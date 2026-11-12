"""Own speech listening for one manager generation.

Config reload builds a new manager, so this owner never mutates its config; it
only reconciles running listeners against the cameras that are enabled right
now (camera power can change at any time through the API or MQTT).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import AppConfig, CameraConfig
from .attach import incident_speech
from .listener import ListenerEvent, SpeechListener
from .segmenter import SpeechEnded, SpeechStarted
from .store import SpeechStore
from .transcriber import TranscriptionJob, TranscriptionResult, TranscriptionService
from .vad import SileroVadModel, VadUnavailable

LOGGER = logging.getLogger("uvicorn.error")

RECONCILE_INTERVAL_SECONDS = 5.0
RETENTION_INTERVAL_SECONDS = 3600.0
# Shorter than this is almost never transcribable speech.
MIN_TRANSCRIBE_SECONDS = 0.5
WHISPER_MODELS = ("tiny.en", "base.en", "small.en", "medium.en")
WHISPER_MARKER_FILE = "openvino_encoder_model.xml"


class SpeechLifecycle:
    def __init__(
        self,
        config: AppConfig,
        *,
        database_dir,
        camera_enabled: Callable[[str], bool],
        audio_available: Callable[[CameraConfig], bool | None] | None = None,
        publish: Callable[[str, dict[str, Any]], Any] | None = None,
        vad_factory: Callable[[str], Any] = SileroVadModel,
        listener_factory: Callable[..., SpeechListener] = SpeechListener,
        transcriber_factory: Callable[..., TranscriptionService] = TranscriptionService,
    ) -> None:
        self.config = config
        self.speech = config.speech
        self.database_dir = database_dir
        self.camera_enabled = camera_enabled
        self.audio_available = audio_available
        self.publish = publish
        self.vad_factory = vad_factory
        self.listener_factory = listener_factory
        self.transcriber_factory = transcriber_factory
        self.cameras = {
            camera.id: camera
            for camera in config.cameras
            if camera.speech.enabled
        }
        self.store: SpeechStore | None = None
        self.transcriber: TranscriptionService | None = None
        self._vad: Any = None
        self._listeners: dict[str, SpeechListener] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "disabled"
        self._reason = ""
        self._last_retention = 0.0

    @property
    def active(self) -> bool:
        return bool(self.speech.enabled and self.cameras)

    def start(self) -> None:
        """Start speech services; failures leave SurvNG running without speech."""
        if not self.speech.enabled:
            self._state, self._reason = "disabled", "speech is disabled"
            return
        if not self.cameras:
            self._state, self._reason = "idle", "no camera has speech listening enabled"
            return
        try:
            self.store = SpeechStore(self.database_dir)
            abandoned = self.store.abandon_pending("restart")
            if abandoned:
                LOGGER.info("Speech: %d pending transcriptions were abandoned at restart", abandoned)
        except Exception as error:
            self._state, self._reason = "unavailable", f"speech database: {type(error).__name__}"
            LOGGER.exception("Speech database could not be opened")
            return
        try:
            self._vad = self.vad_factory(self.speech.vad_model_path)
        except VadUnavailable as error:
            self._state, self._reason = "unavailable", str(error)
            LOGGER.warning("Speech listening unavailable: %s", error)
            return
        except Exception as error:
            self._state, self._reason = "unavailable", f"voice activity model: {type(error).__name__}"
            LOGGER.exception("Speech voice activity model failed to load")
            return
        if any(camera.speech.transcribe for camera in self.cameras.values()):
            self.transcriber = self.transcriber_factory(
                self.speech,
                on_result=self._transcription_result,
                on_dropped=self._transcription_dropped,
            )
            self.transcriber.start()
        self._state, self._reason = "running", ""
        self._stop.clear()
        self._reconcile()
        self._thread = threading.Thread(target=self._run, name="speech-lifecycle", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=RECONCILE_INTERVAL_SECONDS + 2.0)
            self._thread = None
        with self._lock:
            listeners = list(self._listeners.items())
            self._listeners.clear()
        for camera_id, listener in listeners:
            listener.stop()
            self._publish_state(camera_id, False)
        if self.transcriber is not None:
            self.transcriber.stop()
        if self._state == "running":
            self._state = "stopped"

    # Reconciliation ---------------------------------------------------

    def refresh(self) -> None:
        """Apply camera power/detection changes now instead of at the next tick."""
        if self._state != "running" or self._stop.is_set():
            return
        try:
            self._reconcile()
        except Exception:
            LOGGER.exception("Speech reconciliation failed")

    def _run(self) -> None:
        while not self._stop.wait(RECONCILE_INTERVAL_SECONDS):
            try:
                self._reconcile()
                self._apply_retention()
            except Exception:
                LOGGER.exception("Speech reconciliation failed")

    def _reconcile(self) -> None:
        desired = set()
        for camera_id in self.cameras:
            try:
                if self.camera_enabled(camera_id):
                    desired.add(camera_id)
            except Exception:
                continue
        with self._lock:
            if self._stop.is_set():
                return
            stopping = [
                (camera_id, self._listeners.pop(camera_id))
                for camera_id in list(self._listeners)
                if camera_id not in desired
            ]
            starting = [camera_id for camera_id in desired if camera_id not in self._listeners]
            for camera_id in starting:
                camera = self.cameras[camera_id]
                listener = self.listener_factory(
                    camera,
                    ffmpeg_path=self.config.ffmpeg_path,
                    vad_model=self._vad,
                    keep_audio=bool(camera.speech.transcribe and self.transcriber is not None),
                    on_event=self._listener_event,
                    audio_available=self.audio_available,
                )
                self._listeners[camera_id] = listener
                listener.start()
        for camera_id, listener in stopping:
            listener.stop()
            self._publish_state(camera_id, False)

    def _apply_retention(self) -> None:
        if self.store is None:
            return
        now = time.time()
        if now - self._last_retention < RETENTION_INTERVAL_SECONDS:
            return
        self._last_retention = now
        removed = self.store.purge_before(now - self.speech.retention_days * 86400.0)
        if removed:
            LOGGER.info("Speech retention removed %d segments", removed)

    # Events -----------------------------------------------------------

    def _listener_event(self, item: ListenerEvent) -> None:
        if isinstance(item.event, SpeechStarted):
            self._publish_state(item.camera_id, True, started_at=item.started_at)
            return
        if not isinstance(item.event, SpeechEnded) or self.store is None:
            return
        event = item.event
        camera = self.cameras.get(item.camera_id)
        transcribe = bool(
            camera is not None
            and camera.speech.transcribe
            and self.transcriber is not None
            and event.audio is not None
        )
        status, reason = "detected", ""
        if transcribe and event.duration_seconds < MIN_TRANSCRIBE_SECONDS:
            transcribe, status, reason = False, "skipped", "too_short"
        elif transcribe:
            status = "queued"
        segment_id = self.store.insert_segment(
            camera_id=item.camera_id,
            started_at=item.started_at,
            ended_at=float(item.ended_at or item.started_at),
            peak_probability=event.peak_probability,
            status=status,
            reason=reason,
        )
        if not event.split:
            self._publish_state(item.camera_id, False, segment_id=segment_id)
        if transcribe:
            self.transcriber.submit(TranscriptionJob(segment_id, item.camera_id, event.audio))

    def _transcription_dropped(self, job: TranscriptionJob, reason: str) -> None:
        if self.store is not None:
            self.store.set_status(job.segment_id, "skipped", reason=reason[:200])

    def _transcription_result(self, result: TranscriptionResult) -> None:
        if self.store is None:
            return
        if result.status == "transcribing":
            self.store.set_status(result.segment_id, "transcribing")
            return
        self.store.set_transcript(
            result.segment_id,
            status=result.status,
            text=result.text,
            language=result.language,
            score=result.score,
            model=result.model,
            device=result.device,
            reason=result.reason,
        )
        if (
            self.speech.mqtt_publish_transcripts
            and self.publish is not None
            and result.status == "done"
        ):
            self._safe_publish(
                f"camera/{result.camera_id}/speech/transcript",
                {
                    "camera_id": result.camera_id,
                    "segment_id": result.segment_id,
                    "text": result.text,
                    "language": result.language,
                },
            )

    def _publish_state(
        self,
        camera_id: str,
        speaking: bool,
        *,
        started_at: float | None = None,
        segment_id: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "camera_id": camera_id,
            "state": "ON" if speaking else "OFF",
            "speech": speaking,
        }
        if started_at is not None:
            payload["started_at"] = started_at
        if segment_id is not None:
            payload["segment_id"] = segment_id
        self._safe_publish(f"camera/{camera_id}/speech", payload)

    def _safe_publish(self, suffix: str, payload: dict[str, Any]) -> None:
        if self.publish is None:
            return
        try:
            self.publish(suffix, payload)
        except Exception:
            LOGGER.debug("Speech MQTT publish failed for %s", suffix, exc_info=True)

    # Status -----------------------------------------------------------

    def model_status(self) -> dict[str, Any]:
        """Which configured models exist on disk (names only, never paths)."""
        speech = self.speech
        vad_installed = bool(speech.vad_model_path) and Path(speech.vad_model_path).is_file()
        installed: list[str] = []
        if speech.whisper_models_dir:
            root = Path(speech.whisper_models_dir)
            installed = [
                model
                for model in WHISPER_MODELS
                if (root / f"whisper-{model}" / WHISPER_MARKER_FILE).is_file()
            ]
        selected = speech.resolved_whisper_model()
        return {
            "vad_installed": vad_installed,
            "whisper_installed": installed,
            "selected_device": speech.transcription_device,
            "selected_model": selected,
            "selected_model_installed": selected in installed,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            listeners = {camera_id: listener.status() for camera_id, listener in self._listeners.items()}
        cameras = {}
        for camera_id, camera in self.cameras.items():
            cameras[camera_id] = listeners.get(camera_id) or {
                "state": "paused" if self._state == "running" else self._state,
                "transcribe": bool(camera.speech.transcribe),
                "source": camera.speech.source,
            }
        return {
            "enabled": bool(self.speech.enabled),
            "state": self._state,
            "reason": self._reason,
            "cameras": cameras,
            "transcriber": self.transcriber.status() if self.transcriber is not None else None,
            "models": self.model_status(),
        }

    def _read_store(self) -> SpeechStore | None:
        # Past segments stay readable after speech is turned off, but a
        # disabled install never creates the database just to answer a query.
        if self.store is not None:
            return self.store
        if not (Path(self.database_dir) / "speech.sqlite3").exists():
            return None
        return SpeechStore(self.database_dir)

    def segments(self, **filters: Any) -> list[dict[str, Any]]:
        store = self._read_store()
        return store.list_segments(**filters) if store is not None else []

    def incident_speech(self, windows_by_incident) -> dict[str, dict[str, Any]]:
        store = self._read_store()
        if store is None:
            return {}
        return incident_speech(store, windows_by_incident)

    def segment(self, segment_id: int) -> dict[str, Any] | None:
        store = self._read_store()
        return store.get(segment_id) if store is not None else None

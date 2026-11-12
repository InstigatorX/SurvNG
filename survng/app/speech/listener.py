"""Per-camera audio listener: FFmpeg PCM -> voice activity -> speech segments.

Ownership: one listener thread owns one FFmpeg process and one segmenter.
The FFmpeg child writes into a pipe owned by this process, so it exits on
SIGPIPE if SurvNG dies; no stale-process sweep is needed (unlike recorders).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..config import CameraConfig
from ..ffmpeg_input import ffmpeg_input_args
from ..ffmpeg_process import named_ffmpeg_executable
from ..security import redact_secret_text
from .segmenter import FRAME_SAMPLES, SAMPLE_RATE, SegmenterEvent, SpeechEnded, SpeechSegmenter

LOGGER = logging.getLogger("uvicorn.error")

FRAME_BYTES = FRAME_SAMPLES * 2
RESTART_DELAYS_SECONDS = (2.0, 5.0, 15.0, 30.0, 60.0)
NO_AUDIO_RETRY_SECONDS = 600.0
HEALTHY_RUN_SECONDS = 60.0
# Re-anchor sample time to the wall clock when buffering skews it this much.
CLOCK_REANCHOR_SECONDS = 3.0
FAILURE_LOG_INTERVAL_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class ListenerEvent:
    camera_id: str
    event: SegmenterEvent
    started_at: float
    ended_at: float | None


def audio_listener_command(camera: CameraConfig, ffmpeg_path: str) -> list[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        *ffmpeg_input_args(camera, camera.speech.source),
        "-vn",
        "-sn",
        "-dn",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]


def classify_ffmpeg_failure(line: str) -> str:
    text = line.lower()
    if "matches no streams" in text or "does not contain any stream" in text:
        return "no_audio_track"
    if any(token in text for token in ("401", "unauthorized", "forbidden", "403")):
        return "stream_unauthorized"
    if any(token in text for token in (
        "connection refused", "timed out", "timeout", "no route", "404", "not found",
        "network is unreachable", "end of file",
    )):
        return "stream_unavailable"
    if any(token in text for token in ("invalid data", "error while decoding", "decod")):
        return "audio_decode_error"
    return "ffmpeg_error"


class SpeechListener:
    def __init__(
        self,
        camera: CameraConfig,
        *,
        ffmpeg_path: str,
        vad_model: Any,
        keep_audio: bool,
        on_event: Callable[[ListenerEvent], None],
        audio_available: Callable[[CameraConfig], bool | None] | None = None,
        clock: Callable[[], float] = time.time,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        sleep: Callable[[threading.Event, float], bool] | None = None,
    ) -> None:
        self.camera = camera
        self.ffmpeg_path = ffmpeg_path
        self.vad_model = vad_model
        self.keep_audio = keep_audio
        self.on_event = on_event
        self.audio_available = audio_available
        self.clock = clock
        self.popen = popen
        self._wait = sleep or (lambda event, seconds: event.wait(seconds))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._state = "stopped"
        self._failure = ""
        self._restarts = 0
        self._segments = 0
        self._last_speech_at: float | None = None
        self._last_failure_log: dict[str, float] = {}

    # Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            self._state = "starting"
            self._thread = threading.Thread(
                target=self._run,
                name=f"speech-listener-{self.camera.id}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._terminate()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        with self._lock:
            self._thread = None
            self._state = "stopped"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "failure": self._failure,
                "restarts": self._restarts,
                "segments": self._segments,
                "last_speech_at": (
                    datetime.fromtimestamp(self._last_speech_at, timezone.utc).isoformat()
                    if self._last_speech_at is not None
                    else None
                ),
                "transcribe": bool(self.camera.speech.transcribe),
                "source": self.camera.speech.source,
            }

    # Worker ------------------------------------------------------------

    def _set_state(self, state: str, failure: str | None = None) -> None:
        with self._lock:
            self._state = state
            if failure is not None:
                self._failure = failure

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            if self.audio_available is not None:
                try:
                    available = self.audio_available(self.camera)
                except Exception:
                    available = None
                if available is False:
                    self._set_state("no_audio_track", "no_audio_track")
                    self._wait(self._stop, NO_AUDIO_RETRY_SECONDS)
                    continue
            self._set_state("starting")
            started = time.monotonic()
            failure = self._run_once()
            if self._stop.is_set():
                break
            if time.monotonic() - started >= HEALTHY_RUN_SECONDS:
                attempt = 0
            with self._lock:
                self._restarts += 1
            if failure == "no_audio_track":
                self._set_state("no_audio_track", failure)
                delay = NO_AUDIO_RETRY_SECONDS
            else:
                self._set_state("restarting", failure or "stream_ended")
                delay = RESTART_DELAYS_SECONDS[min(attempt, len(RESTART_DELAYS_SECONDS) - 1)]
                attempt += 1
            self._wait(self._stop, delay)
        self._set_state("stopped")

    def _run_once(self) -> str:
        command = audio_listener_command(self.camera, self.ffmpeg_path)
        executable = named_ffmpeg_executable(self.ffmpeg_path, "survng-audio-listener")
        failures: list[str] = []
        try:
            process = self.popen(
                [executable, *command[1:]],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            self._log_failure("ffmpeg_start_failed", redact_secret_text(error))
            return "ffmpeg_start_failed"
        with self._lock:
            self._process = process
        if self._stop.is_set():
            self._terminate()
        stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(process, failures),
            name=f"speech-stderr-{self.camera.id}",
            daemon=True,
        )
        stderr_thread.start()
        try:
            self._consume(process)
        finally:
            self._terminate()
            stderr_thread.join(timeout=2.0)
            with self._lock:
                self._process = None
        return failures[0] if failures else ""

    def _consume(self, process: subprocess.Popen) -> None:
        speech = self.camera.speech
        detector = self.vad_model.stream()
        segmenter = SpeechSegmenter(
            threshold=speech.vad_threshold,
            min_speech_ms=speech.min_speech_ms,
            hangover_ms=speech.hangover_ms,
            pre_roll_ms=speech.pre_roll_ms,
            max_segment_seconds=speech.max_segment_seconds,
            keep_audio=self.keep_audio,
        )
        anchor: float | None = None
        stdout = process.stdout
        try:
            while not self._stop.is_set() and stdout is not None:
                data = stdout.read(FRAME_BYTES)
                if not data or len(data) < FRAME_BYTES:
                    break
                now = self.clock()
                position = segmenter.frames_processed * FRAME_SAMPLES
                if anchor is None:
                    anchor = now - FRAME_SAMPLES / SAMPLE_RATE
                    self._set_state("listening", "")
                elif (
                    not segmenter.speaking
                    and abs(anchor + position / SAMPLE_RATE - now) > CLOCK_REANCHOR_SECONDS
                ):
                    anchor = now - position / SAMPLE_RATE
                frame = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                probability = detector.probability(frame)
                for event in segmenter.push(frame, probability):
                    self._emit(event, anchor)
        finally:
            ended = segmenter.flush()
            if ended is not None and anchor is not None:
                self._emit(ended, anchor)

    def _emit(self, event: SegmenterEvent, anchor: float) -> None:
        started_at = anchor + event.start_sample / SAMPLE_RATE
        ended_at = None
        if isinstance(event, SpeechEnded):
            ended_at = anchor + event.end_sample / SAMPLE_RATE
            with self._lock:
                self._segments += 1
                self._last_speech_at = ended_at
                if self._state == "speaking":
                    self._state = "listening"
        else:
            self._set_state("speaking")
        try:
            self.on_event(ListenerEvent(self.camera.id, event, started_at, ended_at))
        except Exception:
            LOGGER.exception("speech event handling failed for camera %s", self.camera.id)

    def _read_stderr(self, process: subprocess.Popen, failures: list[str]) -> None:
        stream = process.stderr
        if stream is None:
            return
        for raw_line in stream:
            line = raw_line.decode("utf-8", errors="replace").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
            if not line:
                continue
            category = classify_ffmpeg_failure(line)
            if not failures or (category == "no_audio_track" and failures[0] != category):
                failures.insert(0, category)
            self._log_failure(category, redact_secret_text(line)[:300])

    def _log_failure(self, category: str, detail: str) -> None:
        now = time.monotonic()
        with self._lock:
            last = self._last_failure_log.get(category)
            if last is not None and now - last < FAILURE_LOG_INTERVAL_SECONDS:
                return
            self._last_failure_log[category] = now
        LOGGER.warning(
            "Speech listener for camera %s: %s (%s)", self.camera.id, category, detail
        )

    def _terminate(self) -> None:
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.terminate()
            except OSError:
                return
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                process.kill()

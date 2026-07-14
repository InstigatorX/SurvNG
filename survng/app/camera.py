from __future__ import annotations

import logging
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .config import CameraConfig, DetectionZone
from .detector import OpenVinoDetector, objects_to_json
from .events import EventStore
from .ffmpeg_hw import recorded_frame_hw_args
from .onvif_events import OnvifEventListener
from .recorder import Recorder
from .zones import apply_detection_zones, detection_threshold

LOGGER = logging.getLogger(__name__)
RECORDED_EVENT_FRAME_OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)
RECORDED_EVENT_SETTLE_SECONDS = 0.75
RECORDED_EVENT_RETRY_SECONDS = 12.0
RECORDED_EVENT_RETRY_INTERVAL_SECONDS = 1.0
CAPTURE_OPEN_TIMEOUT_MS = 5000
CAPTURE_READ_TIMEOUT_MS = 5000
CAPTURE_STOP_TIMEOUT_SECONDS = 8.0


class CameraWorker:
    def __init__(
        self,
        camera: CameraConfig,
        storage_dir: Path,
        detector: OpenVinoDetector,
        events: EventStore,
        recorder: Recorder,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.camera = camera
        self.storage_dir = storage_dir
        self.detector = detector
        self.events = events
        self.recorder = recorder
        self.event_callback = event_callback
        self.snapshots_dir = storage_dir / "snapshots" / camera.id
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._source_threads: dict[str, threading.Thread] = {}
        self._source_stops: dict[str, threading.Event] = {}
        self._source_frames: dict[str, Any] = {}
        self._source_frame_at: dict[str, str] = {}
        self._source_errors: dict[str, str] = {}
        self.last_error = ""
        self.last_frame_at = ""
        self.last_motion_at = ""
        self.onvif = OnvifEventListener(camera, self.handle_motion_event)

    def start(self) -> None:
        with self._lifecycle_lock:
            self._stop.clear()
            self._start_source("live")
            self.onvif.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            with self._frame_lock:
                stops = list(self._source_stops.values())
                threads = list(self._source_threads.items())
            for stop_event in stops:
                stop_event.set()
            self.onvif.stop()
            deadline = time.monotonic() + CAPTURE_STOP_TIMEOUT_SECONDS
            for _source, thread in threads:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            alive = [source for source, thread in threads if thread.is_alive()]
            if alive:
                LOGGER.error(
                    "camera capture threads did not stop for %s: %s",
                    self.camera.id,
                    ", ".join(alive),
                )
            with self._frame_lock:
                self._source_threads = {
                    source: thread for source, thread in self._source_threads.items()
                    if thread.is_alive()
                }
                self._source_stops = {
                    source: stop for source, stop in self._source_stops.items()
                    if source in self._source_threads
                }
                self._source_frames.clear()
                self._source_frame_at.clear()
                self._source_errors.clear()
            self._thread = self._source_threads.get("live")

    def status(self) -> dict[str, Any]:
        live_thread = self._source_threads.get("live")
        main_thread = self._source_threads.get("main")
        return {
            "id": self.camera.id,
            "name": self.camera.name,
            "running": live_thread is not None and live_thread.is_alive(),
            "main_running": main_thread is not None and main_thread.is_alive(),
            "last_frame_at": self.last_frame_at,
            "main_last_frame_at": self._source_frame_at.get("main", ""),
            "last_error": self.last_error,
            "main_last_error": self._source_errors.get("main", ""),
            "onvif_enabled": self.camera.onvif.enabled,
            "onvif_connected": self.onvif.connected,
            "onvif_last_event_at": self.onvif.last_event_at,
            "last_motion_at": self.last_motion_at,
            "onvif_last_error": self.onvif.last_error,
            "onvif_last_connected_at": self.onvif.last_connected_at,
            "onvif_last_poll_success_at": self.onvif.last_poll_success_at,
            "onvif_last_poll_error": self.onvif.last_poll_error,
            "onvif_last_poll_error_at": self.onvif.last_poll_error_at,
            "onvif_retry_attempts": self.onvif.retry_attempts,
            "onvif_poll_timeouts": self.onvif.poll_timeouts,
            "onvif_poll_errors": self.onvif.poll_errors,
            "onvif_resubscriptions": self.onvif.resubscriptions,
            "onvif_subscription_current_time": self.onvif.subscription_current_time,
            "onvif_subscription_termination_time": self.onvif.subscription_termination_time,
            "onvif_subscription_lifetime_seconds": self.onvif.subscription_lifetime_seconds,
        }

    def update_zones(self, zones: list[DetectionZone]) -> None:
        next_zones = [zone.model_copy(deep=True) for zone in zones]
        with self._lifecycle_lock:
            self.camera.zones = next_zones

    def snapshot(self, source: str = "live") -> bytes | None:
        frame = self._get_latest_frame(source)
        if frame is None:
            return None
        ok, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ok else None

    def mjpeg_frames(self, fps: float = 4.0, source: str = "live"):
        source = self.camera.normalized_source(source)
        delay = 1.0 / max(fps, 1.0)
        while not self._stop.is_set():
            image = self.snapshot(source)
            if image is None:
                time.sleep(delay)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + image
                + b"\r\n"
            )
            time.sleep(delay)

    def handle_motion_event(
        self,
        topic: str = "manual",
        message: str = "",
        event_at: datetime | None = None,
    ) -> None:
        self.last_motion_at = datetime.now(timezone.utc).isoformat()
        if event_at is None:
            event_at = datetime.now(timezone.utc)
        elif event_at.tzinfo is None:
            event_at = event_at.replace(tzinfo=timezone.utc)
        else:
            event_at = event_at.astimezone(timezone.utc)

        if self.event_callback:
            self.event_callback("motion", {
                "camera_id": self.camera.id,
                "timestamp": event_at.isoformat(),
                "source": "manual" if topic.startswith("manual") else "onvif",
            })

        frame, objects, recording_path = self._recorded_motion_frame(event_at)
        snapshot_path = ""
        if frame is not None:
            snapshot_path = self._write_snapshot(frame)
        else:
            objects = [{"status": "no_recorded_frame"}]
        event = self.events.add_event(
            camera_id=self.camera.id,
            kind="motion",
            topic=topic,
            message=message,
            snapshot_path=snapshot_path,
            recording_path=recording_path,
            objects_json=objects_to_json(objects),
            created_at=event_at.isoformat(),
        )
        detected_objects = [
            detected for detected in objects
            if detected.get("label")
        ]
        eligible_objects = [
            detected for detected in objects
            if detected.get("label") and detected.get("incident_eligible") is not False
        ]
        if self.event_callback and detected_objects:
            self.event_callback("object", {
                "event_id": event.get("id"),
                "camera_id": self.camera.id,
                "timestamp": event_at.isoformat(),
                "snapshot_path": snapshot_path,
                "recording_path": recording_path,
                "objects": detected_objects,
                "incident_objects": eligible_objects,
            })

    def _start_source(self, source: str) -> None:
        source = self.camera.normalized_source(source)
        with self._frame_lock:
            thread = self._source_threads.get(source)
            if thread is not None and thread.is_alive():
                return
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run_source,
                args=(source, stop_event),
                name=f"camera-{self.camera.id}-{source}",
                daemon=False,
            )
            self._source_stops[source] = stop_event
            self._source_threads[source] = thread
            if source == "live":
                self._thread = thread
            thread.start()

    def _run_source(self, source: str, stop_event: threading.Event) -> None:
        while not self._stop.is_set() and not stop_event.is_set():
            capture = cv2.VideoCapture()
            try:
                opened = capture.open(
                    self.camera.source_url(source),
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        CAPTURE_OPEN_TIMEOUT_MS,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        CAPTURE_READ_TIMEOUT_MS,
                    ],
                )
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not opened or not capture.isOpened():
                    self._set_source_error(source, "failed to open stream")
                else:
                    self._set_source_error(source, "")
                    while not self._stop.is_set() and not stop_event.is_set():
                        ok, frame = capture.read()
                        if not ok:
                            self._set_source_error(source, "stream read failed")
                            break
                        stamp = datetime.now(timezone.utc).isoformat()
                        with self._frame_lock:
                            self._source_frames[source] = frame.copy()
                            self._source_frame_at[source] = stamp
                            if source == "live":
                                self.last_frame_at = stamp
            except Exception as exc:
                self._set_source_error(source, f"stream error: {str(exc)[:160]}")
                LOGGER.warning("camera stream failed for %s/%s: %s", self.camera.id, source, exc)
            finally:
                capture.release()
            if stop_event.wait(1.0):
                break

    def _set_source_error(self, source: str, message: str) -> None:
        with self._frame_lock:
            self._source_errors[source] = message
            if source == "live":
                self.last_error = message

    def _get_latest_frame(self, source: str = "live") -> Any:
        source = self.camera.normalized_source(source)
        self._start_source(source)
        with self._frame_lock:
            frame = self._source_frames.get(source)
            if frame is None:
                return None
            return frame.copy()

    def _recorded_motion_frame(
        self,
        event_at: datetime,
    ) -> tuple[Any | None, list[dict[str, Any]], str]:
        event_epoch = event_at.timestamp()
        newest_needed = event_epoch + max(RECORDED_EVENT_FRAME_OFFSETS) + RECORDED_EVENT_SETTLE_SECONDS
        wait_seconds = max(0.0, newest_needed - time.time())
        if wait_seconds > 0:
            time.sleep(min(wait_seconds, 3.0))

        deadline = time.time() + RECORDED_EVENT_RETRY_SECONDS
        best_frame: Any | None = None
        best_objects: list[dict[str, Any]] = []
        best_score = -1.0
        best_distance = float("inf")
        best_recording_path = ""

        while True:
            for sample_offset in RECORDED_EVENT_FRAME_OFFSETS:
                target_epoch = event_epoch + sample_offset
                row = self.recorder.recording_at(self.camera.id, target_epoch)
                if row is None:
                    continue

                start_epoch = row.get("start_epoch")
                if start_epoch is None:
                    continue
                frame_offset = max(0.0, target_epoch - float(start_epoch))
                frame = self._read_recorded_frame(Path(row["path"]), frame_offset)
                if frame is None:
                    continue

                threshold = detection_threshold(self.camera, self.detector.config.confidence_threshold)
                objects = self.detector.detect(frame, confidence_threshold=threshold)
                apply_detection_zones(
                    self.camera,
                    objects,
                    int(frame.shape[1]),
                    int(frame.shape[0]),
                    self.detector.config.confidence_threshold,
                )
                score = self._motion_object_score(objects)
                distance = abs(sample_offset)
                if score > best_score or (score == best_score and distance < best_distance):
                    best_frame = frame
                    best_objects = objects
                    best_score = score
                    best_distance = distance
                    best_recording_path = str(row["path"])

            if best_frame is not None or time.time() >= deadline:
                break
            time.sleep(RECORDED_EVENT_RETRY_INTERVAL_SECONDS)

        if best_frame is not None:
            return best_frame, best_objects, best_recording_path

        fallback = self._get_latest_frame()
        if fallback is not None:
            threshold = detection_threshold(self.camera, self.detector.config.confidence_threshold)
            objects = self.detector.detect(fallback, confidence_threshold=threshold)
            apply_detection_zones(
                self.camera,
                objects,
                int(fallback.shape[1]),
                int(fallback.shape[0]),
                self.detector.config.confidence_threshold,
            )
            if objects:
                for detected in objects:
                    detected["frame_source"] = "live_fallback"
                    detected["recording_status"] = "no_recorded_frame"
                return fallback, objects, ""
            return fallback, [{"status": "no_recorded_frame", "frame_source": "live_fallback"}], ""
        return None, [{"status": "no_recorded_frame"}], ""

    def _read_recorded_frame(self, path: Path, offset_seconds: float) -> Any | None:
        if not path.exists():
            return None

        attempts = [0.0, -0.25, 0.25, -0.75, 0.75]
        last_error = ""
        hw_input_args, hw_filter_args = recorded_frame_hw_args(self.recorder.hardware_acceleration)
        decode_plans = [("hardware", hw_input_args, hw_filter_args)] if hw_input_args else []
        decode_plans.append(("cpu", [], []))
        for nudge in attempts:
            sample_at = max(0.0, offset_seconds + nudge)
            for backend, input_args, filter_args in decode_plans:
                command = [
                    self.recorder.ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-fflags",
                    "+discardcorrupt",
                    "-err_detect",
                    "ignore_err",
                    *input_args,
                    "-ss",
                    f"{sample_at:.3f}",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-frames:v",
                    "1",
                    *filter_args,
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "pipe:1",
                ]
                try:
                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=8,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    last_error = f"{backend} timed out"
                    continue

                if result.returncode != 0 or not result.stdout:
                    detail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[0:2]
                    last_error = f"{backend}: {' '.join(detail)[:180]}"
                    continue

                array = np.frombuffer(result.stdout, dtype=np.uint8)
                frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
                last_error = f"{backend}: mjpeg decode returned no frame"

        LOGGER.debug(
            "skipped unreadable recording sample for %s at %.2fs: %s%s",
            self.camera.id,
            offset_seconds,
            path,
            f" ({last_error})" if last_error else "",
        )
        return None

    def _motion_object_score(self, objects: list[dict[str, Any]]) -> float:
        score = 0.0
        for detected in objects:
            label = detected.get("label")
            if not label or detected.get("incident_eligible") is False:
                continue
            confidence = detected.get("confidence")
            if isinstance(confidence, (float, int)):
                score = max(score, float(confidence))
            else:
                score = max(score, 0.01)
        return score

    def _write_snapshot(self, frame: Any) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        path = self.snapshots_dir / f"{stamp}.jpg"
        if not cv2.imwrite(str(path), frame):
            LOGGER.warning("failed to write snapshot for %s to %s", self.camera.id, path)
            return ""
        return str(path)

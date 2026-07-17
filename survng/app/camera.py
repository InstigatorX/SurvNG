from __future__ import annotations

import logging
import queue
import random
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .config import CameraConfig, DetectionZone, MotionQualificationConfig
from .detector import OpenVinoDetector, objects_to_json
from .events import EventStore
from .ffmpeg_hw import recorded_frame_hw_args
from .onvif_events import OnvifEventListener
from .recorder import Recorder
from .motion import BackgroundMotionTracker, MotionQualificationResult, aggregate_mog2_evidence, qualify_motion
from .zones import apply_detection_zones, detection_threshold

LOGGER = logging.getLogger(__name__)
RECORDED_EVENT_FRAME_OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)
RECORDED_EVENT_SETTLE_SECONDS = 0.75
RECORDED_EVENT_RETRY_SECONDS = 12.0
RECORDED_EVENT_RETRY_INTERVAL_SECONDS = 1.0
CAPTURE_OPEN_TIMEOUT_MS = 5000
CAPTURE_READ_TIMEOUT_MS = 5000
CAPTURE_STOP_TIMEOUT_SECONDS = 8.0
FRAME_STALE_SECONDS = 10.0
MAIN_SOURCE_IDLE_SECONDS = 20.0
MOTION_QUEUE_SIZE = 32


class CameraWorker:
    def __init__(
        self,
        camera: CameraConfig,
        storage_dir: Path,
        detector: OpenVinoDetector,
        events: EventStore,
        recorder: Recorder,
        motion_config: MotionQualificationConfig | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.camera = camera
        self.storage_dir = storage_dir
        self.detector = detector
        self.events = events
        self.recorder = recorder
        self.motion_config = motion_config or MotionQualificationConfig()
        self.event_callback = event_callback
        self.snapshots_dir = storage_dir / "snapshots" / camera.id
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._stop.set()
        self._enabled = False
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._source_threads: dict[str, threading.Thread] = {}
        self._source_stops: dict[str, threading.Event] = {}
        self._source_frames: dict[str, Any] = {}
        self._source_frame_at: dict[str, str] = {}
        self._source_frame_monotonic: dict[str, float] = {}
        self._source_last_access: dict[str, float] = {}
        self._source_errors: dict[str, str] = {}
        ring_size = max(
            12,
            round(
                self.motion_config.sample_fps
                * (self.motion_config.window_seconds + self.motion_config.post_trigger_seconds + 3.0)
            ),
        )
        self._motion_frames: deque[tuple[float, np.ndarray]] = deque(maxlen=ring_size)
        self._mog2_samples: deque[tuple[float, dict[str, Any]]] = deque(maxlen=ring_size)
        self._mog2_lock = threading.Lock()
        self._mog2_tracker = BackgroundMotionTracker(
            sample_fps=self.motion_config.sample_fps,
            history_seconds=self.motion_config.mog2_history_seconds,
        )
        self._motion_last_sample = 0.0
        self._motion_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=MOTION_QUEUE_SIZE)
        self._motion_thread: threading.Thread | None = None
        self._motion_stats_lock = threading.Lock()
        self._motion_stats: dict[str, Any] = {
            "triggers": 0,
            "bursts": 0,
            "passed": 0,
            "audit_rejected": 0,
            "suppressed": 0,
            "priority_bypasses": 0,
            "insufficient_frames": 0,
            "inconclusive": 0,
            "dropped_triggers": 0,
            "audit_object_matches": 0,
            "last_result": None,
        }
        self.last_error = ""
        self.last_frame_at = ""
        self.last_motion_at = ""
        self._detection_enabled = True
        self.onvif = OnvifEventListener(camera, self.handle_motion_event)

    def start(self) -> None:
        with self._lifecycle_lock:
            self._enabled = True
            self._stop.clear()
            self._start_source("live")
            if self._motion_thread is None or not self._motion_thread.is_alive():
                self._clear_motion_queue()
                self._motion_thread = threading.Thread(
                    target=self._run_motion_events,
                    name=f"motion-{self.camera.id}",
                    daemon=False,
                )
                self._motion_thread.start()
            self.onvif.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._enabled = False
            self._stop.set()
            with self._frame_lock:
                stops = list(self._source_stops.values())
                threads = list(self._source_threads.items())
            for stop_event in stops:
                stop_event.set()
            self.onvif.stop()
            try:
                self._motion_queue.put_nowait(None)
            except queue.Full:
                pass
            motion_thread = self._motion_thread
            if motion_thread is not None:
                motion_thread.join(timeout=RECORDED_EVENT_RETRY_SECONDS + 10)
                if motion_thread.is_alive():
                    LOGGER.error("motion worker did not stop for %s", self.camera.id)
            self._motion_thread = motion_thread if motion_thread is not None and motion_thread.is_alive() else None
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
                self._source_frame_monotonic.clear()
                self._source_last_access.clear()
                self._source_errors.clear()
                self._motion_frames.clear()
                self._mog2_samples.clear()
                self._motion_last_sample = 0.0
                self.last_frame_at = ""
            with self._mog2_lock:
                self._mog2_tracker.reset()
            self._thread = self._source_threads.get("live")

    def status(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            enabled = self._enabled
        with self._frame_lock:
            live_thread = self._source_threads.get("live")
            main_thread = self._source_threads.get("main")
            live_frame_at = self._source_frame_at.get("live", "")
            main_frame_at = self._source_frame_at.get("main", "")
            live_frame_clock = self._source_frame_monotonic.get("live")
            main_frame_clock = self._source_frame_monotonic.get("main")
            main_error = self._source_errors.get("main", "")
            motion_buffered_frames = len(self._motion_frames)
            motion_frame_shape = list(self._motion_frames[-1][1].shape) if self._motion_frames else None
            mog2_last = dict(self._mog2_samples[-1][1]) if self._mog2_samples else None
        now = time.monotonic()
        live_age = max(0.0, now - live_frame_clock) if live_frame_clock is not None else None
        main_age = max(0.0, now - main_frame_clock) if main_frame_clock is not None else None
        connected = bool(enabled and live_age is not None and live_age <= FRAME_STALE_SECONDS)
        mode, sensitivity, frame_width = self._motion_settings()
        rescue_enabled, rescue_margin = self._motion_rescue_settings()
        with self._motion_stats_lock:
            motion_stats = dict(self._motion_stats)
        return {
            "id": self.camera.id,
            "name": self.camera.name,
            "running": enabled,
            "connected": connected,
            "capture_running": live_thread is not None and live_thread.is_alive(),
            "frame_fresh": connected,
            "last_frame_age_seconds": round(live_age, 3) if live_age is not None else None,
            "main_running": main_thread is not None and main_thread.is_alive(),
            "main_frame_fresh": bool(main_age is not None and main_age <= FRAME_STALE_SECONDS),
            "main_last_frame_age_seconds": round(main_age, 3) if main_age is not None else None,
            "last_frame_at": live_frame_at,
            "main_last_frame_at": main_frame_at,
            "last_error": self.last_error,
            "main_last_error": main_error,
            "onvif_enabled": self.camera.onvif.enabled,
            "onvif_connected": self.onvif.connected,
            "onvif_last_event_at": self.onvif.last_event_at,
            "last_motion_at": self.last_motion_at,
            "detection_enabled": self._detection_enabled,
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
            "motion_qualification": {
                **motion_stats,
                "mode": mode,
                "sensitivity": sensitivity,
                "frame_width": frame_width,
                "borderline_rescue_enabled": rescue_enabled,
                "borderline_margin": rescue_margin,
                "mog2_audit_enabled": self._mog2_audit_enabled(),
                "mog2_history_seconds": self.motion_config.mog2_history_seconds,
                "mog2_last": mog2_last,
                "queue_depth": self._motion_queue.qsize(),
                "buffered_frames": motion_buffered_frames,
                "frame_shape": motion_frame_shape,
            },
        }

    def update_zones(self, zones: list[DetectionZone]) -> None:
        next_zones = [zone.model_copy(deep=True) for zone in zones]
        with self._lifecycle_lock:
            self.camera.zones = next_zones

    def set_detection_enabled(self, enabled: bool) -> None:
        self._detection_enabled = bool(enabled)

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
        if not self._detection_enabled:
            return
        received_at = time.time()
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

        trigger = {
            "topic": topic,
            "message": message,
            "event_at": event_at,
            "received_at": received_at,
        }
        with self._motion_stats_lock:
            self._motion_stats["triggers"] += 1
        try:
            self._motion_queue.put_nowait(trigger)
        except queue.Full:
            try:
                self._motion_queue.get_nowait()
            except queue.Empty:
                pass
            with self._motion_stats_lock:
                self._motion_stats["dropped_triggers"] += 1
            try:
                self._motion_queue.put_nowait(trigger)
            except queue.Full:
                pass

    def _clear_motion_queue(self) -> None:
        while True:
            try:
                self._motion_queue.get_nowait()
            except queue.Empty:
                return

    def _remember_motion_frame(self, frame: np.ndarray, frame_clock: float) -> None:
        interval = 1.0 / max(1.0, self.motion_config.sample_fps)
        with self._frame_lock:
            if frame_clock - self._motion_last_sample < interval * 0.85:
                return
            self._motion_last_sample = frame_clock
        try:
            height, width = frame.shape[:2]
            frame_width = self._motion_settings()[2]
            target_height = max(90, round(height * frame_width / max(1, width)))
            resized = cv2.resize(frame, (frame_width, target_height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        except (cv2.error, ValueError):
            return
        captured_at = time.time()
        mog2_evidence = None
        if self._mog2_audit_enabled():
            with self._mog2_lock:
                mog2_evidence = self._mog2_tracker.update(gray)
        with self._frame_lock:
            self._motion_frames.append((captured_at, gray))
            if mog2_evidence is not None:
                self._mog2_samples.append((captured_at, mog2_evidence))

    def _motion_settings(self) -> tuple[str, str, int]:
        override = self.camera.motion_qualification
        mode = self.motion_config.mode if override.mode == "inherit" else override.mode
        sensitivity = self.motion_config.sensitivity if override.sensitivity == "inherit" else override.sensitivity
        frame_width = int(override.frame_width or self.motion_config.frame_width)
        return mode, sensitivity, frame_width

    def _motion_rescue_settings(self) -> tuple[bool, float]:
        override = self.camera.motion_qualification
        enabled = (
            self.motion_config.borderline_rescue_enabled
            if override.borderline_rescue_enabled is None
            else override.borderline_rescue_enabled
        )
        margin = (
            self.motion_config.borderline_margin
            if override.borderline_margin is None
            else override.borderline_margin
        )
        return bool(enabled), float(margin)

    def _mog2_audit_enabled(self) -> bool:
        override = self.camera.motion_qualification.mog2_audit_enabled
        enabled = self.motion_config.mog2_audit_enabled if override is None else override
        return bool(enabled and self._motion_settings()[0] == "audit")

    def _with_mog2_evidence(
        self,
        result: MotionQualificationResult,
        start_epoch: float,
        end_epoch: float,
    ) -> MotionQualificationResult:
        if not self._mog2_audit_enabled():
            return result
        with self._frame_lock:
            samples = [
                evidence for captured_at, evidence in self._mog2_samples
                if start_epoch <= captured_at <= end_epoch
            ]
        return MotionQualificationResult(
            accepted=result.accepted,
            score=result.score,
            threshold=result.threshold,
            reason=result.reason,
            frame_count=result.frame_count,
            features={**result.features, **aggregate_mog2_evidence(samples)},
        )

    @staticmethod
    def _priority_motion_topic(topic: str) -> bool:
        searchable = topic.lower()
        return topic.startswith("manual") or any(
            word in searchable for word in ("person", "people", "human", "vehicle", "animal", "face")
        )

    def _qualify_motion_burst(
        self,
        event_at: datetime,
        received_at: float,
        sensitivity: str,
    ) -> tuple[MotionQualificationResult, dict[str, Any]]:
        event_epoch = event_at.timestamp()
        anchor = min(event_epoch, received_at) if abs(event_epoch - received_at) <= 10.0 else received_at
        deadline = received_at + self.motion_config.post_trigger_seconds
        best_result: MotionQualificationResult | None = None
        evaluated_windows: set[tuple[float, ...]] = set()
        samples: list[tuple[float, np.ndarray]] = []

        while not self._stop.is_set():
            with self._frame_lock:
                samples = [
                    (captured_at, frame.copy())
                    for captured_at, frame in self._motion_frames
                    if captured_at >= anchor - self.motion_config.window_seconds
                ]

            for end_index in range(3, len(samples)):
                window_end = samples[end_index][0]
                if window_end < received_at:
                    continue
                window_start = window_end - self.motion_config.window_seconds
                window = [item for item in samples[:end_index + 1] if item[0] >= window_start]
                if len(window) < 4 or window[-1][0] - window[0][0] < self.motion_config.window_seconds * 0.45:
                    continue
                key = tuple(round(item[0], 3) for item in window)
                if key in evaluated_windows:
                    continue
                evaluated_windows.add(key)
                result = qualify_motion([item[1] for item in window], sensitivity)
                if best_result is None or result.score > best_result.score:
                    best_result = result
                if result.accepted:
                    return self._with_mog2_evidence(
                        result,
                        anchor - self.motion_config.window_seconds,
                        time.time(),
                    ), {
                        "windows_evaluated": len(evaluated_windows),
                        "event_receipt_delta_seconds": round(received_at - event_epoch, 3),
                    }

            remaining = deadline - time.time()
            if remaining <= 0:
                break
            if self._stop.wait(min(0.2, remaining)):
                break

        diagnostics = {
            "windows_evaluated": len(evaluated_windows),
            "event_receipt_delta_seconds": round(received_at - event_epoch, 3),
        }
        if best_result is None:
            result = MotionQualificationResult(
                True,
                1.0,
                0.0,
                "insufficient_frames",
                len(samples),
                {},
            )
            return self._with_mog2_evidence(
                result,
                anchor - self.motion_config.window_seconds,
                time.time(),
            ), diagnostics
        if best_result.score == 0.0 and not best_result.features.get("global_change"):
            result = MotionQualificationResult(
                True,
                0.0,
                best_result.threshold,
                "no_temporal_signal",
                best_result.frame_count,
                best_result.features,
            )
            return self._with_mog2_evidence(
                result,
                anchor - self.motion_config.window_seconds,
                time.time(),
            ), diagnostics
        return self._with_mog2_evidence(
            best_result,
            anchor - self.motion_config.window_seconds,
            time.time(),
        ), diagnostics

    def _run_motion_events(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._motion_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if first is None or self._stop.is_set():
                return

            triggers = [first]
            quiet_deadline = time.monotonic() + self.motion_config.burst_quiet_seconds
            hard_deadline = time.monotonic() + max(2.0, self.motion_config.burst_quiet_seconds * 4)
            while not self._stop.is_set():
                remaining = min(quiet_deadline, hard_deadline) - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._motion_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    return
                triggers.append(item)
                quiet_deadline = min(
                    hard_deadline,
                    time.monotonic() + self.motion_config.burst_quiet_seconds,
                )

            representative = min(triggers, key=lambda item: item["event_at"])
            event_at = representative["event_at"]
            received_at = min(float(item.get("received_at") or time.time()) for item in triggers)

            mode, sensitivity, frame_width = self._motion_settings()
            rescue_enabled, rescue_margin = self._motion_rescue_settings()
            priority = any(self._priority_motion_topic(str(item["topic"])) for item in triggers)
            diagnostics: dict[str, Any] = {
                "windows_evaluated": 0,
                "event_receipt_delta_seconds": round(received_at - event_at.timestamp(), 3),
            }
            if mode == "off":
                result = MotionQualificationResult(True, 1.0, 0.0, "disabled", 0, {})
            elif priority:
                result = MotionQualificationResult(True, 1.0, 0.0, "priority_topic", 0, {})
            else:
                result, diagnostics = self._qualify_motion_burst(event_at, received_at, sensitivity)

            borderline_candidate = bool(
                rescue_enabled
                and not result.accepted
                and result.score >= max(0.0, result.threshold - rescue_margin)
            )
            qualification = {
                **result.as_dict(),
                **diagnostics,
                "mode": mode,
                "sensitivity": sensitivity,
                "frame_width": frame_width,
                "borderline_rescue_enabled": rescue_enabled,
                "borderline_margin": rescue_margin,
                "borderline_candidate": borderline_candidate,
                "trigger_count": len(triggers),
                "would_suppress": bool(mode == "audit" and not result.accepted),
            }
            effective_accepted = mode != "enforce" or result.accepted or borderline_candidate
            qualification["effective_accepted"] = effective_accepted
            with self._motion_stats_lock:
                self._motion_stats["bursts"] += 1
                if priority:
                    self._motion_stats["priority_bypasses"] += 1
                if result.reason == "insufficient_frames":
                    self._motion_stats["insufficient_frames"] += 1
                if result.reason == "no_temporal_signal":
                    self._motion_stats["inconclusive"] += 1
                if result.accepted:
                    self._motion_stats["passed"] += 1
                elif mode == "enforce" and not borderline_candidate:
                    self._motion_stats["suppressed"] += 1
                elif mode == "audit":
                    self._motion_stats["audit_rejected"] += 1
                self._motion_stats["last_result"] = qualification

            if self.event_callback:
                self.event_callback("motion_qualification", {
                    "camera_id": self.camera.id,
                    "timestamp": event_at.isoformat(),
                    **qualification,
                })
            if not effective_accepted:
                snapshot_path = self._sample_rejected_motion(event_at, result)
                audit = self.events.add_motion_audit(
                    camera_id=self.camera.id,
                    snapshot_path=snapshot_path,
                    created_at=event_at.isoformat(),
                    mode=mode,
                    sensitivity=sensitivity,
                    score=result.score,
                    threshold=result.threshold,
                    reason=result.reason,
                    object_detected=None,
                    trigger_count=len(triggers),
                    features=result.features,
                )
                if self.event_callback:
                    self.event_callback("motion_audit", audit)
                continue

            outcome = self._process_motion_event(
                str(representative["topic"]),
                str(representative["message"]),
                event_at,
                qualification,
            )
            found_object = bool(outcome.get("object_detected"))
            if borderline_candidate and found_object:
                with self._motion_stats_lock:
                    self._motion_stats["borderline_rescues"] = self._motion_stats.get("borderline_rescues", 0) + 1
            elif mode == "enforce" and borderline_candidate:
                with self._motion_stats_lock:
                    self._motion_stats["suppressed"] += 1
            if mode == "audit" and not result.accepted and found_object:
                with self._motion_stats_lock:
                    self._motion_stats["audit_object_matches"] += 1
            if mode in {"audit", "enforce"} and not result.accepted:
                audit = self.events.add_motion_audit(
                    event_id=int(outcome["event_id"]),
                    camera_id=self.camera.id,
                    snapshot_path=str(outcome.get("snapshot_path") or ""),
                    created_at=event_at.isoformat(),
                    mode=mode,
                    sensitivity=sensitivity,
                    score=result.score,
                    threshold=result.threshold,
                    reason=result.reason,
                    object_detected=found_object,
                    trigger_count=len(triggers),
                    features=result.features,
                )
                if self.event_callback:
                    self.event_callback("motion_audit", audit)

    def _sample_rejected_motion(self, event_at: datetime, result: MotionQualificationResult) -> str:
        if self.motion_config.rejected_sample_rate <= 0 or random.random() > self.motion_config.rejected_sample_rate:
            return ""
        frame = self._get_latest_frame("live")
        if frame is None:
            return ""
        directory = self.storage_dir / "motion_samples" / self.camera.id
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stamp = event_at.strftime("%Y%m%d-%H%M%S-%f")
            path = directory / f"{stamp}-{result.score:.3f}-{result.reason}.jpg"
            if cv2.imwrite(str(path), frame):
                samples = sorted(directory.glob("*.jpg"), key=lambda item: item.stat().st_mtime)
                for stale in samples[:-100]:
                    stale.unlink(missing_ok=True)
                return str(path)
        except OSError as error:
            LOGGER.debug("failed to save rejected motion sample for %s: %s", self.camera.id, error)
        return ""

    def _process_motion_event(
        self,
        topic: str,
        message: str,
        event_at: datetime,
        qualification: dict[str, Any],
    ) -> dict[str, Any]:
        frame, objects, recording_path = self._recorded_motion_frame(event_at)
        snapshot_path = ""
        if frame is not None:
            snapshot_path = self._write_snapshot(frame)
        else:
            objects = [{"status": "no_recorded_frame"}]
        eligible_objects = [
            detected for detected in objects
            if detected.get("label") and detected.get("incident_eligible") is not False
        ]
        if qualification.get("borderline_candidate"):
            qualification["rescued_by_object"] = bool(eligible_objects)
            qualification["effective_accepted"] = bool(eligible_objects)
            qualification["would_suppress"] = not bool(eligible_objects)
        objects.append({"status": "motion_qualification", "motion_qualification": qualification})
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
        if self.event_callback:
            self.event_callback("incident", {
                "event_id": event.get("id"),
                "camera_id": self.camera.id,
                "timestamp": event_at.isoformat(),
                "kind": "motion",
            })
        detected_objects = [detected for detected in objects if detected.get("label")]
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
        return {
            "event_id": int(event["id"]),
            "snapshot_path": snapshot_path,
            "object_detected": bool(eligible_objects),
        }

    def _start_source(self, source: str) -> bool:
        source = self.camera.normalized_source(source)
        if self._stop.is_set():
            return False
        with self._frame_lock:
            if self._stop.is_set():
                return False
            thread = self._source_threads.get(source)
            if thread is not None and thread.is_alive():
                return True
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
        return True

    def _source_is_idle(self, source: str) -> bool:
        if source == "live":
            return False
        with self._frame_lock:
            last_access = self._source_last_access.get(source)
        return last_access is None or time.monotonic() - last_access >= MAIN_SOURCE_IDLE_SECONDS

    def _source_finished(self, source: str) -> None:
        current = threading.current_thread()
        with self._frame_lock:
            if self._source_threads.get(source) is not current:
                return
            self._source_threads.pop(source, None)
            self._source_stops.pop(source, None)
            self._source_last_access.pop(source, None)
            if source != "live":
                self._source_frames.pop(source, None)
                self._source_frame_at.pop(source, None)
                self._source_frame_monotonic.pop(source, None)
                self._source_errors.pop(source, None)
            else:
                self._thread = None

    def _run_source(self, source: str, stop_event: threading.Event) -> None:
        try:
            while not self._stop.is_set() and not stop_event.is_set():
                if self._source_is_idle(source):
                    return
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
                            if self._source_is_idle(source):
                                return
                            ok, frame = capture.read()
                            if not ok:
                                self._set_source_error(source, "stream read failed")
                                break
                            stamp = datetime.now(timezone.utc).isoformat()
                            frame_clock = time.monotonic()
                            with self._frame_lock:
                                self._source_frames[source] = frame.copy()
                                self._source_frame_at[source] = stamp
                                self._source_frame_monotonic[source] = frame_clock
                                if source == "live":
                                    self.last_frame_at = stamp
                            if source == "live":
                                self._remember_motion_frame(frame, frame_clock)
                except Exception as exc:
                    self._set_source_error(source, f"stream error: {str(exc)[:160]}")
                    LOGGER.warning("camera stream failed for %s/%s: %s", self.camera.id, source, exc)
                finally:
                    capture.release()
                if stop_event.wait(1.0):
                    break
        finally:
            self._source_finished(source)

    def _set_source_error(self, source: str, message: str) -> None:
        with self._frame_lock:
            self._source_errors[source] = message
            if source == "live":
                self.last_error = message

    def _get_latest_frame(self, source: str = "live") -> Any:
        source = self.camera.normalized_source(source)
        if self._stop.is_set():
            return None
        with self._frame_lock:
            self._source_last_access[source] = time.monotonic()
        if not self._start_source(source):
            return None
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

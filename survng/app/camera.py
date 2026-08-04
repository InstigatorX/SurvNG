from __future__ import annotations

import hashlib
import logging
import queue
import random
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from .config import CameraConfig, DetectionZone, MotionQualificationConfig
from .image_storage import DurableImageWriter
from .onvif_events import OnvifEventListener
from .motion import MotionQualificationResult
from .object_tracking import ObjectTrackingSessionFactory
from .tracking_comparison import sampled_video_frames
from .security import redact_secret_text
from .motion_pipeline import (
    MotionContext,
    MotionDecisionHandlerFactory,
    MotionDebugSnapshotStore,
    MotionEvidenceRepository,
    MotionPipeline,
    MotionScoring,
    RecordedMotionObjectDetectorFactory,
    resolved_trigger_mode,
)

LOGGER = logging.getLogger(__name__)
MOTION_THREAD_STOP_TIMEOUT_SECONDS = 22.0
# OpenCV's FFmpeg calls cannot be interrupted safely from another thread.
# Keep their own deadlines below CameraWorker.stop()'s join budget so a stop
# request can always regain control after a blocked open/read operation.
CAPTURE_OPEN_TIMEOUT_MS = 3000
CAPTURE_READ_TIMEOUT_MS = 5000
CAPTURE_DECODER_THREADS = 1
CAPTURE_STOP_TIMEOUT_SECONDS = 8.0
CAPTURE_RETRY_INITIAL_SECONDS = 1.0
CAPTURE_RETRY_MAX_SECONDS = 30.0
CAPTURE_OPEN_LOCK_POLL_SECONDS = 0.1
CAPTURE_OPEN_CONCURRENCY = 2
CAPTURE_OPEN_SLOTS = threading.BoundedSemaphore(CAPTURE_OPEN_CONCURRENCY)
FRAME_STALE_SECONDS = 10.0
MAIN_SOURCE_IDLE_SECONDS = 20.0
TRACKING_CATCHUP_SECONDS = 10.0
TRACKING_CATCHUP_FRAME_WIDTH = 640
MOTION_QUEUE_SIZE = 32
MOTION_ANALYSIS_QUEUE_SIZE = 1
MOTION_EVENT_MAX_RETRIES = 2
FUSION_STALE_TOLERANCE_SECONDS = 5.0
INCIDENT_ACTIVITY_REASONS = frozenset({"event_state_active", "event_state_cooldown"})


class CameraWorker:
    def __init__(
        self,
        camera: CameraConfig,
        storage_dir: Path,
        motion_config: MotionQualificationConfig | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        motion_pipeline: MotionPipeline,
        motion_observation_pipeline: MotionPipeline,
        motion_fusion_pipeline: MotionPipeline,
        motion_evidence: MotionEvidenceRepository,
        motion_pipeline_origins: dict[str, str],
        motion_decision_handler_factory: MotionDecisionHandlerFactory,
        motion_object_detector_factory: RecordedMotionObjectDetectorFactory,
        object_tracking_session_factory: ObjectTrackingSessionFactory,
        motion_analysis_limiter: threading.BoundedSemaphore,
        image_writer: DurableImageWriter,
        onvif_cache_dir: Path | None = None,
    ) -> None:
        self.camera = camera
        self.storage_dir = storage_dir
        self.motion_config = motion_config or MotionQualificationConfig()
        self.event_callback = event_callback
        self.motion_pipeline = motion_pipeline
        self.motion_observation_pipeline = motion_observation_pipeline
        self.motion_fusion_pipeline = motion_fusion_pipeline
        self.motion_evidence = motion_evidence
        self.motion_pipeline_origins = dict(motion_pipeline_origins)
        self.motion_analysis_limiter = motion_analysis_limiter
        self.image_writer = image_writer
        self.motion_debug = MotionDebugSnapshotStore()
        self._motion_debug_last_run = 0.0
        self.motion_object_detector = motion_object_detector_factory.create(
            camera=camera,
            live_frame_provider=lambda: self._get_latest_frame(),
        )
        self.object_tracking = object_tracking_session_factory.create(
            camera=camera,
            frame_provider=self._get_latest_tracking_frame_with_fallback,
            catchup_frame_provider=self._recorded_tracking_frames,
        )
        self.motion_decision_handler = motion_decision_handler_factory.create(
            camera_id=camera.id,
            detection_provider=lambda event_at: self._recorded_motion_frame(event_at),
            snapshot_writer=lambda frame, event_at: self._write_snapshot(frame, event_at),
            event_callback=self._publish_event_safely if event_callback is not None else None,
        )
        self.snapshots_dir = storage_dir / "snapshots" / camera.id
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._stop.set()
        self._enabled = False
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._motion_analysis_lock = threading.Lock()
        self._motion_fusion_lock = threading.Lock()
        self._motion_fusion_last_at = 0.0
        self._source_threads: dict[str, threading.Thread] = {}
        self._source_stops: dict[str, threading.Event] = {}
        self._source_frames: dict[str, Any] = {}
        self._source_frame_at: dict[str, str] = {}
        self._source_frame_epoch: dict[str, float] = {}
        self._source_frame_monotonic: dict[str, float] = {}
        self._source_frame_dimensions: dict[str, dict[str, int]] = {}
        self._source_frame_times: dict[str, deque[float]] = {
            "live": deque(maxlen=600),
            "main": deque(maxlen=600),
        }
        self._source_capture_stats: dict[str, dict[str, int]] = {
            "live": {"frames_received": 0, "read_failures": 0, "open_failures": 0, "reconnects": 0},
            "main": {"frames_received": 0, "read_failures": 0, "open_failures": 0, "reconnects": 0},
        }
        self._source_last_access: dict[str, float] = {}
        self._source_errors: dict[str, str] = {}
        tracking_buffer_size = max(
            4,
            round(self.object_tracking.config.sample_fps * TRACKING_CATCHUP_SECONDS) + 2,
        )
        self._tracking_frames: deque[tuple[float, np.ndarray]] = deque(
            maxlen=tracking_buffer_size
        )
        self._tracking_last_sample_epoch = 0.0
        ring_size = max(
            12,
            round(
                self.motion_config.sample_fps
                * (self.motion_config.window_seconds + self.motion_config.post_trigger_seconds + 3.0)
            ),
        )
        self._motion_frames: deque[tuple[float, np.ndarray]] = deque(maxlen=ring_size)
        self._motion_last_sample = 0.0
        self._motion_last_continuous_result: MotionQualificationResult | None = None
        self._motion_analysis_last_processed_at = 0.0
        self._motion_primary_last_processed_at = 0.0
        self._motion_analysis_queue: queue.Queue[float | None] = queue.Queue(
            maxsize=MOTION_ANALYSIS_QUEUE_SIZE
        )
        self._motion_analysis_thread: threading.Thread | None = None
        self._motion_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=MOTION_QUEUE_SIZE)
        self._motion_retry_batches: deque[dict[str, Any]] = deque()
        self._motion_thread: threading.Thread | None = None
        self._active_motion_triggers: list[dict[str, Any]] | None = None
        self._active_incident_event_id: int | None = None
        self._adaptive_trigger_pending = False
        self._adaptive_last_completed_at = 0.0
        self._priority_motion_times: deque[float] = deque(maxlen=16)
        self._camera_motion_times: deque[float] = deque(maxlen=32)
        self._visual_backup_last_matched_camera_at = 0.0
        self._visual_backup_analysis_started_at = 0.0
        self._visual_backup_scene_ready = False
        self._visual_backup_stable_since = 0.0
        self._visual_backup_stable_samples = 0
        self._visual_backup_readiness_audited = False
        self._visual_backup_candidate_since = 0.0
        self._visual_backup_last_candidate_at = 0.0
        self._visual_backup_consecutive = 0
        self._visual_backup_trigger_times: deque[float] = deque(maxlen=64)
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
            "analysis_frames_dropped": 0,
            "continuous_frames": 0,
            "continuous_candidates": 0,
            "adaptive_triggers_deferred": 0,
            "visual_backup_candidates": 0,
            "visual_backup_triggers": 0,
            "visual_backup_onvif_matches": 0,
            "visual_backup_rate_limited": 0,
            "visual_backup_not_ready": 0,
            "visual_backup_uncorrelated_objects": 0,
            "analysis_worker_errors": 0,
            "event_worker_errors": 0,
            "event_callback_errors": 0,
            "event_retries": 0,
            "event_retry_drops": 0,
            "stale_fusion_samples": 0,
            "validation_failures": 0,
            "validation_fail_opens": 0,
            "audit_object_matches": 0,
            "suppression_verification_checks": 0,
            "suppression_verification_rescues": 0,
            "last_result": None,
        }
        self.last_error = ""
        self.last_frame_at = ""
        self.last_motion_at = ""
        self._detection_enabled = True
        self.onvif = OnvifEventListener(
            camera,
            self.handle_motion_event,
            cache_dir=onvif_cache_dir or storage_dir / "onvif",
        )

    def start(self) -> None:
        with self._lifecycle_lock:
            self._enabled = True
            self._stop.clear()
            self.object_tracking.set_accepting(self._detection_enabled)
            try:
                if self._motion_analysis_thread is None or not self._motion_analysis_thread.is_alive():
                    self._clear_motion_analysis_queue()
                    analysis_thread = threading.Thread(
                        target=self._run_motion_analysis,
                        name=f"motion-analysis-{self.camera.id}",
                        daemon=False,
                    )
                    self._motion_analysis_thread = analysis_thread
                    try:
                        analysis_thread.start()
                    except BaseException:
                        self._motion_analysis_thread = None
                        raise
                if not self._start_source("live"):
                    raise RuntimeError(f"camera source did not start for {self.camera.id}")
                if self._motion_thread is None or not self._motion_thread.is_alive():
                    self._clear_motion_queue()
                    motion_thread = threading.Thread(
                        target=self._run_motion_events,
                        name=f"motion-{self.camera.id}",
                        daemon=False,
                    )
                    self._motion_thread = motion_thread
                    try:
                        motion_thread.start()
                    except BaseException:
                        self._motion_thread = None
                        raise
                self.onvif.start()
            except BaseException:
                try:
                    self.stop()
                except Exception:
                    LOGGER.exception(
                        "camera startup rollback was incomplete for %s",
                        self.camera.id,
                    )
                raise

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._enabled = False
            self._stop.set()
            self._active_incident_event_id = None
            self.object_tracking.stop()
            with self._frame_lock:
                stops = list(self._source_stops.values())
                threads = list(self._source_threads.items())
            for stop_event in stops:
                stop_event.set()
            self.onvif.stop()
            self._signal_motion_analysis_stop()
            try:
                self._motion_queue.put_nowait(None)
            except queue.Full:
                pass
            motion_thread = self._motion_thread
            if motion_thread is not None:
                motion_thread.join(timeout=MOTION_THREAD_STOP_TIMEOUT_SECONDS)
                if motion_thread.is_alive():
                    LOGGER.error("motion worker did not stop for %s", self.camera.id)
            self._motion_thread = motion_thread if motion_thread is not None and motion_thread.is_alive() else None
            analysis_thread = self._motion_analysis_thread
            if analysis_thread is not None:
                analysis_thread.join(timeout=CAPTURE_STOP_TIMEOUT_SECONDS)
                if analysis_thread.is_alive():
                    LOGGER.error("motion analysis worker did not stop for %s", self.camera.id)
            self._motion_analysis_thread = (
                analysis_thread
                if analysis_thread is not None and analysis_thread.is_alive()
                else None
            )
            deadline = time.monotonic() + CAPTURE_STOP_TIMEOUT_SECONDS
            for _source, thread in threads:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            alive = [source for source, thread in threads if thread.is_alive()]
            if alive:
                logging.getLogger("uvicorn.error").error(
                    "camera capture threads did not stop for %s: %s",
                    self.camera.id,
                    ", ".join(alive),
                )
                current_frames = sys._current_frames()
                for source, thread in threads:
                    if not thread.is_alive() or thread.ident is None:
                        continue
                    frame = current_frames.get(thread.ident)
                    if frame is not None:
                        logging.getLogger("uvicorn.error").error(
                            "camera capture thread stack for %s/%s:\n%s",
                            self.camera.id,
                            source,
                            "".join(traceback.format_stack(frame)),
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
                self._source_frame_epoch.clear()
                self._source_frame_monotonic.clear()
                self._source_last_access.clear()
                self._source_errors.clear()
                self._tracking_frames.clear()
                self._tracking_last_sample_epoch = 0.0
                self._motion_frames.clear()
                self._motion_last_sample = 0.0
                self._motion_last_continuous_result = None
                self._motion_analysis_last_processed_at = 0.0
                self.last_frame_at = ""
            self.motion_evidence.clear()
            motion_workers_stopped = (
                self._motion_thread is None
                and self._motion_analysis_thread is None
            )
            if motion_workers_stopped:
                self.motion_observation_pipeline.runtime.reset()
                self.motion_pipeline.runtime.reset()
                with self._motion_fusion_lock:
                    self.motion_fusion_pipeline.runtime.reset()
                    self._motion_fusion_last_at = 0.0
            else:
                LOGGER.error(
                    "preserving motion runtime for %s because a motion worker is still active",
                    self.camera.id,
                )
            self._active_motion_triggers = None
            self._adaptive_trigger_pending = False
            self._adaptive_last_completed_at = 0.0
            self._priority_motion_times.clear()
            self._camera_motion_times.clear()
            self._visual_backup_last_matched_camera_at = 0.0
            self._visual_backup_analysis_started_at = 0.0
            self._visual_backup_scene_ready = False
            self._visual_backup_stable_since = 0.0
            self._visual_backup_stable_samples = 0
            self._visual_backup_readiness_audited = False
            self._visual_backup_candidate_since = 0.0
            self._visual_backup_last_candidate_at = 0.0
            self._visual_backup_consecutive = 0
            self._visual_backup_trigger_times.clear()
            self._thread = self._source_threads.get("live")
            shutdown_failures: list[str] = []
            if alive:
                shutdown_failures.append(f"capture sources: {', '.join(alive)}")
            if not motion_workers_stopped:
                shutdown_failures.append("motion workers")
            if self.onvif.running:
                shutdown_failures.append("ONVIF worker")
            if self.object_tracking.running():
                shutdown_failures.append("object tracking worker")
            if shutdown_failures:
                raise RuntimeError(
                    f"camera {self.camera.id} did not stop cleanly ({'; '.join(shutdown_failures)})"
                )

    def stop_onvif_events(self) -> None:
        """Release the camera's ONVIF subscription without stopping video."""
        with self._lifecycle_lock:
            self.onvif.stop()

    def close(self) -> None:
        with self._lifecycle_lock:
            active = [
                label
                for label, thread in (
                    ("motion events", self._motion_thread),
                    ("motion analysis", self._motion_analysis_thread),
                )
                if thread is not None and thread.is_alive()
            ]
            if active:
                raise RuntimeError(
                    f"cannot close camera {self.camera.id} pipelines while "
                    f"{', '.join(active)} is running"
                )
            failures: list[BaseException] = []
            for label, pipeline in (
                ("qualification", self.motion_pipeline),
                ("observation", self.motion_observation_pipeline),
                ("fusion", self.motion_fusion_pipeline),
            ):
                try:
                    pipeline.close()
                except BaseException as exc:
                    failures.append(exc)
                    LOGGER.exception(
                        "%s motion pipeline cleanup failed for %s",
                        label,
                        self.camera.id,
                    )
            if failures:
                first_error = failures[0]
                if not isinstance(first_error, Exception):
                    raise first_error
                raise RuntimeError(
                    f"one or more motion pipelines failed to close for {self.camera.id}"
                ) from first_error

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
            stream_dimensions = {
                source: dict(dimensions)
                for source, dimensions in self._source_frame_dimensions.items()
            }
            motion_buffered_frames = len(self._motion_frames)
            motion_frame_shape = list(self._motion_frames[-1][1].shape) if self._motion_frames else None
            capture_stats: dict[str, dict[str, int | float]] = {}
            sample_now = time.monotonic()
            for source in ("live", "main"):
                times = self._source_frame_times[source]
                while times and sample_now - times[0] > 10.0:
                    times.popleft()
                fps = (
                    (len(times) - 1) / max(0.001, times[-1] - times[0])
                    if len(times) >= 2 and sample_now - times[-1] <= FRAME_STALE_SECONDS
                    else 0.0
                )
                capture_stats[source] = {
                    **self._source_capture_stats[source],
                    "fps": round(fps, 2),
                }
        evidence_status = self.motion_evidence.status()
        mog2_status = evidence_status.get("mog2", {})
        now = time.monotonic()
        live_age = max(0.0, now - live_frame_clock) if live_frame_clock is not None else None
        main_age = max(0.0, now - main_frame_clock) if main_frame_clock is not None else None
        connected = bool(enabled and live_age is not None and live_age <= FRAME_STALE_SECONDS)
        mode, sensitivity, frame_width = self._motion_settings()
        stationary_object_tolerance = self._stationary_object_tolerance()
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
            "capture_stats": capture_stats,
            "stream_dimensions": stream_dimensions,
            "onvif_enabled": self.camera.onvif.enabled,
            "onvif_connected": self.onvif.connected,
            "onvif_last_event_at": self.onvif.last_event_at,
            "onvif_last_camera_event_at": self.onvif.last_camera_event_at,
            "onvif_last_motion_event_at": self.onvif.last_motion_event_at,
            "last_motion_at": self.last_motion_at,
            "detection_enabled": self._detection_enabled,
            "object_tracking": self.object_tracking.status(),
            "onvif_last_error": self.onvif.last_error,
            "onvif_last_connected_at": self.onvif.last_connected_at,
            "onvif_last_poll_success_at": self.onvif.last_poll_success_at,
            "onvif_last_poll_error": self.onvif.last_poll_error,
            "onvif_last_poll_error_at": self.onvif.last_poll_error_at,
            "onvif_retry_attempts": self.onvif.retry_attempts,
            "onvif_poll_timeouts": self.onvif.poll_timeouts,
            "onvif_poll_errors": self.onvif.poll_errors,
            "onvif_resubscriptions": self.onvif.resubscriptions,
            "onvif_notifications_received": self.onvif.notifications_received,
            "onvif_motion_events_received": self.onvif.motion_events_received,
            "onvif_inactive_motion_events": self.onvif.inactive_motion_events,
            "onvif_unrecognized_notifications": self.onvif.unrecognized_notifications,
            "onvif_callback_errors": self.onvif.callback_errors,
            "onvif_renewal_attempts": self.onvif.renewal_attempts,
            "onvif_renewals": self.onvif.renewals,
            "onvif_renewal_errors": self.onvif.renewal_errors,
            "onvif_last_renewed_at": self.onvif.last_renewed_at,
            "onvif_subscription_current_time": self.onvif.subscription_current_time,
            "onvif_subscription_termination_time": self.onvif.subscription_termination_time,
            "onvif_subscription_lifetime_seconds": self.onvif.subscription_lifetime_seconds,
            "motion_qualification": {
                **motion_stats,
                "mode": mode,
                "sensitivity": sensitivity,
                "stationary_object_tolerance": stationary_object_tolerance,
                "frame_width": frame_width,
                "camera_mode_background_fps": self.motion_config.camera_mode_background_fps,
                "visual_backup": {
                    "enabled": mode == "camera_rescue",
                    "warmup_seconds": self.motion_config.visual_backup_warmup_seconds,
                    "grace_seconds": self.motion_config.visual_backup_grace_seconds,
                    "minimum_score": self.motion_config.visual_backup_min_score,
                    "score_margin": self.motion_config.visual_backup_score_margin,
                    "minimum_consecutive": self.motion_config.visual_backup_min_consecutive,
                    "cooldown_seconds": self.motion_config.visual_backup_cooldown_seconds,
                    "maximum_triggers_5m": self.motion_config.visual_backup_max_triggers_5m,
                    "scene_ready": self._visual_backup_scene_ready,
                    "stable_samples": self._visual_backup_stable_samples,
                },
                "suppression_verification_rate": self._suppression_verification_rate(),
                "borderline_rescue_enabled": rescue_enabled,
                "borderline_margin": rescue_margin,
                "mog2_audit_enabled": bool(mog2_status.get("enabled", False)),
                "mog2_history_seconds": self.motion_config.mog2_history_seconds,
                "mog2_last": mog2_status.get("last"),
                "evidence_sources": evidence_status,
                "pipeline_origins": dict(self.motion_pipeline_origins),
                "queue_depth": self._motion_queue.qsize(),
                "retry_queue_depth": len(self._motion_retry_batches),
                "analysis_queue_depth": self._motion_analysis_queue.qsize(),
                "analysis_worker_running": bool(
                    self._motion_analysis_thread is not None
                    and self._motion_analysis_thread.is_alive()
                ),
                "event_worker_running": bool(
                    self._motion_thread is not None
                    and self._motion_thread.is_alive()
                ),
                "continuous_last_result": (
                    self._motion_last_continuous_result.as_dict()
                    if self._motion_last_continuous_result is not None
                    else None
                ),
                "buffered_frames": motion_buffered_frames,
                "frame_shape": motion_frame_shape,
                "pipeline": self.motion_pipeline.status(),
                "observation_pipeline": self.motion_observation_pipeline.status(),
                "fusion_pipeline": self.motion_fusion_pipeline.status(),
                "debug": self.motion_debug.status(),
            },
        }

    def update_zones(self, zones: list[DetectionZone]) -> None:
        next_zones = [zone.model_copy(deep=True) for zone in zones]
        with self._lifecycle_lock:
            self.camera.zones = next_zones

    def set_detection_enabled(self, enabled: bool) -> None:
        self._detection_enabled = bool(enabled)
        self.object_tracking.set_accepting(
            self._detection_enabled and not self._stop.is_set()
        )

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

    def _publish_event_safely(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback(event_type, payload)
        except Exception:
            with self._motion_stats_lock:
                self._motion_stats["event_callback_errors"] += 1
            LOGGER.exception(
                "camera event callback failed for %s event=%s",
                self.camera.id,
                event_type,
            )

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

        self._observe_motion_event(topic, message, event_at, received_at)
        configured_mode = self._motion_settings()[0]
        if configured_mode == "adaptive" and not topic.startswith("manual"):
            # ONVIF remains useful diagnostic evidence in visual-triggered mode,
            # but it is never allowed to create an object-detection job.
            return
        if self._priority_motion_topic(topic):
            self._remember_priority_motion(received_at)
        if not topic.startswith("manual"):
            with self._motion_stats_lock:
                self._camera_motion_times.append(received_at)

        self._publish_event_safely("motion", {
            "camera_id": self.camera.id,
            "timestamp": event_at.isoformat(),
            "source": "manual" if topic.startswith("manual") else "onvif",
        })

        self._enqueue_motion_trigger({
            "topic": topic,
            "message": message,
            "event_at": event_at,
            "received_at": received_at,
        })

    def _enqueue_motion_trigger(
        self,
        trigger: dict[str, Any],
        *,
        evict_oldest: bool = True,
    ) -> bool:
        with self._motion_stats_lock:
            self._motion_stats["triggers"] += 1
        try:
            self._motion_queue.put_nowait(trigger)
            return True
        except queue.Full:
            if not evict_oldest:
                with self._motion_stats_lock:
                    self._motion_stats["dropped_triggers"] += 1
                return False
            try:
                self._motion_queue.get_nowait()
            except queue.Empty:
                pass
            with self._motion_stats_lock:
                self._motion_stats["dropped_triggers"] += 1
            try:
                self._motion_queue.put_nowait(trigger)
                return True
            except queue.Full:
                return False

    def _clear_motion_queue(self) -> None:
        self._motion_retry_batches.clear()
        while True:
            try:
                self._motion_queue.get_nowait()
            except queue.Empty:
                return

    def _clear_motion_analysis_queue(self) -> None:
        while True:
            try:
                self._motion_analysis_queue.get_nowait()
            except queue.Empty:
                return

    def _signal_motion_analysis_stop(self) -> None:
        self._clear_motion_analysis_queue()
        try:
            self._motion_analysis_queue.put_nowait(None)
        except queue.Full:
            pass

    def _schedule_motion_analysis(self, captured_at: float) -> None:
        if self._stop.is_set():
            return
        try:
            self._motion_analysis_queue.put_nowait(captured_at)
            return
        except queue.Full:
            try:
                self._motion_analysis_queue.get_nowait()
            except queue.Empty:
                pass
        with self._motion_stats_lock:
            self._motion_stats["analysis_frames_dropped"] += 1
        try:
            self._motion_analysis_queue.put_nowait(captured_at)
        except queue.Full:
            pass

    def _remember_motion_frame(self, frame: np.ndarray, frame_clock: float) -> None:
        if not self._frame_motion_analysis_required():
            return
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
        with self._frame_lock:
            self._motion_frames.append((captured_at, gray))
        self._schedule_motion_analysis(captured_at)

    def _run_motion_analysis(self) -> None:
        while not self._stop.is_set():
            try:
                _scheduled_at = self._motion_analysis_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if _scheduled_at is None or self._stop.is_set():
                return
            try:
                with self.motion_analysis_limiter:
                    if self._stop.is_set():
                        return
                    with self._frame_lock:
                        if not self._motion_frames:
                            continue
                        captured_at, frame = self._motion_frames[-1]
                        frame = frame.copy()
                    if captured_at <= self._motion_analysis_last_processed_at:
                        continue
                    self._motion_analysis_last_processed_at = captured_at
                    if self.motion_observation_pipeline.handles_observation("frame"):
                        observation = MotionContext(
                            camera_id=self.camera.id,
                            captured_at=captured_at,
                            original_frame=frame,
                            configuration={"observation_kind": "frame"},
                            runtime=self.motion_observation_pipeline.runtime,
                        )
                        self.motion_observation_pipeline.process(observation)
                    if (
                        self._continuous_primary_analysis_required()
                        and self._continuous_primary_analysis_due(captured_at)
                    ):
                        self._analyze_continuous_motion(captured_at)
                    elif self.motion_debug.enabled() and time.monotonic() - self._motion_debug_last_run >= 1.0:
                        self._motion_debug_last_run = time.monotonic()
                        self._capture_motion_debug(captured_at)
            except Exception:
                with self._motion_stats_lock:
                    self._motion_stats["analysis_worker_errors"] += 1
                LOGGER.exception("motion analysis cycle failed for %s", self.camera.id)

    def _analyze_continuous_motion(self, captured_at: float) -> None:
        mode, sensitivity, _frame_width = self._motion_settings()
        with self._frame_lock:
            samples = [(timestamp, frame.copy()) for timestamp, frame in list(self._motion_frames)[-2:]]
        if len(samples) < 2:
            return
        try:
            with self._motion_analysis_lock:
                result = self._run_motion_pipeline(
                    [frame for _timestamp, frame in samples],
                    sensitivity,
                    captured_at,
                    [timestamp for timestamp, _frame in samples],
                    isolated=False,
                    capture_debug=self.motion_debug.enabled(),
                    include_telemetry=False,
                )
        except Exception as error:
            LOGGER.warning("continuous motion analysis failed for %s: %s", self.camera.id, error)
            return
        self._motion_last_continuous_result = result
        self._motion_primary_last_processed_at = captured_at
        with self._motion_stats_lock:
            self._motion_stats["continuous_frames"] += 1
            self._motion_stats["continuous_candidates"] += int(result.accepted)
        trigger_mode = self._trigger_mode()
        if trigger_mode == "camera_rescue":
            self._consider_visual_backup(result, samples, captured_at)
            return
        if trigger_mode != "adaptive" or not self._detection_enabled:
            return
        fused = self._with_source_evidence(
            result,
            samples[0][0],
            captured_at,
            include_telemetry=False,
            require_primary_trigger=True,
        )
        self._motion_last_continuous_result = fused
        if not fused.accepted or not self._reserve_adaptive_trigger(captured_at):
            return
        event_at = datetime.fromtimestamp(captured_at, timezone.utc)
        queued = self._enqueue_motion_trigger({
            "topic": "adaptive/motion",
            "message": "adaptive motion transition",
            "event_at": event_at,
            "received_at": captured_at,
            "prequalified": fused,
        }, evict_oldest=False)
        if not queued:
            self._defer_adaptive_trigger(captured_at)
            return
        self.last_motion_at = event_at.isoformat()
        self._publish_event_safely("motion", {
            "camera_id": self.camera.id,
            "timestamp": event_at.isoformat(),
            "source": "adaptive",
        })

    def _reset_visual_backup_candidate(self) -> None:
        self._visual_backup_candidate_since = 0.0
        self._visual_backup_last_candidate_at = 0.0
        self._visual_backup_consecutive = 0

    def _visual_backup_readiness(
        self,
        result: MotionQualificationResult,
        captured_at: float,
    ) -> bool:
        """Require a quiet post-warmup baseline before EMA may rescue events."""
        if (
            self._visual_backup_analysis_started_at <= 0.0
            or captured_at < self._visual_backup_analysis_started_at
        ):
            self._visual_backup_analysis_started_at = captured_at
            self._visual_backup_scene_ready = False
            self._visual_backup_stable_since = 0.0
            self._visual_backup_stable_samples = 0
        if self._visual_backup_scene_ready:
            return True
        if (
            captured_at - self._visual_backup_analysis_started_at
            < self.motion_config.visual_backup_warmup_seconds
        ):
            self._visual_backup_stable_since = 0.0
            self._visual_backup_stable_samples = 0
            return False

        stable_result = bool(
            not result.accepted
            and result.reason not in {
                "global_illumination_change",
                "insufficient_frames",
                "validation_unavailable_fail_open",
            }
        )
        if not stable_result:
            self._visual_backup_stable_since = 0.0
            self._visual_backup_stable_samples = 0
            return False
        if self._visual_backup_stable_since <= 0.0:
            self._visual_backup_stable_since = captured_at
        self._visual_backup_stable_samples += 1
        required_samples = max(3, int(self.motion_config.visual_backup_min_consecutive))
        required_seconds = max(1.5, float(self.motion_config.visual_backup_grace_seconds))
        if (
            self._visual_backup_stable_samples >= required_samples
            and captured_at - self._visual_backup_stable_since >= required_seconds
        ):
            self._visual_backup_scene_ready = True
            self._reset_visual_backup_candidate()
        return self._visual_backup_scene_ready

    def _consider_visual_backup(
        self,
        result: MotionQualificationResult,
        samples: list[tuple[float, np.ndarray]],
        captured_at: float,
    ) -> None:
        if not self._detection_enabled:
            self._reset_visual_backup_candidate()
            return
        scene_ready = self._visual_backup_readiness(result, captured_at)
        required_score = max(
            float(self.motion_config.visual_backup_min_score),
            float(result.threshold) + float(self.motion_config.visual_backup_score_margin),
        )
        strong_candidate = bool(
            result.accepted
            and result.score >= required_score
            and result.reason not in {
                "global_illumination_change",
                "insect_like_motion",
                "persistent_scene_motion",
                "stationary_foreground",
                "stationary_region",
            }
        )
        if not strong_candidate:
            self._reset_visual_backup_candidate()
            return
        if not scene_ready:
            with self._motion_stats_lock:
                self._motion_stats["visual_backup_not_ready"] += 1
            self._record_visual_backup_readiness_audit(result, captured_at)
            self._reset_visual_backup_candidate()
            return

        expected_interval = 1.0 / max(
            0.5,
            min(
                self.motion_config.sample_fps,
                self.motion_config.camera_mode_background_fps,
            ),
        )
        if (
            self._visual_backup_last_candidate_at > 0.0
            and captured_at - self._visual_backup_last_candidate_at
            > expected_interval * 2.5
        ):
            self._reset_visual_backup_candidate()
        if self._visual_backup_candidate_since <= 0.0:
            self._visual_backup_candidate_since = captured_at
        self._visual_backup_last_candidate_at = captured_at
        self._visual_backup_consecutive += 1
        with self._motion_stats_lock:
            self._motion_stats["visual_backup_candidates"] += 1

        if (
            self._visual_backup_consecutive
            < self.motion_config.visual_backup_min_consecutive
            or captured_at - self._visual_backup_candidate_since
            < self.motion_config.visual_backup_grace_seconds
        ):
            return
        with self._motion_stats_lock:
            matched_camera_at = max(
                (
                    observed_at
                    for observed_at in self._camera_motion_times
                    if 0.0 <= captured_at - observed_at
                    <= self.motion_config.visual_backup_cooldown_seconds
                ),
                default=0.0,
            )
            camera_notice_recent = matched_camera_at > 0.0
            if matched_camera_at > self._visual_backup_last_matched_camera_at:
                self._motion_stats["visual_backup_onvif_matches"] += 1
                self._visual_backup_last_matched_camera_at = matched_camera_at
        if camera_notice_recent:
            self._reset_visual_backup_candidate()
            return
        if not self._reserve_visual_backup_trigger(captured_at):
            self._reset_visual_backup_candidate()
            return

        fused = self._with_source_evidence(
            result,
            samples[0][0],
            captured_at,
            include_telemetry=False,
            require_primary_trigger=True,
        )
        self._motion_last_continuous_result = fused
        if not fused.accepted:
            self._defer_adaptive_trigger(captured_at)
            self._reset_visual_backup_candidate()
            return
        features = {
            **fused.features,
            "visual_backup": True,
            "visual_backup_required_score": round(required_score, 4),
            "visual_backup_consecutive": self._visual_backup_consecutive,
            "visual_backup_grace_seconds": self.motion_config.visual_backup_grace_seconds,
        }
        fused = MotionQualificationResult(
            accepted=fused.accepted,
            score=fused.score,
            threshold=fused.threshold,
            reason=fused.reason,
            frame_count=fused.frame_count,
            features=features,
            telemetry=dict(fused.telemetry),
        )
        event_at = datetime.fromtimestamp(captured_at, timezone.utc)
        queued = self._enqueue_motion_trigger({
            "topic": "adaptive/visual_backup",
            "message": "adaptive visual backup after missing camera notice",
            "event_at": event_at,
            "received_at": captured_at,
            "prequalified": fused,
        }, evict_oldest=False)
        if not queued:
            self._defer_adaptive_trigger(captured_at)
            self._reset_visual_backup_candidate()
            return
        with self._motion_stats_lock:
            self._motion_stats["visual_backup_triggers"] += 1
            self._visual_backup_trigger_times.append(captured_at)
        self.last_motion_at = event_at.isoformat()
        self._publish_event_safely("motion", {
            "camera_id": self.camera.id,
            "timestamp": event_at.isoformat(),
            "source": "visual_backup",
        })
        self._reset_visual_backup_candidate()

    def _record_visual_backup_readiness_audit(
        self,
        result: MotionQualificationResult,
        captured_at: float,
    ) -> None:
        if self._visual_backup_readiness_audited:
            return
        self._visual_backup_readiness_audited = True
        event_at = datetime.fromtimestamp(captured_at, timezone.utc)
        try:
            self.motion_decision_handler.record_audit(
                snapshot_path=self._sample_rejected_motion(event_at, result),
                event_at=event_at,
                mode="camera_rescue",
                sensitivity=self._motion_settings()[1],
                score=result.score,
                threshold=result.threshold,
                reason="startup_not_ready",
                object_detected=None,
                trigger_count=0,
                features={
                    **self._audit_features(result),
                    "visual_backup_scene_ready": False,
                    "visual_backup_warmup_seconds": self.motion_config.visual_backup_warmup_seconds,
                },
                category="visual_backup",
            )
        except Exception:
            LOGGER.exception(
                "failed to record visual backup readiness audit for %s",
                self.camera.id,
            )

    def _reserve_visual_backup_trigger(self, captured_at: float) -> bool:
        cutoff = captured_at - 300.0
        with self._motion_stats_lock:
            while (
                self._visual_backup_trigger_times
                and self._visual_backup_trigger_times[0] < cutoff
            ):
                self._visual_backup_trigger_times.popleft()
            limited = bool(
                self._adaptive_trigger_pending
                or (
                    self._adaptive_last_completed_at > 0.0
                    and captured_at - self._adaptive_last_completed_at
                    < self.motion_config.visual_backup_cooldown_seconds
                )
                or len(self._visual_backup_trigger_times)
                >= self.motion_config.visual_backup_max_triggers_5m
            )
            if limited:
                self._motion_stats["visual_backup_rate_limited"] += 1
                return False
            self._adaptive_trigger_pending = True
            return True

    def _reserve_adaptive_trigger(self, captured_at: float) -> bool:
        rearm_seconds = max(
            5.0,
            self.motion_config.window_seconds
            + self.motion_config.post_trigger_seconds
            + self.motion_config.burst_quiet_seconds,
        )
        priority_duplicate = self._matches_recent_priority_motion(captured_at)
        with self._motion_stats_lock:
            if (
                self._adaptive_trigger_pending
                or priority_duplicate
                or captured_at - self._adaptive_last_completed_at < rearm_seconds
            ):
                self._motion_stats["adaptive_triggers_deferred"] += 1
                return False
            self._adaptive_trigger_pending = True
            return True

    def _priority_dedup_seconds(self) -> float:
        return max(
            2.0,
            self.motion_config.post_trigger_seconds
            + self.motion_config.burst_quiet_seconds,
        )

    def _remember_priority_motion(self, observed_at: float) -> None:
        with self._motion_stats_lock:
            self._priority_motion_times.append(observed_at)

    def _matches_recent_priority_motion(self, event_at: float) -> bool:
        tolerance = self._priority_dedup_seconds()
        with self._motion_stats_lock:
            return any(
                abs(event_at - priority_at) <= tolerance
                for priority_at in self._priority_motion_times
            )

    def _defer_adaptive_trigger(self, captured_at: float) -> None:
        with self._motion_stats_lock:
            self._adaptive_trigger_pending = False
            self._adaptive_last_completed_at = captured_at

    def _complete_adaptive_trigger(self, triggers: list[dict[str, Any]]) -> None:
        if not any(str(item.get("topic") or "").startswith("adaptive/") for item in triggers):
            return
        with self._motion_stats_lock:
            self._adaptive_trigger_pending = False
            self._adaptive_last_completed_at = time.time()

    def _capture_motion_debug(self, captured_at: float) -> None:
        with self._frame_lock:
            samples = [
                (timestamp, frame.copy())
                for timestamp, frame in self._motion_frames
            ]
        frames = [frame for _timestamp, frame in samples]
        if len(frames) < 2:
            return
        _mode, sensitivity, _frame_width = self._motion_settings()
        try:
            self._run_motion_pipeline(
                frames,
                sensitivity,
                captured_at,
                [timestamp for timestamp, _frame in samples],
                isolated=True,
                capture_debug=True,
            )
        except Exception as error:
            LOGGER.debug("motion debug capture failed for %s: %s", self.camera.id, error)

    def _observe_motion_event(
        self,
        topic: str,
        message: str,
        event_at: datetime,
        received_at: float,
    ) -> None:
        observation = MotionContext(
            camera_id=self.camera.id,
            captured_at=received_at,
            original_frame=None,
            configuration={
                "observation_kind": "motion_event",
                "event_source": "manual" if topic.startswith("manual") else "onvif",
                "event_topic": topic,
                "event_message": message,
                "event_at": event_at.timestamp(),
            },
            runtime=self.motion_observation_pipeline.runtime,
        )
        try:
            if self.motion_observation_pipeline.handles_observation("motion_event"):
                self.motion_observation_pipeline.process(observation)
        except Exception as error:
            LOGGER.warning(
                "motion event evidence failed for %s: %s",
                self.camera.id,
                error,
            )

    def _motion_settings(self) -> tuple[str, str, int]:
        override = self.camera.motion_qualification
        mode = self.motion_config.mode if override.mode == "inherit" else override.mode
        sensitivity = self.motion_config.sensitivity if override.sensitivity == "inherit" else override.sensitivity
        frame_width = int(override.frame_width or self.motion_config.frame_width)
        return mode, sensitivity, frame_width

    def _stationary_object_tolerance(self) -> str:
        override = self.camera.motion_qualification.stationary_object_tolerance
        if override == "inherit":
            return self.motion_config.stationary_object_tolerance
        return override

    def _trigger_mode(self) -> str:
        return resolved_trigger_mode(self._motion_settings()[0])

    def _fusion_options(self) -> dict[str, Any]:
        return next(
            (
                dict(stage.get("options") or {})
                for stage in self.motion_fusion_pipeline.stage_configuration
                if stage.get("implementation") == "buffered_evidence_fusion"
            ),
            {},
        )

    def _adaptive_analysis_required(self) -> bool:
        if self._trigger_mode() in {"adaptive", "camera_rescue"}:
            return True
        options = self._fusion_options()
        return (
            str(options.get("policy", "audit")).strip().lower() != "bypass"
            and bool(options.get("include_primary", True))
        )

    def _continuous_primary_analysis_required(self) -> bool:
        """Return whether this mode needs frame-driven qualification cycles."""
        return bool(
            self._adaptive_analysis_required()
            and (
                self.motion_pipeline.continuous_analysis
                or self._trigger_mode() in {"adaptive", "camera_rescue"}
            )
        )

    def _continuous_primary_analysis_due(self, captured_at: float) -> bool:
        if self._trigger_mode() == "adaptive":
            return True
        background_fps = min(
            self.motion_config.sample_fps,
            self.motion_config.camera_mode_background_fps,
        )
        interval = 1.0 / max(0.5, background_fps)
        return bool(
            self._motion_primary_last_processed_at <= 0.0
            or captured_at - self._motion_primary_last_processed_at >= interval * 0.85
        )

    def _external_confirmation_required(self) -> bool:
        options = self._fusion_options()
        raw_sources = options.get("sources", [])
        source_values = (raw_sources,) if isinstance(raw_sources, str) else raw_sources
        sources = tuple(
            normalized
            for source in source_values
            if (normalized := str(source).strip().lower())
        ) if isinstance(source_values, (list, tuple)) else ()
        policy = str(options.get("policy", "audit")).strip().lower()
        return bool(
            sources
            and (
                policy in {"all", "weighted"}
                or not bool(options.get("include_primary", True))
            )
        )

    def _frame_motion_analysis_required(self) -> bool:
        return bool(
            self._adaptive_analysis_required()
            or self.motion_observation_pipeline.handles_observation("frame")
            or self.motion_debug.enabled()
        )

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

    def _suppression_verification_rate(self) -> float:
        override = self.camera.motion_qualification.suppression_verification_rate
        return float(
            self.motion_config.suppression_verification_rate
            if override is None
            else override
        )

    @staticmethod
    def _should_verify_suppression(decision_id: str, rate: float) -> bool:
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        sample = int.from_bytes(
            hashlib.sha256(decision_id.encode("utf-8")).digest()[:8],
            "big",
        ) / float(2**64)
        return sample < rate

    @staticmethod
    def _is_borderline_candidate(
        result: MotionQualificationResult,
        enabled: bool,
        margin: float,
    ) -> bool:
        return bool(
            enabled
            and not result.accepted
            and result.reason == "low_score"
            and result.score >= max(0.0, result.threshold - margin)
        )

    def _with_source_evidence(
        self,
        result: MotionQualificationResult,
        start_epoch: float,
        end_epoch: float,
        *,
        include_telemetry: bool = True,
        require_primary_trigger: bool = False,
    ) -> MotionQualificationResult:
        with self._motion_fusion_lock:
            stale_by = self._motion_fusion_last_at - end_epoch
            if 0.0 < stale_by <= FUSION_STALE_TOLERANCE_SECONDS:
                with self._motion_stats_lock:
                    self._motion_stats["stale_fusion_samples"] += 1
                return MotionQualificationResult(
                    accepted=False,
                    score=0.0,
                    threshold=1.0,
                    reason="stale_fusion_evidence",
                    frame_count=result.frame_count,
                    features={
                        **result.features,
                        "stale_fusion_seconds": round(stale_by, 3),
                        "stale_fusion_original_score": result.score,
                    },
                    telemetry=dict(result.telemetry),
                )
            context = MotionContext(
                camera_id=self.camera.id,
                captured_at=end_epoch,
                original_frame=None,
                configuration={
                    "evidence_started_at": start_epoch,
                    "evidence_ended_at": end_epoch,
                    "require_primary_trigger": require_primary_trigger,
                },
                runtime=self.motion_fusion_pipeline.runtime,
                scoring=MotionScoring(
                    accepted=result.accepted,
                    score=result.score,
                    threshold=result.threshold,
                    reason=result.reason,
                    frame_count=result.frame_count,
                    features=dict(result.features),
                ),
            )
            try:
                processed = self.motion_fusion_pipeline.process(context)
            except Exception as error:
                return self._validation_fail_open_result(
                    "motion fusion pipeline",
                    error,
                    result,
                    allow_detection=not (
                        require_primary_trigger and not result.accepted
                    ),
                )
            self._motion_fusion_last_at = end_epoch
        scoring = processed.scoring
        decision = processed.decision
        features = dict(scoring.features)
        features.update(
            {
                "event_state_phase": processed.event_state.phase.value,
                "event_state_transition": processed.event_state.transition_reason,
                "event_state_consecutive_accepts": (
                    processed.event_state.consecutive_accepts
                ),
                "event_state_consecutive_rejects": (
                    processed.event_state.consecutive_rejects
                ),
            }
        )
        if processed.event_state.cooldown_until is not None:
            features["event_state_cooldown_remaining"] = round(
                max(0.0, processed.event_state.cooldown_until - end_epoch),
                3,
            )
        telemetry = dict(result.telemetry)
        if include_telemetry:
            graphs = dict(telemetry.get("graphs") or {})
            graphs.setdefault("qualification", self.motion_pipeline.audit_snapshot())
            graphs["observation"] = self.motion_observation_pipeline.audit_snapshot()
            graphs["fusion"] = self.motion_fusion_pipeline.audit_snapshot(processed.timings)
            telemetry.update({
                "schema_version": 1,
                "origins": dict(self.motion_pipeline_origins),
                "graphs": graphs,
            })
        return MotionQualificationResult(
            accepted=(
                decision.run_object_detection
                if decision is not None
                else scoring.accepted
            ),
            score=decision.score if decision is not None else scoring.score,
            threshold=scoring.threshold,
            reason=decision.reason if decision is not None else scoring.reason,
            frame_count=scoring.frame_count,
            features=features,
            telemetry=telemetry,
        )

    def _reset_motion_fusion_runtime(self) -> None:
        """Reset event-state stages without disturbing unrelated fusion state."""
        event_state_stage_ids = frozenset(
            str(stage.get("stage_id") or "")
            for stage in self.motion_fusion_pipeline.stage_configuration
            if stage.get("implementation") == "score_event_state"
        )
        with self._motion_fusion_lock:
            self.motion_fusion_pipeline.runtime.reset_stages(event_state_stage_ids)

    def _validation_fail_open_result(
        self,
        component: str,
        error: Exception,
        original: MotionQualificationResult | None = None,
        *,
        allow_detection: bool = True,
    ) -> MotionQualificationResult:
        with self._motion_stats_lock:
            self._motion_stats["validation_failures"] += 1
            self._motion_stats["validation_fail_opens"] += int(allow_detection)
        LOGGER.error(
            "%s unavailable for %s; %s (%s)",
            component,
            self.camera.id,
            "allowing object detection" if allow_detection else "preserving rejected primary trigger",
            type(error).__name__,
        )
        features = dict(original.features) if original is not None else {}
        features.update({
            "validation_unavailable": True,
            "validation_fail_open": allow_detection,
            "validation_failure_component": component,
            "validation_failure_type": type(error).__name__,
        })
        return MotionQualificationResult(
            allow_detection,
            1.0 if allow_detection else (original.score if original is not None else 0.0),
            0.0 if allow_detection else (original.threshold if original is not None else 1.0),
            "validation_unavailable_fail_open" if allow_detection else "primary_trigger_rejected",
            original.frame_count if original is not None else 0,
            features,
            telemetry=dict(original.telemetry) if original is not None else {},
        )

    @staticmethod
    def _audit_features(result: MotionQualificationResult) -> dict[str, Any]:
        features = dict(result.features)
        if result.telemetry:
            features["pipeline_telemetry"] = result.telemetry
        return features

    def _with_pipeline_telemetry(
        self,
        result: MotionQualificationResult,
    ) -> MotionQualificationResult:
        if result.telemetry:
            return result
        return MotionQualificationResult(
            accepted=result.accepted,
            score=result.score,
            threshold=result.threshold,
            reason=result.reason,
            frame_count=result.frame_count,
            features=dict(result.features),
            telemetry={
                "schema_version": 1,
                "origins": dict(self.motion_pipeline_origins),
                "graphs": {
                    "qualification": self.motion_pipeline.audit_snapshot(),
                    "observation": self.motion_observation_pipeline.audit_snapshot(),
                    "fusion": self.motion_fusion_pipeline.audit_snapshot(),
                },
            },
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
        if not self._adaptive_analysis_required():
            if self._external_confirmation_required():
                self._stop.wait(self.motion_config.post_trigger_seconds)
            result = MotionQualificationResult(
                False,
                0.0,
                1.0,
                "adaptive_validation_disabled",
                0,
                {},
            )
            return self._with_source_evidence(
                result,
                anchor - self.motion_config.window_seconds,
                time.time(),
            ), {
                "windows_evaluated": 0,
                "event_receipt_delta_seconds": round(received_at - event_epoch, 3),
            }
        deadline = time.monotonic() + self.motion_config.post_trigger_seconds
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
                try:
                    result = self._run_motion_pipeline(
                        [item[1] for item in window],
                        sensitivity,
                        window_end,
                        [item[0] for item in window],
                    )
                except Exception as error:
                    return self._validation_fail_open_result(
                        "adaptive validation pipeline",
                        error,
                    ), {
                        "windows_evaluated": len(evaluated_windows),
                        "event_receipt_delta_seconds": round(received_at - event_epoch, 3),
                    }
                if best_result is None or result.score > best_result.score:
                    best_result = result
                if result.accepted and not self._external_confirmation_required():
                    return self._with_source_evidence(
                        result,
                        anchor - self.motion_config.window_seconds,
                        time.time(),
                    ), {
                        "windows_evaluated": len(evaluated_windows),
                        "event_receipt_delta_seconds": round(received_at - event_epoch, 3),
                    }

            remaining = deadline - time.monotonic()
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
            return self._with_source_evidence(
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
            return self._with_source_evidence(
                result,
                anchor - self.motion_config.window_seconds,
                time.time(),
            ), diagnostics
        return self._with_source_evidence(
            best_result,
            anchor - self.motion_config.window_seconds,
            time.time(),
        ), diagnostics

    def _run_motion_pipeline(
        self,
        frames: list[np.ndarray],
        sensitivity: str,
        captured_at: float,
        frame_timestamps: list[float] | None = None,
        *,
        isolated: bool = True,
        capture_debug: bool = True,
        include_telemetry: bool = True,
    ) -> MotionQualificationResult:
        mode, _resolved_sensitivity, frame_width = self._motion_settings()
        if isolated:
            with self._motion_analysis_lock:
                pipeline = self.motion_pipeline.isolated_copy(clone_runtime=True)
        else:
            pipeline = self.motion_pipeline
        context = MotionContext(
            camera_id=self.camera.id,
            captured_at=captured_at,
            original_frame=frames[-1] if frames else None,
            frame_history=tuple(frames),
            frame_timestamps=tuple(frame_timestamps or ()),
            configuration={
                **self.motion_config.model_dump(mode="python"),
                "camera_id": self.camera.id,
                "mode": mode,
                "sensitivity": sensitivity,
                "stationary_object_tolerance": self._stationary_object_tolerance(),
                "frame_width": frame_width,
                "motion_zones": [
                    zone.model_dump(mode="python")
                    for zone in self.camera.zones
                ],
            },
            runtime=pipeline.runtime,
        )
        try:
            processed = pipeline.process(context)
        finally:
            if isolated:
                pipeline.close()
        if capture_debug:
            self.motion_debug.capture(processed)
        result = processed.scoring
        features = dict(result.features)
        features.setdefault(
            "primary_motion_source",
            self.motion_pipeline.primary_motion_source,
        )
        dominant = processed.dominant_track
        if dominant is not None and dominant.observations:
            features.setdefault(
                "motion_regions",
                [
                    [round(float(value), 5) for value in blob.box]
                    for blob in dominant.observations[-12:]
                ],
            )
            features.setdefault("motion_region_track_id", dominant.track_id)
        telemetry: dict[str, Any] = {}
        if include_telemetry:
            telemetry = {
                "schema_version": 1,
                "origins": dict(self.motion_pipeline_origins),
                "graphs": {
                    "qualification": pipeline.audit_snapshot(processed.timings),
                },
            }
        return MotionQualificationResult(
            accepted=result.accepted,
            score=result.score,
            threshold=result.threshold,
            reason=result.reason,
            frame_count=result.frame_count,
            features=features,
            telemetry=telemetry,
        )

    def set_motion_debug_enabled(self, enabled: bool) -> None:
        self.motion_debug.set_enabled(enabled)

    def motion_debug_status(self) -> dict[str, Any]:
        return self.motion_debug.status()

    def motion_debug_image(self, layer: str) -> bytes | None:
        return self.motion_debug.image(layer)

    def _run_motion_events(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_motion_events_until_error()
                return
            except Exception:
                failed_triggers = self._active_motion_triggers
                self._active_motion_triggers = None
                with self._motion_stats_lock:
                    self._motion_stats["event_worker_errors"] += 1
                    if self._adaptive_trigger_pending:
                        self._adaptive_trigger_pending = False
                        self._adaptive_last_completed_at = time.time()
                LOGGER.exception("motion event cycle failed for %s", self.camera.id)
                if failed_triggers and not self._stop.is_set():
                    self._retry_motion_trigger_batch(failed_triggers)

    def _retry_motion_trigger_batch(self, triggers: list[dict[str, Any]]) -> None:
        retry_count = max(
            (int(item.get("_event_retry_count") or 0) for item in triggers),
            default=0,
        ) + 1
        if retry_count > MOTION_EVENT_MAX_RETRIES:
            with self._motion_stats_lock:
                self._motion_stats["event_retry_drops"] += 1
            return
        retry_triggers = [
            {**item, "_event_retry_count": retry_count}
            for item in triggers
        ]
        if self._stop.wait(0.25 * retry_count):
            return
        wrapper = {
            "topic": "internal/retry_batch",
            "message": f"motion event retry {retry_count}",
            "event_at": min(item["event_at"] for item in retry_triggers),
            "received_at": min(
                float(item.get("received_at") or time.time())
                for item in retry_triggers
            ),
            "_retry_batch": retry_triggers,
        }
        self._motion_retry_batches.append(wrapper)
        with self._motion_stats_lock:
            self._motion_stats["event_retries"] += 1

    def _run_motion_events_until_error(self) -> None:
        while not self._stop.is_set():
            if self._motion_retry_batches:
                first = self._motion_retry_batches.popleft()
            else:
                try:
                    first = self._motion_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
            if first is None or self._stop.is_set():
                return

            retry_batch = first.get("_retry_batch")
            if isinstance(retry_batch, list):
                triggers = [dict(item) for item in retry_batch]
            else:
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
            self._active_motion_triggers = triggers

            priority_triggers = [
                item
                for item in triggers
                if self._priority_motion_topic(str(item["topic"]))
            ]
            representative = min(
                priority_triggers or triggers,
                key=lambda item: item["event_at"],
            )
            event_at = representative["event_at"]
            received_at = min(float(item.get("received_at") or time.time()) for item in triggers)
            decision_id = next(
                (
                    str(item["_motion_decision_id"])
                    for item in triggers
                    if item.get("_motion_decision_id")
                ),
                "",
            )
            if not decision_id:
                decision_id = uuid.uuid4().hex
            for item in triggers:
                item["_motion_decision_id"] = decision_id

            mode, sensitivity, frame_width = self._motion_settings()
            rescue_enabled, rescue_margin = self._motion_rescue_settings()
            priority = bool(priority_triggers)
            adaptive_only = all(
                str(item.get("topic") or "").startswith("adaptive/")
                for item in triggers
            )
            visual_backup_queued = any(
                str(item.get("topic") or "") == "adaptive/visual_backup"
                for item in triggers
            )
            visual_backup = adaptive_only and visual_backup_queued
            if visual_backup_queued and not visual_backup:
                with self._motion_stats_lock:
                    matched_camera_at = max(self._camera_motion_times, default=0.0)
                    if matched_camera_at > self._visual_backup_last_matched_camera_at:
                        self._motion_stats["visual_backup_onvif_matches"] += 1
                        self._visual_backup_last_matched_camera_at = matched_camera_at
            prequalified = [
                item["prequalified"]
                for item in triggers
                if isinstance(item.get("prequalified"), MotionQualificationResult)
            ]
            retry_results = [
                item["_retry_qualification_result"]
                for item in triggers
                if isinstance(
                    item.get("_retry_qualification_result"),
                    MotionQualificationResult,
                )
            ]
            diagnostics: dict[str, Any] = {
                "windows_evaluated": 0,
                "event_receipt_delta_seconds": round(received_at - event_at.timestamp(), 3),
            }
            retry_diagnostics = next(
                (
                    dict(item["_retry_diagnostics"])
                    for item in triggers
                    if isinstance(item.get("_retry_diagnostics"), dict)
                ),
                None,
            )
            if adaptive_only and self._matches_recent_priority_motion(event_at.timestamp()):
                result = MotionQualificationResult(
                    False,
                    0.0,
                    1.0,
                    "priority_event_deduplicated",
                    0,
                    {"primary_motion_source": "adaptive_background"},
                )
            elif mode == "off":
                result = MotionQualificationResult(True, 1.0, 0.0, "disabled", 0, {})
            elif priority:
                result = MotionQualificationResult(
                    True,
                    1.0,
                    0.0,
                    "priority_topic",
                    0,
                    {"primary_motion_source": "onvif_priority"},
                )
            elif retry_results:
                result = max(retry_results, key=lambda item: item.score)
                if retry_diagnostics is not None:
                    diagnostics = retry_diagnostics
            elif prequalified:
                result = self._with_pipeline_telemetry(
                    max(prequalified, key=lambda item: item.score)
                )
            else:
                result, diagnostics = self._qualify_motion_burst(event_at, received_at, sensitivity)

            if self._stop.is_set():
                self._complete_adaptive_trigger(triggers)
                self._active_motion_triggers = None
                return

            borderline_candidate = self._is_borderline_candidate(
                result,
                rescue_enabled,
                rescue_margin,
            )
            suppression_verification_candidate = bool(
                mode in {"camera", "camera_rescue", "adaptive", "enforce"}
                and not result.accepted
                and not borderline_candidate
                and not result.reason.startswith("event_state_")
                and self._should_verify_suppression(
                    decision_id,
                    self._suppression_verification_rate(),
                )
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
                "suppression_verification_rate": self._suppression_verification_rate(),
                "suppression_verification_candidate": suppression_verification_candidate,
                "trigger_count": len(triggers),
                "trigger_source": "visual_backup" if visual_backup else (
                    "adaptive" if adaptive_only else "camera"
                ),
                "retry_count": max(
                    (int(item.get("_event_retry_count") or 0) for item in triggers),
                    default=0,
                ),
                "would_suppress": bool(
                    mode in {"audit", "camera", "camera_rescue", "adaptive", "enforce"}
                    and not result.accepted
                ),
            }
            if mode == "off":
                effective_accepted = True
            elif mode == "audit":
                effective_accepted = True
            else:
                effective_accepted = bool(
                    result.accepted
                    or borderline_candidate
                    or suppression_verification_candidate
                )
            qualification["effective_accepted"] = effective_accepted
            retry_attempt = qualification["retry_count"] > 0
            for item in triggers:
                item["_retry_qualification_result"] = result
                item["_retry_diagnostics"] = dict(diagnostics)
            with self._motion_stats_lock:
                if not retry_attempt:
                    self._motion_stats["bursts"] += 1
                    if priority:
                        self._motion_stats["priority_bypasses"] += 1
                    if result.reason == "insufficient_frames":
                        self._motion_stats["insufficient_frames"] += 1
                    if result.reason == "no_temporal_signal":
                        self._motion_stats["inconclusive"] += 1
                    if result.accepted:
                        self._motion_stats["passed"] += 1
                    elif (
                        mode in {"camera", "camera_rescue", "adaptive", "enforce"}
                        and not borderline_candidate
                        and not suppression_verification_candidate
                    ):
                        self._motion_stats["suppressed"] += 1
                    elif mode == "audit":
                        self._motion_stats["audit_rejected"] += 1
                self._motion_stats["last_result"] = qualification

            if not retry_attempt:
                self._publish_event_safely("motion_qualification", {
                    "camera_id": self.camera.id,
                    "timestamp": event_at.isoformat(),
                    **qualification,
                })
            if not effective_accepted:
                try:
                    snapshot_path = next(
                        (
                            str(item["_motion_audit_snapshot_path"])
                            for item in triggers
                            if item.get("_motion_audit_snapshot_path")
                        ),
                        None,
                    )
                    if snapshot_path is None:
                        snapshot_path = (
                            ""
                            if result.reason in INCIDENT_ACTIVITY_REASONS
                            else self._sample_rejected_motion(event_at, result)
                        )
                        for item in triggers:
                            item["_motion_audit_snapshot_path"] = snapshot_path
                    self.motion_decision_handler.record_audit(
                        decision_id=decision_id,
                        snapshot_path=snapshot_path,
                        event_at=event_at,
                        mode=mode,
                        sensitivity=sensitivity,
                        score=result.score,
                        threshold=result.threshold,
                        reason=result.reason,
                        object_detected=None,
                        trigger_count=len(triggers),
                        features=self._audit_features(result),
                        related_event_id=self._related_incident_event_id(result),
                    )
                finally:
                    self._complete_adaptive_trigger(triggers)
                self._active_motion_triggers = None
                continue

            try:
                outcome = self._process_motion_event(
                    str(representative["topic"]),
                    str(representative["message"]),
                    event_at,
                    qualification,
                    require_eligible_object=bool(
                        visual_backup
                        or borderline_candidate
                        or suppression_verification_candidate
                    ),
                    require_motion_correlation=visual_backup,
                )
                event_id = outcome.get("event_id")
                if event_id is not None:
                    self._active_incident_event_id = int(event_id)
                    # The incident is durable once the handler returns. Later audit or
                    # notification failures must not replay detection and duplicate it.
                    self._active_motion_triggers = None
                object_outcome = outcome.get("object_detected")
                found_object = object_outcome is True
                if borderline_candidate and found_object:
                    with self._motion_stats_lock:
                        self._motion_stats["borderline_rescues"] = self._motion_stats.get("borderline_rescues", 0) + 1
                elif mode in {"camera", "camera_rescue", "adaptive", "enforce"} and borderline_candidate:
                    with self._motion_stats_lock:
                        self._motion_stats["suppressed"] += 1
                if suppression_verification_candidate:
                    with self._motion_stats_lock:
                        self._motion_stats["suppression_verification_checks"] += 1
                        if found_object:
                            self._motion_stats["suppression_verification_rescues"] += 1
                        else:
                            self._motion_stats["suppressed"] += 1
                if mode == "audit" and not result.accepted and found_object:
                    with self._motion_stats_lock:
                        self._motion_stats["audit_object_matches"] += 1
                if visual_backup:
                    correlation = outcome.get("motion_correlation")
                    if (
                        outcome.get("rejection_reason") == "object_not_motion_correlated"
                        and isinstance(correlation, dict)
                    ):
                        with self._motion_stats_lock:
                            self._motion_stats["visual_backup_uncorrelated_objects"] += int(
                                correlation.get("eligible_object_count") or 0
                            )
                    if not found_object:
                        # A backup with no eligible object must not leave the
                        # fusion event state ACTIVE. A real camera notice that
                        # arrives just after the backup still deserves its own
                        # ordinary detector pass.
                        self._reset_motion_fusion_runtime()
                    self.motion_decision_handler.record_audit(
                        event_id=int(event_id) if event_id is not None else None,
                        decision_id=decision_id if event_id is None else "",
                        snapshot_path=str(outcome.get("snapshot_path") or ""),
                        event_at=event_at,
                        mode=mode,
                        sensitivity=sensitivity,
                        score=result.score,
                        threshold=result.threshold,
                        reason=str(outcome.get("rejection_reason") or "visual_backup_trigger"),
                        object_detected=object_outcome,
                        trigger_count=len(triggers),
                        features={
                            **self._audit_features(result),
                            "visual_backup_original_reason": result.reason,
                            "motion_correlation": correlation,
                        },
                        category="visual_backup",
                    )
                elif mode in {"audit", "camera", "camera_rescue", "adaptive", "enforce"} and not result.accepted:
                    audit_snapshot_path = str(outcome.get("snapshot_path") or "")
                    if not audit_snapshot_path and event_id is None:
                        audit_snapshot_path = self._sample_rejected_motion(event_at, result)
                    self.motion_decision_handler.record_audit(
                        event_id=int(event_id) if event_id is not None else None,
                        decision_id=decision_id if event_id is None else "",
                        snapshot_path=audit_snapshot_path,
                        event_at=event_at,
                        mode=mode,
                        sensitivity=sensitivity,
                        score=result.score,
                        threshold=result.threshold,
                        reason=result.reason,
                        object_detected=object_outcome,
                        trigger_count=len(triggers),
                        features={
                            **self._audit_features(result),
                            "suppression_verification": suppression_verification_candidate,
                        },
                    )
            finally:
                self._complete_adaptive_trigger(triggers)
            self._active_motion_triggers = None

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
            path = self.image_writer.write(
                directory,
                f"{stamp}-{result.score:.3f}-{result.reason}",
                frame,
            )
            if path is not None:
                try:
                    samples = []
                    for item in self.image_writer.stored_images(directory):
                        try:
                            samples.append((item.stat().st_mtime_ns, item))
                        except FileNotFoundError:
                            continue
                    for _modified, stale in sorted(samples)[:-100]:
                        try:
                            stale.unlink(missing_ok=True)
                        except OSError as error:
                            LOGGER.debug(
                                "failed to prune rejected motion sample %s: %s",
                                stale,
                                error,
                            )
                except OSError as error:
                    LOGGER.debug(
                        "failed to enumerate rejected motion samples for %s: %s",
                        self.camera.id,
                        error,
                    )
                return str(path)
            LOGGER.warning("failed to encode rejected motion sample for %s", self.camera.id)
        except OSError as error:
            LOGGER.debug("failed to save rejected motion sample for %s: %s", self.camera.id, error)
        return ""

    def _process_motion_event(
        self,
        topic: str,
        message: str,
        event_at: datetime,
        qualification: dict[str, Any],
        *,
        require_eligible_object: bool = False,
        require_motion_correlation: bool = False,
    ) -> dict[str, Any]:
        if self.object_tracking.config.enabled:
            # Main-stream capture opens concurrently with recorded validation,
            # so the tracking handoff does not pay another RTSP startup delay.
            self._get_latest_tracking_frame("main")
        outcome = self.motion_decision_handler.handle(
            topic,
            message,
            event_at,
            qualification,
            require_eligible_object=require_eligible_object,
            require_motion_correlation=require_motion_correlation,
        )
        if outcome.event_id is not None and outcome.object_detected:
            initial_tracking_frame = None
            if self.object_tracking.config.enabled and outcome.snapshot_path:
                initial_tracking_frame = cv2.imread(outcome.snapshot_path)
            self.object_tracking.start(
                outcome.event_id,
                event_at,
                list(outcome.detected_objects),
                initial_tracking_frame,
            )
        return outcome.as_dict()

    def _related_incident_event_id(self, result: MotionQualificationResult) -> int | None:
        if result.reason not in INCIDENT_ACTIVITY_REASONS:
            return None
        return self._active_incident_event_id

    def _start_source(self, source: str) -> bool:
        source = self.camera.normalized_source(source)
        if self._stop.is_set():
            return False
        with self._frame_lock:
            if self._stop.is_set():
                return False
            thread = self._source_threads.get(source)
            existing_stop = self._source_stops.get(source)
            if (
                thread is not None
                and thread.is_alive()
                and existing_stop is not None
                and not existing_stop.is_set()
            ):
                return True
            stop_event = threading.Event()
            if source == "main":
                self._tracking_frames.clear()
                self._tracking_last_sample_epoch = 0.0
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
            try:
                thread.start()
            except BaseException:
                if self._source_threads.get(source) is thread:
                    self._source_threads.pop(source, None)
                    self._source_stops.pop(source, None)
                    if source == "live":
                        self._thread = None
                raise
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
                self._source_frame_epoch.pop(source, None)
                self._source_frame_monotonic.pop(source, None)
                self._source_errors.pop(source, None)
                if source == "main":
                    self._tracking_frames.clear()
                    self._tracking_last_sample_epoch = 0.0
            else:
                self._thread = None

    def _run_source(self, source: str, stop_event: threading.Event) -> None:
        retry_delay = CAPTURE_RETRY_INITIAL_SECONDS
        try:
            while not self._stop.is_set() and not stop_event.is_set():
                if self._source_is_idle(source):
                    return
                capture = cv2.VideoCapture()
                failure_reason = ""
                try:
                    opened = self._open_capture(capture, source, stop_event)
                    if self._stop.is_set() or stop_event.is_set():
                        return
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not opened or not capture.isOpened():
                        failure_reason = "failed to open stream"
                        with self._frame_lock:
                            self._source_capture_stats[source]["open_failures"] += 1
                        self._set_source_error(source, failure_reason)
                    else:
                        with self._frame_lock:
                            source_stats = self._source_capture_stats[source]
                            if source_stats["frames_received"] > 0:
                                source_stats["reconnects"] += 1
                        self._set_source_error(source, "")
                        while not self._stop.is_set() and not stop_event.is_set():
                            if self._source_is_idle(source):
                                return
                            ok, frame = capture.read()
                            if not ok:
                                if self._stop.is_set() or stop_event.is_set():
                                    break
                                failure_reason = "stream read failed"
                                with self._frame_lock:
                                    self._source_capture_stats[source]["read_failures"] += 1
                                self._set_source_error(source, failure_reason)
                                break
                            if self._stop.is_set() or stop_event.is_set():
                                break
                            retry_delay = CAPTURE_RETRY_INITIAL_SECONDS
                            frame_epoch = time.time()
                            stamp = datetime.fromtimestamp(frame_epoch, timezone.utc).isoformat()
                            frame_clock = time.monotonic()
                            with self._frame_lock:
                                self._source_frames[source] = frame.copy()
                                self._source_frame_at[source] = stamp
                                self._source_frame_epoch[source] = frame_epoch
                                self._source_frame_monotonic[source] = frame_clock
                                self._source_frame_dimensions[source] = {
                                    "width": int(frame.shape[1]),
                                    "height": int(frame.shape[0]),
                                }
                                self._source_frame_times[source].append(frame_clock)
                                self._source_capture_stats[source]["frames_received"] += 1
                                if source == "live":
                                    self.last_frame_at = stamp
                            if source == "live":
                                self._remember_motion_frame(frame, frame_clock)
                            elif source == "main":
                                self._remember_tracking_frame(frame, frame_epoch)
                except Exception as exc:
                    failure_reason = f"stream error: {redact_secret_text(exc)[:160]}"
                    self._set_source_error(source, failure_reason)
                    LOGGER.warning(
                        "camera stream failed for %s/%s: %s",
                        self.camera.id,
                        source,
                        failure_reason,
                    )
                finally:
                    capture.release()
                if self._stop.is_set() or stop_event.is_set() or self._source_is_idle(source):
                    break
                LOGGER.info(
                    "camera=%s source=%s retry_delay=%s failure_reason=%s",
                    self.camera.id,
                    source,
                    retry_delay,
                    failure_reason,
                )
                wait_delay = retry_delay
                if source != "live":
                    with self._frame_lock:
                        last_access = self._source_last_access.get(source)
                    if last_access is not None:
                        idle_in = max(
                            0.0,
                            MAIN_SOURCE_IDLE_SECONDS - (time.monotonic() - last_access),
                        )
                        wait_delay = min(wait_delay, idle_in)
                if stop_event.wait(wait_delay):
                    break
                retry_delay = min(retry_delay * 2.0, CAPTURE_RETRY_MAX_SECONDS)
        finally:
            self._source_finished(source)

    def _open_capture(
        self,
        capture: Any,
        source: str,
        stop_event: threading.Event,
    ) -> bool:
        # OpenCV/FFmpeg opens can wait in native code where Python cannot
        # interrupt them. Bound admission at the Python boundary so queued
        # cameras can observe shutdown and the admitted worst case still fits
        # inside the capture-thread join budget.
        while not self._stop.is_set() and not stop_event.is_set():
            if not CAPTURE_OPEN_SLOTS.acquire(timeout=CAPTURE_OPEN_LOCK_POLL_SECONDS):
                continue
            try:
                if self._stop.is_set() or stop_event.is_set():
                    return False
                return bool(capture.open(
                    self.camera.source_url(source),
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_N_THREADS,
                        CAPTURE_DECODER_THREADS,
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        CAPTURE_OPEN_TIMEOUT_MS,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        CAPTURE_READ_TIMEOUT_MS,
                    ],
                ))
            finally:
                CAPTURE_OPEN_SLOTS.release()
        return False

    def _set_source_error(self, source: str, message: str) -> None:
        message = redact_secret_text(message)
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
            frame_clock = self._source_frame_monotonic.get(source)
            if (
                frame is None
                or frame_clock is None
                or time.monotonic() - frame_clock > FRAME_STALE_SECONDS
            ):
                return None
            return frame.copy()

    def _get_latest_tracking_frame(
        self,
        source: str = "main",
    ) -> tuple[np.ndarray, float, float] | None:
        source = self.camera.normalized_source(source)
        if self._stop.is_set():
            return None
        with self._frame_lock:
            self._source_last_access[source] = time.monotonic()
        if not self._start_source(source):
            return None
        with self._frame_lock:
            frame = self._source_frames.get(source)
            captured_at = self._source_frame_epoch.get(source)
            frame_clock = self._source_frame_monotonic.get(source)
            if (
                frame is None
                or captured_at is None
                or frame_clock is None
                or time.monotonic() - frame_clock > FRAME_STALE_SECONDS
            ):
                return None
            return frame.copy(), captured_at, frame_clock

    def _get_latest_tracking_frame_with_fallback(
        self,
    ) -> tuple[np.ndarray, float, float] | None:
        """Prefer main-stream detail without leaving a cold-start frame gap.

        Main capture is started on the first request, but a high-resolution HEVC
        stream may not yield a frame until its next keyframe. The live stream is
        already continuously captured, so it provides a timestamped bridge until
        main capture is ready. ObjectTrackingSession rescales detections from each
        source into the incident coordinate space before association.
        """
        main_sample = self._get_latest_tracking_frame("main")
        if main_sample is not None:
            return main_sample
        return self._get_latest_tracking_frame("live")

    def _remember_tracking_frame(self, frame: np.ndarray, captured_at: float) -> None:
        """Retain a bounded, detector-sized main-stream history for live catch-up."""
        interval = 1.0 / max(0.1, float(self.object_tracking.config.sample_fps))
        with self._frame_lock:
            if captured_at - self._tracking_last_sample_epoch < interval * 0.9:
                return
            self._tracking_last_sample_epoch = captured_at
        height, width = frame.shape[:2]
        if width > TRACKING_CATCHUP_FRAME_WIDTH:
            scale = TRACKING_CATCHUP_FRAME_WIDTH / width
            frame = cv2.resize(
                frame,
                (TRACKING_CATCHUP_FRAME_WIDTH, max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            frame = frame.copy()
        with self._frame_lock:
            self._tracking_frames.append((captured_at, frame))

    def _recorded_tracking_frames(
        self,
        start_epoch: float,
        end_epoch: float,
        sample_fps: float,
        frame_width: int,
    ) -> Iterator[tuple[float, np.ndarray]]:
        """Yield recorded and buffered frames that bridge detection to live tracking."""
        if end_epoch <= start_epoch or frame_width <= 0:
            return
        samples: list[tuple[float, np.ndarray]] = []
        rows = self.motion_object_detector.recorder.recording_rows_between(
            self.camera.id,
            start_epoch,
            end_epoch,
            source="main",
        )
        interval = 1.0 / max(0.1, float(sample_fps))
        last_epoch = start_epoch - interval
        for row in rows:
            row_start = float(row.get("start_epoch") or 0.0)
            row_end = float(row.get("end_epoch") or row_start)
            sample_start = max(start_epoch, row_start)
            sample_end = min(end_epoch, row_end)
            duration = sample_end - sample_start
            if duration <= 0.0:
                continue
            path = Path(str(row.get("path") or ""))
            if not path.is_file():
                continue
            try:
                for captured_at, frame in sampled_video_frames(
                    path,
                    start_epoch=sample_start,
                    sample_fps=sample_fps,
                    duration_seconds=duration,
                    ffmpeg_path=self.motion_object_detector.recorder.ffmpeg_path,
                    maximum_width=frame_width,
                    start_offset_seconds=max(0.0, sample_start - row_start),
                    probe_path=path,
                ):
                    if captured_at <= last_epoch + interval * 0.5:
                        continue
                    if captured_at > end_epoch + 1e-6:
                        break
                    last_epoch = captured_at
                    samples.append((captured_at, frame))
            except RuntimeError as error:
                LOGGER.warning(
                    "recorded tracking catch-up skipped %s/%s: %s",
                    self.camera.id,
                    path.name,
                    redact_secret_text(error),
                )
        with self._frame_lock:
            samples.extend(
                (captured_at, frame)
                for captured_at, frame in self._tracking_frames
                if start_epoch <= captured_at <= end_epoch
            )
        last_epoch = start_epoch - interval
        for captured_at, frame in sorted(samples, key=lambda sample: sample[0]):
            if captured_at <= last_epoch + interval * 0.5:
                continue
            last_epoch = captured_at
            yield captured_at, frame

    def _recorded_motion_frame(
        self,
        event_at: datetime,
    ) -> tuple[Any | None, list[dict[str, Any]], str]:
        return self.motion_object_detector.detect(event_at)

    def _write_snapshot(self, frame: Any, event_at: datetime | None = None) -> str:
        captured_at = event_at or datetime.now(timezone.utc)
        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=timezone.utc)
        else:
            captured_at = captured_at.astimezone(timezone.utc)
        event_stamp = captured_at.strftime("%Y%m%d-%H%M%S-%f")
        stamp = f"{event_stamp}-{time.time_ns() % 1_000_000_000:09d}"
        path = self.image_writer.write(self.snapshots_dir, stamp, frame)
        if path is None:
            LOGGER.warning("failed to encode snapshot for %s", self.camera.id)
            return ""
        return str(path)

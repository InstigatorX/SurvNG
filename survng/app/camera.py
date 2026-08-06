from __future__ import annotations

import logging
import queue
import random
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from .camera_capture import (
    FRAME_STALE_SECONDS,
    CaptureBackend,
    CameraCaptureService,
    CapturedFrame,
    CaptureOpenLimiter,
    OpenCvFfmpegCaptureBackend,
)
from .config import CameraConfig, DetectionZone, MotionQualificationConfig
from .image_storage import DurableImageWriter
from .onvif_events import OnvifEventListener
from .motion import MotionQualificationResult
from .motion_analysis import FairMotionAnalysisLimiter
from .motion_analysis_service import MotionAnalysisHooks, MotionAnalysisService
from .motion_coordinator import (
    VisualBackupCoordinator,
    VisualBackupPolicy,
)
from .motion_events import (
    MotionEventCoordinator,
    MotionTrigger,
    MotionTriggerBatch,
    RetryDisposition,
)
from .motion_decisions import (
    MotionDecisionHooks,
    MotionDecisionOrchestrator,
    audit_features,
    is_borderline_candidate,
    priority_motion_topic,
    should_verify_suppression,
)
from .motion_incidents import MotionIncidentService
from .object_tracking import ObjectTrackingSession, ObjectTrackingSessionFactory
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
CAPTURE_STOP_TIMEOUT_SECONDS = 8.0
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
        motion_analysis_limiter: FairMotionAnalysisLimiter,
        image_writer: DurableImageWriter,
        onvif_cache_dir: Path | None = None,
        capture_backend: CaptureBackend | None = None,
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
        effective_capture_backend = capture_backend or OpenCvFfmpegCaptureBackend(
            CaptureOpenLimiter()
        )
        self.motion_debug = MotionDebugSnapshotStore()
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
        self.motion_incidents = MotionIncidentService(
            camera_id=camera.id,
            decision_processor=self.motion_decision_handler,
            tracking_provider=lambda: self.object_tracking,
            prewarm_tracking=lambda: self._get_latest_tracking_frame("main"),
            image_reader=lambda path: cv2.imread(path),
        )
        self.snapshots_dir = storage_dir / "snapshots" / camera.id
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._stop.set()
        self._enabled = False
        self._accepting_motion_events = True
        self._lifecycle_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._motion_analysis_lock = threading.Lock()
        self._motion_fusion_lock = threading.Lock()
        self._motion_fusion_last_at = 0.0
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
        self.motion_events = MotionEventCoordinator(
            queue_size=MOTION_QUEUE_SIZE,
            retry_limit=MOTION_EVENT_MAX_RETRIES,
        )
        self._active_incident_event_id: int | None = None
        self.visual_backup = VisualBackupCoordinator()
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
            "analysis_wait_ms_total": 0.0,
            "analysis_wait_ms_max": 0.0,
            "continuous_frames": 0,
            "continuous_candidates": 0,
            "adaptive_triggers_deferred": 0,
            "visual_backup_candidates": 0,
            "visual_backup_triggers": 0,
            "visual_backup_onvif_matches": 0,
            "visual_backup_rate_limited": 0,
            "visual_backup_not_ready": 0,
            "visual_backup_not_promoted": 0,
            "visual_backup_uncorrelated_objects": 0,
            "illumination_evaluations": 0,
            "illumination_candidates": 0,
            "illumination_filtered": 0,
            "illumination_verification_probes": 0,
            "illumination_verification_rescues": 0,
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
        self.motion_decisions = MotionDecisionOrchestrator(
            camera_id=camera.id,
            events=self.motion_events,
            audit_recorder=self.motion_decision_handler,
            burst_quiet_seconds=lambda: self.motion_config.burst_quiet_seconds,
            hooks=MotionDecisionHooks(
                motion_settings=lambda: self._motion_settings(),
                rescue_settings=lambda: self._motion_rescue_settings(),
                suppression_verification_rate=(
                    lambda: self._suppression_verification_rate()
                ),
                matches_recent_priority_motion=(
                    lambda observed_at: self._matches_recent_priority_motion(observed_at)
                ),
                qualify_motion_burst=(
                    lambda event_at, received_at, sensitivity: self._qualify_motion_burst(
                        event_at, received_at, sensitivity
                    )
                ),
                with_pipeline_telemetry=(
                    lambda result: self._with_pipeline_telemetry(result)
                ),
                process_incident=lambda *args, **kwargs: self._process_motion_event(
                    *args, **kwargs
                ),
                sample_rejected_motion=(
                    lambda event_at, result: self._sample_rejected_motion(
                        event_at, result
                    )
                ),
                related_incident_event_id=(
                    lambda result: self._related_incident_event_id(result)
                ),
                reset_motion_fusion_runtime=(
                    lambda: self._reset_motion_fusion_runtime()
                ),
                record_visual_camera_match=(
                    lambda observed_at: self._record_visual_camera_match(observed_at)
                ),
                complete_adaptive_trigger=(
                    lambda triggers: self._complete_adaptive_trigger(triggers)
                ),
                set_active_incident_event_id=(
                    lambda event_id: self._set_active_incident_event_id(event_id)
                ),
                publish_event=lambda event_type, payload: self._publish_event_safely(
                    event_type, payload
                ),
                record_decision_stats=lambda **kwargs: self._record_decision_stats(
                    **kwargs
                ),
                increment_stat=lambda name, amount: self._increment_motion_stat(
                    name, amount
                ),
            ),
        )
        self.motion_analysis = MotionAnalysisService(
            camera_id=camera.id,
            frame_lock=self._frame_lock,
            analysis_lock=self._motion_analysis_lock,
            ring_size=ring_size,
            queue_size=MOTION_ANALYSIS_QUEUE_SIZE,
            limiter=self.motion_analysis_limiter,
            observation_pipeline=self.motion_observation_pipeline,
            events=self.motion_events,
            visual_backup=self.visual_backup,
            audit_recorder=self.motion_decision_handler,
            debug_store=self.motion_debug,
            hooks=MotionAnalysisHooks(
                frame_analysis_required=lambda: self._frame_motion_analysis_required(),
                sample_fps=lambda: self.motion_config.sample_fps,
                frame_width=lambda: self._motion_settings()[2],
                motion_settings=lambda: self._motion_settings(),
                continuous_primary_required=(
                    lambda: self._continuous_primary_analysis_required()
                ),
                continuous_primary_due=(
                    lambda captured_at, last_processed_at: (
                        self._continuous_primary_analysis_due(
                            captured_at,
                            last_processed_at=last_processed_at,
                        )
                    )
                ),
                execute_continuous=(
                    lambda captured_at: self._analyze_continuous_motion(captured_at)
                ),
                execute_debug_capture=(
                    lambda captured_at: self._capture_motion_debug(captured_at)
                ),
                adaptive_rearm_seconds=lambda: self._adaptive_rearm_seconds(),
                priority_dedup_seconds=lambda: self._priority_dedup_seconds(),
                run_pipeline=lambda *args, **kwargs: self._run_motion_pipeline(
                    *args, **kwargs
                ),
                illumination_filter_enabled=(
                    lambda: self._illumination_filter_enabled()
                ),
                trigger_mode=lambda: self._trigger_mode(),
                detection_enabled=lambda: self._detection_enabled,
                with_source_evidence=lambda *args, **kwargs: self._with_source_evidence(
                    *args, **kwargs
                ),
                visual_backup_settings=lambda: self._visual_backup_settings(),
                visual_backup_policy=lambda: self._visual_backup_policy(),
                suppression_verification_rate=(
                    lambda: self._suppression_verification_rate()
                ),
                visual_backup_warmup_seconds=(
                    lambda: self.motion_config.visual_backup_warmup_seconds
                ),
                sample_rejected_motion=(
                    lambda event_at, result: self._sample_rejected_motion(
                        event_at, result
                    )
                ),
                publish_event=lambda event_type, payload: self._publish_event_safely(
                    event_type, payload
                ),
                set_last_motion_at=lambda value: self._set_last_motion_at(value),
                increment_stat=lambda name, amount: self._increment_motion_stat(
                    name, amount
                ),
                record_analysis_wait=(
                    lambda wait_ms: self._record_analysis_wait(wait_ms)
                ),
            ),
        )
        self.last_motion_at = ""
        self._detection_enabled = True
        self.capture = CameraCaptureService(
            camera_id=camera.id,
            source_url=camera.source_url,
            backend=effective_capture_backend,
            frame_observer=self._capture_frame,
            source_started_observer=self._capture_source_started,
            source_stopped_observer=self._capture_source_stopped,
        )
        self.onvif = OnvifEventListener(
            camera,
            self.handle_motion_event,
            cache_dir=onvif_cache_dir or storage_dir / "onvif",
        )

    # Transitional compatibility accessors keep diagnostics and existing
    # integrations stable while MotionEventCoordinator owns the runtime.
    @property
    def _motion_queue(self) -> queue.Queue:
        return self.motion_events.queue

    @property
    def _motion_retry_batches(self) -> deque[dict[str, Any]]:
        return self.motion_events.retry_batches

    @property
    def _motion_thread(self) -> threading.Thread | None:
        return self.motion_events.thread

    @_motion_thread.setter
    def _motion_thread(self, value: threading.Thread | None) -> None:
        self.motion_events.thread = value

    @property
    def _active_motion_triggers(self) -> MotionTriggerBatch | None:
        return self.motion_events.active_triggers

    @_active_motion_triggers.setter
    def _active_motion_triggers(
        self,
        value: MotionTriggerBatch | list[MotionTrigger | dict[str, Any]] | None,
    ) -> None:
        self.motion_events.set_active(
            None if value is None else MotionTriggerBatch.coerce(value)
        )

    @property
    def _adaptive_trigger_pending(self) -> bool:
        return self.motion_events.adaptive_trigger_pending

    @_adaptive_trigger_pending.setter
    def _adaptive_trigger_pending(self, value: bool) -> None:
        self.motion_events.adaptive_trigger_pending = value

    @property
    def _adaptive_last_completed_at(self) -> float:
        return self.motion_events.adaptive_last_completed_at

    @_adaptive_last_completed_at.setter
    def _adaptive_last_completed_at(self, value: float) -> None:
        self.motion_events.adaptive_last_completed_at = value

    @property
    def _priority_motion_times(self) -> deque[float]:
        return self.motion_events.priority_motion_times

    @property
    def _camera_motion_times(self) -> deque[float]:
        return self.motion_events.camera_motion_times

    @property
    def _motion_frames(self) -> deque[tuple[float, np.ndarray]]:
        return self.motion_analysis.frames

    @property
    def _motion_color_frames(self) -> deque[tuple[float, np.ndarray]]:
        return self.motion_analysis.color_frames

    @property
    def _motion_analysis_queue(self) -> queue.Queue[float | None]:
        return self.motion_analysis.queue

    @property
    def _motion_analysis_thread(self) -> threading.Thread | None:
        return self.motion_analysis.thread

    @_motion_analysis_thread.setter
    def _motion_analysis_thread(self, value: threading.Thread | None) -> None:
        self.motion_analysis.thread = value

    @property
    def _motion_last_sample(self) -> float:
        return self.motion_analysis.last_sample_clock

    @_motion_last_sample.setter
    def _motion_last_sample(self, value: float) -> None:
        self.motion_analysis.last_sample_clock = value

    @property
    def _motion_last_continuous_result(self) -> MotionQualificationResult | None:
        return self.motion_analysis.last_continuous_result

    @_motion_last_continuous_result.setter
    def _motion_last_continuous_result(
        self,
        value: MotionQualificationResult | None,
    ) -> None:
        self.motion_analysis.last_continuous_result = value

    @property
    def _motion_analysis_last_processed_at(self) -> float:
        return self.motion_analysis.last_processed_at

    @_motion_analysis_last_processed_at.setter
    def _motion_analysis_last_processed_at(self, value: float) -> None:
        self.motion_analysis.last_processed_at = value

    @property
    def _motion_primary_last_processed_at(self) -> float:
        return self.motion_analysis.primary_last_processed_at

    @_motion_primary_last_processed_at.setter
    def _motion_primary_last_processed_at(self, value: float) -> None:
        self.motion_analysis.primary_last_processed_at = value

    @property
    def _motion_debug_last_run(self) -> float:
        return self.motion_analysis.debug_last_run_clock

    @_motion_debug_last_run.setter
    def _motion_debug_last_run(self, value: float) -> None:
        self.motion_analysis.debug_last_run_clock = value

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._enabled:
                return
            residual_workers = [
                label
                for label, running in (
                    (
                        "motion events",
                        self._motion_thread is not None
                        and self._motion_thread.is_alive(),
                    ),
                    (
                        "motion analysis",
                        self._motion_analysis_thread is not None
                        and self._motion_analysis_thread.is_alive(),
                    ),
                    (
                        "capture",
                        any(thread.is_alive() for thread in self.capture.threads().values()),
                    ),
                    ("ONVIF", self.onvif.running),
                    ("object tracking", self.object_tracking.running()),
                )
                if running
            ]
            if residual_workers:
                raise RuntimeError(
                    f"cannot start camera {self.camera.id} while stale workers remain: "
                    f"{', '.join(residual_workers)}"
                )
            self._enabled = True
            self._accepting_motion_events = True
            self._stop.clear()
            self.object_tracking.set_accepting(self._detection_enabled)
            try:
                self.motion_analysis.start(self._stop)
                if not self.capture.start():
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
            self._accepting_motion_events = False
            self._stop.set()
            self._active_incident_event_id = None
            # Stop accepting camera I/O and release the scarce ONVIF
            # subscription before waiting for tracking inference to finish.
            self.capture.request_stop()
            self.onvif.stop()
            self.object_tracking.stop()
            self.motion_analysis.request_stop()
            self.motion_events.signal_stop()
            motion_thread = self._motion_thread
            if motion_thread is not None:
                motion_thread.join(timeout=MOTION_THREAD_STOP_TIMEOUT_SECONDS)
                if motion_thread.is_alive():
                    LOGGER.error("motion worker did not stop for %s", self.camera.id)
            self._motion_thread = motion_thread if motion_thread is not None and motion_thread.is_alive() else None
            analysis_stopped = self.motion_analysis.wait_stopped(
                CAPTURE_STOP_TIMEOUT_SECONDS
            )
            if not analysis_stopped:
                LOGGER.error("motion analysis worker did not stop for %s", self.camera.id)
            alive_threads = self.capture.wait_stopped(CAPTURE_STOP_TIMEOUT_SECONDS)
            alive = sorted(alive_threads)
            if alive:
                logging.getLogger("uvicorn.error").error(
                    "camera capture threads did not stop for %s: %s",
                    self.camera.id,
                    ", ".join(alive),
                )
                current_frames = sys._current_frames()
                for source, thread in alive_threads.items():
                    if thread.ident is None:
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
                self._tracking_frames.clear()
                self._tracking_last_sample_epoch = 0.0
            event_worker_stopped = self._motion_thread is None
            motion_workers_stopped = event_worker_stopped and analysis_stopped
            if motion_workers_stopped:
                self.motion_analysis.reset()
                self.motion_evidence.clear()
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
            if motion_workers_stopped:
                self.motion_events.reset()
            else:
                LOGGER.error(
                    "preserving motion event state for %s because a motion worker is still active",
                    self.camera.id,
                )
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
            try:
                self.capture.close()
            except BaseException as exc:
                failures.append(exc)
                LOGGER.exception("capture cleanup failed for %s", self.camera.id)
            if failures:
                first_error = failures[0]
                if not isinstance(first_error, Exception):
                    raise first_error
                raise RuntimeError(
                    f"one or more camera resources failed to close for {self.camera.id}"
                ) from first_error

    def status(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            enabled = self._enabled
        capture_status = self.capture.status()
        motion_analysis_status = self.motion_analysis.status()
        live_frame_at = str(capture_status["live_frame_at"])
        main_frame_at = str(capture_status["main_frame_at"])
        live_frame_clock = capture_status["live_frame_monotonic"]
        main_frame_clock = capture_status["main_frame_monotonic"]
        evidence_status = self.motion_evidence.status()
        mog2_status = evidence_status.get("mog2", {})
        now = time.monotonic()
        live_age = max(0.0, now - live_frame_clock) if live_frame_clock is not None else None
        main_age = max(0.0, now - main_frame_clock) if main_frame_clock is not None else None
        connected = bool(enabled and live_age is not None and live_age <= FRAME_STALE_SECONDS)
        mode, sensitivity, frame_width = self._motion_settings()
        stationary_object_tolerance = self._stationary_object_tolerance()
        rescue_enabled, rescue_margin = self._motion_rescue_settings()
        visual_backup = self._visual_backup_settings()
        visual_backup_status = self.motion_analysis.visual_backup_snapshot()
        with self._motion_stats_lock:
            motion_stats = dict(self._motion_stats)
        return {
            "id": self.camera.id,
            "name": self.camera.name,
            "running": enabled,
            "connected": connected,
            "capture_running": bool(capture_status["live_running"]),
            "frame_fresh": connected,
            "last_frame_age_seconds": round(live_age, 3) if live_age is not None else None,
            "main_running": bool(capture_status["main_running"]),
            "main_frame_fresh": bool(main_age is not None and main_age <= FRAME_STALE_SECONDS),
            "main_last_frame_age_seconds": round(main_age, 3) if main_age is not None else None,
            "last_frame_at": live_frame_at,
            "main_last_frame_at": main_frame_at,
            "last_error": capture_status["last_error"],
            "main_last_error": capture_status["main_error"],
            "capture_stats": capture_status["capture_stats"],
            "stream_dimensions": capture_status["stream_dimensions"],
            "onvif_enabled": self.camera.onvif.enabled,
            "onvif_connected": self.onvif.connected,
            "onvif_last_event_at": self.onvif.last_event_at,
            "onvif_last_camera_event_at": self.onvif.last_camera_event_at,
            "onvif_last_motion_event_at": self.onvif.last_motion_event_at,
            "last_motion_at": self.last_motion_at,
            "detection_enabled": self._detection_enabled,
            "object_tracking": {
                **self.object_tracking.status(),
                **self.motion_incidents.status(),
            },
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
                "illumination_filter_enabled": self._illumination_filter_enabled(),
                "frame_width": frame_width,
                "camera_mode_background_fps": self.motion_config.camera_mode_background_fps,
                "visual_backup": {
                    "enabled": mode == "camera_rescue",
                    "warmup_seconds": self.motion_config.visual_backup_warmup_seconds,
                    "grace_seconds": visual_backup["grace_seconds"],
                    "minimum_score": visual_backup["minimum_score"],
                    "score_margin": self.motion_config.visual_backup_score_margin,
                    "minimum_consecutive": visual_backup["minimum_consecutive"],
                    "cooldown_seconds": visual_backup["cooldown_seconds"],
                    "maximum_triggers_5m": visual_backup["maximum_triggers_5m"],
                    **visual_backup_status,
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
                "analysis_queue_depth": motion_analysis_status["queue_depth"],
                "analysis_worker_running": motion_analysis_status["worker_running"],
                "event_worker_running": bool(
                    self._motion_thread is not None
                    and self._motion_thread.is_alive()
                ),
                "continuous_last_result": motion_analysis_status[
                    "continuous_last_result"
                ],
                "buffered_frames": motion_analysis_status["buffered_frames"],
                "frame_shape": motion_analysis_status["frame_shape"],
                "color_buffered_frames": motion_analysis_status[
                    "color_buffered_frames"
                ],
                "color_frame_shape": motion_analysis_status["color_frame_shape"],
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

    def create_object_tracking_session(
        self,
        factory: ObjectTrackingSessionFactory,
    ) -> ObjectTrackingSession:
        """Build a replacement tracking session without changing camera I/O."""
        return factory.create(
            camera=self.camera,
            frame_provider=self._get_latest_tracking_frame_with_fallback,
            catchup_frame_provider=self._recorded_tracking_frames,
        )

    def pause_object_tracking_session(self) -> None:
        """Quiesce tracking before an inference engine transition."""
        with self._lifecycle_lock:
            if not self.object_tracking.stop():
                raise RuntimeError(
                    f"object tracking session did not stop for {self.camera.id}"
                )

    def resume_object_tracking_session(self) -> None:
        """Restore tracking eligibility after a cancelled transition."""
        with self._lifecycle_lock:
            self.object_tracking.set_accepting(
                self._detection_enabled and not self._stop.is_set()
            )

    def replace_object_tracking_session(
        self,
        replacement: ObjectTrackingSession,
    ) -> ObjectTrackingSession:
        """Atomically replace tracking while preserving capture and ONVIF state."""
        with self._lifecycle_lock:
            previous = self.object_tracking
            if replacement is previous:
                return previous
            if not previous.stop():
                raise RuntimeError(
                    f"object tracking session did not stop for {self.camera.id}"
                )
            try:
                self.object_tracking = replacement
                tracking_buffer_size = max(
                    4,
                    round(
                        replacement.config.sample_fps * TRACKING_CATCHUP_SECONDS
                    ) + 2,
                )
                with self._frame_lock:
                    self._tracking_frames = deque(
                        self._tracking_frames,
                        maxlen=tracking_buffer_size,
                    )
                    self._tracking_last_sample_epoch = 0.0
                replacement.set_accepting(
                    self._detection_enabled and not self._stop.is_set()
                )
            except BaseException:
                try:
                    replacement.stop()
                except Exception:
                    LOGGER.exception(
                        "replacement object tracking cleanup failed for %s",
                        self.camera.id,
                    )
                finally:
                    self.object_tracking = previous
                try:
                    previous.set_accepting(
                        self._detection_enabled and not self._stop.is_set()
                    )
                except Exception:
                    LOGGER.exception(
                        "previous object tracking session restore failed for %s",
                        self.camera.id,
                    )
                raise
            return previous

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

    def _increment_motion_stat(self, name: str, amount: int = 1) -> None:
        with self._motion_stats_lock:
            self._motion_stats[name] = int(self._motion_stats.get(name) or 0) + amount

    def _record_analysis_wait(self, wait_ms: float) -> None:
        with self._motion_stats_lock:
            self._motion_stats["analysis_wait_ms_total"] += wait_ms
            self._motion_stats["analysis_wait_ms_max"] = max(
                float(self._motion_stats["analysis_wait_ms_max"]),
                wait_ms,
            )

    def _set_last_motion_at(self, value: str) -> None:
        self.last_motion_at = value

    def _record_decision_stats(
        self,
        *,
        result: MotionQualificationResult,
        qualification: dict[str, Any],
        retry_attempt: bool,
        priority: bool,
        mode: str,
        borderline_candidate: bool,
        suppression_verification_candidate: bool,
    ) -> None:
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

    def _record_visual_camera_match(self, observed_at: float) -> bool:
        return self.motion_analysis.record_visual_camera_match(observed_at)

    def _set_active_incident_event_id(self, event_id: int) -> None:
        self._active_incident_event_id = event_id

    def handle_motion_event(
        self,
        topic: str = "manual",
        message: str = "",
        event_at: datetime | None = None,
    ) -> None:
        if not self._accepting_motion_events or not self._detection_enabled:
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
            self.motion_events.remember_camera_motion(received_at)

        self._publish_event_safely("motion", {
            "camera_id": self.camera.id,
            "timestamp": event_at.isoformat(),
            "source": "manual" if topic.startswith("manual") else "onvif",
        })

        self._enqueue_motion_trigger(MotionTrigger(
            topic=topic,
            message=message,
            event_at=event_at,
            received_at=received_at,
        ))

    def _enqueue_motion_trigger(
        self,
        trigger: MotionTrigger | dict[str, Any],
        *,
        evict_oldest: bool = True,
    ) -> bool:
        return self.motion_events.enqueue(
            trigger,
            evict_oldest=evict_oldest,
            on_trigger=self._increment_motion_stat,
            on_drop=self._increment_motion_stat,
        )

    def _clear_motion_queue(self) -> None:
        self.motion_events.clear()

    def _clear_motion_analysis_queue(self) -> None:
        self.motion_analysis.clear_queue()

    def _signal_motion_analysis_stop(self) -> None:
        self.motion_analysis.request_stop()

    def _schedule_motion_analysis(self, captured_at: float) -> None:
        self.motion_analysis.schedule(captured_at, self._stop)

    def _remember_motion_frame(
        self,
        frame: np.ndarray,
        frame_clock: float,
        captured_at: float | None = None,
    ) -> None:
        self.motion_analysis.remember_frame(
            frame,
            frame_clock,
            self._stop,
            captured_at,
        )

    def _run_motion_analysis(self) -> None:
        self.motion_analysis.run(self._stop)

    def _analyze_continuous_motion(self, captured_at: float) -> None:
        self.motion_analysis.analyze_continuous(captured_at)

    def _reset_visual_backup_candidate(self) -> None:
        self.motion_analysis.reset_visual_backup_candidate()

    def _visual_backup_readiness(
        self,
        result: MotionQualificationResult,
        captured_at: float,
    ) -> bool:
        """Compatibility facade for tests and existing runtime callers."""
        return self.motion_analysis.visual_backup_readiness(result, captured_at)

    @property
    def _visual_backup_scene_ready(self) -> bool:
        return self.motion_analysis.visual_backup_scene_ready

    @property
    def _visual_backup_stable_samples(self) -> int:
        return self.motion_analysis.visual_backup_stable_samples

    def _consider_visual_backup(
        self,
        result: MotionQualificationResult,
        samples: list[tuple[float, np.ndarray]],
        captured_at: float,
    ) -> None:
        self.motion_analysis.consider_visual_backup(result, samples, captured_at)

    def _record_visual_backup_readiness_audit(
        self,
        result: MotionQualificationResult,
        captured_at: float,
    ) -> None:
        self.motion_analysis.record_visual_backup_readiness_audit(
            result,
            captured_at,
        )

    def _reserve_adaptive_trigger(self, captured_at: float) -> bool:
        return self.motion_analysis.reserve_adaptive_trigger(captured_at)

    def _adaptive_rearm_seconds(self) -> float:
        return max(
            5.0,
            self.motion_config.window_seconds
            + self.motion_config.post_trigger_seconds
            + self.motion_config.burst_quiet_seconds,
        )

    def _priority_dedup_seconds(self) -> float:
        return max(
            2.0,
            self.motion_config.post_trigger_seconds
            + self.motion_config.burst_quiet_seconds,
        )

    def _remember_priority_motion(self, observed_at: float) -> None:
        self.motion_events.remember_priority(observed_at)

    def _matches_recent_priority_motion(self, event_at: float) -> bool:
        return self.motion_events.matches_recent_priority(
            event_at,
            rearm_seconds=self._priority_dedup_seconds(),
        )

    def _defer_adaptive_trigger(self, captured_at: float) -> None:
        self.motion_events.defer_adaptive(captured_at)

    def _complete_adaptive_trigger(self, triggers: MotionTriggerBatch) -> None:
        self.motion_events.complete_adaptive(triggers, time.time())

    def _capture_motion_debug(self, captured_at: float) -> None:
        self.motion_analysis.capture_debug(captured_at)

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

    def _visual_backup_settings(self) -> dict[str, float | int]:
        override = self.camera.motion_qualification
        return {
            "grace_seconds": float(
                self.motion_config.visual_backup_grace_seconds
                if override.visual_backup_grace_seconds is None
                else override.visual_backup_grace_seconds
            ),
            "minimum_score": float(
                self.motion_config.visual_backup_min_score
                if override.visual_backup_min_score is None
                else override.visual_backup_min_score
            ),
            "minimum_consecutive": int(
                self.motion_config.visual_backup_min_consecutive
                if override.visual_backup_min_consecutive is None
                else override.visual_backup_min_consecutive
            ),
            "cooldown_seconds": float(
                self.motion_config.visual_backup_cooldown_seconds
                if override.visual_backup_cooldown_seconds is None
                else override.visual_backup_cooldown_seconds
            ),
            "maximum_triggers_5m": int(
                self.motion_config.visual_backup_max_triggers_5m
                if override.visual_backup_max_triggers_5m is None
                else override.visual_backup_max_triggers_5m
            ),
        }

    def _visual_backup_policy(self) -> VisualBackupPolicy:
        settings = self._visual_backup_settings()
        return VisualBackupPolicy(
            warmup_seconds=self.motion_config.visual_backup_warmup_seconds,
            grace_seconds=float(settings["grace_seconds"]),
            minimum_score=float(settings["minimum_score"]),
            score_margin=self.motion_config.visual_backup_score_margin,
            minimum_consecutive=int(settings["minimum_consecutive"]),
            cooldown_seconds=float(settings["cooldown_seconds"]),
            maximum_triggers_5m=int(settings["maximum_triggers_5m"]),
            sample_fps=self.motion_config.sample_fps,
            background_fps=self.motion_config.camera_mode_background_fps,
        )

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

    def _continuous_primary_analysis_due(
        self,
        captured_at: float,
        *,
        last_processed_at: float | None = None,
    ) -> bool:
        if self._trigger_mode() == "adaptive":
            return True
        background_fps = min(
            self.motion_config.sample_fps,
            self.motion_config.camera_mode_background_fps,
        )
        interval = 1.0 / max(0.5, background_fps)
        previous = (
            self._motion_primary_last_processed_at
            if last_processed_at is None
            else last_processed_at
        )
        return bool(
            previous <= 0.0 or captured_at - previous >= interval * 0.85
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

    def _illumination_filter_enabled(self) -> bool:
        override = self.camera.motion_qualification.illumination_filter_enabled
        return bool(
            self.motion_config.illumination_filter_enabled
            if override is None
            else override
        )

    @staticmethod
    def _should_verify_suppression(decision_id: str, rate: float) -> bool:
        return should_verify_suppression(decision_id, rate)

    @staticmethod
    def _is_borderline_candidate(
        result: MotionQualificationResult,
        enabled: bool,
        margin: float,
    ) -> bool:
        return is_borderline_candidate(result, enabled, margin)

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
        return audit_features(result)

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
        return priority_motion_topic(topic)

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
            samples = self.motion_analysis.samples_since(
                anchor - self.motion_config.window_seconds
            )

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
                "illumination_filter_enabled": self._illumination_filter_enabled(),
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
        self.motion_decisions.run(self._stop)

    def _retry_motion_trigger_batch(
        self,
        triggers: MotionTriggerBatch | list[dict[str, Any]],
    ) -> RetryDisposition:
        return self.motion_decisions.retry_batch(triggers, self._stop)

    def _run_motion_events_until_error(self) -> None:
        self.motion_decisions.run_until_error(self._stop)

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
        outcome = self.motion_incidents.process(
            topic,
            message,
            event_at,
            qualification,
            require_eligible_object=require_eligible_object,
            require_motion_correlation=require_motion_correlation,
        )
        return outcome.as_dict()

    def _related_incident_event_id(self, result: MotionQualificationResult) -> int | None:
        if result.reason not in INCIDENT_ACTIVITY_REASONS:
            return None
        return self._active_incident_event_id

    def _capture_frame(self, frame: CapturedFrame) -> None:
        if frame.source == "live":
            self._remember_motion_frame(
                frame.image,
                frame.captured_at_monotonic,
                frame.captured_at_epoch,
            )
        elif frame.source == "main":
            self._remember_tracking_frame(frame.image, frame.captured_at_epoch)

    def _capture_source_started(self, source: str) -> None:
        if source == "main":
            with self._frame_lock:
                self._tracking_frames.clear()
                self._tracking_last_sample_epoch = 0.0

    def _capture_source_stopped(self, source: str) -> None:
        if source == "main":
            with self._frame_lock:
                self._tracking_frames.clear()
                self._tracking_last_sample_epoch = 0.0

    def _get_latest_frame(self, source: str = "live") -> Any:
        source = self.camera.normalized_source(source)
        if self._stop.is_set():
            return None
        frame = self.capture.request_frame(source)
        return frame.image if frame is not None else None

    def _get_latest_tracking_frame(
        self,
        source: str = "main",
    ) -> tuple[np.ndarray, float, float] | None:
        source = self.camera.normalized_source(source)
        if self._stop.is_set():
            return None
        frame = self.capture.request_frame(source)
        if frame is None:
            return None
        return frame.image, frame.captured_at_epoch, frame.captured_at_monotonic

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

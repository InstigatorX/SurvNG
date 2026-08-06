from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .camera_capture import (
    CaptureBackend,
    CameraCaptureService,
    CapturedFrame,
    CaptureOpenLimiter,
    OpenCvFfmpegCaptureBackend,
)
from .camera_status import CameraStatusHooks, CameraStatusService
from .camera_media import CameraMediaService
from .config import CameraConfig, DetectionZone, MotionQualificationConfig
from .image_storage import DurableImageWriter
from .onvif_events import OnvifEventListener
from .motion import MotionQualificationResult
from .motion_analysis import FairMotionAnalysisLimiter
from .motion_analysis_service import MotionAnalysisHooks, MotionAnalysisService
from .motion_qualification_service import (
    MotionQualificationHooks,
    MotionQualificationService,
)
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
from .object_tracking_lifecycle import ObjectTrackingLifecycle
from .tracking_frames import TrackingFrameService
from .motion_pipeline import (
    MotionDecisionHandlerFactory,
    MotionDebugSnapshotStore,
    MotionEvidenceRepository,
    MotionPipeline,
    RecordedMotionObjectDetectorFactory,
)

LOGGER = logging.getLogger(__name__)
MOTION_THREAD_STOP_TIMEOUT_SECONDS = 22.0
# OpenCV's FFmpeg calls cannot be interrupted safely from another thread.
# Keep their own deadlines below CameraWorker.stop()'s join budget so a stop
# request can always regain control after a blocked open/read operation.
CAPTURE_STOP_TIMEOUT_SECONDS = 8.0
MOTION_QUEUE_SIZE = 32
MOTION_ANALYSIS_QUEUE_SIZE = 1
MOTION_EVENT_MAX_RETRIES = 2
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
        self._stop = threading.Event()
        self._stop.set()
        self._enabled = False
        self._detection_enabled = True
        self._accepting_motion_events = True
        self._lifecycle_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        effective_capture_backend = capture_backend or OpenCvFfmpegCaptureBackend(
            CaptureOpenLimiter()
        )
        self.motion_debug = MotionDebugSnapshotStore()
        self.motion_object_detector = motion_object_detector_factory.create(
            camera=camera,
            live_frame_provider=lambda: self._get_latest_frame(),
        )
        self.media = CameraMediaService(
            camera=camera,
            storage_dir=storage_dir,
            image_writer=image_writer,
            motion_detector=self.motion_object_detector,
            frame_provider=lambda source: self._get_latest_frame(source),
            rejected_sample_rate=lambda: self.motion_config.rejected_sample_rate,
            stop_requested=self._stop.is_set,
        )
        self.tracking_lifecycle = ObjectTrackingLifecycle(
            camera=camera,
            factory=object_tracking_session_factory,
            frame_provider=self._get_latest_tracking_frame_with_fallback,
            catchup_frame_provider=self._recorded_tracking_frames,
            prewarm_frame_provider=lambda: self._get_latest_tracking_frame("main"),
            history=lambda: self.tracking_frames,
            accepting=lambda: self._detection_enabled and not self._stop.is_set(),
            lifecycle_lock=self._lifecycle_lock,
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
            tracking_enabled=self.tracking_lifecycle.enabled,
            has_trackable_objects=self.tracking_lifecycle.has_trackable_objects,
            start_tracking=self.tracking_lifecycle.start_incident,
            prewarm_tracking=self.tracking_lifecycle.prewarm,
            image_reader=self.media.read_image,
        )
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
        self.motion_qualification = MotionQualificationService(
            camera=camera,
            config=self.motion_config,
            qualification_pipeline=self.motion_pipeline,
            observation_pipeline=self.motion_observation_pipeline,
            fusion_pipeline=self.motion_fusion_pipeline,
            pipeline_origins=self.motion_pipeline_origins,
            debug_store=self.motion_debug,
            stop_event=self._stop,
            hooks=MotionQualificationHooks(
                samples_since=lambda captured_at: self.motion_analysis.samples_since(
                    captured_at
                ),
                increment_stat=lambda name, amount: self._increment_motion_stat(
                    name, amount
                ),
            ),
        )
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
            analysis_lock=self.motion_qualification.analysis_lock,
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
        self.capture = CameraCaptureService(
            camera_id=camera.id,
            source_url=camera.source_url,
            backend=effective_capture_backend,
            frame_observer=self._capture_frame,
            source_started_observer=self._capture_source_started,
            source_stopped_observer=self._capture_source_stopped,
        )
        self.tracking_frames = TrackingFrameService(
            camera=camera,
            capture=self.capture,
            recorder=self.motion_object_detector.recorder,
            stop_event=self._stop,
            sample_fps=self.tracking_lifecycle.sample_fps,
        )
        self.onvif = OnvifEventListener(
            camera,
            self.handle_motion_event,
            cache_dir=onvif_cache_dir or storage_dir / "onvif",
        )
        self.status_reporter = CameraStatusService(
            camera=camera,
            motion_config=self.motion_config,
            capture=self.capture,
            motion_analysis=self.motion_analysis,
            motion_evidence=self.motion_evidence,
            onvif=self.onvif,
            qualification_pipeline=self.motion_pipeline,
            observation_pipeline=self.motion_observation_pipeline,
            fusion_pipeline=self.motion_fusion_pipeline,
            debug_store=self.motion_debug,
            pipeline_origins=self.motion_pipeline_origins,
            hooks=CameraStatusHooks(
                runtime_state=self._status_runtime_state,
                motion_settings=self._motion_settings,
                stationary_object_tolerance=self._stationary_object_tolerance,
                rescue_settings=self._motion_rescue_settings,
                visual_backup_settings=self._visual_backup_settings,
                illumination_filter_enabled=self._illumination_filter_enabled,
                suppression_verification_rate=self._suppression_verification_rate,
                motion_stats=self._motion_stats_snapshot,
                object_tracking_status=self.tracking_lifecycle.status,
                incident_status=self.motion_incidents.status,
                event_worker_running=lambda: bool(
                    self._motion_thread is not None
                    and self._motion_thread.is_alive()
                ),
                event_queue_depth=self._motion_queue.qsize,
                retry_queue_depth=self.motion_events.retry_queue_depth,
            ),
        )

    # Transitional compatibility accessors keep diagnostics and existing
    # integrations stable while MotionEventCoordinator owns the runtime.
    @property
    def _motion_queue(self) -> queue.Queue:
        return self.motion_events.queue

    @property
    def _motion_retry_batches(self) -> deque[MotionTrigger]:
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
    def _tracking_frames(self) -> deque[tuple[float, np.ndarray]]:
        return self.tracking_frames.frames

    @property
    def object_tracking(self) -> ObjectTrackingSession:
        """Compatibility view of the session owned by the lifecycle coordinator."""
        return self.tracking_lifecycle.current()

    @object_tracking.setter
    def object_tracking(self, value: ObjectTrackingSession) -> None:
        self.tracking_lifecycle.bind_for_compatibility(value)

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
                    ("object tracking", self.tracking_lifecycle.running()),
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
            try:
                self.tracking_lifecycle.sync_accepting()
                # Clear state left by the previous run before producers start.
                # Capture and motion analysis may enqueue a legitimate trigger
                # as soon as they start, and that work must not be discarded.
                self._clear_motion_queue()
                self.motion_analysis.start(self._stop)
                if not self.capture.start():
                    raise RuntimeError(f"camera source did not start for {self.camera.id}")
                if self._motion_thread is None or not self._motion_thread.is_alive():
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
                except BaseException:
                    LOGGER.exception(
                        "camera startup rollback was incomplete for %s",
                        self.camera.id,
                    )
                raise

    def stop(self) -> None:
        with self._lifecycle_lock:
            shutdown_errors: list[tuple[str, BaseException]] = []

            def attempt(label: str, operation: Callable[[], Any]) -> Any:
                try:
                    return operation()
                except BaseException as error:
                    shutdown_errors.append((label, error))
                    LOGGER.exception("%s stop failed for %s", label, self.camera.id)
                    return None

            self._enabled = False
            self._accepting_motion_events = False
            self._stop.set()
            self._active_incident_event_id = None
            # Stop accepting camera I/O and release the scarce ONVIF
            # subscription before waiting for tracking inference to finish.
            attempt("capture", self.capture.request_stop)
            attempt("ONVIF", self.onvif.stop)
            tracking_stopped = attempt(
                "object tracking",
                self.tracking_lifecycle.stop,
            )
            if tracking_stopped is False:
                shutdown_errors.append((
                    "object tracking",
                    RuntimeError("object tracking session did not stop"),
                ))
            attempt("motion analysis", self.motion_analysis.request_stop)
            attempt("motion events", self.motion_events.signal_stop)
            motion_thread = self._motion_thread
            if motion_thread is not None:
                attempt(
                    "motion event worker join",
                    lambda: motion_thread.join(
                        timeout=MOTION_THREAD_STOP_TIMEOUT_SECONDS
                    ),
                )
                motion_thread_alive = attempt(
                    "motion event worker status",
                    motion_thread.is_alive,
                )
                if motion_thread_alive is not False:
                    LOGGER.error("motion worker did not stop for %s", self.camera.id)
            else:
                motion_thread_alive = False
            self._motion_thread = motion_thread if motion_thread_alive is not False else None
            analysis_stopped = attempt(
                "motion analysis wait",
                lambda: self.motion_analysis.wait_stopped(
                    CAPTURE_STOP_TIMEOUT_SECONDS
                ),
            )
            analysis_stopped = analysis_stopped is True
            if not analysis_stopped:
                LOGGER.error("motion analysis worker did not stop for %s", self.camera.id)
            alive_threads = attempt(
                "capture wait",
                lambda: self.capture.wait_stopped(CAPTURE_STOP_TIMEOUT_SECONDS),
            )
            if not isinstance(alive_threads, dict):
                alive_threads = {}
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
            attempt("tracking frame history", self.tracking_frames.clear)
            event_worker_stopped = self._motion_thread is None
            motion_workers_stopped = event_worker_stopped and analysis_stopped
            if motion_workers_stopped:
                attempt("motion analysis reset", self.motion_analysis.reset)
                attempt("motion evidence reset", self.motion_evidence.clear)
                attempt(
                    "motion qualification reset",
                    self.motion_qualification.reset_runtime,
                )
            else:
                LOGGER.error(
                    "preserving motion runtime for %s because a motion worker is still active",
                    self.camera.id,
                )
            if motion_workers_stopped:
                attempt("motion event state reset", self.motion_events.reset)
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
            onvif_running = attempt(
                "ONVIF status",
                lambda: self.onvif.running,
            )
            if onvif_running is not False:
                shutdown_failures.append("ONVIF worker")
            tracking_running = attempt(
                "object tracking status",
                self.tracking_lifecycle.running,
            )
            if tracking_running is not False:
                shutdown_failures.append("object tracking worker")
            shutdown_failures.extend(
                f"{label} cleanup"
                for label, _error in shutdown_errors
                if f"{label} cleanup" not in shutdown_failures
            )
            if shutdown_failures:
                failure = RuntimeError(
                    f"camera {self.camera.id} did not stop cleanly ({'; '.join(shutdown_failures)})"
                )
                if shutdown_errors:
                    first_error = shutdown_errors[0][1]
                    if not isinstance(first_error, Exception):
                        raise first_error
                    raise failure from first_error
                raise failure

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

    def _status_runtime_state(self) -> tuple[bool, bool, str]:
        with self._lifecycle_lock:
            return self._enabled, self._detection_enabled, self.last_motion_at

    def _motion_stats_snapshot(self) -> dict[str, Any]:
        with self._motion_stats_lock:
            return dict(self._motion_stats)

    def status(self) -> dict[str, Any]:
        return self.status_reporter.snapshot()

    def update_zones(self, zones: list[DetectionZone]) -> None:
        next_zones = [zone.model_copy(deep=True) for zone in zones]
        with self._lifecycle_lock:
            self.camera.zones = next_zones

    def set_detection_enabled(self, enabled: bool) -> None:
        with self._lifecycle_lock:
            previous = self._detection_enabled
            self._detection_enabled = bool(enabled)
            try:
                self.tracking_lifecycle.sync_accepting()
            except BaseException:
                self._detection_enabled = previous
                try:
                    self.tracking_lifecycle.sync_accepting()
                except BaseException:
                    LOGGER.exception(
                        "object tracking eligibility rollback failed for %s",
                        self.camera.id,
                    )
                raise

    def create_object_tracking_session(
        self,
        factory: ObjectTrackingSessionFactory,
    ) -> ObjectTrackingSession:
        """Build a replacement tracking session without changing camera I/O."""
        return self.tracking_lifecycle.create(factory)

    def pause_object_tracking_session(self) -> None:
        """Quiesce tracking before an inference engine transition."""
        self.tracking_lifecycle.pause()

    def resume_object_tracking_session(self) -> None:
        """Restore tracking eligibility after a cancelled transition."""
        self.tracking_lifecycle.sync_accepting()

    def replace_object_tracking_session(
        self,
        replacement: ObjectTrackingSession,
    ) -> ObjectTrackingSession:
        """Atomically replace tracking while preserving capture and ONVIF state."""
        return self.tracking_lifecycle.replace(replacement)

    def snapshot(self, source: str = "live") -> bytes | None:
        return self.media.snapshot(source)

    def mjpeg_frames(self, fps: float = 4.0, source: str = "live") -> Iterator[bytes]:
        yield from self.media.mjpeg_frames(fps, source)

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
        self.motion_qualification.observe_event(topic, message, event_at, received_at)

    def _motion_settings(self) -> tuple[str, str, int]:
        return self.motion_qualification.settings()

    def _stationary_object_tolerance(self) -> str:
        return self.motion_qualification.stationary_object_tolerance()

    def _visual_backup_settings(self) -> dict[str, float | int]:
        return self.motion_qualification.visual_backup_settings()

    def _visual_backup_policy(self) -> VisualBackupPolicy:
        return self.motion_qualification.visual_backup_policy()

    def _trigger_mode(self) -> str:
        return self.motion_qualification.trigger_mode()

    def _fusion_options(self) -> dict[str, Any]:
        return self.motion_qualification.fusion_options()

    def _adaptive_analysis_required(self) -> bool:
        return self.motion_qualification.adaptive_analysis_required()

    def _continuous_primary_analysis_required(self) -> bool:
        return self.motion_qualification.continuous_primary_required()

    def _continuous_primary_analysis_due(
        self,
        captured_at: float,
        *,
        last_processed_at: float | None = None,
    ) -> bool:
        previous = (
            self._motion_primary_last_processed_at
            if last_processed_at is None
            else last_processed_at
        )
        return self.motion_qualification.continuous_primary_due(
            captured_at, previous
        )

    def _external_confirmation_required(self) -> bool:
        return self.motion_qualification.external_confirmation_required()

    def _frame_motion_analysis_required(self) -> bool:
        return self.motion_qualification.frame_analysis_required()

    def _motion_rescue_settings(self) -> tuple[bool, float]:
        return self.motion_qualification.rescue_settings()

    def _suppression_verification_rate(self) -> float:
        return self.motion_qualification.suppression_verification_rate()

    def _illumination_filter_enabled(self) -> bool:
        return self.motion_qualification.illumination_filter_enabled()

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
        return self.motion_qualification.with_source_evidence(
            result,
            start_epoch,
            end_epoch,
            include_telemetry=include_telemetry,
            require_primary_trigger=require_primary_trigger,
        )

    def _reset_motion_fusion_runtime(self) -> None:
        self.motion_qualification.reset_event_state_runtime()

    def _validation_fail_open_result(
        self,
        component: str,
        error: Exception,
        original: MotionQualificationResult | None = None,
        *,
        allow_detection: bool = True,
    ) -> MotionQualificationResult:
        return self.motion_qualification.validation_fail_open_result(
            component,
            error,
            original,
            allow_detection=allow_detection,
        )

    @staticmethod
    def _audit_features(result: MotionQualificationResult) -> dict[str, Any]:
        return audit_features(result)

    def _with_pipeline_telemetry(
        self,
        result: MotionQualificationResult,
    ) -> MotionQualificationResult:
        return self.motion_qualification.with_pipeline_telemetry(result)

    @staticmethod
    def _priority_motion_topic(topic: str) -> bool:
        return priority_motion_topic(topic)

    def _qualify_motion_burst(
        self,
        event_at: datetime,
        received_at: float,
        sensitivity: str,
    ) -> tuple[MotionQualificationResult, dict[str, Any]]:
        return self.motion_qualification.qualify_burst(
            event_at, received_at, sensitivity
        )

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
        return self.motion_qualification.run_pipeline(
            frames,
            sensitivity,
            captured_at,
            frame_timestamps,
            isolated=isolated,
            capture_debug=capture_debug,
            include_telemetry=include_telemetry,
        )

    def set_motion_debug_enabled(self, enabled: bool) -> None:
        self.motion_qualification.set_debug_enabled(enabled)

    def motion_debug_status(self) -> dict[str, Any]:
        return self.motion_qualification.debug_status()

    def motion_debug_image(self, layer: str) -> bytes | None:
        return self.motion_qualification.debug_image(layer)

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
        return self.media.sample_rejected_motion(event_at, result)

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
            self.tracking_frames.clear()

    def _capture_source_stopped(self, source: str) -> None:
        if source == "main":
            self.tracking_frames.clear()

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
        return self.tracking_frames.latest(source)

    def _get_latest_tracking_frame_with_fallback(
        self,
    ) -> tuple[np.ndarray, float, float] | None:
        return self._get_latest_tracking_frame("main") or self._get_latest_tracking_frame(
            "live"
        )

    def _remember_tracking_frame(self, frame: np.ndarray, captured_at: float) -> None:
        self.tracking_frames.remember(frame, captured_at)

    def _recorded_tracking_frames(
        self,
        start_epoch: float,
        end_epoch: float,
        sample_fps: float,
        frame_width: int,
    ) -> Iterator[tuple[float, np.ndarray]]:
        yield from self.tracking_frames.recorded_frames(
            start_epoch,
            end_epoch,
            sample_fps,
            frame_width,
        )

    def _recorded_motion_frame(
        self,
        event_at: datetime,
    ) -> tuple[Any | None, list[dict[str, Any]], str]:
        return self.media.detect_recorded_motion(event_at)

    def _write_snapshot(self, frame: Any, event_at: datetime | None = None) -> str:
        return self.media.write_snapshot(frame, event_at)

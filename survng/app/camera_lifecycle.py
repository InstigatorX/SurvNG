"""Camera runtime state and coordinated worker lifecycle."""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable

from .camera_capture import CameraCaptureService
from .motion_analysis_service import MotionAnalysisService
from .motion_events import MotionEventCoordinator
from .motion_pipeline import MotionEvidenceRepository, MotionPipeline
from .motion_qualification_service import MotionQualificationService
from .object_tracking_lifecycle import ObjectTrackingLifecycle
from .onvif_events import OnvifEventListener
from .security import redact_secret_text
from .tracking_frames import TrackingFrameService

LOGGER = logging.getLogger(__name__)
MOTION_THREAD_STOP_TIMEOUT_SECONDS = 22.0
# OpenCV's FFmpeg calls cannot be interrupted safely from another thread.
# Keep their own deadlines below stop()'s join budget so a stop request can
# always regain control after a blocked open/read operation.
CAPTURE_STOP_TIMEOUT_SECONDS = 8.0


class CameraLifecyclePhase(StrEnum):
    """Observable phases for one camera runtime generation."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(slots=True)
class CameraRuntimeState:
    """Mutable camera lifecycle state shared through one explicit owner."""

    # This lock protects only small in-memory state transitions. Blocking I/O,
    # joins, and resource cleanup must never execute while it is held.
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    phase: CameraLifecyclePhase = CameraLifecyclePhase.STOPPED
    enabled: bool = False
    detection_enabled: bool = True
    accepting_motion_events: bool = False
    active_incident_event_id: int | None = None
    generation: int = 0
    transition_count: int = 0
    last_transition_at: str = ""
    phase_started_at_monotonic: float = field(default_factory=time.monotonic)
    last_phase_duration_ms: float = 0.0
    last_failure: str = ""

    def __post_init__(self) -> None:
        self.stop_event.set()


class CameraLifecycleService:
    """Start, stop, and close the workers composing one camera runtime."""

    def __init__(
        self,
        *,
        camera_id: str,
        state: CameraRuntimeState,
        capture: CameraCaptureService,
        onvif: OnvifEventListener,
        tracking: ObjectTrackingLifecycle,
        motion_analysis: MotionAnalysisService,
        motion_events: MotionEventCoordinator,
        tracking_frames: TrackingFrameService,
        motion_evidence: MotionEvidenceRepository,
        motion_qualification: MotionQualificationService,
        motion_pipelines: tuple[tuple[str, MotionPipeline], ...],
        run_motion_events: Callable[[], None],
    ) -> None:
        self.camera_id = camera_id
        self.state = state
        self.capture = capture
        self.onvif = onvif
        self.tracking = tracking
        self.motion_analysis = motion_analysis
        self.motion_events = motion_events
        self.tracking_frames = tracking_frames
        self.motion_evidence = motion_evidence
        self.motion_qualification = motion_qualification
        self.motion_pipelines = motion_pipelines
        self.run_motion_events = run_motion_events
        # Serialize lifecycle commands without blocking state readers. This may
        # be held across joins and camera I/O; state.lock may not.
        self._operation_lock = threading.Lock()

    def start(self) -> None:
        with self._operation_lock:
            with self.state.lock:
                if self.state.phase is CameraLifecyclePhase.RUNNING:
                    return
                if self.state.phase is CameraLifecyclePhase.CLOSED:
                    raise RuntimeError(f"camera {self.camera_id} is closed")
            residual_workers = self._residual_workers()
            if residual_workers:
                message = (
                    f"cannot start camera {self.camera_id} while stale workers remain: "
                    f"{', '.join(residual_workers)}"
                )
                with self.state.lock:
                    self._transition_locked(
                        CameraLifecyclePhase.FAILED,
                        failure=message,
                    )
                raise RuntimeError(message)
            with self.state.lock:
                self.state.enabled = True
                self.state.accepting_motion_events = True
                self.state.stop_event.clear()
                self.state.generation += 1
                self._transition_locked(CameraLifecyclePhase.STARTING)
            try:
                self.tracking.sync_accepting()
                # Clear previous-run state before any producer can enqueue new work.
                self.motion_events.clear()
                self.motion_analysis.start(self.state.stop_event)
                if not self.capture.start():
                    raise RuntimeError(
                        f"camera source did not start for {self.camera_id}"
                    )
                motion_thread = self.motion_events.thread
                if motion_thread is None or not motion_thread.is_alive():
                    motion_thread = threading.Thread(
                        target=self.run_motion_events,
                        name=f"motion-{self.camera_id}",
                        daemon=False,
                    )
                    self.motion_events.thread = motion_thread
                    try:
                        motion_thread.start()
                    except BaseException:
                        self.motion_events.thread = None
                        raise
                self.onvif.start()
            except BaseException as startup_error:
                try:
                    self._stop_runtime(
                        final_phase=CameraLifecyclePhase.FAILED,
                        failure=f"{type(startup_error).__name__}: {startup_error}",
                    )
                except BaseException as rollback_error:
                    LOGGER.error(
                        "camera startup rollback was incomplete for %s: %s",
                        self.camera_id,
                        redact_secret_text(rollback_error),
                    )
                if not isinstance(startup_error, Exception):
                    raise
                raise RuntimeError(
                    redact_secret_text(
                        f"{type(startup_error).__name__}: {startup_error}"
                    )
                ) from None
            with self.state.lock:
                self._transition_locked(CameraLifecyclePhase.RUNNING)

    def stop(self) -> None:
        with self._operation_lock:
            self._stop_runtime()

    def _stop_runtime(
        self,
        *,
        final_phase: CameraLifecyclePhase = CameraLifecyclePhase.STOPPED,
        failure: str = "",
    ) -> None:
        shutdown_errors: list[tuple[str, BaseException]] = []

        def attempt(label: str, operation: Callable[[], Any]) -> Any:
            try:
                return operation()
            except BaseException as error:
                shutdown_errors.append((label, error))
                LOGGER.error(
                    "%s stop failed for %s: %s",
                    label,
                    self.camera_id,
                    redact_secret_text(error),
                )
                return None

        with self.state.lock:
            if self.state.phase is CameraLifecyclePhase.CLOSED:
                return
            self.state.enabled = False
            self.state.accepting_motion_events = False
            self.state.stop_event.set()
            self.state.active_incident_event_id = None
            self._transition_locked(CameraLifecyclePhase.STOPPING)

        # Release scarce camera I/O before waiting for inference workers.
        attempt("capture", self.capture.request_stop)
        attempt("ONVIF", self.onvif.stop)
        tracking_stopped = attempt("object tracking", self.tracking.stop)
        if tracking_stopped is False:
            shutdown_errors.append((
                "object tracking",
                RuntimeError("object tracking session did not stop"),
            ))
        attempt("motion analysis", self.motion_analysis.request_stop)
        attempt("motion events", self.motion_events.signal_stop)

        motion_thread = self.motion_events.thread
        motion_thread_alive: object = False
        if motion_thread is not None:
            attempt(
                "motion event worker join",
                lambda: motion_thread.join(timeout=MOTION_THREAD_STOP_TIMEOUT_SECONDS),
            )
            motion_thread_alive = attempt(
                "motion event worker status",
                motion_thread.is_alive,
            )
            if motion_thread_alive is not False:
                LOGGER.error("motion worker did not stop for %s", self.camera_id)
        self.motion_events.thread = (
            motion_thread if motion_thread_alive is not False else None
        )

        analysis_stopped = attempt(
            "motion analysis wait",
            lambda: self.motion_analysis.wait_stopped(CAPTURE_STOP_TIMEOUT_SECONDS),
        ) is True
        if not analysis_stopped:
            LOGGER.error("motion analysis worker did not stop for %s", self.camera_id)
        alive_threads = attempt(
            "capture wait",
            lambda: self.capture.wait_stopped(CAPTURE_STOP_TIMEOUT_SECONDS),
        )
        if not isinstance(alive_threads, dict):
            alive_threads = {}
        alive = sorted(alive_threads)
        self._log_alive_capture_threads(alive_threads)

        attempt("tracking frame history", self.tracking_frames.clear)
        motion_workers_stopped = (
            self.motion_events.thread is None and analysis_stopped
        )
        if motion_workers_stopped:
            attempt("motion analysis reset", self.motion_analysis.reset)
            attempt("motion evidence reset", self.motion_evidence.clear)
            attempt(
                "motion qualification reset",
                self.motion_qualification.reset_runtime,
            )
            attempt("motion event state reset", self.motion_events.reset)
        else:
            LOGGER.error(
                "preserving motion runtime for %s because a motion worker is still active",
                self.camera_id,
            )

        shutdown_failures: list[str] = []
        if alive:
            shutdown_failures.append(f"capture sources: {', '.join(alive)}")
        if not motion_workers_stopped:
            shutdown_failures.append("motion workers")
        if attempt("ONVIF status", lambda: self.onvif.running) is not False:
            shutdown_failures.append("ONVIF worker")
        if attempt("object tracking status", self.tracking.running) is not False:
            shutdown_failures.append("object tracking worker")
        shutdown_failures.extend(
            f"{label} cleanup"
            for label, _error in shutdown_errors
            if f"{label} cleanup" not in shutdown_failures
        )
        if shutdown_failures:
            message = (
                f"camera {self.camera_id} did not stop cleanly "
                f"({'; '.join(shutdown_failures)})"
            )
            with self.state.lock:
                self._transition_locked(
                    CameraLifecyclePhase.FAILED,
                    failure=message,
                )
            stop_error = RuntimeError(message)
            if shutdown_errors:
                first_error = shutdown_errors[0][1]
                if not isinstance(first_error, Exception):
                    raise first_error
                raise stop_error from None
            raise stop_error
        with self.state.lock:
            self._transition_locked(final_phase, failure=failure)

    def stop_onvif_events(self) -> None:
        with self._operation_lock:
            self.onvif.stop()

    def close(self) -> None:
        with self._operation_lock:
            with self.state.lock:
                if self.state.phase is CameraLifecyclePhase.CLOSED:
                    return
                if self.state.phase in {
                    CameraLifecyclePhase.STARTING,
                    CameraLifecyclePhase.RUNNING,
                    CameraLifecyclePhase.STOPPING,
                }:
                    raise RuntimeError(
                        f"cannot close camera {self.camera_id} while lifecycle "
                        f"phase is {self.state.phase.value}"
                    )
            residual = self._residual_workers()
            cleanup_error: BaseException | None = None
            if residual:
                try:
                    self._stop_runtime()
                except BaseException as error:
                    cleanup_error = error
                residual = self._residual_workers()
            if cleanup_error is not None and not isinstance(cleanup_error, Exception):
                raise cleanup_error
            if residual:
                message = (
                    f"cannot close camera {self.camera_id} while owned workers "
                    f"remain: {', '.join(residual)}"
                )
                with self.state.lock:
                    self._transition_locked(
                        CameraLifecyclePhase.FAILED,
                        failure=message,
                    )
                close_error = RuntimeError(message)
                if cleanup_error is not None:
                    if not isinstance(cleanup_error, Exception):
                        raise cleanup_error
                    raise close_error from None
                raise close_error
            failures: list[BaseException] = []
            for label, pipeline in self.motion_pipelines:
                try:
                    pipeline.close()
                except BaseException as error:
                    failures.append(error)
                    LOGGER.error(
                        "%s motion pipeline cleanup failed for %s: %s",
                        label,
                        self.camera_id,
                        redact_secret_text(error),
                    )
            try:
                self.capture.close()
            except BaseException as error:
                failures.append(error)
                LOGGER.error(
                    "capture cleanup failed for %s: %s",
                    self.camera_id,
                    redact_secret_text(error),
                )
            if failures:
                first_error = failures[0]
                with self.state.lock:
                    self._transition_locked(
                        CameraLifecyclePhase.FAILED,
                        failure=f"{type(first_error).__name__}: {first_error}",
                    )
                if not isinstance(first_error, Exception):
                    raise first_error
                raise RuntimeError(
                    f"one or more camera resources failed to close for "
                    f"{self.camera_id}"
                ) from None
            with self.state.lock:
                self.state.enabled = False
                self.state.accepting_motion_events = False
                self.state.stop_event.set()
                self._transition_locked(CameraLifecyclePhase.CLOSED)

    def set_detection_enabled(self, enabled: bool) -> None:
        with self._operation_lock:
            with self.state.lock:
                previous = self.state.detection_enabled
                self.state.detection_enabled = bool(enabled)
            try:
                self.tracking.sync_accepting()
            except BaseException:
                with self.state.lock:
                    self.state.detection_enabled = previous
                try:
                    self.tracking.sync_accepting()
                except BaseException:
                    LOGGER.exception(
                        "object tracking eligibility rollback failed for %s",
                        self.camera_id,
                    )
                raise

    def runtime_status(self) -> dict[str, Any]:
        """Return a non-blocking lifecycle baseline for telemetry."""
        with self.state.lock:
            enabled = self.state.enabled
            detection_enabled = self.state.detection_enabled
            accepting_motion_events = self.state.accepting_motion_events
            phase = self.state.phase
            generation = self.state.generation
            transition_count = self.state.transition_count
            last_transition_at = self.state.last_transition_at
            phase_age_seconds = max(
                0.0,
                time.monotonic() - self.state.phase_started_at_monotonic,
            )
            last_phase_duration_ms = self.state.last_phase_duration_ms
            last_failure = self.state.last_failure
        workers = self._residual_workers()
        return {
            "phase": phase.value,
            "enabled": enabled,
            "detection_enabled": detection_enabled,
            "accepting_motion_events": accepting_motion_events,
            "generation": generation,
            "transition_count": transition_count,
            "last_transition_at": last_transition_at,
            "phase_age_seconds": round(phase_age_seconds, 3),
            "last_phase_duration_ms": round(last_phase_duration_ms, 3),
            "last_failure": last_failure,
            "active_workers": workers,
            "active_worker_count": len(workers),
        }

    def _transition_locked(
        self,
        phase: CameraLifecyclePhase,
        *,
        failure: str = "",
    ) -> None:
        """Record a phase transition while the short-lived state lock is held."""
        transitioned_at_monotonic = time.monotonic()
        self.state.last_phase_duration_ms = max(
            0.0,
            (
                transitioned_at_monotonic
                - self.state.phase_started_at_monotonic
            )
            * 1000.0,
        )
        self.state.phase_started_at_monotonic = transitioned_at_monotonic
        self.state.phase = phase
        self.state.transition_count += 1
        self.state.last_transition_at = datetime.now(timezone.utc).isoformat()
        if failure:
            self.state.last_failure = redact_secret_text(failure)[:1000]
        elif phase in {
            CameraLifecyclePhase.STARTING,
            CameraLifecyclePhase.RUNNING,
            CameraLifecyclePhase.STOPPED,
            CameraLifecyclePhase.CLOSED,
        }:
            self.state.last_failure = ""

    def _residual_workers(self) -> list[str]:
        return [
            label
            for label, running in (
                (
                    "motion events",
                    self.motion_events.thread is not None
                    and self.motion_events.thread.is_alive(),
                ),
                (
                    "motion analysis",
                    self.motion_analysis.thread is not None
                    and self.motion_analysis.thread.is_alive(),
                ),
                (
                    "capture",
                    any(
                        thread.is_alive()
                        for thread in self.capture.threads().values()
                    ),
                ),
                ("ONVIF", self.onvif.running),
                ("object tracking", self.tracking.running()),
            )
            if running
        ]

    def _log_alive_capture_threads(
        self,
        alive_threads: dict[str, threading.Thread],
    ) -> None:
        if not alive_threads:
            return
        runtime_logger = logging.getLogger("uvicorn.error")
        runtime_logger.error(
            "camera capture threads did not stop for %s: %s",
            self.camera_id,
            ", ".join(sorted(alive_threads)),
        )
        current_frames = sys._current_frames()
        for source, thread in alive_threads.items():
            if thread.ident is None:
                continue
            frame = current_frames.get(thread.ident)
            if frame is not None:
                runtime_logger.error(
                    "camera capture thread stack for %s/%s:\n%s",
                    self.camera_id,
                    source,
                    "".join(traceback.format_stack(frame)),
                )

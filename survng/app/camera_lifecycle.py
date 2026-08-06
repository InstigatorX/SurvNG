"""Camera runtime state and coordinated worker lifecycle."""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from .camera_capture import CameraCaptureService
from .motion_analysis_service import MotionAnalysisService
from .motion_events import MotionEventCoordinator
from .motion_pipeline import MotionEvidenceRepository, MotionPipeline
from .motion_qualification_service import MotionQualificationService
from .object_tracking_lifecycle import ObjectTrackingLifecycle
from .onvif_events import OnvifEventListener
from .tracking_frames import TrackingFrameService

LOGGER = logging.getLogger(__name__)
MOTION_THREAD_STOP_TIMEOUT_SECONDS = 22.0
# OpenCV's FFmpeg calls cannot be interrupted safely from another thread.
# Keep their own deadlines below stop()'s join budget so a stop request can
# always regain control after a blocked open/read operation.
CAPTURE_STOP_TIMEOUT_SECONDS = 8.0


@dataclass(slots=True)
class CameraRuntimeState:
    """Mutable camera lifecycle state shared through one explicit owner."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    enabled: bool = False
    detection_enabled: bool = True
    accepting_motion_events: bool = True
    active_incident_event_id: int | None = None

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

    def start(self) -> None:
        with self.state.lock:
            if self.state.enabled:
                return
            residual_workers = self._residual_workers()
            if residual_workers:
                raise RuntimeError(
                    f"cannot start camera {self.camera_id} while stale workers remain: "
                    f"{', '.join(residual_workers)}"
                )
            self.state.enabled = True
            self.state.accepting_motion_events = True
            self.state.stop_event.clear()
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
            except BaseException:
                try:
                    self.stop()
                except BaseException:
                    LOGGER.exception(
                        "camera startup rollback was incomplete for %s",
                        self.camera_id,
                    )
                raise

    def stop(self) -> None:
        with self.state.lock:
            shutdown_errors: list[tuple[str, BaseException]] = []

            def attempt(label: str, operation: Callable[[], Any]) -> Any:
                try:
                    return operation()
                except BaseException as error:
                    shutdown_errors.append((label, error))
                    LOGGER.exception("%s stop failed for %s", label, self.camera_id)
                    return None

            self.state.enabled = False
            self.state.accepting_motion_events = False
            self.state.stop_event.set()
            self.state.active_incident_event_id = None
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
                    lambda: motion_thread.join(
                        timeout=MOTION_THREAD_STOP_TIMEOUT_SECONDS
                    ),
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
                lambda: self.motion_analysis.wait_stopped(
                    CAPTURE_STOP_TIMEOUT_SECONDS
                ),
            ) is True
            if not analysis_stopped:
                LOGGER.error(
                    "motion analysis worker did not stop for %s", self.camera_id
                )
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
                    "preserving motion runtime for %s because a motion worker "
                    "is still active",
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
                failure = RuntimeError(
                    f"camera {self.camera_id} did not stop cleanly "
                    f"({'; '.join(shutdown_failures)})"
                )
                if shutdown_errors:
                    first_error = shutdown_errors[0][1]
                    if not isinstance(first_error, Exception):
                        raise first_error
                    raise failure from first_error
                raise failure

    def stop_onvif_events(self) -> None:
        with self.state.lock:
            self.onvif.stop()

    def close(self) -> None:
        with self.state.lock:
            active = [
                label
                for label, thread in (
                    ("motion events", self.motion_events.thread),
                    ("motion analysis", self.motion_analysis.thread),
                )
                if thread is not None and thread.is_alive()
            ]
            if active:
                raise RuntimeError(
                    f"cannot close camera {self.camera_id} pipelines while "
                    f"{', '.join(active)} is running"
                )
            failures: list[BaseException] = []
            for label, pipeline in self.motion_pipelines:
                try:
                    pipeline.close()
                except BaseException as error:
                    failures.append(error)
                    LOGGER.exception(
                        "%s motion pipeline cleanup failed for %s",
                        label,
                        self.camera_id,
                    )
            try:
                self.capture.close()
            except BaseException as error:
                failures.append(error)
                LOGGER.exception("capture cleanup failed for %s", self.camera_id)
            if failures:
                first_error = failures[0]
                if not isinstance(first_error, Exception):
                    raise first_error
                raise RuntimeError(
                    f"one or more camera resources failed to close for "
                    f"{self.camera_id}"
                ) from first_error

    def set_detection_enabled(self, enabled: bool) -> None:
        with self.state.lock:
            previous = self.state.detection_enabled
            self.state.detection_enabled = bool(enabled)
            try:
                self.tracking.sync_accepting()
            except BaseException:
                self.state.detection_enabled = previous
                try:
                    self.tracking.sync_accepting()
                except BaseException:
                    LOGGER.exception(
                        "object tracking eligibility rollback failed for %s",
                        self.camera_id,
                    )
                raise

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

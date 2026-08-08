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
from .motion_runtime import MotionRuntimeService
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


@dataclass(frozen=True, slots=True)
class CameraStopTicket:
    """Identity of one camera generation's idempotent stop transaction."""

    camera_id: str
    generation: int
    requested_at_monotonic: float


@dataclass(slots=True)
class _PendingCameraShutdown:
    ticket: CameraStopTicket
    errors: list[tuple[str, BaseException]]
    final_phase: CameraLifecyclePhase
    failure: str


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
        motion_runtime: MotionRuntimeService,
        tracking_frames: TrackingFrameService,
    ) -> None:
        self.camera_id = camera_id
        self.state = state
        self.capture = capture
        self.onvif = onvif
        self.tracking = tracking
        self.motion_runtime = motion_runtime
        self.tracking_frames = tracking_frames
        # Serialize lifecycle commands without blocking state readers. This may
        # be held across joins and camera I/O; state.lock may not.
        self._operation_lock = threading.Lock()
        self._pending_shutdown: _PendingCameraShutdown | None = None

    def start(self) -> None:
        with self._operation_lock:
            with self.state.lock:
                if self.state.phase is CameraLifecyclePhase.RUNNING:
                    return
                if self.state.phase is CameraLifecyclePhase.CLOSED:
                    raise RuntimeError(f"camera {self.camera_id} is closed")
                if self.state.phase in {
                    CameraLifecyclePhase.STARTING,
                    CameraLifecyclePhase.STOPPING,
                }:
                    raise RuntimeError(
                        f"cannot start camera {self.camera_id} while lifecycle "
                        f"phase is {self.state.phase.value}"
                    )
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
                detection_enabled = self.state.detection_enabled
                self._transition_locked(CameraLifecyclePhase.STARTING)
            try:
                self.tracking.sync_accepting()
                # Clear previous-run state before any producer can enqueue new work.
                self.motion_runtime.start(self.state.stop_event)
                if not self.capture.start():
                    raise RuntimeError(
                        f"camera source did not start for {self.camera_id}"
                    )
                # ONVIF events only feed object-detection admission. Preserve
                # live capture for a detection-disabled camera without holding
                # a camera subscription that cannot produce useful work.
                if detection_enabled:
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

    def request_stop(self) -> CameraStopTicket | None:
        """Broadcast a nonblocking stop request for every owned component."""
        with self._operation_lock:
            return self._request_stop_runtime()

    def wait_stopped(
        self,
        deadline: float,
        ticket: CameraStopTicket | None = None,
    ) -> bool:
        """Wait for a requested stop using one fleet-owned absolute deadline."""
        with self._operation_lock:
            return self._wait_stop_runtime(deadline, ticket)

    def _stop_runtime(
        self,
        *,
        final_phase: CameraLifecyclePhase = CameraLifecyclePhase.STOPPED,
        failure: str = "",
    ) -> None:
        ticket = self._request_stop_runtime(
            final_phase=final_phase,
            failure=failure,
        )
        if ticket is None:
            return
        deadline = time.monotonic() + max(
            MOTION_THREAD_STOP_TIMEOUT_SECONDS,
            CAPTURE_STOP_TIMEOUT_SECONDS,
        )
        if not self._wait_stop_runtime(deadline, ticket):
            raise RuntimeError(f"camera {self.camera_id} did not stop cleanly")

    def _request_stop_runtime(
        self,
        *,
        final_phase: CameraLifecyclePhase = CameraLifecyclePhase.STOPPED,
        failure: str = "",
    ) -> CameraStopTicket | None:
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
                return None
            if self.state.phase is CameraLifecyclePhase.STOPPING:
                pending = self._pending_shutdown
                if pending is None:
                    raise RuntimeError(
                        f"camera {self.camera_id} is stopping without a stop ticket"
                    )
                return pending.ticket
            ticket = CameraStopTicket(
                camera_id=self.camera_id,
                generation=self.state.generation,
                requested_at_monotonic=time.monotonic(),
            )
            self.state.enabled = False
            self.state.accepting_motion_events = False
            self.state.stop_event.set()
            self.state.active_incident_event_id = None
            self._transition_locked(CameraLifecyclePhase.STOPPING)

        # Broadcast first so all independent workers consume the same later
        # wait budget instead of nesting their individual timeout budgets.
        attempt("capture", self.capture.request_stop)
        attempt("ONVIF", self.onvif.request_stop)
        attempt("object tracking", self.tracking.request_stop)
        attempt("motion runtime", self.motion_runtime.request_stop)
        self._pending_shutdown = _PendingCameraShutdown(
            ticket=ticket,
            errors=shutdown_errors,
            final_phase=final_phase,
            failure=failure,
        )
        return ticket

    def _wait_stop_runtime(
        self,
        deadline: float,
        ticket: CameraStopTicket | None = None,
    ) -> bool:
        with self.state.lock:
            if self.state.phase is CameraLifecyclePhase.CLOSED:
                return True
            pending = self._pending_shutdown
            if pending is None:
                return self.state.phase is CameraLifecyclePhase.STOPPED
            expected = ticket or pending.ticket
            if (
                expected.camera_id != self.camera_id
                or expected != pending.ticket
                or expected.generation != self.state.generation
                or self.state.phase is not CameraLifecyclePhase.STOPPING
            ):
                return False
        shutdown_errors = pending.errors

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

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

        tracking_stopped = attempt(
            "object tracking wait",
            lambda: self.tracking.wait_stopped(remaining()),
        ) is True
        motion_workers_stopped = attempt(
            "motion runtime wait",
            lambda: self.motion_runtime.wait_stopped(
                analysis_timeout=remaining(),
                decision_timeout=remaining(),
            ),
        ) is True
        if not motion_workers_stopped:
            LOGGER.error("motion workers did not stop for %s", self.camera_id)
        alive_threads = attempt(
            "capture wait",
            lambda: self.capture.wait_stopped(remaining()),
        )
        if not isinstance(alive_threads, dict):
            alive_threads = {}
        alive = sorted(alive_threads)
        self._log_alive_capture_threads(alive_threads)

        onvif_stopped = attempt(
            "ONVIF wait",
            lambda: self.onvif.wait_stopped(remaining()),
        ) is True

        attempt("tracking frame history", self.tracking_frames.clear)
        shutdown_failures: list[str] = []
        if alive:
            shutdown_failures.append(f"capture sources: {', '.join(alive)}")
        if not motion_workers_stopped:
            shutdown_failures.append("motion workers")
        if not onvif_stopped or attempt(
            "ONVIF status", lambda: self.onvif.running
        ) is not False:
            shutdown_failures.append("ONVIF worker")
        if not tracking_stopped or attempt(
            "object tracking status", self.tracking.running
        ) is not False:
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
            if (
                self._pending_shutdown is not pending
                or self.state.generation != pending.ticket.generation
                or self.state.phase is not CameraLifecyclePhase.STOPPING
            ):
                return False
            self._transition_locked(
                pending.final_phase,
                failure=pending.failure,
            )
            self._pending_shutdown = None
        return True

    def stop_onvif_events(self) -> None:
        with self._operation_lock:
            self.onvif.stop()

    def request_onvif_stop(self) -> None:
        self.onvif.request_stop()

    def wait_onvif_stopped(self, deadline: float) -> bool:
        return self.onvif.wait_stopped(max(0.0, deadline - time.monotonic()))

    def active_workers(self) -> list[str]:
        return self._residual_workers()

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
            try:
                self.motion_runtime.close()
            except BaseException as error:
                failures.append(error)
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
            desired = bool(enabled)
            with self.state.lock:
                previous = self.state.detection_enabled
                phase = self.state.phase
                if previous is desired:
                    return
                self.state.detection_enabled = desired
            try:
                self.tracking.sync_accepting()
                if phase is CameraLifecyclePhase.RUNNING:
                    if desired:
                        self.onvif.start()
                    else:
                        self.onvif.stop()
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
                if phase is CameraLifecyclePhase.RUNNING:
                    try:
                        if previous:
                            self.onvif.start()
                        else:
                            self.onvif.stop()
                    except BaseException:
                        LOGGER.exception(
                            "ONVIF eligibility rollback failed for %s",
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
        allowed = {
            CameraLifecyclePhase.STOPPED: {
                CameraLifecyclePhase.STARTING,
                CameraLifecyclePhase.STOPPING,
                CameraLifecyclePhase.FAILED,
                CameraLifecyclePhase.CLOSED,
            },
            CameraLifecyclePhase.STARTING: {
                CameraLifecyclePhase.RUNNING,
                CameraLifecyclePhase.STOPPING,
                CameraLifecyclePhase.FAILED,
            },
            CameraLifecyclePhase.RUNNING: {
                CameraLifecyclePhase.STOPPING,
                CameraLifecyclePhase.FAILED,
            },
            CameraLifecyclePhase.STOPPING: {
                CameraLifecyclePhase.STOPPED,
                CameraLifecyclePhase.FAILED,
            },
            CameraLifecyclePhase.FAILED: {
                CameraLifecyclePhase.STARTING,
                CameraLifecyclePhase.STOPPING,
                CameraLifecyclePhase.CLOSED,
            },
            CameraLifecyclePhase.CLOSED: set(),
        }
        if phase is self.state.phase:
            if failure:
                self.state.last_failure = redact_secret_text(failure)[:1000]
            return
        if phase not in allowed[self.state.phase]:
            raise RuntimeError(
                f"invalid camera lifecycle transition for {self.camera_id}: "
                f"{self.state.phase.value} -> {phase.value}"
            )
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
        workers = list(self.motion_runtime.active_workers())
        workers.extend(
            label
            for label, running in (
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
        )
        return workers

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

"""Lifecycle ownership for the process-wide camera fleet."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .camera_startup import CameraStartupCoordinator, CameraStartupTask
from .config import CameraConfig
from .object_activity import AttributionMode
from .security import redact_secret_text

LOGGER = logging.getLogger("uvicorn.error")
ONVIF_RELEASE_TIMEOUT_SECONDS = 15.0
CAMERA_SHUTDOWN_TIMEOUT_SECONDS = 30.0


class CameraFleetWorker(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def request_stop(self) -> Any | None: ...
    def wait_stopped(self, deadline: float, ticket: Any | None = None) -> bool: ...
    def active_workers(self) -> list[str]: ...
    def close(self) -> None: ...
    def stop_onvif_events(self) -> None: ...
    def request_onvif_stop(self) -> None: ...
    def wait_onvif_stopped(self, deadline: float) -> bool: ...
    def live_capture_ready(self) -> bool: ...
    def set_detection_enabled(self, enabled: bool) -> None: ...
    def reconfigure_object_activity_attribution(self, mode: AttributionMode) -> None: ...


class CameraFleetRecorder(Protocol):
    def set_camera_enabled(self, camera_id: str, enabled: bool) -> None: ...
    def start(self, camera: CameraConfig, source: str) -> None: ...


class CameraFleetStatePublisher(Protocol):
    def publish_camera_state(self, camera_id: str, enabled: bool) -> None: ...


@dataclass(frozen=True, slots=True)
class CameraFleetFailure:
    label: str
    error: Exception


class CameraFleetOperationError(RuntimeError):
    def __init__(
        self,
        action: str,
        failures: Sequence[CameraFleetFailure],
        *,
        residual_camera_ids: Sequence[str] = (),
    ) -> None:
        self.action = action
        self.failures = tuple(failures)
        self.residual_camera_ids = tuple(sorted(set(residual_camera_ids)))
        labels = ", ".join(failure.label for failure in self.failures)
        super().__init__(f"camera fleet {action} failed: {labels}")


class CameraFleetLifecycle:
    """Own camera admission, early ONVIF release, and bounded fleet teardown."""

    def __init__(
        self,
        *,
        cameras: Sequence[CameraConfig],
        workers: Mapping[str, CameraFleetWorker],
        recorder: CameraFleetRecorder,
        startup: CameraStartupCoordinator,
        state_publisher: CameraFleetStatePublisher,
    ) -> None:
        self.cameras = tuple(cameras)
        self.workers = dict(workers)
        camera_ids = [camera.id for camera in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera fleet contains duplicate camera IDs")
        if set(camera_ids) != set(self.workers):
            missing = sorted(set(camera_ids) - set(self.workers))
            unexpected = sorted(set(self.workers) - set(camera_ids))
            raise ValueError(
                "camera fleet worker mismatch "
                f"(missing={missing}, unexpected={unexpected})"
            )
        self.recorder = recorder
        self.startup = startup
        self._state_publisher = state_publisher
        self._stopping = threading.Event()
        self._cameras_by_id = {camera.id: camera for camera in self.cameras}
        self._state_lock = threading.Lock()
        self._camera_enabled = {camera.id: True for camera in self.cameras}
        self._residual_lock = threading.Lock()
        self._shutdown_residuals: set[str] = set()
        self._onvif_residuals: set[str] = set()
        self._closed_workers: set[str] = set()

    def recorder_keys(self) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        for camera in self.cameras:
            if camera.record:
                keys.add((camera.id, "main"))
            if camera.record_sub and camera.live_stream_url:
                keys.add((camera.id, "live"))
        return keys

    def camera(self, camera_id: str) -> CameraConfig | None:
        return self._cameras_by_id.get(camera_id)

    def set_camera_enabled(self, camera_id: str, enabled: bool) -> bool:
        with self._state_lock:
            if camera_id not in self._camera_enabled:
                return False
            self._camera_enabled[camera_id] = bool(enabled)
            return True

    def camera_enabled(self, camera_id: str) -> bool:
        with self._state_lock:
            return self._camera_enabled.get(camera_id, False)

    def start_camera(self, camera_id: str) -> bool:
        worker = self.workers.get(camera_id)
        if worker is None or self._stopping.is_set():
            return False
        worker.start()
        return True

    def stop_camera(self, camera_id: str) -> bool:
        worker = self.workers.get(camera_id)
        if worker is None or self._stopping.is_set():
            return False
        worker.stop()
        return True

    def prepare_startup(
        self,
        *,
        camera_enabled: Mapping[str, bool],
        recording_enabled: Mapping[str, bool],
        detection_enabled: Mapping[str, bool],
        recording_is_enabled: Callable[[str], bool] | None = None,
    ) -> tuple[CameraStartupTask, ...]:
        """Apply immutable preferences and build one admission generation."""
        self._stopping.clear()
        camera_preferences = {
            camera.id: bool(camera_enabled.get(camera.id, True))
            for camera in self.cameras
        }
        with self._state_lock:
            self._camera_enabled = dict(camera_preferences)
        recording_preferences = {
            camera.id: bool(recording_enabled.get(camera.id, True))
            for camera in self.cameras
        }
        recording_predicate = recording_is_enabled or (
            lambda camera_id: recording_preferences[camera_id]
        )
        detection_preferences = {
            camera.id: bool(detection_enabled.get(camera.id, True))
            for camera in self.cameras
        }
        tasks: list[CameraStartupTask] = []
        for camera in self.cameras:
            worker = self.workers[camera.id]
            self.recorder.set_camera_enabled(
                camera.id,
                camera_preferences[camera.id] and recording_preferences[camera.id],
            )
            worker.set_detection_enabled(detection_preferences[camera.id])
            tasks.append(CameraStartupTask(
                camera_id=camera.id,
                is_enabled=lambda camera_id=camera.id: self._admission_enabled(
                    camera_id
                ),
                start_camera=worker.start,
                capture_ready=worker.live_capture_ready,
                start_recorders=lambda camera=camera: self._start_recorders(
                    camera,
                    recording_predicate,
                ),
                publish_state=lambda camera_id=camera.id: (
                    self._publish_current_state(camera_id)
                ),
            ))
        return tuple(tasks)

    def _publish_state(self, camera_id: str, enabled: bool) -> None:
        self._state_publisher.publish_camera_state(camera_id, enabled)

    def _publish_current_state(self, camera_id: str) -> None:
        self._publish_state(camera_id, self.camera_enabled(camera_id))

    def _admission_enabled(self, camera_id: str) -> bool:
        return not self._stopping.is_set() and self.camera_enabled(camera_id)

    def start_admission(
        self,
        tasks: Sequence[CameraStartupTask],
        *,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        self.startup.start(tasks, on_complete=on_complete)

    def wait(self, timeout: float | None = None) -> bool:
        return self.startup.wait(timeout)

    def status(self) -> dict[str, object]:
        with self._residual_lock:
            lifecycle = {
                "shutdown_residual_camera_ids": sorted(self._shutdown_residuals),
                "onvif_residual_camera_ids": sorted(self._onvif_residuals),
            }
        return {**self.startup.status(), **lifecycle}

    def cancel_admission(self) -> None:
        self._stopping.set()
        if not self.startup.cancel():
            raise RuntimeError("camera startup coordinator did not stop")

    def release_onvif(self, *, timeout: float | None = None) -> None:
        failures: list[CameraFleetFailure] = []
        for camera_id, worker in self.workers.items():
            try:
                worker.request_onvif_stop()
            except Exception as error:
                failures.append(CameraFleetFailure(camera_id, error))
        deadline = time.monotonic() + max(
            0.0,
            ONVIF_RELEASE_TIMEOUT_SECONDS if timeout is None else timeout,
        )
        active: set[str] = set()
        for camera_id, worker in self.workers.items():
            try:
                if not worker.wait_onvif_stopped(deadline):
                    active.add(camera_id)
            except Exception as error:
                failures.append(CameraFleetFailure(camera_id, error))
                active.add(camera_id)
        with self._residual_lock:
            self._onvif_residuals = set(active)
        failures.extend(CameraFleetFailure(
            camera_id,
            RuntimeError("ONVIF release timed out"),
        ) for camera_id in active)
        if failures:
            raise CameraFleetOperationError(
                "ONVIF release",
                failures,
                residual_camera_ids=active,
            )

    def quiesce_onvif(self, *, timeout: float | None = None) -> None:
        """Stop new admission before releasing every existing subscription."""
        try:
            self.cancel_admission()
        except Exception as error:
            # Admission may still be inside CameraLifecycle.start() while
            # holding its operation lock. Do not attempt ONVIF or camera stop
            # work until the coordinator and all of its workers have joined.
            residuals = set(self.workers)
            with self._residual_lock:
                self._shutdown_residuals = set(residuals)
            raise CameraFleetOperationError(
                "startup cancellation",
                [CameraFleetFailure("startup admission", error)],
                residual_camera_ids=residuals,
            ) from None
        try:
            self.release_onvif(timeout=timeout)
        except CameraFleetOperationError as error:
            raise CameraFleetOperationError(
                "ONVIF quiescence",
                error.failures,
                residual_camera_ids=error.residual_camera_ids,
            ) from None

    def stop_workers(
        self,
        *,
        timeout: float | None = None,
    ) -> None:
        self._stopping.set()
        failures: list[CameraFleetFailure] = []
        stop_tickets: dict[str, Any] = {}
        for camera_id, worker in self.workers.items():
            try:
                ticket = worker.request_stop()
                if ticket is not None:
                    stop_tickets[camera_id] = ticket
            except Exception as error:
                failures.append(CameraFleetFailure(camera_id, error))
                LOGGER.error(
                    "camera fleet shutdown request failed for %s: %s",
                    camera_id,
                    redact_secret_text(error),
                )
        deadline = time.monotonic() + max(
            0.0,
            CAMERA_SHUTDOWN_TIMEOUT_SECONDS if timeout is None else timeout,
        )
        active: set[str] = set()
        for camera_id, worker in self.workers.items():
            try:
                if not worker.wait_stopped(
                    deadline,
                    stop_tickets.get(camera_id),
                ):
                    active.add(camera_id)
            except Exception as error:
                failures.append(CameraFleetFailure(camera_id, error))
                try:
                    has_residuals = bool(worker.active_workers())
                except Exception as status_error:
                    failures.append(CameraFleetFailure(camera_id, status_error))
                    has_residuals = True
                if has_residuals:
                    active.add(camera_id)
        with self._residual_lock:
            self._shutdown_residuals = set(active)
        failures.extend(CameraFleetFailure(
            camera_id,
            RuntimeError("camera shutdown timed out"),
        ) for camera_id in active)
        if active:
            raise CameraFleetOperationError(
                "shutdown",
                failures,
                residual_camera_ids=active,
            )
        for camera_id, worker in self.workers.items():
            if camera_id in self._closed_workers:
                continue
            try:
                worker.close()
                self._closed_workers.add(camera_id)
            except Exception as error:
                failures.append(CameraFleetFailure(camera_id, error))
                LOGGER.exception("camera resource cleanup failed for %s", camera_id)
        if failures:
            raise CameraFleetOperationError("shutdown", failures)

    def _start_recorders(
        self,
        camera: CameraConfig,
        recording_is_enabled: Callable[[str], bool],
    ) -> None:
        if (
            self._stopping.is_set()
            or not self.camera_enabled(camera.id)
            or not recording_is_enabled(camera.id)
        ):
            return
        if camera.record:
            self.recorder.start(camera, "main")
        if (
            not self._stopping.is_set()
            and camera.record_sub
            and camera.live_stream_url
        ):
            self.recorder.start(camera, "live")

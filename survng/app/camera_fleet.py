"""Lifecycle ownership for the process-wide camera fleet."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .camera_startup import CameraStartupCoordinator, CameraStartupTask
from .config import CameraConfig
from .security import redact_secret_text

LOGGER = logging.getLogger("uvicorn.error")
ONVIF_RELEASE_TIMEOUT_SECONDS = 15.0
CAMERA_SHUTDOWN_TIMEOUT_SECONDS = 30.0


class CameraFleetWorker(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
    def stop_onvif_events(self) -> None: ...
    def live_capture_ready(self) -> bool: ...
    def set_detection_enabled(self, enabled: bool) -> None: ...


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
    def __init__(self, action: str, failures: Sequence[CameraFleetFailure]) -> None:
        self.action = action
        self.failures = tuple(failures)
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
        self._publisher_lock = threading.Lock()
        self._state_publisher = state_publisher
        self._stopping = threading.Event()
        self._cameras_by_id = {camera.id: camera for camera in self.cameras}
        self._state_lock = threading.Lock()
        self._camera_enabled = {camera.id: True for camera in self.cameras}

    def replace_state_publisher(self, publisher: CameraFleetStatePublisher) -> None:
        with self._publisher_lock:
            self._state_publisher = publisher

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
                    recording_preferences,
                ),
                publish_state=lambda camera_id=camera.id: (
                    self._publish_current_state(camera_id)
                ),
            ))
        return tuple(tasks)

    def _publish_state(self, camera_id: str, enabled: bool) -> None:
        with self._publisher_lock:
            publisher = self._state_publisher
        publisher.publish_camera_state(camera_id, enabled)

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
        return self.startup.status()

    def cancel_admission(self) -> None:
        self._stopping.set()
        if not self.startup.cancel():
            raise RuntimeError("camera startup coordinator did not stop")

    def release_onvif(self, *, timeout: float | None = None) -> None:
        failures, active = self._parallel(
            action="ONVIF release",
            operation=lambda worker: worker.stop_onvif_events(),
            thread_prefix="release-onvif",
            timeout=(
                ONVIF_RELEASE_TIMEOUT_SECONDS if timeout is None else timeout
            ),
        )
        failures.extend(
            CameraFleetFailure(camera_id, RuntimeError("ONVIF release timed out"))
            for camera_id in active
        )
        if failures:
            raise CameraFleetOperationError("ONVIF release", failures)

    def quiesce_onvif(self, *, timeout: float | None = None) -> None:
        """Stop new admission before releasing every existing subscription."""
        failures: list[CameraFleetFailure] = []
        try:
            self.cancel_admission()
        except Exception as error:
            failures.append(CameraFleetFailure("startup admission", error))
        try:
            self.release_onvif(timeout=timeout)
        except CameraFleetOperationError as error:
            failures.extend(error.failures)
        if failures:
            raise CameraFleetOperationError("ONVIF quiescence", failures)

    def stop_workers(
        self,
        *,
        timeout: float | None = None,
    ) -> None:
        self._stopping.set()
        failures, active = self._parallel(
            action="shutdown",
            operation=lambda worker: worker.stop(),
            thread_prefix="stop-camera",
            timeout=(CAMERA_SHUTDOWN_TIMEOUT_SECONDS if timeout is None else timeout),
        )
        failures.extend(
            CameraFleetFailure(camera_id, RuntimeError("camera shutdown timed out"))
            for camera_id in active
        )
        for camera_id, worker in self.workers.items():
            if camera_id in active:
                continue
            try:
                worker.close()
            except Exception as error:
                failures.append(CameraFleetFailure(camera_id, error))
                LOGGER.exception("camera resource cleanup failed for %s", camera_id)
        if failures:
            raise CameraFleetOperationError("shutdown", failures)

    def _start_recorders(
        self,
        camera: CameraConfig,
        recording_enabled: Mapping[str, bool],
    ) -> None:
        if (
            self._stopping.is_set()
            or not self.camera_enabled(camera.id)
            or not recording_enabled[camera.id]
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

    def _parallel(
        self,
        *,
        action: str,
        operation: Callable[[CameraFleetWorker], None],
        thread_prefix: str,
        timeout: float,
    ) -> tuple[list[CameraFleetFailure], set[str]]:
        failures: list[CameraFleetFailure] = []
        failures_lock = threading.Lock()

        def invoke(camera_id: str, worker: CameraFleetWorker) -> None:
            try:
                operation(worker)
            except Exception as error:
                with failures_lock:
                    failures.append(CameraFleetFailure(camera_id, error))
                LOGGER.error(
                    "camera fleet %s failed for %s: %s",
                    action,
                    camera_id,
                    redact_secret_text(error),
                )

        threads = [
            (camera_id, threading.Thread(
                target=invoke,
                args=(camera_id, worker),
                name=f"{thread_prefix}-{camera_id}",
                daemon=True,
            ))
            for camera_id, worker in self.workers.items()
        ]
        for _camera_id, thread in threads:
            thread.start()
        deadline = time.monotonic() + max(0.0, timeout)
        for _camera_id, thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        active = {
            camera_id for camera_id, thread in threads if thread.is_alive()
        }
        with failures_lock:
            return list(failures), active

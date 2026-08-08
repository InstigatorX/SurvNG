"""Transactional runtime control for camera, recording, and detection state."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from .camera_fleet import CameraFleetLifecycle
from .config import AppConfig, CameraConfig
from .mqtt_lifecycle import MqttLifecycle
from .recording_lifecycle import RecordingLifecycle
from .runtime_monitor import ApplicationRuntimeMonitor


LOGGER = logging.getLogger("uvicorn.error")


class DetectionControlWorker(Protocol):
    def set_detection_enabled(self, enabled: bool) -> None: ...


class CameraControlService:
    """Own desired runtime state and apply camera-control transactions.

    Preferences are restored from the last atomically persisted control state.
    Missing, malformed, and removed-camera values safely fall back to on.
    Configuration-generation reloads can still transfer an active snapshot.
    """

    def __init__(
        self,
        *,
        cameras: Sequence[CameraConfig],
        workers: Mapping[str, DetectionControlWorker],
        recording: RecordingLifecycle,
        fleet: CameraFleetLifecycle,
        mqtt: MqttLifecycle,
        runtime_monitor: ApplicationRuntimeMonitor,
        state_path: Path,
        legacy_state_paths: Sequence[Path] = (),
    ) -> None:
        self._cameras = {camera.id: camera for camera in cameras}
        self._workers = dict(workers)
        if set(self._cameras) != set(self._workers):
            raise ValueError("camera control workers do not match configured cameras")
        self._recording = recording
        self._fleet = fleet
        self._mqtt = mqtt
        self._runtime_monitor = runtime_monitor
        self._state_path = state_path
        self._legacy_state_paths = tuple(
            path for path in legacy_state_paths if path != state_path
        )
        self._lock = threading.RLock()
        self._recording_enabled: dict[str, bool] = {}
        self._detection_enabled: dict[str, bool] = {}
        self._camera_enabled = {camera_id: True for camera_id in self._cameras}
        self._accepting_commands = True
        self._load_persisted_state()

    def _load_persisted_state(self) -> None:
        state_path: Path | None = None
        for candidate in (self._state_path, *self._legacy_state_paths):
            try:
                candidate.stat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                LOGGER.warning(
                    "camera control state could not be inspected at %s: %s; using defaults",
                    candidate,
                    exc,
                )
                return
            state_path = candidate
            break
        if state_path is None:
            return
        try:
            if state_path.stat().st_size > 1024 * 1024:
                raise ValueError("camera control state exceeds 1 MiB")
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("camera control state must be an object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "camera control state could not be restored from %s: %s; using defaults",
                state_path,
                exc,
            )
            return

        if state_path != self._state_path:
            LOGGER.info(
                "restored legacy camera control state from %s; next write migrates it to %s",
                state_path,
                self._state_path,
            )

        camera_ids = set(self._cameras)

        def boolean_map(name: str) -> dict[str, bool]:
            values = payload.get(name)
            if not isinstance(values, dict):
                return {}
            return {
                camera_id: enabled
                for camera_id, enabled in values.items()
                if camera_id in camera_ids and isinstance(enabled, bool)
            }

        self._recording_enabled = boolean_map("recording_enabled")
        self._detection_enabled = boolean_map("detection_enabled")
        saved_cameras = boolean_map("camera_enabled")
        self._camera_enabled = {
            camera_id: saved_cameras.get(camera_id, True)
            for camera_id in camera_ids
        }

    def quiesce(self) -> None:
        """Reject new commands after any in-flight control transaction drains."""
        with self._lock:
            self._accepting_commands = False

    def recording_enabled(self, camera_id: str) -> bool:
        with self._lock:
            return self._recording_enabled.get(camera_id, True)

    def detection_enabled(self, camera_id: str) -> bool:
        with self._lock:
            return self._detection_enabled.get(camera_id, True)

    def camera_enabled(self, camera_id: str) -> bool:
        with self._lock:
            return self._camera_enabled.get(camera_id, False)

    def snapshot(self) -> dict[str, dict[str, bool]]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, dict[str, bool]]:
        return {
            "recording_enabled": dict(self._recording_enabled),
            "detection_enabled": dict(self._detection_enabled),
            "camera_enabled": dict(self._camera_enabled),
        }

    def apply(
        self,
        preferences: Mapping[str, Mapping[str, bool]],
        *,
        persist: bool = False,
    ) -> None:
        with self._lock:
            previous = self._snapshot_locked()
            camera_ids = set(self._workers)
            self._recording_enabled = {
                camera_id: bool(enabled)
                for camera_id, enabled in preferences.get("recording_enabled", {}).items()
                if camera_id in camera_ids
            }
            self._detection_enabled = {
                camera_id: bool(enabled)
                for camera_id, enabled in preferences.get("detection_enabled", {}).items()
                if camera_id in camera_ids
            }
            self._camera_enabled = {
                camera_id: bool(preferences.get("camera_enabled", {}).get(camera_id, True))
                for camera_id in camera_ids
            }
            try:
                if persist:
                    self._persist_locked()
            except BaseException:
                self._restore_locked(previous)
                raise

    def persist(self) -> None:
        with self._lock:
            self._persist_locked()

    def startup_preferences(self) -> dict[str, dict[str, bool]]:
        """Return one coherent preference snapshot for camera admission."""
        return self.snapshot()

    def set_recording(self, camera_id: str, enabled: bool) -> bool:
        with self._lock:
            if not self._command_allowed(camera_id):
                return False
            previous = self._recording_enabled.get(camera_id, True)
            self._update_preference_locked("recording_enabled", camera_id, enabled)
            should_run = self._camera_enabled[camera_id] and bool(enabled)
            try:
                self._recording.set_camera_enabled(camera_id, should_run)
                if should_run:
                    self._recording.start_camera(self._cameras[camera_id])
            except BaseException:
                self._restore_preference_locked(
                    "recording_enabled",
                    camera_id,
                    previous,
                )
                previous_should_run = self._camera_enabled[camera_id] and previous
                try:
                    self._recording.set_camera_enabled(
                        camera_id,
                        previous_should_run,
                    )
                    if previous_should_run:
                        self._recording.start_camera(self._cameras[camera_id])
                except Exception:
                    LOGGER.exception(
                        "failed to restore recorder state for %s",
                        camera_id,
                    )
                raise
            self._mqtt.publish_camera_feature_state(camera_id, "recording", bool(enabled))
            self._runtime_monitor.publish_camera_status(camera_id)
            return True

    def set_detection(self, camera_id: str, enabled: bool) -> bool:
        with self._lock:
            if not self._command_allowed(camera_id):
                return False
            previous = self._detection_enabled.get(camera_id, True)
            self._update_preference_locked("detection_enabled", camera_id, enabled)
            try:
                self._workers[camera_id].set_detection_enabled(enabled)
            except BaseException:
                self._restore_preference_locked(
                    "detection_enabled",
                    camera_id,
                    previous,
                )
                raise
            self._mqtt.publish_camera_feature_state(camera_id, "detection", bool(enabled))
            self._runtime_monitor.publish_camera_status(camera_id)
            return True

    def start_camera(self, camera_id: str) -> bool:
        with self._lock:
            if not self._command_allowed(camera_id):
                return False
            previous = self._camera_enabled[camera_id]
            self._update_preference_locked("camera_enabled", camera_id, True)
            try:
                if not self._fleet.set_camera_enabled(camera_id, True):
                    raise RuntimeError(f"camera {camera_id} is not in the fleet")
                recording_enabled = self._recording_enabled.get(camera_id, True)
                self._recording.set_camera_enabled(camera_id, recording_enabled)
                if not self._fleet.start_camera(camera_id):
                    raise RuntimeError(f"camera {camera_id} could not start")
            except Exception:
                self._camera_enabled[camera_id] = previous
                self._fleet.set_camera_enabled(camera_id, previous)
                try:
                    self._persist_locked()
                except Exception:
                    LOGGER.exception("failed to roll back camera power state for %s", camera_id)
                self._recording.set_camera_enabled(
                    camera_id,
                    previous and self._recording_enabled.get(camera_id, True),
                )
                raise
            if self._recording_enabled.get(camera_id, True):
                self._recording.start_camera(self._cameras[camera_id])
            self._mqtt.publish_camera_state(camera_id, True)
            self._runtime_monitor.publish_camera_status(camera_id)
            return True

    def stop_camera(self, camera_id: str) -> bool:
        with self._lock:
            if not self._command_allowed(camera_id):
                return False
            self._update_preference_locked("camera_enabled", camera_id, False)
            try:
                if not self._fleet.set_camera_enabled(camera_id, False):
                    raise RuntimeError(f"camera {camera_id} is not in the fleet")
                self._recording.set_camera_enabled(camera_id, False)
                if not self._fleet.stop_camera(camera_id):
                    raise RuntimeError(f"camera {camera_id} could not stop")
            except Exception:
                # A partially stopped camera cannot truthfully be advertised as
                # enabled. Persist the safe off state and leave recovery explicit.
                self._camera_enabled[camera_id] = False
                self._fleet.set_camera_enabled(camera_id, False)
                self._recording.set_camera_enabled(camera_id, False)
                self._mqtt.publish_camera_state(camera_id, False)
                self._runtime_monitor.publish_camera_status(camera_id)
                raise
            self._mqtt.publish_camera_state(camera_id, False)
            self._runtime_monitor.publish_camera_status(camera_id)
            return True

    def reconfigure_recorders(
        self,
        config: AppConfig,
        *,
        restart_recorders: bool,
    ) -> None:
        """Serialize recorder cutover against camera recording commands."""
        with self._lock:
            self._recording.reconfigure(
                config,
                list(self._cameras.values()),
                self._desired_recording_state_locked(),
                restart_recorders=restart_recorders,
            )

    def _desired_recording_state_locked(self) -> dict[str, bool]:
        return {
            camera_id: (
                self._camera_enabled.get(camera_id, True)
                and self._recording_enabled.get(camera_id, True)
            )
            for camera_id in self._cameras
        }

    def _command_allowed(self, camera_id: str) -> bool:
        return self._accepting_commands and camera_id in self._cameras

    def _update_preference_locked(
        self,
        category: str,
        camera_id: str,
        enabled: bool,
    ) -> None:
        values = self._preference_map(category)
        had_previous = camera_id in values
        previous = values.get(camera_id)
        values[camera_id] = bool(enabled)
        try:
            self._persist_locked()
        except BaseException:
            if had_previous:
                values[camera_id] = bool(previous)
            else:
                values.pop(camera_id, None)
            raise

    def _preference_map(self, category: str) -> dict[str, bool]:
        if category == "recording_enabled":
            return self._recording_enabled
        if category == "detection_enabled":
            return self._detection_enabled
        if category == "camera_enabled":
            return self._camera_enabled
        raise ValueError(f"unknown runtime preference category {category!r}")

    def _restore_preference_locked(
        self,
        category: str,
        camera_id: str,
        enabled: bool,
    ) -> None:
        """Restore memory and disk after a runtime transition rejects a change."""
        self._preference_map(category)[camera_id] = bool(enabled)
        try:
            self._persist_locked()
        except Exception:
            LOGGER.exception(
                "failed to persist rolled-back %s state for %s",
                category,
                camera_id,
            )

    def _restore_locked(self, snapshot: Mapping[str, Mapping[str, bool]]) -> None:
        self._recording_enabled = dict(snapshot["recording_enabled"])
        self._detection_enabled = dict(snapshot["detection_enabled"])
        self._camera_enabled = dict(snapshot["camera_enabled"])

    def _persist_locked(self) -> None:
        payload = self._snapshot_locked()
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._state_path.parent,
                prefix=f".{self._state_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._state_path)
            temporary = None
            try:
                directory_fd = os.open(self._state_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

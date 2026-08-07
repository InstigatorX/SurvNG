"""Stable lifecycle ownership for replaceable MQTT client generations."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .config import MqttConfig
from .mqtt import MqttService


class MqttLifecycle:
    """Own MQTT generations while presenting one stable publisher identity."""

    def __init__(
        self,
        config: MqttConfig,
        service_factory: Callable[[MqttConfig], MqttService],
        *,
        service: MqttService | None = None,
    ) -> None:
        self._factory = service_factory
        self._lock = threading.RLock()
        self._current_lock = threading.Lock()
        self._service = service or service_factory(config)
        self._closed = False

    @property
    def service(self) -> MqttService:
        with self._current_lock:
            return self._service

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("MQTT lifecycle is closed")
            self._service.start()

    def stop(self, *, lifecycle: str = "stopping") -> None:
        with self._lock:
            if lifecycle == "stopping":
                self._service.stop()
            else:
                self._service.stop(lifecycle=lifecycle)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._service.stop()

    def reconfigure(
        self,
        config: MqttConfig,
        *,
        running: bool,
        allowed: Callable[[], bool] | None = None,
    ) -> None:
        """Cut over to one prepared generation or restore the prior client."""
        with self._lock:
            if self._closed:
                raise RuntimeError("MQTT lifecycle is closed")
            if allowed is not None and not allowed():
                raise RuntimeError("application manager is stopping")
            previous = self._service
            replacement = self._factory(config)
            previous.stop(lifecycle="restarting", require_quiesced=True)
            with self._current_lock:
                self._service = replacement
            try:
                replacement.start(raise_on_failure=True)
                if running:
                    replacement.set_server_lifecycle("running")
            except BaseException:
                try:
                    replacement.stop(lifecycle="restarting")
                finally:
                    with self._current_lock:
                        self._service = previous
                previous.start(raise_on_failure=True)
                if running:
                    previous.set_server_lifecycle("running")
                raise

    def _current(self) -> MqttService:
        with self._current_lock:
            return self._service

    # These explicit forwarding methods make this lifecycle a stable typed
    # publisher for camera admission and application event producers. They
    # snapshot a generation under the lock, then avoid holding the lifecycle
    # lock across broker I/O or callbacks.
    def set_server_lifecycle(self, lifecycle: str, *, refresh_status: bool = True) -> None:
        self._current().set_server_lifecycle(lifecycle, refresh_status=refresh_status)

    def publish_camera_state(self, camera_id: str, running: bool) -> None:
        self._current().publish_camera_state(camera_id, running)

    def publish_camera_feature_state(self, camera_id: str, feature: str, enabled: bool) -> None:
        self._current().publish_camera_feature_state(camera_id, feature, enabled)

    def publish_discovery(self, cameras: list[dict[str, Any]]) -> None:
        self._current().publish_discovery(cameras)

    def remove_zone_discovery(
        self,
        camera_id: str,
        zones: list[dict[str, Any]],
        model_classes: list[str],
    ) -> None:
        self._current().remove_zone_discovery(camera_id, zones, model_classes)

    def track_incident(
        self,
        event: dict[str, Any],
        camera_name: str,
        base_path: str = "",
        allow_new: bool = True,
    ) -> None:
        self._current().track_incident(
            event,
            camera_name,
            base_path,
            allow_new=allow_new,
        )

    def publish(self, suffix: str, payload: dict[str, Any], retain: bool = False) -> None:
        self._current().publish(suffix, payload, retain=retain)

    def publish_zone_objects(
        self,
        camera_id: str,
        zones: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> None:
        self._current().publish_zone_objects(camera_id, zones, payload)

    def status(self) -> dict[str, Any]:
        return self._current().status()

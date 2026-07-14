from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from typing import Any, Callable

from .config import MqttConfig

LOGGER = logging.getLogger(__name__)


class MqttService:
    def __init__(
        self,
        config: MqttConfig,
        power_callback: Callable[[str, bool], bool],
        recording_callback: Callable[[str, bool], bool],
        detection_callback: Callable[[str, bool], bool],
        connected_callback: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self.power_callback = power_callback
        self.recording_callback = recording_callback
        self.detection_callback = detection_callback
        self.connected_callback = connected_callback
        self.client: Any = None
        self.connected = False
        self.last_connected_at = ""
        self.last_error = ""
        self.messages_published = 0
        self.commands_received = 0
        self._lock = threading.Lock()

    @property
    def prefix(self) -> str:
        return self.config.topic_prefix.strip().strip("/") or "survng"

    def start(self) -> None:
        if not self.config.enabled or not self.config.host.strip() or self.client is not None:
            return
        try:
            import paho.mqtt.client as mqtt

            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.config.client_id.strip() or "survng",
                protocol=mqtt.MQTTv311,
            )
            if self.config.username:
                client.username_pw_set(self.config.username, self.config.password or None)
            if self.config.tls:
                client.tls_set()
            client.reconnect_delay_set(min_delay=1, max_delay=60)
            client.will_set(
                f"{self.prefix}/status",
                json.dumps({"online": False}),
                qos=self.config.qos,
                retain=True,
            )
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.on_message = self._on_message
            self.client = client
            client.connect_async(self.config.host.strip(), self.config.port, keepalive=45)
            client.loop_start()
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.exception("failed to start MQTT client")

    def stop(self) -> None:
        client = self.client
        if client is None:
            return
        try:
            if self.connected:
                self.publish("status", {"online": False}, retain=True)
                client.disconnect()
            client.loop_stop()
        except Exception:
            LOGGER.exception("failed to stop MQTT client cleanly")
        finally:
            self.connected = False
            self.client = None

    def publish(self, suffix: str, payload: dict[str, Any], retain: bool = False) -> None:
        self.publish_topic(f"{self.prefix}/{suffix.strip('/')}", payload, retain=retain)

    def publish_topic(self, topic: str, payload: dict[str, Any], retain: bool = False) -> None:
        client = self.client
        if client is None or not self.connected:
            return
        try:
            info = client.publish(
                topic,
                json.dumps(payload, separators=(",", ":"), default=str),
                qos=self.config.qos,
                retain=retain,
            )
            if info.rc == 0:
                with self._lock:
                    self.messages_published += 1
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.warning("MQTT publish failed for %s: %s", topic, exc)

    def remove_retained_topic(self, topic: str) -> None:
        client = self.client
        if client is None or not self.connected:
            return
        try:
            info = client.publish(topic, "", qos=self.config.qos, retain=True)
            if info.rc == 0:
                with self._lock:
                    self.messages_published += 1
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.warning("MQTT retained-topic removal failed for %s: %s", topic, exc)

    @staticmethod
    def _slug(value: str, fallback: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or fallback

    def publish_discovery(self, cameras: list[dict[str, Any]]) -> None:
        if not self.config.discovery_enabled:
            return
        discovery_prefix = self.config.discovery_prefix.strip().strip("/") or "homeassistant"
        availability = {
            "availability_topic": f"{self.prefix}/status",
            "availability_template": "{{ value_json.online }}",
            "payload_available": "True",
            "payload_not_available": "False",
        }
        for camera in cameras:
            camera_id = str(camera.get("id") or "")
            if not camera_id:
                continue
            camera_name = str(camera.get("name") or camera_id)
            camera_slug = self._slug(camera_id, "camera")
            object_id = f"survng_{camera_slug}"
            device = {
                "identifiers": [f"survng_{camera_id}"],
                "name": camera_name,
                "manufacturer": "SurvNG",
                "model": "Network Camera",
            }
            state_topic = f"{self.prefix}/camera/{camera_id}"
            entities = [
                ("switch", "power", {
                    "name": "Power",
                    "unique_id": f"survng_{camera_id}_power",
                    "state_topic": f"{state_topic}/state",
                    "value_template": "{{ value_json.state }}",
                    "command_topic": f"{state_topic}/power/set",
                    "payload_on": "ON",
                    "payload_off": "OFF",
                    "state_on": "ON",
                    "state_off": "OFF",
                    "icon": "mdi:cctv",
                }),
                ("switch", "recording", {
                    "name": "Recording",
                    "unique_id": f"survng_{camera_id}_recording",
                    "state_topic": f"{state_topic}/recording/state",
                    "value_template": "{{ value_json.state }}",
                    "command_topic": f"{state_topic}/recording/set",
                    "payload_on": "ON",
                    "payload_off": "OFF",
                    "state_on": "ON",
                    "state_off": "OFF",
                    "icon": "mdi:record-rec",
                }),
                ("switch", "detection", {
                    "name": "Detection",
                    "unique_id": f"survng_{camera_id}_detection",
                    "state_topic": f"{state_topic}/detection/state",
                    "value_template": "{{ value_json.state }}",
                    "command_topic": f"{state_topic}/detection/set",
                    "payload_on": "ON",
                    "payload_off": "OFF",
                    "state_on": "ON",
                    "state_off": "OFF",
                    "icon": "mdi:motion-sensor",
                }),
                ("binary_sensor", "motion", {
                    "name": "Motion",
                    "unique_id": f"survng_{camera_id}_motion",
                    "state_topic": f"{state_topic}/motion",
                    "value_template": "{{ 'ON' if value_json.camera_id else 'OFF' }}",
                    "payload_on": "ON",
                    "payload_off": "OFF",
                    "device_class": "motion",
                    "off_delay": 10,
                }),
                ("binary_sensor", "object", {
                    "name": "Object",
                    "unique_id": f"survng_{camera_id}_object",
                    "state_topic": f"{state_topic}/object",
                    "value_template": "{{ 'ON' if value_json.objects | length > 0 else 'OFF' }}",
                    "payload_on": "ON",
                    "payload_off": "OFF",
                    "device_class": "occupancy",
                    "off_delay": 15,
                }),
                ("sensor", "classes", {
                    "name": "Last Objects",
                    "unique_id": f"survng_{camera_id}_classes",
                    "state_topic": f"{state_topic}/object",
                    "value_template": "{{ value_json.classes | join(', ') }}",
                    "json_attributes_topic": f"{state_topic}/object",
                    "icon": "mdi:shape",
                }),
            ]
            self.remove_retained_topic(
                f"{discovery_prefix}/sensor/{object_id}/zones/config"
            )
            for zone in camera.get("zones") or []:
                zone_name = str(zone.get("name") or "").strip()
                if not zone_name:
                    continue
                zone_slug = self._slug(zone_name, "zone")
                self.remove_retained_topic(
                    f"{discovery_prefix}/binary_sensor/{object_id}/zone_{zone_slug}/config"
                )
                zone_object_id = f"survng_{camera_slug}_zone_{zone_slug}"
                zone_state_topic = f"{self.prefix}/zone/{camera_id}/{zone_slug}"
                self.remove_retained_topic(
                    f"{discovery_prefix}/binary_sensor/{zone_object_id}/object/config"
                )
                for class_name in camera.get("model_classes") or []:
                    class_slug = self._slug(str(class_name), "object")
                    self.remove_retained_topic(
                        f"{discovery_prefix}/binary_sensor/{zone_object_id}/class_{class_slug}/config"
                    )
                if zone.get("enabled") is False:
                    continue
                zone_device = {
                    "identifiers": [f"survng_{camera_id}_zone_{zone_slug}"],
                    "name": f"{camera_name} {zone_name}",
                    "manufacturer": "SurvNG",
                    "model": "Detection Zone",
                    "via_device": f"survng_{camera_id}",
                }
                zone_entities = [("binary_sensor", "object", {
                    "name": "Any Object",
                    "unique_id": f"survng_{camera_id}_zone_{zone_slug}_object",
                    "state_topic": f"{zone_state_topic}/object",
                    "value_template": "{{ 'ON' if value_json.detected else 'OFF' }}",
                    "payload_on": "ON",
                    "payload_off": "OFF",
                    "device_class": "occupancy",
                    "off_delay": 15,
                    "icon": "mdi:vector-polygon",
                    "json_attributes_topic": f"{zone_state_topic}/object",
                })]
                for class_name in zone.get("object_classes") or []:
                    normalized_class = str(class_name or "").strip()
                    if not normalized_class:
                        continue
                    class_slug = self._slug(normalized_class, "object")
                    zone_entities.append(("binary_sensor", f"class_{class_slug}", {
                        "name": normalized_class.replace("_", " ").title(),
                        "unique_id": f"survng_{camera_id}_zone_{zone_slug}_{class_slug}",
                        "state_topic": f"{zone_state_topic}/class/{class_slug}",
                        "value_template": "{{ 'ON' if value_json.detected else 'OFF' }}",
                        "payload_on": "ON",
                        "payload_off": "OFF",
                        "device_class": "occupancy",
                        "off_delay": 15,
                        "icon": "mdi:shape",
                        "json_attributes_topic": f"{zone_state_topic}/class/{class_slug}",
                    }))
                for component, entity_name, entity in zone_entities:
                    payload = {
                        **entity,
                        **availability,
                        "device": zone_device,
                        "origin": {"name": "SurvNG"},
                    }
                    topic = (
                        f"{discovery_prefix}/{component}/{zone_object_id}/{entity_name}/config"
                    )
                    self.publish_topic(topic, payload, retain=True)
            for component, entity_name, entity in entities:
                payload = {**entity, **availability, "device": device, "origin": {"name": "SurvNG"}}
                topic = f"{discovery_prefix}/{component}/{object_id}/{entity_name}/config"
                self.publish_topic(topic, payload, retain=True)

    def remove_zone_discovery(
        self,
        camera_id: str,
        zones: list[dict[str, Any]],
        model_classes: list[str],
    ) -> None:
        if not self.config.discovery_enabled:
            return
        discovery_prefix = self.config.discovery_prefix.strip().strip("/") or "homeassistant"
        camera_slug = self._slug(camera_id, "camera")
        camera_object_id = f"survng_{camera_slug}"
        for zone in zones:
            zone_name = str(zone.get("name") or "").strip()
            if not zone_name:
                continue
            zone_slug = self._slug(zone_name, "zone")
            zone_object_id = f"survng_{camera_slug}_zone_{zone_slug}"
            self.remove_retained_topic(
                f"{discovery_prefix}/binary_sensor/{camera_object_id}/zone_{zone_slug}/config"
            )
            self.remove_retained_topic(
                f"{discovery_prefix}/binary_sensor/{zone_object_id}/object/config"
            )
            class_names = {
                str(value).strip()
                for value in [*model_classes, *(zone.get("object_classes") or [])]
                if str(value).strip()
            }
            for class_name in class_names:
                class_slug = self._slug(class_name, "object")
                self.remove_retained_topic(
                    f"{discovery_prefix}/binary_sensor/{zone_object_id}/class_{class_slug}/config"
                )

    def publish_zone_objects(
        self,
        camera_id: str,
        zones: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> None:
        objects = payload.get("objects") or []
        for zone in zones:
            zone_name = str(zone.get("name") or "").strip()
            if not zone_name or zone.get("enabled") is False:
                continue
            zone_objects = [
                item for item in objects
                if zone_name in (item.get("zones") or [])
            ]
            if not zone_objects:
                continue
            zone_slug = self._slug(zone_name, "zone")
            zone_classes = sorted({
                str(item.get("label")) for item in zone_objects if item.get("label")
            })
            zone_payload = {
                "detected": True,
                "camera_id": camera_id,
                "zone": zone_name,
                "event_id": payload.get("event_id"),
                "created_at": payload.get("created_at"),
                "classes": zone_classes,
                "objects": zone_objects,
            }
            base_topic = f"zone/{camera_id}/{zone_slug}"
            self.publish(f"{base_topic}/object", zone_payload)
            configured_classes = {
                str(value).strip().lower()
                for value in (zone.get("object_classes") or [])
                if str(value).strip()
            }
            for class_name in zone_classes:
                if configured_classes and class_name.lower() not in configured_classes:
                    continue
                class_slug = self._slug(class_name, "object")
                class_objects = [
                    item for item in zone_objects
                    if str(item.get("label") or "").lower() == class_name.lower()
                ]
                self.publish(
                    f"{base_topic}/class/{class_slug}",
                    {**zone_payload, "class": class_name, "objects": class_objects},
                )

    def publish_camera_state(self, camera_id: str, running: bool) -> None:
        self.publish(
            f"camera/{camera_id}/state",
            {"camera_id": camera_id, "state": "ON" if running else "OFF", "running": running},
            retain=True,
        )

    def publish_camera_feature_state(self, camera_id: str, feature: str, enabled: bool) -> None:
        if feature not in {"recording", "detection"}:
            return
        self.publish(
            f"camera/{camera_id}/{feature}/state",
            {"camera_id": camera_id, "state": "ON" if enabled else "OFF", "enabled": enabled},
            retain=True,
        )

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "configured": bool(self.config.host.strip()),
            "connected": self.connected,
            "host": self.config.host,
            "port": self.config.port,
            "topic_prefix": self.prefix,
            "last_connected_at": self.last_connected_at,
            "last_error": self.last_error,
            "messages_published": self.messages_published,
            "commands_received": self.commands_received,
        }

    def _on_connect(self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
        if bool(getattr(reason_code, "is_failure", reason_code != 0)):
            self.connected = False
            self.last_error = f"broker rejected connection: {reason_code}"
            return
        self.connected = True
        self.last_error = ""
        self.last_connected_at = datetime.now(timezone.utc).isoformat()
        client.subscribe(f"{self.prefix}/camera/+/power/set", qos=self.config.qos)
        client.subscribe(f"{self.prefix}/camera/+/recording/set", qos=self.config.qos)
        client.subscribe(f"{self.prefix}/camera/+/detection/set", qos=self.config.qos)
        self.publish("status", {"online": True, "connected_at": self.last_connected_at}, retain=True)
        if self.connected_callback:
            threading.Thread(target=self.connected_callback, daemon=True).start()
        LOGGER.info("MQTT connected to %s:%s", self.config.host, self.config.port)

    def _on_disconnect(self, client: Any, userdata: Any, disconnect_flags: Any, reason_code: Any, properties: Any) -> None:
        self.connected = False
        if bool(getattr(reason_code, "is_failure", reason_code != 0)):
            self.last_error = f"disconnected: {reason_code}"
            LOGGER.warning("MQTT disconnected unexpectedly: %s", reason_code)

    def _on_message(self, client: Any, userdata: Any, message: Any) -> None:
        parts = message.topic.split("/")
        prefix_parts = self.prefix.split("/")
        expected_length = len(prefix_parts) + 4
        if len(parts) != expected_length or parts[:len(prefix_parts)] != prefix_parts:
            return
        if parts[-1] != "set" or parts[-4] != "camera":
            return
        camera_id = parts[-3]
        feature = parts[-2]
        callbacks = {
            "power": self.power_callback,
            "recording": self.recording_callback,
            "detection": self.detection_callback,
        }
        callback = callbacks.get(feature)
        if callback is None:
            return
        raw = message.payload.decode("utf-8", errors="replace").strip()
        try:
            decoded = json.loads(raw)
            value = decoded.get("state", decoded.get(feature, decoded)) if isinstance(decoded, dict) else decoded
        except json.JSONDecodeError:
            value = raw
        normalized = str(value).strip().upper()
        if normalized not in {"ON", "OFF", "TRUE", "FALSE", "1", "0"}:
            LOGGER.warning("ignored invalid MQTT camera %s command for %s: %s", feature, camera_id, raw[:100])
            return
        turn_on = normalized in {"ON", "TRUE", "1"}
        with self._lock:
            self.commands_received += 1
        threading.Thread(target=self._apply_control, args=(camera_id, feature, turn_on, callback), daemon=True).start()

    def _apply_control(
        self,
        camera_id: str,
        feature: str,
        turn_on: bool,
        callback: Callable[[str, bool], bool],
    ) -> None:
        try:
            applied = callback(camera_id, turn_on)
            if not applied:
                self.publish(f"camera/{camera_id}/command/error", {"error": "camera not found"})
                return
        except Exception as exc:
            LOGGER.exception("MQTT camera %s command failed for %s", feature, camera_id)
            self.publish(f"camera/{camera_id}/command/error", {"error": str(exc)})

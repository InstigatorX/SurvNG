from __future__ import annotations

import json
import logging
import queue
import re
import threading
from datetime import datetime, timezone
from typing import Any, Callable

from .config import MqttConfig
from .incident_utils import (
    DEFAULT_INCIDENT_GAP_SECONDS,
    event_epoch,
    stable_incident_id,
    stable_incident_key,
)
from .security import redact_secret_text

LOGGER = logging.getLogger(__name__)
MQTT_COMMAND_QUEUE_SIZE = 64
MQTT_COMMAND_STOP_TIMEOUT_SECONDS = 5.0
MQTT_COMMAND_MAX_PAYLOAD_BYTES = 4096
MQTT_SERVER_MONITOR_STOP_TIMEOUT_SECONDS = 5.0


class MqttService:
    def __init__(
        self,
        config: MqttConfig,
        power_callback: Callable[[str, bool], bool],
        recording_callback: Callable[[str, bool], bool],
        detection_callback: Callable[[str, bool], bool],
        connected_callback: Callable[[], None] | None = None,
        server_status_callback: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.power_callback = power_callback
        self.recording_callback = recording_callback
        self.detection_callback = detection_callback
        self.connected_callback = connected_callback
        self.server_status_callback = server_status_callback
        self.client: Any = None
        self.connected = False
        self.last_connected_at = ""
        self.last_error = ""
        self.server_status_error = ""
        self.messages_published = 0
        self.publish_failures = 0
        self.commands_received = 0
        self.commands_rejected = 0
        self.command_errors = 0
        self.subscription_failures = 0
        self.command_subscriptions_active = False
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._command_queue: queue.Queue[
            tuple[str, str, bool, Callable[[str, bool], bool]] | None
        ] = queue.Queue(maxsize=MQTT_COMMAND_QUEUE_SIZE)
        self._command_stop = threading.Event()
        self._command_thread: threading.Thread | None = None
        self._incident_lock = threading.RLock()
        self._pending_incidents: dict[str, dict[str, Any]] = {}
        self._accept_incidents = True
        self._server_lifecycle = "starting"
        self._server_monitor_stop = threading.Event()
        self._server_monitor_thread: threading.Thread | None = None
        self._last_server_state: tuple[str, str, str] | None = None
        self._server_state_payload: dict[str, Any] = {}

    @property
    def prefix(self) -> str:
        return self.config.topic_prefix.strip().strip("/") or "survng"

    def start(self) -> None:
        with self._lifecycle_lock:
            self._accept_incidents = True
            if not self.config.enabled or not self.config.host.strip() or self.client is not None:
                return
            client: Any = None
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
                    json.dumps({"online": False, "lifecycle": "offline"}),
                    qos=self.config.qos,
                    retain=True,
                )
                client.on_connect = self._on_connect
                client.on_disconnect = self._on_disconnect
                client.on_message = self._on_message
                self._start_command_worker()
                self._start_server_monitor()
                self.client = client
                connect_result = client.connect_async(
                    self.config.host.strip(),
                    self.config.port,
                    keepalive=45,
                )
                # Paho 2.x documents connect_async() as returning None after it
                # validates and stores the connection parameters. Older client
                # releases and test doubles may return MQTT_ERR_SUCCESS instead.
                if (
                    connect_result is not None
                    and int(connect_result) != int(mqtt.MQTT_ERR_SUCCESS)
                ):
                    raise RuntimeError(f"MQTT connect setup failed: rc={connect_result}")
                loop_result = client.loop_start()
                if int(loop_result) != int(mqtt.MQTT_ERR_SUCCESS):
                    raise RuntimeError(f"MQTT network loop failed to start: rc={loop_result}")
            except Exception as exc:
                self.last_error = redact_secret_text(exc)
                self.connected = False
                self.client = None
                if client is not None:
                    try:
                        client.loop_stop()
                    except Exception:
                        LOGGER.debug("failed to roll back MQTT network loop", exc_info=True)
                self._stop_command_worker()
                self._stop_server_monitor()
                LOGGER.exception("failed to start MQTT client")

    def stop(self, *, lifecycle: str = "stopping") -> None:
        with self._lifecycle_lock:
            client = self.client
            self._accept_incidents = False
            was_connected = bool(client is not None and self.connected)
            if was_connected:
                try:
                    self.flush_incidents()
                except Exception:
                    LOGGER.exception("failed to flush pending MQTT incidents")
            try:
                self._cancel_incident_timers()
            except Exception:
                LOGGER.exception("failed to cancel pending MQTT incident timers")
            if client is not None:
                if was_connected:
                    self.set_server_lifecycle(lifecycle, refresh_status=False)
                    self.publish(
                        "status",
                        {"online": False, "lifecycle": self._server_lifecycle},
                        retain=True,
                    )
                # Invalidate the client before disconnecting so callbacks
                # racing with shutdown cannot restore connected state or
                # enqueue commands for a service that is stopping.
                self.client = None
                self.connected = False
                self.command_subscriptions_active = False
                if was_connected:
                    try:
                        client.disconnect()
                    except Exception:
                        LOGGER.exception("failed to disconnect MQTT client cleanly")
                try:
                    client.loop_stop()
                except Exception:
                    LOGGER.exception("failed to stop MQTT network loop cleanly")
            self.connected = False
            self.command_subscriptions_active = False
            self.client = None
            self._stop_command_worker()
            self._stop_server_monitor()

    def _start_server_monitor(self) -> None:
        if not self.config.server_status_enabled or self.server_status_callback is None:
            return
        thread = self._server_monitor_thread
        if thread is not None and thread.is_alive():
            return
        self._server_monitor_stop.clear()
        thread = threading.Thread(
            target=self._run_server_monitor,
            name="mqtt-server-monitor",
            daemon=True,
        )
        self._server_monitor_thread = thread
        thread.start()

    def _stop_server_monitor(self) -> None:
        self._server_monitor_stop.set()
        thread = self._server_monitor_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=MQTT_SERVER_MONITOR_STOP_TIMEOUT_SECONDS)
        if thread is not None and thread.is_alive():
            LOGGER.error("MQTT server monitor did not stop")
        else:
            self._server_monitor_thread = None

    def _run_server_monitor(self) -> None:
        interval = float(self.config.server_metrics_interval_seconds)
        while not self._server_monitor_stop.wait(interval):
            if self.connected:
                self.publish_server_status()

    def set_server_lifecycle(self, lifecycle: str, *, refresh_status: bool = True) -> None:
        normalized = str(lifecycle or "").strip().lower()
        if normalized not in {"starting", "running", "stopping", "restarting"}:
            raise ValueError(f"invalid server lifecycle: {lifecycle}")
        self._server_lifecycle = normalized
        if self.connected:
            self.publish(
                "status",
                {
                    "online": True,
                    "lifecycle": normalized,
                    "connected_at": self.last_connected_at,
                },
                retain=True,
            )
            if refresh_status:
                self.publish_server_status()
            else:
                now = datetime.now(timezone.utc).isoformat()
                with self._lock:
                    state = dict(self._server_state_payload)
                state.update({"lifecycle": normalized, "updated_at": now})
                self.publish("server/state", state, retain=True)
                self._publish_server_transition(state, now)

    def _publish_server_transition(self, state: dict[str, Any], created_at: str) -> None:
        fingerprint = (
            self._server_lifecycle,
            str(state.get("health") or "unknown"),
            str(state.get("activity") or "idle"),
        )
        with self._lock:
            changed = fingerprint != self._last_server_state
            if changed:
                self._last_server_state = fingerprint
        if changed:
            self.publish(
                "server/event",
                {
                    "event": "state_changed",
                    "lifecycle": fingerprint[0],
                    "health": fingerprint[1],
                    "activity": fingerprint[2],
                    "created_at": created_at,
                },
            )

    def publish_server_status(self) -> None:
        if not self.config.server_status_enabled or self.server_status_callback is None:
            return
        try:
            snapshot = dict(self.server_status_callback() or {})
            state = dict(snapshot.get("state") or {})
            metrics = dict(snapshot.get("metrics") or {})
            now = datetime.now(timezone.utc).isoformat()
            state.update({
                "lifecycle": self._server_lifecycle,
                "updated_at": now,
            })
            metrics["updated_at"] = now
            with self._lock:
                self._server_state_payload = dict(state)
            self.publish("server/state", state, retain=True)
            self.publish("server/metrics", metrics, retain=True)
            self._publish_server_transition(state, now)
            self.server_status_error = ""
        except Exception as exc:
            self.server_status_error = redact_secret_text(exc)
            LOGGER.exception("failed to publish MQTT server telemetry")

    def _start_command_worker(self) -> None:
        thread = self._command_thread
        if thread is not None and thread.is_alive():
            if self._command_stop.is_set():
                raise RuntimeError("previous MQTT command worker is still stopping")
            return
        while True:
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break
        self._command_stop.clear()
        thread = threading.Thread(
            target=self._run_commands,
            name="mqtt-commands",
            # Command callbacks are bounded by normal application operations,
            # but a third-party or filesystem call can still wedge. stop()
            # waits for the worker; daemon status is the final process-exit
            # safeguard after that deadline expires.
            daemon=True,
        )
        self._command_thread = thread
        thread.start()

    def _stop_command_worker(self) -> None:
        self._command_stop.set()
        while True:
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._command_queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._command_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=MQTT_COMMAND_STOP_TIMEOUT_SECONDS)
        if thread is not None and thread.is_alive():
            LOGGER.error("MQTT command worker did not stop")
        else:
            self._command_thread = None

    def _run_commands(self) -> None:
        while not self._command_stop.is_set():
            try:
                command = self._command_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if command is None or self._command_stop.is_set():
                return
            self._apply_control(*command)

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
            else:
                with self._lock:
                    self.publish_failures += 1
                self.last_error = f"publish failed for {topic}: rc={info.rc}"
        except Exception as exc:
            with self._lock:
                self.publish_failures += 1
            self.last_error = redact_secret_text(exc)
            LOGGER.warning("MQTT publish failed for %s: %s", topic, self.last_error)

    def remove_retained_topic(self, topic: str) -> None:
        client = self.client
        if client is None or not self.connected:
            return
        try:
            info = client.publish(topic, "", qos=self.config.qos, retain=True)
            if info.rc == 0:
                with self._lock:
                    self.messages_published += 1
            else:
                with self._lock:
                    self.publish_failures += 1
                self.last_error = f"retained-topic removal failed for {topic}: rc={info.rc}"
        except Exception as exc:
            with self._lock:
                self.publish_failures += 1
            self.last_error = redact_secret_text(exc)
            LOGGER.warning("MQTT retained-topic removal failed for %s: %s", topic, self.last_error)

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
        self._publish_server_discovery(discovery_prefix, availability)
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

    def _publish_server_discovery(
        self,
        discovery_prefix: str,
        availability: dict[str, str],
    ) -> None:
        if not self.config.server_status_enabled:
            return
        object_id = "survng_server"
        state_topic = f"{self.prefix}/server/state"
        metrics_topic = f"{self.prefix}/server/metrics"
        device = {
            "identifiers": [object_id],
            "name": self.config.server_name,
            "manufacturer": "SurvNG",
            "model": "Video Surveillance Server",
        }
        entities: list[tuple[str, str, dict[str, Any]]] = [
            ("sensor", "lifecycle", {
                "name": "Lifecycle",
                "unique_id": "survng_server_lifecycle",
                "state_topic": state_topic,
                "value_template": "{{ value_json.lifecycle }}",
                "device_class": "enum",
                "options": ["starting", "running", "stopping", "restarting"],
                "icon": "mdi:server",
                "json_attributes_topic": state_topic,
            }),
            ("sensor", "health", {
                "name": "Health",
                "unique_id": "survng_server_health",
                "state_topic": state_topic,
                "value_template": "{{ value_json.health }}",
                "device_class": "enum",
                "options": ["ok", "degraded", "fault"],
                "icon": "mdi:heart-pulse",
            }),
            ("sensor", "activity", {
                "name": "Activity",
                "unique_id": "survng_server_activity",
                "state_topic": state_topic,
                "value_template": "{{ value_json.activity }}",
                "icon": "mdi:progress-wrench",
            }),
            ("binary_sensor", "problem", {
                "name": "Problem",
                "unique_id": "survng_server_problem",
                "state_topic": state_topic,
                "value_template": "{{ 'ON' if value_json.health in ['degraded', 'fault'] else 'OFF' }}",
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": "problem",
            }),
            ("sensor", "uptime", {
                "name": "Uptime",
                "unique_id": "survng_server_uptime",
                "state_topic": state_topic,
                "value_template": "{{ value_json.uptime_seconds }}",
                "device_class": "duration",
                "unit_of_measurement": "s",
                "state_class": "measurement",
                "icon": "mdi:timer-outline",
            }),
            ("sensor", "cameras", {
                "name": "Cameras Running",
                "unique_id": "survng_server_cameras_running",
                "state_topic": metrics_topic,
                "value_template": "{{ value_json.cameras_running }}",
                "state_class": "measurement",
                "icon": "mdi:cctv",
                "json_attributes_topic": metrics_topic,
            }),
            ("sensor", "recorders", {
                "name": "Recorders Running",
                "unique_id": "survng_server_recorders_running",
                "state_topic": metrics_topic,
                "value_template": "{{ value_json.recorders_running }}",
                "state_class": "measurement",
                "icon": "mdi:record-rec",
            }),
            ("sensor", "cpu", {
                "name": "CPU Load",
                "unique_id": "survng_server_cpu_load",
                "state_topic": metrics_topic,
                "value_template": "{{ value_json.cpu_load_percent }}",
                "unit_of_measurement": "%",
                "state_class": "measurement",
                "icon": "mdi:cpu-64-bit",
            }),
            ("sensor", "memory", {
                "name": "Memory Used",
                "unique_id": "survng_server_memory_used",
                "state_topic": metrics_topic,
                "value_template": "{{ value_json.memory_used_percent }}",
                "unit_of_measurement": "%",
                "state_class": "measurement",
                "icon": "mdi:memory",
            }),
            ("sensor", "storage", {
                "name": "Storage Free",
                "unique_id": "survng_server_storage_free",
                "state_topic": metrics_topic,
                "value_template": "{{ value_json.storage_free_percent }}",
                "unit_of_measurement": "%",
                "state_class": "measurement",
                "icon": "mdi:harddisk",
            }),
            ("sensor", "detector", {
                "name": "Detector",
                "unique_id": "survng_server_detector",
                "state_topic": metrics_topic,
                "value_template": "{{ value_json.detector_state }}",
                "icon": "mdi:brain",
            }),
            ("sensor", "object_queue", {
                "name": "Object Queue",
                "unique_id": "survng_server_object_queue",
                "state_topic": metrics_topic,
                "value_template": "{{ value_json.object_queue_depth }}",
                "state_class": "measurement",
                "icon": "mdi:tray-full",
            }),
        ]
        for component, entity_name, entity in entities:
            payload = {
                **entity,
                **availability,
                "device": device,
                "origin": {"name": "SurvNG"},
            }
            self.publish_topic(
                f"{discovery_prefix}/{component}/{object_id}/{entity_name}/config",
                payload,
                retain=True,
            )

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

    @staticmethod
    def _event_objects(event: dict[str, Any]) -> list[dict[str, Any]]:
        raw = event.get("objects")
        if raw is None:
            try:
                raw = json.loads(str(event.get("objects_json") or "[]"))
            except (json.JSONDecodeError, TypeError, ValueError):
                raw = []
        if not isinstance(raw, list):
            return []
        detected: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("label") or item.get("incident_eligible") is False:
                continue
            try:
                confidence = float(item.get("confidence") or 0)
            except (TypeError, ValueError):
                continue
            if confidence > 0:
                detected.append(item)
        return detected

    @classmethod
    def _incident_payload(cls, pending: dict[str, Any], state: str) -> dict[str, Any]:
        events = sorted(pending["events"].values(), key=event_epoch)
        first = events[0]
        last = events[-1]

        def representative_score(event: dict[str, Any]) -> tuple[int, float, int, int]:
            objects = cls._event_objects(event)
            return (
                int(bool(objects)),
                max((float(item.get("confidence") or 0) for item in objects), default=0.0),
                int(bool(event.get("snapshot_path"))),
                int(event.get("id") or 0),
            )

        representative = max(events, key=representative_score)
        object_summaries: dict[str, dict[str, Any]] = {}
        for event in events:
            for item in cls._event_objects(event):
                label = str(item.get("label") or "").strip()
                key = label.lower()
                current = object_summaries.setdefault(key, {
                    "label": label,
                    "confidence": 0.0,
                    "zones": set(),
                    "count": 0,
                })
                current["confidence"] = max(current["confidence"], float(item.get("confidence") or 0))
                current["zones"].update(str(zone) for zone in item.get("zones", []) if zone)
                current["count"] += 1
        objects = [
            {
                **summary,
                "confidence": round(float(summary["confidence"]), 4),
                "zones": sorted(summary["zones"]),
            }
            for summary in sorted(object_summaries.values(), key=lambda item: str(item["label"]).lower())
        ]
        first_id = first.get("id")
        incident_id = stable_incident_id(str(pending["camera_id"]), first_id)
        base_path = str(pending.get("base_path") or "").rstrip("/")
        representative_id = int(representative.get("id") or 0)
        return {
            "schema_version": 1,
            "type": "incident",
            "state": state,
            "incident_id": incident_id,
            "incident_key": stable_incident_key(str(pending["camera_id"]), first_id),
            "camera_id": pending["camera_id"],
            "camera_name": pending["camera_name"],
            "started_at": first.get("created_at"),
            "ended_at": last.get("created_at"),
            "duration_seconds": round(max(0.0, event_epoch(last) - event_epoch(first)), 3),
            "event_count": len(events),
            "event_ids": [int(event.get("id") or 0) for event in events],
            "object_event_count": sum(bool(cls._event_objects(event)) for event in events),
            "has_objects": bool(objects),
            "classes": [item["label"] for item in objects],
            "zones": sorted({zone for item in objects for zone in item["zones"]}),
            "objects": objects,
            "representative_event_id": representative_id,
            "snapshot_url": f"{base_path}/api/events/{representative_id}/snapshot.jpg",
            "incidents_url": f"{base_path}/incidents",
        }

    def track_incident(
        self,
        event: dict[str, Any],
        camera_name: str,
        base_path: str = "",
        allow_new: bool = True,
    ) -> None:
        if not self.config.enabled or not self.config.incident_events_enabled or not self._accept_incidents:
            return
        camera_id = str(event.get("camera_id") or "")
        event_id = int(event.get("id") or 0)
        if not camera_id or event_id <= 0 or not event.get("created_at"):
            return

        publish_payloads: list[dict[str, Any]] = []
        with self._incident_lock:
            pending = self._pending_incidents.get(camera_id)
            if not allow_new and (pending is None or event_id not in pending["events"]):
                return
            if pending and event_epoch(event) - float(pending["last_epoch"]) > DEFAULT_INCIDENT_GAP_SECONDS:
                timer = pending.get("timer")
                if timer is not None:
                    timer.cancel()
                publish_payloads.append(self._incident_payload(pending, "complete"))
                self._pending_incidents.pop(camera_id, None)
                pending = None

            state = "updated"
            if pending is None:
                if not allow_new:
                    return
                pending = {
                    "camera_id": camera_id,
                    "camera_name": camera_name or camera_id,
                    "base_path": base_path,
                    "events": {},
                    "last_epoch": event_epoch(event),
                    "generation": 0,
                    "timer": None,
                }
                self._pending_incidents[camera_id] = pending
                state = "new"
            pending["camera_name"] = camera_name or camera_id
            pending["base_path"] = base_path
            pending["events"][event_id] = dict(event)
            pending["last_epoch"] = max(float(pending["last_epoch"]), event_epoch(event))
            pending["generation"] += 1
            timer = pending.get("timer")
            if timer is not None:
                timer.cancel()
            timer = threading.Timer(
                DEFAULT_INCIDENT_GAP_SECONDS,
                self._settle_incident,
                args=(camera_id, int(pending["generation"])),
            )
            timer.daemon = True
            pending["timer"] = timer
            timer.start()
            publish_payloads.append(self._incident_payload(pending, state))

        for payload in publish_payloads:
            self.publish("events/incidents", payload, retain=False)

    def _settle_incident(self, camera_id: str, generation: int) -> None:
        payload: dict[str, Any] | None = None
        with self._incident_lock:
            pending = self._pending_incidents.get(camera_id)
            if pending is None or int(pending["generation"]) != generation:
                return
            if not self.connected:
                timer = threading.Timer(5.0, self._settle_incident, args=(camera_id, generation))
                timer.daemon = True
                pending["timer"] = timer
                timer.start()
                return
            self._pending_incidents.pop(camera_id, None)
            payload = self._incident_payload(pending, "complete")
        self.publish("events/incidents", payload, retain=False)

    def flush_incidents(self) -> None:
        with self._incident_lock:
            pending_incidents = list(self._pending_incidents.values())
            self._pending_incidents.clear()
            for pending in pending_incidents:
                timer = pending.get("timer")
                if timer is not None:
                    timer.cancel()
        for pending in pending_incidents:
            self.publish("events/incidents", self._incident_payload(pending, "complete"), retain=False)

    def _cancel_incident_timers(self) -> None:
        with self._incident_lock:
            pending_incidents = list(self._pending_incidents.values())
            self._pending_incidents.clear()
        for pending in pending_incidents:
            timer = pending.get("timer")
            if timer is not None:
                timer.cancel()

    def status(self) -> dict[str, Any]:
        with self._incident_lock:
            pending_incidents = len(self._pending_incidents)
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
            "publish_failures": self.publish_failures,
            "commands_received": self.commands_received,
            "commands_rejected": self.commands_rejected,
            "command_errors": self.command_errors,
            "subscription_failures": self.subscription_failures,
            "command_subscriptions_active": self.command_subscriptions_active,
            "command_queue_depth": self._command_queue.qsize(),
            "command_worker_running": bool(
                self._command_thread is not None and self._command_thread.is_alive()
            ),
            "incident_events_enabled": self.config.incident_events_enabled,
            "incident_topic": f"{self.prefix}/events/incidents",
            "pending_incidents": pending_incidents,
            "server_status_enabled": self.config.server_status_enabled,
            "server_state_topic": f"{self.prefix}/server/state",
            "server_metrics_topic": f"{self.prefix}/server/metrics",
            "server_metrics_interval_seconds": self.config.server_metrics_interval_seconds,
            "server_lifecycle": self._server_lifecycle,
            "server_monitor_running": bool(
                self._server_monitor_thread is not None
                and self._server_monitor_thread.is_alive()
            ),
            "server_status_error": self.server_status_error,
        }

    def _on_connect(self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
        if client is not self.client:
            return
        if bool(getattr(reason_code, "is_failure", reason_code != 0)):
            self.connected = False
            self.last_error = f"broker rejected connection: {reason_code}"
            return
        self.connected = True
        self.command_subscriptions_active = False
        self.last_error = ""
        self.last_connected_at = datetime.now(timezone.utc).isoformat()
        try:
            subscriptions = [
                client.subscribe(f"{self.prefix}/camera/+/power/set", qos=self.config.qos),
                client.subscribe(f"{self.prefix}/camera/+/recording/set", qos=self.config.qos),
                client.subscribe(f"{self.prefix}/camera/+/detection/set", qos=self.config.qos),
            ]
            failed = [result for result in subscriptions if int(result[0]) != 0]
            if failed:
                self.subscription_failures += 1
                self.last_error = "one or more MQTT command subscriptions failed"
            else:
                self.command_subscriptions_active = True
        except Exception as exc:
            self.subscription_failures += 1
            self.last_error = f"MQTT command subscription failed: {redact_secret_text(exc)}"
            LOGGER.exception("failed to subscribe to MQTT command topics")
        self.publish(
            "status",
            {
                "online": True,
                "lifecycle": self._server_lifecycle,
                "connected_at": self.last_connected_at,
            },
            retain=True,
        )
        if self.connected_callback:
            try:
                self.connected_callback()
            except Exception:
                LOGGER.exception("MQTT connected callback failed")
        self.publish_server_status()
        LOGGER.info("MQTT connected to %s:%s", self.config.host, self.config.port)

    def _on_disconnect(self, client: Any, userdata: Any, disconnect_flags: Any, reason_code: Any, properties: Any) -> None:
        if client is not self.client:
            return
        self.connected = False
        self.command_subscriptions_active = False
        if bool(getattr(reason_code, "is_failure", reason_code != 0)):
            self.last_error = f"disconnected: {reason_code}"
            LOGGER.warning("MQTT disconnected unexpectedly: %s", reason_code)

    def _on_message(self, client: Any, userdata: Any, message: Any) -> None:
        if client is not self.client or not self.connected:
            return
        if bool(getattr(message, "retain", False)):
            with self._lock:
                self.commands_rejected += 1
            LOGGER.warning("ignored retained MQTT command on %s", message.topic)
            return
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
        if len(message.payload) > MQTT_COMMAND_MAX_PAYLOAD_BYTES:
            with self._lock:
                self.commands_rejected += 1
            LOGGER.warning("ignored oversized MQTT command on %s", message.topic)
            return
        raw = message.payload.decode("utf-8", errors="replace").strip()
        try:
            decoded = json.loads(raw)
            value = decoded.get("state", decoded.get(feature, decoded)) if isinstance(decoded, dict) else decoded
        except json.JSONDecodeError:
            value = raw
        normalized = str(value).strip().upper()
        if normalized not in {"ON", "OFF", "TRUE", "FALSE", "1", "0"}:
            with self._lock:
                self.commands_rejected += 1
            LOGGER.warning("ignored invalid MQTT camera %s command for %s: %s", feature, camera_id, raw[:100])
            return
        turn_on = normalized in {"ON", "TRUE", "1"}
        command = (camera_id, feature, turn_on, callback)
        try:
            self._command_queue.put_nowait(command)
        except queue.Full:
            with self._lock:
                self.commands_rejected += 1
            self.last_error = "MQTT command queue is full"
            LOGGER.warning("ignored MQTT camera command because the command queue is full")
            return
        with self._lock:
            self.commands_received += 1

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
            with self._lock:
                self.command_errors += 1
            error = redact_secret_text(exc)
            LOGGER.error("MQTT camera %s command failed for %s: %s", feature, camera_id, error)
            self.publish(f"camera/{camera_id}/command/error", {"error": error})

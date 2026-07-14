from __future__ import annotations

import json
import threading
import unittest
from types import SimpleNamespace

from survng.app.config import MqttConfig
from survng.app.mqtt import MqttService


class FakePublishResult:
    rc = 0


class FakeClient:
    def __init__(self) -> None:
        self.published: list[tuple[str, str, int, bool]] = []

    def publish(self, topic: str, payload: str, qos: int, retain: bool):
        self.published.append((topic, payload, qos, retain))
        return FakePublishResult()


class MqttServiceTest(unittest.TestCase):
    def service(self, recording_callback=lambda camera_id, enabled: True, detection_callback=lambda camera_id, enabled: True):
        service = MqttService(
            MqttConfig(enabled=True, host="broker", discovery_enabled=True),
            lambda camera_id, enabled: True,
            recording_callback,
            detection_callback,
        )
        service.client = FakeClient()
        service.connected = True
        return service

    def test_discovery_includes_recording_and_detection_switches(self) -> None:
        service = self.service()
        service.publish_discovery([{"id": "gate", "name": "Gate", "zones": []}])

        topics = {item[0] for item in service.client.published}
        self.assertIn("homeassistant/switch/survng_gate/recording/config", topics)
        self.assertIn("homeassistant/switch/survng_gate/detection/config", topics)

    def test_feature_state_is_retained(self) -> None:
        service = self.service()
        service.publish_camera_feature_state("gate", "recording", False)

        topic, raw_payload, _qos, retained = service.client.published[-1]
        self.assertEqual(topic, "survng/camera/gate/recording/state")
        self.assertTrue(retained)
        self.assertEqual(json.loads(raw_payload)["state"], "OFF")

    def test_recording_and_detection_commands_use_their_callbacks(self) -> None:
        calls: list[tuple[str, str, bool]] = []
        completed = threading.Event()

        def callback(feature: str):
            def apply(camera_id: str, enabled: bool) -> bool:
                calls.append((feature, camera_id, enabled))
                if len(calls) == 2:
                    completed.set()
                return True
            return apply

        service = self.service(callback("recording"), callback("detection"))
        service._on_message(None, None, SimpleNamespace(
            topic="survng/camera/gate/recording/set",
            payload=b"OFF",
        ))
        service._on_message(None, None, SimpleNamespace(
            topic="survng/camera/gate/detection/set",
            payload=b'{"state":"ON"}',
        ))

        self.assertTrue(completed.wait(1))
        self.assertCountEqual(calls, [
            ("recording", "gate", False),
            ("detection", "gate", True),
        ])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def subscribe(self, topic: str, qos: int):
        return (0, 1)

    def disconnect(self) -> None:
        pass

    def loop_stop(self) -> None:
        pass


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
        service._start_command_worker()
        self.addCleanup(service.stop)
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
        service._on_message(service.client, None, SimpleNamespace(
            topic="survng/camera/gate/recording/set",
            payload=b"OFF",
        ))
        service._on_message(service.client, None, SimpleNamespace(
            topic="survng/camera/gate/detection/set",
            payload=b'{"state":"ON"}',
        ))

        self.assertTrue(completed.wait(1))
        self.assertCountEqual(calls, [
            ("recording", "gate", False),
            ("detection", "gate", True),
        ])

    def test_incident_lifecycle_uses_stable_id_and_non_retained_topic(self) -> None:
        service = self.service()
        service.track_incident({
            "id": 41,
            "camera_id": "front-door",
            "created_at": "2026-07-17T12:00:00+00:00",
            "snapshot_path": "/storage/front-door.jpg",
            "objects_json": json.dumps([{
                "label": "person",
                "confidence": 0.82,
                "zones": ["Porch"],
                "incident_eligible": True,
            }]),
        }, "Front Door", "/survng")
        service.track_incident({
            "id": 42,
            "camera_id": "front-door",
            "created_at": "2026-07-17T12:00:10+00:00",
            "snapshot_path": "/storage/front-door-2.jpg",
            "objects_json": json.dumps([{
                "label": "person",
                "confidence": 0.91,
                "zones": ["Porch", "Walkway"],
                "incident_eligible": True,
            }, {
                "label": "dog",
                "confidence": 0.73,
                "zones": ["Walkway"],
                "incident_eligible": True,
            }]),
        }, "Front Door", "/survng")
        service.flush_incidents()

        publications = [
            (topic, json.loads(payload), retained)
            for topic, payload, _qos, retained in service.client.published
            if topic == "survng/events/incidents"
        ]
        self.assertEqual([payload["state"] for _, payload, _ in publications], ["new", "updated", "complete"])
        self.assertEqual({payload["incident_id"] for _, payload, _ in publications}, {"incident-front-door-41"})
        complete = publications[-1][1]
        self.assertEqual(complete["event_ids"], [41, 42])
        self.assertEqual(complete["event_count"], 2)
        self.assertEqual(complete["classes"], ["dog", "person"])
        self.assertEqual(complete["zones"], ["Porch", "Walkway"])
        self.assertEqual(complete["representative_event_id"], 42)
        self.assertEqual(complete["snapshot_url"], "/survng/api/events/42/snapshot.jpg")
        self.assertTrue(all(retained is False for _, _, retained in publications))

    def test_incident_events_can_be_disabled(self) -> None:
        service = self.service()
        service.config.incident_events_enabled = False
        service.track_incident({
            "id": 41,
            "camera_id": "front-door",
            "created_at": "2026-07-17T12:00:00+00:00",
        }, "Front Door")

        self.assertEqual(service.client.published, [])

    def test_manual_update_only_changes_an_incident_that_is_still_pending(self) -> None:
        service = self.service()
        event = {
            "id": 41,
            "camera_id": "front-door",
            "created_at": "2026-07-17T12:00:00+00:00",
            "objects_json": "[]",
        }
        service.track_incident(event, "Front Door")
        service.track_incident({
            **event,
            "objects_json": json.dumps([{"label": "car", "confidence": 0.9}]),
        }, "Front Door", allow_new=False)
        service.flush_incidents()
        complete = json.loads(service.client.published[-1][1])
        self.assertEqual(complete["classes"], ["car"])

        service.track_incident(event, "Front Door", allow_new=False)
        self.assertEqual(len(service.client.published), 3)

    def test_retained_commands_are_rejected_without_replaying_state(self) -> None:
        callback = Mock(return_value=True)
        service = self.service(recording_callback=callback)

        service._on_message(service.client, None, SimpleNamespace(
            topic="survng/camera/gate/recording/set",
            payload=b"OFF",
            retain=True,
        ))

        callback.assert_not_called()
        self.assertEqual(service.commands_received, 0)
        self.assertEqual(service.commands_rejected, 1)

    def test_stale_client_callbacks_are_ignored(self) -> None:
        callback = Mock(return_value=True)
        service = self.service(recording_callback=callback)

        service._on_message(object(), None, SimpleNamespace(
            topic="survng/camera/gate/recording/set",
            payload=b"OFF",
        ))
        service._on_disconnect(object(), None, None, 1, None)

        callback.assert_not_called()
        self.assertTrue(service.connected)

    def test_shutdown_invalidates_client_before_disconnect_callbacks(self) -> None:
        connected = Mock()
        service = self.service()
        service.connected_callback = connected
        client = service.client
        client.disconnect = Mock(side_effect=lambda: service._on_connect(
            client, None, None, 0, None
        ))

        service.stop()

        self.assertFalse(service.connected)
        self.assertIsNone(service.client)
        connected.assert_not_called()

    def test_connect_tracks_command_subscription_health(self) -> None:
        connected = Mock()
        service = self.service()
        service.connected_callback = connected

        service._on_connect(service.client, None, None, 0, None)

        self.assertTrue(service.command_subscriptions_active)
        self.assertEqual(service.subscription_failures, 0)
        connected.assert_called_once_with()

    def test_oversized_command_is_rejected_before_parsing(self) -> None:
        callback = Mock(return_value=True)
        service = self.service(recording_callback=callback)

        service._on_message(service.client, None, SimpleNamespace(
            topic="survng/camera/gate/recording/set",
            payload=b"1" * 5000,
        ))

        callback.assert_not_called()
        self.assertEqual(service.commands_rejected, 1)

    def test_start_failure_releases_client_and_command_worker(self) -> None:
        service = MqttService(
            MqttConfig(enabled=True, host="broker"),
            lambda camera_id, enabled: True,
            lambda camera_id, enabled: True,
            lambda camera_id, enabled: True,
        )
        client = Mock()
        client.connect_async.side_effect = RuntimeError("connect setup failed")
        with patch("paho.mqtt.client.Client", return_value=client):
            service.start()

        self.assertIsNone(service.client)
        self.assertIsNone(service._command_thread)
        client.loop_stop.assert_called_once_with()
        self.assertIn("connect setup failed", service.last_error)

    def test_nonzero_network_loop_result_rolls_back_startup(self) -> None:
        service = MqttService(
            MqttConfig(enabled=True, host="broker"),
            lambda camera_id, enabled: True,
            lambda camera_id, enabled: True,
            lambda camera_id, enabled: True,
        )
        client = Mock()
        client.connect_async.return_value = 0
        client.loop_start.return_value = 3
        with patch("paho.mqtt.client.Client", return_value=client):
            service.start()

        self.assertIsNone(service.client)
        self.assertIsNone(service._command_thread)
        client.loop_stop.assert_called_once_with()
        self.assertIn("network loop failed", service.last_error)


if __name__ == "__main__":
    unittest.main()

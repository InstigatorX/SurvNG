from __future__ import annotations

import unittest
from unittest.mock import Mock, call

from survng.app.config import MqttConfig
from survng.app.mqtt_lifecycle import MqttLifecycle


class MqttLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MqttConfig(enabled=True, host="broker")
        self.previous = Mock()
        self.replacement = Mock()
        self.lifecycle = MqttLifecycle(
            self.config,
            lambda _config: self.replacement,
            service=self.previous,
        )

    def test_stable_publisher_routes_to_replacement_generation(self) -> None:
        publisher = self.lifecycle

        self.lifecycle.reconfigure(self.config, running=True)
        publisher.publish_camera_state("gate", True)

        self.previous.stop.assert_called_once_with(
            lifecycle="restarting",
            require_quiesced=True,
        )
        self.replacement.start.assert_called_once_with(raise_on_failure=True)
        self.replacement.set_server_lifecycle.assert_called_once_with("running")
        self.replacement.publish_camera_state.assert_called_once_with("gate", True)
        self.previous.publish_camera_state.assert_not_called()

    def test_failed_replacement_restores_previous_generation(self) -> None:
        self.replacement.start.side_effect = RuntimeError("start failed")

        with self.assertRaisesRegex(RuntimeError, "start failed"):
            self.lifecycle.reconfigure(self.config, running=True)

        self.assertIs(self.lifecycle.service, self.previous)
        self.replacement.stop.assert_called_once_with(lifecycle="restarting")
        self.previous.start.assert_called_once_with(raise_on_failure=True)
        self.previous.set_server_lifecycle.assert_called_once_with("running")

    def test_unquiesced_previous_generation_prevents_cutover(self) -> None:
        self.previous.stop.side_effect = RuntimeError("workers did not quiesce")

        with self.assertRaisesRegex(RuntimeError, "did not quiesce"):
            self.lifecycle.reconfigure(self.config, running=True)

        self.assertIs(self.lifecycle.service, self.previous)
        self.replacement.start.assert_not_called()

    def test_cutover_guard_is_checked_after_lifecycle_admission(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "manager is stopping"):
            self.lifecycle.reconfigure(
                self.config,
                running=True,
                allowed=lambda: False,
            )

        self.assertIs(self.lifecycle.service, self.previous)
        self.previous.stop.assert_not_called()
        self.replacement.start.assert_not_called()

    def test_inflight_command_can_publish_while_old_generation_drains(self) -> None:
        self.previous.stop.side_effect = lambda **_kwargs: (
            self.lifecycle.publish_camera_feature_state("gate", "recording", False)
        )

        self.lifecycle.reconfigure(self.config, running=False)

        self.previous.publish_camera_feature_state.assert_called_once_with(
            "gate", "recording", False
        )
        self.assertIs(self.lifecycle.service, self.replacement)

    def test_close_is_idempotent_and_terminal(self) -> None:
        self.lifecycle.close()
        self.lifecycle.close()

        self.previous.stop.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.lifecycle.start()

    def test_forwarders_preserve_arguments(self) -> None:
        self.lifecycle.publish("camera/gate/motion", {"active": True}, retain=True)
        self.lifecycle.publish_camera_feature_state("gate", "recording", False)
        self.lifecycle.publish_zone_objects("gate", [], {"objects": []})

        self.assertEqual(
            self.previous.method_calls,
            [
                call.publish("camera/gate/motion", {"active": True}, retain=True),
                call.publish_camera_feature_state("gate", "recording", False),
                call.publish_zone_objects("gate", [], {"objects": []}),
            ],
        )


if __name__ == "__main__":
    unittest.main()

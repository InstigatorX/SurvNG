from __future__ import annotations

import threading
import unittest
from dataclasses import replace
from unittest.mock import Mock

from survng.app.config import AppConfig
from survng.app.manager_reload import ManagerGenerationLifecycle, ManagerReloadHooks


class ManagerGenerationLifecycleTest(unittest.TestCase):
    def lifecycle(self, factory: Mock, hooks: ManagerReloadHooks) -> ManagerGenerationLifecycle:
        return ManagerGenerationLifecycle(
            lock=threading.RLock(),
            stopping=threading.Event(),
            manager_factory=factory,
            hooks=hooks,
        )

    def hooks(self) -> ManagerReloadHooks:
        return ManagerReloadHooks(
            active_storage_tasks=lambda _manager: [],
            active_ai_operations=lambda: {},
            prewarmer_running=lambda: False,
            stop_prewarmer=Mock(),
            start_prewarmer=Mock(),
            save_config=Mock(),
            publish_runtime=Mock(),
            refresh_runtime_caches=Mock(),
            storage_error=RuntimeError,
            ai_error=RuntimeError,
        )

    def test_preference_read_failure_closes_unpublished_candidate(self) -> None:
        previous = Mock()
        previous.runtime_preferences.side_effect = RuntimeError("state unavailable")
        candidate = Mock()
        hooks = self.hooks()
        lifecycle = self.lifecycle(Mock(return_value=candidate), hooks)

        with self.assertRaisesRegex(RuntimeError, "before the active manager was stopped"):
            lifecycle.reload(AppConfig(), previous, AppConfig(base_path="/new"), persist=True)

        candidate.stop_all.assert_called_once_with()
        previous.stop_all_with_runtime_preferences.assert_not_called()
        hooks.publish_runtime.assert_not_called()

    def test_success_publishes_only_after_candidate_is_started_and_persisted(self) -> None:
        order: list[str] = []
        previous = Mock()
        previous.runtime_preferences.return_value = {"camera_enabled": {}}
        previous.stop_all_with_runtime_preferences.side_effect = lambda: order.append("stop-old")
        candidate = Mock()
        candidate.start_all.side_effect = lambda: order.append("start-new")
        hooks = self.hooks()
        hooks.save_config.side_effect = lambda *_args, **_kwargs: order.append("save")
        hooks.publish_runtime.side_effect = lambda *_args: order.append("publish")

        self.lifecycle(Mock(return_value=candidate), hooks).reload(
            AppConfig(), previous, AppConfig(base_path="/new"), persist=True
        )

        self.assertEqual(order, ["stop-old", "start-new", "save", "publish"])
        hooks.refresh_runtime_caches.assert_called_once_with()

    def test_active_generation_requests_drain_before_previous_manager_stops(self) -> None:
        order: list[str] = []
        previous = Mock()
        previous.runtime_preferences.return_value = {"camera_enabled": {}}
        previous.stop_all_with_runtime_preferences.side_effect = lambda: order.append("stop-old")
        candidate = Mock()
        wait_for_manager_idle = Mock(
            side_effect=lambda active: order.append("drain-requests") or active is previous
        )
        hooks = replace(
            self.hooks(),
            wait_for_manager_idle=wait_for_manager_idle,
        )

        self.lifecycle(Mock(return_value=candidate), hooks).reload(
            AppConfig(), previous, AppConfig(base_path="/new"), persist=False
        )

        self.assertEqual(order[:2], ["drain-requests", "stop-old"])
        wait_for_manager_idle.assert_called_once_with(previous)

    def test_reload_refuses_to_stop_manager_when_request_drain_times_out(self) -> None:
        previous = Mock()
        previous.runtime_preferences.return_value = {"camera_enabled": {}}
        candidate = Mock()
        hooks = replace(
            self.hooks(),
            wait_for_manager_idle=Mock(return_value=False),
        )

        with self.assertRaisesRegex(RuntimeError, "before the active manager was stopped"):
            self.lifecycle(Mock(return_value=candidate), hooks).reload(
                AppConfig(), previous, AppConfig(base_path="/new"), persist=False
            )

        previous.stop_all_with_runtime_preferences.assert_not_called()
        candidate.stop_all.assert_called_once_with()

    def test_failed_candidate_cleanup_prevents_overlapping_recovery(self) -> None:
        previous = Mock()
        previous.runtime_preferences.return_value = {}
        candidate = Mock()
        candidate.stop_all.side_effect = RuntimeError("worker still alive")
        recovery = Mock()
        factory = Mock(side_effect=[candidate, recovery])
        hooks = self.hooks()
        hooks.save_config.side_effect = OSError("configuration disk unavailable")
        lifecycle = self.lifecycle(factory, hooks)
        effective = AppConfig(base_path="/new")
        with self.assertRaisesRegex(RuntimeError, "replacement manager.*restart"):
            lifecycle.reload(AppConfig(), previous, effective, persist=True)
        self.assertEqual(factory.call_count, 1)
        recovery.start_all.assert_not_called()
        hooks.publish_runtime.assert_called_once_with(effective, candidate)
        self.assertTrue(lifecycle._stopping.is_set())

    def test_failed_recovery_is_closed(self) -> None:
        previous = Mock()
        previous.runtime_preferences.return_value = {}
        candidate = Mock()
        candidate.start_all.side_effect = RuntimeError("candidate failed")
        recovery = Mock()
        recovery.start_all.side_effect = RuntimeError("recovery failed")
        hooks = self.hooks()
        lifecycle = self.lifecycle(Mock(side_effect=[candidate, recovery]), hooks)
        with self.assertRaisesRegex(RuntimeError, "could not be restored"):
            lifecycle.reload(AppConfig(), previous, AppConfig(), persist=False)
        recovery.stop_all.assert_called_once_with()
        self.assertTrue(lifecycle._stopping.is_set())

    def test_previous_shutdown_failure_keeps_previous_owner(self) -> None:
        previous = Mock()
        previous.runtime_preferences.return_value = {}
        previous.stop_all_with_runtime_preferences.side_effect = RuntimeError("inference still alive")
        candidate = Mock()
        factory = Mock(return_value=candidate)
        hooks = self.hooks()
        lifecycle = self.lifecycle(factory, hooks)
        original = AppConfig()
        with self.assertRaisesRegex(RuntimeError, "previous manager shutdown"):
            lifecycle.reload(original, previous, AppConfig(), persist=False)
        candidate.start_all.assert_not_called()
        hooks.publish_runtime.assert_called_once_with(original, previous)
        self.assertEqual(factory.call_count, 1)
        self.assertTrue(lifecycle._stopping.is_set())

    def test_failed_recovery_cleanup_preserves_owner_for_final_shutdown(self) -> None:
        previous = Mock()
        previous.runtime_preferences.return_value = {}
        candidate = Mock()
        candidate.start_all.side_effect = RuntimeError("candidate failed")
        recovery = Mock()
        recovery.start_all.side_effect = RuntimeError("recovery failed")
        recovery.stop_all.side_effect = RuntimeError("recovery workers still alive")
        hooks = self.hooks()
        lifecycle = self.lifecycle(Mock(side_effect=[candidate, recovery]), hooks)
        original = AppConfig()
        with self.assertRaisesRegex(RuntimeError, "could not be restored"):
            lifecycle.reload(original, previous, AppConfig(), persist=False)
        hooks.publish_runtime.assert_called_once_with(original, recovery)
        self.assertTrue(lifecycle._stopping.is_set())


if __name__ == "__main__":
    unittest.main()

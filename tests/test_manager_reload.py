from __future__ import annotations

import threading
import unittest
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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import threading
import unittest
from dataclasses import replace
from unittest.mock import Mock

from survng.app.config import AppConfig
from survng.app.manager_reload import (
    ManagerGenerationLifecycle,
    ManagerReloadHooks,
    ManagerReloadRestartPending,
)


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

    def test_success_persists_first_and_publishes_only_after_candidate_is_started(self) -> None:
        order: list[str] = []
        previous = Mock()
        previous.runtime_preferences.return_value = {"camera_enabled": {}}
        previous.stop_all_with_runtime_preferences.side_effect = lambda: order.append("stop-old")
        candidate = Mock()
        candidate.start_all.side_effect = lambda: order.append("start-new")
        hooks = replace(
            self.hooks(),
            record_change=Mock(side_effect=lambda *_args: order.append("record")),
        )
        hooks.save_config.side_effect = lambda *_args, **_kwargs: order.append("save")
        hooks.publish_runtime.side_effect = lambda *_args: order.append("publish")

        self.lifecycle(Mock(return_value=candidate), hooks).reload(
            AppConfig(), previous, AppConfig(base_path="/new"), persist=True
        )

        self.assertEqual(order, ["save", "stop-old", "start-new", "record", "publish"])
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
        candidate.start_all.side_effect = RuntimeError("camera failed to start")
        candidate.stop_all.side_effect = RuntimeError("worker still alive")
        recovery = Mock()
        factory = Mock(side_effect=[candidate, recovery])
        hooks = self.hooks()
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

    def _dead_end(self, kind: str) -> tuple[Mock, Mock, AppConfig]:
        """A reload that cannot recover in-process; returns (previous, factory, config)."""
        previous = Mock()
        previous.runtime_preferences.return_value = {}
        candidate = Mock()
        recovery = Mock()
        if kind == "previous_shutdown":
            previous.stop_all_with_runtime_preferences.side_effect = RuntimeError("alive")
            return previous, Mock(return_value=candidate), AppConfig()
        if kind == "candidate_cleanup":
            candidate.start_all.side_effect = RuntimeError("candidate failed")
            candidate.stop_all.side_effect = RuntimeError("worker still alive")
            return previous, Mock(side_effect=[candidate, recovery]), AppConfig()
        candidate.start_all.side_effect = RuntimeError("candidate failed")
        recovery.start_all.side_effect = RuntimeError("recovery failed")
        return previous, Mock(side_effect=[candidate, recovery]), AppConfig()

    def test_unrecoverable_reload_requests_a_supervisor_restart_after_publishing(self) -> None:
        for kind, message in (
            ("previous_shutdown", "previous manager shutdown is still active"),
            ("candidate_cleanup", "replacement manager cleanup is incomplete"),
            ("recovery", "previous manager could not be restored"),
        ):
            with self.subTest(kind=kind):
                order: list[str] = []
                hooks = replace(
                    self.hooks(),
                    publish_runtime=Mock(side_effect=lambda *_args: order.append("publish")),
                    request_supervisor_restart=Mock(
                        side_effect=lambda reason: order.append("restart") or True
                    ),
                )
                previous, factory, original = self._dead_end(kind)
                lifecycle = self.lifecycle(factory, hooks)

                with self.assertRaisesRegex(
                    RuntimeError, f"{message}; SurvNG is restarting through its supervisor"
                ):
                    lifecycle.reload(original, previous, AppConfig(base_path="/new"), persist=False)

                hooks.request_supervisor_restart.assert_called_once()
                self.assertIn(message, hooks.request_supervisor_restart.call_args.args[0])
                self.assertTrue(lifecycle._stopping.is_set())
                if "publish" in order:
                    self.assertLess(order.index("publish"), order.index("restart"))

    def test_a_restart_keeps_the_saved_change_only_when_it_was_never_tried(self) -> None:
        for kind, kept in (
            # The old generation would not stop: the new config never ran,
            # so the restart applies it.
            ("previous_shutdown", True),
            # The new config ran and failed: what is on disk is the old one.
            ("candidate_cleanup", False),
            ("recovery", False),
        ):
            with self.subTest(kind=kind):
                hooks = replace(
                    self.hooks(),
                    request_supervisor_restart=Mock(return_value=True),
                    record_change=Mock(),
                )
                previous, factory, original = self._dead_end(kind)
                requested = AppConfig(base_path="/new")
                with self.assertRaises(ManagerReloadRestartPending) as raised:
                    self.lifecycle(factory, hooks).reload(original, previous, requested, persist=True)
                self.assertIs(raised.exception.persisted, kept)
                self.assertTrue(raised.exception.restarting)
                saved = [call.args[0] for call in hooks.save_config.call_args_list]
                self.assertEqual(saved, [requested] if kept else [requested, original])
                # History shows only a change that runs, now or after the restart.
                if kept:
                    hooks.record_change.assert_called_once_with(original, requested)
                else:
                    hooks.record_change.assert_not_called()
                if kept:
                    self.assertIn("to apply the saved configuration", str(raised.exception))

    def test_without_a_managed_server_the_error_asks_for_a_manual_restart(self) -> None:
        hooks = replace(self.hooks(), request_supervisor_restart=Mock(return_value=False))
        previous, factory, original = self._dead_end("previous_shutdown")
        with self.assertRaisesRegex(RuntimeError, "; restart SurvNG through its supervisor"):
            self.lifecycle(factory, hooks).reload(original, previous, AppConfig(), persist=False)

    def test_a_failing_restart_request_does_not_hide_the_reload_failure(self) -> None:
        hooks = replace(
            self.hooks(), request_supervisor_restart=Mock(side_effect=RuntimeError("no server")),
        )
        previous, factory, original = self._dead_end("previous_shutdown")
        with self.assertRaisesRegex(RuntimeError, "previous manager shutdown is still active"):
            self.lifecycle(factory, hooks).reload(original, previous, AppConfig(), persist=False)

    def test_recoverable_reload_failures_do_not_restart_the_process(self) -> None:
        previous = Mock()
        previous.runtime_preferences.return_value = {}
        candidate = Mock()
        candidate.start_all.side_effect = RuntimeError("candidate failed")
        hooks = replace(self.hooks(), request_supervisor_restart=Mock(return_value=True))
        lifecycle = self.lifecycle(Mock(side_effect=[candidate, Mock()]), hooks)

        with self.assertRaisesRegex(RuntimeError, "previous configuration was restored"):
            lifecycle.reload(AppConfig(), previous, AppConfig(base_path="/new"), persist=False)

        hooks.request_supervisor_restart.assert_not_called()
        self.assertFalse(lifecycle._stopping.is_set())


if __name__ == "__main__":
    unittest.main()

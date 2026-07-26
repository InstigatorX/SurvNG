from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from survng.app.config import AppConfig
from survng.app import main


class ConfigReloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_config = main.config
        self.previous_manager = main.manager

    def tearDown(self) -> None:
        main.config = self.previous_config
        main.manager = self.previous_manager

    def test_failed_replacement_restores_previous_manager_without_persisting(self) -> None:
        active = Mock()
        active.runtime_preferences.return_value = {
            "recording_enabled": {"gate": False},
            "detection_enabled": {"gate": True},
            "camera_enabled": {"gate": True},
        }
        active.stop_all_with_runtime_preferences.return_value = {
            "recording_enabled": {},
            "detection_enabled": {},
        }
        candidate = Mock()
        candidate.start_all.side_effect = RuntimeError("startup failed")
        recovery = Mock()
        main.config = AppConfig(base_path="/old")
        main.manager = active

        with (
            patch("survng.app.main.AppManager", side_effect=[candidate, recovery]),
            patch("survng.app.main._stop_recording_prewarmer"),
            patch("survng.app.main._start_recording_prewarmer"),
            patch("survng.app.main.save_config") as save,
        ):
            with self.assertRaisesRegex(RuntimeError, "previous configuration was restored"):
                main.reload_manager(AppConfig(base_path="/new"))

        active.stop_all_with_runtime_preferences.assert_called_once_with()
        candidate.stop_all.assert_called_once_with()
        recovery.start_all.assert_called_once_with()
        recovery.apply_runtime_preferences.assert_called_once_with(
            active.runtime_preferences.return_value,
            persist=True,
        )
        save.assert_not_called()
        self.assertIs(main.manager, recovery)
        self.assertEqual(main.config.base_path, "/old")

    def test_successful_replacement_starts_before_atomic_persistence_and_swap(self) -> None:
        actions: list[str] = []
        active = Mock()
        active.runtime_preferences.return_value = {
            "recording_enabled": {},
            "detection_enabled": {},
            "camera_enabled": {},
        }
        active.stop_all_with_runtime_preferences.side_effect = lambda: (
            actions.append("old-stop")
            or {"recording_enabled": {}, "detection_enabled": {}}
        )
        candidate = Mock()
        candidate.start_all.side_effect = lambda: actions.append("new-start")
        main.config = AppConfig(base_path="/old")
        main.manager = active

        with (
            patch("survng.app.main.AppManager", return_value=candidate),
            patch("survng.app.main._stop_recording_prewarmer"),
            patch("survng.app.main._start_recording_prewarmer"),
            patch(
                "survng.app.main.save_config",
                side_effect=lambda *_args, **_kwargs: actions.append("save"),
            ),
        ):
            effective = main.reload_manager(AppConfig(base_path="/new"))

        self.assertEqual(actions, ["old-stop", "new-start", "save"])
        self.assertIs(main.manager, candidate)
        self.assertIs(main.config, effective)
        self.assertEqual(main.config.base_path, "/new")

    def test_persistence_failure_stops_candidate_and_restores_previous_manager(self) -> None:
        active = Mock()
        active.runtime_preferences.return_value = {
            "recording_enabled": {"removed": False},
            "detection_enabled": {"removed": False},
            "camera_enabled": {"removed": False},
        }
        active.stop_all_with_runtime_preferences.return_value = {
            "recording_enabled": {},
            "detection_enabled": {},
        }
        candidate = Mock()
        recovery = Mock()
        main.config = AppConfig(base_path="/old")
        main.manager = active

        with (
            patch("survng.app.main.AppManager", side_effect=[candidate, recovery]),
            patch("survng.app.main._stop_recording_prewarmer"),
            patch("survng.app.main._start_recording_prewarmer"),
            patch("survng.app.main.save_config", side_effect=OSError("disk full")),
        ):
            with self.assertRaisesRegex(RuntimeError, "previous configuration was restored"):
                main.reload_manager(AppConfig(base_path="/new"))

        candidate.start_all.assert_called_once_with()
        candidate.stop_all.assert_called_once_with()
        recovery.start_all.assert_called_once_with()
        recovery.apply_runtime_preferences.assert_called_once_with(
            active.runtime_preferences.return_value,
            persist=True,
        )
        self.assertIs(main.manager, recovery)
        self.assertEqual(main.config.base_path, "/old")

    def test_start_does_not_revive_a_prewarmer_that_is_still_stopping(self) -> None:
        original_thread = main.RECORDING_PREWARM_THREAD
        original_stop_state = main.RECORDING_PREWARM_STOP.is_set()
        active = Mock()
        active.is_alive.return_value = True
        main.RECORDING_PREWARM_THREAD = active
        main.RECORDING_PREWARM_STOP.set()
        try:
            main._start_recording_prewarmer()
            self.assertTrue(main.RECORDING_PREWARM_STOP.is_set())
            active.start.assert_not_called()
        finally:
            main.RECORDING_PREWARM_THREAD = original_thread
            if original_stop_state:
                main.RECORDING_PREWARM_STOP.set()
            else:
                main.RECORDING_PREWARM_STOP.clear()

    def test_stop_reports_a_prewarmer_that_cannot_be_reaped(self) -> None:
        original_thread = main.RECORDING_PREWARM_THREAD
        original_stop_state = main.RECORDING_PREWARM_STOP.is_set()
        active = Mock()
        active.is_alive.return_value = True
        main.RECORDING_PREWARM_THREAD = active
        try:
            with patch("survng.app.main.RECORDING_PREWARM_PROCESS", None):
                with self.assertRaisesRegex(RuntimeError, "did not stop"):
                    main._stop_recording_prewarmer()
        finally:
            main.RECORDING_PREWARM_THREAD = original_thread
            if original_stop_state:
                main.RECORDING_PREWARM_STOP.set()
            else:
                main.RECORDING_PREWARM_STOP.clear()


if __name__ == "__main__":
    unittest.main()

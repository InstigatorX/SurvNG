from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from survng.app.config import AppConfig, CameraConfig
from survng.app import main


class ConfigReloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_config = main.config
        self.previous_manager = main.manager
        main.APPLICATION_STOPPING.clear()

    def tearDown(self) -> None:
        main.APPLICATION_STOPPING.clear()
        main.config = self.previous_config
        main.manager = self.previous_manager

    def test_manager_reload_is_refused_during_shutdown(self) -> None:
        active = Mock()
        main.config = AppConfig(base_path="/old")
        main.manager = active
        main.APPLICATION_STOPPING.set()

        with patch("survng.app.main.AppManager") as manager_factory:
            with self.assertRaisesRegex(RuntimeError, "shutting down"):
                main.reload_manager(AppConfig(base_path="/new"))

        manager_factory.assert_not_called()
        active.stop_all_with_runtime_preferences.assert_not_called()

    def test_manager_reload_is_refused_while_ai_uses_active_manager(self) -> None:
        active = Mock()
        main.config = AppConfig(base_path="/old")
        main.manager = active

        with (
            patch.object(main, "_active_ai_operations", return_value={"assistant": 1}),
            patch("survng.app.main.AppManager") as manager_factory,
            self.assertRaises(main.AiOperationsActiveError),
        ):
            main.reload_manager(AppConfig(base_path="/new"))

        manager_factory.assert_not_called()
        active.stop_all_with_runtime_preferences.assert_not_called()

    def test_manager_reload_is_refused_without_stopping_active_storage_work(self) -> None:
        active = Mock()
        active.recorder.retention_status.return_value = {"state": "idle"}
        main.config = AppConfig(base_path="/old")
        main.manager = active

        with (
            patch.object(
                main.STORAGE_MAINTENANCE,
                "status",
                return_value={"status": "running", "mode": "repair"},
            ),
            patch.object(main.STORAGE_MAINTENANCE, "stop") as stop,
            patch("survng.app.main.AppManager") as manager_factory,
        ):
            with self.assertRaisesRegex(main.StorageTasksActiveError, "storage repair"):
                main.reload_manager(AppConfig(base_path="/new"))

        stop.assert_not_called()
        manager_factory.assert_not_called()
        active.stop_all_with_runtime_preferences.assert_not_called()
        self.assertIs(main.manager, active)
        self.assertEqual(main.config.base_path, "/old")

    def test_bounded_retention_cleanup_does_not_block_manager_reload(self) -> None:
        active = Mock()
        active.recorder.retention_status.return_value = {"state": "cleaning"}

        with patch.object(
            main.STORAGE_MAINTENANCE,
            "status",
            return_value={"status": "idle"},
        ):
            tasks = main._active_storage_tasks(active)

        self.assertEqual(tasks, [])

    def test_active_media_export_blocks_manager_reload(self) -> None:
        active = Mock()
        exports = Mock()
        exports.active_jobs.return_value = [{"kind": "timelapse", "status": "running"}]

        with (
            patch.object(main.STORAGE_MAINTENANCE, "status", return_value={"status": "idle"}),
            patch.object(main, "MEDIA_EXPORTS", exports),
        ):
            tasks = main._active_storage_tasks(active)

        self.assertEqual(tasks, ["media timelapse export"])

    def test_active_media_export_blocks_ffmpeg_hot_reconfiguration(self) -> None:
        active = Mock()
        exports = Mock()
        exports.active_jobs.return_value = [{"kind": "recording", "status": "running"}]
        main.config = AppConfig(ffmpeg_path="/old/ffmpeg")
        main.manager = active

        with (
            patch.object(main, "MEDIA_EXPORTS", exports),
            patch.object(main, "save_config") as save,
            self.assertRaisesRegex(main.StorageTasksActiveError, "media recording export"),
        ):
            main.apply_config_update(AppConfig(ffmpeg_path="/new/ffmpeg"))

        save.assert_not_called()
        active.reconfigure_recorders.assert_not_called()
        self.assertEqual(main.config.ffmpeg_path, "/old/ffmpeg")

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

    def test_app_only_settings_hot_apply_without_restarting_cameras(self) -> None:
        active = Mock()
        current = AppConfig(base_path="/old", recording_cache_max_gb=5)
        active.config = current
        main.config = current
        main.manager = active
        incoming = current.model_copy(update={
            "base_path": "/new",
            "recording_cache_max_gb": 10,
            "event_clip_before_seconds": 8,
        })

        with (
            patch("survng.app.main.reload_manager") as reload,
            patch("survng.app.main.save_config") as save,
        ):
            effective, result = main.apply_config_update(incoming)

        reload.assert_not_called()
        save.assert_called_once_with(effective, assign_ids=False)
        active.reconfigure_mqtt.assert_not_called()
        active.recorder.reconfigure_retention.assert_not_called()
        active.reconfigure_image_storage.assert_not_called()
        self.assertEqual(result["apply_mode"], "hot")
        self.assertFalse(result["camera_workers_restarted"])
        self.assertIs(main.config, effective)
        self.assertIs(active.config, effective)

    def test_image_storage_change_hot_applies_without_restarting_cameras(self) -> None:
        active = Mock()
        current = AppConfig()
        active.config = current
        main.config = current
        main.manager = active
        incoming = current.model_copy(deep=True)
        incoming.image_storage.quality = 90

        with (
            patch("survng.app.main.reload_manager") as reload,
            patch("survng.app.main.save_config"),
        ):
            effective, result = main.apply_config_update(incoming)

        reload.assert_not_called()
        active.reconfigure_image_storage.assert_called_once_with(effective.image_storage)
        active.reconfigure_mqtt.assert_not_called()
        active.recorder.reconfigure_retention.assert_not_called()
        self.assertEqual(result["apply_mode"], "hot")
        self.assertFalse(result["camera_workers_restarted"])

    def test_mqtt_change_restarts_only_mqtt(self) -> None:
        active = Mock()
        current = AppConfig()
        active.config = current
        main.config = current
        main.manager = active
        incoming = current.model_copy(deep=True)
        incoming.mqtt.host = "broker.local"
        incoming.mqtt.enabled = True

        with (
            patch("survng.app.main.reload_manager") as reload,
            patch("survng.app.main.save_config"),
        ):
            _effective, result = main.apply_config_update(incoming)

        reload.assert_not_called()
        active.reconfigure_mqtt.assert_called_once_with(incoming.mqtt)
        active.recorder.reconfigure_retention.assert_not_called()
        self.assertEqual(result["subsystems_restarted"], ["mqtt"])
        self.assertFalse(result["camera_workers_restarted"])

    def test_camera_retention_override_hot_applies_only_retention(self) -> None:
        active = Mock()
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
        current = AppConfig(cameras=[camera])
        active.config = current
        main.config = current
        main.manager = active
        incoming = current.model_copy(deep=True)
        incoming.cameras[0].retention.main_days = 14

        with (
            patch("survng.app.main.reload_manager") as reload,
            patch("survng.app.main.save_config"),
        ):
            effective, result = main.apply_config_update(incoming)

        reload.assert_not_called()
        active.reconfigure_mqtt.assert_not_called()
        active.recorder.reconfigure_retention.assert_called_once_with(
            effective.retention,
            effective.cameras,
        )
        self.assertIn("retention", result["hot_updated"])
        self.assertFalse(result["camera_workers_restarted"])

    def test_camera_owned_change_uses_full_manager_reload(self) -> None:
        active = Mock()
        current = AppConfig()
        active.config = current
        main.config = current
        main.manager = active
        replacement = current.model_copy(update={"storage_dir": "/new/storage"})
        effective = replacement.model_copy(deep=True)

        with patch("survng.app.main.reload_manager", return_value=effective) as reload:
            result_config, result = main.apply_config_update(replacement)

        reload.assert_called_once_with(effective, assign_ids=False, persist=True)
        self.assertIs(result_config, effective)
        self.assertEqual(result["apply_mode"], "manager_reload")
        self.assertTrue(result["camera_workers_restarted"])

    def test_recorder_setting_restarts_only_recorders(self) -> None:
        active = Mock()
        current = AppConfig(recording_segment_seconds=10)
        active.config = current
        main.config = current
        main.manager = active
        incoming = current.model_copy(update={"recording_segment_seconds": 30})

        with (
            patch("survng.app.main.reload_manager") as reload,
            patch("survng.app.main.save_config"),
        ):
            effective, result = main.apply_config_update(incoming)

        reload.assert_not_called()
        active.reconfigure_recorders.assert_called_once_with(effective)
        active.reconfigure_mqtt.assert_not_called()
        self.assertEqual(result["subsystems_restarted"], ["recorders"])
        self.assertFalse(result["camera_workers_restarted"])

    def test_hot_apply_persistence_failure_rolls_runtime_back(self) -> None:
        active = Mock()
        current = AppConfig()
        active.config = current
        main.config = current
        main.manager = active
        incoming = current.model_copy(deep=True)
        incoming.retention.main_days = 3
        incoming.mqtt.host = "new-broker.local"
        active.reconfigure_mqtt.side_effect = [RuntimeError("mqtt failed"), None]

        with patch("survng.app.main.save_config") as save:
            with self.assertRaisesRegex(RuntimeError, "mqtt failed"):
                main.apply_config_update(incoming)

        self.assertEqual(active.reconfigure_mqtt.call_args_list, [
            unittest.mock.call(incoming.mqtt),
            unittest.mock.call(current.mqtt),
        ])
        active.recorder.reconfigure_retention.assert_not_called()
        self.assertEqual(save.call_count, 2)
        self.assertEqual(save.call_args_list[0].args[0], incoming)
        self.assertIs(save.call_args_list[1].args[0], current)
        self.assertIs(main.config, current)
        self.assertIs(active.config, current)

    def test_hot_apply_persistence_failure_does_not_touch_runtime(self) -> None:
        active = Mock()
        current = AppConfig()
        active.config = current
        main.config = current
        main.manager = active
        incoming = current.model_copy(deep=True)
        incoming.mqtt.host = "new-broker.local"

        with patch("survng.app.main.save_config", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                main.apply_config_update(incoming)

        active.reconfigure_mqtt.assert_not_called()
        active.recorder.reconfigure_retention.assert_not_called()
        self.assertIs(main.config, current)
        self.assertIs(active.config, current)

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

from __future__ import annotations

import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from survng.app.config import AppConfig, CameraConfig
from survng.app.manager import AppManager


def manager_with_mocks() -> AppManager:
    manager = object.__new__(AppManager)
    camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
    manager.config = AppConfig(cameras=[camera])
    manager._lifecycle_lock = threading.RLock()
    manager._stopping = False
    manager._started = False
    manager._closed = False
    manager._camera_enabled = {"gate": True}
    manager._recording_enabled = {}
    manager._detection_enabled = {}
    manager.detector = Mock()
    manager.faces = Mock()
    manager.recorder = Mock()
    manager.mqtt = Mock()
    manager.state_events = Mock()
    manager.workers = {"gate": Mock()}
    manager._start_state_monitor = Mock()
    manager._stop_state_monitor = Mock()
    manager._save_runtime_state = Mock()
    return manager


class ManagerLifecycleTest(unittest.TestCase):
    def test_camera_state_fingerprint_includes_trigger_health_changes(self) -> None:
        status = {
            "id": "gate",
            "onvif_connected": True,
            "onvif_notifications_received": 10,
            "onvif_motion_events_received": 2,
            "onvif_renewals": 1,
            "motion_qualification": {},
        }
        original = AppManager._camera_state_fingerprint(status)

        self.assertNotEqual(
            original,
            AppManager._camera_state_fingerprint({
                **status,
                "onvif_motion_events_received": 3,
            }),
        )
        self.assertNotEqual(
            original,
            AppManager._camera_state_fingerprint({
                **status,
                "onvif_renewal_errors": 1,
            }),
        )

    def test_real_empty_manager_starts_and_stops_all_background_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AppManager(AppConfig(storage_dir=tmpdir))
            manager.start_all()
            self.assertTrue(manager._started)
            self.assertTrue(manager._state_monitor_thread.is_alive())

            manager.stop_all()

        self.assertTrue(manager._closed)
        self.assertIsNone(manager._state_monitor_thread)
        self.assertIsNone(manager.recorder._index_thread)
        self.assertIsNone(manager.recorder._watchdog_thread)

    def test_manager_keeps_databases_and_onvif_caches_outside_media_storage(self) -> None:
        with tempfile.TemporaryDirectory() as storage, tempfile.TemporaryDirectory() as database:
            camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
            manager = AppManager(AppConfig(
                storage_dir=storage,
                database_dir=database,
                cameras=[camera],
            ))

            self.assertEqual(manager.events.db_path, Path(database) / "survng.sqlite3")
            self.assertEqual(manager.faces.db_path, Path(database) / "survng.sqlite3")
            self.assertEqual(manager.database_dir, Path(database))
            self.assertEqual(manager.workers["gate"].onvif._cache_dir, Path(database) / "onvif")

            manager.stop_all()

    def test_startup_failure_rolls_back_every_started_subsystem(self) -> None:
        manager = manager_with_mocks()
        manager.workers["gate"].start.side_effect = RuntimeError("capture failed")

        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            manager.start_all()

        manager.mqtt.stop.assert_called_once_with()
        manager.workers["gate"].stop.assert_called_once_with()
        manager.workers["gate"].close.assert_called_once_with()
        manager.faces.close.assert_called_once_with()
        manager.detector.stop.assert_called_once_with()
        manager.recorder.stop_all.assert_called_once_with()
        manager.state_events.close.assert_called_once_with()
        self.assertTrue(manager._closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            manager.start_all()

    def test_shutdown_continues_after_one_cleanup_failure(self) -> None:
        manager = manager_with_mocks()
        manager._started = True
        manager.faces.close.side_effect = RuntimeError("face close failed")

        with self.assertRaisesRegex(RuntimeError, "face recognition"):
            manager.stop_all()

        manager.detector.stop.assert_called_once_with()
        manager.recorder.stop_all.assert_called_once_with()
        manager.state_events.close.assert_called_once_with()
        self.assertTrue(manager._closed)

    def test_start_is_idempotent_and_stop_is_terminal(self) -> None:
        manager = manager_with_mocks()

        manager.start_all()
        manager.start_all()
        manager.stop_all()
        manager.stop_all()

        manager.detector.start.assert_called_once_with()
        manager.workers["gate"].start.assert_called_once_with()
        manager.workers["gate"].stop.assert_called_once_with()
        manager.detector.stop.assert_called_once_with()

    def test_runtime_preferences_are_filtered_for_a_replacement_manager(self) -> None:
        manager = manager_with_mocks()

        manager.apply_runtime_preferences(
            {
                "recording_enabled": {"gate": False, "removed": False},
                "detection_enabled": {"gate": True, "removed": True},
            }
        )

        self.assertEqual(
            manager.runtime_preferences(),
            {
                "recording_enabled": {"gate": False},
                "detection_enabled": {"gate": True},
                "camera_enabled": {"gate": True},
            },
        )

    def test_invalid_runtime_state_values_do_not_enable_features(self) -> None:
        self.assertEqual(
            AppManager._boolean_preferences(
                {"recording_enabled": {"gate": "false", "foyer": False}},
                "recording_enabled",
            ),
            {"foyer": False},
        )

    def test_runtime_preference_write_failure_rolls_back_memory(self) -> None:
        manager = manager_with_mocks()
        manager._recording_enabled = {"gate": True}
        manager._save_runtime_state.side_effect = OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            manager.set_recording("gate", False)

        self.assertEqual(manager._recording_enabled, {"gate": True})
        manager.recorder.set_camera_enabled.assert_not_called()

    def test_disabled_camera_remains_stopped_during_manager_start(self) -> None:
        manager = manager_with_mocks()
        manager._camera_enabled["gate"] = False

        manager.start_all()

        manager.workers["gate"].start.assert_not_called()
        manager.recorder.set_camera_enabled.assert_called_with("gate", False)
        manager.mqtt.publish_camera_state.assert_called_with("gate", False)
        manager.stop_all()

    def test_failed_camera_start_restores_previous_recorder_state(self) -> None:
        manager = manager_with_mocks()
        manager.workers["gate"].start.side_effect = RuntimeError("capture failed")

        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            manager.start_camera("gate")

        self.assertTrue(manager._camera_enabled["gate"])
        self.assertEqual(
            manager.recorder.set_camera_enabled.call_args_list[-1].args,
            ("gate", True),
        )


if __name__ == "__main__":
    unittest.main()

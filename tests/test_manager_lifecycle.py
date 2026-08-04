from __future__ import annotations

import json
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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
    def test_mqtt_server_health_accepts_loaded_isolated_detector(self) -> None:
        manager = manager_with_mocks()
        manager.config = AppConfig(
            cameras=manager.config.cameras,
            detector={"enabled": True, "device": "GPU"},
        )
        manager._started = True
        manager._process_started_monotonic = 0.0
        manager._process_started_at = "2026-07-31T12:00:00+00:00"
        manager.statuses = Mock(return_value=[{
            "id": "gate",
            "running": True,
            "recording": True,
            "sub_recording": False,
        }])
        manager.detector_status = Mock(return_value={
            "enabled": True,
            "loaded_backend": "openvino",
            "configured_device": "GPU",
            "openvino_loaded": True,
            "runtime": {"queue_depth": 0},
            "isolation": {"enabled": True, "worker_alive": True},
        })
        manager.recorder.retention_status.return_value = {"state": "idle"}

        server = manager._mqtt_server_status()

        self.assertEqual(server["state"]["health"], "ok")
        self.assertEqual(server["metrics"]["detector_state"], "ready")
        self.assertEqual(server["metrics"]["detector_device"], "GPU")

    def test_mqtt_server_health_faults_when_isolated_detector_worker_is_dead(self) -> None:
        status = {
            "enabled": True,
            "loaded_backend": "openvino",
            "openvino_loaded": True,
            "isolation": {"enabled": True, "worker_alive": False},
        }

        self.assertFalse(AppManager._detector_runtime_ready(status))

    def test_mqtt_reconfiguration_does_not_touch_camera_workers_or_recorders(self) -> None:
        manager = manager_with_mocks()
        previous = manager.mqtt
        replacement = Mock()

        with patch("survng.app.manager.MqttService", return_value=replacement) as service:
            manager.reconfigure_mqtt(manager.config.mqtt)

        previous.stop.assert_called_once_with(lifecycle="restarting")
        replacement.start.assert_called_once_with()
        service.assert_called_once()
        manager.workers["gate"].stop.assert_not_called()
        manager.recorder.stop_all.assert_not_called()

    def test_failed_mqtt_reconfiguration_restores_previous_runtime(self) -> None:
        manager = manager_with_mocks()
        previous = manager.mqtt
        replacement = Mock()
        replacement.start.side_effect = RuntimeError("mqtt start failed")

        with patch("survng.app.manager.MqttService", return_value=replacement):
            with self.assertRaisesRegex(RuntimeError, "mqtt start failed"):
                manager.reconfigure_mqtt(manager.config.mqtt)

        self.assertIs(manager.mqtt, previous)
        replacement.stop.assert_called_once_with(lifecycle="restarting")
        previous.start.assert_called_once_with()
        manager.workers["gate"].stop.assert_not_called()

    def test_recorder_reconfiguration_keeps_camera_capture_running(self) -> None:
        manager = manager_with_mocks()
        manager._started = True
        next_config = manager.config.model_copy(update={"recording_segment_seconds": 30})

        manager.reconfigure_recorders(next_config)

        self.assertEqual(manager.recorder.set_camera_enabled.call_args_list, [
            unittest.mock.call("gate", False),
            unittest.mock.call("gate", True),
        ])
        manager.recorder.reconfigure_runtime.assert_called_once_with(
            ffmpeg_path=next_config.ffmpeg_path,
            hardware_acceleration=next_config.hardware_acceleration,
            segment_seconds=30,
        )
        manager.recorder.start.assert_called_once_with(manager.config.cameras[0], "main")
        manager.workers["gate"].stop.assert_not_called()
        manager.workers["gate"].start.assert_not_called()

    def test_tracking_reconfiguration_keeps_camera_capture_and_recorders_running(self) -> None:
        manager = manager_with_mocks()
        manager.events = Mock()
        manager.publish_event = Mock()
        manager.person_reidentifier = Mock()
        manager.face_recognizer = Mock()
        manager.appearance_index = Mock()
        manager.object_tracking_session_factory = Mock()
        manager._object_tracking_limiter = threading.BoundedSemaphore(2)
        manager.detector.config = manager.config.detector
        replacement = Mock()
        previous = Mock()
        worker = manager.workers["gate"]
        worker.create_object_tracking_session.return_value = replacement
        worker.replace_object_tracking_session.return_value = previous
        next_detector = manager.config.detector.model_copy(deep=True)
        next_detector.tracking.sample_fps = 3.0

        with patch("survng.app.manager.ObjectTrackingSessionFactory") as factory_type:
            factory_type.return_value = Mock()
            manager.reconfigure_object_tracking(next_detector)

        worker.create_object_tracking_session.assert_called_once_with(
            factory_type.return_value
        )
        worker.replace_object_tracking_session.assert_called_once_with(replacement)
        manager.detector.update_runtime_config.assert_called_once_with(next_detector)
        worker.stop.assert_not_called()
        worker.start.assert_not_called()
        manager.recorder.stop_all.assert_not_called()

    def test_inference_reconfiguration_keeps_camera_capture_running(self) -> None:
        manager = manager_with_mocks()
        manager.detector.config = manager.config.detector
        manager.face_recognizer = Mock()
        manager.person_reidentifier = Mock()
        next_detector = manager.config.detector.model_copy(deep=True)
        next_detector.device = "GPU"

        manager.reconfigure_inference(next_detector, {"object"})

        manager.detector.reconfigure_roles.assert_called_once_with(
            next_detector,
            {"object"},
        )
        manager.faces.close.assert_not_called()
        manager.workers["gate"].stop.assert_not_called()
        manager.workers["gate"].start.assert_not_called()
        manager.recorder.stop_all.assert_not_called()

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
        self.assertNotEqual(
            original,
            AppManager._camera_state_fingerprint({
                **status,
                "stream_dimensions": {"live": {"width": 896, "height": 672}},
            }),
        )

    def test_real_empty_manager_starts_and_stops_all_background_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AppManager(AppConfig(storage_dir=tmpdir))
            manager.start_all()
            self.assertTrue(manager._started)
            self.assertTrue(manager._state_monitor_thread.is_alive())
            server = manager._mqtt_server_status()
            self.assertEqual(server["state"]["health"], "ok")
            self.assertEqual(server["metrics"]["cameras_total"], 0)
            self.assertEqual(server["metrics"]["detector_state"], "disabled")

            manager.stop_all()

        self.assertTrue(manager._closed)
        self.assertIsNone(manager._state_monitor_thread)
        self.assertIsNone(manager.recorder._index_thread)
        self.assertIsNone(manager.recorder._watchdog_thread)

    def test_constructor_failure_closes_services_created_before_workers(self) -> None:
        detector = Mock()
        faces = Mock()
        recorder = Mock()
        state_events = Mock()
        mqtt = Mock()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("survng.app.manager.InferenceSupervisor", return_value=detector),
            patch("survng.app.manager.FaceStore", return_value=faces),
            patch("survng.app.manager.Recorder", return_value=recorder),
            patch("survng.app.manager.StateEventBroker", return_value=state_events),
            patch("survng.app.manager.MqttService", return_value=mqtt),
            patch.object(AppManager, "_create_camera_worker", side_effect=RuntimeError("worker failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                AppManager(AppConfig(
                    storage_dir=tmpdir,
                    cameras=[CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")],
                ))

        mqtt.stop.assert_called_once_with()
        faces.close.assert_called_once_with()
        detector.stop.assert_called_once_with()
        recorder.stop_all.assert_called_once_with()
        state_events.close.assert_called_once_with()

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

    def test_camera_close_runs_even_when_camera_stop_fails(self) -> None:
        manager = manager_with_mocks()
        manager._started = True
        manager.workers["gate"].stop.side_effect = RuntimeError("stop failed")

        with self.assertRaisesRegex(RuntimeError, "camera gate"):
            manager.stop_all()

        manager.workers["gate"].close.assert_called_once_with()
        manager.detector.stop.assert_called_once_with()
        manager.recorder.stop_all.assert_called_once_with()

    def test_manager_camera_shutdown_deadline_does_not_block_other_cleanup(self) -> None:
        manager = manager_with_mocks()
        manager._started = True
        release = threading.Event()
        manager.workers["gate"].stop.side_effect = lambda: release.wait(timeout=1)

        try:
            with (
                patch("survng.app.manager.CAMERA_SHUTDOWN_TIMEOUT_SECONDS", 0.02),
                self.assertRaisesRegex(RuntimeError, "camera gate"),
            ):
                manager.stop_all()
        finally:
            release.set()

        manager.workers["gate"].close.assert_not_called()
        manager.detector.stop.assert_called_once_with()
        manager.recorder.stop_all.assert_called_once_with()

    def test_onvif_release_deadline_is_bounded(self) -> None:
        manager = manager_with_mocks()
        release = threading.Event()
        manager.workers["gate"].stop_onvif_events.side_effect = (
            lambda: release.wait(timeout=1)
        )

        try:
            with (
                patch("survng.app.manager.ONVIF_RELEASE_TIMEOUT_SECONDS", 0.02),
                self.assertRaisesRegex(RuntimeError, "gate"),
            ):
                manager.release_onvif_subscriptions()
        finally:
            release.set()

    def test_shutdown_releases_onvif_before_other_camera_components(self) -> None:
        manager = manager_with_mocks()
        manager._started = True
        order: list[str] = []
        manager.workers["gate"].stop_onvif_events.side_effect = (
            lambda: order.append("onvif")
        )
        manager.workers["gate"].stop.side_effect = lambda: order.append("camera")
        manager.mqtt.stop.side_effect = lambda: order.append("mqtt")

        manager.stop_all()

        self.assertLess(order.index("onvif"), order.index("mqtt"))
        self.assertLess(order.index("onvif"), order.index("camera"))

    def test_early_onvif_release_keeps_video_worker_running(self) -> None:
        manager = manager_with_mocks()

        manager.release_onvif_subscriptions()

        manager.workers["gate"].stop_onvif_events.assert_called_once_with()
        manager.workers["gate"].stop.assert_not_called()
        manager.recorder.stop_all.assert_not_called()

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

    def test_persisted_runtime_preferences_roll_back_on_write_failure(self) -> None:
        manager = manager_with_mocks()
        previous = manager.runtime_preferences()
        manager._save_runtime_state.side_effect = OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            manager.apply_runtime_preferences(
                {
                    "recording_enabled": {"gate": False},
                    "detection_enabled": {"gate": False},
                    "camera_enabled": {"gate": False},
                },
                persist=True,
            )

        self.assertEqual(manager.runtime_preferences(), previous)

    def test_fresh_process_ignores_saved_runtime_state_and_defaults_all_controls_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "runtime_state.json").write_text(
                json.dumps({
                    "recording_enabled": {"gate": False},
                    "detection_enabled": {"gate": False},
                    "camera_enabled": {"gate": False},
                }),
                encoding="utf-8",
            )
            manager = AppManager(AppConfig(
                storage_dir=tmpdir,
                cameras=[CameraConfig(
                    id="gate",
                    name="Gate",
                    stream_url="rtsp://camera/main",
                )],
            ))

            self.assertEqual(
                manager.runtime_preferences(),
                {
                    "recording_enabled": {},
                    "detection_enabled": {},
                    "camera_enabled": {"gate": True},
                },
            )
            self.assertTrue(manager.recording_enabled("gate"))
            self.assertTrue(manager.detection_enabled("gate"))

            manager.stop_all()

    def test_runtime_preference_write_failure_rolls_back_memory(self) -> None:
        manager = manager_with_mocks()
        manager._recording_enabled = {"gate": True}
        manager._save_runtime_state.side_effect = OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            manager.set_recording("gate", False)

        self.assertEqual(manager._recording_enabled, {"gate": True})
        manager.recorder.set_camera_enabled.assert_not_called()

    def test_runtime_state_write_is_atomic_and_cleans_failed_temporary_file(self) -> None:
        manager = manager_with_mocks()
        with tempfile.TemporaryDirectory() as tmpdir:
            manager._runtime_state_lock = threading.Lock()
            manager._runtime_state_path = Path(tmpdir) / "runtime_state.json"
            manager._runtime_state_path.write_text('{"original": true}\n', encoding="utf-8")
            with patch("survng.app.manager.json.dump", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    AppManager._save_runtime_state(manager)
            contents = manager._runtime_state_path.read_text(encoding="utf-8")
            temporary_files = list(Path(tmpdir).glob(".runtime_state.json.*.tmp"))

        self.assertEqual(contents, '{"original": true}\n')
        self.assertEqual(temporary_files, [])

    def test_disabled_camera_remains_stopped_during_manager_start(self) -> None:
        manager = manager_with_mocks()
        manager._camera_enabled["gate"] = False

        manager.start_all()

        manager._save_runtime_state.assert_called_once_with()
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

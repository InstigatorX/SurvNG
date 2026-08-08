from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

from survng.app.camera_control import CameraControlService
from survng.app.config import AppConfig, CameraConfig


def control_service(
    state_path: Path,
) -> tuple[CameraControlService, Mock, Mock, Mock, Mock]:
    camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
    worker = Mock()
    recording = Mock()
    fleet = Mock()
    fleet_state = {"gate": True}
    fleet.set_camera_enabled.side_effect = lambda camera_id, enabled: (
        fleet_state.__setitem__(camera_id, bool(enabled)) is None
    )
    fleet.camera_enabled.side_effect = lambda camera_id: fleet_state.get(camera_id, False)
    fleet.start_camera.return_value = True
    fleet.stop_camera.return_value = True
    mqtt = Mock()
    monitor = Mock()
    controls = CameraControlService(
        cameras=[camera],
        workers={"gate": worker},
        recording=recording,
        fleet=fleet,
        mqtt=mqtt,
        runtime_monitor=monitor,
        state_path=state_path,
    )
    return controls, worker, recording, fleet, mqtt


class CameraControlServiceTest(unittest.TestCase):
    def test_restores_persisted_controls_for_configured_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime_state.json"
            path.write_text(
                '{"recording_enabled":{"gate":false},'
                '"detection_enabled":{"gate":false},'
                '"camera_enabled":{"gate":false,"removed":false}}',
                encoding="utf-8",
            )

            controls, _worker, _recording, _fleet, _mqtt = control_service(path)

            self.assertEqual(controls.snapshot(), {
                "recording_enabled": {"gate": False},
                "detection_enabled": {"gate": False},
                "camera_enabled": {"gate": False},
            })

    def test_legacy_media_state_is_loaded_then_migrated_to_local_state_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = root / "database" / "runtime_state.json"
            legacy = root / "media" / "runtime_state.json"
            legacy.parent.mkdir(parents=True)
            legacy.write_text(
                '{"recording_enabled":{"gate":false},'
                '"detection_enabled":{"gate":false},'
                '"camera_enabled":{"gate":true}}',
                encoding="utf-8",
            )
            camera = CameraConfig(
                id="gate",
                name="Gate",
                stream_url="rtsp://camera/main",
            )
            worker = Mock()
            controls = CameraControlService(
                cameras=[camera],
                workers={"gate": worker},
                recording=Mock(),
                fleet=Mock(),
                mqtt=Mock(),
                runtime_monitor=Mock(),
                state_path=local,
                legacy_state_paths=(legacy,),
            )

            self.assertFalse(controls.recording_enabled("gate"))
            self.assertFalse(controls.detection_enabled("gate"))
            self.assertFalse(local.exists())

            controls.persist()

            self.assertTrue(local.exists())
            self.assertIn('"gate": false', local.read_text(encoding="utf-8"))

    def test_malformed_persisted_controls_fall_back_to_safe_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime_state.json"
            path.write_text('{"camera_enabled":{"gate":"false"}}', encoding="utf-8")

            controls, _worker, _recording, _fleet, _mqtt = control_service(path)

            self.assertEqual(controls.snapshot(), {
                "recording_enabled": {},
                "detection_enabled": {},
                "camera_enabled": {"gate": True},
            })

    def test_preference_write_failure_rolls_back_before_runtime_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            controls, _worker, recording, _fleet, _mqtt = control_service(
                Path(tmpdir) / "missing" / "runtime_state.json"
            )
            controls._persist_locked = Mock(side_effect=OSError("disk full"))

            with self.assertRaisesRegex(OSError, "disk full"):
                controls.set_recording("gate", False)

            self.assertTrue(controls.recording_enabled("gate"))
            recording.set_camera_enabled.assert_not_called()

    def test_failed_recorder_transition_restores_persisted_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime_state.json"
            controls, _worker, recording, _fleet, _mqtt = control_service(path)
            recording.set_camera_enabled.side_effect = [RuntimeError("recorder busy"), None]

            with self.assertRaisesRegex(RuntimeError, "recorder busy"):
                controls.set_recording("gate", False)

            self.assertTrue(controls.recording_enabled("gate"))
            restored = CameraControlService(
                cameras=list(controls._cameras.values()),
                workers=controls._workers,
                recording=controls._recording,
                fleet=controls._fleet,
                mqtt=controls._mqtt,
                runtime_monitor=controls._runtime_monitor,
                state_path=path,
            )
            self.assertTrue(restored.recording_enabled("gate"))
            self.assertEqual(
                recording.set_camera_enabled.call_args_list[-1].args,
                ("gate", True),
            )

    def test_failed_detection_transition_restores_persisted_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime_state.json"
            controls, worker, _recording, _fleet, _mqtt = control_service(path)
            worker.set_detection_enabled.side_effect = RuntimeError("ONVIF busy")

            with self.assertRaisesRegex(RuntimeError, "ONVIF busy"):
                controls.set_detection("gate", False)

            self.assertTrue(controls.detection_enabled("gate"))
            payload = path.read_text(encoding="utf-8")
            self.assertIn('"gate": true', payload)

    def test_apply_filters_removed_cameras_and_rolls_back_failed_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            controls, _worker, _recording, _fleet, _mqtt = control_service(
                Path(tmpdir) / "runtime_state.json"
            )
            controls.apply({
                "recording_enabled": {"gate": False, "removed": False},
                "detection_enabled": {"gate": True, "removed": True},
                "camera_enabled": {"gate": False, "removed": False},
            })
            previous = controls.snapshot()
            controls._persist_locked = Mock(side_effect=OSError("disk full"))

            with self.assertRaisesRegex(OSError, "disk full"):
                controls.apply({"camera_enabled": {"gate": True}}, persist=True)

            self.assertEqual(controls.snapshot(), previous)

    def test_failed_camera_start_restores_desired_and_recorder_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            controls, _worker, recording, fleet, _mqtt = control_service(
                Path(tmpdir) / "runtime_state.json"
            )
            controls.apply({"camera_enabled": {"gate": False}})
            fleet.start_camera.side_effect = RuntimeError("capture failed")

            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                controls.start_camera("gate")

            self.assertFalse(controls.camera_enabled("gate"))
            self.assertFalse(fleet.camera_enabled("gate"))
            recording.set_camera_enabled.assert_called_with("gate", False)

    def test_failed_camera_stop_remains_truthfully_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            controls, _worker, recording, fleet, mqtt = control_service(
                Path(tmpdir) / "runtime_state.json"
            )
            fleet.stop_camera.side_effect = RuntimeError("stop failed")

            with self.assertRaisesRegex(RuntimeError, "stop failed"):
                controls.stop_camera("gate")

            self.assertFalse(controls.camera_enabled("gate"))
            self.assertFalse(fleet.camera_enabled("gate"))
            recording.set_camera_enabled.assert_called_with("gate", False)
            mqtt.publish_camera_state.assert_called_once_with("gate", False)

    def test_quiesce_waits_for_inflight_transaction_and_rejects_late_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            controls, worker, _recording, _fleet, _mqtt = control_service(
                Path(tmpdir) / "runtime_state.json"
            )
            entered = threading.Event()
            release = threading.Event()

            def block_detection(_enabled: bool) -> None:
                entered.set()
                release.wait(1.0)

            worker.set_detection_enabled.side_effect = block_detection
            command = threading.Thread(
                target=controls.set_detection,
                args=("gate", False),
            )
            command.start()
            self.assertTrue(entered.wait(1.0))
            quiesced = threading.Event()
            shutdown = threading.Thread(
                target=lambda: (controls.quiesce(), quiesced.set()),
            )
            shutdown.start()
            time.sleep(0.02)
            self.assertFalse(quiesced.is_set())
            release.set()
            command.join(1.0)
            shutdown.join(1.0)

            self.assertTrue(quiesced.is_set())
            self.assertFalse(controls.set_detection("gate", True))
            self.assertEqual(worker.set_detection_enabled.call_count, 1)

    def test_recorder_reconfiguration_serializes_recording_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            controls, _worker, recording, _fleet, _mqtt = control_service(
                Path(tmpdir) / "runtime_state.json"
            )
            entered = threading.Event()
            release = threading.Event()

            def block_reconfigure(*_args, **_kwargs) -> None:
                entered.set()
                release.wait(1.0)

            recording.reconfigure.side_effect = block_reconfigure
            cutover = threading.Thread(
                target=controls.reconfigure_recorders,
                args=(AppConfig(),),
                kwargs={"restart_recorders": True},
            )
            cutover.start()
            self.assertTrue(entered.wait(1.0))
            command = threading.Thread(
                target=controls.set_recording,
                args=("gate", False),
            )
            command.start()
            time.sleep(0.02)
            recording.set_camera_enabled.assert_not_called()
            release.set()
            cutover.join(1.0)
            command.join(1.0)

            recording.set_camera_enabled.assert_called_once_with("gate", False)


if __name__ == "__main__":
    unittest.main()

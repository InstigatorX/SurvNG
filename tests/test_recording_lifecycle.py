from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, call

from survng.app.config import AppConfig, CameraConfig
from survng.app.recording_lifecycle import RecordingLifecycle


class RecordingLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://camera/main",
            live_stream_url="rtsp://camera/live",
            record=True,
            record_sub=True,
        )
        self.config = AppConfig(cameras=[self.camera])
        self.recorder = Mock()
        self.recorder.ffmpeg_path = self.config.ffmpeg_path
        self.recorder.hardware_acceleration = self.config.hardware_acceleration
        self.recorder.segment_seconds = self.config.recording_segment_seconds
        self.lifecycle = RecordingLifecycle(
            config=self.config,
            storage_dir=Path("."),
            protected_recording_paths=set,
            recorder=self.recorder,
        )

    def test_shared_services_start_after_stale_cleanup(self) -> None:
        order: list[str] = []
        self.recorder.cleanup_stale_recorders.side_effect = lambda _keys: order.append("cleanup")
        self.recorder.start_indexer.side_effect = lambda _cameras: order.append("index")
        self.recorder.start_watchdog.side_effect = lambda _cameras: order.append("watchdog")

        timings = self.lifecycle.start_services(
            [self.camera],
            {("gate", "main"), ("gate", "live")},
        )

        self.assertEqual(order, ["cleanup", "index", "watchdog"])
        self.assertGreaterEqual(timings.cleanup_seconds, 0.0)
        self.assertGreaterEqual(timings.services_seconds, 0.0)

    def test_partial_service_start_is_rolled_back(self) -> None:
        self.recorder.start_watchdog.side_effect = RuntimeError("watchdog failed")

        with self.assertRaisesRegex(RuntimeError, "watchdog failed"):
            self.lifecycle.start_services([self.camera], {("gate", "main")})

        self.recorder.stop_all.assert_called_once_with()

    def test_reconfigure_restarts_only_desired_camera_streams(self) -> None:
        next_config = self.config.model_copy(update={"recording_segment_seconds": 30})

        self.lifecycle.reconfigure(
            next_config,
            [self.camera],
            {"gate": True},
            restart_recorders=True,
        )

        self.assertEqual(
            self.recorder.set_camera_enabled.call_args_list,
            [call("gate", False), call("gate", True)],
        )
        self.recorder.reconfigure_runtime.assert_called_once_with(
            ffmpeg_path=next_config.ffmpeg_path,
            hardware_acceleration=next_config.hardware_acceleration,
            segment_seconds=30,
        )
        self.assertEqual(
            self.recorder.start.call_args_list,
            [call(self.camera, "main"), call(self.camera, "live")],
        )

    def test_failed_recorder_start_restores_previous_generation(self) -> None:
        next_config = self.config.model_copy(update={"recording_segment_seconds": 30})
        self.recorder.start.side_effect = [RuntimeError("start failed"), None, None]

        with self.assertRaisesRegex(RuntimeError, "start failed"):
            self.lifecycle.reconfigure(
                next_config,
                [self.camera],
                {"gate": True},
                restart_recorders=True,
            )

        self.assertEqual(self.recorder.reconfigure_runtime.call_count, 2)
        self.recorder.reconfigure_runtime.assert_has_calls([
            call(
                ffmpeg_path=next_config.ffmpeg_path,
                hardware_acceleration=next_config.hardware_acceleration,
                segment_seconds=30,
            ),
            call(
                ffmpeg_path=self.config.ffmpeg_path,
                hardware_acceleration=self.config.hardware_acceleration,
                segment_seconds=self.config.recording_segment_seconds,
            ),
        ])
        self.assertEqual(
            self.recorder.set_camera_enabled.call_args_list,
            [
                call("gate", False),
                call("gate", True),
                call("gate", False),
                call("gate", True),
            ],
        )

    def test_close_is_idempotent_and_terminal(self) -> None:
        self.lifecycle.close()
        self.lifecycle.close()

        self.recorder.stop_all.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.lifecycle.start_services([], set())


if __name__ == "__main__":
    unittest.main()

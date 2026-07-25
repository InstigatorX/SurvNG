from __future__ import annotations

import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import cv2

from survng.app.camera import CameraWorker
from survng.app.config import CameraConfig, MotionQualificationConfig


class DummyDetector:
    def __init__(self) -> None:
        self.calls = 0
        self.config = SimpleNamespace(confidence_threshold=0.5)

    def detect(self, frame, confidence_threshold=None):
        self.calls += 1
        return [
            {
                "label": "car",
                "confidence": 0.8,
                "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            }
        ]


class DummyEvents:
    pass


class DummyRecorder:
    ffmpeg_path = "ffmpeg"

    def recording_at(self, camera_id: str, epoch: float):
        return None


class CameraWorkerTest(unittest.TestCase):
    def test_capture_limits_ffmpeg_decoder_threads(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        capture = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = CameraWorker(camera, Path(tmpdir), DummyDetector(), DummyEvents(), DummyRecorder())
            worker._stop.clear()

            def stop_after_open(*_args):
                worker._stop.set()
                return False

            capture.open.side_effect = stop_after_open
            with patch("survng.app.camera.cv2.VideoCapture", return_value=capture):
                worker._run_source("live", __import__("threading").Event())

        _, backend, options = capture.open.call_args.args
        self.assertEqual(backend, cv2.CAP_FFMPEG)
        self.assertEqual(options[options.index(cv2.CAP_PROP_N_THREADS) + 1], 1)
        capture.release.assert_called_once()

    def test_borderline_candidate_is_rescued_by_eligible_object(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        events = Mock()
        events.add_event.return_value = {"id": 42}
        qualification = {
            "borderline_candidate": True,
            "effective_accepted": True,
            "would_suppress": True,
        }
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        objects = [{"label": "dog", "confidence": 0.8, "incident_eligible": True}]
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = CameraWorker(camera, Path(tmpdir), DummyDetector(), events, DummyRecorder())
            with (
                patch.object(worker, "_recorded_motion_frame", return_value=(frame, objects, "recording.mp4")),
                patch.object(worker, "_write_snapshot", return_value="snapshot.jpg"),
            ):
                outcome = worker._process_motion_event("motion", "message", datetime.now(timezone.utc), qualification)

        self.assertTrue(outcome["object_detected"])
        self.assertTrue(qualification["rescued_by_object"])
        self.assertTrue(qualification["effective_accepted"])
        self.assertFalse(qualification["would_suppress"])

    def test_borderline_candidate_without_object_remains_suppressed(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        events = Mock()
        events.add_event.return_value = {"id": 43}
        qualification = {
            "borderline_candidate": True,
            "effective_accepted": True,
            "would_suppress": False,
        }
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = CameraWorker(camera, Path(tmpdir), DummyDetector(), events, DummyRecorder())
            with (
                patch.object(worker, "_recorded_motion_frame", return_value=(frame, [], "recording.mp4")),
                patch.object(worker, "_write_snapshot", return_value="snapshot.jpg"),
            ):
                outcome = worker._process_motion_event("motion", "message", datetime.now(timezone.utc), qualification)

        self.assertFalse(outcome["object_detected"])
        self.assertFalse(qualification["rescued_by_object"])
        self.assertFalse(qualification["effective_accepted"])
        self.assertTrue(qualification["would_suppress"])

    def test_motion_ring_uses_per_camera_analysis_width(self) -> None:
        camera = CameraConfig.model_validate({
            "id": "back-middle",
            "name": "Back Middle",
            "stream_url": "rtsp://example.invalid/main",
            "motion_qualification": {"frame_width": 480},
        })
        config = MotionQualificationConfig(frame_width=320, sample_fps=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = CameraWorker(camera, Path(tmpdir), DummyDetector(), DummyEvents(), DummyRecorder(), config)
            worker._remember_motion_frame(np.zeros((720, 1280, 3), dtype=np.uint8), time.monotonic())

            self.assertEqual(worker._motion_settings(), ("audit", "balanced", 480))
            self.assertEqual(worker.status()["motion_qualification"]["frame_width"], 480)
            self.assertEqual(worker.status()["motion_qualification"]["frame_shape"], [270, 480])
            self.assertTrue(worker.status()["motion_qualification"]["mog2_audit_enabled"])
            self.assertIsNotNone(worker.status()["motion_qualification"]["mog2_last"])

    def test_powered_off_worker_does_not_start_snapshot_source(self) -> None:
        camera = CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = CameraWorker(camera, Path(tmpdir), DummyDetector(), DummyEvents(), DummyRecorder())
            with patch.object(worker, "_start_source", wraps=worker._start_source) as start_source:
                self.assertIsNone(worker.snapshot("main"))

        start_source.assert_not_called()

    def test_status_separates_power_capture_and_frame_freshness(self) -> None:
        camera = CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = CameraWorker(camera, Path(tmpdir), DummyDetector(), DummyEvents(), DummyRecorder())
            worker._enabled = True
            worker._stop.clear()
            worker._source_threads["live"] = Mock(is_alive=lambda: True)
            worker._source_frame_at["live"] = "2026-07-14T12:00:00+00:00"
            worker._source_frame_monotonic["live"] = time.monotonic()
            status = worker.status()

        self.assertTrue(status["running"])
        self.assertTrue(status["capture_running"])
        self.assertTrue(status["connected"])
        self.assertTrue(status["frame_fresh"])

    def test_only_main_source_expires_when_idle(self) -> None:
        camera = CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = CameraWorker(camera, Path(tmpdir), DummyDetector(), DummyEvents(), DummyRecorder())
            worker._source_last_access["main"] = time.monotonic() - 60

            self.assertTrue(worker._source_is_idle("main"))
            self.assertFalse(worker._source_is_idle("live"))

    def test_disabled_detection_ignores_motion_event(self) -> None:
        camera = CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        detector = DummyDetector()
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = CameraWorker(camera, Path(tmpdir), detector, DummyEvents(), DummyRecorder())
            worker.set_detection_enabled(False)
            worker.handle_motion_event("onvif/motion", "motion")

        self.assertEqual(worker.last_motion_at, "")
        self.assertEqual(detector.calls, 0)

    def test_motion_handler_enqueues_without_running_detection_inline(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = CameraWorker(camera, Path(tmpdir), DummyDetector(), DummyEvents(), DummyRecorder())
            with patch.object(worker, "_recorded_motion_frame") as recorded_frame:
                worker.handle_motion_event("onvif/motion", "motion")

            self.assertEqual(worker._motion_queue.qsize(), 1)
            recorded_frame.assert_not_called()

    def test_motion_worker_coalesces_a_trigger_burst(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(mode="off", burst_quiet_seconds=0.1, window_seconds=0.8)
        published = []
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = CameraWorker(
                camera,
                Path(tmpdir),
                DummyDetector(),
                DummyEvents(),
                DummyRecorder(),
                config,
                lambda event_type, payload: published.append((event_type, payload)),
            )
            worker._stop.clear()
            with patch.object(
                worker,
                "_process_motion_event",
                return_value={"event_id": 1, "snapshot_path": "", "object_detected": False},
            ) as process_event:
                thread = worker._motion_thread = __import__("threading").Thread(target=worker._run_motion_events)
                thread.start()
                now = datetime.now(timezone.utc)
                worker.handle_motion_event("onvif/motion", "first", now)
                worker.handle_motion_event("onvif/motion", "second", now)
                deadline = time.monotonic() + 2
                while process_event.call_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.02)
                worker._stop.set()
                worker._motion_queue.put_nowait(None)
                thread.join(timeout=2)

            self.assertEqual(process_event.call_count, 1)
            qualifications = [payload for event_type, payload in published if event_type == "motion_qualification"]
            self.assertEqual(len(qualifications), 1)
            self.assertEqual(qualifications[0]["trigger_count"], 2)

    def test_all_zero_rolling_windows_fail_open_as_inconclusive(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main")
        config = MotionQualificationConfig(post_trigger_seconds=0.5, window_seconds=0.8)
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = CameraWorker(camera, Path(tmpdir), DummyDetector(), DummyEvents(), DummyRecorder(), config)
            worker._stop.clear()
            received_at = time.time() - 1.0
            for index in range(5):
                worker._motion_frames.append((received_at - 0.8 + index * 0.2, np.zeros((90, 160), dtype=np.uint8)))

            result, diagnostics = worker._qualify_motion_burst(
                datetime.fromtimestamp(received_at, timezone.utc),
                received_at,
                "balanced",
            )

            self.assertTrue(result.accepted)
            self.assertEqual(result.reason, "no_temporal_signal")
            self.assertGreater(diagnostics["windows_evaluated"], 0)
            self.assertEqual(result.features["mog2_warmed"], 0.0)

    def test_motion_event_runs_detection_on_live_fallback(self) -> None:
        camera = CameraConfig(
            id="back-middle",
            name="Back Middle",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        )
        detector = DummyDetector()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            worker = CameraWorker(camera, Path(tmpdir), detector, DummyEvents(), DummyRecorder())
            with (
                patch("survng.app.camera.RECORDED_EVENT_SETTLE_SECONDS", 0.0),
                patch("survng.app.camera.RECORDED_EVENT_RETRY_SECONDS", 0.0),
                patch.object(worker, "_get_latest_frame", lambda source="live": frame.copy()),
            ):
                fallback, objects, recording_path = worker._recorded_motion_frame(
                    datetime(2026, 7, 11, 15, 36, 57, tzinfo=timezone.utc)
                )

        self.assertIsNotNone(fallback)
        self.assertEqual(recording_path, "")
        self.assertEqual(detector.calls, 1)
        self.assertEqual(objects[0]["label"], "car")
        self.assertEqual(objects[0]["frame_source"], "live_fallback")
        self.assertEqual(objects[0]["recording_status"], "no_recorded_frame")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from survng.app.camera import CameraWorker
from survng.app.config import CameraConfig


class DummyDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
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

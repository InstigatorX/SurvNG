from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from survng.app.config import CameraConfig
from survng.app.motion_pipeline.object_detection import (
    _RecordedDetectionSample,
    RecordedMotionObjectDetector,
    _temporal_consensus,
)


def detected(
    label: str,
    confidence: float,
    box: tuple[int, int, int, int],
) -> dict:
    x1, y1, x2, y2 = box
    return {
        "label": label,
        "confidence": confidence,
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "incident_eligible": True,
    }


def sample(offset: float, objects: list[dict]) -> _RecordedDetectionSample:
    return _RecordedDetectionSample(
        offset=offset,
        frame=np.full((20, 30, 3), int((offset + 1) * 10), dtype=np.uint8),
        objects=objects,
        recording_path=f"sample-{offset}.mp4",
    )


class RecordedObjectConsensusTest(unittest.TestCase):
    def test_single_high_confidence_outlier_is_not_incident_eligible(self) -> None:
        samples = [
            sample(-1.0, []),
            sample(-0.5, [detected("robot_lawnmower", 0.83, (100, 100, 140, 140))]),
            sample(0.0, []),
            sample(0.5, []),
            sample(1.0, []),
        ]

        selected, objects = _temporal_consensus(samples, minimum_confirmations=2)

        self.assertEqual(selected.offset, -0.5)
        self.assertEqual(len(objects), 1)
        self.assertFalse(objects[0]["incident_eligible"])
        self.assertFalse(objects[0]["temporal_consensus"])
        self.assertEqual(objects[0]["temporal_observations"], 1)
        self.assertEqual(objects[0]["temporal_required_observations"], 2)

    def test_repeatable_label_uses_median_instead_of_peak_confidence(self) -> None:
        samples = [
            sample(-0.5, [detected("car", 0.75, (100, 100, 150, 150))]),
            sample(0.0, [detected("car", 0.99, (104, 101, 154, 151))]),
            sample(0.5, [detected("car", 0.80, (108, 102, 158, 152))]),
        ]

        _selected, objects = _temporal_consensus(samples, minimum_confirmations=2)

        self.assertEqual(len(objects), 1)
        self.assertTrue(objects[0]["incident_eligible"])
        self.assertEqual(objects[0]["confidence"], 0.8)
        self.assertEqual(objects[0]["temporal_peak_confidence"], 0.99)
        self.assertEqual(objects[0]["temporal_observations"], 3)
        self.assertEqual(objects[0]["temporal_sample_offset_seconds"], 0.0)

    def test_spatial_track_votes_prevent_high_confidence_label_outliers(self) -> None:
        samples = [
            sample(-1.0, [detected("robot_lawnmower", 0.93, (100, 100, 150, 150))]),
            sample(-0.5, [detected("car", 0.76, (102, 100, 152, 150))]),
            sample(0.0, [detected("robot_lawnmower", 0.91, (104, 100, 154, 150))]),
            sample(0.5, [detected("car", 0.78, (106, 100, 156, 150))]),
            sample(1.0, [detected("car", 0.80, (108, 100, 158, 150))]),
        ]

        _selected, objects = _temporal_consensus(samples, minimum_confirmations=2)

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["label"], "car")
        self.assertEqual(objects[0]["confidence"], 0.78)
        self.assertEqual(objects[0]["temporal_label_votes"], {
            "robot_lawnmower": 2,
            "car": 3,
        })
        self.assertEqual(objects[0]["temporal_track_observations"], 5)

    def test_distant_one_frame_objects_do_not_form_false_consensus(self) -> None:
        samples = [
            sample(-0.5, [detected("person", 0.90, (10, 10, 30, 50))]),
            sample(0.5, [detected("person", 0.88, (500, 400, 520, 440))]),
        ]

        _selected, objects = _temporal_consensus(samples, minimum_confirmations=2)

        self.assertEqual(len(objects), 1)
        self.assertFalse(objects[0]["incident_eligible"])
        self.assertEqual(objects[0]["temporal_observations"], 1)

    def test_class_override_can_require_stronger_confirmation(self) -> None:
        samples = [
            sample(-0.5, [detected("robot_lawnmower", 0.84, (100, 100, 140, 140))]),
            sample(0.0, [detected("robot_lawnmower", 0.82, (102, 100, 142, 140))]),
            sample(0.5, []),
        ]

        _selected, objects = _temporal_consensus(
            samples,
            minimum_confirmations=2,
            class_confirmations={"robot_lawnmower": 3},
        )

        self.assertFalse(objects[0]["incident_eligible"])
        self.assertEqual(objects[0]["temporal_observations"], 2)
        self.assertEqual(objects[0]["temporal_required_observations"], 3)

    def test_detector_retries_missing_offsets_until_confirmation_is_possible(self) -> None:
        event_epoch = 1_800_000_000.0
        calls_by_offset: dict[float, int] = {}

        class Recorder:
            ffmpeg_path = "ffmpeg"
            hardware_acceleration = "none"

            def recording_at(self, _camera_id: str, epoch: float):
                offset = round(epoch - event_epoch, 1)
                calls_by_offset[offset] = calls_by_offset.get(offset, 0) + 1
                if offset == -1.0 or (
                    offset == -0.5 and calls_by_offset[offset] >= 2
                ):
                    return {"path": f"sample-{offset}.mp4", "start_epoch": epoch}
                return None

        class Detector:
            config = SimpleNamespace(
                confidence_threshold=0.5,
                require_incident_zone=False,
                event_confirmation_frames=2,
                event_class_confirmation_frames={},
            )

            def __init__(self) -> None:
                self.calls = 0

            def detect(self, _frame, confidence_threshold=None):
                self.calls += 1
                return [detected("car", 0.8, (2, 2, 12, 12))]

        detector = Detector()
        backend = RecordedMotionObjectDetector(
            CameraConfig(
                id="gate",
                name="Gate",
                stream_url="rtsp://example.invalid/main",
            ),
            detector,
            Recorder(),
            lambda: None,
        )
        with (
            patch(
                "survng.app.motion_pipeline.object_detection.RECORDED_EVENT_SETTLE_SECONDS",
                0.0,
            ),
            patch(
                "survng.app.motion_pipeline.object_detection.RECORDED_EVENT_RETRY_SECONDS",
                0.1,
            ),
            patch(
                "survng.app.motion_pipeline.object_detection.RECORDED_EVENT_RETRY_INTERVAL_SECONDS",
                0.001,
            ),
            patch.object(
                backend,
                "_read_recorded_frame",
                return_value=np.zeros((20, 20, 3), dtype=np.uint8),
            ),
        ):
            _frame, objects, _path = backend.detect(
                datetime.fromtimestamp(event_epoch, timezone.utc)
            )

        self.assertEqual(detector.calls, 2)
        self.assertGreaterEqual(calls_by_offset[-0.5], 2)
        self.assertTrue(objects[0]["incident_eligible"])
        self.assertEqual(objects[0]["temporal_observations"], 2)


if __name__ == "__main__":
    unittest.main()

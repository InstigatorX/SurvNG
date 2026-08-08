from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
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
    def test_dedicated_face_detector_replaces_generic_face_boxes_as_auxiliary_evidence(self) -> None:
        class Detector:
            config = SimpleNamespace(
                confidence_threshold=0.45,
                event_class_confidence_thresholds={},
                require_incident_zone=False,
            )

            def detect(self, _frame, confidence_threshold=None):
                return [
                    detected("person", 0.9, (10, 5, 80, 95)),
                    detected("face", 0.8, (28, 12, 52, 40)),
                ]

            def detect_faces(self, _frame):
                return [{
                    "label": "face",
                    "confidence": 0.75,
                    "box": {"x1": 23, "y1": 10, "x2": 48, "y2": 38},
                    "detection_source": "dedicated_face",
                }]

        backend = RecordedMotionObjectDetector(
            CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            Detector(),
            SimpleNamespace(),
            lambda: None,
        )

        objects = backend._detect_objects(np.zeros((100, 100, 3), dtype=np.uint8))

        faces = [item for item in objects if item["label"] == "face"]
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["detection_source"], "dedicated_face")
        self.assertFalse(faces[0]["incident_eligible"])
        self.assertTrue(faces[0]["confidence_eligible"])
        self.assertIn("face_quality_score", faces[0])

    def test_auxiliary_faces_are_retained_only_alongside_confirmed_incident_objects(self) -> None:
        person = detected("person", 0.9, (2, 2, 25, 19))
        face = {
            **detected("face", 0.8, (5, 3, 14, 13)),
            "incident_eligible": False,
            "auxiliary_detection": True,
            "detection_source": "dedicated_face",
            "face_quality_score": 0.8,
        }
        samples = [
            sample(0.0, [dict(person), dict(face)]),
            sample(0.5, [dict(person), dict(face)]),
        ]

        _selected, objects = _temporal_consensus(samples, minimum_confirmations=2)

        retained_face = next(item for item in objects if item["label"] == "face")
        self.assertTrue(retained_face["temporal_consensus"])
        self.assertFalse(retained_face["incident_eligible"])

    def test_best_face_quality_selects_temporal_incident_snapshot(self) -> None:
        person = detected("person", 0.9, (2, 2, 25, 19))

        def face(quality: float) -> dict:
            return {
                **detected("face", 0.8, (5, 3, 14, 13)),
                "incident_eligible": False,
                "auxiliary_detection": True,
                "face_quality_score": quality,
            }

        samples = [
            sample(0.0, [dict(person), face(0.2)]),
            sample(0.5, [dict(person), face(0.9)]),
        ]

        selected, _objects = _temporal_consensus(samples, minimum_confirmations=2)

        self.assertEqual(selected.offset, 0.5)

    def test_visual_quality_breaks_ties_between_equally_valid_object_frames(self) -> None:
        checker = np.indices((80, 80)).sum(axis=0) % 2
        sharp = np.repeat((checker * 255).astype(np.uint8)[..., None], 3, axis=2)
        blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
        object_in_both = detected("person", 0.82, (8, 8, 72, 72))
        samples = [
            _RecordedDetectionSample(0.0, blurred, [dict(object_in_both)], "blurred.mp4"),
            _RecordedDetectionSample(0.5, sharp, [dict(object_in_both)], "sharp.mp4"),
        ]

        selected, objects = _temporal_consensus(samples, minimum_confirmations=2)

        self.assertEqual(selected.recording_path, "sharp.mp4")
        self.assertTrue(objects[0]["temporal_consensus"])
        self.assertGreater(objects[0]["snapshot_sharpness_score"], 0.5)
        self.assertGreater(objects[0]["snapshot_quality_score"], 0.5)

    def test_recorded_frame_transport_is_lossless_bmp_instead_of_mjpeg(self) -> None:
        expected = np.arange(24 * 32 * 3, dtype=np.uint8).reshape((24, 32, 3))
        success, encoded = cv2.imencode(".bmp", expected)
        self.assertTrue(success)
        backend = RecordedMotionObjectDetector(
            CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            SimpleNamespace(config=SimpleNamespace()),
            SimpleNamespace(ffmpeg_path="ffmpeg", hardware_acceleration="off"),
            lambda: None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            recording = Path(tmpdir) / "sample.mp4"
            recording.touch()
            completed = SimpleNamespace(returncode=0, stdout=encoded.tobytes(), stderr=b"")
            with patch(
                "survng.app.motion_pipeline.object_detection.subprocess.run",
                return_value=completed,
            ) as run:
                actual = backend._read_recorded_frame(recording, 1.25)

        self.assertTrue(np.array_equal(actual, expected))
        command = run.call_args.args[0]
        self.assertIn("bmp", command)
        self.assertIn("bgr24", command)
        self.assertNotIn("mjpeg", command)

    def test_per_class_confidence_sets_inference_floor_and_object_eligibility(self) -> None:
        class Detector:
            config = SimpleNamespace(
                confidence_threshold=0.45,
                event_class_confidence_thresholds={"car": 0.7, "person": 0.3},
                require_incident_zone=False,
            )

            def __init__(self) -> None:
                self.requested_threshold = None

            def detect(self, _frame, confidence_threshold=None):
                self.requested_threshold = confidence_threshold
                return [
                    detected("car", 0.69, (2, 2, 12, 12)),
                    detected("person", 0.35, (14, 2, 24, 12)),
                ]

        detector = Detector()
        backend = RecordedMotionObjectDetector(
            CameraConfig(
                id="gate",
                name="Gate",
                stream_url="rtsp://example.invalid/main",
            ),
            detector,
            SimpleNamespace(),
            lambda: None,
        )

        objects = backend._detect_objects(np.zeros((20, 30, 3), dtype=np.uint8))

        self.assertEqual(detector.requested_threshold, 0.3)
        self.assertFalse(objects[0]["incident_eligible"])
        self.assertEqual(objects[0]["confidence_threshold"], 0.7)
        self.assertEqual(objects[0]["detection_frame_width"], 30)
        self.assertEqual(objects[0]["detection_frame_height"], 20)
        self.assertTrue(objects[1]["incident_eligible"])
        self.assertEqual(objects[1]["confidence_threshold"], 0.3)

    def test_non_finite_confidence_cannot_win_temporal_selection(self) -> None:
        samples = [
            sample(-0.5, [detected("ghost", float("nan"), (10, 10, 30, 30))]),
            sample(0.0, [detected("car", 0.8, (100, 100, 150, 150))]),
        ]

        selected, objects = _temporal_consensus(samples, minimum_confirmations=1)

        self.assertEqual(selected.offset, 0.0)
        self.assertEqual(objects[0]["label"], "car")
        self.assertEqual(objects[0]["confidence"], 0.8)

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
        self.assertGreater(objects[0]["temporal_center_displacement_ratio"], 0)
        self.assertGreaterEqual(
            objects[0]["temporal_center_path_ratio"],
            objects[0]["temporal_center_displacement_ratio"],
        )

    def test_temporal_consensus_marks_object_that_appears_after_empty_sample(self) -> None:
        samples = [
            sample(-0.5, []),
            sample(0.0, [detected("car", 0.82, (100, 100, 160, 160))]),
            sample(0.5, [detected("car", 0.84, (101, 100, 161, 160))]),
        ]

        _selected, objects = _temporal_consensus(samples, minimum_confirmations=2)

        self.assertTrue(objects[0]["temporal_newly_appeared"])
        self.assertEqual(objects[0]["temporal_first_observation_offset_seconds"], 0.0)
        self.assertEqual(objects[0]["temporal_last_observation_offset_seconds"], 0.5)

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

    def test_zone_admission_in_one_frame_carries_across_the_same_confirmed_track(self) -> None:
        admitted = detected("car", 0.82, (100, 100, 180, 180))
        admitted["zones"] = ["Driveway"]
        corroborating = detected("car", 0.88, (120, 100, 200, 180))
        corroborating["incident_eligible"] = False
        corroborating["zones"] = []
        samples = [
            sample(0.0, [admitted]),
            sample(0.5, [corroborating]),
        ]

        selected, objects = _temporal_consensus(samples, minimum_confirmations=2)

        self.assertEqual(selected.offset, 0.0)
        self.assertTrue(objects[0]["incident_eligible"])
        self.assertTrue(objects[0]["temporal_consensus"])
        self.assertEqual(objects[0]["temporal_observations"], 2)
        self.assertEqual(objects[0]["temporal_incident_observations"], 1)
        self.assertEqual(objects[0]["zones"], ["Driveway"])

    def test_track_never_admitted_by_incident_policy_remains_ineligible(self) -> None:
        first = detected("car", 0.91, (100, 100, 180, 180))
        second = detected("car", 0.93, (120, 100, 200, 180))
        first["incident_eligible"] = False
        second["incident_eligible"] = False

        _selected, objects = _temporal_consensus(
            [sample(0.0, [first]), sample(0.5, [second])],
            minimum_confirmations=2,
        )

        self.assertFalse(objects[0]["incident_eligible"])
        self.assertFalse(objects[0]["temporal_consensus"])
        self.assertEqual(objects[0]["temporal_observations"], 2)
        self.assertEqual(objects[0]["temporal_incident_observations"], 0)

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
            patch(
                "survng.app.motion_pipeline.object_detection.time.time",
                return_value=event_epoch + 20.0,
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

    def test_detector_retries_when_sample_count_is_met_without_object_consensus(self) -> None:
        event_epoch = 1_800_000_000.0
        calls_by_offset: dict[float, int] = {}

        class Recorder:
            ffmpeg_path = "ffmpeg"
            hardware_acceleration = "none"

            def recording_at(self, _camera_id: str, epoch: float):
                offset = round(epoch - event_epoch, 1)
                calls_by_offset[offset] = calls_by_offset.get(offset, 0) + 1
                if offset in {-1.0, -0.5} or (offset == 0.0 and calls_by_offset[offset] >= 2):
                    return {"path": f"sample-{offset}.mp4", "start_epoch": epoch}
                return None

        class Detector:
            config = SimpleNamespace(
                confidence_threshold=0.5,
                require_incident_zone=False,
                event_confirmation_frames=2,
                event_class_confirmation_frames={},
            )

            def detect(self, frame, confidence_threshold=None):
                return [detected("car", 0.85, (2, 2, 12, 12))] if int(frame[0, 0, 0]) != 2 else []

        backend = RecordedMotionObjectDetector(
            CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            Detector(),
            Recorder(),
            lambda: None,
        )

        def read_frame(path, _offset, **_kwargs):
            value = 2 if "-0.5" in str(path) else 1
            return np.full((20, 20, 3), value, dtype=np.uint8)

        with (
            patch("survng.app.motion_pipeline.object_detection.RECORDED_EVENT_SETTLE_SECONDS", 0.0),
            patch("survng.app.motion_pipeline.object_detection.RECORDED_EVENT_RETRY_SECONDS", 0.1),
            patch("survng.app.motion_pipeline.object_detection.RECORDED_EVENT_RETRY_INTERVAL_SECONDS", 0.001),
            patch(
                "survng.app.motion_pipeline.object_detection.time.time",
                return_value=event_epoch + 20.0,
            ),
            patch.object(backend, "_read_recorded_frame", side_effect=read_frame),
        ):
            _frame, objects, _path = backend.detect(datetime.fromtimestamp(event_epoch, timezone.utc))

        self.assertGreaterEqual(calls_by_offset[0.0], 2)
        self.assertTrue(objects[0]["incident_eligible"])
        self.assertEqual(objects[0]["temporal_observations"], 2)

    def test_detector_uses_sparse_late_stage_when_object_appears_after_trigger(self) -> None:
        event_epoch = 1_800_000_000.0
        requested_offsets: list[float] = []

        class Recorder:
            ffmpeg_path = "ffmpeg"
            hardware_acceleration = "none"

            def recording_at(self, _camera_id: str, epoch: float):
                offset = round(epoch - event_epoch, 1)
                requested_offsets.append(offset)
                return {"path": f"sample-{offset}.mp4", "start_epoch": epoch}

        class Detector:
            config = SimpleNamespace(
                confidence_threshold=0.5,
                require_incident_zone=False,
                event_confirmation_frames=2,
                event_class_confirmation_frames={},
            )

            def detect(self, frame, confidence_threshold=None):
                if int(frame[0, 0, 0]) < 8:
                    return []
                return [detected("car", 0.86, (2, 2, 12, 12))]

        backend = RecordedMotionObjectDetector(
            CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            Detector(),
            Recorder(),
            lambda: None,
        )

        def read_frame(path, _offset, **_kwargs):
            offset = float(str(path).removeprefix("sample-").removesuffix(".mp4"))
            return np.full((20, 20, 3), max(0, int(offset)), dtype=np.uint8)

        with (
            patch("survng.app.motion_pipeline.object_detection.time.time", return_value=event_epoch + 20.0),
            patch.object(backend, "_read_recorded_frame", side_effect=read_frame),
        ):
            _frame, objects, _path = backend.detect(datetime.fromtimestamp(event_epoch, timezone.utc))

        self.assertIn(8.0, requested_offsets)
        self.assertIn(8.5, requested_offsets)
        self.assertNotIn(12.0, requested_offsets)
        self.assertTrue(objects[0]["incident_eligible"])
        self.assertEqual(objects[0]["temporal_observations"], 2)
        self.assertEqual(objects[0]["temporal_sample_offset_seconds"], 8.0)


if __name__ == "__main__":
    unittest.main()

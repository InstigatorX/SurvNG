from __future__ import annotations

import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from survng.app.config import DetectorConfig
from survng.app.detector import (
    OpenVinoDetector,
    _class_aware_nms,
    detection_failure,
    merge_manual_detection_objects,
)


class _FailingNet:
    def setInput(self, _tensor) -> None:
        raise RuntimeError("inference failed")


def make_detector(labels: list[str] | None = None) -> OpenVinoDetector:
    detector = OpenVinoDetector(DetectorConfig(enabled=False))
    detector.labels = labels or ["person", "car"]
    detector.config.confidence_threshold = 0.5
    detector.config.nms_threshold = 0.45
    detector.input_shape = (640, 640)
    return detector


class OpenVinoDetectorTest(unittest.TestCase):
    def test_detection_failure_reports_worker_error(self) -> None:
        self.assertEqual(
            detection_failure([{"status": "detector_unavailable", "error": "timed out"}]),
            "timed out",
        )
        self.assertEqual(detection_failure([]), "")
        self.assertEqual(detection_failure(["legacy", None]), "")  # type: ignore[list-item]

    def test_manual_detection_merge_preserves_incident_metadata(self) -> None:
        existing = '[{"label":"old"},{"status":"no_recorded_frame"},{"status":"motion_qualification","motion_qualification":{"score":0.7}},{"status":"object_tracking","object_tracking":{"state":"complete"}}]'
        detected = [{"label": "person", "confidence": 0.9}]

        merged = merge_manual_detection_objects(existing, detected)

        self.assertEqual(merged[0]["label"], "person")
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[1]["status"], "motion_qualification")
        self.assertEqual(merged[2]["status"], "object_tracking")

    def test_output_format_distinguishes_two_class_raw_yolo_from_e2e(self) -> None:
        detector = make_detector(["person", "car"])
        detector.output_layers = [SimpleNamespace(shape=(1, 6, 8400))]
        detector.output_layer = detector.output_layers[0]
        self.assertEqual(detector._detect_output_format(), "yolo")

        detector.output_layers = [SimpleNamespace(shape=(1, 300, 6))]
        detector.output_layer = detector.output_layers[0]
        self.assertEqual(detector._detect_output_format(), "yolo-e2e")

    def test_failed_inference_releases_active_counter_and_records_failure(self) -> None:
        detector = make_detector()
        detector.enabled = True
        detector.cv_net = _FailingNet()
        detector.output_format = "yolo"

        with self.assertRaisesRegex(ValueError, "uint8 BGR"):
            detector.detect(np.zeros((10, 10, 3), dtype=np.float32))

        runtime = detector.status()["runtime"]
        self.assertEqual(runtime["active_inferences"], 0)
        self.assertEqual(runtime["pending_frames"], 0)
        self.assertEqual(runtime["total_inferences"], 1)
        self.assertEqual(runtime["failed_inferences"], 1)

    def test_non_finite_override_threshold_is_rejected_and_restored(self) -> None:
        detector = make_detector()
        detector.enabled = True
        detector.cv_net = _FailingNet()
        detector.output_format = "yolo"

        with self.assertRaisesRegex(ValueError, "finite"):
            detector.detect(
                np.zeros((10, 10, 3), dtype=np.uint8),
                confidence_threshold=float("nan"),
            )

        self.assertEqual(detector.config.confidence_threshold, 0.5)

    def test_yolo_parser_skips_non_finite_rows_without_losing_valid_detection(self) -> None:
        detector = make_detector(["person"])
        metadata = {
            "image_width": 640.0,
            "image_height": 640.0,
            "scale": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
        }
        output = np.array(
            [[[100.0, 100.0, 40.0, 60.0, 0.9], [np.nan, 20.0, 4.0, 4.0, 0.99]]],
            dtype=np.float32,
        )

        objects = detector._parse_yolo_output(output, metadata)

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["label"], "person")

    def test_yolo_nms_preserves_overlapping_different_classes(self) -> None:
        detector = make_detector(["person", "car"])
        metadata = {
            "image_width": 640.0,
            "image_height": 640.0,
            "scale": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
        }
        output = np.array([[
            [100.0, 100.0, 60.0, 80.0, 0.90, 0.05],
            [100.0, 100.0, 60.0, 80.0, 0.05, 0.88],
        ]], dtype=np.float32)

        objects = detector._parse_yolo_output(output, metadata)

        self.assertEqual([item["label"] for item in objects], ["person", "car"])

    def test_vectorized_yolo_parser_matches_scalar_reference(self) -> None:
        detector = make_detector(["person", "car", "dog", "truck", "face", "bird"])
        metadata = {
            "image_width": 1920.0,
            "image_height": 1080.0,
            "scale": 1 / 3,
            "pad_x": 0.0,
            "pad_y": 140.0,
        }
        rng = np.random.default_rng(20260804)
        rows = np.zeros((8400, 10), dtype=np.float32)
        rows[:, 0] = rng.uniform(-100, 740, size=8400)
        rows[:, 1] = rng.uniform(-100, 740, size=8400)
        rows[:, 2] = rng.uniform(1, 300, size=8400)
        rows[:, 3] = rng.uniform(1, 300, size=8400)
        rows[:, 4:] = rng.uniform(0.0, 0.49, size=(8400, 6))
        credible = rng.choice(8400, size=160, replace=False)
        classes = rng.integers(0, 6, size=len(credible))
        rows[credible, 4 + classes] = rng.uniform(0.5, 0.99, size=len(credible))
        rows[1, 4] = np.nan
        rows[2, 0] = np.inf
        rows[3, 2] = -1
        rows[4, 3] = 0

        for output in (rows[np.newaxis, ...], rows.T[np.newaxis, ...]):
            metrics: dict[str, float | int] = {}
            vectorized = detector._parse_yolo_output(output, metadata, metrics=metrics)
            scalar = detector._parse_yolo_output_scalar_reference(output, metadata)

            self.assertEqual(vectorized, scalar)
            self.assertEqual(metrics["raw"], 8400)
            self.assertGreater(metrics["confidence"], 0)
            self.assertEqual(metrics["selected"], len(vectorized))
            self.assertGreaterEqual(float(metrics["decode_ms"]), 0.0)
            self.assertGreaterEqual(float(metrics["nms_ms"]), 0.0)

    def test_vectorized_yolo_decode_is_faster_than_scalar_reference(self) -> None:
        detector = make_detector(["person", "car"])
        metadata = {
            "image_width": 640.0,
            "image_height": 640.0,
            "scale": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
        }
        output = np.zeros((1, 6, 8400), dtype=np.float32)
        output[:, :4, :] = 100.0
        output[:, 4:, :] = 0.1
        output[0, 4, :24] = 0.9
        detector._parse_yolo_output(output, metadata)
        detector._parse_yolo_output_scalar_reference(output, metadata)

        started = time.perf_counter()
        for _ in range(10):
            detector._parse_yolo_output(output, metadata)
        vectorized_ms = (time.perf_counter() - started) * 1000 / 10
        started = time.perf_counter()
        for _ in range(2):
            detector._parse_yolo_output_scalar_reference(output, metadata)
        scalar_ms = (time.perf_counter() - started) * 1000 / 2

        self.assertLess(vectorized_ms, scalar_ms * 0.5)

    def test_class_aware_nms_falls_back_when_batched_nms_fails(self) -> None:
        boxes = [[10, 10, 30, 30], [11, 11, 30, 30], [10, 10, 30, 30]]
        scores = [0.9, 0.8, 0.85]
        class_ids = [0, 0, 1]
        expected = _class_aware_nms(boxes, scores, class_ids, 0.5, 0.45)

        with patch(
            "survng.app.detector.cv2.dnn.NMSBoxesBatched",
            side_effect=cv2.error("unavailable"),
        ):
            fallback = _class_aware_nms(boxes, scores, class_ids, 0.5, 0.45)

        self.assertEqual(fallback, expected)

    def test_runtime_stats_expose_stage_percentiles_and_candidate_counts(self) -> None:
        detector = make_detector()
        detector._active_inferences = 1
        detector._finish_inference(
            {
                "queue": 1.0,
                "preprocess": 2.0,
                "inference": 3.0,
                "postprocess": 4.0,
                "postprocess_decode": 2.5,
                "postprocess_nms": 1.0,
                "postprocess_result": 0.5,
                "total": 10.0,
            },
            [{"label": "person"}],
            succeeded=True,
            candidate_counts={
                "raw": 8400,
                "confidence": 12,
                "valid_boxes": 11,
                "selected": 3,
            },
        )

        runtime = detector.status()["runtime"]

        self.assertEqual(runtime["stages"]["percentiles_ms"]["postprocess"]["p95"], 4.0)
        self.assertEqual(runtime["candidates"]["last"]["raw"], 8400)
        self.assertEqual(runtime["candidates"]["average"]["selected"], 3.0)

    def test_ssd_parser_clamps_boxes_and_supports_one_based_labels(self) -> None:
        detector = make_detector(["person", "car"])
        output = np.array(
            [[[[0.0, 1.0, 0.9, -0.1, 0.2, 1.2, 0.8], [0.0, 2.0, 0.8, np.nan, 0.0, 1.0, 1.0]]]],
            dtype=np.float32,
        )

        objects = detector._parse_ssd_output(output, 100, 50)

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["label"], "person")
        self.assertEqual(objects[0]["box"], {"x1": 0, "y1": 10, "x2": 100, "y2": 40})

    def test_raw_yolo_segmentation_output_is_transposed_scored_and_nms_filtered(self) -> None:
        detector = make_detector(["person"])
        metadata = {
            "image_width": 640.0,
            "image_height": 640.0,
            "scale": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
        }
        rows = np.zeros((1201, 37), dtype=np.float32)
        rows[0, :5] = [100.0, 100.0, 40.0, 60.0, 0.9]
        rows[1, :5] = [101.0, 101.0, 40.0, 60.0, 0.8]
        detections = rows.T[np.newaxis, ...]
        prototypes = np.zeros((1, 32, 8, 8), dtype=np.float32)

        objects = detector._parse_yolo_seg_outputs([detections, prototypes], metadata)

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["label"], "person")
        self.assertEqual(objects[0]["box"], {"x1": 80, "y1": 70, "x2": 120, "y2": 130})

    def test_e2e_segmentation_output_is_not_confused_with_two_class_raw_output(self) -> None:
        detector = make_detector(["person", "car"])
        metadata = {
            "image_width": 640.0,
            "image_height": 640.0,
            "scale": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
        }
        detections = np.zeros((1, 300, 38), dtype=np.float32)
        detections[0, 0, :6] = [50.0, 60.0, 150.0, 180.0, 0.9, 1.0]
        prototypes = np.zeros((1, 32, 8, 8), dtype=np.float32)

        objects = detector._parse_yolo_seg_outputs([detections, prototypes], metadata)

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["label"], "car")
        self.assertEqual(objects[0]["box"], {"x1": 50, "y1": 60, "x2": 150, "y2": 180})


if __name__ == "__main__":
    unittest.main()

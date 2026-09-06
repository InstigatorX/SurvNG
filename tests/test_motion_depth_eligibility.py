from __future__ import annotations

import json
import time
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from survng.app.config import CameraConfig, DepthConfig
from survng.app.depth_estimation import OpenVinoDepthEstimator
from survng.app.motion_pipeline.decision_handler import MotionDecisionHandler
from survng.app.motion_pipeline.object_detection import (
    _RecordedDetectionSample,
    _temporal_consensus,
    RecordedMotionObjectDetector,
)
from survng.app.zones import apply_detection_zones


class MotionDepthEligibilityTest(unittest.TestCase):
    def test_recorded_depth_preserves_temporal_and_distance_vetoes_in_incident_handoff(self) -> None:
        def detected(label, confidence, box):
            x1, y1, x2, y2 = box
            return {
                "label": label,
                "confidence": confidence,
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "incident_eligible": True,
            }

        event_at = datetime.fromtimestamp(1_800_000_000.0, timezone.utc)
        camera = CameraConfig(
            id="gate", name="Gate", stream_url="rtsp://example.invalid/main",
            zones=[{
                "name": "Approach", "behavior": "incident", "max_depth_m": 25.0,
                "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}],
            }],
        )
        frame = np.zeros((100, 240, 3), dtype=np.uint8)
        samples = []
        for offset in (0.0, 0.5):
            objects = [
                detected("person", 0.9, (20, 20, 60, 80)),
                detected("dog", 0.9, (190, 20, 220, 80)),
            ]
            if offset == 0.0:
                objects.append(detected("car", 0.9, (100, 20, 130, 80)))
            apply_detection_zones(camera, objects, 240, 100, 0.5, True, {})
            samples.append(_RecordedDetectionSample(offset, frame.copy(), objects, "event.mp4"))
        selected, objects = _temporal_consensus(samples, minimum_confirmations=2)
        before_depth = {item["label"]: item for item in objects}
        self.assertTrue(before_depth["person"]["incident_eligible"])
        self.assertTrue(before_depth["dog"]["incident_eligible"])
        self.assertFalse(before_depth["car"]["temporal_eligible"])

        depth_config = DepthConfig(enabled=True, max_incident_distance_m=15.0)
        estimator = OpenVinoDepthEstimator.__new__(OpenVinoDepthEstimator)
        estimator.depth_config = depth_config
        estimator._last_inference_ms = 0.0
        depth_map = np.full((100, 240), 5.0, dtype=np.float32)
        depth_map[:, 160:] = 20.0
        estimator.estimate_depth_map = lambda _frame: depth_map
        backend = RecordedMotionObjectDetector(
            camera,
            SimpleNamespace(
                config=SimpleNamespace(confidence_threshold=0.5, depth=depth_config),
                estimate_depth_for_objects=estimator.enrich_objects,
            ),
            SimpleNamespace(), lambda: None,
        )
        result = backend._recorded_result(
            selected, objects, samples, {"detection_enrichment_ms": 0.0}, time.monotonic(),
            refinement_pending=False, event_epoch=event_at.timestamp(),
        )
        enriched = {item["label"]: item for item in result.objects}
        self.assertTrue(enriched["person"]["incident_eligible"])
        self.assertFalse(enriched["car"]["incident_eligible"])
        self.assertFalse(enriched["car"]["temporal_eligible"])
        self.assertFalse(enriched["dog"]["incident_eligible"])
        self.assertTrue(enriched["dog"]["depth_filtered"])
        self.assertEqual({item["depth_zone_matched"] for item in enriched.values()}, {"Approach"})

        events = Mock()
        events.add_event.return_value = {"id": 1}
        publish = Mock()
        handler = MotionDecisionHandler(
            camera.id, events, lambda _at: result,
            lambda _frame, _at: "snapshot.jpg", json.dumps,
            event_callback=publish,
        )
        outcome = handler.handle("manual", "test", event_at, {}, require_eligible_object=True)
        self.assertEqual([item["label"] for item in outcome.detected_objects], ["person"])
        payload = next(call.args[1] for call in publish.call_args_list if call.args[0] == "object")
        self.assertEqual([item["label"] for item in payload["incident_objects"]], ["person"])
        stored = json.loads(events.add_event.call_args.kwargs["objects_json"])
        self.assertEqual(
            {item["label"]: item["incident_eligible"] for item in stored if item.get("label")},
            {"person": True, "car": False, "dog": False},
        )

from __future__ import annotations

import unittest

import numpy as np

from survng.app.config import DepthConfig, DetectionZone
from survng.app.depth_estimation import (
    depth_motion_evidence_values,
    encode_depth_heatmap,
    sample_bbox_depth_stats,
    scale_depth_map,
)
from survng.app.zones import apply_depth_zone_filters


class DepthEstimationTests(unittest.TestCase):
    def test_sample_bbox_depth_stats_ignores_invalid_pixels(self) -> None:
        depth_map = np.array(
            [
                [1.0, 2.0, 99.0],
                [3.0, 4.0, np.nan],
                [5.0, 6.0, 7.0],
            ],
            dtype=np.float32,
        )
        stats = sample_bbox_depth_stats(
            depth_map,
            {"x1": 0, "y1": 0, "x2": 2, "y2": 2},
            min_m=0.5,
            max_m=10.0,
        )
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats["median_m"], 2.5)
        self.assertEqual(stats["min_m"], 1.0)
        self.assertEqual(stats["max_m"], 4.0)

    def test_scale_depth_map_reverses_letterbox(self) -> None:
        depth_map = np.arange(16, dtype=np.float32).reshape(4, 4)
        metadata = {
            "scale": 0.5,
            "pad_x": 1.0,
            "pad_y": 1.0,
            "input_width": 4.0,
            "input_height": 4.0,
            "image_width": 4.0,
            "image_height": 4.0,
        }
        restored = scale_depth_map(depth_map, (4, 4), metadata)
        self.assertEqual(restored.shape, (4, 4))

    def test_encode_depth_heatmap_returns_png_bytes(self) -> None:
        depth_map = np.linspace(1.0, 10.0, 16, dtype=np.float32).reshape(4, 4)
        encoded = encode_depth_heatmap(depth_map, max_width=4)
        self.assertTrue(encoded.startswith(b"\x89PNG"))

    def test_enrich_objects_can_force_debug_heatmap(self) -> None:
        from survng.app.depth_estimation import OpenVinoDepthEstimator

        estimator = OpenVinoDepthEstimator.__new__(OpenVinoDepthEstimator)
        estimator.depth_config = DepthConfig(store_heatmap=False, heatmap_max_width=64)
        estimator._last_inference_ms = 12.0
        depth_map = np.linspace(2.0, 8.0, 16, dtype=np.float32).reshape(4, 4)
        estimator.estimate_depth_map = lambda _frame: depth_map  # type: ignore[method-assign]
        objects = [{"label": "person", "box": {"x1": 0, "y1": 0, "x2": 2, "y2": 2}}]
        enriched, metadata = estimator.enrich_objects(
            np.zeros((4, 4, 3), dtype=np.uint8),
            objects,
            include_heatmap=True,
        )
        self.assertEqual(enriched[0]["depth_stats"]["median_m"], 3.0)
        self.assertTrue(metadata.get("heatmap_png", b"").startswith(b"\x89PNG"))
        self.assertEqual(metadata.get("heatmap_range_m"), {"min_m": 0.05, "max_m": 150.0})

    def test_depth_motion_evidence_values(self) -> None:
        values = depth_motion_evidence_values(
            [
                {"depth_stats": {"median_m": 4.0}},
                {"depth_stats": {"median_m": 8.0}},
            ],
            captured_at=100.0,
            frame_offset_s=0.5,
        )
        self.assertEqual(values["nearest_m"], 4.0)
        self.assertEqual(values["farthest_m"], 8.0)
        self.assertEqual(values["median_m"], 6.0)
        self.assertEqual(values["foreground_score"], 0.867)
        self.assertEqual(values["score"], values["foreground_score"])
        self.assertEqual(values["warmed"], 1.0)

    def test_apply_depth_zone_filters_incident_band(self) -> None:
        from survng.app.config import CameraConfig

        camera = CameraConfig(
            id="cam",
            name="Cam",
            stream_url="rtsp://example.invalid/main",
            zones=[
                DetectionZone(
                    name="Near Door",
                    behavior="incident",
                    min_depth_m=1.0,
                    max_depth_m=6.0,
                    object_classes=["person"],
                    points=[{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
                )
            ],
        )
        objects = [
            {
                "label": "person",
                "depth_stats": {"median_m": 3.5},
                "incident_eligible": False,
                "spatial_zones": ["Near Door"],
            },
            {
                "label": "person",
                "depth_stats": {"median_m": 20.0},
                "incident_eligible": True,
                "spatial_zones": ["Near Door"],
            },
        ]
        filtered = apply_depth_zone_filters(camera, objects)
        self.assertTrue(filtered[0]["incident_eligible"])
        self.assertEqual(filtered[0]["depth_zone_matched"], "Near Door")
        self.assertTrue(filtered[1]["incident_eligible"])
        self.assertNotIn("depth_zone_matched", filtered[1])

    def test_apply_depth_zone_filters_requires_matching_spatial_zone_and_prioritizes_ignore(self) -> None:
        from survng.app.config import CameraConfig

        camera = CameraConfig(
            id="cam",
            name="Cam",
            stream_url="rtsp://example.invalid/main",
            zones=[
                DetectionZone(name="Near", behavior="incident", min_depth_m=1.0, max_depth_m=6.0),
                DetectionZone(name="Private", behavior="ignore", min_depth_m=1.0, max_depth_m=6.0),
            ],
        )
        outside, overlapping = apply_depth_zone_filters(camera, [
            {"label": "person", "depth_stats": {"median_m": 3.5}, "incident_eligible": False, "spatial_zones": ["Elsewhere"]},
            {"label": "person", "depth_stats": {"median_m": 3.5}, "incident_eligible": True, "spatial_zones": ["Near", "Private"]},
        ])

        self.assertFalse(outside["incident_eligible"])
        self.assertNotIn("depth_zone_matched", outside)
        self.assertFalse(overlapping["incident_eligible"])
        self.assertTrue(overlapping["depth_zone_filtered"])


class DepthConfigTests(unittest.TestCase):
    def test_depth_config_defaults(self) -> None:
        config = DepthConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.input_size, 768)
        self.assertEqual(config.max_distance_m, 150.0)

    def test_depth_ranges_must_not_be_inverted(self) -> None:
        from pydantic import ValidationError

        with self.assertRaisesRegex(ValidationError, "minimum distance"):
            DepthConfig(min_distance_m=10.0, max_distance_m=2.0)
        with self.assertRaisesRegex(ValidationError, "minimum depth"):
            DetectionZone(name="Near", min_depth_m=10.0, max_depth_m=2.0)


if __name__ == "__main__":
    unittest.main()

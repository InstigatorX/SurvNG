from __future__ import annotations

import unittest

from survng.app.config import CameraConfig, DetectionZone
from survng.app.zones import apply_detection_zones


class DetectionZoneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = CameraConfig(
            id="lower-garage",
            name="Lower Garage",
            stream_url="rtsp://example.invalid/main",
            zones=[DetectionZone(
                name="Lower Driveway",
                object_classes=["car"],
                points=[
                    {"x": 0.0, "y": 0.8817273328921245},
                    {"x": 0.0, "y": 1.0},
                    {"x": 0.7305788015962833, "y": 1.0},
                    {"x": 0.9135408600869617, "y": 0.5467769688947717},
                    {"x": 0.5384031210911907, "y": 0.3815287888815354},
                ],
            )],
        )

    def test_bottom_center_on_zone_boundary_is_included(self) -> None:
        objects = [{
            "label": "car",
            "confidence": 0.9126,
            "box": {"x1": 438, "y1": 897, "x2": 1914, "y2": 2160},
        }]

        apply_detection_zones(self.camera, objects, 3840, 2160, 0.35)

        self.assertEqual(objects[0]["zone_point"], {"x": 0.30625, "y": 1.0})
        self.assertEqual(objects[0]["zones"], ["Lower Driveway"])
        self.assertTrue(objects[0]["incident_eligible"])

    def test_bottom_boundary_outside_polygon_remains_excluded(self) -> None:
        objects = [{
            "label": "car",
            "confidence": 0.95,
            "box": {"x1": 2880, "y1": 897, "x2": 3264, "y2": 2160},
        }]

        apply_detection_zones(self.camera, objects, 3840, 2160, 0.35)

        self.assertEqual(objects[0]["zone_point"], {"x": 0.8, "y": 1.0})
        self.assertEqual(objects[0]["zones"], [])
        self.assertFalse(objects[0]["incident_eligible"])


if __name__ == "__main__":
    unittest.main()

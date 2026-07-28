from __future__ import annotations

import unittest

from survng.app.config import CameraConfig, DetectionZone
from survng.app.zones import apply_detection_zones, detection_threshold


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

    def test_full_frame_mode_allows_object_outside_incident_zone(self) -> None:
        self.camera.require_incident_zone = False
        objects = [{
            "label": "car",
            "confidence": 0.95,
            "box": {"x1": 2880, "y1": 897, "x2": 3264, "y2": 2160},
        }]

        apply_detection_zones(self.camera, objects, 3840, 2160, 0.35)

        self.assertEqual(objects[0]["zones"], [])
        self.assertTrue(objects[0]["incident_eligible"])

    def test_global_full_frame_mode_is_used_when_camera_inherits(self) -> None:
        objects = [{
            "label": "car",
            "confidence": 0.95,
            "box": {"x1": 2880, "y1": 897, "x2": 3264, "y2": 2160},
        }]

        apply_detection_zones(
            self.camera,
            objects,
            3840,
            2160,
            0.35,
            require_incident_zone=False,
        )

        self.assertTrue(objects[0]["incident_eligible"])

    def test_camera_can_require_zone_when_global_mode_allows_anywhere(self) -> None:
        self.camera.require_incident_zone = True
        objects = [{
            "label": "car",
            "confidence": 0.95,
            "box": {"x1": 2880, "y1": 897, "x2": 3264, "y2": 2160},
        }]

        apply_detection_zones(
            self.camera,
            objects,
            3840,
            2160,
            0.35,
            require_incident_zone=False,
        )

        self.assertFalse(objects[0]["incident_eligible"])

    def test_full_frame_mode_preserves_zone_specific_lower_threshold(self) -> None:
        self.camera.require_incident_zone = False
        self.camera.zones[0].confidence_threshold = 0.2
        objects = [{
            "label": "car",
            "confidence": 0.3,
            "box": {"x1": 438, "y1": 897, "x2": 1914, "y2": 2160},
        }]

        apply_detection_zones(self.camera, objects, 3840, 2160, 0.45)

        self.assertEqual(objects[0]["zones"], ["Lower Driveway"])
        self.assertTrue(objects[0]["incident_eligible"])

    def test_ignore_zone_still_wins_in_full_frame_mode(self) -> None:
        self.camera.require_incident_zone = False
        self.camera.zones[0].behavior = "ignore"
        objects = [{
            "label": "car",
            "confidence": 0.95,
            "box": {"x1": 438, "y1": 897, "x2": 1914, "y2": 2160},
        }]

        apply_detection_zones(self.camera, objects, 3840, 2160, 0.35)

        self.assertEqual(objects[0]["zones"], ["Lower Driveway"])
        self.assertFalse(objects[0]["incident_eligible"])

    def test_reapplication_clears_stale_zone_metadata(self) -> None:
        detected = {
            "label": "car",
            "confidence": 0.95,
            "box": {"x1": 2880, "y1": 897, "x2": 3264, "y2": 2160},
            "zones": ["stale"],
            "zone_matches": [{"name": "stale"}],
            "zone_point": {"x": 0.1, "y": 0.1},
            "incident_eligible": True,
        }

        apply_detection_zones(self.camera, [detected], 3840, 2160, 0.35)

        self.assertEqual(detected["zones"], [])
        self.assertEqual(detected["zone_matches"], [])
        self.assertEqual(detected["zone_point"], {"x": 0.8, "y": 1.0})
        self.assertFalse(detected["incident_eligible"])

    def test_malformed_detector_values_fail_closed_without_crashing(self) -> None:
        objects = [
            {"label": "car", "confidence": "bad", "box": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
            {"label": "car", "confidence": 0.9, "box": {"x1": 0, "y1": 0, "x2": float("nan"), "y2": 1}},
            {"label": "car", "confidence": 0.9, "box": "bad"},
        ]

        apply_detection_zones(self.camera, objects, 3840, 2160, 0.35)

        self.assertTrue(all(item["incident_eligible"] is False for item in objects))
        self.assertTrue(all(item["zones"] == [] for item in objects))

    def test_non_mapping_detector_entry_is_ignored_without_crashing(self) -> None:
        malformed = ["legacy"]

        result = apply_detection_zones(self.camera, malformed, 3840, 2160, 0.35)  # type: ignore[arg-type]

        self.assertEqual(result, ["legacy"])

    def test_detection_threshold_uses_lowest_enabled_zone_threshold(self) -> None:
        self.camera.zones[0].confidence_threshold = 0.2
        self.assertEqual(detection_threshold(self.camera, 0.45), 0.2)
        self.camera.zones[0].enabled = False
        self.assertEqual(detection_threshold(self.camera, 0.45), 0.45)

    def test_no_zone_configuration_also_clears_stale_annotations(self) -> None:
        self.camera.zones = []
        detected = {
            "label": "car",
            "confidence": 0.9,
            "zones": ["old"],
            "zone_matches": [{"name": "old"}],
            "zone_point": {"x": 0.2, "y": 0.3},
            "incident_eligible": False,
        }

        apply_detection_zones(self.camera, [detected], 3840, 2160, 0.35)

        self.assertEqual(detected["zones"], [])
        self.assertEqual(detected["zone_matches"], [])
        self.assertNotIn("zone_point", detected)
        self.assertTrue(detected["incident_eligible"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from survng.app.camera_routes import match_camera_route
from survng.app.config import CameraTransitionRoute


class CameraRouteMatchTest(unittest.TestCase):
    def test_direction_and_time_window_are_required(self) -> None:
        route = CameraTransitionRoute(
            from_camera="back-left",
            to_camera="gate",
            min_seconds=2,
            max_seconds=12,
        )

        match = match_camera_route([route], "back-left", "gate", 4)
        self.assertIsNotNone(match)
        self.assertEqual(match.from_camera, "back-left")
        self.assertIsNone(match_camera_route([route], "gate", "back-left", 4))
        self.assertIsNone(match_camera_route([route], "back-left", "gate", 20))

    def test_bidirectional_route_accepts_reverse_travel(self) -> None:
        route = CameraTransitionRoute(
            from_camera="gate",
            to_camera="lower-garage",
            max_seconds=20,
            bidirectional=True,
            name="Driveway",
        )

        match = match_camera_route([route], "lower-garage", "gate", 8)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "Driveway")
        self.assertEqual(match.as_dict()["route_to_camera"], "gate")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import cv2
import numpy as np

from survng.app.motion import qualify_motion


class MotionQualificationTest(unittest.TestCase):
    def test_coherent_interior_motion_is_accepted(self) -> None:
        frames = []
        for index in range(9):
            frame = np.zeros((180, 320), dtype=np.uint8)
            cv2.rectangle(frame, (70 + index * 8, 60), (105 + index * 8, 125), 255, -1)
            frames.append(frame)

        result = qualify_motion(frames, "balanced")

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "qualified")
        self.assertGreater(result.features["continuity"], 0.8)

    def test_erratic_edge_motion_is_rejected_at_balanced_sensitivity(self) -> None:
        frames = []
        for x, y in [(4, 5), (15, 40), (3, 90), (18, 130), (5, 50), (20, 160), (2, 100), (16, 15), (4, 70)]:
            frame = np.zeros((180, 320), dtype=np.uint8)
            cv2.circle(frame, (x, y), 8, 255, -1)
            frames.append(frame)

        result = qualify_motion(frames, "balanced")

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "edge_motion")

    def test_global_brightness_change_is_rejected(self) -> None:
        frames = [np.full((180, 320), index * 20, dtype=np.uint8) for index in range(9)]

        result = qualify_motion(frames)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "global_change")

    def test_insufficient_frames_fail_open(self) -> None:
        result = qualify_motion([np.zeros((180, 320), dtype=np.uint8)] * 3)

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "insufficient_frames")


if __name__ == "__main__":
    unittest.main()

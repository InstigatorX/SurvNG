from __future__ import annotations

import unittest

import cv2
import numpy as np

from survng.app.motion import BackgroundMotionTracker, aggregate_mog2_evidence, qualify_motion


class MotionQualificationTest(unittest.TestCase):
    def test_mog2_tracker_persists_slow_foreground_blob(self) -> None:
        tracker = BackgroundMotionTracker(sample_fps=5, history_seconds=20)
        background = np.zeros((180, 320), dtype=np.uint8)
        for _ in range(12):
            tracker.update(background)

        evidence = []
        for index in range(9):
            frame = background.copy()
            cv2.rectangle(frame, (35 + index * 3, 55), (80 + index * 3, 145), 255, -1)
            evidence.append(tracker.update(frame))

        aggregate = aggregate_mog2_evidence(evidence)
        self.assertEqual(aggregate["mog2_warmed"], 1.0)
        self.assertGreater(aggregate["mog2_track_persistence"], 0.7)
        self.assertGreater(aggregate["mog2_track_hits"], 5)
        self.assertGreater(aggregate["mog2_score"], 0.7)
        self.assertEqual(len(aggregate["mog2_tracks"]), 1)
        track = aggregate["mog2_tracks"][0]
        self.assertEqual(track["id"], 1)
        self.assertEqual(len(track["box"]), 4)
        self.assertGreater(len(track["path"]), 5)
        self.assertTrue(all(0.0 <= coordinate <= 1.0 for point in track["path"] for coordinate in point))

    def test_mog2_tracker_reports_warmup_without_motion_decision(self) -> None:
        tracker = BackgroundMotionTracker(sample_fps=5, history_seconds=20)
        evidence = [tracker.update(np.zeros((90, 160), dtype=np.uint8)) for _ in range(3)]

        aggregate = aggregate_mog2_evidence(evidence)
        self.assertEqual(aggregate["mog2_warmed"], 0.0)
        self.assertNotIn("mog2_score", aggregate)

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

    def test_coherent_motion_entering_from_edge_is_accepted(self) -> None:
        frames = []
        for index in range(9):
            frame = np.zeros((180, 320), dtype=np.uint8)
            bottom = 195 - index * 9
            cv2.rectangle(frame, (145, bottom - 55), (180, bottom), 255, -1)
            frames.append(frame)

        result = qualify_motion(frames, "balanced")

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "qualified")
        self.assertGreater(result.features["inward_progress"], 0.4)

    def test_stable_track_along_edge_receives_coherence_relief(self) -> None:
        frames = []
        for index in range(9):
            frame = np.zeros((270, 480), dtype=np.uint8)
            cv2.rectangle(frame, (390 - index * 10, 225), (430 - index * 10, 260), 255, -1)
            frames.append(frame)

        result = qualify_motion(frames, "balanced")

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "coherent_edge_track")
        self.assertGreater(result.features["coherent_edge_track"], 0.4)

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

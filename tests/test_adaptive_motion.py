from __future__ import annotations

import unittest

import cv2
import numpy as np

from survng.app.motion_pipeline import (
    MotionContext,
    MotionPipelineFactory,
    MotionStageConfig,
    adaptive_motion_stage_configs,
    build_builtin_motion_registry,
)
from survng.app.motion_types import MotionBlob, MotionFrameBlobs


def moving_subject_frames(count: int = 10) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for index in range(count):
        frame = np.full((180, 320), 30, dtype=np.uint8)
        cv2.rectangle(frame, (60 + index * 7, 50), (95 + index * 7, 140), 180, -1)
        frames.append(frame)
    return frames


class AdaptiveMotionPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "gate",
            adaptive_motion_stage_configs(),
            required_artifacts={"scoring"},
        )

    def tearDown(self) -> None:
        self.pipeline.close()

    def process(
        self,
        frames: list[np.ndarray],
        *,
        captured_at: float = 100.0,
        configuration: dict | None = None,
    ) -> MotionContext:
        return self.pipeline.process(MotionContext(
            camera_id="gate",
            captured_at=captured_at,
            original_frame=frames[-1],
            frame_history=tuple(frames),
            configuration={
                "sensitivity": "balanced",
                "sample_fps": 5.0,
                **(configuration or {}),
            },
            runtime=self.pipeline.runtime,
        ))

    def process_timed(
        self,
        frames: list[np.ndarray],
        timestamps: list[float],
    ) -> MotionContext:
        return self.pipeline.process(MotionContext(
            camera_id="gate",
            captured_at=timestamps[-1],
            original_frame=frames[-1],
            frame_history=tuple(frames),
            frame_timestamps=tuple(timestamps),
            configuration={"sensitivity": "balanced", "sample_fps": 5.0},
            runtime=self.pipeline.runtime,
        ))

    def test_coherent_subject_is_accepted_with_adaptive_artifacts(self) -> None:
        result = self.process(moving_subject_frames())

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "qualified")
        self.assertIsNotNone(result.background_image)
        self.assertEqual(len(result.difference_history), 9)
        self.assertEqual(len(result.threshold_mask_history), 9)
        self.assertTrue(result.blobs)
        self.assertTrue(result.tracked_objects)
        self.assertGreater(result.dominant_track.consecutive_frames, 2)
        self.assertTrue(any(track.velocity != (0.0, 0.0) for track in result.tracked_objects))
        self.assertIn("adaptive_thresholds", result.debug.values)
        self.assertIn("scene_noise", result.scoring.features)

    def test_global_illumination_change_is_rejected_and_learned_quickly(self) -> None:
        frames = [
            np.full((180, 320), 20 + index * 15, dtype=np.uint8)
            for index in range(10)
        ]

        result = self.process(frames)

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "global_illumination_change")
        self.assertGreater(result.scoring.features["global_change"], 0.55)
        self.assertGreater(
            max(result.debug.values["background_learning_rates"]),
            0.1,
        )

    def test_random_sensor_noise_does_not_create_motion(self) -> None:
        random = np.random.default_rng(4)
        frames = [
            np.clip(30 + random.normal(0, 5, (180, 320)), 0, 255).astype(np.uint8)
            for _ in range(10)
        ]

        result = self.process(frames)

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "no_motion_blobs")
        self.assertGreater(result.debug.values["threshold_noise"], 1.0)

    def test_tiny_erratic_edge_motion_receives_insect_penalty(self) -> None:
        frames: list[np.ndarray] = []
        positions = [(1, 2), (8, 20), (2, 50), (10, 80), (2, 120), (11, 60), (1, 20), (9, 100), (2, 50), (8, 150)]
        for x, y in positions:
            frame = np.zeros((180, 320), dtype=np.uint8)
            cv2.circle(frame, (x, y), 3, 255, -1)
            frames.append(frame)

        result = self.process(frames)

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "insect_like_motion")
        self.assertGreaterEqual(result.scoring.features["insect_penalty"], 0.55)

    def test_ignore_zone_filters_overlapping_components(self) -> None:
        zones = [{
            "name": "tree",
            "enabled": True,
            "behavior": "ignore",
            "points": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.55, "y": 0.0},
                {"x": 0.55, "y": 1.0},
                {"x": 0.0, "y": 1.0},
            ],
        }]

        result = self.process(
            moving_subject_frames(),
            configuration={"motion_zones": zones},
        )

        self.assertFalse(result.scoring.accepted)
        self.assertGreater(result.debug.values["blob_rejections"]["zone"], 0)

    def test_background_and_tracks_are_isolated_per_camera_pipeline(self) -> None:
        other = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "foyer",
            adaptive_motion_stage_configs(),
        )
        try:
            gate = self.process(moving_subject_frames())
            dark = [np.zeros((180, 320), dtype=np.uint8) for _ in range(10)]
            foyer = other.process(MotionContext(
                camera_id="foyer",
                captured_at=100.0,
                original_frame=dark[-1],
                frame_history=tuple(dark),
                configuration={"sensitivity": "balanced", "sample_fps": 5.0},
                runtime=other.runtime,
            ))

            self.assertGreater(float(np.mean(gate.background_image)), 0.0)
            self.assertEqual(float(np.mean(foyer.background_image)), 0.0)
            self.assertNotEqual(self.pipeline.runtime, other.runtime)
        finally:
            other.close()

    def test_tracker_expires_motion_across_a_gap_longer_than_configured(self) -> None:
        tracker = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "gate",
            [MotionStageConfig(
                "tracking",
                "persistent_centroid_tracker",
                {"maximum_missed_frames": 3},
            )],
            initial_artifacts={"filtered_blob_history"},
        )
        blob = MotionBlob(
            box=(0.2, 0.2, 0.3, 0.5),
            centroid=(0.25, 0.35),
            area_pixels=100,
            area_ratio=0.01,
        )
        visible = MotionFrameBlobs(10000, 100, 0.01, (blob,))
        empty = MotionFrameBlobs(10000, 0, 0.0, ())
        history = (visible, visible, empty, empty, empty, empty, empty, visible, visible, visible)
        try:
            result = tracker.process(MotionContext(
                camera_id="gate",
                captured_at=102.0,
                original_frame=None,
                configuration={"sample_fps": 5.0},
                runtime=tracker.runtime,
                filtered_blob_history=history,
            ))
            self.assertLessEqual(result.dominant_track.consecutive_frames, 3)
        finally:
            tracker.close()

    def test_overlapping_replay_does_not_retrain_adaptive_threshold(self) -> None:
        frames = moving_subject_frames(6)
        timestamps = [100.0 + index * 0.2 for index in range(len(frames))]
        self.process_timed(frames, timestamps)
        threshold_state = self.pipeline.runtime.stage_state["threshold"]
        before = (threshold_state.threshold_ema, threshold_state.noise_ema)

        self.process_timed(frames, timestamps)

        self.assertEqual(
            (threshold_state.threshold_ema, threshold_state.noise_ema),
            before,
        )

    def test_motion_score_accumulates_across_incremental_invocations(self) -> None:
        frames = moving_subject_frames(5)
        first = self.process_timed(frames[:3], [100.0, 100.2, 100.4])
        second = self.process_timed(frames[2:5], [100.4, 100.6, 100.8])

        self.assertGreater(
            second.dominant_track.accumulated_score,
            first.dominant_track.accumulated_score,
        )

    def test_blob_reports_only_the_zone_it_overlaps(self) -> None:
        zones = [
            {
                "name": "left",
                "enabled": True,
                "behavior": "incident",
                "points": [{"x": 0, "y": 0}, {"x": 0.5, "y": 0}, {"x": 0.5, "y": 1}, {"x": 0, "y": 1}],
            },
            {
                "name": "right",
                "enabled": True,
                "behavior": "incident",
                "points": [{"x": 0.5, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0.5, "y": 1}],
            },
        ]

        result = self.process(moving_subject_frames(4), configuration={"motion_zones": zones})

        self.assertTrue(result.blobs)
        self.assertTrue(all("right" not in blob.zone_names for blob in result.blobs))
        self.assertTrue(any(blob.zone_names == ("left",) for blob in result.blobs))


if __name__ == "__main__":
    unittest.main()

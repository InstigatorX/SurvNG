from __future__ import annotations

import unittest

import cv2
import numpy as np

from survng.app.motion_pipeline import (
    MotionContext,
    MotionPipelineFactory,
    MotionScoring,
    MotionStageConfig,
    adaptive_motion_stage_configs,
    build_builtin_motion_registry,
)
from survng.app.motion_types import MotionBlob, MotionFrameBlobs, MotionTrack


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

    def test_global_change_scoring_uses_background_stage_threshold(self) -> None:
        stages = adaptive_motion_stage_configs()
        background = stages[1]
        stages[1] = MotionStageConfig(
            background.stage_id,
            background.implementation,
            {**background.options, "global_change_ratio": 0.9},
        )
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "gate-custom",
            stages,
        )
        frames: list[np.ndarray] = []
        for index in range(10):
            frame = np.full((180, 320), 20, dtype=np.uint8)
            frame[:, :190] = 20 + index * 15
            frames.append(frame)
        try:
            result = pipeline.process(MotionContext(
                camera_id="gate-custom",
                captured_at=100.0,
                original_frame=frames[-1],
                frame_history=tuple(frames),
                configuration={"sensitivity": "balanced", "sample_fps": 5.0},
                runtime=pipeline.runtime,
            ))

            self.assertEqual(result.debug.values["global_change_threshold"], 0.9)
            self.assertEqual(result.scoring.features["global_change_threshold"], 0.9)
            self.assertNotEqual(result.scoring.reason, "global_illumination_change")
        finally:
            pipeline.close()

    def test_background_learning_accounts_for_time_between_analysis_cycles(self) -> None:
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "elapsed-background",
            [MotionStageConfig("background", "adaptive_ema_background")],
            initial_artifacts={"processed_frame_history"},
        )
        frame = np.full((40, 60), 30, dtype=np.uint8)
        try:
            first = pipeline.process(MotionContext(
                camera_id="elapsed-background",
                captured_at=100.2,
                original_frame=frame,
                configuration={"sample_fps": 5.0},
                runtime=pipeline.runtime,
                processed_frame_history=(frame, frame),
                frame_timestamps=(100.0, 100.2),
            ))
            second = pipeline.process(MotionContext(
                camera_id="elapsed-background",
                captured_at=101.0,
                original_frame=frame,
                configuration={"sample_fps": 5.0},
                runtime=pipeline.runtime,
                processed_frame_history=(frame, frame),
                frame_timestamps=(100.8, 101.0),
            ))
        finally:
            pipeline.close()

        self.assertAlmostEqual(first.debug.values["background_learning_rates"][-1], 0.025, places=3)
        self.assertGreater(second.debug.values["background_learning_rates"][-1], 0.09)

    def test_persistent_foreground_is_selectively_learned_faster(self) -> None:
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "persistent-background",
            [MotionStageConfig("background", "adaptive_ema_background", {
                "motion_learning_scale": 0.01,
                "stationary_learning_seconds": 1.0,
                "stationary_learning_rate": 0.2,
            })],
            initial_artifacts={"processed_frame_history"},
        )
        base = np.zeros((50, 80), dtype=np.uint8)
        changed = base.copy()
        cv2.rectangle(changed, (20, 15), (45, 35), 180, -1)
        try:
            pipeline.process(MotionContext(
                camera_id="persistent-background",
                captured_at=100.2,
                original_frame=base,
                configuration={"sample_fps": 5.0},
                runtime=pipeline.runtime,
                processed_frame_history=(base, base),
                frame_timestamps=(100.0, 100.2),
            ))
            pipeline.process(MotionContext(
                camera_id="persistent-background",
                captured_at=100.8,
                original_frame=changed,
                configuration={"sample_fps": 5.0},
                runtime=pipeline.runtime,
                processed_frame_history=(changed, changed),
                frame_timestamps=(100.6, 100.8),
            ))
            result = pipeline.process(MotionContext(
                camera_id="persistent-background",
                captured_at=101.4,
                original_frame=changed,
                configuration={"sample_fps": 5.0},
                runtime=pipeline.runtime,
                processed_frame_history=(changed, changed),
                frame_timestamps=(101.2, 101.4),
            ))
        finally:
            pipeline.close()

        self.assertGreater(result.debug.values["background_persistent_change_ratios"][-1], 0.0)
        self.assertGreater(result.debug.values["background_moving_learning_rates"][-1], 0.45)

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

    def test_new_stationary_foreground_is_not_credible_motion(self) -> None:
        base = np.full((180, 320), 30, dtype=np.uint8)
        changed = base.copy()
        cv2.rectangle(changed, (70, 50), (120, 140), 180, -1)
        previous = base
        result: MotionContext | None = None
        for index in range(20):
            captured_at = 100.2 + index * 0.2
            result = self.process_timed(
                [previous, changed],
                [captured_at - 0.2, captured_at],
            )
            previous = changed

        self.assertIsNotNone(result)
        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "stationary_foreground")
        self.assertEqual(result.scoring.features["net_displacement"], 0.0)

    def test_illumination_filter_rejects_clear_brightness_change_when_enabled(self) -> None:
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "light-change",
            [MotionStageConfig(
                "illumination",
                "illumination_change_filter",
                {"minimum_evidence_frames": 2, "rejection_threshold": 0.82},
            )],
            initial_artifacts={"motion_mask_history", "scoring"},
        )
        try:
            rng = np.random.default_rng(7)
            texture = rng.integers(35, 220, (90, 160), dtype=np.uint8)
            base = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
            darker = np.clip(base.astype(np.float32) * 0.62, 0, 255).astype(np.uint8)
            darkest = np.clip(base.astype(np.float32) * 0.42, 0, 255).astype(np.uint8)
            mask = np.full(base.shape[:2], 255, dtype=np.uint8)
            result = pipeline.process(MotionContext(
                camera_id="light-change",
                captured_at=100.0,
                original_frame=darkest,
                frame_history=(base, darker, darkest),
                motion_mask_history=(mask, mask),
                configuration={"illumination_filter_enabled": True},
                runtime=pipeline.runtime,
                scoring=MotionScoring(
                    accepted=True,
                    score=0.82,
                    threshold=0.48,
                    reason="qualified",
                ),
            ))
        finally:
            pipeline.close()

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "illumination_change")
        self.assertTrue(result.scoring.features["illumination_would_reject"])

    def test_illumination_filter_observes_without_rejecting_when_disabled(self) -> None:
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "light-observe",
            [MotionStageConfig("illumination", "illumination_change_filter")],
            initial_artifacts={"motion_mask_history", "scoring"},
        )
        try:
            rng = np.random.default_rng(13)
            texture = rng.integers(35, 220, (90, 160), dtype=np.uint8)
            base = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
            darker = np.clip(base.astype(np.float32) * 0.65, 0, 255).astype(np.uint8)
            darkest = np.clip(base.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
            mask = np.full(base.shape[:2], 255, dtype=np.uint8)
            result = pipeline.process(MotionContext(
                camera_id="light-observe",
                captured_at=100.0,
                original_frame=darkest,
                frame_history=(base, darker, darkest),
                motion_mask_history=(mask, mask),
                configuration={"illumination_filter_enabled": False},
                runtime=pipeline.runtime,
                scoring=MotionScoring(
                    accepted=True,
                    score=0.82,
                    threshold=0.48,
                    reason="qualified",
                ),
            ))
        finally:
            pipeline.close()

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "qualified")
        self.assertTrue(result.scoring.features["illumination_would_reject"])

    def test_illumination_filter_fails_open_for_physical_structure_change(self) -> None:
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "physical-change",
            [MotionStageConfig("illumination", "illumination_change_filter")],
            initial_artifacts={"motion_mask_history", "scoring"},
        )
        try:
            rng = np.random.default_rng(11)
            base = rng.integers(20, 210, (90, 160, 3), dtype=np.uint8)
            first = base.copy()
            second = base.copy()
            cv2.rectangle(first, (25, 20), (65, 75), (20, 30, 230), -1)
            cv2.rectangle(second, (70, 20), (110, 75), (20, 30, 230), -1)
            first_mask = np.zeros(base.shape[:2], dtype=np.uint8)
            second_mask = np.zeros(base.shape[:2], dtype=np.uint8)
            cv2.rectangle(first_mask, (25, 20), (65, 75), 255, -1)
            cv2.rectangle(second_mask, (25, 20), (110, 75), 255, -1)
            result = pipeline.process(MotionContext(
                camera_id="physical-change",
                captured_at=100.0,
                original_frame=second,
                frame_history=(base, first, second),
                motion_mask_history=(first_mask, second_mask),
                configuration={"illumination_filter_enabled": True},
                runtime=pipeline.runtime,
                scoring=MotionScoring(
                    accepted=True,
                    score=0.82,
                    threshold=0.48,
                    reason="qualified",
                ),
            ))
        finally:
            pipeline.close()

        self.assertTrue(result.scoring.accepted)
        self.assertFalse(result.scoring.features["illumination_would_reject"])

    def test_stationary_object_tolerance_selects_safe_scoring_thresholds(self) -> None:
        scorer = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "tolerance",
            [MotionStageConfig("scoring", "adaptive_motion_score")],
            initial_artifacts={"tracked_objects", "processed_frame_history"},
        )
        try:
            for tolerance, expected in {
                "low": (0.006, 0.015),
                "balanced": (0.01, 0.025),
                "high": (0.02, 0.05),
            }.items():
                result = scorer.process(MotionContext(
                    camera_id="tolerance",
                    captured_at=100.0,
                    original_frame=None,
                    configuration={
                        "sensitivity": "balanced",
                        "stationary_object_tolerance": tolerance,
                    },
                    runtime=scorer.runtime,
                    processed_frame_history=(np.zeros((10, 10), dtype=np.uint8),),
                ))
                self.assertEqual(
                    result.debug.values["stationary_displacement_ratio"],
                    expected[0],
                )
                self.assertEqual(
                    result.debug.values["stationary_path_ratio"],
                    expected[1],
                )
        finally:
            scorer.close()

    def test_small_subpixel_jitter_is_not_credible_motion(self) -> None:
        frames: list[np.ndarray] = []
        for index in range(10):
            frame = np.full((180, 320), 30, dtype=np.uint8)
            x = 150 + (index % 2)
            cv2.rectangle(frame, (x, 90), (x + 6, 96), 210, -1)
            frames.append(frame)

        result = self.process(frames)

        self.assertFalse(result.scoring.accepted)
        self.assertIn(
            result.scoring.reason,
            {
                "micro_jitter",
                "insect_like_motion",
                "stationary_foreground",
                "low_persistence",
            },
        )

    def test_rejected_nuisance_track_does_not_hide_credible_motion_track(self) -> None:
        scorer = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "multi",
            [MotionStageConfig("scoring", "adaptive_motion_score")],
            initial_artifacts={"tracked_objects", "processed_frame_history"},
        )

        def track(track_id: int, path: tuple[tuple[float, float], ...], area: float) -> MotionTrack:
            blobs = tuple(
                MotionBlob(
                    box=(x, y, x + 0.08, y + 0.2),
                    centroid=(x + 0.04, y + 0.1),
                    area_pixels=area * 57600,
                    area_ratio=area,
                    fill_ratio=0.7,
                    edge_distance=0.2,
                )
                for x, y in path
            )
            return MotionTrack(
                track_id=track_id,
                box=blobs[-1].box,
                path=tuple(blob.centroid for blob in blobs),
                observations=blobs,
                observation_frames=tuple(range(len(blobs))),
                active_history=tuple(True for _ in blobs),
                changed_pixels=(),
                changed_ratios=(),
                first_seen=100.0,
                last_seen=100.0 + (len(blobs) - 1) * 0.2,
                consecutive_started_at=100.0,
                consecutive_frames=len(blobs),
                size_history=tuple(area for _ in blobs),
            )

        nuisance = track(1, ((0.3, 0.3),) * 5, 0.02)
        moving = track(2, ((0.1, 0.4), (0.12, 0.4)), 0.005)
        try:
            result = scorer.process(MotionContext(
                camera_id="multi",
                captured_at=101.0,
                original_frame=None,
                configuration={"sensitivity": "balanced", "sample_fps": 5.0},
                runtime=scorer.runtime,
                processed_frame_history=(np.zeros((10, 10), dtype=np.uint8),),
                tracked_objects=[nuisance, moving],
            ))
        finally:
            scorer.close()

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.dominant_track.track_id, 2)

    def test_slow_persistent_drift_is_stationary_foreground(self) -> None:
        scorer = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "drift",
            [MotionStageConfig("scoring", "adaptive_motion_score")],
            initial_artifacts={"tracked_objects", "processed_frame_history"},
        )
        path = tuple((0.4 + index * 0.0004, 0.4) for index in range(20))
        blobs = tuple(
            MotionBlob(
                box=(x, y, x + 0.1, y + 0.2),
                centroid=(x, y),
                area_pixels=576,
                area_ratio=0.01,
                fill_ratio=0.7,
                edge_distance=0.2,
            )
            for x, y in path
        )
        drift = MotionTrack(
            track_id=1,
            box=blobs[-1].box,
            path=path,
            observations=blobs,
            observation_frames=tuple(range(20)),
            active_history=tuple(True for _ in blobs),
            changed_pixels=(),
            changed_ratios=(),
            first_seen=100.0,
            last_seen=103.8,
            consecutive_started_at=100.0,
            consecutive_frames=20,
            size_history=tuple(0.01 for _ in blobs),
        )
        try:
            result = scorer.process(MotionContext(
                camera_id="drift",
                captured_at=103.8,
                original_frame=None,
                configuration={"sensitivity": "balanced", "sample_fps": 5.0},
                runtime=scorer.runtime,
                processed_frame_history=(np.zeros((10, 10), dtype=np.uint8),),
                tracked_objects=[drift],
            ))
        finally:
            scorer.close()

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "stationary_foreground")

    def test_parked_vehicle_contour_oscillation_is_not_credible_motion(self) -> None:
        scorer = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "parked-vehicle",
            [MotionStageConfig("scoring", "adaptive_motion_score")],
            initial_artifacts={"tracked_objects", "processed_frame_history"},
        )
        path = (
            (0.73, 0.72), (0.73, 0.66), (0.69, 0.63), (0.69, 0.63),
            (0.73, 0.72), (0.73, 0.66), (0.73, 0.66), (0.73, 0.66),
        )
        blobs = tuple(
            MotionBlob(
                box=(x - 0.05, y - 0.08, x + 0.05, y + 0.08),
                centroid=(x, y),
                area_pixels=730,
                area_ratio=0.0127,
                fill_ratio=0.46,
                edge_distance=0.2,
            )
            for x, y in path
        )
        parked = MotionTrack(
            track_id=1,
            box=blobs[-1].box,
            path=path,
            observations=blobs,
            observation_frames=tuple(range(len(blobs))),
            active_history=tuple(True for _ in blobs),
            changed_pixels=(),
            changed_ratios=(),
            first_seen=100.0,
            last_seen=103.5,
            consecutive_started_at=100.0,
            consecutive_frames=len(blobs),
            size_history=tuple(blob.area_ratio for blob in blobs),
        )
        try:
            result = scorer.process(MotionContext(
                camera_id="parked-vehicle",
                captured_at=103.5,
                original_frame=None,
                configuration={
                    "sensitivity": "balanced",
                    "sample_fps": 5.0,
                    "stationary_object_tolerance": "balanced",
                },
                runtime=scorer.runtime,
                processed_frame_history=(np.zeros((10, 10), dtype=np.uint8),),
                tracked_objects=[parked],
            ))
        finally:
            scorer.close()

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "stationary_foreground")
        self.assertLess(result.scoring.features["motion_progress"], 0.32)
        self.assertEqual(result.scoring.features["stationary_object_tolerance"], "balanced")

    def test_stationary_region_memory_survives_motion_track_id_reset(self) -> None:
        scorer = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "stationary-memory",
            [MotionStageConfig("scoring", "adaptive_motion_score")],
            initial_artifacts={"tracked_objects", "processed_frame_history"},
        )

        def motion_track(track_id: int, path: tuple[tuple[float, float], ...], started_at: float) -> MotionTrack:
            blobs = tuple(
                MotionBlob(
                    box=(x - 0.06, y - 0.08, x + 0.06, y + 0.08),
                    centroid=(x, y),
                    area_pixels=900,
                    area_ratio=0.016,
                    fill_ratio=0.7,
                    edge_distance=0.2,
                )
                for x, y in path
            )
            return MotionTrack(
                track_id=track_id,
                box=blobs[-1].box,
                path=path,
                observations=blobs,
                observation_frames=tuple(range(len(blobs))),
                active_history=tuple(True for _ in blobs),
                changed_pixels=(),
                changed_ratios=(),
                first_seen=started_at,
                last_seen=started_at + (len(blobs) - 1) * 0.2,
                consecutive_started_at=started_at,
                consecutive_frames=len(blobs),
                size_history=tuple(blob.area_ratio for blob in blobs),
            )

        def score(track: MotionTrack, captured_at: float) -> MotionContext:
            return scorer.process(MotionContext(
                camera_id="stationary-memory",
                captured_at=captured_at,
                original_frame=None,
                configuration={
                    "sensitivity": "balanced",
                    "sample_fps": 5.0,
                    "stationary_object_tolerance": "balanced",
                },
                runtime=scorer.runtime,
                processed_frame_history=(np.zeros((10, 10), dtype=np.uint8),),
                tracked_objects=[track],
            ))

        try:
            first = score(motion_track(1, ((0.44, 0.4),) * 5, 100.0), 100.8)
            second = score(
                motion_track(2, ((0.40, 0.4), (0.46, 0.4), (0.44, 0.4)), 102.0),
                102.4,
            )
        finally:
            scorer.close()

        self.assertEqual(first.scoring.reason, "stationary_foreground")
        self.assertFalse(second.scoring.accepted)
        self.assertEqual(second.scoring.reason, "stationary_region")
        self.assertGreaterEqual(second.scoring.features["stationary_region_count"], 1)

    def test_long_running_directional_motion_remains_credible(self) -> None:
        scorer = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "persistent",
            [MotionStageConfig("scoring", "adaptive_motion_score")],
            initial_artifacts={"tracked_objects", "processed_frame_history"},
        )
        path = tuple((0.2 + index * 0.005, 0.4) for index in range(40))
        blobs = tuple(
            MotionBlob(
                box=(x, y, x + 0.1, y + 0.2),
                centroid=(x, y),
                area_pixels=576,
                area_ratio=0.01,
                fill_ratio=0.7,
                edge_distance=0.2,
            )
            for x, y in path
        )
        persistent = MotionTrack(
            track_id=1,
            box=blobs[-1].box,
            path=path,
            observations=blobs,
            observation_frames=tuple(range(40)),
            active_history=tuple(True for _ in blobs),
            changed_pixels=(),
            changed_ratios=(),
            first_seen=100.0,
            last_seen=107.8,
            consecutive_started_at=100.0,
            consecutive_frames=40,
            size_history=tuple(0.01 for _ in blobs),
        )
        try:
            result = scorer.process(MotionContext(
                camera_id="persistent",
                captured_at=107.8,
                original_frame=None,
                configuration={"sensitivity": "balanced", "sample_fps": 5.0},
                runtime=scorer.runtime,
                processed_frame_history=(np.zeros((10, 10), dtype=np.uint8),),
                tracked_objects=[persistent],
            ))
        finally:
            scorer.close()

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "qualified")

    def test_long_running_contained_oscillation_becomes_scene_activity(self) -> None:
        scorer = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "persistent-oscillation",
            [MotionStageConfig("scoring", "adaptive_motion_score")],
            initial_artifacts={"tracked_objects", "processed_frame_history"},
        )
        path = tuple(
            (0.4 + (0.025 if index % 2 else -0.025), 0.4)
            for index in range(40)
        )
        blobs = tuple(
            MotionBlob(
                box=(x, y, x + 0.1, y + 0.2),
                centroid=(x, y),
                area_pixels=576,
                area_ratio=0.01,
                fill_ratio=0.7,
                edge_distance=0.2,
            )
            for x, y in path
        )
        persistent = MotionTrack(
            track_id=1,
            box=blobs[-1].box,
            path=path,
            observations=blobs,
            observation_frames=tuple(range(40)),
            active_history=tuple(True for _ in blobs),
            changed_pixels=(),
            changed_ratios=(),
            first_seen=100.0,
            last_seen=107.8,
            consecutive_started_at=100.0,
            consecutive_frames=40,
            size_history=tuple(0.01 for _ in blobs),
        )
        try:
            result = scorer.process(MotionContext(
                camera_id="persistent-oscillation",
                captured_at=107.8,
                original_frame=None,
                configuration={"sensitivity": "balanced", "sample_fps": 5.0},
                runtime=scorer.runtime,
                processed_frame_history=(np.zeros((10, 10), dtype=np.uint8),),
                tracked_objects=[persistent],
            ))
        finally:
            scorer.close()

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "persistent_scene_motion")

    def test_background_learning_rate_is_elapsed_time_based(self) -> None:
        base = np.full((180, 320), 30, dtype=np.uint8)
        brighter = np.full((180, 320), 90, dtype=np.uint8)
        short = self.process_timed([base, brighter], [100.0, 100.2])
        short_rate = short.debug.values["background_learning_rates"][-1]

        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "elapsed",
            adaptive_motion_stage_configs(),
            required_artifacts={"scoring"},
        )
        try:
            long = pipeline.process(MotionContext(
                camera_id="elapsed",
                captured_at=102.0,
                original_frame=brighter,
                frame_history=(base, brighter),
                frame_timestamps=(100.0, 102.0),
                configuration={"sensitivity": "balanced", "sample_fps": 5.0},
                runtime=pipeline.runtime,
            ))
        finally:
            pipeline.close()

        self.assertGreater(long.debug.values["background_learning_rates"][-1], short_rate)

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

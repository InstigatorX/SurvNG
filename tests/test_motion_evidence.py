from __future__ import annotations

import unittest

import cv2
import numpy as np

from survng.app.motion import aggregate_mog2_evidence
from survng.app.motion_pipeline import (
    EVIDENCE_REPOSITORY_SERVICE,
    MotionContext,
    MotionEvidenceRepository,
    MotionPipelineFactory,
    MotionScoring,
    MotionStageDependencies,
    build_builtin_motion_registry,
    motion_fusion_stage_configs,
    motion_observation_stage_configs,
)


def pipeline_factory(repository: MotionEvidenceRepository) -> MotionPipelineFactory:
    return MotionPipelineFactory(
        build_builtin_motion_registry(),
        dependencies=MotionStageDependencies(
            services={EVIDENCE_REPOSITORY_SERVICE: repository},
        ),
    )


class MotionEvidenceTest(unittest.TestCase):
    def test_repository_bounds_samples_and_filters_windows(self) -> None:
        repository = MotionEvidenceRepository("gate", max_samples_per_source=2)
        repository.append("mog2", 1.0, {"score": 0.1})
        repository.append("mog2", 2.0, {"score": 0.2})
        repository.append("mog2", 3.0, {"score": 0.3})

        samples = repository.window("mog2", 1.5, 2.5)

        self.assertEqual([sample.captured_at for sample in samples], [2.0])
        self.assertEqual(repository.last("mog2").values["score"], 0.3)
        self.assertEqual(repository.status()["mog2"]["sample_count"], 2)

    def test_mog2_source_owns_tracker_in_per_camera_runtime(self) -> None:
        repository = MotionEvidenceRepository("gate", max_samples_per_source=64)
        factory = pipeline_factory(repository)
        pipeline = factory.create(
            "gate",
            motion_observation_stage_configs(
                mog2_enabled=True,
                sample_fps=5.0,
                mog2_history_seconds=20.0,
            ),
        )
        frames = [np.zeros((180, 320), dtype=np.uint8) for _ in range(12)]
        for index in range(9):
            frame = np.zeros((180, 320), dtype=np.uint8)
            cv2.rectangle(frame, (35 + index * 3, 55), (80 + index * 3, 145), 255, -1)
            frames.append(frame)

        for index, frame in enumerate(frames):
            pipeline.process(
                MotionContext(
                    camera_id="gate",
                    captured_at=float(index),
                    original_frame=frame,
                    configuration={},
                    runtime=pipeline.runtime,
                )
            )

        aggregate = aggregate_mog2_evidence(
            [dict(sample.values) for sample in repository.window("mog2", 0.0, 30.0)]
        )
        self.assertEqual(aggregate["mog2_warmed"], 1.0)
        self.assertGreater(aggregate["mog2_track_hits"], 5)
        self.assertIn("mog2_source", pipeline.runtime.stage_state)

    def test_disabled_mog2_source_does_not_allocate_runtime_or_samples(self) -> None:
        repository = MotionEvidenceRepository("gate")
        factory = pipeline_factory(repository)
        pipeline = factory.create(
            "gate",
            motion_observation_stage_configs(
                mog2_enabled=False,
                sample_fps=5.0,
                mog2_history_seconds=30.0,
            ),
        )

        pipeline.process(
            MotionContext(
                camera_id="gate",
                captured_at=1.0,
                original_frame=np.zeros((90, 160), dtype=np.uint8),
                configuration={},
                runtime=pipeline.runtime,
            )
        )

        self.assertIsNone(repository.last("mog2"))
        self.assertEqual(pipeline.runtime.stage_state, {})
        self.assertFalse(repository.status()["mog2"]["enabled"])

    def test_fusion_preserves_primary_decision_and_adds_windowed_evidence(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append("mog2", 10.0, {"warmed": 0.0, "foreground_ratio": 0.1})
        repository.append("mog2", 11.0, {"warmed": 1.0, "score": 0.8, "track_hits": 6})
        repository.append("mog2", 20.0, {"warmed": 1.0, "score": 0.2, "track_hits": 2})
        factory = pipeline_factory(repository)
        pipeline = factory.create(
            "gate",
            motion_fusion_stage_configs(),
            initial_artifacts={"scoring"},
        )
        context = MotionContext(
            camera_id="gate",
            captured_at=12.0,
            original_frame=None,
            configuration={"evidence_started_at": 9.0, "evidence_ended_at": 12.0},
            runtime=pipeline.runtime,
            scoring=MotionScoring(
                accepted=False,
                score=0.42,
                threshold=0.48,
                reason="edge_motion",
                frame_count=9,
                features={"continuity": 0.2},
            ),
        )

        result = pipeline.process(context)

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.score, 0.42)
        self.assertEqual(result.scoring.reason, "edge_motion")
        self.assertEqual(result.scoring.features["mog2_score"], 0.8)
        self.assertEqual(result.scoring.features["mog2_track_hits"], 6)
        self.assertEqual(result.source_evidence["mog2"]["mog2_score"], 0.8)

    def test_fusion_requires_explicit_scoring_input_and_repository(self) -> None:
        repository = MotionEvidenceRepository("gate")
        factory = pipeline_factory(repository)

        with self.assertRaisesRegex(ValueError, "scoring"):
            factory.create("gate", motion_fusion_stage_configs())

        factory_without_repository = MotionPipelineFactory(build_builtin_motion_registry())
        with self.assertRaisesRegex(ValueError, EVIDENCE_REPOSITORY_SERVICE):
            factory_without_repository.create(
                "gate",
                motion_fusion_stage_configs(),
                initial_artifacts={"scoring"},
            )


if __name__ == "__main__":
    unittest.main()

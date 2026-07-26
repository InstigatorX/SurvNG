from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from survng.app.motion import aggregate_mog2_evidence
from survng.app.motion_pipeline import (
    EVIDENCE_REPOSITORY_SERVICE,
    MotionContext,
    MotionEvidenceRepository,
    MotionPipelineFactory,
    MotionScoring,
    MotionStageConfig,
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


def apply_fusion_policy(
    repository: MotionEvidenceRepository,
    *,
    policy: str,
    primary_accepted: bool,
    primary_score: float,
    options: dict[str, object] | None = None,
) -> MotionContext:
    pipeline = pipeline_factory(repository).create(
        "gate",
        [
            MotionStageConfig(
                "fusion",
                "buffered_evidence_fusion",
                {"sources": ["mog2"], "policy": policy, **(options or {})},
            )
        ],
        initial_artifacts={"scoring"},
    )
    return pipeline.process(
        MotionContext(
            camera_id="gate",
            captured_at=12.0,
            original_frame=None,
            configuration={"evidence_started_at": 9.0, "evidence_ended_at": 12.0},
            runtime=pipeline.runtime,
            scoring=MotionScoring(
                accepted=primary_accepted,
                score=primary_score,
                threshold=0.48,
                reason="qualified" if primary_accepted else "edge_motion",
                frame_count=9,
            ),
        )
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

    def test_repository_accepts_parallel_source_writers(self) -> None:
        repository = MotionEvidenceRepository("gate", max_samples_per_source=200)

        def append_sample(index: int) -> None:
            source = "mog2" if index % 2 else "onvif"
            repository.append(source, float(index), {"score": index / 100.0})

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(append_sample, range(100)))

        status = repository.status()
        self.assertEqual(status["mog2"]["sample_count"], 50)
        self.assertEqual(status["onvif"]["sample_count"], 50)

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
        self.assertNotIn("mog2", repository.status())
        self.assertFalse(pipeline.handles_observation("frame"))
        self.assertEqual(pipeline.status()["execution_groups"], [{
            "mode": "sequential",
            "stages": ["onvif_source"],
        }])

    def test_onvif_source_normalizes_motion_event_without_touching_mog2_runtime(self) -> None:
        repository = MotionEvidenceRepository("gate")
        pipeline = pipeline_factory(repository).create(
            "gate",
            motion_observation_stage_configs(
                mog2_enabled=True,
                sample_fps=5.0,
                mog2_history_seconds=30.0,
            ),
        )

        result = pipeline.process(
            MotionContext(
                camera_id="gate",
                captured_at=12.0,
                original_frame=None,
                configuration={
                    "observation_kind": "motion_event",
                    "event_source": "onvif",
                    "event_topic": "RuleEngine/CellMotionDetector/Motion",
                    "event_message": "State=true",
                    "event_at": 11.75,
                },
                runtime=pipeline.runtime,
            )
        )

        evidence = repository.last("onvif")
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.values["score"], 0.55)
        self.assertFalse(evidence.values["priority"])
        self.assertEqual(evidence.values["event_at"], 11.75)
        self.assertEqual(evidence.values["received_at"], 12.0)
        self.assertEqual(result.source_evidence["onvif"]["event_source"], "onvif")
        self.assertNotIn("mog2_source", pipeline.runtime.stage_state)

    def test_onvif_source_scores_semantic_topic_as_priority(self) -> None:
        repository = MotionEvidenceRepository("gate")
        pipeline = pipeline_factory(repository).create(
            "gate",
            motion_observation_stage_configs(
                mog2_enabled=True,
                sample_fps=5.0,
                mog2_history_seconds=30.0,
            ),
        )

        pipeline.process(
            MotionContext(
                camera_id="gate",
                captured_at=12.0,
                original_frame=None,
                configuration={
                    "observation_kind": "motion_event",
                    "event_topic": "RuleEngine/PeopleDetector/Person",
                    "event_message": "State=true",
                },
                runtime=pipeline.runtime,
            )
        )

        evidence = repository.last("onvif")
        self.assertTrue(evidence.values["priority"])
        self.assertEqual(evidence.values["score"], 0.95)

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

    def test_any_policy_can_rescue_primary_rejection(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append("mog2", 11.0, {"warmed": 1.0, "score": 0.8})

        result = apply_fusion_policy(
            repository,
            policy="any",
            primary_accepted=False,
            primary_score=0.2,
        )

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.score, 0.8)
        self.assertEqual(result.scoring.reason, "fusion_any_accepted")
        self.assertEqual(result.scoring.features["fusion_source_votes"], {"mog2": True})

    def test_all_policy_can_reject_primary_without_source_consensus(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append("mog2", 11.0, {"warmed": 1.0, "score": 0.2})

        result = apply_fusion_policy(
            repository,
            policy="all",
            primary_accepted=True,
            primary_score=0.8,
        )

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.score, 0.2)
        self.assertEqual(result.scoring.reason, "fusion_all_rejected")

    def test_weighted_policy_uses_configured_weights_and_threshold(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append("mog2", 11.0, {"warmed": 1.0, "score": 0.8})

        result = apply_fusion_policy(
            repository,
            policy="weighted",
            primary_accepted=False,
            primary_score=0.2,
            options={
                "source_weights": {"primary": 1.0, "mog2": 3.0},
                "weighted_threshold": 0.6,
            },
        )

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.score, 0.65)
        self.assertEqual(result.scoring.reason, "fusion_weighted_accepted")

    def test_policy_fails_open_to_primary_when_source_is_not_warmed(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append("mog2", 11.0, {"warmed": 0.0, "score": 0.9})

        result = apply_fusion_policy(
            repository,
            policy="all",
            primary_accepted=True,
            primary_score=0.7,
        )

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.score, 0.7)
        self.assertFalse(result.scoring.features["fusion_applied"])
        self.assertEqual(result.scoring.features["fusion_reason"], "insufficient_sources")

    def test_generic_registered_source_can_participate_without_fusion_changes(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append(
            "onvif",
            11.0,
            {"warmed": 1.0, "score": 1.0, "topic": "RuleEngine/CellMotionDetector"},
        )

        result = apply_fusion_policy(
            repository,
            policy="any",
            primary_accepted=False,
            primary_score=0.2,
            options={"sources": ["onvif"]},
        )

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.features["onvif_score"], 1.0)
        self.assertEqual(
            result.scoring.features["onvif_topic"],
            "RuleEngine/CellMotionDetector",
        )


if __name__ == "__main__":
    unittest.main()

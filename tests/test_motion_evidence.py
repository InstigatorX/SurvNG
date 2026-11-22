from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

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
    require_primary_trigger: bool = False,
) -> MotionContext:
    pipeline = pipeline_factory(repository).create(
        "gate",
        [
            MotionStageConfig(
                "fusion",
                "buffered_evidence_fusion",
                {"policy": policy, **(options or {})},
            )
        ],
        initial_artifacts={"scoring"},
    )
    return pipeline.process(
        MotionContext(
            camera_id="gate",
            captured_at=12.0,
            original_frame=None,
            configuration={
                "evidence_started_at": 9.0,
                "evidence_ended_at": 12.0,
                "require_primary_trigger": require_primary_trigger,
            },
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
        repository.append("aux", 1.0, {"score": 0.1})
        repository.append("aux", 2.0, {"score": 0.2})
        repository.append("aux", 3.0, {"score": 0.3})

        samples = repository.window("aux", 1.5, 2.5)

        self.assertEqual([sample.captured_at for sample in samples], [2.0])
        self.assertEqual(repository.last("aux").values["score"], 0.3)
        self.assertEqual(repository.status()["aux"]["sample_count"], 2)

    def test_repository_accepts_parallel_source_writers(self) -> None:
        repository = MotionEvidenceRepository("gate", max_samples_per_source=200)

        def append_sample(index: int) -> None:
            source = "aux" if index % 2 else "onvif"
            repository.append(source, float(index), {"score": index / 100.0})

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(append_sample, range(100)))

        status = repository.status()
        self.assertEqual(status["aux"]["sample_count"], 50)
        self.assertEqual(status["onvif"]["sample_count"], 50)

    def test_onvif_source_normalizes_motion_event(self) -> None:
        repository = MotionEvidenceRepository("gate")
        pipeline = pipeline_factory(repository).create(
            "gate",
            motion_observation_stage_configs(),
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

    def test_onvif_source_scores_semantic_topic_as_priority(self) -> None:
        repository = MotionEvidenceRepository("gate")
        pipeline = pipeline_factory(repository).create(
            "gate",
            motion_observation_stage_configs(),
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

    def test_audit_fusion_preserves_primary_decision(self) -> None:
        for accepted in (True, False):
            result = apply_fusion_policy(
                MotionEvidenceRepository("gate"),
                policy="audit",
                primary_accepted=accepted,
                primary_score=0.42,
            )
            with self.subTest(accepted=accepted):
                self.assertEqual(result.scoring.accepted, accepted)
                self.assertEqual(result.scoring.score, 0.42)
                self.assertEqual(result.scoring.reason, "qualified" if accepted else "edge_motion")

    def test_bypass_policy_accepts_without_visual_validators(self) -> None:
        result = apply_fusion_policy(
            MotionEvidenceRepository("gate"),
            policy="bypass",
            primary_accepted=False,
            primary_score=0.1,
        )

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "validation_disabled")

    def test_required_primary_cannot_be_bypassed(self) -> None:
        result = apply_fusion_policy(
            MotionEvidenceRepository("gate"),
            policy="bypass",
            primary_accepted=False,
            primary_score=0.1,
            require_primary_trigger=True,
        )

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "primary_trigger_rejected")

    def test_fusion_requires_explicit_scoring_input(self) -> None:
        factory = pipeline_factory(MotionEvidenceRepository("gate"))

        with self.assertRaisesRegex(ValueError, "scoring"):
            factory.create("gate", motion_fusion_stage_configs())

    def test_fusion_stage_rejects_sources_and_unsupported_policies(self) -> None:
        factory = pipeline_factory(MotionEvidenceRepository("gate"))
        for options in ({"sources": ["aux"]}, {"policy": "any"}, {"policy": "weighted"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                factory.create(
                    "gate",
                    [MotionStageConfig("fusion", "buffered_evidence_fusion", options)],
                    initial_artifacts={"scoring"},
                )

if __name__ == "__main__":
    unittest.main()

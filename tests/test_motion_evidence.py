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
                {"sources": ["aux"], "policy": policy, **(options or {})},
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

    def test_fusion_preserves_primary_decision_and_adds_windowed_evidence(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append("aux", 10.0, {"warmed": 0.0, "score": 0.1})
        repository.append("aux", 11.0, {"warmed": 1.0, "score": 0.8})
        repository.append("aux", 20.0, {"warmed": 1.0, "score": 0.2})
        factory = pipeline_factory(repository)
        pipeline = factory.create(
            "gate",
            [
                MotionStageConfig(
                    "evidence_fusion",
                    "buffered_evidence_fusion",
                    {"sources": ["aux"], "policy": "audit"},
                ),
            ],
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
        self.assertEqual(result.scoring.features["aux_score"], 0.8)
        self.assertEqual(result.scoring.features["aux_sample_count"], 2)
        self.assertEqual(result.source_evidence["aux"]["aux_score"], 0.8)

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
        repository.append("aux", 11.0, {"warmed": 1.0, "score": 0.8})

        result = apply_fusion_policy(
            repository,
            policy="any",
            primary_accepted=False,
            primary_score=0.2,
        )

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.score, 0.8)
        self.assertEqual(result.scoring.reason, "fusion_any_accepted")
        self.assertEqual(result.scoring.features["fusion_source_votes"], {"aux": True})

    def test_all_policy_can_reject_primary_without_source_consensus(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append("aux", 11.0, {"warmed": 1.0, "score": 0.2})

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
        repository.append("aux", 11.0, {"warmed": 1.0, "score": 0.8})

        result = apply_fusion_policy(
            repository,
            policy="weighted",
            primary_accepted=False,
            primary_score=0.2,
            options={
                "source_weights": {"primary": 1.0, "aux": 3.0},
                "weighted_threshold": 0.6,
            },
        )

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.score, 0.65)
        self.assertEqual(result.scoring.reason, "fusion_weighted_accepted")

    def test_policy_fails_open_to_primary_when_source_is_not_warmed(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append("aux", 11.0, {"warmed": 0.0, "score": 0.9})

        result = apply_fusion_policy(
            repository,
            policy="all",
            primary_accepted=True,
            primary_score=0.7,
        )

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.score, 1.0)
        self.assertEqual(result.scoring.reason, "validation_unavailable_fail_open")
        self.assertFalse(result.scoring.features["fusion_applied"])
        self.assertEqual(result.scoring.features["fusion_reason"], "insufficient_sources")

    def test_policy_fails_open_even_when_primary_rejected(self) -> None:
        repository = MotionEvidenceRepository("gate")

        result = apply_fusion_policy(
            repository,
            policy="all",
            primary_accepted=False,
            primary_score=0.1,
        )

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "validation_unavailable_fail_open")

    def test_policy_fails_closed_when_required_source_is_unavailable(self) -> None:
        result = apply_fusion_policy(
            MotionEvidenceRepository("gate"),
            policy="all",
            primary_accepted=True,
            primary_score=0.8,
            options={"fail_open": False},
        )

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.score, 0.8)
        self.assertEqual(result.scoring.reason, "validation_unavailable_fail_closed")
        self.assertFalse(result.scoring.features["fusion_applied"])
        self.assertEqual(result.scoring.features["fusion_reason"], "insufficient_sources")

    def test_bypass_policy_accepts_without_visual_validators(self) -> None:
        result = apply_fusion_policy(
            MotionEvidenceRepository("gate"),
            policy="bypass",
            primary_accepted=False,
            primary_score=0.1,
        )

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "validation_disabled")

    def test_supporting_source_only_policy_does_not_include_adaptive_score(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append("aux", 11.0, {"warmed": 1.0, "score": 0.8})

        result = apply_fusion_policy(
            repository,
            policy="all",
            primary_accepted=False,
            primary_score=0.1,
            options={"include_primary": False},
        )

        self.assertTrue(result.scoring.accepted)
        self.assertEqual(result.scoring.score, 0.8)
        self.assertFalse(result.scoring.features["fusion_primary_included"])

    def test_required_primary_cannot_be_rescued_by_supporting_source(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append("aux", 11.0, {"warmed": 1.0, "score": 0.9})

        result = apply_fusion_policy(
            repository,
            policy="any",
            primary_accepted=False,
            primary_score=0.1,
            require_primary_trigger=True,
        )

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.reason, "primary_trigger_rejected")

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

    def test_depth_object_evidence_stage_exposes_windowed_samples(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append(
            "depth_object",
            10.5,
            {
                "score": 0.82,
                "warmed": 1.0,
                "nearest_m": 5.4,
                "median_m": 6.1,
                "object_count": 2,
            },
        )
        pipeline = pipeline_factory(repository).create(
            "gate",
            [
                MotionStageConfig(
                    "depth_evidence",
                    "depth_object_evidence",
                    {"enabled": True},
                )
            ],
            initial_artifacts={"scoring"},
        )
        result = pipeline.process(
            MotionContext(
                camera_id="gate",
                captured_at=12.0,
                original_frame=None,
                configuration={"evidence_started_at": 9.0, "evidence_ended_at": 12.0},
                runtime=pipeline.runtime,
                scoring=MotionScoring(
                    accepted=True,
                    score=0.7,
                    threshold=0.48,
                    reason="qualified",
                    frame_count=9,
                ),
            )
        )

        self.assertEqual(result.source_evidence["depth_object"]["depth_object_score"], 0.82)
        self.assertEqual(result.scoring.features["depth_object_nearest_m"], 5.4)
        self.assertEqual(result.scoring.features["depth_object_sample_count"], 1)

    def test_depth_object_fusion_audit_preserves_primary_decision(self) -> None:
        repository = MotionEvidenceRepository("gate")
        repository.append(
            "depth_object",
            11.0,
            {
                "foreground_score": 0.75,
                "nearest_m": 7.5,
                "median_m": 8.0,
                "object_count": 1,
            },
        )
        result = apply_fusion_policy(
            repository,
            policy="audit",
            primary_accepted=False,
            primary_score=0.42,
            options={"sources": ["depth_object"]},
        )

        self.assertFalse(result.scoring.accepted)
        self.assertEqual(result.scoring.score, 0.42)
        self.assertEqual(result.scoring.features["depth_object_score"], 0.75)
        self.assertEqual(result.scoring.features["depth_object_nearest_m"], 7.5)


if __name__ == "__main__":
    unittest.main()

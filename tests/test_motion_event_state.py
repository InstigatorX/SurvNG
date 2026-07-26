from __future__ import annotations

import unittest

from survng.app.motion_pipeline import (
    MotionContext,
    MotionEventPhase,
    MotionPipelineFactory,
    MotionScoring,
    MotionStageConfig,
    build_builtin_motion_registry,
)


def state_pipeline(
    camera_id: str,
    *,
    activation_frames: int = 1,
    release_frames: int = 1,
    cooldown_seconds: float = 0.0,
    state_timeout_seconds: float = 10.0,
):
    return MotionPipelineFactory(build_builtin_motion_registry()).create(
        camera_id,
        [
            MotionStageConfig(
                "event_state",
                "score_event_state",
                {
                    "activation_frames": activation_frames,
                    "release_frames": release_frames,
                    "cooldown_seconds": cooldown_seconds,
                    "state_timeout_seconds": state_timeout_seconds,
                },
            ),
            MotionStageConfig("trigger", "score_trigger"),
        ],
        initial_artifacts={"scoring"},
    )


def process_score(pipeline, captured_at: float, accepted: bool) -> MotionContext:
    return pipeline.process(
        MotionContext(
            camera_id=pipeline.camera_id,
            captured_at=captured_at,
            original_frame=None,
            configuration={},
            runtime=pipeline.runtime,
            scoring=MotionScoring(
                accepted=accepted,
                score=0.8 if accepted else 0.2,
                threshold=0.48,
                reason="qualified" if accepted else "edge_motion",
                frame_count=9,
            ),
        )
    )


class MotionEventStateTest(unittest.TestCase):
    def test_default_state_machine_preserves_immediate_decisions(self) -> None:
        pipeline = state_pipeline("gate")

        accepted = process_score(pipeline, 10.0, True)
        rejected = process_score(pipeline, 11.0, False)

        self.assertEqual(accepted.event_state.phase, MotionEventPhase.ACTIVE)
        self.assertTrue(accepted.decision.run_object_detection)
        self.assertEqual(accepted.decision.reason, "qualified")
        self.assertEqual(rejected.event_state.phase, MotionEventPhase.REJECTED)
        self.assertFalse(rejected.decision.run_object_detection)
        self.assertEqual(rejected.decision.reason, "edge_motion")

    def test_activation_hysteresis_requires_consecutive_accepts(self) -> None:
        pipeline = state_pipeline("gate", activation_frames=2)

        candidate = process_score(pipeline, 10.0, True)
        active = process_score(pipeline, 11.0, True)

        self.assertEqual(candidate.event_state.phase, MotionEventPhase.CANDIDATE)
        self.assertEqual(candidate.event_state.consecutive_accepts, 1)
        self.assertFalse(candidate.decision.run_object_detection)
        self.assertEqual(candidate.decision.reason, "event_state_candidate")
        self.assertEqual(active.event_state.phase, MotionEventPhase.ACTIVE)
        self.assertEqual(active.event_state.consecutive_accepts, 2)
        self.assertTrue(active.decision.run_object_detection)
        self.assertTrue(active.event_state.event_key.startswith("gate:"))

    def test_release_hysteresis_holds_active_state_until_threshold(self) -> None:
        pipeline = state_pipeline("gate", release_frames=2)
        process_score(pipeline, 10.0, True)

        held = process_score(pipeline, 11.0, False)
        released = process_score(pipeline, 12.0, False)

        self.assertEqual(held.event_state.phase, MotionEventPhase.ACTIVE)
        self.assertEqual(held.event_state.transition_reason, "release_pending")
        self.assertFalse(held.decision.run_object_detection)
        self.assertEqual(held.decision.reason, "event_state_active")
        self.assertEqual(released.event_state.phase, MotionEventPhase.REJECTED)
        self.assertFalse(released.decision.run_object_detection)

    def test_cooldown_blocks_reactivation_until_deadline(self) -> None:
        pipeline = state_pipeline(
            "gate",
            cooldown_seconds=5.0,
            state_timeout_seconds=1.0,
        )
        process_score(pipeline, 10.0, True)
        cooling = process_score(pipeline, 11.0, False)
        blocked = process_score(pipeline, 13.0, True)
        reactivated = process_score(pipeline, 16.0, True)

        self.assertEqual(cooling.event_state.phase, MotionEventPhase.COOLDOWN)
        self.assertEqual(cooling.event_state.cooldown_until, 16.0)
        self.assertFalse(blocked.decision.run_object_detection)
        self.assertEqual(blocked.decision.reason, "event_state_cooldown")
        self.assertEqual(reactivated.event_state.phase, MotionEventPhase.ACTIVE)
        self.assertTrue(reactivated.decision.run_object_detection)

    def test_state_timeout_prevents_stale_active_hold(self) -> None:
        pipeline = state_pipeline(
            "gate",
            release_frames=3,
            state_timeout_seconds=5.0,
        )
        process_score(pipeline, 10.0, True)

        result = process_score(pipeline, 20.0, False)

        self.assertEqual(result.event_state.phase, MotionEventPhase.REJECTED)
        self.assertFalse(result.decision.run_object_detection)

    def test_runtime_state_is_isolated_per_camera(self) -> None:
        gate = state_pipeline("gate", activation_frames=2)
        foyer = state_pipeline("foyer", activation_frames=2)

        process_score(gate, 10.0, True)
        foyer_candidate = process_score(foyer, 10.0, True)
        gate_active = process_score(gate, 11.0, True)

        self.assertEqual(foyer_candidate.event_state.phase, MotionEventPhase.CANDIDATE)
        self.assertEqual(gate_active.event_state.phase, MotionEventPhase.ACTIVE)
        self.assertIsNot(gate.runtime, foyer.runtime)

    def test_continuous_activity_triggers_object_detection_only_once(self) -> None:
        pipeline = state_pipeline("gate")

        first = process_score(pipeline, 10.0, True)
        continuous = process_score(pipeline, 11.0, True)

        self.assertTrue(first.decision.run_object_detection)
        self.assertFalse(continuous.decision.run_object_detection)
        self.assertEqual(continuous.event_state.phase, MotionEventPhase.ACTIVE)
        self.assertEqual(continuous.decision.reason, "event_state_active")


if __name__ == "__main__":
    unittest.main()

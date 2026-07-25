from __future__ import annotations

from typing import Any, Mapping

from ..motion import qualify_motion
from .context import MotionContext, MotionEventPhase, MotionScoring, TriggerDecision
from .registry import MotionStageDependencies, MotionStageRegistration, MotionStageRegistry


class LegacyQualificationStage:
    def __init__(self, stage_id: str = "legacy_qualifier") -> None:
        self._stage_id = stage_id

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        sensitivity = str(context.configuration.get("sensitivity") or "balanced")
        result = qualify_motion(list(context.frame_history), sensitivity)
        context.scoring = MotionScoring(
            accepted=result.accepted,
            score=result.score,
            threshold=result.threshold,
            reason=result.reason,
            frame_count=result.frame_count,
            features=dict(result.features),
        )
        context.event_state.phase = (
            MotionEventPhase.ACTIVE if result.accepted else MotionEventPhase.REJECTED
        )
        context.event_state.updated_at = context.captured_at
        context.decision = TriggerDecision(
            run_object_detection=result.accepted,
            reason=result.reason,
            score=result.score,
            evidence_sources=("legacy_frame_difference",),
        )
        return context


def _build_legacy_stage(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> LegacyQualificationStage:
    del options, dependencies
    return LegacyQualificationStage(stage_id)


def register_legacy_motion_stage(registry: MotionStageRegistry) -> None:
    registry.register(
        MotionStageRegistration(
            implementation="legacy_qualifier",
            builder=_build_legacy_stage,
            requires=frozenset({"frame_history", "configuration"}),
            provides=frozenset({"scoring", "event_state", "decision"}),
            graph="qualification",
            category="compatibility",
            display_name="Classic motion analysis",
            description="Runs the original all-in-one SurvNG motion algorithm.",
        )
    )

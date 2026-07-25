from __future__ import annotations

from .context import (
    MotionBlob,
    MotionContext,
    MotionDebugData,
    MotionEventPhase,
    MotionEventState,
    MotionScoring,
    MotionTrack,
    StageTiming,
    TriggerDecision,
)
from .contracts import MotionPipelineObserver, MotionStage
from .factory import MotionPipelineFactory, MotionStageConfig
from .legacy import LegacyQualificationStage, register_legacy_motion_stage
from .pipeline import LoggingMotionPipelineObserver, MotionPipeline
from .registry import MotionStageDependencies, MotionStageRegistration, MotionStageRegistry
from .runtime import MotionRuntimeState


def build_builtin_motion_registry() -> MotionStageRegistry:
    registry = MotionStageRegistry()
    register_legacy_motion_stage(registry)
    return registry


def build_legacy_motion_pipeline(camera_id: str) -> MotionPipeline:
    factory = MotionPipelineFactory(
        build_builtin_motion_registry(),
        observer=LoggingMotionPipelineObserver(),
    )
    return factory.create(
        camera_id,
        [MotionStageConfig(stage_id="qualification", implementation="legacy_qualifier")],
    )


__all__ = [
    "LegacyQualificationStage",
    "LoggingMotionPipelineObserver",
    "MotionBlob",
    "MotionContext",
    "MotionDebugData",
    "MotionEventPhase",
    "MotionEventState",
    "MotionPipeline",
    "MotionPipelineFactory",
    "MotionPipelineObserver",
    "MotionRuntimeState",
    "MotionScoring",
    "MotionStage",
    "MotionStageConfig",
    "MotionStageDependencies",
    "MotionStageRegistration",
    "MotionStageRegistry",
    "MotionTrack",
    "StageTiming",
    "TriggerDecision",
    "build_builtin_motion_registry",
    "build_legacy_motion_pipeline",
    "register_legacy_motion_stage",
]

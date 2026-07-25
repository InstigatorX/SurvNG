from __future__ import annotations

from .context import (
    MotionBlob,
    MotionContext,
    MotionDebugData,
    MotionEventPhase,
    MotionEventState,
    MotionFrameBlobs,
    MotionScoring,
    MotionTrack,
    StageTiming,
    TriggerDecision,
)
from .contracts import MotionPipelineObserver, MotionStage
from .configuration import ResolvedMotionPipelineGraphs, resolve_motion_pipeline_graphs
from .catalog import (
    MotionPipelinePreset,
    builtin_motion_pipeline_presets,
    motion_pipeline_catalog,
)
from .defaults import (
    default_motion_fusion_stage_configs as motion_fusion_stage_configs,
    default_motion_observation_stage_configs as motion_observation_stage_configs,
    default_motion_stage_configs,
)
from .decision_handler import (
    MotionDecisionHandler,
    MotionDecisionHandlerFactory,
    MotionDecisionOutcome,
)
from .decision_stages import (
    MotionEventStateStage,
    ObjectDetectionTriggerStage,
    register_decision_stages,
)
from .evidence import MotionEvidenceRepository, MotionEvidenceSample
from .evidence_stages import (
    EVIDENCE_REPOSITORY_SERVICE,
    BufferedMotionFusionStage,
    Mog2EvidenceSourceStage,
    OnvifEventEvidenceStage,
    register_evidence_stages,
)
from .factory import MotionPipelineFactory, MotionStageConfig
from .image_stages import (
    BlobExtractionStage,
    BlobFilteringStage,
    DominantCentroidTrackingStage,
    FixedThresholdStage,
    FrameDifferenceStage,
    LegacyMotionScoringStage,
    MotionFramePreprocessorStage,
    MotionScoringStage,
    OpenCloseMorphologyStage,
    register_image_motion_stages,
)
from .legacy import LegacyQualificationStage, register_legacy_motion_stage
from .pipeline import LoggingMotionPipelineObserver, MotionPipeline
from .object_detection import RecordedMotionObjectDetector, RecordedMotionObjectDetectorFactory
from .registry import (
    MotionStageDependencies,
    MotionStageOption,
    MotionStageRegistration,
    MotionStageRegistry,
)
from .runtime import MotionRuntimeState


def build_builtin_motion_registry() -> MotionStageRegistry:
    registry = MotionStageRegistry()
    register_legacy_motion_stage(registry)
    register_image_motion_stages(registry)
    register_evidence_stages(registry)
    register_decision_stages(registry)
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


def build_default_motion_pipeline(camera_id: str) -> MotionPipeline:
    factory = MotionPipelineFactory(
        build_builtin_motion_registry(),
        observer=LoggingMotionPipelineObserver(),
    )
    return factory.create(camera_id, default_motion_stage_configs())


__all__ = [
    "LegacyQualificationStage",
    "LegacyMotionScoringStage",
    "BlobExtractionStage",
    "BlobFilteringStage",
    "BufferedMotionFusionStage",
    "DominantCentroidTrackingStage",
    "LoggingMotionPipelineObserver",
    "MotionBlob",
    "MotionContext",
    "MotionDebugData",
    "MotionDecisionHandler",
    "MotionDecisionHandlerFactory",
    "MotionDecisionOutcome",
    "MotionEventPhase",
    "MotionEventState",
    "MotionEvidenceRepository",
    "MotionEvidenceSample",
    "MotionFramePreprocessorStage",
    "MotionFrameBlobs",
    "MotionEventStateStage",
    "MotionPipeline",
    "MotionPipelinePreset",
    "MotionPipelineFactory",
    "MotionPipelineObserver",
    "MotionRuntimeState",
    "MotionScoring",
    "MotionScoringStage",
    "MotionStage",
    "MotionStageOption",
    "MotionStageConfig",
    "MotionStageDependencies",
    "MotionStageRegistration",
    "MotionStageRegistry",
    "MotionTrack",
    "Mog2EvidenceSourceStage",
    "OnvifEventEvidenceStage",
    "OpenCloseMorphologyStage",
    "ObjectDetectionTriggerStage",
    "RecordedMotionObjectDetector",
    "RecordedMotionObjectDetectorFactory",
    "ResolvedMotionPipelineGraphs",
    "FixedThresholdStage",
    "FrameDifferenceStage",
    "StageTiming",
    "TriggerDecision",
    "EVIDENCE_REPOSITORY_SERVICE",
    "build_builtin_motion_registry",
    "builtin_motion_pipeline_presets",
    "build_default_motion_pipeline",
    "build_legacy_motion_pipeline",
    "default_motion_stage_configs",
    "motion_fusion_stage_configs",
    "motion_pipeline_catalog",
    "motion_observation_stage_configs",
    "register_legacy_motion_stage",
    "register_image_motion_stages",
    "register_evidence_stages",
    "register_decision_stages",
    "resolve_motion_pipeline_graphs",
]

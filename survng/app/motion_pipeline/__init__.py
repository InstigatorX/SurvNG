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
from .evidence import MotionEvidenceRepository, MotionEvidenceSample
from .evidence_stages import (
    EVIDENCE_REPOSITORY_SERVICE,
    BufferedMotionFusionStage,
    Mog2EvidenceSourceStage,
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
    MotionEventStateStage,
    MotionFramePreprocessorStage,
    MotionScoringStage,
    ObjectDetectionTriggerStage,
    OpenCloseMorphologyStage,
    register_image_motion_stages,
)
from .legacy import LegacyQualificationStage, register_legacy_motion_stage
from .pipeline import LoggingMotionPipelineObserver, MotionPipeline
from .object_detection import RecordedMotionObjectDetector, RecordedMotionObjectDetectorFactory
from .registry import MotionStageDependencies, MotionStageRegistration, MotionStageRegistry
from .runtime import MotionRuntimeState


def build_builtin_motion_registry() -> MotionStageRegistry:
    registry = MotionStageRegistry()
    register_legacy_motion_stage(registry)
    register_image_motion_stages(registry)
    register_evidence_stages(registry)
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
    "MotionPipelineFactory",
    "MotionPipelineObserver",
    "MotionRuntimeState",
    "MotionScoring",
    "MotionScoringStage",
    "MotionStage",
    "MotionStageConfig",
    "MotionStageDependencies",
    "MotionStageRegistration",
    "MotionStageRegistry",
    "MotionTrack",
    "Mog2EvidenceSourceStage",
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
    "build_default_motion_pipeline",
    "build_legacy_motion_pipeline",
    "default_motion_stage_configs",
    "motion_fusion_stage_configs",
    "motion_observation_stage_configs",
    "register_legacy_motion_stage",
    "register_image_motion_stages",
    "register_evidence_stages",
    "resolve_motion_pipeline_graphs",
]

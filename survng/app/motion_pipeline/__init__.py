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
from .adaptive_stages import (
    AdaptiveBlobFilterStage,
    AdaptiveEmaBackgroundStage,
    AdaptiveMotionScoringStage,
    AdaptiveStatisticalThresholdStage,
    ConnectedComponentBlobStage,
    EmaZoneExclusionStage,
    IlluminationChangeFilterStage,
    PersistentCentroidTrackerStage,
    register_adaptive_motion_stages,
)
from .contracts import (
    MotionPipelineObserver,
    MotionRuntimeSnapshot,
    MotionStage,
    MotionStageLifecycle,
)
from .configuration import (
    ResolvedMotionPipelineGraphs,
    resolve_motion_pipeline_graphs,
    resolved_trigger_mode,
)
from .catalog import (
    MotionPipelinePreset,
    builtin_motion_pipeline_presets,
    motion_pipeline_catalog,
)
from .defaults import (
    adaptive_motion_stage_configs,
    default_motion_fusion_stage_configs as motion_fusion_stage_configs,
    default_motion_observation_stage_configs as motion_observation_stage_configs,
    default_motion_stage_configs,
)
from .decision_handler import (
    MotionDecisionHandler,
    MotionDecisionHandlerFactory,
    MotionDecisionOutcome,
)
from .debug import MotionDebugSnapshot, MotionDebugSnapshotStore
from .evidence import MotionEvidenceRepository, MotionEvidenceSample
from .evidence_stages import (
    EVIDENCE_REPOSITORY_SERVICE,
    BufferedMotionFusionStage,
    DepthObjectEvidenceStage,
    OnvifEventEvidenceStage,
    register_evidence_stages,
)
from .factory import MotionPipelineFactory, MotionStageConfig
from .guided import (
    analysis_preset_selections,
    guided_fusion_settings,
    identify_analysis_preset,
    update_guided_fusion,
)
from .image_stages import (
    BlobExtractionStage,
    BlobFilteringStage,
    DominantCentroidTrackingStage,
    FixedThresholdStage,
    FrameDifferenceStage,
    MotionFramePreprocessorStage,
    MotionScoringStage,
    OpenCloseMorphologyStage,
    register_image_motion_stages,
)
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
    register_image_motion_stages(registry)
    register_adaptive_motion_stages(registry)
    register_evidence_stages(registry)
    return registry


def build_default_motion_pipeline(camera_id: str) -> MotionPipeline:
    factory = MotionPipelineFactory(
        build_builtin_motion_registry(),
        observer=LoggingMotionPipelineObserver(),
    )
    return factory.create(camera_id, default_motion_stage_configs())


__all__ = [
    "AdaptiveBlobFilterStage",
    "AdaptiveEmaBackgroundStage",
    "AdaptiveMotionScoringStage",
    "AdaptiveStatisticalThresholdStage",
    "BlobExtractionStage",
    "BlobFilteringStage",
    "BufferedMotionFusionStage",
    "ConnectedComponentBlobStage",
    "EmaZoneExclusionStage",
    "IlluminationChangeFilterStage",
    "DominantCentroidTrackingStage",
    "LoggingMotionPipelineObserver",
    "MotionBlob",
    "MotionContext",
    "MotionDebugData",
    "MotionDebugSnapshot",
    "MotionDebugSnapshotStore",
    "MotionDecisionHandler",
    "MotionDecisionHandlerFactory",
    "MotionDecisionOutcome",
    "MotionEventPhase",
    "MotionEventState",
    "MotionEvidenceRepository",
    "MotionEvidenceSample",
    "MotionFramePreprocessorStage",
    "MotionFrameBlobs",
    "MotionPipeline",
    "MotionPipelinePreset",
    "MotionPipelineFactory",
    "MotionPipelineObserver",
    "MotionRuntimeState",
    "MotionRuntimeSnapshot",
    "MotionScoring",
    "MotionScoringStage",
    "MotionStage",
    "MotionStageLifecycle",
    "MotionStageOption",
    "MotionStageConfig",
    "analysis_preset_selections",
    "MotionStageDependencies",
    "MotionStageRegistration",
    "MotionStageRegistry",
    "MotionTrack",
    "DepthObjectEvidenceStage",
    "OnvifEventEvidenceStage",
    "OpenCloseMorphologyStage",
    "PersistentCentroidTrackerStage",
    "RecordedMotionObjectDetector",
    "RecordedMotionObjectDetectorFactory",
    "ResolvedMotionPipelineGraphs",
    "FixedThresholdStage",
    "FrameDifferenceStage",
    "StageTiming",
    "TriggerDecision",
    "EVIDENCE_REPOSITORY_SERVICE",
    "adaptive_motion_stage_configs",
    "build_builtin_motion_registry",
    "builtin_motion_pipeline_presets",
    "build_default_motion_pipeline",
    "guided_fusion_settings",
    "identify_analysis_preset",
    "default_motion_stage_configs",
    "motion_fusion_stage_configs",
    "motion_pipeline_catalog",
    "motion_observation_stage_configs",
    "register_image_motion_stages",
    "register_adaptive_motion_stages",
    "register_evidence_stages",
    "resolve_motion_pipeline_graphs",
    "resolved_trigger_mode",
    "update_guided_fusion",
]

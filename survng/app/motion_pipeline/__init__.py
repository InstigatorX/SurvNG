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
from .decision_handler import (
    MotionDecisionHandler,
    MotionDecisionHandlerFactory,
    MotionDecisionOutcome,
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


def default_motion_stage_configs() -> list[MotionStageConfig]:
    return [
        MotionStageConfig(stage_id="preprocess", implementation="gray_blur"),
        MotionStageConfig(stage_id="difference", implementation="frame_difference"),
        MotionStageConfig(
            stage_id="threshold",
            implementation="fixed_threshold",
            options={"value": 18},
        ),
        MotionStageConfig(
            stage_id="morphology",
            implementation="open_close",
            options={"kernel_size": 3, "close_iterations": 2},
        ),
        MotionStageConfig(stage_id="blob_extract", implementation="contour_blobs"),
        MotionStageConfig(
            stage_id="blob_filter",
            implementation="minimum_area",
            options={"minimum_area_ratio": 0.0003},
        ),
        MotionStageConfig(
            stage_id="tracking",
            implementation="dominant_centroid",
            options={
                "minimum_active_area_ratio": 0.0008,
                "minimum_changed_ratio": 0.003,
            },
        ),
        MotionStageConfig(stage_id="scoring", implementation="default_motion_score"),
        MotionStageConfig(stage_id="event_state", implementation="score_event_state"),
        MotionStageConfig(stage_id="trigger", implementation="score_trigger"),
    ]


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
    "OpenCloseMorphologyStage",
    "ObjectDetectionTriggerStage",
    "RecordedMotionObjectDetector",
    "RecordedMotionObjectDetectorFactory",
    "FixedThresholdStage",
    "FrameDifferenceStage",
    "StageTiming",
    "TriggerDecision",
    "build_builtin_motion_registry",
    "build_default_motion_pipeline",
    "build_legacy_motion_pipeline",
    "default_motion_stage_configs",
    "register_legacy_motion_stage",
    "register_image_motion_stages",
]

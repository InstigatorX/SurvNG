from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..motion import (
    difference_motion_frames,
    extract_motion_blobs,
    filter_motion_blobs,
    morphology_motion_masks,
    preprocess_motion_frames,
    score_motion_masks,
    score_motion_track,
    threshold_motion_differences,
    track_dominant_motion,
)
from .context import MotionContext, MotionEventPhase, MotionScoring, TriggerDecision
from .registry import (
    MotionStageDependencies,
    MotionStageOption,
    MotionStageRegistration,
    MotionStageRegistry,
)


class MotionFramePreprocessorStage:
    def __init__(self, stage_id: str) -> None:
        self._stage_id = stage_id

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        if context.processed_frame_history:
            context.processed_frame = context.processed_frame_history[-1]
            return context
        prepared = preprocess_motion_frames(list(context.frame_history))
        context.processed_frame_history = tuple(prepared)
        context.processed_frame = prepared[-1] if prepared else None
        return context


class FrameDifferenceStage:
    def __init__(self, stage_id: str) -> None:
        self._stage_id = stage_id

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        differences = difference_motion_frames(list(context.processed_frame_history))
        context.difference_history = tuple(differences)
        context.difference_image = differences[-1] if differences else None
        return context


class FixedThresholdStage:
    def __init__(self, stage_id: str, threshold_value: int = 18) -> None:
        self._stage_id = stage_id
        self.threshold_value = max(0, min(255, int(threshold_value)))

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        masks = threshold_motion_differences(
            list(context.difference_history),
            self.threshold_value,
        )
        context.threshold_mask_history = tuple(masks)
        context.binary_motion_mask = masks[-1] if masks else None
        return context


class OpenCloseMorphologyStage:
    def __init__(
        self,
        stage_id: str,
        kernel_size: int = 3,
        close_iterations: int = 2,
    ) -> None:
        self._stage_id = stage_id
        normalized_kernel = max(1, int(kernel_size))
        self.kernel_size = normalized_kernel if normalized_kernel % 2 else normalized_kernel + 1
        self.close_iterations = max(0, int(close_iterations))

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        masks = morphology_motion_masks(
            list(context.threshold_mask_history),
            self.kernel_size,
            self.close_iterations,
        )
        context.motion_mask_history = tuple(masks)
        context.binary_motion_mask = masks[-1] if masks else None
        return context


class BlobExtractionStage:
    def __init__(self, stage_id: str) -> None:
        self._stage_id = stage_id

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        history = extract_motion_blobs(
            list(context.processed_frame_history),
            list(context.motion_mask_history),
        )
        context.raw_blob_history = tuple(history)
        return context


class BlobFilteringStage:
    def __init__(self, stage_id: str, minimum_area_ratio: float = 0.0003) -> None:
        self._stage_id = stage_id
        self.minimum_area_ratio = max(0.0, float(minimum_area_ratio))

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        history = filter_motion_blobs(
            list(context.raw_blob_history),
            self.minimum_area_ratio,
        )
        context.filtered_blob_history = tuple(history)
        context.blobs = [blob for frame in history for blob in frame.blobs]
        return context


class DominantCentroidTrackingStage:
    def __init__(
        self,
        stage_id: str,
        minimum_active_area_ratio: float = 0.0008,
        minimum_changed_ratio: float = 0.003,
    ) -> None:
        self._stage_id = stage_id
        self.minimum_active_area_ratio = max(0.0, float(minimum_active_area_ratio))
        self.minimum_changed_ratio = max(0.0, float(minimum_changed_ratio))

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        track = track_dominant_motion(
            list(context.filtered_blob_history),
            self.minimum_active_area_ratio,
            self.minimum_changed_ratio,
        )
        context.dominant_track = track
        context.tracked_objects = [track] if track.observations else []
        return context


class MotionScoringStage:
    def __init__(self, stage_id: str) -> None:
        self._stage_id = stage_id

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        if context.dominant_track is None:
            raise ValueError("dominant motion track is required")
        sensitivity = str(context.configuration.get("sensitivity") or "balanced")
        result = score_motion_track(
            context.dominant_track,
            len(context.processed_frame_history),
            sensitivity,
        )
        scored_track = replace(context.dominant_track, score=result.score)
        context.dominant_track = scored_track
        context.tracked_objects = [scored_track] if scored_track.observations else []
        context.scoring = MotionScoring(
            accepted=result.accepted,
            score=result.score,
            threshold=result.threshold,
            reason=result.reason,
            frame_count=result.frame_count,
            features=dict(result.features),
        )
        return context


class LegacyMotionScoringStage:
    """Compatibility stage retaining combined blob, score, state, and trigger policy."""

    def __init__(self, stage_id: str) -> None:
        self._stage_id = stage_id

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        sensitivity = str(context.configuration.get("sensitivity") or "balanced")
        result = score_motion_masks(
            list(context.processed_frame_history),
            list(context.motion_mask_history),
            sensitivity,
        )
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
            evidence_sources=("frame_difference",),
        )
        return context


def _build_preprocessor(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> MotionFramePreprocessorStage:
    del options, dependencies
    return MotionFramePreprocessorStage(stage_id)


def _build_difference(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> FrameDifferenceStage:
    del options, dependencies
    return FrameDifferenceStage(stage_id)


def _build_threshold(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> FixedThresholdStage:
    del dependencies
    return FixedThresholdStage(stage_id, int(options.get("value", 18)))


def _build_morphology(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> OpenCloseMorphologyStage:
    del dependencies
    return OpenCloseMorphologyStage(
        stage_id,
        kernel_size=int(options.get("kernel_size", 3)),
        close_iterations=int(options.get("close_iterations", 2)),
    )


def _build_scorer(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> LegacyMotionScoringStage:
    del options, dependencies
    return LegacyMotionScoringStage(stage_id)


def _build_blob_extractor(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> BlobExtractionStage:
    del options, dependencies
    return BlobExtractionStage(stage_id)


def _build_blob_filter(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> BlobFilteringStage:
    del dependencies
    return BlobFilteringStage(stage_id, float(options.get("minimum_area_ratio", 0.0003)))


def _build_tracker(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> DominantCentroidTrackingStage:
    del dependencies
    return DominantCentroidTrackingStage(
        stage_id,
        minimum_active_area_ratio=float(options.get("minimum_active_area_ratio", 0.0008)),
        minimum_changed_ratio=float(options.get("minimum_changed_ratio", 0.003)),
    )


def _build_motion_scorer(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> MotionScoringStage:
    del options, dependencies
    return MotionScoringStage(stage_id)


def register_image_motion_stages(registry: MotionStageRegistry) -> None:
    registry.register(
        MotionStageRegistration(
            implementation="gray_blur",
            builder=_build_preprocessor,
            requires=frozenset({"frame_history"}),
            provides=frozenset({"processed_frame", "processed_frame_history"}),
            graph="qualification",
            category="preprocessing",
            display_name="Grayscale smoothing",
            description="Reduces image noise before motion is measured.",
        )
    )
    registry.register(
        MotionStageRegistration(
            implementation="frame_difference",
            builder=_build_difference,
            requires=frozenset({"processed_frame_history"}),
            provides=frozenset({"difference_image", "difference_history"}),
            graph="qualification",
            category="difference",
            display_name="Frame-to-frame difference",
            description="Measures changes between consecutive video frames.",
        )
    )
    registry.register(
        MotionStageRegistration(
            implementation="fixed_threshold",
            builder=_build_threshold,
            requires=frozenset({"difference_history"}),
            provides=frozenset({"binary_motion_mask", "threshold_mask_history"}),
            graph="qualification",
            category="threshold",
            display_name="Fixed change threshold",
            description="Separates meaningful pixel changes from minor video noise.",
            options=(MotionStageOption(
                "value", "Change threshold", "integer", 18,
                "Higher values ignore more subtle changes.", 0, 255,
            ),),
        )
    )
    registry.register(
        MotionStageRegistration(
            implementation="open_close",
            builder=_build_morphology,
            requires=frozenset({"threshold_mask_history"}),
            provides=frozenset({"binary_motion_mask", "motion_mask_history"}),
            graph="qualification",
            category="morphology",
            display_name="Mask cleanup",
            description="Removes speckles and joins nearby motion areas.",
            options=(
                MotionStageOption("kernel_size", "Cleanup size", "integer", 3, minimum=1, maximum=15),
                MotionStageOption("close_iterations", "Join passes", "integer", 2, minimum=0, maximum=10),
            ),
        )
    )
    registry.register(
        MotionStageRegistration(
            implementation="legacy_motion_scorer",
            builder=_build_scorer,
            requires=frozenset({"processed_frame_history", "motion_mask_history"}),
            provides=frozenset({"scoring", "event_state", "decision"}),
            graph="qualification",
            category="compatibility",
            display_name="Classic combined scorer",
            description="Preserves the earlier combined mask, score, and trigger behavior.",
        )
    )
    registry.register(
        MotionStageRegistration(
            implementation="contour_blobs",
            builder=_build_blob_extractor,
            requires=frozenset({"processed_frame_history", "motion_mask_history"}),
            provides=frozenset({"raw_blob_history"}),
            graph="qualification",
            category="blob_extraction",
            display_name="Contour extraction",
            description="Turns connected motion-mask regions into candidate objects.",
        )
    )
    registry.register(
        MotionStageRegistration(
            implementation="minimum_area",
            builder=_build_blob_filter,
            requires=frozenset({"raw_blob_history"}),
            provides=frozenset({"filtered_blob_history", "blobs"}),
            graph="qualification",
            category="blob_filtering",
            display_name="Minimum area filter",
            description="Removes motion regions that are too small to be useful.",
            options=(MotionStageOption(
                "minimum_area_ratio", "Minimum area", "number", 0.0003,
                "Smallest accepted region as a portion of the frame.", 0, 1, advanced=True,
            ),),
        )
    )
    registry.register(
        MotionStageRegistration(
            implementation="dominant_centroid",
            builder=_build_tracker,
            requires=frozenset({"filtered_blob_history"}),
            provides=frozenset({"dominant_track", "tracked_objects"}),
            graph="qualification",
            category="tracking",
            display_name="Centroid motion tracking",
            description="Tracks the dominant moving region by its center point.",
            options=(
                MotionStageOption("minimum_active_area_ratio", "Minimum active area", "number", 0.0008, minimum=0, maximum=1, advanced=True),
                MotionStageOption("minimum_changed_ratio", "Minimum changed area", "number", 0.003, minimum=0, maximum=1, advanced=True),
            ),
        )
    )
    registry.register(
        MotionStageRegistration(
            implementation="default_motion_score",
            builder=_build_motion_scorer,
            requires=frozenset({"dominant_track", "processed_frame_history"}),
            provides=frozenset({"scoring"}),
            graph="qualification",
            category="scoring",
            display_name="SurvNG motion score",
            description="Scores motion persistence, continuity, area, and image stability.",
        )
    )

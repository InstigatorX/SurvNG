"""Image stages shared by the EMA qualification graph.

The fixed-threshold frame-difference graph that also lived here was retired;
``configuration.RETIRED_QUALIFICATION_IMPLEMENTATIONS`` names its stages.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..motion import morphology_motion_masks, preprocess_motion_frames
from .context import MotionContext
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


def _build_preprocessor(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> MotionFramePreprocessorStage:
    del options, dependencies
    return MotionFramePreprocessorStage(stage_id)


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

from __future__ import annotations

from dataclasses import dataclass

from ..config import (
    CameraMotionQualificationConfig,
    MotionQualificationConfig,
    MotionStageSelection,
)
from .defaults import (
    default_motion_fusion_stage_configs,
    default_motion_observation_stage_configs,
    default_motion_stage_configs,
)
from .factory import MotionStageConfig


@dataclass(frozen=True, slots=True)
class ResolvedMotionPipelineGraphs:
    qualification: tuple[MotionStageConfig, ...]
    observation: tuple[MotionStageConfig, ...]
    fusion: tuple[MotionStageConfig, ...]
    origins: dict[str, str]


def _stage_configs(selections: list[MotionStageSelection]) -> tuple[MotionStageConfig, ...]:
    return tuple(
        MotionStageConfig(
            stage_id=selection.stage_id,
            implementation=selection.implementation,
            options=dict(selection.options),
        )
        for selection in selections
    )


def _resolve_graph(
    graph_name: str,
    global_stages: list[MotionStageSelection],
    camera_stages: list[MotionStageSelection] | None,
    defaults: list[MotionStageConfig],
) -> tuple[tuple[MotionStageConfig, ...], str]:
    if camera_stages is not None:
        if not camera_stages:
            raise ValueError(f"camera motion {graph_name} graph cannot be empty")
        return _stage_configs(camera_stages), "camera"
    if global_stages:
        return _stage_configs(global_stages), "global"
    return tuple(defaults), "default"


def resolve_motion_pipeline_graphs(
    global_config: MotionQualificationConfig,
    camera_config: CameraMotionQualificationConfig,
) -> ResolvedMotionPipelineGraphs:
    mode = global_config.mode if camera_config.mode == "inherit" else camera_config.mode
    mog2_requested = (
        global_config.mog2_audit_enabled
        if camera_config.mog2_audit_enabled is None
        else camera_config.mog2_audit_enabled
    )
    qualification, qualification_origin = _resolve_graph(
        "qualification",
        global_config.pipeline.qualification,
        camera_config.pipeline.qualification,
        default_motion_stage_configs(),
    )
    observation, observation_origin = _resolve_graph(
        "observation",
        global_config.pipeline.observation,
        camera_config.pipeline.observation,
        default_motion_observation_stage_configs(
            mog2_enabled=bool(mog2_requested and mode == "audit"),
            sample_fps=global_config.sample_fps,
            mog2_history_seconds=global_config.mog2_history_seconds,
        ),
    )
    fusion, fusion_origin = _resolve_graph(
        "fusion",
        global_config.pipeline.fusion,
        camera_config.pipeline.fusion,
        default_motion_fusion_stage_configs(),
    )
    return ResolvedMotionPipelineGraphs(
        qualification=qualification,
        observation=observation,
        fusion=fusion,
        origins={
            "qualification": qualification_origin,
            "observation": observation_origin,
            "fusion": fusion_origin,
        },
    )

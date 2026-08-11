from __future__ import annotations

from dataclasses import dataclass

from ..config import (
    CameraMotionQualificationConfig,
    MotionQualificationConfig,
    MotionStageSelection,
)
from .defaults import (
    adaptive_motion_stage_configs,
    default_motion_fusion_stage_configs,
    default_motion_observation_stage_configs,
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
            parallel_group=selection.parallel_group,
        )
        for selection in selections
    )


def _resolve_graph(
    graph_name: str,
    global_stages: list[MotionStageSelection],
    camera_stages: list[MotionStageSelection] | None,
    defaults: list[MotionStageConfig],
) -> tuple[tuple[MotionStageConfig, ...], str]:
    def migrate_legacy(
        stages: list[MotionStageSelection],
        origin: str,
    ) -> tuple[tuple[MotionStageConfig, ...], str]:
        if (
            len(stages) == 1
            and stages[0].implementation in {"legacy_qualifier", "legacy_motion_scorer"}
        ):
            return tuple(defaults), f"{origin}_legacy_migrated"
        return _stage_configs(stages), origin

    if camera_stages is not None:
        if not camera_stages:
            raise ValueError(f"camera motion {graph_name} graph cannot be empty")
        return migrate_legacy(camera_stages, "camera")
    if global_stages:
        return migrate_legacy(global_stages, "global")
    return tuple(defaults), "default"


def _normalized_sources(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return ()
    return tuple(dict.fromkeys(
        normalized
        for source in values
        if (normalized := str(source).strip().lower())
    ))


def _validate_trigger_source_separation(
    mode: str,
    fusion: tuple[MotionStageConfig, ...],
) -> None:
    if mode not in {"camera", "camera_rescue", "adaptive"}:
        return
    for stage in fusion:
        if stage.implementation != "buffered_evidence_fusion":
            continue
        sources = _normalized_sources(stage.options.get("sources", []))
        if "onvif" in sources:
            raise ValueError(
                "ONVIF cannot be a validation source in camera/adaptive mode; "
                "it is the camera trigger in camera modes and diagnostic-only in adaptive mode"
            )


def resolved_trigger_mode(mode: str) -> str:
    """Resolve legacy modes to an explicit trigger model."""
    if mode in {"adaptive", "enforce"}:
        return "adaptive"
    if mode == "camera_rescue":
        return "camera_rescue"
    return "camera"


def resolve_motion_pipeline_graphs(
    global_config: MotionQualificationConfig,
    camera_config: CameraMotionQualificationConfig,
) -> ResolvedMotionPipelineGraphs:
    mode = global_config.mode if camera_config.mode == "inherit" else camera_config.mode
    qualification, qualification_origin = _resolve_graph(
        "qualification",
        global_config.pipeline.qualification,
        camera_config.pipeline.qualification,
        adaptive_motion_stage_configs(),
    )
    fusion, fusion_origin = _resolve_graph(
        "fusion",
        global_config.pipeline.fusion,
        camera_config.pipeline.fusion,
        default_motion_fusion_stage_configs(),
    )
    _validate_trigger_source_separation(mode, fusion)
    observation, observation_origin = _resolve_graph(
        "observation",
        global_config.pipeline.observation,
        camera_config.pipeline.observation,
        default_motion_observation_stage_configs(),
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

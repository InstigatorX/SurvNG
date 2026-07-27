from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..config import MotionStageSelection
from .catalog import builtin_motion_pipeline_presets
from .factory import MotionStageConfig


def _normalized_stage(stage: MotionStageConfig) -> tuple[str, str, dict[str, Any], str]:
    return (
        stage.stage_id,
        stage.implementation,
        dict(stage.options),
        stage.parallel_group,
    )


def identify_analysis_preset(stages: Sequence[MotionStageConfig]) -> str:
    normalized = [_normalized_stage(stage) for stage in stages]
    for preset in builtin_motion_pipeline_presets():
        if preset.graph == "qualification" and normalized == [
            _normalized_stage(stage) for stage in preset.stages
        ]:
            return preset.preset_id
    return "custom"


def analysis_preset_selections(preset_id: str) -> list[MotionStageSelection]:
    for preset in builtin_motion_pipeline_presets():
        if preset.graph == "qualification" and preset.preset_id == preset_id:
            return stage_selections(preset.stages)
    raise ValueError(f"unknown motion analysis preset: {preset_id}")


def stage_selections(stages: Sequence[MotionStageConfig]) -> list[MotionStageSelection]:
    return [
        MotionStageSelection(
            stage_id=stage.stage_id,
            implementation=stage.implementation,
            options=dict(stage.options),
            parallel_group=stage.parallel_group,
        )
        for stage in stages
    ]


def guided_fusion_settings(stages: Sequence[MotionStageConfig]) -> dict[str, Any]:
    for stage in stages:
        if stage.implementation == "buffered_evidence_fusion":
            raw_sources = stage.options.get("sources", [])
            source_values = (raw_sources,) if isinstance(raw_sources, str) else raw_sources
            sources = list(dict.fromkeys(
                normalized
                for source in source_values
                if (normalized := str(source).strip().lower())
            )) if isinstance(source_values, (list, tuple)) else []
            return {
                "guided": True,
                "policy": str(stage.options.get("policy", "audit")).strip().lower(),
                "sources": sources,
                "source_thresholds": dict(stage.options.get("source_thresholds", {})),
                "source_weights": dict(stage.options.get("source_weights", {})),
                "weighted_threshold": float(stage.options.get("weighted_threshold", 0.5)),
                "include_primary": bool(stage.options.get("include_primary", True)),
                "fail_open": bool(stage.options.get("fail_open", True)),
            }
    return {"guided": False}


def update_guided_fusion(
    stages: Sequence[MotionStageConfig],
    setting: str,
    value: Any,
) -> list[MotionStageSelection]:
    selections = stage_selections(stages)
    fusion = next(
        (
            stage
            for stage in selections
            if stage.implementation == "buffered_evidence_fusion"
        ),
        None,
    )
    if fusion is None:
        raise ValueError("the active custom decision pipeline has no guided fusion stage")
    if setting == "fusion_policy":
        fusion.options["policy"] = value
    elif setting == "fusion_sources":
        fusion.options["sources"] = list(value)
    else:
        raise ValueError(f"unsupported guided fusion setting: {setting}")
    return selections

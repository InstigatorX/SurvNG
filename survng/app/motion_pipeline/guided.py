from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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

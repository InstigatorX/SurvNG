from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .defaults import adaptive_motion_stage_configs, default_motion_stage_configs
from .factory import MotionStageConfig
from .registry import MotionStageRegistry


@dataclass(frozen=True, slots=True)
class MotionPipelinePreset:
    preset_id: str
    label: str
    description: str
    graph: str
    stages: tuple[MotionStageConfig, ...]
    recommended: bool = False

    def as_dict(self, registry: MotionStageRegistry) -> dict[str, Any]:
        registered = set(registry.implementations())
        unavailable = sorted({
            stage.implementation
            for stage in self.stages
            if stage.implementation not in registered
        })
        return {
            "id": self.preset_id,
            "label": self.label,
            "description": self.description,
            "graph": self.graph,
            "recommended": self.recommended,
            "available": not unavailable,
            "unavailable_implementations": unavailable,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "implementation": stage.implementation,
                    "options": dict(stage.options),
                    **(
                        {"parallel_group": stage.parallel_group}
                        if stage.parallel_group
                        else {}
                    ),
                }
                for stage in self.stages
            ],
        }


def builtin_motion_pipeline_presets() -> tuple[MotionPipelinePreset, ...]:
    return (
        MotionPipelinePreset(
            preset_id="adaptive",
            label="Adaptive motion analysis",
            description=(
                "Learns each camera scene and automatically adjusts for lighting, "
                "sensor noise, nuisance regions, and insect-like motion."
            ),
            graph="qualification",
            stages=tuple(adaptive_motion_stage_configs()),
            recommended=True,
        ),
        MotionPipelinePreset(
            preset_id="modular",
            label="Modular motion analysis",
            description=(
                "Original fixed-threshold multi-stage analysis retained for comparison "
                "and rollback."
            ),
            graph="qualification",
            stages=tuple(default_motion_stage_configs()),
        ),
        MotionPipelinePreset(
            preset_id="classic",
            label="Classic compatibility",
            description=(
                "Original all-in-one SurvNG analysis for troubleshooting or behavior "
                "comparison."
            ),
            graph="qualification",
            stages=(
                MotionStageConfig(
                    stage_id="qualification",
                    implementation="legacy_qualifier",
                ),
            ),
        ),
    )


def motion_pipeline_catalog(registry: MotionStageRegistry) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stages": list(registry.catalog()),
        "presets": [
            preset.as_dict(registry)
            for preset in builtin_motion_pipeline_presets()
        ],
    }

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .contracts import MotionPipelineObserver
from .pipeline import MotionErrorPolicy, MotionPipeline
from .registry import MotionStageDependencies, MotionStageRegistry


@dataclass(frozen=True, slots=True)
class MotionStageConfig:
    stage_id: str
    implementation: str
    options: Mapping[str, Any] = field(default_factory=dict)


class MotionPipelineFactory:
    def __init__(
        self,
        registry: MotionStageRegistry,
        dependencies: MotionStageDependencies | None = None,
        observer: MotionPipelineObserver | None = None,
    ) -> None:
        self.registry = registry
        self.dependencies = dependencies or MotionStageDependencies()
        self.observer = observer

    def create(
        self,
        camera_id: str,
        stage_configs: Sequence[MotionStageConfig],
        error_policy: MotionErrorPolicy = "raise",
        initial_artifacts: Iterable[str] = (),
        required_artifacts: Iterable[str] = (),
    ) -> MotionPipeline:
        stage_ids: set[str] = set()
        available_artifacts = {
            "original_frame",
            "frame_history",
            "configuration",
            "runtime",
            *initial_artifacts,
        }
        stages = []
        for stage_config in stage_configs:
            stage_id = stage_config.stage_id.strip()
            if not stage_id:
                raise ValueError("motion stage ID cannot be empty")
            if stage_id in stage_ids:
                raise ValueError(f"duplicate motion stage ID: {stage_id}")
            registration = self.registry.registration(stage_config.implementation)
            missing = registration.requires - available_artifacts
            if missing:
                raise ValueError(
                    f"motion stage {stage_id!r} requires unavailable artifacts: {', '.join(sorted(missing))}"
                )
            stages.append(registration.builder(stage_id, stage_config.options, self.dependencies))
            stage_ids.add(stage_id)
            available_artifacts.update(registration.provides)
        missing_outputs = set(required_artifacts) - available_artifacts
        if missing_outputs:
            raise ValueError(
                "motion pipeline does not provide required artifacts: "
                + ", ".join(sorted(missing_outputs))
            )
        return MotionPipeline(
            camera_id=camera_id,
            stages=stages,
            observer=self.observer,
            error_policy=error_policy,
            stage_configuration=[
                {
                    "stage_id": stage.stage_id,
                    "implementation": stage.implementation,
                    "options": dict(stage.options),
                }
                for stage in stage_configs
            ],
        )

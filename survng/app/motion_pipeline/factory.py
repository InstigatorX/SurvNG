from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .contracts import MotionPipelineObserver, MotionStage
from .pipeline import MotionErrorPolicy, MotionPipeline
from .registry import MotionStageDependencies, MotionStageRegistry


@dataclass(frozen=True, slots=True)
class MotionStageConfig:
    stage_id: str
    implementation: str
    options: Mapping[str, Any] = field(default_factory=dict)
    parallel_group: str = ""


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
        registrations = []
        execution_groups: list[list[int]] = []
        current_parallel_group = ""
        for stage_config in stage_configs:
            stage_id = stage_config.stage_id.strip()
            if not stage_id:
                raise ValueError("motion stage ID cannot be empty")
            if stage_id in stage_ids:
                raise ValueError(f"duplicate motion stage ID: {stage_id}")
            registration = self.registry.registration(stage_config.implementation)
            parallel_group = stage_config.parallel_group.strip()
            if parallel_group and parallel_group == current_parallel_group:
                group_available = available_artifacts - set().union(
                    *(registrations[index].provides for index in execution_groups[-1])
                )
            else:
                group_available = available_artifacts
            missing = registration.requires - group_available
            if missing:
                raise ValueError(
                    f"motion stage {stage_id!r} requires unavailable artifacts: {', '.join(sorted(missing))}"
                )
            stages.append(registration.builder(stage_id, stage_config.options, self.dependencies))
            registrations.append(registration)
            stage_ids.add(stage_id)
            available_artifacts.update(registration.provides)
            if parallel_group and parallel_group == current_parallel_group:
                existing_provides = set().union(
                    *(registrations[index].provides for index in execution_groups[-1])
                )
                conflicting = (existing_provides & registration.provides) - {
                    "source_evidence",
                    "debug",
                }
                if conflicting:
                    raise ValueError(
                        f"parallel motion group {parallel_group!r} has conflicting outputs: "
                        + ", ".join(sorted(conflicting))
                    )
                execution_groups[-1].append(len(stages) - 1)
            else:
                execution_groups.append([len(stages) - 1])
            current_parallel_group = parallel_group
        missing_outputs = set(required_artifacts) - available_artifacts
        if missing_outputs:
            raise ValueError(
                "motion pipeline does not provide required artifacts: "
                + ", ".join(sorted(missing_outputs))
            )
        stage_blueprint = tuple(
            (
                stage_config.stage_id.strip(),
                registration.builder,
                dict(stage_config.options),
            )
            for stage_config, registration in zip(
                stage_configs,
                registrations,
                strict=True,
            )
        )

        def build_isolated_stages() -> tuple[MotionStage, ...]:
            return tuple(
                builder(stage_id, dict(options), self.dependencies)
                for stage_id, builder, options in stage_blueprint
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
                    **(
                        {"parallel_group": stage.parallel_group}
                        if stage.parallel_group
                        else {}
                    ),
                }
                for stage in stage_configs
            ],
            execution_groups=execution_groups,
            stage_provides={
                stage.stage_id: registration.provides
                for stage, registration in zip(stages, registrations, strict=True)
            },
            continuous_analysis=any(
                registration.continuous_analysis
                for registration in registrations
            ),
            motion_sources=tuple(
                dict.fromkeys(
                    registration.motion_source
                    for registration in registrations
                    if registration.motion_source
                )
            ),
            stage_factory=build_isolated_stages,
        )

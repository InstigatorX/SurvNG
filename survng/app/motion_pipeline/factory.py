from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contracts import MotionPipelineObserver, MotionStage
from .pipeline import MotionErrorPolicy, MotionPipeline
from .registry import (
    MotionStageDependencies,
    MotionStageRegistration,
    MotionStageRegistry,
)


_MERGEABLE_PARALLEL_OUTPUTS = frozenset({"source_evidence", "debug"})


@dataclass(frozen=True, slots=True)
class MotionStageConfig:
    stage_id: str
    implementation: str
    options: Mapping[str, Any] = field(default_factory=dict)
    parallel_group: str = ""


def _effective_observation_kinds(
    stage: MotionStage,
    registration: MotionStageRegistration,
) -> frozenset[str] | None:
    if registration.observation_kinds is not None:
        return registration.observation_kinds
    declared = getattr(stage, "observation_kinds", None)
    if declared is None:
        return None
    return frozenset(str(kind) for kind in declared)


def _observation_kinds_overlap(
    left: frozenset[str] | None,
    right: frozenset[str] | None,
) -> bool:
    return left is None or right is None or bool(left & right)


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
        effective_observation_kinds: list[frozenset[str] | None] = []
        execution_groups: list[list[int]] = []
        current_parallel_group = ""
        current_group_available = set(available_artifacts)
        requires_observation_kind = False
        artifact_providers: dict[str, list[frozenset[str] | None]] = {}
        for stage_config in stage_configs:
            stage_id = stage_config.stage_id.strip()
            if not stage_id:
                raise ValueError("motion stage ID cannot be empty")
            if stage_id in stage_ids:
                raise ValueError(f"duplicate motion stage ID: {stage_id}")
            registration = self.registry.registration(stage_config.implementation)
            parallel_group = stage_config.parallel_group.strip()
            continues_parallel_group = bool(
                parallel_group and parallel_group == current_parallel_group
            )
            if not continues_parallel_group:
                current_group_available = set(available_artifacts)
            missing = registration.requires - current_group_available
            if missing:
                raise ValueError(
                    f"motion stage {stage_id!r} requires unavailable artifacts: {', '.join(sorted(missing))}"
                )
            stage = registration.builder(stage_id, stage_config.options, self.dependencies)
            stages.append(stage)
            registrations.append(registration)
            stage_observation_kinds = _effective_observation_kinds(stage, registration)
            effective_observation_kinds.append(stage_observation_kinds)
            for artifact in registration.provides - _MERGEABLE_PARALLEL_OUTPUTS:
                previous_providers = artifact_providers.setdefault(artifact, [])
                if any(
                    not _observation_kinds_overlap(previous, stage_observation_kinds)
                    for previous in previous_providers
                ):
                    requires_observation_kind = True
                previous_providers.append(stage_observation_kinds)
            stage_ids.add(stage_id)
            available_artifacts.update(registration.provides)
            if continues_parallel_group:
                overlapping_stages = (
                    index
                    for index in execution_groups[-1]
                    if _observation_kinds_overlap(
                        effective_observation_kinds[index],
                        effective_observation_kinds[-1],
                    )
                )
                existing_provides = set().union(
                    *(registrations[index].provides for index in overlapping_stages)
                )
                conflicting = (
                    existing_provides & registration.provides
                ) - _MERGEABLE_PARALLEL_OUTPUTS
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
        observation_kinds = sorted({
            kind
            for kinds in effective_observation_kinds
            for kind in (kinds or ())
        })
        def validate_observation_graph(
            label: str,
            runnable_for: Callable[[int], bool],
        ) -> None:
            kind_available = {
                "original_frame",
                "frame_history",
                "configuration",
                "runtime",
                *initial_artifacts,
            }
            has_runnable_stage = False
            for group in execution_groups:
                runnable = [
                    index
                    for index in group
                    if runnable_for(index)
                ]
                if not runnable:
                    continue
                has_runnable_stage = True
                for index in runnable:
                    missing = registrations[index].requires - kind_available
                    if missing:
                        stage_id = stage_configs[index].stage_id.strip()
                        raise ValueError(
                            f"motion stage {stage_id!r} requires unavailable artifacts "
                            f"for {label}: "
                            + ", ".join(sorted(missing))
                        )
                for index in runnable:
                    kind_available.update(registrations[index].provides)
            kind_missing_outputs = set(required_artifacts) - kind_available
            if has_runnable_stage and kind_missing_outputs:
                raise ValueError(
                    f"motion pipeline does not provide required artifacts for {label}: "
                    + ", ".join(sorted(kind_missing_outputs))
                )

        for observation_kind in observation_kinds:
            validate_observation_graph(
                f"observation {observation_kind!r}",
                lambda index, kind=observation_kind: (
                    effective_observation_kinds[index] is None
                    or kind in effective_observation_kinds[index]
                ),
            )
        if any(kinds is None for kinds in effective_observation_kinds):
            validate_observation_graph(
                "unrestricted observations",
                lambda index: effective_observation_kinds[index] is None,
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
            stage_observation_kinds={
                stage.stage_id: kinds
                for stage, kinds in zip(stages, effective_observation_kinds, strict=True)
            },
            requires_observation_kind=requires_observation_kind,
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

from __future__ import annotations

import math
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
        constructed_stages: list[MotionStage] = []
        try:
            return self._create(
                camera_id,
                stage_configs,
                error_policy,
                initial_artifacts,
                required_artifacts,
                constructed_stages,
            )
        except BaseException:
            self._close_stages(constructed_stages)
            raise

    def _create(
        self,
        camera_id: str,
        stage_configs: Sequence[MotionStageConfig],
        error_policy: MotionErrorPolicy,
        initial_artifacts: Iterable[str],
        required_artifacts: Iterable[str],
        constructed_stages: list[MotionStage],
    ) -> MotionPipeline:
        stage_ids: set[str] = set()
        available_artifacts = {
            "original_frame",
            "frame_history",
            "configuration",
            "runtime",
            *initial_artifacts,
        }
        stages = constructed_stages
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
            self._validate_options(stage_id, registration, stage_config.options)
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
            try:
                stage = registration.builder(
                    stage_id,
                    stage_config.options,
                    self.dependencies,
                )
            except Exception as error:
                raise ValueError(
                    f"motion stage {stage_id!r} implementation "
                    f"{stage_config.implementation!r} could not be configured: {error}"
                ) from error
            stages.append(stage)
            if not isinstance(stage, MotionStage):
                raise ValueError(
                    f"motion stage {stage_id!r} implementation "
                    f"{stage_config.implementation!r} did not return a MotionStage"
                )
            if stage.stage_id != stage_id:
                raise ValueError(
                    f"motion stage builder returned ID {stage.stage_id!r}; "
                    f"expected {stage_id!r}"
                )
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
            isolated_stages: list[MotionStage] = []
            try:
                for stage_id, builder, options in stage_blueprint:
                    stage = builder(stage_id, dict(options), self.dependencies)
                    isolated_stages.append(stage)
                    if not isinstance(stage, MotionStage):
                        raise ValueError(
                            f"isolated motion stage {stage_id!r} did not return a MotionStage"
                        )
                    if stage.stage_id != stage_id:
                        raise ValueError(
                            f"isolated motion stage builder returned ID {stage.stage_id!r}; "
                            f"expected {stage_id!r}"
                        )
            except BaseException:
                self._close_stages(isolated_stages)
                raise
            return tuple(isolated_stages)

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

    @staticmethod
    def _validate_options(
        stage_id: str,
        registration: MotionStageRegistration,
        values: Mapping[str, Any],
    ) -> None:
        """Validate configured values against the stage's public option contract."""
        if not registration.options:
            return
        definitions = {option.key: option for option in registration.options}
        unknown = sorted(set(values) - set(definitions))
        if unknown:
            raise ValueError(
                f"motion stage {stage_id!r} has unknown options: "
                + ", ".join(unknown)
            )
        for key, value in values.items():
            option = definitions[key]
            label = f"motion stage {stage_id!r} option {key!r}"
            if option.value_type == "boolean":
                valid = isinstance(value, bool)
            elif option.value_type == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif option.value_type == "number":
                valid = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                )
            elif option.value_type == "string":
                valid = isinstance(value, str)
            elif option.value_type == "string_list":
                # A single string remains accepted for compatibility with the
                # original configuration format; builders normalize it to one item.
                valid = isinstance(value, str) or (
                    isinstance(value, (list, tuple))
                    and all(isinstance(item, str) for item in value)
                )
            else:
                valid = isinstance(value, Mapping)
            if not valid:
                raise ValueError(f"{label} must be a valid {option.value_type}")
            if option.value_type in {"integer", "number"}:
                numeric = float(value)
                if option.minimum is not None and numeric < option.minimum:
                    raise ValueError(f"{label} must be at least {option.minimum:g}")
                if option.maximum is not None and numeric > option.maximum:
                    raise ValueError(f"{label} must be at most {option.maximum:g}")
            if option.choices and value not in option.choices:
                raise ValueError(
                    f"{label} must be one of: " + ", ".join(option.choices)
                )

    def _close_stages(self, stages: Iterable[MotionStage]) -> None:
        closed: set[int] = set()
        for stage in reversed(tuple(stages)):
            identity = id(stage)
            if identity in closed:
                continue
            closed.add(identity)
            close = getattr(stage, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException:
                self.dependencies.logger.exception(
                    "Motion stage cleanup failed during pipeline construction stage=%s",
                    getattr(stage, "stage_id", "<invalid>"),
                )

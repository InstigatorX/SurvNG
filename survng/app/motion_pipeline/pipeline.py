from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .context import MotionContext, StageTiming
from .contracts import MotionPipelineObserver, MotionStage, NullMotionPipelineObserver
from .runtime import MotionRuntimeState


LOGGER = logging.getLogger(__name__)
MotionErrorPolicy = Literal["raise", "skip"]
_SENSITIVE_OPTION_PARTS = ("api_key", "password", "secret", "token", "credential")


def _audit_safe_value(value: object, depth: int = 0) -> object:
    if depth >= 5:
        return "[truncated]"
    if isinstance(value, Mapping):
        safe: dict[str, object] = {}
        for raw_key, item in list(value.items())[:64]:
            key = str(raw_key)[:100]
            if any(part in key.lower() for part in _SENSITIVE_OPTION_PARTS):
                safe[key] = "[redacted]"
            else:
                safe[key] = _audit_safe_value(item, depth + 1)
        return safe
    if isinstance(value, (list, tuple)):
        return [_audit_safe_value(item, depth + 1) for item in value[:64]]
    if isinstance(value, str):
        return value[:500]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:500]


@dataclass(slots=True)
class _StageMetrics:
    calls: int = 0
    failures: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0
    max_ms: float = 0.0

    def record(self, duration_ms: float, succeeded: bool) -> None:
        self.calls += 1
        self.failures += int(not succeeded)
        self.total_ms += duration_ms
        self.last_ms = duration_ms
        self.max_ms = max(self.max_ms, duration_ms)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "last_ms": round(self.last_ms, 3),
            "average_ms": round(self.total_ms / self.calls, 3) if self.calls else 0.0,
            "max_ms": round(self.max_ms, 3),
        }


class LoggingMotionPipelineObserver(NullMotionPipelineObserver):
    def __init__(self, slow_stage_ms: float = 25.0) -> None:
        self.slow_stage_ms = max(0.0, float(slow_stage_ms))

    def stage_completed(self, timing: StageTiming, context: MotionContext) -> None:
        if timing.duration_ms >= self.slow_stage_ms:
            LOGGER.debug(
                "Slow motion stage camera=%s stage=%s duration_ms=%.3f",
                context.camera_id,
                timing.stage_id,
                timing.duration_ms,
            )

    def stage_failed(self, timing: StageTiming, context: MotionContext, error: Exception) -> None:
        LOGGER.exception(
            "Motion stage failed camera=%s stage=%s duration_ms=%.3f: %s",
            context.camera_id,
            timing.stage_id,
            timing.duration_ms,
            error,
        )


class MotionPipeline:
    def __init__(
        self,
        camera_id: str,
        stages: Sequence[MotionStage],
        observer: MotionPipelineObserver | None = None,
        error_policy: MotionErrorPolicy = "raise",
        stage_configuration: Sequence[Mapping[str, Any]] = (),
        execution_groups: Sequence[Sequence[int]] = (),
        stage_provides: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        if not stages:
            raise ValueError("motion pipeline requires at least one stage")
        stage_ids = [stage.stage_id for stage in stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("motion pipeline stage IDs must be unique")
        if error_policy not in {"raise", "skip"}:
            raise ValueError(f"unsupported motion error policy: {error_policy}")
        self.camera_id = camera_id
        self.stages = tuple(stages)
        self.runtime = MotionRuntimeState(camera_id=camera_id)
        self.observer = observer or NullMotionPipelineObserver()
        self.error_policy = error_policy
        self.stage_configuration = tuple(dict(item) for item in stage_configuration)
        normalized_groups = tuple(tuple(int(index) for index in group) for group in execution_groups)
        self.execution_groups = normalized_groups or tuple((index,) for index in range(len(stages)))
        flattened = [index for group in self.execution_groups for index in group]
        if flattened != list(range(len(stages))):
            raise ValueError("motion execution groups must contain every stage exactly once in order")
        self.stage_provides = dict(stage_provides or {})
        parallel_workers = max((len(group) for group in self.execution_groups), default=1)
        self._executor = (
            ThreadPoolExecutor(
                max_workers=parallel_workers,
                thread_name_prefix=f"motion-branch-{camera_id}",
            )
            if parallel_workers > 1
            else None
        )
        self._metrics = {stage.stage_id: _StageMetrics() for stage in self.stages}
        self._metrics_lock = threading.Lock()

    def process(self, context: MotionContext) -> MotionContext:
        if context.camera_id != self.camera_id:
            raise ValueError(
                f"pipeline for {self.camera_id!r} cannot process context for {context.camera_id!r}"
            )
        if context.runtime is not self.runtime:
            raise ValueError("motion context must use the pipeline's per-camera runtime state")

        current = context
        for group in self.execution_groups:
            if len(group) == 1:
                current = self._process_stage(self.stages[group[0]], current)
            else:
                current = self._process_parallel_group(group, current)
        return current

    def isolated_copy(self, *, clone_runtime: bool = False) -> "MotionPipeline":
        """Create a disposable pipeline that cannot mutate this pipeline's runtime or metrics."""
        pipeline = MotionPipeline(
            camera_id=self.camera_id,
            stages=self.stages,
            observer=self.observer,
            error_policy=self.error_policy,
            stage_configuration=self.stage_configuration,
            execution_groups=self.execution_groups,
            stage_provides=self.stage_provides,
        )
        if clone_runtime:
            pipeline.runtime = self.runtime.clone()
        return pipeline

    def uses_implementation(self, implementation: str) -> bool:
        return any(
            item.get("implementation") == implementation
            for item in self.stage_configuration
        )

    def handles_observation(self, kind: str) -> bool:
        return any(
            kinds is None or kind in kinds
            for stage in self.stages
            for kinds in (getattr(stage, "observation_kinds", None),)
        )

    def _process_stage(self, stage: MotionStage, context: MotionContext) -> MotionContext:
        self.observer.stage_started(stage.stage_id, context)
        started_ns = time.perf_counter_ns()
        try:
            next_context = stage.process(context)
            if not isinstance(next_context, MotionContext):
                raise TypeError(f"motion stage {stage.stage_id!r} returned an invalid context")
            if next_context.camera_id != self.camera_id or next_context.runtime is not self.runtime:
                raise ValueError(
                    f"motion stage {stage.stage_id!r} replaced the camera identity or runtime state"
                )
        except Exception as error:
            timing = self._record_timing(stage.stage_id, started_ns, False)
            context.timings[stage.stage_id] = timing
            self.observer.stage_failed(timing, context, error)
            if self.error_policy == "raise":
                raise
            return context
        timing = self._record_timing(stage.stage_id, started_ns, True)
        next_context.timings[stage.stage_id] = timing
        self.observer.stage_completed(timing, next_context)
        return next_context

    def _process_parallel_group(
        self,
        group: tuple[int, ...],
        context: MotionContext,
    ) -> MotionContext:
        if self._executor is None:
            raise RuntimeError("parallel motion executor is unavailable")
        branches = [context.fork() for _index in group]
        futures = [
            self._executor.submit(self._process_stage, self.stages[index], branch)
            for index, branch in zip(group, branches, strict=True)
        ]
        results = [future.result() for future in futures]
        for index, result in zip(group, results, strict=True):
            self._merge_branch(context, result, self.stage_provides.get(self.stages[index].stage_id, frozenset()))
            context.timings.update(result.timings)
            context.debug.values.update(result.debug.values)
            context.debug.images.update(result.debug.images)
        return context

    @staticmethod
    def _merge_branch(
        context: MotionContext,
        branch: MotionContext,
        artifacts: frozenset[str],
    ) -> None:
        for artifact in artifacts:
            if artifact == "source_evidence":
                context.source_evidence.update(branch.source_evidence)
            elif artifact == "debug":
                context.debug.values.update(branch.debug.values)
                context.debug.images.update(branch.debug.images)
            elif hasattr(context, artifact):
                setattr(context, artifact, getattr(branch, artifact))

    def _record_timing(self, stage_id: str, started_ns: int, succeeded: bool) -> StageTiming:
        duration_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        with self._metrics_lock:
            self._metrics[stage_id].record(duration_ms, succeeded)
        return StageTiming(stage_id=stage_id, duration_ms=duration_ms, succeeded=succeeded)

    def status(self) -> dict[str, object]:
        with self._metrics_lock:
            stages = {stage_id: metrics.as_dict() for stage_id, metrics in self._metrics.items()}
        return {
            "camera_id": self.camera_id,
            "runtime_generation": self.runtime.generation,
            "error_policy": self.error_policy,
            "configuration": [dict(item) for item in self.stage_configuration],
            "execution_groups": [
                {
                    "mode": "parallel" if len(group) > 1 else "sequential",
                    "stages": [self.stages[index].stage_id for index in group],
                }
                for group in self.execution_groups
            ],
            "stages": stages,
        }

    def audit_snapshot(
        self,
        timings: Mapping[str, StageTiming] | None = None,
    ) -> dict[str, object]:
        """Return bounded, JSON-safe pipeline telemetry for a persisted decision."""
        status = self.status()
        invocation_timings = {
            stage_id: {
                "duration_ms": round(timing.duration_ms, 3),
                "succeeded": timing.succeeded,
            }
            for stage_id, timing in (timings or {}).items()
        }
        return {
            "configuration": _audit_safe_value(status["configuration"]),
            "execution_groups": status["execution_groups"],
            "invocation_timings": invocation_timings,
            "stage_metrics": status["stages"],
        }

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

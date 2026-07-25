from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Literal, Sequence

from .context import MotionContext, StageTiming
from .contracts import MotionPipelineObserver, MotionStage, NullMotionPipelineObserver
from .runtime import MotionRuntimeState


LOGGER = logging.getLogger(__name__)
MotionErrorPolicy = Literal["raise", "skip"]


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
        for stage in self.stages:
            self.observer.stage_started(stage.stage_id, current)
            started_ns = time.perf_counter_ns()
            succeeded = False
            try:
                next_context = stage.process(current)
                if not isinstance(next_context, MotionContext):
                    raise TypeError(f"motion stage {stage.stage_id!r} returned an invalid context")
                if next_context.camera_id != self.camera_id or next_context.runtime is not self.runtime:
                    raise ValueError(
                        f"motion stage {stage.stage_id!r} replaced the camera identity or runtime state"
                    )
                current = next_context
                succeeded = True
            except Exception as error:
                timing = self._record_timing(stage.stage_id, started_ns, False)
                current.timings[stage.stage_id] = timing
                self.observer.stage_failed(timing, current, error)
                if self.error_policy == "raise":
                    raise
                continue

            timing = self._record_timing(stage.stage_id, started_ns, succeeded)
            current.timings[stage.stage_id] = timing
            self.observer.stage_completed(timing, current)
        return current

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
            "stages": stages,
        }

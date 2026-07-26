from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from .context import MotionContext, StageTiming


@runtime_checkable
class MotionStage(Protocol):
    @property
    def stage_id(self) -> str:
        ...

    def process(self, context: MotionContext) -> MotionContext:
        ...


@runtime_checkable
class MotionStageLifecycle(Protocol):
    """Optional cleanup contract for stages that own native resources."""

    def close(self) -> None:
        ...


@runtime_checkable
class MotionRuntimeSnapshot(Protocol):
    """Opt-in contract for native or otherwise non-deepcopyable stage state."""

    def snapshot(self) -> Self:
        ...


class MotionPipelineObserver(Protocol):
    def stage_started(self, stage_id: str, context: MotionContext) -> None:
        ...

    def stage_completed(self, timing: StageTiming, context: MotionContext) -> None:
        ...

    def stage_failed(self, timing: StageTiming, context: MotionContext, error: Exception) -> None:
        ...


class NullMotionPipelineObserver:
    def stage_started(self, stage_id: str, context: MotionContext) -> None:
        del stage_id, context

    def stage_completed(self, timing: StageTiming, context: MotionContext) -> None:
        del timing, context

    def stage_failed(self, timing: StageTiming, context: MotionContext, error: Exception) -> None:
        del timing, context, error

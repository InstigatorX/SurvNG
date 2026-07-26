from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .context import MotionContext, MotionEventPhase, TriggerDecision
from .registry import (
    MotionStageDependencies,
    MotionStageOption,
    MotionStageRegistration,
    MotionStageRegistry,
)


@dataclass(slots=True)
class _EventRuntime:
    phase: MotionEventPhase = MotionEventPhase.IDLE
    event_key: str = ""
    started_at: float | None = None
    updated_at: float | None = None
    consecutive_accepts: int = 0
    consecutive_rejects: int = 0
    cooldown_until: float | None = None
    transition_reason: str = "initialized"

    def reset(self, reason: str) -> None:
        self.phase = MotionEventPhase.IDLE
        self.event_key = ""
        self.started_at = None
        self.consecutive_accepts = 0
        self.consecutive_rejects = 0
        self.cooldown_until = None
        self.transition_reason = reason


class MotionEventStateStage:
    def __init__(
        self,
        stage_id: str,
        *,
        activation_frames: int = 1,
        release_frames: int = 1,
        cooldown_seconds: float = 0.0,
        state_timeout_seconds: float = 10.0,
    ) -> None:
        self._stage_id = stage_id
        self.activation_frames = max(1, int(activation_frames))
        self.release_frames = max(1, int(release_frames))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.state_timeout_seconds = max(0.0, float(state_timeout_seconds))

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        state = context.runtime.state_for(self.stage_id, _EventRuntime)
        now = context.captured_at
        if state.updated_at is not None and now < state.updated_at:
            state.reset("clock_reset")

        if state.phase == MotionEventPhase.COOLDOWN:
            if state.cooldown_until is not None and now < state.cooldown_until:
                state.consecutive_accepts = 0
                state.consecutive_rejects = 0
                state.transition_reason = "cooldown_active"
                state.updated_at = now
                self._publish(context, state)
                return context
            state.reset("cooldown_complete")
        elif (
            state.updated_at is not None
            and self.state_timeout_seconds > 0
            and now - state.updated_at > self.state_timeout_seconds
        ):
            state.reset("state_timeout")

        if context.scoring.accepted:
            state.consecutive_accepts += 1
            state.consecutive_rejects = 0
            if state.phase == MotionEventPhase.ACTIVE:
                state.transition_reason = "active_confirmed"
            elif state.consecutive_accepts >= self.activation_frames:
                state.phase = MotionEventPhase.ACTIVE
                state.started_at = now
                state.event_key = f"{context.camera_id}:{round(now * 1000)}"
                state.transition_reason = "activation_threshold"
            else:
                state.phase = MotionEventPhase.CANDIDATE
                state.transition_reason = "activation_pending"
        else:
            state.consecutive_accepts = 0
            state.consecutive_rejects += 1
            if (
                state.phase == MotionEventPhase.ACTIVE
                and state.consecutive_rejects < self.release_frames
            ):
                state.transition_reason = "release_pending"
            elif state.phase == MotionEventPhase.ACTIVE and self.cooldown_seconds > 0:
                state.phase = MotionEventPhase.COOLDOWN
                state.cooldown_until = now + self.cooldown_seconds
                state.transition_reason = "cooldown_started"
            else:
                state.phase = MotionEventPhase.REJECTED
                state.event_key = ""
                state.started_at = None
                state.cooldown_until = None
                state.transition_reason = "rejected"

        state.updated_at = now
        self._publish(context, state)
        return context

    @staticmethod
    def _publish(context: MotionContext, state: _EventRuntime) -> None:
        context.event_state.phase = state.phase
        context.event_state.event_key = state.event_key
        context.event_state.started_at = state.started_at
        context.event_state.updated_at = state.updated_at
        context.event_state.consecutive_accepts = state.consecutive_accepts
        context.event_state.consecutive_rejects = state.consecutive_rejects
        context.event_state.cooldown_until = state.cooldown_until
        context.event_state.transition_reason = state.transition_reason


class ObjectDetectionTriggerStage:
    def __init__(self, stage_id: str) -> None:
        self._stage_id = stage_id

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        active = context.event_state.phase == MotionEventPhase.ACTIVE
        newly_active = active and context.event_state.transition_reason == "activation_threshold"
        if newly_active and context.scoring.accepted:
            reason = context.scoring.reason
        elif active:
            reason = "event_state_active"
        elif context.event_state.phase == MotionEventPhase.CANDIDATE:
            reason = "event_state_candidate"
        elif context.event_state.phase == MotionEventPhase.COOLDOWN:
            reason = "event_state_cooldown"
        else:
            reason = context.scoring.reason
        primary_source = str(
            context.scoring.features.get("primary_motion_source")
            or context.configuration.get("primary_motion_source")
            or "frame_difference"
        )
        sources = (primary_source, *sorted(context.source_evidence))
        context.decision = TriggerDecision(
            run_object_detection=newly_active,
            reason=reason,
            score=context.scoring.score,
            evidence_sources=tuple(dict.fromkeys(sources)),
        )
        return context


def _build_event_state(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> MotionEventStateStage:
    del dependencies
    return MotionEventStateStage(
        stage_id,
        activation_frames=int(options.get("activation_frames", 1)),
        release_frames=int(options.get("release_frames", 1)),
        cooldown_seconds=float(options.get("cooldown_seconds", 0.0)),
        state_timeout_seconds=float(options.get("state_timeout_seconds", 10.0)),
    )


def _build_trigger(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> ObjectDetectionTriggerStage:
    del options, dependencies
    return ObjectDetectionTriggerStage(stage_id)


def register_decision_stages(registry: MotionStageRegistry) -> None:
    registry.register(
        MotionStageRegistration(
            implementation="score_event_state",
            builder=_build_event_state,
            requires=frozenset({"scoring"}),
            provides=frozenset({"event_state"}),
            graph="fusion",
            category="event_state",
            display_name="Motion event stability",
            description="Controls how quickly motion starts, ends, and becomes eligible again.",
            options=(
                MotionStageOption("activation_frames", "Signals to start", "integer", 1, minimum=1, maximum=20),
                MotionStageOption("release_frames", "Quiet signals to end", "integer", 1, minimum=1, maximum=20),
                MotionStageOption("cooldown_seconds", "Pause after event", "number", 0.0, minimum=0, maximum=300),
                MotionStageOption("state_timeout_seconds", "Unfinished event timeout", "number", 10.0, minimum=0, maximum=300, advanced=True),
            ),
        )
    )
    registry.register(
        MotionStageRegistration(
            implementation="score_trigger",
            builder=_build_trigger,
            requires=frozenset({"scoring", "event_state"}),
            provides=frozenset({"decision"}),
            graph="fusion",
            category="trigger",
            display_name="Object detection trigger",
            description="Turns an active motion event into an object-detection decision.",
        )
    )

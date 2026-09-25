"""Low-latency activity transitions projected from admitted motion observations."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Callable

from .domain_events import MotionObserved

LOGGER = logging.getLogger(__name__)


class ActivityState(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class ActivityTransitionKind(StrEnum):
    STARTED = "started"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class ActivityTransition:
    camera_id: str
    activity_id: str
    generation: int
    sequence: int
    transition: ActivityTransitionKind
    state: ActivityState
    observed_at: str
    emitted_at: str
    source: str
    sources: tuple[str, ...]
    reason: str
    idle_after_seconds: float
    schema_version: int = 1

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "camera_id": self.camera_id,
            "activity_id": self.activity_id,
            "generation": self.generation,
            "sequence": self.sequence,
            "transition": self.transition.value,
            "state": self.state.value,
            "observed_at": self.observed_at,
            "emitted_at": self.emitted_at,
            "source": self.source,
            "sources": list(self.sources),
            "reason": self.reason,
            "idle_after_seconds": self.idle_after_seconds,
        }


@dataclass(slots=True)
class _CameraActivity:
    generation: int
    sequence: int = 0
    activity_id: str = ""
    observed_at: str = ""
    source: str = ""
    sources: set[str] = field(default_factory=set)
    timer: threading.Timer | None = None
    timer_token: int = 0

    @property
    def active(self) -> bool:
        return bool(self.activity_id)


class ActivityEventBus:
    """Convert motion observations into explicit, generation-safe activity edges.

    Motion qualification and incident persistence remain authoritative in their
    existing services. This bus only owns the short inactivity window used by
    integrations and user interfaces.
    """

    def __init__(
        self,
        publish: Callable[[ActivityTransition], None],
        *,
        idle_after_seconds: float = 10.0,
        utcnow: Callable[[], datetime] | None = None,
    ) -> None:
        self._publish = publish
        self.idle_after_seconds = max(0.1, float(idle_after_seconds))
        self._utcnow = utcnow or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._operations = threading.RLock()
        self._cameras: dict[str, _CameraActivity] = {}
        self._closed = False

    def start_generation(self, camera_id: str, generation: int) -> None:
        transition: ActivityTransition | None = None
        with self._operations:
            with self._lock:
                if self._closed:
                    raise RuntimeError("activity event bus is closed")
                current = self._cameras.get(camera_id)
                if current is not None and current.generation == generation:
                    return
                if current is not None:
                    transition = self._stop_locked(
                        camera_id,
                        current,
                        reason="generation_replaced",
                    )
                    sequence = current.sequence
                else:
                    sequence = 0
                self._cameras[camera_id] = _CameraActivity(
                    generation=int(generation),
                    sequence=sequence,
                )
            self._dispatch(transition)

    def observe(
        self,
        observation: MotionObserved,
        generation: int,
    ) -> ActivityTransition | None:
        transition: ActivityTransition | None = None
        with self._operations:
            with self._lock:
                if self._closed:
                    return None
                current = self._cameras.get(observation.camera_id)
                if current is None or current.generation != generation:
                    if current is not None:
                        self._cancel_timer_locked(current)
                        sequence = current.sequence
                    else:
                        sequence = 0
                    current = _CameraActivity(
                        generation=int(generation),
                        sequence=sequence,
                    )
                    self._cameras[observation.camera_id] = current
                current.observed_at = observation.timestamp
                current.source = observation.source
                current.sources.add(observation.source)
                if not current.active:
                    current.activity_id = uuid.uuid4().hex
                    current.sequence += 1
                    transition = self._transition(
                        observation.camera_id,
                        current,
                        ActivityTransitionKind.STARTED,
                        ActivityState.ACTIVE,
                        reason="motion_observed",
                    )
                self._schedule_timeout_locked(observation.camera_id, current)
            self._dispatch(transition)
        return transition

    def stop_generation(
        self,
        camera_id: str,
        generation: int,
        *,
        reason: str = "camera_stopped",
    ) -> ActivityTransition | None:
        transition: ActivityTransition | None = None
        with self._operations:
            with self._lock:
                current = self._cameras.get(camera_id)
                if current is None or current.generation != generation:
                    return None
                transition = self._stop_locked(camera_id, current, reason=reason)
            self._dispatch(transition)
        return transition

    def snapshot(self, camera_id: str) -> dict[str, object]:
        """Return current state for retained integrations without emitting an edge."""
        with self._lock:
            current = self._cameras.get(camera_id)
            if current is None:
                current = _CameraActivity(generation=0)
            transition = self._transition(
                camera_id,
                current,
                (
                    ActivityTransitionKind.STARTED
                    if current.active
                    else ActivityTransitionKind.STOPPED
                ),
                (
                    ActivityState.ACTIVE
                    if current.active
                    else ActivityState.INACTIVE
                ),
                reason="state_snapshot",
            )
        return transition.to_payload()

    def close(self) -> None:
        transitions: list[ActivityTransition] = []
        with self._operations:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                for camera_id, current in self._cameras.items():
                    transition = self._stop_locked(
                        camera_id,
                        current,
                        reason="service_stopped",
                    )
                    if transition is not None:
                        transitions.append(transition)
                self._cameras.clear()
            for transition in transitions:
                self._dispatch(transition)

    def _schedule_timeout_locked(
        self,
        camera_id: str,
        current: _CameraActivity,
    ) -> None:
        self._cancel_timer_locked(current)
        current.timer_token += 1
        token = current.timer_token
        timer = threading.Timer(
            self.idle_after_seconds,
            self._timeout,
            args=(camera_id, current.generation, token),
        )
        timer.daemon = True
        current.timer = timer
        timer.start()

    def _timeout(self, camera_id: str, generation: int, token: int) -> None:
        transition: ActivityTransition | None = None
        with self._operations:
            with self._lock:
                if self._closed:
                    return
                current = self._cameras.get(camera_id)
                if (
                    current is None
                    or current.generation != generation
                    or current.timer_token != token
                ):
                    return
                current.timer = None
                transition = self._stop_locked(
                    camera_id,
                    current,
                    reason="inactivity_timeout",
                )
            self._dispatch(transition)

    def _stop_locked(
        self,
        camera_id: str,
        current: _CameraActivity,
        *,
        reason: str,
    ) -> ActivityTransition | None:
        self._cancel_timer_locked(current)
        if not current.active:
            return None
        current.sequence += 1
        transition = self._transition(
            camera_id,
            current,
            ActivityTransitionKind.STOPPED,
            ActivityState.INACTIVE,
            reason=reason,
        )
        current.activity_id = ""
        current.observed_at = ""
        current.source = ""
        current.sources.clear()
        return transition

    @staticmethod
    def _cancel_timer_locked(current: _CameraActivity) -> None:
        timer = current.timer
        current.timer = None
        if timer is not None:
            timer.cancel()

    def _transition(
        self,
        camera_id: str,
        current: _CameraActivity,
        kind: ActivityTransitionKind,
        state: ActivityState,
        *,
        reason: str,
    ) -> ActivityTransition:
        return ActivityTransition(
            camera_id=camera_id,
            activity_id=current.activity_id,
            generation=current.generation,
            sequence=current.sequence,
            transition=kind,
            state=state,
            observed_at=current.observed_at,
            emitted_at=self._utcnow().isoformat(),
            source=current.source,
            sources=tuple(sorted(current.sources)),
            reason=reason,
            idle_after_seconds=self.idle_after_seconds,
        )

    def _dispatch(self, transition: ActivityTransition | None) -> None:
        if transition is None:
            return
        try:
            self._publish(transition)
        except Exception:
            LOGGER.exception(
                "activity transition callback failed for camera=%s transition=%s",
                transition.camera_id,
                transition.transition.value,
            )

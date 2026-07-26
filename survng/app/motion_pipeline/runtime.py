from __future__ import annotations

import logging
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, cast


T = TypeVar("T")
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MotionRuntimeState:
    """Mutable state owned by exactly one camera pipeline."""

    camera_id: str
    generation: int = 1
    stage_state: dict[str, Any] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def state_for(self, stage_id: str, factory: Callable[[], T]) -> T:
        with self._lock:
            if stage_id not in self.stage_state:
                self.stage_state[stage_id] = factory()
            return cast(T, self.stage_state[stage_id])

    def reset(self) -> None:
        with self._lock:
            states = tuple(self.stage_state.values())
            self.stage_state.clear()
            self.generation += 1
        self._close_states(states)

    def close(self) -> None:
        """Release runtime-owned resources without creating a new generation."""
        with self._lock:
            states = tuple(self.stage_state.values())
            self.stage_state.clear()
        self._close_states(states)

    @staticmethod
    def _close_states(states: tuple[Any, ...]) -> None:
        for state in states:
            close = getattr(state, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception:
                LOGGER.exception("motion runtime state cleanup failed")

    def clone(self) -> "MotionRuntimeState":
        """Copy stage state for a disposable replay without sharing mutable state."""
        with self._lock:
            copied: dict[str, Any] = {}
            for stage_id, state in self.stage_state.items():
                snapshot = getattr(state, "snapshot", None)
                clone = getattr(state, "clone", None)
                try:
                    if callable(snapshot):
                        copied[stage_id] = snapshot()
                    elif callable(clone):
                        copied[stage_id] = clone()
                    else:
                        copied[stage_id] = deepcopy(state)
                except Exception as error:
                    raise TypeError(
                        f"motion runtime state {stage_id!r} must implement snapshot() "
                        "when it owns native or non-copyable resources"
                    ) from error
            return MotionRuntimeState(
                camera_id=self.camera_id,
                generation=self.generation,
                stage_state=copied,
            )

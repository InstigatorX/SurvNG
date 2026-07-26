from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, cast


T = TypeVar("T")


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
            self.stage_state.clear()
            self.generation += 1

    def clone(self) -> "MotionRuntimeState":
        """Copy stage state for a disposable replay without sharing mutable state."""
        with self._lock:
            copied: dict[str, Any] = {}
            for stage_id, state in self.stage_state.items():
                clone = getattr(state, "clone", None)
                copied[stage_id] = clone() if callable(clone) else deepcopy(state)
            return MotionRuntimeState(
                camera_id=self.camera_id,
                generation=self.generation,
                stage_state=copied,
            )

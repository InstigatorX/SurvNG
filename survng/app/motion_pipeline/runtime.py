from __future__ import annotations

import threading
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


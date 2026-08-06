"""Fair, bounded scheduling for continuous visual-motion analysis."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Iterator


@dataclass(slots=True)
class _Waiter:
    camera_id: str
    sequence: int
    queued_at: float


class FairMotionAnalysisLimiter:
    """Bound concurrent analysis while giving overdue cameras their turn.

    A normal semaphore permits a busy camera to repeatedly reacquire a slot while
    another camera remains queued.  This scheduler instead selects the waiting
    camera least recently granted a slot (then FIFO), preserving the same fixed
    capacity without allowing starvation.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("motion analysis capacity must be positive")
        self.capacity = capacity
        self._condition = threading.Condition()
        self._active = 0
        self._sequence = 0
        self._pending: dict[str, _Waiter] = {}
        self._last_granted: dict[str, int] = {}
        self._grants = 0

    def _next_waiter(self) -> _Waiter | None:
        if not self._pending:
            return None
        return min(
            self._pending.values(),
            key=lambda waiter: (
                self._last_granted.get(waiter.camera_id, -1),
                waiter.sequence,
            ),
        )

    @contextmanager
    def acquire(
        self,
        camera_id: str,
        *,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[float | None]:
        """Acquire a fair slot, or yield ``None`` when the wait is cancelled."""
        started = time.monotonic()
        acquired = False
        with self._condition:
            self._sequence += 1
            waiter = _Waiter(camera_id, self._sequence, started)
            # A worker only has one queued unit at a time. Replacing a stale
            # request keeps the latest-frame queue bounded per camera.
            self._pending[camera_id] = waiter
            self._condition.notify_all()
            while self._active >= self.capacity or self._next_waiter() is not waiter:
                if self._pending.get(camera_id) is not waiter:
                    break
                if cancel_event is not None and cancel_event.is_set():
                    if self._pending.get(camera_id) is waiter:
                        self._pending.pop(camera_id, None)
                    self._condition.notify_all()
                    break
                self._condition.wait(timeout=0.1 if cancel_event is not None else None)
            else:
                self._pending.pop(camera_id, None)
                self._active += 1
                self._grants += 1
                self._last_granted[camera_id] = self._grants
                acquired = True
        if not acquired:
            yield None
            return
        try:
            yield max(0.0, time.monotonic() - started)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def status(self) -> dict[str, int]:
        with self._condition:
            return {
                "capacity": self.capacity,
                "active": self._active,
                "pending": len(self._pending),
            }

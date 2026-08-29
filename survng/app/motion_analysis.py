"""Fair, bounded scheduling for continuous visual-motion analysis."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Callable, Iterator


@dataclass(slots=True)
class _Waiter:
    camera_id: str
    sequence: int
    queued_at: float
    on_available: Callable[[], None] | None = None
    notified: bool = False


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

    def _available_callbacks_locked(self) -> list[Callable[[], None]]:
        """Wake only the fair head; acquisition advances the next waiter."""
        if self._active >= self.capacity:
            return []
        waiter = self._next_waiter()
        # Blocking acquire() waiters are awakened by the condition. Waking
        # only one nonblocking head prevents capacity-two callbacks from
        # executing out of order and stranding an already-notified waiter.
        if (
            waiter is None
            or waiter.on_available is None
            or waiter.notified
        ):
            return []
        waiter.notified = True
        return [waiter.on_available]

    @staticmethod
    def _notify_callbacks(callbacks: list[Callable[[], None]]) -> None:
        for callback in callbacks:
            try:
                callback()
            except Exception:
                # A wakeup is advisory; the owner also retries on its bounded
                # queue timeout and unconditionally cancels during teardown.
                continue

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
        callbacks: list[Callable[[], None]] = []
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
                callbacks = self._available_callbacks_locked()
        self._notify_callbacks(callbacks)
        if not acquired:
            yield None
            return
        try:
            yield max(0.0, time.monotonic() - started)
        finally:
            callbacks: list[Callable[[], None]]
            with self._condition:
                self._active -= 1
                self._condition.notify_all()
                callbacks = self._available_callbacks_locked()
            self._notify_callbacks(callbacks)

    @contextmanager
    def try_acquire(
        self,
        camera_id: str,
        *,
        on_available: Callable[[], None] | None = None,
    ) -> Iterator[float | None]:
        """Acquire without blocking while retaining a fair pending request.

        Callers can continue lightweight per-frame work and retry later. The
        first retry whose request is next in fair order receives the slot and
        observes the total time since that camera first became pending.
        """
        acquired = False
        callbacks: list[Callable[[], None]] = []
        with self._condition:
            waiter = self._pending.get(camera_id)
            if waiter is None:
                self._sequence += 1
                waiter = _Waiter(
                    camera_id,
                    self._sequence,
                    time.monotonic(),
                    on_available=on_available,
                )
                self._pending[camera_id] = waiter
            elif on_available is not None:
                waiter.on_available = on_available
            if self._active < self.capacity and self._next_waiter() is waiter:
                self._pending.pop(camera_id, None)
                self._active += 1
                self._grants += 1
                self._last_granted[camera_id] = self._grants
                acquired = True
                callbacks = self._available_callbacks_locked()
        self._notify_callbacks(callbacks)
        if not acquired:
            yield None
            return
        try:
            yield max(0.0, time.monotonic() - waiter.queued_at)
        finally:
            callbacks: list[Callable[[], None]]
            with self._condition:
                self._active -= 1
                self._condition.notify_all()
                callbacks = self._available_callbacks_locked()
            self._notify_callbacks(callbacks)

    def set_capacity(self, capacity: int) -> None:
        """Raise or lower the fleet-wide analysis ceiling without dropping work.

        Active cameras keep their current slot. A lower ceiling simply waits
        for those leases to end before granting additional waiters.
        """
        if capacity < 1:
            raise ValueError("motion analysis capacity must be positive")
        callbacks: list[Callable[[], None]]
        with self._condition:
            self.capacity = int(capacity)
            self._condition.notify_all()
            callbacks = self._available_callbacks_locked()
        self._notify_callbacks(callbacks)

    def cancel(self, camera_id: str) -> None:
        """Remove a camera's ungranted request during shutdown."""
        callbacks: list[Callable[[], None]]
        with self._condition:
            if self._pending.pop(camera_id, None) is not None:
                self._condition.notify_all()
            callbacks = self._available_callbacks_locked()
        self._notify_callbacks(callbacks)

    def status(self) -> dict[str, int]:
        with self._condition:
            return {
                "capacity": self.capacity,
                "active": self._active,
                "pending": len(self._pending),
            }

"""Short-lived leases that protect an application-manager generation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar


ManagerT = TypeVar("ManagerT")


class ManagerAccessCoordinator:
    """Allow bounded I/O outside the reload lock without retiring its owner.

    A lease is registered while the generation lock is held, closing the gap
    between selecting a manager and announcing its use. Reload holds that same
    generation lock while waiting, whereas lease release uses only this
    coordinator's condition, so completion cannot deadlock with cutover.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active: dict[int, int] = {}

    @contextmanager
    def lease(
        self,
        generation_lock: threading.RLock,
        get_manager: Callable[[], ManagerT],
    ) -> Iterator[ManagerT]:
        with generation_lock:
            manager = get_manager()
            identity = id(manager)
            with self._condition:
                self._active[identity] = self._active.get(identity, 0) + 1
        try:
            yield manager
        finally:
            with self._condition:
                remaining = self._active.get(identity, 0) - 1
                if remaining > 0:
                    self._active[identity] = remaining
                else:
                    self._active.pop(identity, None)
                    self._condition.notify_all()

    def wait_idle(self, manager: object, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        identity = id(manager)
        with self._condition:
            while self._active.get(identity, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def active_leases(self, manager: object) -> int:
        with self._condition:
            return self._active.get(id(manager), 0)

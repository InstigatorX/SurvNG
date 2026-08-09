"""Short-lived leases that protect an application-manager generation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, ParamSpec, TypeVar


ManagerT = TypeVar("ManagerT")
ReturnT = TypeVar("ReturnT")
Params = ParamSpec("Params")


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


@contextmanager
def manager_generation_lease(
    access: ManagerAccessCoordinator | None,
    generation_lock: threading.RLock | None,
    get_manager: Callable[[], ManagerT],
) -> Iterator[ManagerT]:
    """Protect one bounded use of the selected manager generation."""
    if access is not None:
        if generation_lock is None:
            raise RuntimeError("manager generation lease requires a generation lock")
        with access.lease(generation_lock, get_manager) as manager:
            yield manager
        return
    if generation_lock is not None:
        with generation_lock:
            yield get_manager()
        return
    yield get_manager()


def guard_manager_generation(
    access: ManagerAccessCoordinator | None,
    generation_lock: threading.RLock | None,
    get_manager: Callable[[], Any],
) -> Callable[[Callable[Params, ReturnT]], Callable[Params, ReturnT]]:
    """Decorate a bounded sync or async operation with a generation lease."""

    def decorate(operation: Callable[Params, ReturnT]) -> Callable[Params, ReturnT]:
        if iscoroutinefunction(operation):
            @wraps(operation)
            async def async_guarded(*args: Params.args, **kwargs: Params.kwargs) -> Any:
                if access is not None:
                    if generation_lock is None:
                        raise RuntimeError(
                            "manager generation guard requires a generation lock"
                        )
                    with access.lease(generation_lock, get_manager):
                        return await operation(*args, **kwargs)
                if generation_lock is not None:
                    with generation_lock:
                        return await operation(*args, **kwargs)
                return await operation(*args, **kwargs)

            return async_guarded

        @wraps(operation)
        def guarded(*args: Params.args, **kwargs: Params.kwargs) -> ReturnT:
            if access is not None:
                if generation_lock is None:
                    raise RuntimeError(
                        "manager generation guard requires a generation lock"
                    )
                with access.lease(generation_lock, get_manager):
                    return operation(*args, **kwargs)
            if generation_lock is not None:
                with generation_lock:
                    return operation(*args, **kwargs)
            return operation(*args, **kwargs)

        return guarded

    return decorate

"""An async manager guard registers its lease without blocking the event loop."""

from __future__ import annotations

import asyncio
import threading

import pytest

from survng.app.manager_access import ManagerAccessCoordinator, guard_manager_generation


def test_async_guard_waits_for_a_reload_without_blocking_the_loop() -> None:
    access = ManagerAccessCoordinator()
    # A plain Lock stands in for the reload lock: a timer thread releases it,
    # which an RLock would only allow its owner to do.
    reload_lock = threading.Lock()
    manager = object()

    @guard_manager_generation(access, reload_lock, lambda: manager)
    async def operation() -> str:
        assert access.active_leases(manager) == 1
        return "done"

    async def scenario() -> tuple[str, int]:
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        beat = asyncio.create_task(heartbeat())
        reload_lock.acquire()  # a reload in progress
        threading.Timer(0.2, reload_lock.release).start()
        try:
            return await operation(), ticks
        finally:
            beat.cancel()

    result, ticks = asyncio.run(scenario())

    assert result == "done"
    assert ticks >= 5  # the loop kept running while the lease waited
    assert access.active_leases(manager) == 0


def test_cancelled_registration_does_not_leak_a_lease() -> None:
    access = ManagerAccessCoordinator()
    reload_lock = threading.Lock()
    manager = object()

    @guard_manager_generation(access, reload_lock, lambda: manager)
    async def operation() -> None:
        return None

    async def scenario() -> None:
        reload_lock.acquire()
        task = asyncio.create_task(operation())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        reload_lock.release()
        await asyncio.sleep(0.2)  # let the worker thread finish registering

    asyncio.run(scenario())

    assert access.wait_idle(manager, 1.0)


def test_async_guard_refuses_to_hold_a_bare_lock_across_awaits() -> None:
    with pytest.raises(TypeError):
        @guard_manager_generation(None, threading.RLock(), lambda: object())
        async def operation() -> None:
            return None

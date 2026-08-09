from __future__ import annotations

import asyncio
import threading

from survng.app.manager_access import (
    ManagerAccessCoordinator,
    guard_manager_generation,
)


def test_generation_lease_releases_reload_lock_during_io_but_blocks_retirement() -> None:
    generation_lock = threading.RLock()
    access = ManagerAccessCoordinator()
    manager = object()
    entered = threading.Event()
    release = threading.Event()

    def request() -> None:
        with access.lease(generation_lock, lambda: manager) as selected:
            assert selected is manager
            entered.set()
            assert release.wait(1.0)

    thread = threading.Thread(target=request)
    thread.start()
    assert entered.wait(1.0)

    assert generation_lock.acquire(timeout=0.1)
    generation_lock.release()
    assert access.active_leases(manager) == 1
    assert not access.wait_idle(manager, 0.01)

    release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    assert access.wait_idle(manager, 0.1)


def test_lease_registration_is_atomic_with_manager_selection() -> None:
    generation_lock = threading.RLock()
    access = ManagerAccessCoordinator()
    first = object()
    second = object()
    active = {"manager": first}

    with access.lease(generation_lock, lambda: active["manager"]) as selected:
        with generation_lock:
            active["manager"] = second
        assert selected is first
        assert access.active_leases(first) == 1
        assert access.active_leases(second) == 0

    assert access.wait_idle(first, 0.1)


def test_guard_holds_generation_lease_without_holding_reload_lock() -> None:
    generation_lock = threading.RLock()
    access = ManagerAccessCoordinator()
    manager = object()

    @guard_manager_generation(access, generation_lock, lambda: manager)
    def operation() -> object:
        assert generation_lock.acquire(timeout=0.1)
        generation_lock.release()
        assert access.active_leases(manager) == 1
        return manager

    assert operation() is manager
    assert access.active_leases(manager) == 0


def test_async_guard_keeps_lease_across_await() -> None:
    generation_lock = threading.RLock()
    access = ManagerAccessCoordinator()
    manager = object()

    @guard_manager_generation(access, generation_lock, lambda: manager)
    async def operation() -> None:
        await asyncio.sleep(0)
        assert access.active_leases(manager) == 1

    asyncio.run(operation())
    assert access.active_leases(manager) == 0

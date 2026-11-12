"""The MCP live frame must not hold the process reload lock while it encodes."""

from __future__ import annotations

import threading

from survng.app.manager_access import ManagerAccessCoordinator
from survng.app.mcp_routes import McpDependencies, McpEndpoint


def test_live_frame_work_runs_outside_the_generation_lock() -> None:
    lock = threading.RLock()
    access = ManagerAccessCoordinator()
    manager = object()
    started = threading.Event()
    release = threading.Event()
    deps = McpDependencies(
        get_config=lambda: None,
        get_manager=lambda: manager,
        manager_lock=lock,
        intelligence=object(),
        application_stopping=threading.Event(),
        call_limiter=threading.BoundedSemaphore(1),
        manager_access=access,
    )
    endpoint = McpEndpoint(deps)

    def block(active: object) -> None:
        assert active is manager
        started.set()
        assert release.wait(2)

    worker = threading.Thread(target=lambda: endpoint._with_generation_lease(block))
    worker.start()
    assert started.wait(2)
    acquired = threading.Event()

    def take_lock() -> None:
        with lock:
            acquired.set()

    other = threading.Thread(target=take_lock)
    other.start()
    assert acquired.wait(2)
    assert access.active_leases(manager) == 1
    release.set()
    worker.join(2)
    other.join(2)
    assert access.wait_idle(manager, 1.0)
    assert not worker.is_alive()

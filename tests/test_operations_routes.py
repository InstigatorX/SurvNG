from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from survng.app.operations_routes import (
    OperationsRouteDependencies,
    create_operations_router,
)


def _endpoint(router, path: str, method: str):
    return next(
        route.endpoint
        for route in router.routes
        if route.path == path and method in route.methods
    )


def _router(request_server_restart: Mock):
    return create_operations_router(OperationsRouteDependencies(
        get_manager=Mock(),
        manager_lock=threading.RLock(),
        log_rows=lambda: (),
        storage_maintenance=Mock(),
        request_server_restart=request_server_restart,
        product_update=Mock(),
    ))


def test_restart_server_returns_scheduled_restart() -> None:
    request_restart = Mock(return_value={
        "ok": True,
        "status": "restart_scheduled",
        "instance_id": "old-instance",
    })

    result = _endpoint(
        _router(request_restart),
        "/api/system/restart",
        "POST",
    )()

    assert result["status"] == "restart_scheduled"
    request_restart.assert_called_once_with()


def test_restart_server_reports_restart_safety_conflict() -> None:
    request_restart = Mock(side_effect=RuntimeError("storage repair is active"))

    with pytest.raises(HTTPException) as raised:
        _endpoint(
            _router(request_restart),
            "/api/system/restart",
            "POST",
        )()

    assert raised.value.status_code == 409
    assert raised.value.detail == "storage repair is active"

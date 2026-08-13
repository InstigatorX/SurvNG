from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from survng.app import main


def test_server_restart_is_scheduled_after_response_without_blocking() -> None:
    scheduled_target = None

    class ImmediateThread:
        def __init__(self, *, target, **_kwargs) -> None:
            nonlocal scheduled_target
            scheduled_target = target

        def start(self) -> None:
            return None

    main.SERVER_RESTART_SCHEDULED = False
    main.APPLICATION_STOPPING.clear()
    with (
        patch.object(main, "_active_storage_tasks", return_value=[]),
        patch.object(main.threading, "Thread", ImmediateThread),
    ):
        result = main._request_server_restart()

    assert result == {
        "ok": True,
        "status": "restart_scheduled",
        "instance_id": main.PROCESS_INSTANCE_ID,
    }
    assert scheduled_target is not None
    assert main.SERVER_RESTART_SCHEDULED is True
    assert main.APPLICATION_STOPPING.is_set()
    main.SERVER_RESTART_SCHEDULED = False
    main.APPLICATION_STOPPING.clear()


def test_server_restart_refuses_active_storage_work() -> None:
    main.SERVER_RESTART_SCHEDULED = False
    main.APPLICATION_STOPPING.clear()
    with patch.object(main, "_active_storage_tasks", return_value=["media recording export"]):
        with pytest.raises(RuntimeError, match="media recording export"):
            main._request_server_restart()

    assert main.SERVER_RESTART_SCHEDULED is False
    assert not main.APPLICATION_STOPPING.is_set()


def test_ai_shutdown_drain_waits_for_registered_work() -> None:
    main._begin_ai_operation("test-drain")
    release = threading.Timer(0.02, main._end_ai_operation, args=("test-drain",))
    release.start()
    try:
        assert main._wait_for_ai_operations(1.0) == {}
    finally:
        release.join(timeout=1)
        main._end_ai_operation("test-drain")
    assert main._active_ai_operations() == {}


def test_ai_shutdown_drain_reports_work_that_exceeds_deadline() -> None:
    main._begin_ai_operation("test-timeout")
    try:
        assert main._wait_for_ai_operations(0.0) == {"test-timeout": 1}
    finally:
        main._end_ai_operation("test-timeout")
    assert main._active_ai_operations() == {}


def test_lifespan_starts_calibration_monitor_through_intelligence_owner() -> None:
    assert main._active_ai_operations() == {}
    async def exercise() -> None:
        async def monitor_forever() -> None:
            await asyncio.Event().wait()

        active_manager = SimpleNamespace(
            start_all=Mock(),
            stop_all=Mock(),
            detector=SimpleNamespace(stop_resource_tracker=Mock()),
        )
        exports = SimpleNamespace(start=Mock(), stop=Mock(return_value=True))
        monitor = AsyncMock(side_effect=monitor_forever)

        with (
            patch.object(main, "manager", active_manager),
            patch.object(main, "_record_process_lifecycle"),
            patch.object(main, "_start_face_observation_sync"),
            patch.object(
                main._recording_media_runtime,
                "_media_export_manager",
                return_value=exports,
            ),
            patch.object(
                main._recording_media_runtime,
                "_start_recording_prewarmer",
            ),
            patch.object(
                main._recording_media_runtime,
                "_stop_recording_prewarmer",
            ),
            patch.object(
                main._intelligence_route_bundle.service,
                "_calibration_followup_monitor",
                monitor,
            ),
            patch.object(main.STORAGE_MAINTENANCE, "stop", return_value=True),
        ):
            async with main.lifespan(main.app):
                await asyncio.sleep(0)

        monitor.assert_awaited_once_with()
        active_manager.start_all.assert_called_once_with()
        active_manager.stop_all.assert_called_once_with()
        exports.start.assert_called_once_with()
        exports.stop.assert_called_once_with(timeout=10.0)

    asyncio.run(exercise())

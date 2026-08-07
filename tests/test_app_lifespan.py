from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from survng.app import main


def test_lifespan_starts_calibration_monitor_through_intelligence_owner() -> None:
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

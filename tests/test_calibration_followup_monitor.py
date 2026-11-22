"""The calibration follow-up monitor survives failures and respects shutdown."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from survng.app.intelligence_routes import IntelligenceService


def _service(**overrides) -> IntelligenceService:
    events = Mock()
    events.calibration_change_sets.return_value = []
    events.calibration_runs.return_value = []
    manager = Mock(events=events)
    deps = SimpleNamespace(
        application_stopping=threading.Event(),
        manager_lock=threading.RLock(),
        get_manager=Mock(return_value=manager),
        get_config=Mock(),
        begin_ai_operation=Mock(),
        end_ai_operation=Mock(),
    )
    for name, value in overrides.items():
        setattr(deps, name, value)
    service = IntelligenceService.__new__(IntelligenceService)
    IntelligenceService.__init__(service, deps)  # type: ignore[arg-type]
    return service


def test_a_tick_started_during_shutdown_starts_nothing() -> None:
    service = _service()
    service.deps.application_stopping.set()

    service._calibration_followup_tick()

    service.deps.get_manager.assert_not_called()


def test_one_failed_tick_does_not_end_the_monitor(monkeypatch) -> None:
    service = _service()
    ticks = 0

    def tick() -> None:
        nonlocal ticks
        ticks += 1
        if ticks == 1:
            raise RuntimeError("database or disk is full")
        service.deps.application_stopping.set()

    async def no_wait(_seconds: float) -> None:
        return None

    service._calibration_followup_tick = tick  # type: ignore[method-assign]
    monkeypatch.setattr("survng.app.intelligence_routes.asyncio.sleep", no_wait)

    asyncio.run(service._calibration_followup_monitor())

    assert ticks == 2


def test_due_evaluation_is_registered_before_the_reload_lock_is_released() -> None:
    created = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    service = _service()
    events = service.deps.get_manager().events
    events.calibration_change_sets.return_value = [
        {"id": 3, "action": "apply", "status": "collecting",
         "created_at": created, "evaluation_hours": 24},
    ]
    service._schedule_system_review = Mock()  # type: ignore[method-assign]
    held_when_registered: list[bool] = []
    service.deps.begin_ai_operation = Mock(
        side_effect=lambda _operation: held_when_registered.append(
            service.deps.manager_lock._is_owned()
        )
    )
    service._run_calibration_evaluation = Mock()  # type: ignore[method-assign]

    service._calibration_followup_tick()

    events.update_calibration_change_set_status.assert_called_once_with(3, "reviewing")
    assert held_when_registered == [True]

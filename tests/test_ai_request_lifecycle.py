"""Blocking AI work retains admission and generation ownership after cancellation."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from survng.app.assistant import AssistantChatRequest
from survng.app.audit_ai import AuditAiError
from survng.app.config import AppConfig, AuditAiConfig
from survng.app.intelligence_routes import IntelligenceService


@pytest.mark.parametrize("operation", ["assistant", "motion_audit"])
def test_cancelled_request_retains_admission_until_provider_finishes(operation):
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    limiter = threading.BoundedSemaphore(1)
    config = AppConfig(audit_ai=AuditAiConfig(
        enabled=True, assistant_enabled=True, provider="gemini", api_key="test-key",
    ))
    manager = SimpleNamespace(
        storage_dir="unused", media_storage=None,
        events=SimpleNamespace(get_motion_audit=lambda _: {"camera_id": "gate"}),
    )
    deps = SimpleNamespace(
        manager_lock=threading.RLock(), application_stopping=threading.Event(),
        get_config=lambda: config,
        get_manager=lambda: manager, get_assistant_limiter=lambda: limiter,
        get_audit_ai_limiter=lambda: limiter, begin_ai_operation=Mock(),
        end_ai_operation=lambda _: completed.set(),
    )
    service = IntelligenceService(deps)
    service._assistant_catalog = Mock(return_value={})
    service._audit_ai_context = Mock(return_value={})

    def provider(*_args, **_kwargs):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test provider was not released")
        raise AuditAiError("injected provider failure")

    async def invoke():
        if operation == "assistant":
            return await service.assistant_chat(AssistantChatRequest(message="status"))
        return await service.motion_audit_ai_analyze(1)

    async def scenario():
        task = asyncio.create_task(invoke())
        try:
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(.01)
            assert entered.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not completed.is_set(), "request cancellation retired live AI work"
            with pytest.raises(HTTPException) as rejected:
                await invoke()
            assert rejected.value.status_code == 429
        finally:
            release.set()
            for _ in range(200):
                if completed.is_set():
                    break
                await asyncio.sleep(.01)
        assert completed.is_set()
        assert limiter.acquire(blocking=False)
        limiter.release()

    with (
        patch("survng.app.intelligence_routes.AssistantProvider.plan", side_effect=provider),
        patch("survng.app.intelligence_routes.AuditAiAdvisor.analyze", side_effect=provider),
        patch("survng.app.intelligence_routes.event_snapshot_path", return_value="unused.jpg"),
    ):
        asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["assistant", "motion_audit"])
def test_ai_work_is_not_admitted_after_shutdown_starts(operation):
    stopping = threading.Event()
    stopping.set()
    deps = SimpleNamespace(manager_lock=threading.RLock(), application_stopping=stopping)
    service = IntelligenceService(deps)
    async def scenario():
        with pytest.raises(HTTPException) as rejected:
            if operation == "assistant":
                await service.assistant_chat(AssistantChatRequest(message="status"))
            else:
                await service.motion_audit_ai_analyze(1)
        assert rejected.value.status_code == 503
    asyncio.run(scenario())

"""MCP endpoint for Grok Bot: bearer auth, scoped tools, and sanitized evidence."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from types import SimpleNamespace

import cv2
import httpx2 as httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from survng.app import main
from survng.app.config import ApiAuthConfig, ApiTokenConfig, AppConfig, WebAuthConfig, WebUserConfig
from survng.app.intelligence_routes import IntelligenceDependencies, IntelligenceService
from survng.app import mcp_routes
from survng.app.mcp_routes import MCP_PROTOCOL_VERSION, McpDependencies, create_mcp_router
from survng.app.security import (
    SESSION_COOKIE_NAME,
    encode_session,
    hash_api_token,
    hash_password,
)

SECRET = "bot-secret"
STREAM_URL = "rtsp://viewer:secret@camera.local/main"
STORAGE_PATH = "/var/lib/survng/recordings"


class EventStore:
    def __init__(self, event=None, incidents=None) -> None:
        self.event = event
        self.incidents = list(incidents or [])

    def get(self, event_id: int):
        if self.event is not None and int(self.event.get("id") or 0) == int(event_id):
            return self.event
        return None

    def list_scene_incidents(self, **_kwargs):
        return list(self.incidents)

    def calibration_runs(self, _limit, include_result=False):
        del include_result
        return []


def token_config(*scopes: str, enabled: bool = True, secret: str = SECRET) -> AppConfig:
    return AppConfig(api_auth=ApiAuthConfig(
        enabled=enabled,
        tokens=[ApiTokenConfig(
            id="bot",
            name="Grok Bot",
            token_hash=hash_api_token(secret),
            scopes=list(scopes),
        )],
    ))


def _health_status(_active):
    return {
        "cameras": {"total": 1, "online": 0},
        "detector": {"runtime": {}, "reid": {}},
        "storage": {"recordings": STORAGE_PATH},
        "mqtt": {"stream_url": STREAM_URL},
        "go2rtc": {},
    }


def build_client(
    tmp_path,
    config: AppConfig,
    *,
    call_limiter: threading.BoundedSemaphore | None = None,
    system_status=None,
    manager_lock: threading.RLock | None = None,
):
    storage = tmp_path / "media"
    storage.mkdir()
    frame = np.zeros((100, 2000, 3), dtype=np.uint8)
    frame[:, :, 2] = 255
    snapshot = storage / "snap.jpg"
    assert cv2.imwrite(str(snapshot), frame)
    incident = {
        "id": "incident-gate-1",
        "incident_id": "incident-gate-1",
        "camera_id": "gate",
        "camera_ids": ["gate"],
        "representative_event_id": 7,
        "start_at": "2026-09-26T20:00:00-04:00",
        "labels": ["person"],
        "zones": [STORAGE_PATH],
        "summary": STREAM_URL,
        "events": [{
            "id": 7,
            "snapshot_path": str(snapshot),
            "recording_path": str(storage / "clip.mp4"),
            "objects": [{"label": "person", "confidence": 0.9}],
        }],
        "scene_objects": [{"label": "person", "confidence": 0.9}],
    }
    manager = SimpleNamespace(
        storage_dir=storage,
        media_storage=None,
        events=EventStore({"id": 5, "snapshot_path": "snap.jpg"}, [incident]),
        started=[],
        statuses=lambda: [{
            "id": "gate",
            "running": False,
            "connected": False,
            "frame_fresh": False,
            "recording": False,
            "last_frame_age_seconds": 12,
            "last_error": STREAM_URL,
            "stream_url": STREAM_URL,
        }],
        recorder=SimpleNamespace(retention_status=lambda: {
            "state": "idle",
            "enabled": True,
            "plan": {},
            "last_run": {},
        }),
        incidents=SimpleNamespace(snapshot=lambda: [{
            "schema_version": 2,
            "incident_id": "incident-gate-new",
            "camera_id": "gate",
            "state": "new",
            "incident_path": "/incidents/incident-gate-new",
            "snapshot_url": "https://nvr.example/survng/api/events/8/snapshot.jpg",
            "stream_url": STREAM_URL,
        }, {
            "schema_version": 2,
            "incident_id": "incident-gate-1",
            "camera_id": "gate",
            "state": "complete",
            "incident_path": "/incidents/incident-gate-1",
            "snapshot_url": "https://nvr.example/survng/api/events/7/snapshot.jpg",
            "stream_url": STREAM_URL,
        }]),
        incident_notification_allowed=lambda _payload: True,
        incident_notification_payload=lambda payload: dict(payload),
    )

    def start_camera(camera_id: str) -> bool:
        manager.started.append(("start", camera_id))
        return camera_id == "gate"

    manager.start_camera = start_camera
    manager.stop_camera = lambda camera_id: False
    manager.set_recording = lambda camera_id, enabled: False
    manager.set_detection = lambda camera_id, enabled: False
    queries = SimpleNamespace(
        hydrate=lambda _manager, items: items,
        with_faces=lambda _manager, items: items,
        resolve_event=lambda _manager, _event_id: None,
    )
    lock = manager_lock or threading.RLock()
    service = IntelligenceService(IntelligenceDependencies(
        get_config=lambda: config,
        get_manager=lambda: manager,
        manager_lock=lock,
        get_audit_ai_limiter=lambda: None,
        get_assistant_limiter=lambda: None,
        application_stopping=threading.Event(),
        incident_queries=queries,
        system_telemetry=SimpleNamespace(system_status=system_status or _health_status),
        apply_config_update=lambda *_args, **_kwargs: None,
        begin_ai_operation=lambda *_args, **_kwargs: None,
        end_ai_operation=lambda *_args, **_kwargs: None,
        media_export_manager=lambda: None,
    ))
    app = FastAPI()
    app.include_router(create_mcp_router(McpDependencies(
        get_config=lambda: config,
        get_manager=lambda: manager,
        manager_lock=lock,
        intelligence=service,
        application_stopping=threading.Event(),
        call_limiter=call_limiter or threading.BoundedSemaphore(2),
    )))
    return TestClient(app), manager


def post_rpc(client: TestClient, method: str, params=None, *, token: str | None = SECRET, rpc_id=1):
    body = {"jsonrpc": "2.0", "method": method, "id": rpc_id}
    if params is not None:
        body["params"] = params
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return client.post("/api/mcp", json=body, headers=headers)


def tool_names(payload: dict) -> set[str]:
    return {item["name"] for item in payload["result"]["tools"]}


def tool_text(payload: dict) -> str:
    content = payload["result"]["content"]
    return "".join(item.get("text", "") for item in content if item.get("type") == "text")


READ_TOOLS = {
    "get_system_health",
    "get_camera_health",
    "explain_configuration",
    "inspect_incident",
    "search_incidents",
    "semantic_search_recordings",
    "summarize_recent_activity",
    "trace_across_cameras",
    "get_incident_image",
    "recent_incidents",
}
CONTROL_TOOLS = {"start_camera", "stop_camera", "set_recording", "set_detection"}


def test_initialize_advertises_the_protocol_version(tmp_path) -> None:
    client, _manager = build_client(tmp_path, token_config("read"))
    response = post_rpc(client, "initialize", {})
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert body["result"]["serverInfo"]["name"] == "survng"


@pytest.mark.parametrize("enabled,token", [
    (False, SECRET),
    (True, None),
    (True, "wrong-token"),
])
def test_mcp_requires_an_enabled_api_token(tmp_path, enabled, token) -> None:
    client, _manager = build_client(tmp_path, token_config("read", enabled=enabled))
    response = post_rpc(client, "tools/list", token=token)
    assert response.status_code == 401


def test_non_utf8_body_is_a_parse_error(tmp_path) -> None:
    client, _manager = build_client(tmp_path, token_config("read"))
    response = client.post(
        "/api/mcp",
        content=b"\xff",
        headers={
            "Authorization": f"Bearer {SECRET}",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["id"] is None
    assert body["error"]["code"] == -32700
    assert body["error"]["message"] == "parse error"


def test_initialize_rejects_a_body_over_the_cap_without_reading_the_rest(tmp_path) -> None:
    client, _manager = build_client(tmp_path, token_config("read"))
    response = client.post(
        "/api/mcp",
        content=b"{" + (b"x" * mcp_routes._MAX_BODY_BYTES),
        headers={
            "Authorization": f"Bearer {SECRET}",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "request is too large"

    pulled = 0

    async def receive() -> dict:
        nonlocal pulled
        pulled += 1
        if pulled == 1:
            return {
                "type": "http.request",
                "body": b"x" * (mcp_routes._MAX_BODY_BYTES + 1),
                "more_body": True,
            }
        return {
            "type": "http.request",
            "body": b"y" * 4096,
            "more_body": False,
        }

    request = Request({"type": "http", "method": "POST", "headers": []}, receive)
    with pytest.raises(mcp_routes.McpProtocolError) as exc:
        asyncio.run(mcp_routes._read_capped_body(request))
    assert exc.value.status_code == 400
    assert pulled == 1

    declared = Request(
        {
            "type": "http",
            "method": "POST",
            "headers": [(b"content-length", b"999999999")],
        },
        receive,
    )
    pulled = 0
    with pytest.raises(mcp_routes.McpProtocolError) as declared_error:
        asyncio.run(mcp_routes._read_capped_body(declared))
    assert declared_error.value.status_code == 400
    assert pulled == 0


def test_read_token_lists_investigation_tools_only(tmp_path) -> None:
    client, manager = build_client(tmp_path, token_config("read"))
    listed = tool_names(post_rpc(client, "tools/list").json())
    assert READ_TOOLS <= listed
    assert listed.isdisjoint(CONTROL_TOOLS)
    assert "create_media_export" not in listed
    assert "analyze_incident_visual" not in listed

    hidden = post_rpc(client, "tools/call", {"name": "start_camera", "arguments": {"camera_id": "gate"}})
    assert hidden.status_code == 200
    assert hidden.json()["error"]["code"] == -32602
    assert manager.started == []


def test_search_and_health_responses_omit_secrets_and_paths(tmp_path) -> None:
    client, _manager = build_client(tmp_path, token_config("read"))
    health = post_rpc(client, "tools/call", {"name": "get_system_health", "arguments": {}})
    search = post_rpc(client, "tools/call", {"name": "search_incidents", "arguments": {}})
    assert health.status_code == 200
    assert search.status_code == 200
    combined = health.text + search.text
    assert "secret" not in combined
    assert STREAM_URL not in combined
    assert STORAGE_PATH not in combined
    assert "[URL omitted]" in health.text
    assert "[filesystem path omitted]" in health.text
    assert "person" in search.text


def test_incident_image_returns_a_bounded_jpeg_or_not_found(tmp_path) -> None:
    client, _manager = build_client(tmp_path, token_config("read"))
    found = post_rpc(client, "tools/call", {"name": "get_incident_image", "arguments": {"event_id": 5}})
    assert found.status_code == 200
    body = found.json()["result"]
    assert body["isError"] is False
    image = next(item for item in body["content"] if item["type"] == "image")
    assert image["mimeType"] == "image/jpeg"
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(image["data"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None
    assert max(decoded.shape[:2]) <= 1280
    assert decoded.shape[1] == 1280

    missing = post_rpc(client, "tools/call", {"name": "get_incident_image", "arguments": {"event_id": 99}})
    assert missing.json()["result"]["isError"] is True
    assert "not found" in tool_text(missing.json())


def test_incident_image_refuses_oversized_frames_outside_the_manager_lock(tmp_path, monkeypatch) -> None:
    lock = threading.RLock()
    client, manager = build_client(tmp_path, token_config("read"), manager_lock=lock)
    decoded_outside_lock = threading.Event()
    original = mcp_routes._bounded_jpeg

    def bounded(path):
        assert lock.acquire(blocking=False)
        lock.release()
        decoded_outside_lock.set()
        return original(path)

    monkeypatch.setattr(mcp_routes, "_bounded_jpeg", bounded)
    found = post_rpc(client, "tools/call", {"name": "get_incident_image", "arguments": {"event_id": 5}})
    assert found.json()["result"]["isError"] is False
    assert decoded_outside_lock.is_set()

    snapshot = manager.storage_dir / "snap.jpg"
    snapshot.write_bytes(b"\xff\xd8" + (b"\x00" * 64) + b"\xff\xd9")
    monkeypatch.setattr(mcp_routes, "_MAX_SNAPSHOT_BYTES", 32)
    oversized = post_rpc(client, "tools/call", {"name": "get_incident_image", "arguments": {"event_id": 5}})
    assert oversized.json()["result"]["isError"] is True
    assert "unavailable" in tool_text(oversized.json())

    monkeypatch.setattr(mcp_routes, "_MAX_SNAPSHOT_BYTES", 12 * 1024 * 1024)
    snapshot.write_bytes(bytes([
        0xFF, 0xD8,
        0xFF, 0xC0, 0x00, 0x0B, 0x08,
        0x4E, 0x20,
        0x4E, 0x20,
        0x01, 0x01, 0x11, 0x00,
        0xFF, 0xD9,
    ]))
    huge = post_rpc(client, "tools/call", {"name": "get_incident_image", "arguments": {"event_id": 5}})
    assert huge.json()["result"]["isError"] is True
    assert "unavailable" in tool_text(huge.json())


def test_tool_text_drops_embedded_paths_urls_and_vectors(tmp_path) -> None:
    client, manager = build_client(tmp_path, token_config("read"))
    leaked_path = "/var/lib/survng/models/image.xml"
    manager.semantic_search = SimpleNamespace(
        search_text=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(
            f"semantic image model artifact is missing: {leaked_path}; open failed {STREAM_URL}"
        )),
    )
    manager.semantic_search_status = lambda: {
        "enabled": True,
        "state": "unavailable",
        "error": f"model directory does not exist: {leaked_path}",
    }
    manager.events.incidents[0]["events"][0]["faces"] = [{
        "name": "Ada",
        "embedding": [0.25] * 48,
    }]

    search = post_rpc(client, "tools/call", {"name": "search_incidents", "arguments": {}})
    semantic = post_rpc(client, "tools/call", {
        "name": "semantic_search_recordings",
        "arguments": {"query": "red jacket"},
    })
    combined = search.text + semantic.text
    assert search.status_code == 200
    assert semantic.status_code == 200
    assert leaked_path not in combined
    assert STREAM_URL not in combined
    assert "secret" not in combined
    assert "0.25" not in combined
    assert "Ada" in search.text
    assert "/recordings/search" in semantic.text
    assert "[filesystem path omitted]" in semantic.text
    assert "[URL omitted]" in semantic.text


def test_parallel_tool_calls_receive_429(tmp_path) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_status(_active):
        entered.set()
        assert release.wait(3)
        return _health_status(_active)

    client, _manager = build_client(
        tmp_path,
        token_config("read"),
        call_limiter=threading.BoundedSemaphore(1),
        system_status=blocking_status,
    )

    async def scenario():
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            async def call():
                return await http.post(
                    "/api/mcp",
                    headers={"Authorization": f"Bearer {SECRET}"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": "get_system_health", "arguments": {}},
                    },
                )

            first = asyncio.create_task(call())
            assert await asyncio.to_thread(entered.wait, 3)
            second = await call()
            release.set()
            first_response = await first
            return first_response, second

    first_response, second = asyncio.run(scenario())
    assert first_response.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"] == "2"
    assert second.json()["error"]["message"] == "SurvNG is busy; try again shortly"


def test_recent_incidents_keep_the_app_href_and_drop_urls(tmp_path) -> None:
    client, _manager = build_client(tmp_path, token_config("read"))
    response = post_rpc(client, "tools/call", {"name": "recent_incidents", "arguments": {"limit": 1}})
    payload = json.loads(tool_text(response.json()))
    assert len(payload["incidents"]) == 1
    incident = payload["incidents"][0]
    assert incident["href"] == "/incidents/incident-gate-new"
    assert incident["schema_version"] == 2
    assert "secret" not in response.text
    assert "nvr.example" not in response.text


def test_camera_control_token_can_start_a_camera(tmp_path) -> None:
    client, manager = build_client(tmp_path, token_config("camera:control"))
    listed = tool_names(post_rpc(client, "tools/list").json())
    assert "start_camera" in listed
    assert "get_system_health" not in listed
    assert "create_media_export" not in listed

    started = post_rpc(client, "tools/call", {"name": "start_camera", "arguments": {"camera_id": "gate"}})
    assert started.json()["result"]["isError"] is False
    assert json.loads(tool_text(started.json())) == {"ok": True, "camera_id": "gate"}
    assert manager.started == [("start", "gate")]

    missing = post_rpc(client, "tools/call", {"name": "start_camera", "arguments": {"camera_id": "missing"}})
    assert missing.json()["result"]["isError"] is True


def test_admin_token_can_request_a_media_export(tmp_path) -> None:
    client, _manager = build_client(tmp_path, token_config("admin"))
    listed = tool_names(post_rpc(client, "tools/list").json())
    assert "create_media_export" in listed
    assert "start_camera" in listed
    response = post_rpc(client, "tools/call", {
        "name": "create_media_export",
        "arguments": {
            "camera_id": "gate",
            "start_at": "2026-09-26T20:00:00-04:00",
            "end_at": "2026-09-26T20:01:00-04:00",
            "export_kind": "recording",
        },
    })
    assert response.status_code == 200
    assert response.json()["result"]["isError"] is False
    assert "media_export" in tool_text(response.json())


def _middleware_scope(path: str, headers: list[tuple[bytes, bytes]]) -> dict:
    return {
        "type": "http",
        "scheme": "http",
        "method": "POST",
        "path": path,
        "headers": [(b"host", b"survng.local"), *headers],
    }


async def _receive() -> dict:
    await asyncio.sleep(0)
    return {"type": "http.disconnect"}


def test_middleware_admits_any_bearer_scope_and_rejects_sessions() -> None:
    async def exercise(config_changes, headers, *, admitted: bool, path: str = "/api/mcp") -> None:
        called = False

        async def inner(_scope, _receive_message, _send) -> None:
            nonlocal called
            called = True

        messages: list[dict] = []

        async def collect(message: dict) -> None:
            messages.append(message)

        middleware = main.SecurityBoundaryMiddleware(inner)
        with config_changes:
            await middleware(_middleware_scope(path, headers), _receive, collect)
        assert called is admitted
        if not admitted:
            assert messages[0]["status"] == 401

    read_auth = ApiAuthConfig(enabled=True, tokens=[ApiTokenConfig(
        id="bot", name="Grok Bot", token_hash=hash_api_token(SECRET), scopes=["read"],
    )])
    control_auth = ApiAuthConfig(enabled=True, tokens=[ApiTokenConfig(
        id="bot", name="Grok Bot", token_hash=hash_api_token(SECRET), scopes=["camera:control"],
    )])
    bearer = [(b"authorization", f"Bearer {SECRET}".encode())]
    user = WebUserConfig(
        id="alex",
        username="alex",
        role="admin",
        password_hash=hash_password("correct-horse"),
    )
    session = encode_session(user.id, "a" * 64)
    web_auth = WebAuthConfig(enabled=True, session_key="a" * 64, users=[user])
    from unittest.mock import patch
    asyncio.run(exercise(
        patch.object(main.config, "api_auth", read_auth),
        bearer,
        admitted=True,
    ))
    asyncio.run(exercise(
        patch.object(main.config, "api_auth", control_auth),
        bearer,
        admitted=True,
    ))
    asyncio.run(exercise(
        patch.object(main.config, "api_auth", read_auth),
        [(b"authorization", b"Bearer wrong")],
        admitted=False,
    ))
    asyncio.run(exercise(
        patch.multiple(main.config, api_auth=read_auth, web_auth=web_auth),
        [(b"cookie", f"{SESSION_COOKIE_NAME}={session}".encode())],
        admitted=False,
    ))
    asyncio.run(exercise(
        patch.multiple(
            main.config,
            api_auth=ApiAuthConfig(enabled=False, tokens=read_auth.tokens),
            base_path="/survng",
        ),
        bearer,
        admitted=False,
        path="/survng/api/mcp",
    ))
    asyncio.run(exercise(
        patch.multiple(main.config, api_auth=read_auth, base_path="/survng"),
        bearer,
        admitted=True,
        path="/survng/api/mcp",
    ))

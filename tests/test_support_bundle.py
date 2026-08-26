from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from survng.app.security import required_api_scope
from survng.app.support_bundle import (
    SupportBundleDependencies,
    create_support_bundle_router,
)


class _Config:
    def model_dump(self, *, mode: str) -> dict:
        return {
            "api_token": "do-not-export",
            "storage_dir": "/srv/survng",
            "camera": {"main_stream_url": "rtsp://admin:secret@camera/stream"},
            "detection_enabled": True,
        }


def test_support_bundle_is_bounded_redacted_and_downloadable() -> None:
    telemetry = SimpleNamespace(
        operational_event_history=lambda **_: [{"summary": "camera recovered"}],
        diagnostic_sessions=lambda **_: [{"id": "abc", "scope": "system"}],
    )
    manager = SimpleNamespace(
        telemetry=telemetry,
        statuses=lambda: [{"id": "gate", "connected": True, "snapshot_path": "/private.jpg"}],
    )
    app = __import__("fastapi").FastAPI()
    app.include_router(create_support_bundle_router(SupportBundleDependencies(
        get_manager=lambda: manager,
        get_config=lambda: _Config(),
        system_status=lambda _: {"version": "1.0", "storage_path": "/private"},
        log_rows=lambda: [{"level": "ERROR", "message": "authorization: Bearer abc"}],
    )))

    response = TestClient(app).get("/api/support-bundle")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    encoded = json.dumps(body)
    assert body["manifest"]["schema_version"] == 1
    assert body["manifest"]["privacy"]["media_included"] is False
    assert "do-not-export" not in encoded
    assert "secret@" not in encoded
    assert "/srv/survng" not in encoded
    assert "/private.jpg" not in encoded
    assert "Bearer abc" not in encoded
    assert "rtsp://" not in encoded


def test_support_bundle_collection_failure_is_reported_without_failing_download() -> None:
    app = __import__("fastapi").FastAPI()
    app.include_router(create_support_bundle_router(SupportBundleDependencies(
        get_manager=lambda: (_ for _ in ()).throw(RuntimeError("secret path /home/user")),
        get_config=lambda: (_ for _ in ()).throw(RuntimeError("config failed")),
        system_status=lambda _: {},
        log_rows=lambda: (),
    )))
    response = TestClient(app).get("/api/support-bundle")
    assert response.status_code == 200
    assert response.json()["runtime"]["collection_error"] == "manager"
    assert "/home/user" not in response.text


def test_support_bundle_requires_admin_scope() -> None:
    assert required_api_scope("GET", "/api/support-bundle") == "admin"

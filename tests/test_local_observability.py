from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from survng import ctl
from survng.app.config import AppConfig
from survng.app.local_observability import (
    LocalObservabilityServer,
    MAX_RECENT_LOG_BYTES,
    MAX_RECENT_LOG_ROWS,
    build_runtime_status,
    request_runtime_status,
)


def _manager(secret: str = "must-not-leak") -> SimpleNamespace:
    return SimpleNamespace(
        statuses=Mock(return_value=[{
            "id": "gate",
            "name": "Gate",
            "running": True,
            "connected": True,
            "capture_connectivity": "healthy",
            "last_frame_age_seconds": 0.2,
            "recording": True,
            "recording_enabled": True,
            "detection_enabled": True,
            "onvif_connected": True,
            "last_error": secret,
            "stream_url": f"rtsp://admin:{secret}@camera/live",
            "object_tracking": {
                "active": True,
                "running": True,
                "capacity_requests": 7,
                "capacity_waits": 2,
                "capacity_timeouts": 1,
                "capacity_wait_seconds_last": 0.5,
                "private_key": secret,
            },
        }]),
        detector_status=Mock(return_value={
            "ready": True,
            "configured_device": "GPU",
            "runtime": {
                "queue_depth": 1,
                "total_inferences": 30,
                "failed_inferences": 2,
                "last_error": secret,
            },
            "isolation": {
                "configured_workers": 2,
                "alive_workers": 2,
                "pending_requests": 1,
            },
            "model_path": f"/models/{secret}",
        }),
        recorder=SimpleNamespace(retention_status=Mock(return_value={
            "state": "idle",
            "last_plan_at": "2026-08-29T12:00:00+00:00",
            "plan": {
                "storage": {
                    "total_bytes": 1000,
                    "used_bytes": 400,
                    "free_bytes": 600,
                    "free_percent": 60.0,
                },
                "path": f"/storage/{secret}",
            },
        })),
        inference=SimpleNamespace(tracking_limiter=SimpleNamespace(status=Mock(
            return_value={
                "active": 1,
                "baseline": 3,
                "burst_limit": 5,
                "burst_enabled": True,
                "burst_admissions": 4,
                "burst_denials": 1,
            }
        ))),
    )


def test_runtime_status_is_effective_and_strictly_allowlisted() -> None:
    secret = "secret-token-password"
    config = AppConfig(
        detector={
            "tracking": {
                "max_active_cameras": 3,
                "burst_max_active_cameras": 5,
                "capacity_wait_seconds": 8,
            }
        },
        cameras=[{
            "id": "gate",
            "name": "Gate",
            "stream_url": f"rtsp://admin:{secret}@camera/live",
        }],
    )

    payload = build_runtime_status(
        config,
        _manager(secret),
        instance_id="instance-one",
        uptime_seconds=12.34,
        stopping=False,
    )

    assert payload["tracking"]["settings"]["max_active_cameras"] == 3
    assert payload["tracking"]["settings"]["burst_max_active_cameras"] == 5
    assert payload["tracking"]["settings"]["capacity_wait_seconds"] == 8
    assert payload["tracking"]["capacity"]["active"] == 1
    assert payload["tracking"]["activity_since_restart"] == {
        "requests": 7,
        "waits": 2,
        "timeouts": 1,
    }
    assert payload["detector"]["ready"] is True
    assert payload["storage"]["free_bytes"] == 600
    assert payload["cameras"][0]["connectivity"] == "healthy"
    encoded = json.dumps(payload)
    assert secret not in encoded
    assert "rtsp://" not in encoded
    assert "stream_url" not in encoded
    assert "private_key" not in encoded


def test_runtime_status_log_tail_is_bounded_allowlisted_and_redacted() -> None:
    secret = "observer-secret-value"
    log_rows = [
        {
            "time": f"2026-08-29T12:00:{index:02d}+00:00",
            "level": "ERROR",
            "logger": "survng.camera",
            "message": (
                f"camera failed password={secret} "
                f"at rtsp://admin:{secret}@camera-{index}/live "
                f"see https://example.test/private/{secret}"
            ),
            "exception": f"traceback leaked {secret}",
            "structured_extra": {"api_token": secret},
        }
        for index in range(MAX_RECENT_LOG_ROWS + 25)
    ]

    payload = build_runtime_status(
        AppConfig(),
        _manager(secret),
        instance_id="instance-one",
        uptime_seconds=12.34,
        stopping=False,
        log_rows=log_rows,
    )

    recent = payload["recent_logs"]
    assert len(recent["entries"]) <= MAX_RECENT_LOG_ROWS
    assert recent["serialized_bytes"] <= MAX_RECENT_LOG_BYTES
    assert recent["truncated"] is True
    assert set(recent["entries"][0]) == {
        "timestamp",
        "level",
        "logger",
        "message",
    }
    encoded = json.dumps(recent)
    assert secret not in encoded
    assert "rtsp://" not in encoded
    assert "https://" not in encoded
    assert "[redacted-url]" in encoded
    assert "traceback" not in encoded
    assert "structured_extra" not in encoded


def test_owner_only_socket_serves_status_and_is_removed(tmp_path: Path) -> None:
    async def exercise() -> None:
        socket_path = tmp_path / "runtime" / "observability.sock"
        server = LocalObservabilityServer(
            lambda: {"schema_version": 1, "tracking": {"capacity": {"active": 2}}},
            socket_path,
        )
        await server.start()
        try:
            assert stat.S_IMODE(socket_path.parent.stat().st_mode) == 0o700
            assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
            payload = await asyncio.to_thread(request_runtime_status, socket_path)
            assert payload["tracking"]["capacity"]["active"] == 2
        finally:
            await server.stop()
        assert not socket_path.exists()

    asyncio.run(exercise())


def test_cli_status_prints_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        ctl,
        "request_runtime_status",
        lambda _path, timeout: {"schema_version": 1, "process": {"stopping": False}},
    )

    assert ctl.main(["status", "--socket", "/tmp/test.sock", "--compact"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": 1,
        "process": {"stopping": False},
    }


def test_server_refuses_insecure_parent_directory(tmp_path: Path) -> None:
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    server = LocalObservabilityServer(lambda: {}, parent / "status.sock")

    async def exercise() -> None:
        try:
            await server.start()
        except PermissionError as error:
            assert "mode 0700" in str(error)
        else:
            raise AssertionError("insecure socket parent should be rejected")

    asyncio.run(exercise())

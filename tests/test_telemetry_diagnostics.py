from datetime import datetime, timezone

import pytest

from survng.app.telemetry_diagnostics import DiagnosticTelemetryController
from survng.app.telemetry_store import TelemetryStore


def test_manual_camera_session_flushes_prebuffer_and_scopes_future_samples(tmp_path) -> None:
    store = TelemetryStore(tmp_path)
    controller = DiagnosticTelemetryController(store)
    sampled_at = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    statuses = [
        {"id": "gate", "connected": True, "capture_stats": {"live": {"fps": 10}}},
        {"id": "foyer", "connected": True, "capture_stats": {"live": {"fps": 20}}},
    ]
    controller.observe(statuses, detector_runtime={}, now_monotonic=10, sampled_at=sampled_at)
    session = controller.start(scope="camera", camera_id="gate", duration_seconds=900)
    exported = controller.export(session["id"])

    assert exported is not None
    assert list(exported["samples"][0]["payload"]["cameras"]) == ["gate"]


def test_diagnostic_scope_and_duration_are_strict(tmp_path) -> None:
    controller = DiagnosticTelemetryController(TelemetryStore(tmp_path))

    with pytest.raises(ValueError, match="scope"):
        controller.start(scope="everything", duration_seconds=900)
    with pytest.raises(ValueError, match="duration"):
        controller.start(scope="system", duration_seconds=120)
    with pytest.raises(ValueError, match="camera id"):
        controller.start(scope="camera", duration_seconds=900)


def test_sustained_camera_outage_creates_one_automatic_session(tmp_path) -> None:
    store = TelemetryStore(tmp_path)
    controller = DiagnosticTelemetryController(store)
    started = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    status = [{"id": "gate", "expected_enabled": True, "connected": False}]
    for index in range(5):
        controller.observe(
            status,
            detector_runtime={},
            now_monotonic=10 + index * 5,
            sampled_at=started,
        )

    sessions = store.diagnostic_sessions(active_only=True, now=started)
    assert len(sessions) == 1
    assert sessions[0]["trigger_kind"] == "camera_unavailable"
    events = store.operational_event_history(hours=1, now=started)
    assert [event["kind"] for event in events] == ["camera_unavailable"]

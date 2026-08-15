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

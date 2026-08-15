from datetime import datetime, timedelta, timezone

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


def test_camera_outages_are_not_reported_during_startup_admission(tmp_path) -> None:
    store = TelemetryStore(tmp_path)
    controller = DiagnosticTelemetryController(store)
    started = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    status = [{"id": "gate", "expected_enabled": True, "connected": False}]
    for index in range(6):
        controller.observe(
            status,
            detector_runtime={},
            camera_startup_status={"complete": False},
            now_monotonic=10 + index * 5,
            sampled_at=started,
        )

    assert not store.diagnostic_sessions(active_only=True, now=started)
    assert not store.operational_event_history(hours=1, now=started)

    for index in range(3):
        controller.observe(
            status,
            detector_runtime={},
            camera_startup_status={"complete": True},
            now_monotonic=40 + index * 5,
            sampled_at=started,
        )
    assert len(store.diagnostic_sessions(active_only=True, now=started)) == 1


def test_degraded_startup_camera_gets_recovery_grace_before_outage(tmp_path) -> None:
    store = TelemetryStore(tmp_path)
    controller = DiagnosticTelemetryController(store)
    started = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    status = [{"id": "gate", "expected_enabled": True, "connected": False}]
    startup = {
        "complete": True,
        "cameras": {
            "gate": {
                "phase": "degraded",
                "completed_at": started.isoformat(),
            }
        },
    }
    for index in range(12):
        controller.observe(
            status,
            detector_runtime={},
            camera_startup_status=startup,
            now_monotonic=10 + index * 5,
            sampled_at=started + timedelta(seconds=index * 5),
        )
    assert not store.operational_event_history(hours=1, now=started + timedelta(seconds=60))

    for index in range(3):
        controller.observe(
            status,
            detector_runtime={},
            camera_startup_status=startup,
            now_monotonic=100 + index * 5,
            sampled_at=started + timedelta(seconds=90 + index * 5),
        )
    assert len(store.diagnostic_sessions(active_only=True, now=started + timedelta(seconds=100))) == 1


def test_sustained_detector_backlog_uses_available_runtime_signal(tmp_path) -> None:
    store = TelemetryStore(tmp_path)
    controller = DiagnosticTelemetryController(store)
    started = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    for index in range(3):
        controller.observe(
            [],
            detector_runtime={"queue_depth": 1},
            now_monotonic=10 + index * 5,
            sampled_at=started,
        )

    events = store.operational_event_history(hours=1, now=started)
    assert [event["kind"] for event in events] == ["detector_backlog"]


def test_diagnostic_payload_redacts_credentials_and_normalizes_nonfinite_values(
    tmp_path,
) -> None:
    store = TelemetryStore(tmp_path)
    controller = DiagnosticTelemetryController(store)
    started = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    controller.observe(
        [],
        detector_runtime={
            "last_error": "rtsp://admin:secret@camera/live token=abc",
            "average_inference_ms": float("nan"),
        },
        now_monotonic=10,
        sampled_at=started,
    )
    session = controller.start(
        scope="detector", duration_seconds=900, started_at=started
    )

    exported = controller.export(session["id"])
    assert exported is not None
    detector = exported["samples"][0]["payload"]["detector_runtime"]
    assert "secret" not in detector["last_error"]
    assert "token=abc" not in detector["last_error"]
    assert detector["average_inference_ms"] is None

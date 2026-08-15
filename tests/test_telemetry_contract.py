from survng.app.telemetry_contract import (
    CAMERA_METRICS,
    DIAGNOSTIC_DURATIONS_SECONDS,
    DIAGNOSTIC_SCOPES,
    SYSTEM_METRICS,
    MetricKind,
    TelemetryRetentionPolicy,
)


def test_operational_metric_names_are_unique_and_typed() -> None:
    metrics = SYSTEM_METRICS + CAMERA_METRICS
    names = [metric.name for metric in metrics]

    assert len(names) == len(set(names))
    assert all(metric.kind in MetricKind for metric in metrics)
    assert all(metric.unit for metric in metrics)


def test_default_retention_is_bounded_for_production() -> None:
    policy = TelemetryRetentionPolicy()

    assert policy.raw_days == 2
    assert policy.quarter_hour_days == 30
    assert policy.hourly_days == 365
    assert policy.operational_events_days == 90
    assert policy.diagnostic_retention_days == 7
    assert policy.operational_budget_bytes < policy.diagnostic_budget_bytes


def test_diagnostics_are_scoped_and_expiring() -> None:
    assert DIAGNOSTIC_SCOPES == {"system", "detector", "storage", "camera"}
    assert DIAGNOSTIC_DURATIONS_SECONDS == (900, 3600, 21600, 86400)


def test_legacy_runtime_telemetry_exists_only_in_versioned_migration() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert "runtime_telemetry_samples" not in (root / "survng/app/events.py").read_text()
    assert "system_lifecycle_events" not in (root / "survng/app/events.py").read_text()
    assert "record_runtime_telemetry" not in (
        root / "survng/app/runtime_monitor.py"
    ).read_text()

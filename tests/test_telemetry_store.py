from datetime import datetime, timedelta, timezone

from survng.app.telemetry_contract import TelemetryRetentionPolicy
from survng.app.telemetry_store import (
    CameraTelemetryBucket,
    SystemTelemetryBucket,
    TelemetryStore,
)


def test_store_persists_typed_system_and_camera_buckets(tmp_path) -> None:
    store = TelemetryStore(tmp_path)
    sampled_at = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)

    store.write_buckets(
        SystemTelemetryBucket(
            sampled_at=sampled_at,
            cpu_load_percent=12.5,
            application_rss_bytes=1234,
        ),
        [
            CameraTelemetryBucket(
                sampled_at=sampled_at,
                camera_id="gate",
                available=1.0,
                live_fps=9.8,
                ema_frames_sampled=120,
            )
        ],
    )

    assert store.system_history(
        since=sampled_at - timedelta(minutes=1), resolution_minutes=1
    )[0]["application_rss_bytes"] == 1234
    camera = store.camera_history(
        since=sampled_at - timedelta(minutes=1),
        resolution_minutes=1,
        camera_id="gate",
    )[0]
    assert camera["live_fps"] == 9.8
    assert camera["ema_frames_sampled"] == 120


def test_store_is_independent_and_retention_is_resolution_aware(tmp_path) -> None:
    policy = TelemetryRetentionPolicy(raw_days=2, quarter_hour_days=30, hourly_days=365)
    store = TelemetryStore(tmp_path, retention=policy)
    now = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    old = now - timedelta(days=3)

    for resolution in (1, 15, 60):
        store.write_buckets(
            SystemTelemetryBucket(sampled_at=old, resolution_minutes=resolution),
            [CameraTelemetryBucket(sampled_at=old, camera_id="gate", resolution_minutes=resolution)],
        )
    store.enforce_retention(now=now)

    assert not store.system_history(
        since=now - timedelta(days=10), resolution_minutes=1
    )
    assert store.system_history(
        since=now - timedelta(days=10), resolution_minutes=15
    )
    assert store.system_history(
        since=now - timedelta(days=10), resolution_minutes=60
    )
    assert store.path.name == "telemetry.sqlite3"


def test_operational_event_details_are_compact_and_database_is_measurable(tmp_path) -> None:
    store = TelemetryStore(tmp_path)
    store.record_operational_event(
        occurred_at=datetime.now(timezone.utc),
        kind="camera_unavailable",
        scope="camera",
        camera_id="gate",
        summary="Gate stream unavailable",
        details={"attempt": 2},
    )

    assert store.database_bytes() > 0


def test_rollups_average_gauges_and_sum_interval_counters(tmp_path) -> None:
    store = TelemetryStore(tmp_path)
    start = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    for minute, fps, interruptions in ((0, 10.0, 1), (1, 20.0, 2)):
        sampled_at = start + timedelta(minutes=minute)
        store.write_buckets(
            SystemTelemetryBucket(sampled_at=sampled_at, cpu_load_percent=fps),
            [
                CameraTelemetryBucket(
                    sampled_at=sampled_at,
                    camera_id="gate",
                    available=1.0,
                    live_fps=fps,
                    capture_interruptions=interruptions,
                )
            ],
        )
    store.refresh_rollups(sampled_at=start + timedelta(minutes=1))

    system = store.system_history(since=start, resolution_minutes=15)[0]
    camera = store.camera_history(
        since=start, resolution_minutes=15, camera_id="gate"
    )[0]
    assert system["cpu_load_percent"] == 15.0
    assert camera["live_fps"] == 15.0
    assert camera["capture_interruptions"] == 3


def test_operational_history_combines_camera_and_system_metrics(tmp_path) -> None:
    store = TelemetryStore(tmp_path)
    sampled_at = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    store.write_buckets(
        SystemTelemetryBucket(
            sampled_at=sampled_at,
            cpu_load_percent=25.0,
            inference_ms=18.0,
        ),
        [
            CameraTelemetryBucket(
                sampled_at=sampled_at,
                camera_id="gate",
                available=1.0,
                ema_frames_sampled=98,
                ema_frames_superseded=2,
                ema_credible_episodes=2,
                object_checks_admitted=1,
                object_checks_completed=1,
            )
        ],
    )

    row = store.operational_history(
        hours=2, bucket_minutes=1, camera_id="gate", now=sampled_at
    )[0]
    assert row["camera_availability_percent"] == 100.0
    assert row["analysis_coverage_percent"] == 98.0
    assert row["cpu_load_percent"] == 25.0
    assert row["ema_credible_episodes"] == 2
    assert store.sample_times(hours=2, now=sampled_at) == [sampled_at.isoformat()]


def test_diagnostic_sessions_expire_and_export_bounded_samples(tmp_path) -> None:
    store = TelemetryStore(tmp_path)
    started = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    session = store.create_diagnostic_session(
        scope="camera", camera_id="gate", duration_seconds=900, started_at=started
    )
    store.write_diagnostic_samples(
        [session["id"]], sampled_at=started, payload={"cameras": {"gate": {"fps": 10}}}
    )

    exported = store.diagnostic_export(session["id"])
    assert exported is not None
    assert exported["samples"][0]["payload"]["cameras"]["gate"]["fps"] == 10
    assert store.diagnostic_sessions(active_only=True, now=started)

    store.enforce_retention(now=started + timedelta(minutes=16))
    assert store.diagnostic_export(session["id"]) is None


def test_diagnostic_payload_budget_removes_oldest_samples(tmp_path) -> None:
    store = TelemetryStore(
        tmp_path,
        retention=TelemetryRetentionPolicy(diagnostic_budget_bytes=40),
    )
    started = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    session = store.create_diagnostic_session(
        scope="system", duration_seconds=900, started_at=started
    )
    for second in range(3):
        store.write_diagnostic_samples(
            [session["id"]],
            sampled_at=started + timedelta(seconds=second),
            payload={"detail": "x" * 30},
        )

    assert store.enforce_diagnostic_budget() > 0
    exported = store.diagnostic_export(session["id"])
    assert exported is not None
    assert sum(len(str(row["payload"])) for row in exported["samples"]) <= 40


def test_operational_events_coalesce_within_window(tmp_path) -> None:
    store = TelemetryStore(tmp_path)
    now = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    first = store.record_or_coalesce_operational_event(
        occurred_at=now,
        kind="camera_unavailable",
        scope="camera",
        camera_id="gate",
        summary="Gate unavailable",
    )
    second = store.record_or_coalesce_operational_event(
        occurred_at=now + timedelta(minutes=1),
        kind="camera_unavailable",
        scope="camera",
        camera_id="gate",
        summary="Gate still unavailable",
    )

    assert first["id"] == second["id"]
    assert second == {"id": first["id"], "count": 2, "coalesced": True}
    assert store.operational_event_history(hours=2, now=now + timedelta(minutes=1))[0]["count"] == 2

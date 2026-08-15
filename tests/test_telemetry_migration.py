import base64
import json
import sqlite3
import zlib
from datetime import datetime, timedelta, timezone

from survng.app.telemetry_migration import migrate_legacy_runtime_telemetry
from survng.app.telemetry_store import TelemetryStore


def _encoded(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return "zlib:" + base64.b64encode(zlib.compress(raw)).decode()


def test_legacy_migration_is_complete_idempotent_and_drops_source(tmp_path) -> None:
    event_db = tmp_path / "survng.sqlite3"
    with sqlite3.connect(event_db) as conn:
        conn.execute(
            "create table runtime_telemetry_samples (sampled_at text primary key,payload_json text not null)"
        )
        conn.execute(
            "create table system_lifecycle_events ("
            "id integer primary key,instance_id text,kind text,occurred_at text,details_json text)"
        )
        start = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
        for minute, failures, sampled in ((0, 1, 100), (1, 3, 125)):
            payload = {
                "system_runtime": {"cpu_load_percent": 20 + minute},
                "process_memory": {"rss_bytes": 1000 + minute},
                "worker_memory": {"total_rss_bytes": 2000 + minute},
                "cameras": {
                    "gate": {
                        "enabled": True,
                        "connected": True,
                        "frame_age_seconds": 0.1,
                        "capture": {"live": {"fps": 10, "read_failures": failures}, "main": {}},
                        "analysis_frames_dropped": 0,
                        "analysis_runtime": {"frames_sampled": sampled},
                        "event_runtime": {},
                    }
                },
            }
            conn.execute(
                "insert into runtime_telemetry_samples values (?,?)",
                ((start + timedelta(minutes=minute)).isoformat(), _encoded(payload)),
            )
        conn.execute(
            "insert into system_lifecycle_events "
            "(instance_id,kind,occurred_at,details_json) values (?,?,?,?)",
            ("instance-a", "startup_ready", start.isoformat(), '{"cameras":13}'),
        )
    store = TelemetryStore(tmp_path)

    result = migrate_legacy_runtime_telemetry(event_db, store, batch_size=1)

    assert result == {"migrated": 2, "complete": True}
    history = store.operational_history(
        hours=2,
        bucket_minutes=1,
        camera_id="gate",
        now=start + timedelta(minutes=1),
    )
    assert history[-1]["capture_interruptions"] == 2
    assert history[-1]["analysis_frames_sampled"] == 25
    assert store.lifecycle_events(hours=1, now=start)[0]["details"] == {"cameras": 13}
    with sqlite3.connect(event_db) as conn:
        assert conn.execute(
            "select 1 from sqlite_master where name='runtime_telemetry_samples'"
        ).fetchone() is None
        assert conn.execute(
            "select 1 from sqlite_master where name='system_lifecycle_events'"
        ).fetchone() is None
    assert migrate_legacy_runtime_telemetry(event_db, store) == {"migrated": 0, "complete": True}


def test_legacy_migration_skips_corrupt_and_malformed_payloads(tmp_path) -> None:
    event_db = tmp_path / "survng.sqlite3"
    sampled_at = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    with sqlite3.connect(event_db) as conn:
        conn.execute(
            "create table runtime_telemetry_samples (sampled_at text primary key,payload_json text not null)"
        )
        conn.execute(
            "insert into runtime_telemetry_samples values (?,?)",
            (sampled_at.isoformat(), "zlib:not-base64"),
        )
        conn.execute(
            "insert into runtime_telemetry_samples values (?,?)",
            (
                (sampled_at + timedelta(minutes=1)).isoformat(),
                json.dumps(
                    {
                        "process_memory": {"rss_bytes": "invalid"},
                        "cameras": ["not", "a", "mapping"],
                    }
                ),
            ),
        )

    result = migrate_legacy_runtime_telemetry(event_db, TelemetryStore(tmp_path))

    assert result == {"migrated": 1, "complete": True}
    with sqlite3.connect(event_db) as conn:
        assert conn.execute(
            "select 1 from sqlite_master where name='runtime_telemetry_samples'"
        ).fetchone() is None

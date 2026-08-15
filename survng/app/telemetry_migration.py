"""One-time conversion of legacy EventStore telemetry into typed buckets."""

from __future__ import annotations

import base64
import json
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .telemetry_store import (
    CameraTelemetryBucket,
    SystemTelemetryBucket,
    TelemetryStore,
)


MIGRATION_KEY = "legacy_event_telemetry_migrated_v2"


def _decode(value: object) -> dict[str, Any]:
    text = str(value or "{}")
    if text.startswith("zlib:"):
        text = zlib.decompress(base64.b64decode(text[5:])).decode()
    parsed = json.loads(text)
    return parsed if isinstance(parsed, dict) else {}


def _delta(current: int, previous: int | None) -> int:
    if previous is None:
        return 0
    return max(0, current - previous) if current >= previous else max(0, current)


def _finite(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def migrate_legacy_runtime_telemetry(
    event_database: Path,
    store: TelemetryStore,
    *,
    batch_size: int = 250,
) -> dict[str, int | bool]:
    """Convert all legacy rows, then drop the legacy table after success."""
    if store.metadata_value(MIGRATION_KEY) == "1":
        return {"migrated": 0, "complete": True}
    source = sqlite3.connect(Path(event_database), timeout=10.0)
    source.row_factory = sqlite3.Row
    try:
        tables = {
            str(row[0])
            for row in source.execute(
                "select name from sqlite_master where type='table' and name in "
                "('runtime_telemetry_samples','system_lifecycle_events')"
            )
        }
        if not tables:
            store.set_metadata_value(MIGRATION_KEY, "1")
            return {"migrated": 0, "complete": True}
        previous: dict[str, dict[str, int]] = {}
        migrated = 0
        last_sampled_at = ""
        while "runtime_telemetry_samples" in tables:
            rows = source.execute(
                "select sampled_at,payload_json from runtime_telemetry_samples "
                "where sampled_at>? order by sampled_at limit ?",
                (last_sampled_at, max(1, int(batch_size))),
            ).fetchall()
            if not rows:
                break
            systems: list[SystemTelemetryBucket] = []
            cameras: list[CameraTelemetryBucket] = []
            for row in rows:
                last_sampled_at = str(row["sampled_at"])
                try:
                    sampled_at = datetime.fromisoformat(last_sampled_at.replace("Z", "+00:00"))
                    if sampled_at.tzinfo is None:
                        sampled_at = sampled_at.replace(tzinfo=timezone.utc)
                    payload = _decode(row["payload_json"])
                except (TypeError, ValueError, json.JSONDecodeError, zlib.error):
                    continue
                process = dict(payload.get("process_memory") or {})
                workers = dict(payload.get("worker_memory") or {})
                runtime = dict(payload.get("system_runtime") or {})
                systems.append(
                    SystemTelemetryBucket(
                        sampled_at=sampled_at,
                        cpu_load_percent=_finite(runtime.get("cpu_load_percent")),
                        memory_used_percent=_finite(runtime.get("memory_used_percent")),
                        application_rss_bytes=int(process.get("rss_bytes") or 0),
                        worker_rss_bytes=int(workers.get("total_rss_bytes") or 0),
                        inference_ms=_finite(runtime.get("inference_ms")),
                    )
                )
                for camera_id, item in dict(payload.get("cameras") or {}).items():
                    if not isinstance(item, dict):
                        continue
                    capture = dict(item.get("capture") or {})
                    live = dict(capture.get("live") or {})
                    main = dict(capture.get("main") or {})
                    analysis = dict(item.get("analysis_runtime") or {})
                    event_runtime = dict(item.get("event_runtime") or {})
                    decisions = dict(
                        (event_runtime.get("episode") or {}).get("decision_counts") or {}
                    )
                    current = {
                        "capture_interruptions": int(live.get("read_failures") or 0)
                        + int(live.get("open_failures") or 0)
                        + int(main.get("read_failures") or 0)
                        + int(main.get("open_failures") or 0),
                        "ema_frames_sampled": int(analysis.get("frames_sampled") or 0),
                        "ema_frames_superseded": int(item.get("analysis_frames_dropped") or 0),
                        "ema_credible_episodes": int(decisions.get("request_reserved") or 0),
                        "object_checks_admitted": int(decisions.get("request_admitted") or 0),
                        "object_checks_completed": int(decisions.get("request_completed") or 0),
                        "object_check_failures": int(decisions.get("detector_failed") or 0),
                    }
                    old = previous.get(str(camera_id), {})
                    deltas = {key: _delta(value, old.get(key)) for key, value in current.items()}
                    previous[str(camera_id)] = current
                    frame_age = item.get("frame_age_seconds")
                    fresh = frame_age is None or float(frame_age) <= 5.0
                    enabled = bool(item.get("enabled", True))
                    cameras.append(
                        CameraTelemetryBucket(
                            sampled_at=sampled_at,
                            camera_id=str(camera_id),
                            expected=float(enabled),
                            available=float(not enabled or (bool(item.get("connected")) and fresh)),
                            live_fps=float(live.get("fps") or 0.0),
                            main_fps=float(main.get("fps") or 0.0),
                            **deltas,
                        )
                    )
                migrated += 1
            store.write_bucket_batch(systems, cameras)
        if "runtime_telemetry_samples" in tables:
            store.rebuild_rollups()
            source.execute("drop table runtime_telemetry_samples")
        if "system_lifecycle_events" in tables:
            lifecycle_rows = source.execute(
                "select instance_id,kind,occurred_at,details_json "
                "from system_lifecycle_events order by occurred_at,id"
            ).fetchall()
            store.import_lifecycle_events(dict(row) for row in lifecycle_rows)
            source.execute("drop table system_lifecycle_events")
        source.commit()
        store.set_metadata_value(MIGRATION_KEY, "1")
        return {"migrated": migrated, "complete": True}
    finally:
        source.close()

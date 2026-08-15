"""One-time conversion of legacy EventStore telemetry into typed buckets."""

from __future__ import annotations

import base64
import binascii
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


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


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


def _integer(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0


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
                except (
                    TypeError,
                    ValueError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    zlib.error,
                    binascii.Error,
                ):
                    continue
                process = _mapping(payload.get("process_memory"))
                workers = _mapping(payload.get("worker_memory"))
                runtime = _mapping(payload.get("system_runtime"))
                systems.append(
                    SystemTelemetryBucket(
                        sampled_at=sampled_at,
                        cpu_load_percent=_finite(runtime.get("cpu_load_percent")),
                        memory_used_percent=_finite(runtime.get("memory_used_percent")),
                        application_rss_bytes=_integer(process.get("rss_bytes")),
                        worker_rss_bytes=_integer(workers.get("total_rss_bytes")),
                        inference_ms=_finite(runtime.get("inference_ms")),
                    )
                )
                for camera_id, item in _mapping(payload.get("cameras")).items():
                    if not isinstance(item, dict):
                        continue
                    capture = _mapping(item.get("capture"))
                    live = _mapping(capture.get("live"))
                    main = _mapping(capture.get("main"))
                    analysis = _mapping(item.get("analysis_runtime"))
                    event_runtime = _mapping(item.get("event_runtime"))
                    decisions = _mapping(_mapping(event_runtime.get("episode")).get("decision_counts"))
                    current = {
                        "capture_interruptions": _integer(live.get("read_failures"))
                        + _integer(live.get("open_failures"))
                        + _integer(main.get("read_failures"))
                        + _integer(main.get("open_failures")),
                        "ema_frames_sampled": _integer(analysis.get("frames_sampled")),
                        "ema_frames_superseded": _integer(item.get("analysis_frames_dropped")),
                        "ema_credible_episodes": _integer(decisions.get("request_reserved")),
                        "object_checks_admitted": _integer(decisions.get("request_admitted")),
                        "object_checks_completed": _integer(decisions.get("request_completed")),
                        "object_check_failures": _integer(decisions.get("detector_failed")),
                    }
                    old = previous.get(str(camera_id), {})
                    deltas = {key: _delta(value, old.get(key)) for key, value in current.items()}
                    previous[str(camera_id)] = current
                    raw_frame_age = item.get("frame_age_seconds")
                    frame_age = _finite(raw_frame_age)
                    fresh = raw_frame_age is None or (
                        frame_age is not None and frame_age <= 5.0
                    )
                    enabled = bool(item.get("enabled", True))
                    cameras.append(
                        CameraTelemetryBucket(
                            sampled_at=sampled_at,
                            camera_id=str(camera_id),
                            expected=float(enabled),
                            available=float(enabled and bool(item.get("connected")) and fresh),
                            live_fps=_finite(live.get("fps")) or 0.0,
                            main_fps=_finite(main.get("fps")) or 0.0,
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

"""Compact, isolated persistence for operational and diagnostic telemetry."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .telemetry_contract import TelemetryRetentionPolicy


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class SystemTelemetryBucket:
    sampled_at: datetime
    resolution_minutes: int = 1
    cpu_load_percent: float | None = None
    memory_used_percent: float | None = None
    application_rss_bytes: int = 0
    worker_rss_bytes: int = 0
    inference_ms: float | None = None
    gpu_utilization_percent: float | None = None
    detector_requests: int = 0
    detector_failures: int = 0
    detector_capacity_delays: int = 0
    database_write_contention: int = 0


@dataclass(frozen=True, slots=True)
class CameraTelemetryBucket:
    sampled_at: datetime
    camera_id: str
    resolution_minutes: int = 1
    available: float = 0.0
    live_fps: float = 0.0
    main_fps: float = 0.0
    capture_interruptions: int = 0
    ema_frames_sampled: int = 0
    ema_frames_superseded: int = 0
    ema_credible_episodes: int = 0
    object_checks_admitted: int = 0
    object_checks_completed: int = 0
    object_check_failures: int = 0
    tracking_requested: int = 0
    tracking_completed: int = 0
    tracking_delayed: int = 0
    tracking_skipped: int = 0
    incidents_created: int = 0


SYSTEM_COLUMNS = tuple(
    field
    for field in SystemTelemetryBucket.__dataclass_fields__
    if field not in {"sampled_at", "resolution_minutes"}
)
CAMERA_COLUMNS = tuple(
    field
    for field in CameraTelemetryBucket.__dataclass_fields__
    if field not in {"sampled_at", "camera_id", "resolution_minutes"}
)


class TelemetryStore:
    """Own the telemetry database; it never shares EventStore's writer lock."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        database_dir: Path,
        *,
        retention: TelemetryRetentionPolicy | None = None,
    ) -> None:
        self.path = Path(database_dir) / "telemetry.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retention = retention or TelemetryRetentionPolicy()
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=2.0)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma journal_mode=wal")
        conn.execute("pragma synchronous=normal")
        conn.execute("pragma busy_timeout=2000")
        return conn

    def _initialize(self) -> None:
        system_fields = ",\n".join(
            f"{name} {'integer' if name.endswith(('_bytes', 'requests', 'failures', 'delays', 'contention')) else 'real'}"
            for name in SYSTEM_COLUMNS
        )
        camera_fields = ",\n".join(
            f"{name} {'integer' if name not in {'available', 'live_fps', 'main_fps'} else 'real'} not null default 0"
            for name in CAMERA_COLUMNS
        )
        with self._lock, self._connect() as conn:
            conn.executescript(
                f"""
                create table if not exists telemetry_metadata (
                    key text primary key,
                    value text not null
                );
                create table if not exists system_metric_buckets (
                    sampled_at text not null,
                    resolution_minutes integer not null,
                    {system_fields},
                    primary key (resolution_minutes, sampled_at)
                ) without rowid;
                create table if not exists camera_metric_buckets (
                    sampled_at text not null,
                    camera_id text not null,
                    resolution_minutes integer not null,
                    {camera_fields},
                    primary key (resolution_minutes, camera_id, sampled_at)
                ) without rowid;
                create table if not exists operational_events (
                    id integer primary key,
                    occurred_at text not null,
                    kind text not null,
                    scope text not null default 'system',
                    camera_id text not null default '',
                    summary text not null,
                    count integer not null default 1,
                    details_json text not null default '{{}}'
                );
                create index if not exists operational_events_time
                    on operational_events (occurred_at);
                create table if not exists diagnostic_sessions (
                    id text primary key,
                    scope text not null,
                    camera_id text not null default '',
                    started_at text not null,
                    expires_at text not null,
                    stopped_at text,
                    trigger_kind text not null default 'manual'
                );
                create table if not exists diagnostic_samples (
                    session_id text not null references diagnostic_sessions(id) on delete cascade,
                    sampled_at text not null,
                    payload_json text not null,
                    primary key (session_id, sampled_at)
                ) without rowid;
                """
            )
            conn.execute(
                "insert or replace into telemetry_metadata (key, value) values ('schema_version', ?)",
                (str(self.SCHEMA_VERSION),),
            )

    @staticmethod
    def _system_values(sample: SystemTelemetryBucket) -> tuple[Any, ...]:
        values = asdict(sample)
        return (
            _utc(sample.sampled_at).isoformat(),
            max(1, int(sample.resolution_minutes)),
            *(values[name] for name in SYSTEM_COLUMNS),
        )

    @staticmethod
    def _camera_values(sample: CameraTelemetryBucket) -> tuple[Any, ...]:
        values = asdict(sample)
        return (
            _utc(sample.sampled_at).isoformat(),
            str(sample.camera_id),
            max(1, int(sample.resolution_minutes)),
            *(values[name] for name in CAMERA_COLUMNS),
        )

    def write_buckets(
        self,
        system: SystemTelemetryBucket,
        cameras: Iterable[CameraTelemetryBucket],
    ) -> None:
        system_names = "sampled_at,resolution_minutes," + ",".join(SYSTEM_COLUMNS)
        camera_names = "sampled_at,camera_id,resolution_minutes," + ",".join(CAMERA_COLUMNS)
        system_placeholders = ",".join("?" for _ in range(2 + len(SYSTEM_COLUMNS)))
        camera_placeholders = ",".join("?" for _ in range(3 + len(CAMERA_COLUMNS)))
        camera_values = [self._camera_values(sample) for sample in cameras]
        with self._lock, self._connect() as conn:
            conn.execute(
                f"insert or replace into system_metric_buckets ({system_names}) values ({system_placeholders})",
                self._system_values(system),
            )
            if camera_values:
                conn.executemany(
                    f"insert or replace into camera_metric_buckets ({camera_names}) values ({camera_placeholders})",
                    camera_values,
                )

    @staticmethod
    def _bucket_start(value: datetime, resolution_minutes: int) -> datetime:
        seconds = max(1, int(resolution_minutes)) * 60
        epoch = int(_utc(value).timestamp())
        return datetime.fromtimestamp((epoch // seconds) * seconds, timezone.utc)

    def refresh_rollups(self, *, sampled_at: datetime) -> None:
        """Refresh the current 15-minute and hourly buckets from minute rows."""
        for resolution in (15, 60):
            start = self._bucket_start(sampled_at, resolution)
            end = start + timedelta(minutes=resolution)
            self._refresh_system_rollup(start=start, end=end, resolution=resolution)
            self._refresh_camera_rollup(start=start, end=end, resolution=resolution)

    def _refresh_system_rollup(
        self, *, start: datetime, end: datetime, resolution: int
    ) -> None:
        gauge_columns = (
            "cpu_load_percent",
            "memory_used_percent",
            "application_rss_bytes",
            "worker_rss_bytes",
            "inference_ms",
            "gpu_utilization_percent",
        )
        counter_columns = tuple(name for name in SYSTEM_COLUMNS if name not in gauge_columns)
        expressions = [f"avg({name}) as {name}" for name in gauge_columns]
        expressions.extend(f"sum({name}) as {name}" for name in counter_columns)
        with self._connect() as conn:
            row = conn.execute(
                "select " + ",".join(expressions) + " from system_metric_buckets "
                "where resolution_minutes=1 and sampled_at>=? and sampled_at<?",
                (start.isoformat(), end.isoformat()),
            ).fetchone()
        if row is None or all(row[name] is None for name in SYSTEM_COLUMNS):
            return
        values = {
            name: (0 if row[name] is None and name in counter_columns else row[name])
            for name in SYSTEM_COLUMNS
        }
        self.write_buckets(
            SystemTelemetryBucket(
                sampled_at=start,
                resolution_minutes=resolution,
                **values,
            ),
            [],
        )

    def _refresh_camera_rollup(
        self, *, start: datetime, end: datetime, resolution: int
    ) -> None:
        gauge_columns = ("available", "live_fps", "main_fps")
        counter_columns = tuple(name for name in CAMERA_COLUMNS if name not in gauge_columns)
        expressions = [f"avg({name}) as {name}" for name in gauge_columns]
        expressions.extend(f"sum({name}) as {name}" for name in counter_columns)
        with self._connect() as conn:
            rows = conn.execute(
                "select camera_id," + ",".join(expressions) + " from camera_metric_buckets "
                "where resolution_minutes=1 and sampled_at>=? and sampled_at<? group by camera_id",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        cameras = [
            CameraTelemetryBucket(
                sampled_at=start,
                camera_id=str(row["camera_id"]),
                resolution_minutes=resolution,
                **{
                    name: (0 if row[name] is None and name in counter_columns else row[name])
                    for name in CAMERA_COLUMNS
                },
            )
            for row in rows
        ]
        if not cameras:
            return
        # Preserve the system rollup written above while replacing camera rows.
        with self._lock, self._connect() as conn:
            names = "sampled_at,camera_id,resolution_minutes," + ",".join(CAMERA_COLUMNS)
            placeholders = ",".join("?" for _ in range(3 + len(CAMERA_COLUMNS)))
            conn.executemany(
                f"insert or replace into camera_metric_buckets ({names}) values ({placeholders})",
                [self._camera_values(sample) for sample in cameras],
            )

    def system_history(
        self, *, since: datetime, resolution_minutes: int
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from system_metric_buckets where resolution_minutes = ? "
                "and sampled_at >= ? order by sampled_at",
                (max(1, int(resolution_minutes)), _utc(since).isoformat()),
            ).fetchall()
        return [dict(row) for row in rows]

    def camera_history(
        self,
        *,
        since: datetime,
        resolution_minutes: int,
        camera_id: str = "",
    ) -> list[dict[str, Any]]:
        sql = (
            "select * from camera_metric_buckets where resolution_minutes = ? "
            "and sampled_at >= ?"
        )
        parameters: list[Any] = [max(1, int(resolution_minutes)), _utc(since).isoformat()]
        if camera_id:
            sql += " and camera_id = ?"
            parameters.append(camera_id)
        sql += " order by sampled_at, camera_id"
        with self._connect() as conn:
            rows = conn.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]

    def record_operational_event(
        self,
        *,
        occurred_at: datetime,
        kind: str,
        summary: str,
        scope: str = "system",
        camera_id: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "insert into operational_events "
                "(occurred_at,kind,scope,camera_id,summary,details_json) values (?,?,?,?,?,?)",
                (
                    _utc(occurred_at).isoformat(),
                    kind,
                    scope,
                    camera_id,
                    summary,
                    json.dumps(dict(details or {}), separators=(",", ":")),
                ),
            )

    def enforce_retention(self, *, now: datetime | None = None) -> None:
        current = _utc(now or datetime.now(timezone.utc))
        cutoffs = {
            1: current - timedelta(days=self.retention.raw_days),
            15: current - timedelta(days=self.retention.quarter_hour_days),
            60: current - timedelta(days=self.retention.hourly_days),
        }
        with self._lock, self._connect() as conn:
            for resolution, cutoff in cutoffs.items():
                conn.execute(
                    "delete from system_metric_buckets where resolution_minutes = ? and sampled_at < ?",
                    (resolution, cutoff.isoformat()),
                )
                conn.execute(
                    "delete from camera_metric_buckets where resolution_minutes = ? and sampled_at < ?",
                    (resolution, cutoff.isoformat()),
                )
            conn.execute(
                "delete from operational_events where occurred_at < ?",
                (
                    current
                    .__sub__(timedelta(days=self.retention.operational_events_days))
                    .isoformat(),
                ),
            )
            conn.execute(
                "delete from diagnostic_samples where session_id in "
                "(select id from diagnostic_sessions where expires_at < ?)",
                (current.isoformat(),),
            )
            conn.execute(
                "delete from diagnostic_sessions where expires_at < ?",
                (current.isoformat(),),
            )

    def database_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            try:
                total += (Path(f"{self.path}{suffix}")).stat().st_size
            except OSError:
                continue
        return total

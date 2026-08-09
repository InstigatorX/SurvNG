from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .incident_utils import event_snapshot_path, portable_media_path


class EventStore:
    COMPACT_COLUMNS = (
        "id, camera_id, kind, snapshot_path, recording_path, objects_json, created_at"
    )
    TRACKING_COMPARISON_HISTORY_PER_CAMERA = 100
    TRACKING_COMPARISON_VERDICTS = {
        "survng_hybrid",
        "ultralytics_botsort",
        "ultralytics_deepocsort",
        "ultralytics_fasttrack",
        "inconclusive",
    }

    def __init__(self, storage_dir: Path, database_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir
        self.db_path = (database_dir or storage_dir) / "survng.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 10000")
        conn.execute("pragma foreign_keys = on")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("pragma journal_mode = wal")
            conn.execute("pragma synchronous = normal")
            conn.execute(
                """
                create table if not exists events (
                    id integer primary key autoincrement,
                    camera_id text not null,
                    kind text not null,
                    topic text,
                    message text,
                    snapshot_path text,
                    recording_path text,
                    objects_json text not null default '[]',
                    created_at text not null
                )
                """
            )
            conn.execute(
                "create index if not exists idx_events_created_at on events(created_at desc)"
            )
            conn.execute(
                "create index if not exists idx_events_camera_created_at on events(camera_id, created_at desc)"
            )
            conn.execute(
                "create table if not exists survng_metadata (key text primary key, value text not null)"
            )
            conn.execute(
                """
                create table if not exists runtime_telemetry_samples (
                    sampled_at text primary key,
                    payload_json text not null
                )
                """
            )
            conn.execute(
                "create index if not exists idx_runtime_telemetry_sampled_at on runtime_telemetry_samples(sampled_at)"
            )
            conn.execute(
                """
                create table if not exists system_lifecycle_events (
                    id integer primary key autoincrement,
                    instance_id text not null,
                    kind text not null,
                    occurred_at text not null,
                    details_json text not null default '{}'
                )
                """
            )
            conn.execute(
                "create index if not exists idx_system_lifecycle_occurred_at "
                "on system_lifecycle_events(occurred_at)"
            )
            conn.execute(
                """
                create table if not exists motion_audits (
                    id integer primary key autoincrement,
                    event_id integer,
                    related_event_id integer,
                    decision_id text,
                    camera_id text not null,
                    snapshot_path text not null default '',
                    created_at text not null,
                    mode text not null,
                    sensitivity text not null,
                    score real not null,
                    threshold real not null,
                    reason text not null,
                    object_detected integer,
                    trigger_count integer not null default 1,
                    features_json text not null default '{}',
                    category text not null default 'qualification'
                )
                """
            )
            columns = {
                str(row["name"])
                for row in conn.execute("pragma table_info(motion_audits)").fetchall()
            }
            if "decision_id" not in columns:
                conn.execute("alter table motion_audits add column decision_id text")
            if "related_event_id" not in columns:
                conn.execute("alter table motion_audits add column related_event_id integer")
            if "category" not in columns:
                conn.execute(
                    "alter table motion_audits add column category text not null default 'qualification'"
                )
            conn.execute(
                "create index if not exists idx_motion_audits_created_at on motion_audits(created_at desc, id desc)"
            )
            conn.execute(
                "create index if not exists idx_motion_audits_camera_created_at on motion_audits(camera_id, created_at desc)"
            )
            conn.execute(
                "create index if not exists idx_motion_audits_category_created_at on motion_audits(category, created_at desc, id desc)"
            )
            conn.execute(
                "create index if not exists idx_motion_audits_related_event on motion_audits(related_event_id, created_at) where related_event_id is not null"
            )

            conn.execute(
                "create unique index if not exists idx_motion_audits_event on motion_audits(event_id) where event_id is not null"
            )
            conn.execute(
                "create unique index if not exists idx_motion_audits_decision on motion_audits(decision_id) where decision_id is not null and decision_id != ''"
            )
            conn.execute(
                """
                create table if not exists motion_ai_reviews (
                    id integer primary key autoincrement,
                    camera_id text not null,
                    status text not null,
                    audits_considered integer not null default 0,
                    images_available integer not null default 0,
                    analyzed integer not null default 0,
                    failed integer not null default 0,
                    result_json text not null default '{}',
                    error text not null default '',
                    created_at text not null,
                    updated_at text not null
                )
                """
            )
            conn.execute(
                "create index if not exists idx_motion_ai_reviews_camera_created on motion_ai_reviews(camera_id, created_at desc, id desc)"
            )
            conn.execute(
                """
                create table if not exists camera_intelligence_evaluations (
                    id integer primary key autoincrement,
                    camera_id text not null,
                    baseline_review_id integer not null unique,
                    followup_review_id integer,
                    status text not null default 'collecting',
                    evaluation_hours real not null default 24,
                    applied_changes_json text not null default '[]',
                    baseline_result_json text not null default '{}',
                    followup_result_json text not null default '{}',
                    comparison_json text not null default '{}',
                    error text not null default '',
                    applied_at text not null,
                    updated_at text not null,
                    completed_at text
                )
                """
            )
            conn.execute(
                "create index if not exists idx_camera_intelligence_evaluations_camera_applied on camera_intelligence_evaluations(camera_id, applied_at desc, id desc)"
            )
            conn.execute(
                """
                create table if not exists calibration_runs (
                    id integer primary key autoincrement,
                    status text not null default 'queued',
                    mode text not null,
                    camera_ids_json text not null default '[]',
                    configuration_fingerprint text not null,
                    result_json text not null default '{}',
                    error text not null default '',
                    created_at text not null,
                    updated_at text not null,
                    completed_at text
                )
                """
            )
            conn.execute(
                "create index if not exists idx_calibration_runs_created on calibration_runs(created_at desc, id desc)"
            )
            conn.execute(
                """
                create table if not exists calibration_change_sets (
                    id integer primary key autoincrement,
                    run_id integer,
                    parent_change_set_id integer,
                    action text not null default 'apply',
                    status text not null default 'applied',
                    evaluation_hours real not null default 24,
                    configuration_fingerprint_before text not null,
                    configuration_fingerprint_after text not null,
                    changes_json text not null default '[]',
                    apply_result_json text not null default '{}',
                    evaluation_json text not null default '{}',
                    created_at text not null,
                    updated_at text not null,
                    foreign key(run_id) references calibration_runs(id),
                    foreign key(parent_change_set_id) references calibration_change_sets(id)
                )
                """
            )
            conn.execute(
                "create index if not exists idx_calibration_change_sets_created on calibration_change_sets(created_at desc, id desc)"
            )
            conn.execute(
                "update calibration_runs set status = 'interrupted', error = 'SurvNG restarted before calibration completed', updated_at = ? where status in ('queued', 'running')",
                (datetime.now(timezone.utc).isoformat(),),
            )
            conn.execute(
                "update calibration_change_sets set status = 'evaluation_failed', evaluation_json = ?, updated_at = ? where status = 'reviewing'",
                (
                    json.dumps({
                        "outcome": "failed",
                        "error": "SurvNG restarted during calibration follow-up; run the evaluation again",
                    }, separators=(",", ":")),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.execute(
                """
                create table if not exists tracking_comparisons (
                    id integer primary key autoincrement,
                    event_id integer not null unique,
                    camera_id text not null,
                    event_created_at text not null default '',
                    result_json text not null,
                    verdict text not null default '',
                    reviewed_at text,
                    created_at text not null
                )
                """
            )
            conn.execute(
                "create index if not exists idx_tracking_comparisons_camera_created on tracking_comparisons(camera_id, created_at desc, id desc)"
            )
            conn.execute(
                "update motion_ai_reviews set status = 'interrupted', error = 'SurvNG restarted before this review completed' where status in ('queued', 'running')"
            )
            conn.execute(
                """
                update camera_intelligence_evaluations
                set status = 'collecting', followup_review_id = null,
                    error = 'SurvNG restarted during the follow-up; run it again',
                    updated_at = ?
                where status = 'reviewing'
                """,
                (datetime.now(timezone.utc).isoformat(),),
            )
            # Older active/cooldown observations predate durable linkage. The
            # state reason guarantees they belong to an already-created event;
            # attach them to the nearest preceding event for the same camera.
            conn.execute(
                """
                update motion_audits as audit
                set related_event_id = (
                    select event.id from events as event
                    where event.camera_id = audit.camera_id
                      and julianday(event.created_at) <= julianday(audit.created_at)
                      and (julianday(audit.created_at) - julianday(event.created_at)) * 86400.0 <= 300.0
                    order by julianday(event.created_at) desc, event.id desc
                    limit 1
                )
                where audit.related_event_id is null
                  and audit.event_id is null
                  and audit.reason in ('event_state_active', 'event_state_cooldown')
                  and exists (
                    select 1 from events as event
                    where event.camera_id = audit.camera_id
                      and julianday(event.created_at) <= julianday(audit.created_at)
                      and (julianday(audit.created_at) - julianday(event.created_at)) * 86400.0 <= 300.0
                )
                """
            )
            storage_root = str(self.storage_dir.resolve())
            if self._metadata_value(conn, "portable_media_paths") != "1":
                self._rebase_media_paths(conn)
                self._set_metadata_value(conn, "portable_media_paths", "1")
            if self._metadata_value(conn, "event_storage_root") != storage_root:
                self._set_metadata_value(conn, "event_storage_root", storage_root)
            try:
                backfill_after_id = int(
                    self._metadata_value(conn, "motion_audit_backfill_event_id") or 0
                )
            except ValueError:
                backfill_after_id = 0
            latest_event_id = int(
                conn.execute("select coalesce(max(id), 0) from events").fetchone()[0]
            )
            if latest_event_id > backfill_after_id:
                self._backfill_motion_audits(conn, after_event_id=backfill_after_id)
                self._set_metadata_value(
                    conn,
                    "motion_audit_backfill_event_id",
                    str(latest_event_id),
                )

    @staticmethod
    def _calibration_run_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        for column, target, fallback in (
            ("camera_ids_json", "camera_ids", []),
            ("result_json", "result", {}),
        ):
            if column not in payload:
                payload[target] = fallback
                continue
            try:
                payload[target] = json.loads(str(payload.pop(column) or ""))
            except (json.JSONDecodeError, TypeError):
                payload[target] = fallback
        return payload

    def create_calibration_run(
        self,
        *,
        mode: str,
        camera_ids: list[str],
        configuration_fingerprint: str,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert into calibration_runs (
                    status, mode, camera_ids_json, configuration_fingerprint,
                    created_at, updated_at
                ) values ('queued', ?, ?, ?, ?, ?)
                """,
                (
                    mode,
                    json.dumps(camera_ids, separators=(",", ":")),
                    configuration_fingerprint,
                    now,
                    now,
                ),
            )
            run_id = int(cursor.lastrowid)
        return self.get_calibration_run(run_id) or {}

    def update_calibration_run(
        self,
        run_id: int,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        completed_at = now if status in {"completed", "failed", "interrupted"} else None
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                update calibration_runs
                set status = ?, result_json = coalesce(?, result_json), error = ?,
                    updated_at = ?, completed_at = coalesce(?, completed_at)
                where id = ?
                """,
                (
                    status,
                    (
                        json.dumps(result, separators=(",", ":"), allow_nan=False)
                        if result is not None
                        else None
                    ),
                    error,
                    now,
                    completed_at,
                    int(run_id),
                ),
            )
        return self.get_calibration_run(run_id) or {}

    def get_calibration_run(self, run_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from calibration_runs where id = ?",
                (int(run_id),),
            ).fetchone()
        return self._calibration_run_row(row)

    def calibration_runs(
        self,
        limit: int = 20,
        *,
        include_result: bool = False,
    ) -> list[dict[str, Any]]:
        columns = "*" if include_result else """
            id, status, mode, camera_ids_json, configuration_fingerprint,
            error, created_at, updated_at, completed_at
        """
        with self._connect() as conn:
            rows = conn.execute(
                f"select {columns} from calibration_runs order by created_at desc, id desc limit ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [item for row in rows if (item := self._calibration_run_row(row))]

    def calibration_rollback_change_ids(self, parent_change_set_id: int) -> set[str]:
        """Return source change IDs already reversed by child rollback entries."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                select changes_json from calibration_change_sets
                where parent_change_set_id = ? and action = 'rollback'
                """,
                (int(parent_change_set_id),),
            ).fetchall()
        change_ids: set[str] = set()
        for row in rows:
            try:
                changes = json.loads(str(row["changes_json"] or "[]"))
            except (json.JSONDecodeError, TypeError):
                continue
            for change in changes:
                source_id = str(change.get("source_change_id") or "")
                if source_id:
                    change_ids.add(source_id)
        return change_ids

    @staticmethod
    def _calibration_change_set_row(
        row: sqlite3.Row | None,
    ) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        for column, target, fallback in (
            ("changes_json", "changes", []),
            ("apply_result_json", "apply_result", {}),
            ("evaluation_json", "evaluation", {}),
        ):
            try:
                payload[target] = json.loads(str(payload.pop(column) or ""))
            except (json.JSONDecodeError, TypeError):
                payload[target] = fallback
        return payload

    def create_calibration_change_set(
        self,
        *,
        run_id: int | None,
        parent_change_set_id: int | None,
        action: str,
        status: str,
        evaluation_hours: float,
        configuration_fingerprint_before: str,
        configuration_fingerprint_after: str,
        changes: list[dict[str, Any]],
        apply_result: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert into calibration_change_sets (
                    run_id, parent_change_set_id, action, status,
                    evaluation_hours, configuration_fingerprint_before,
                    configuration_fingerprint_after, changes_json,
                    apply_result_json, created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    parent_change_set_id,
                    action,
                    status,
                    max(24.0, min(float(evaluation_hours), 168.0)),
                    configuration_fingerprint_before,
                    configuration_fingerprint_after,
                    json.dumps(changes, separators=(",", ":"), allow_nan=False),
                    json.dumps(apply_result, separators=(",", ":"), allow_nan=False),
                    now,
                    now,
                ),
            )
            change_set_id = int(cursor.lastrowid)
        return self.get_calibration_change_set(change_set_id) or {}

    def get_calibration_change_set(
        self,
        change_set_id: int,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from calibration_change_sets where id = ?",
                (int(change_set_id),),
            ).fetchone()
        return self._calibration_change_set_row(row)

    def calibration_change_sets(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from calibration_change_sets order by created_at desc, id desc limit ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [
            item for row in rows
            if (item := self._calibration_change_set_row(row))
        ]

    def update_calibration_evaluation(
        self,
        change_set_id: int,
        evaluation: dict[str, Any],
        *,
        status: str,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                update calibration_change_sets
                set status = ?, evaluation_json = ?, updated_at = ? where id = ?
                """,
                (
                    status,
                    json.dumps(evaluation, separators=(",", ":"), allow_nan=False),
                    now,
                    int(change_set_id),
                ),
            )
        return self.get_calibration_change_set(change_set_id) or {}

    def update_calibration_change_set_status(
        self,
        change_set_id: int,
        status: str,
    ) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            conn.execute(
                "update calibration_change_sets set status = ?, updated_at = ? where id = ?",
                (status, datetime.now(timezone.utc).isoformat(), int(change_set_id)),
            )
        return self.get_calibration_change_set(change_set_id) or {}

    def protected_recording_paths(self) -> set[str]:
        """Return continuous segments still referenced by incident history."""
        recordings_root = (self.storage_dir / "recordings").resolve()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT recording_path FROM events WHERE recording_path != ''"
            ).fetchall()
        protected: set[str] = set()
        for row in rows:
            raw_path = str(row["recording_path"] or "")
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.storage_dir / path
            resolved = path.resolve(strict=False)
            try:
                resolved.relative_to(recordings_root)
            except ValueError:
                continue
            protected.add(str(resolved))
        return protected

    @staticmethod
    def _result_json_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        try:
            result = json.loads(str(payload.pop("result_json") or "{}"))
        except (json.JSONDecodeError, TypeError):
            result = {}
        payload["result"] = result if isinstance(result, dict) else {}
        return payload

    def save_tracking_comparison(
        self,
        *,
        event_id: int,
        camera_id: str,
        event_created_at: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_event_id = int(event_id)
        if normalized_event_id <= 0:
            raise ValueError("tracking comparison event id must be positive")
        normalized_camera_id = str(camera_id or "").strip()
        if not normalized_camera_id:
            raise ValueError("tracking comparison camera id is required")
        result_json = json.dumps(result, separators=(",", ":"), allow_nan=False)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                insert into tracking_comparisons (
                    event_id, camera_id, event_created_at, result_json, created_at
                ) values (?, ?, ?, ?, ?)
                on conflict(event_id) do update set
                    camera_id = excluded.camera_id,
                    event_created_at = excluded.event_created_at,
                    result_json = excluded.result_json,
                    verdict = '',
                    reviewed_at = null,
                    created_at = excluded.created_at
                """,
                (
                    normalized_event_id,
                    normalized_camera_id,
                    str(event_created_at or ""),
                    result_json,
                    now,
                ),
            )
            row = conn.execute(
                "select * from tracking_comparisons where event_id = ?",
                (normalized_event_id,),
            ).fetchone()
            conn.execute(
                """
                delete from tracking_comparisons
                where camera_id = ? and id not in (
                    select id from tracking_comparisons
                    where camera_id = ?
                    order by created_at desc, id desc
                    limit ?
                )
                """,
                (
                    normalized_camera_id,
                    normalized_camera_id,
                    self.TRACKING_COMPARISON_HISTORY_PER_CAMERA,
                ),
            )
        comparison = self._result_json_row(row)
        if comparison is None:
            raise RuntimeError("tracking comparison could not be persisted")
        return comparison

    def set_tracking_comparison_verdict(
        self,
        comparison_id: int,
        verdict: str,
    ) -> dict[str, Any] | None:
        normalized_verdict = str(verdict or "").strip()
        if normalized_verdict not in self.TRACKING_COMPARISON_VERDICTS:
            raise ValueError("invalid tracking comparison verdict")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "update tracking_comparisons set verdict = ?, reviewed_at = ? where id = ?",
                (normalized_verdict, now, int(comparison_id)),
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "select * from tracking_comparisons where id = ?",
                (int(comparison_id),),
            ).fetchone()
        return self._result_json_row(row)

    def tracking_comparison_history(
        self,
        *,
        camera_id: str = "",
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 100))
        normalized_camera_id = str(camera_id or "").strip()
        where = "where camera_id = ?" if normalized_camera_id else ""
        values: list[Any] = [normalized_camera_id] if normalized_camera_id else []
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select * from tracking_comparisons
                {where}
                order by created_at desc, id desc
                limit ?
                """,
                [*values, bounded_limit],
            ).fetchall()
        return [self._result_json_row(row) or {} for row in rows]

    def tracking_comparison_summary(self, *, camera_id: str = "") -> dict[str, Any]:
        normalized_camera_id = str(camera_id or "").strip()
        where = "where camera_id = ?" if normalized_camera_id else ""
        values: tuple[Any, ...] = (normalized_camera_id,) if normalized_camera_id else ()
        with self._connect() as conn:
            rows = conn.execute(
                f"select verdict, count(*) as count from tracking_comparisons {where} group by verdict",
                values,
            ).fetchall()
        counts = {"unreviewed": 0, **{value: 0 for value in self.TRACKING_COMPARISON_VERDICTS}}
        for row in rows:
            key = str(row["verdict"] or "unreviewed")
            if key in counts:
                counts[key] = int(row["count"])
        return {
            "camera_id": normalized_camera_id,
            "total": sum(counts.values()),
            "reviewed": sum(counts[value] for value in self.TRACKING_COMPARISON_VERDICTS),
            "verdicts": counts,
        }

    @staticmethod
    def _metadata_value(conn: sqlite3.Connection, key: str) -> str:
        row = conn.execute(
            "select value from survng_metadata where key = ?",
            (key,),
        ).fetchone()
        return str(row[0]) if row is not None else ""

    @staticmethod
    def _set_metadata_value(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            """
            insert into survng_metadata (key, value) values (?, ?)
            on conflict(key) do update set value = excluded.value
            """,
            (key, value),
        )

    @staticmethod
    def _qualification_from_objects(objects_json: str) -> dict[str, Any] | None:
        try:
            objects = json.loads(objects_json or "[]")
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(objects, list):
            return None
        return next(
            (
                item.get("motion_qualification")
                for item in objects
                if isinstance(item, dict)
                if item.get("status") == "motion_qualification"
                and isinstance(item.get("motion_qualification"), dict)
            ),
            None,
        )

    def _backfill_motion_audits(
        self,
        conn: sqlite3.Connection,
        *,
        after_event_id: int = 0,
    ) -> None:
        rows = conn.execute(
            """
            select id, camera_id, snapshot_path, created_at, objects_json
            from events
            where id > ?
              and objects_json like '%motion_qualification%'
              and id not in (select event_id from motion_audits where event_id is not null)
            order by id asc
            """,
            (max(0, int(after_event_id)),),
        ).fetchall()
        inserts: list[tuple[Any, ...]] = []
        for row in rows:
            qualification = self._qualification_from_objects(str(row["objects_json"] or ""))
            if not qualification:
                continue
            visual_backup = qualification.get("trigger_source") == "visual_backup"
            if not qualification.get("would_suppress") and not visual_backup:
                continue
            try:
                objects = json.loads(str(row["objects_json"] or "[]"))
            except (json.JSONDecodeError, TypeError):
                objects = []
            object_detected = any(
                isinstance(item, dict)
                and item.get("label")
                and item.get("incident_eligible") is not False
                for item in objects
            ) if isinstance(objects, list) else False
            raw_features = qualification.get("features")
            features = dict(raw_features) if isinstance(raw_features, dict) else {}
            reason = str(qualification.get("reason") or "rejected")
            category = "qualification"
            if visual_backup:
                features["visual_backup_original_reason"] = reason
                reason = "visual_backup_trigger"
                category = "visual_backup"
            inserts.append((
                int(row["id"]),
                str(row["camera_id"]),
                str(row["snapshot_path"] or ""),
                str(row["created_at"]),
                str(qualification.get("mode") or "audit"),
                str(qualification.get("sensitivity") or "balanced"),
                float(qualification.get("score") or 0.0),
                float(qualification.get("threshold") or 0.0),
                reason,
                int(object_detected),
                max(1, int(qualification.get("trigger_count") or 1)),
                json.dumps(features, separators=(",", ":")),
                category,
            ))
        if inserts:
            conn.executemany(
                """
                insert or ignore into motion_audits (
                    event_id, camera_id, snapshot_path, created_at, mode,
                    sensitivity, score, threshold, reason, object_detected,
                    trigger_count, features_json, category
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                inserts,
            )

    def _rebase_media_paths(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "select id, snapshot_path, recording_path from events where snapshot_path != '' or recording_path != ''"
        ).fetchall()
        updates: list[tuple[str, str, int]] = []
        for row in rows:
            snapshot_path = portable_media_path(self.storage_dir, row["snapshot_path"])
            recording_path = portable_media_path(self.storage_dir, row["recording_path"])
            if snapshot_path != str(row["snapshot_path"] or "") or recording_path != str(row["recording_path"] or ""):
                updates.append((snapshot_path, recording_path, int(row["id"])))
        if updates:
            conn.executemany(
                "update events set snapshot_path = ?, recording_path = ? where id = ?",
                updates,
            )

        audit_rows = conn.execute(
            "select id, snapshot_path from motion_audits where snapshot_path != ''"
        ).fetchall()
        audit_updates: list[tuple[str, int]] = []
        for row in audit_rows:
            raw_path = str(row["snapshot_path"] or "")
            portable_path = portable_media_path(self.storage_dir, raw_path)
            if portable_path != raw_path:
                audit_updates.append((portable_path, int(row["id"])))
        if audit_updates:
            conn.executemany(
                "update motion_audits set snapshot_path = ? where id = ?",
                audit_updates,
            )

        face_table = conn.execute(
            "select 1 from sqlite_master where type = 'table' and name = 'face_observations'"
        ).fetchone()
        if face_table:
            face_rows = conn.execute(
                "select id, snapshot_path from face_observations where snapshot_path != ''"
            ).fetchall()
            face_updates: list[tuple[str, int]] = []
            for row in face_rows:
                raw_path = str(row["snapshot_path"] or "")
                portable_path = portable_media_path(self.storage_dir, raw_path)
                if portable_path != raw_path:
                    face_updates.append((portable_path, int(row["id"])))
            if face_updates:
                conn.executemany(
                    "update face_observations set snapshot_path = ? where id = ?",
                    face_updates,
                )

    def add_event(
        self,
        camera_id: str,
        kind: str,
        topic: str = "",
        message: str = "",
        snapshot_path: str = "",
        recording_path: str = "",
        objects_json: str = "[]",
        created_at: str | None = None,
    ) -> dict[str, Any]:
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()
        snapshot_path = portable_media_path(self.storage_dir, snapshot_path)
        recording_path = portable_media_path(self.storage_dir, recording_path)
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert into events (
                    camera_id, kind, topic, message, snapshot_path,
                    recording_path, objects_json, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    kind,
                    topic,
                    message,
                    snapshot_path,
                    recording_path,
                    objects_json,
                    created_at,
                ),
            )
            event_id = cursor.lastrowid
        return {
            "id": event_id,
            "camera_id": camera_id,
            "kind": kind,
            "topic": topic,
            "message": message,
            "snapshot_path": snapshot_path,
            "recording_path": recording_path,
            "objects_json": objects_json,
            "created_at": created_at,
        }

    def telemetry_activity(
        self,
        *,
        hours: int = 24,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Aggregate bounded event/object history into UTC hourly buckets."""
        bounded_hours = max(1, min(int(hours), 168))
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        current_hour = current.replace(minute=0, second=0, microsecond=0)
        first_hour = current_hour - timedelta(hours=bounded_hours - 1)
        one_hour_ago = current - timedelta(hours=1)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                select camera_id, created_at, objects_json
                from events
                where created_at >= ?
                order by created_at, id
                """,
                (first_hour.isoformat(),),
            ).fetchall()

        def empty_counts() -> dict[str, Any]:
            return {"events": 0, "object_incidents": 0, "objects": 0, "labels": {}}

        def add_counts(target: dict[str, Any], detected: list[dict[str, Any]]) -> None:
            target["events"] += 1
            if detected:
                target["object_incidents"] += 1
            target["objects"] += len(detected)
            labels = target["labels"]
            for item in detected:
                label = str(item.get("label") or "unknown")
                labels[label] = int(labels.get(label, 0)) + 1

        hourly = [
            {"started_at": (first_hour + timedelta(hours=index)).isoformat(), **empty_counts()}
            for index in range(bounded_hours)
        ]

        def empty_hourly() -> list[dict[str, Any]]:
            return [
                {"started_at": item["started_at"], **empty_counts()}
                for item in hourly
            ]

        overall_24h = empty_counts()
        overall_1h = empty_counts()
        cameras: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                created_at = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                created_at = created_at.astimezone(timezone.utc)
            except (TypeError, ValueError):
                continue
            bucket_index = int((created_at.replace(minute=0, second=0, microsecond=0) - first_hour).total_seconds() // 3600)
            if bucket_index < 0 or bucket_index >= bounded_hours:
                continue
            try:
                objects = json.loads(str(row["objects_json"] or "[]"))
            except (TypeError, ValueError):
                objects = []
            detected = [
                item for item in objects
                if isinstance(item, dict)
                and item.get("label")
                and item.get("incident_eligible") is not False
                and not item.get("status")
            ] if isinstance(objects, list) else []
            camera_id = str(row["camera_id"])
            camera = cameras.setdefault(
                camera_id,
                {"last_24h": empty_counts(), "last_hour": empty_counts(), "hourly": empty_hourly()},
            )
            add_counts(hourly[bucket_index], detected)
            add_counts(camera["hourly"][bucket_index], detected)
            add_counts(overall_24h, detected)
            add_counts(camera["last_24h"], detected)
            if created_at >= one_hour_ago:
                add_counts(overall_1h, detected)
                add_counts(camera["last_hour"], detected)

        return {
            "hours": bounded_hours,
            "started_at": first_hour.isoformat(),
            "last_hour": overall_1h,
            "last_24h": overall_24h,
            "hourly": hourly,
            "by_camera": cameras,
        }

    def record_runtime_telemetry(
        self,
        camera_statuses: list[dict[str, Any]],
        *,
        sampled_at: datetime | None = None,
        process_memory: dict[str, Any] | None = None,
        worker_memory: dict[str, Any] | None = None,
        memory_maintenance: dict[str, Any] | None = None,
        system_runtime: dict[str, Any] | None = None,
    ) -> None:
        """Persist one compact camera-health sample per UTC minute for seven days."""
        current = sampled_at or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc).replace(second=0, microsecond=0)
        cameras: dict[str, Any] = {}
        for status in camera_statuses:
            camera_id = str(status.get("id") or "")
            if not camera_id:
                continue
            motion = status.get("motion_qualification") or {}
            tracking = status.get("object_tracking") or {}
            lifecycle = status.get("lifecycle") or {}
            cameras[camera_id] = {
                "enabled": bool(
                    status.get("expected_enabled", lifecycle.get("enabled", True))
                ),
                "connected": bool(status.get("connected")),
                "frame_age_seconds": status.get("last_frame_age_seconds"),
                "main_frame_age_seconds": status.get("main_last_frame_age_seconds"),
                "capture": status.get("capture_stats") or {},
                "analysis_frames_dropped": int(motion.get("analysis_frames_dropped") or 0),
                "analysis_runtime": motion.get("analysis_runtime") or {},
                "event_runtime": motion.get("event_runtime") or {},
                "tracking_active": bool(tracking.get("active")),
            }
        payload = json.dumps(
            {
                "cameras": cameras,
                "process_memory": process_memory or {},
                "worker_memory": worker_memory or {},
                "memory_maintenance": memory_maintenance or {},
                "system_runtime": system_runtime or {},
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        cutoff = current - timedelta(days=8)
        with self._lock, self._connect() as conn:
            conn.execute(
                "insert or replace into runtime_telemetry_samples (sampled_at, payload_json) values (?, ?)",
                (current.isoformat(), payload),
            )
            conn.execute(
                "delete from runtime_telemetry_samples where sampled_at < ?",
                (cutoff.isoformat(),),
            )

    def record_lifecycle_event(
        self,
        instance_id: str,
        kind: str,
        *,
        occurred_at: datetime | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Persist a process lifecycle transition used to explain telemetry gaps."""
        allowed = {
            "startup_started",
            "startup_ready",
            "shutdown_requested",
            "shutdown_completed",
        }
        if kind not in allowed:
            raise ValueError(f"unsupported lifecycle event: {kind}")
        current = occurred_at or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        payload = json.dumps(details or {}, separators=(",", ":"), allow_nan=False)
        cutoff = current - timedelta(days=8)
        with self._lock, self._connect() as conn:
            conn.execute(
                "insert into system_lifecycle_events "
                "(instance_id, kind, occurred_at, details_json) values (?, ?, ?, ?)",
                (str(instance_id), kind, current.isoformat(), payload),
            )
            conn.execute(
                "delete from system_lifecycle_events where occurred_at < ?",
                (cutoff.isoformat(),),
            )

    def lifecycle_events(
        self,
        *,
        hours: int = 168,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        start = current - timedelta(hours=max(1, min(int(hours), 24 * 8)))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "select instance_id, kind, occurred_at, details_json "
                "from system_lifecycle_events where occurred_at >= ? "
                "order by occurred_at, id",
                (start.isoformat(),),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                details = json.loads(str(row["details_json"] or "{}"))
            except json.JSONDecodeError:
                details = {}
            result.append(
                {
                    "instance_id": str(row["instance_id"]),
                    "kind": str(row["kind"]),
                    "occurred_at": str(row["occurred_at"]),
                    "details": details if isinstance(details, dict) else {},
                }
            )
        return result

    def runtime_telemetry_sample_times(
        self,
        *,
        hours: int = 168,
        now: datetime | None = None,
    ) -> list[str]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        start = current - timedelta(hours=max(1, min(int(hours), 24 * 8)))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "select sampled_at from runtime_telemetry_samples "
                "where sampled_at >= ? order by sampled_at",
                (start.isoformat(),),
            ).fetchall()
        return [str(row["sampled_at"]) for row in rows]

    def process_memory_history(
        self,
        *,
        hours: int,
        bucket_minutes: int,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return persisted main-process memory samples for leak diagnosis."""
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        bounded_hours = max(1, min(int(hours), 24 * 8))
        bucket_seconds = max(60, min(int(bucket_minutes), 60) * 60)
        start = current - timedelta(hours=bounded_hours)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "select sampled_at, payload_json from runtime_telemetry_samples "
                "where sampled_at >= ? order by sampled_at",
                (start.isoformat(),),
            ).fetchall()

        buckets: dict[int, dict[str, Any]] = {}
        for row in rows:
            try:
                sampled = datetime.fromisoformat(
                    str(row["sampled_at"]).replace("Z", "+00:00")
                )
                payload = json.loads(str(row["payload_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            memory = payload.get("process_memory") if isinstance(payload, dict) else None
            if not isinstance(memory, dict) or not memory:
                continue
            malloc = memory.get("malloc")
            malloc = malloc if isinstance(malloc, dict) else {}
            worker_memory = payload.get("worker_memory")
            worker_memory = worker_memory if isinstance(worker_memory, dict) else {}
            maintenance = payload.get("memory_maintenance")
            maintenance = maintenance if isinstance(maintenance, dict) else {}
            bucket_epoch = int(sampled.timestamp() // bucket_seconds) * bucket_seconds
            buckets[bucket_epoch] = {
                "sampled_at": sampled.astimezone(timezone.utc).isoformat(),
                "rss_bytes": int(memory.get("rss_bytes") or 0),
                "anonymous_rss_bytes": int(memory.get("anonymous_rss_bytes") or 0),
                "pss_bytes": int(memory.get("pss_bytes") or 0),
                "private_dirty_bytes": int(memory.get("private_dirty_bytes") or 0),
                "anonymous_huge_pages_bytes": int(
                    memory.get("anonymous_huge_pages_bytes") or 0
                ),
                "malloc_allocated_bytes": int(malloc.get("allocated_bytes") or 0),
                "malloc_free_bytes": int(malloc.get("free_bytes") or 0),
                "malloc_mmap_bytes": int(malloc.get("mmap_bytes") or 0),
                "worker_rss_bytes": int(worker_memory.get("total_rss_bytes") or 0),
                "worker_pss_bytes": int(worker_memory.get("total_pss_bytes") or 0),
                "allocator_trim_count": int(maintenance.get("successful_trims") or 0),
                "allocator_trim_reclaimed_bytes": int(
                    maintenance.get("reclaimed_total_bytes") or 0
                ),
                "threads": int(memory.get("threads") or 0),
                "file_descriptors": int(memory.get("file_descriptors") or 0),
            }
        return [buckets[bucket] for bucket in sorted(buckets)]

    def runtime_telemetry_history(
        self,
        *,
        hours: int,
        bucket_minutes: int,
        camera_id: str = "",
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        bounded_hours = max(1, min(int(hours), 24 * 8))
        bucket_seconds = max(60, min(int(bucket_minutes), 60) * 60)
        start = current - timedelta(hours=bounded_hours)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "select sampled_at, payload_json from runtime_telemetry_samples where sampled_at >= ? order by sampled_at",
                (start.isoformat(),),
            ).fetchall()

        buckets: dict[int, dict[str, Any]] = {}
        previous_counters: dict[str, dict[str, int]] = {}
        for row in rows:
            try:
                sampled = datetime.fromisoformat(str(row["sampled_at"]).replace("Z", "+00:00"))
                payload = json.loads(str(row["payload_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            cameras = payload.get("cameras") if isinstance(payload, dict) else {}
            if not isinstance(cameras, dict):
                continue
            if camera_id:
                selected = (
                    [(camera_id, cameras[camera_id])]
                    if isinstance(cameras.get(camera_id), dict)
                    else []
                )
            else:
                selected = [
                    (key, value)
                    for key, value in cameras.items()
                    if isinstance(value, dict)
                ]
            bucket_epoch = int(sampled.timestamp() // bucket_seconds) * bucket_seconds
            bucket = buckets.setdefault(bucket_epoch, {
                "sampled_at": datetime.fromtimestamp(bucket_epoch, timezone.utc).isoformat(),
                "live_fps_total": 0.0,
                "main_fps_total": 0.0,
                "live_fps_samples": 0,
                "main_fps_samples": 0,
                "tracking_active_max": 0,
                "capture_read_failures": 0,
                "capture_open_failures": 0,
                "main_capture_starts": 0,
                "analysis_frames_dropped": 0,
                "capture_observer_p99_ms": 0.0,
                "capture_to_analysis_p95_ms": 0.0,
                "preprocess_p99_ms": 0.0,
                "motion_copy_bytes": 0,
                "event_evictions": 0,
                "event_rejections": 0,
                "event_retry_drops": 0,
                "analysis_frames_sampled": 0,
                "camera_availability_total": 0.0,
                "camera_availability_samples": 0,
                "expected_cameras": 0,
                "unavailable_cameras": 0,
                "cpu_load_percent_total": 0.0,
                "memory_used_percent_total": 0.0,
                "inference_ms_total": 0.0,
                "system_runtime_samples": 0,
                "inference_samples": 0,
            })
            system_runtime = payload.get("system_runtime") or {}
            if isinstance(system_runtime, dict):
                cpu_load = system_runtime.get("cpu_load_percent")
                memory_used = system_runtime.get("memory_used_percent")
                if isinstance(cpu_load, (int, float)) and math.isfinite(cpu_load):
                    bucket["cpu_load_percent_total"] += max(0.0, float(cpu_load))
                    if isinstance(memory_used, (int, float)) and math.isfinite(memory_used):
                        bucket["memory_used_percent_total"] += max(0.0, float(memory_used))
                    bucket["system_runtime_samples"] += 1
                inference_ms = system_runtime.get("inference_ms")
                if isinstance(inference_ms, (int, float)) and math.isfinite(inference_ms):
                    bucket["inference_ms_total"] += max(0.0, float(inference_ms))
                    bucket["inference_samples"] += 1
            live_values: list[float] = []
            main_values: list[float] = []
            active = 0
            expected = 0
            available = 0
            for selected_id, item in selected:
                capture = item.get("capture") or {}
                live = capture.get("live") or {}
                main = capture.get("main") or {}
                analysis_runtime = item.get("analysis_runtime") or {}
                event_runtime = item.get("event_runtime") or {}
                live_fps = float(live.get("fps") or 0.0)
                main_fps = float(main.get("fps") or 0.0)
                if camera_id or bool(item.get("connected")):
                    live_values.append(live_fps)
                if camera_id or main_fps > 0:
                    main_values.append(main_fps)
                active += int(bool(item.get("tracking_active")))
                if bool(item.get("enabled", True)):
                    expected += 1
                    frame_age = item.get("frame_age_seconds")
                    frame_fresh = frame_age is None or float(frame_age) <= 5.0
                    available += int(bool(item.get("connected")) and frame_fresh)
                bucket["capture_observer_p99_ms"] = max(
                    float(bucket["capture_observer_p99_ms"]),
                    float(live.get("observer_p99_ms") or 0.0),
                    float(main.get("observer_p99_ms") or 0.0),
                )
                bucket["capture_to_analysis_p95_ms"] = max(
                    float(bucket["capture_to_analysis_p95_ms"]),
                    float(analysis_runtime.get("capture_to_analysis_p95_ms") or 0.0),
                )
                bucket["preprocess_p99_ms"] = max(
                    float(bucket["preprocess_p99_ms"]),
                    float(analysis_runtime.get("preprocess_p99_ms") or 0.0),
                )
                counters = {
                    "capture_read_failures": int(live.get("read_failures") or 0) + int(main.get("read_failures") or 0),
                    "capture_open_failures": int(live.get("open_failures") or 0) + int(main.get("open_failures") or 0),
                    "main_capture_starts": int(main.get("starts") or 0),
                    "analysis_frames_dropped": int(item.get("analysis_frames_dropped") or 0),
                    "analysis_frames_sampled": int(analysis_runtime.get("frames_sampled") or 0),
                    "motion_copy_bytes": int(analysis_runtime.get("copy_bytes") or 0),
                    "event_evictions": int(event_runtime.get("evicted") or 0),
                    "event_rejections": int(event_runtime.get("rejected") or 0),
                    "event_retry_drops": int(event_runtime.get("retries_dropped") or 0),
                }
                previous = previous_counters.get(selected_id)
                if previous is not None:
                    for key, value in counters.items():
                        bucket[key] += max(0, value - previous[key]) if value >= previous[key] else max(0, value)
                previous_counters[selected_id] = counters
            if expected:
                bucket["camera_availability_total"] += (available / expected) * 100.0
                bucket["camera_availability_samples"] += 1
                bucket["expected_cameras"] = max(int(bucket["expected_cameras"]), expected)
                bucket["unavailable_cameras"] = max(
                    int(bucket["unavailable_cameras"]),
                    expected - available,
                )
            if live_values:
                bucket["live_fps_total"] += sum(live_values) / len(live_values)
                bucket["live_fps_samples"] += 1
            if main_values:
                bucket["main_fps_total"] += sum(main_values) / len(main_values)
                bucket["main_fps_samples"] += 1
            if selected:
                bucket["tracking_active_max"] = max(bucket["tracking_active_max"], active)

        result: list[dict[str, Any]] = []
        for bucket_epoch in sorted(buckets):
            bucket = buckets[bucket_epoch]
            live_samples = max(1, int(bucket["live_fps_samples"]))
            main_samples = max(1, int(bucket["main_fps_samples"]))
            availability_samples = max(1, int(bucket["camera_availability_samples"]))
            system_runtime_samples = int(bucket["system_runtime_samples"])
            inference_samples = int(bucket["inference_samples"])
            analyzed = int(bucket["analysis_frames_sampled"])
            superseded = int(bucket["analysis_frames_dropped"])
            analysis_total = analyzed + superseded
            result.append({
                "sampled_at": bucket["sampled_at"],
                "live_fps": round(float(bucket["live_fps_total"]) / live_samples, 2),
                "main_fps": round(float(bucket["main_fps_total"]) / main_samples, 2),
                "tracking_active_max": int(bucket["tracking_active_max"]),
                "capture_read_failures": int(bucket["capture_read_failures"]),
                "capture_open_failures": int(bucket["capture_open_failures"]),
                "capture_interruptions": int(bucket["capture_read_failures"])
                + int(bucket["capture_open_failures"]),
                "main_capture_starts": int(bucket["main_capture_starts"]),
                "analysis_frames_dropped": int(bucket["analysis_frames_dropped"]),
                "analysis_frames_sampled": analyzed,
                "analysis_coverage_percent": (
                    round((analyzed / analysis_total) * 100.0, 3)
                    if analysis_total else None
                ),
                "camera_availability_percent": (
                    round(float(bucket["camera_availability_total"]) / availability_samples, 2)
                    if bucket["camera_availability_samples"] else None
                ),
                "expected_cameras": int(bucket["expected_cameras"]),
                "unavailable_cameras": int(bucket["unavailable_cameras"]),
                "cpu_load_percent": (
                    round(float(bucket["cpu_load_percent_total"]) / system_runtime_samples, 2)
                    if system_runtime_samples else None
                ),
                "memory_used_percent": (
                    round(float(bucket["memory_used_percent_total"]) / system_runtime_samples, 2)
                    if system_runtime_samples else None
                ),
                "inference_ms": (
                    round(float(bucket["inference_ms_total"]) / inference_samples, 2)
                    if inference_samples else None
                ),
                "capture_observer_p99_ms": round(float(bucket["capture_observer_p99_ms"]), 3),
                "capture_to_analysis_p95_ms": round(float(bucket["capture_to_analysis_p95_ms"]), 3),
                "preprocess_p99_ms": round(float(bucket["preprocess_p99_ms"]), 3),
                "motion_copy_bytes": int(bucket["motion_copy_bytes"]),
                "event_evictions": int(bucket["event_evictions"]),
                "event_rejections": int(bucket["event_rejections"]),
                "event_retry_drops": int(bucket["event_retry_drops"]),
                "event_delivery_failures": int(bucket["event_evictions"])
                + int(bucket["event_rejections"])
                + int(bucket["event_retry_drops"]),
            })
        return result

    def tracking_capacity_activity(
        self,
        *,
        hours: int,
        bucket_minutes: int,
        camera_id: str = "",
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        bounded_hours = max(1, min(int(hours), 24 * 31))
        bucket_seconds = max(60, min(int(bucket_minutes), 60) * 60)
        start = current - timedelta(hours=bounded_hours)
        query = "select camera_id, created_at, objects_json from events where created_at >= ?"
        parameters: list[Any] = [start.isoformat()]
        if camera_id:
            query += " and camera_id = ?"
            parameters.append(camera_id)
        query += " order by created_at, id"
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        buckets: dict[int, dict[str, Any]] = {}
        for row in rows:
            try:
                created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
                objects = json.loads(str(row["objects_json"] or "[]"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            created = created.astimezone(timezone.utc)
            tracking = next((
                item.get("object_tracking")
                for item in objects
                if isinstance(item, dict)
                and item.get("status") == "object_tracking"
                and isinstance(item.get("object_tracking"), dict)
            ), None) if isinstance(objects, list) else None
            if not tracking:
                continue
            bucket_epoch = int(created.timestamp() // bucket_seconds) * bucket_seconds
            bucket = buckets.setdefault(bucket_epoch, {
                "sampled_at": datetime.fromtimestamp(bucket_epoch, timezone.utc).isoformat(),
                "attempts": 0,
                "waited": 0,
                "skipped": 0,
                "wait_seconds_total": 0.0,
                "wait_seconds_max": 0.0,
            })
            wait_seconds = max(0.0, float(tracking.get("capacity_wait_seconds") or 0.0))
            bucket["attempts"] += 1
            bucket["waited"] += int(wait_seconds >= 0.01)
            bucket["skipped"] += int(tracking.get("state") == "skipped_capacity")
            bucket["wait_seconds_total"] += wait_seconds
            bucket["wait_seconds_max"] = max(bucket["wait_seconds_max"], wait_seconds)
        first_bucket = int(start.timestamp() // bucket_seconds) * bucket_seconds
        last_bucket = int(current.timestamp() // bucket_seconds) * bucket_seconds
        result: list[dict[str, Any]] = []
        for bucket_epoch in range(first_bucket, last_bucket + 1, bucket_seconds):
            bucket = buckets.get(bucket_epoch, {
                "sampled_at": datetime.fromtimestamp(bucket_epoch, timezone.utc).isoformat(),
                "attempts": 0,
                "waited": 0,
                "skipped": 0,
                "wait_seconds_total": 0.0,
                "wait_seconds_max": 0.0,
            })
            wait_total = float(bucket["wait_seconds_total"])
            result.append({
                "sampled_at": bucket["sampled_at"],
                "attempts": int(bucket["attempts"]),
                "waited": int(bucket["waited"]),
                "skipped": int(bucket["skipped"]),
                "wait_seconds_average": round(
                    wait_total / max(1, int(bucket["waited"])),
                    3,
                ),
                "wait_seconds_max": round(float(bucket["wait_seconds_max"]), 3),
            })
        return result

    def add_motion_audit(
        self,
        *,
        camera_id: str,
        snapshot_path: str,
        created_at: str,
        mode: str,
        sensitivity: str,
        score: float,
        threshold: float,
        reason: str,
        object_detected: bool | None,
        trigger_count: int,
        features: dict[str, Any],
        category: str = "qualification",
        event_id: int | None = None,
        related_event_id: int | None = None,
        decision_id: str = "",
    ) -> dict[str, Any]:
        normalized_object_detected = (
            None if object_detected is None else int(object_detected)
        )
        normalized_trigger_count = max(1, int(trigger_count))
        snapshot_path = portable_media_path(self.storage_dir, snapshot_path)
        normalized_decision_id = str(decision_id or "").strip()
        normalized_category = str(category or "qualification").strip().lower()
        if normalized_category not in {"qualification", "visual_backup"}:
            raise ValueError("invalid motion audit category")
        normalized_related_event_id = (
            int(related_event_id) if related_event_id is not None else None
        )
        if normalized_related_event_id is not None and normalized_related_event_id <= 0:
            raise ValueError("related motion event id must be positive")
        if len(normalized_decision_id) > 128:
            raise ValueError("motion audit decision_id must be at most 128 characters")
        normalized_score = float(score)
        normalized_threshold = float(threshold)
        if not math.isfinite(normalized_score) or not math.isfinite(normalized_threshold):
            raise ValueError("motion audit score and threshold must be finite")
        features_json = json.dumps(
            features or {},
            separators=(",", ":"),
            allow_nan=False,
        )
        replaced_snapshot = ""
        with self._lock, self._connect() as conn:
            audit_id: int | None = None
            if normalized_decision_id:
                existing = conn.execute(
                    "select id, snapshot_path from motion_audits where decision_id = ?",
                    (normalized_decision_id,),
                ).fetchone()
                if existing is not None:
                    audit_id = int(existing["id"])
                    replaced_snapshot = str(existing["snapshot_path"] or "")
                    conn.execute(
                        """
                        update motion_audits
                        set event_id = coalesce(?, event_id),
                            related_event_id = coalesce(?, related_event_id), camera_id = ?,
                            snapshot_path = ?, created_at = ?, mode = ?,
                            sensitivity = ?, score = ?, threshold = ?, reason = ?,
                            object_detected = ?, trigger_count = ?, features_json = ?,
                            category = ?
                        where id = ?
                        """,
                        (
                            event_id,
                            normalized_related_event_id,
                            camera_id,
                            snapshot_path,
                            created_at,
                            mode,
                            sensitivity,
                            normalized_score,
                            normalized_threshold,
                            reason,
                            normalized_object_detected,
                            normalized_trigger_count,
                            features_json,
                            normalized_category,
                            audit_id,
                        ),
                    )
            elif event_id is None:
                existing = conn.execute(
                    """
                    select id from motion_audits
                    where event_id is null and camera_id = ? and created_at = ?
                      and mode = ? and sensitivity = ? and reason = ?
                      and snapshot_path = ? and score = ? and threshold = ?
                      and object_detected is ? and trigger_count = ?
                      and related_event_id is ?
                      and features_json = ? and category = ?
                    order by id asc limit 1
                    """,
                    (
                        camera_id,
                        created_at,
                        mode,
                        sensitivity,
                        reason,
                        snapshot_path,
                        normalized_score,
                        normalized_threshold,
                        normalized_object_detected,
                        normalized_trigger_count,
                        normalized_related_event_id,
                        features_json,
                        normalized_category,
                    ),
                ).fetchone()
                if existing is not None:
                    audit_id = int(existing["id"])
            if audit_id is None:
                cursor = conn.execute(
                    """
                    insert or ignore into motion_audits (
                        event_id, related_event_id, decision_id, camera_id, snapshot_path, created_at, mode,
                        sensitivity, score, threshold, reason, object_detected,
                        trigger_count, features_json, category
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        normalized_related_event_id,
                        normalized_decision_id or None,
                        camera_id,
                        snapshot_path,
                        created_at,
                        mode,
                        sensitivity,
                        normalized_score,
                        normalized_threshold,
                        reason,
                        normalized_object_detected,
                        normalized_trigger_count,
                        features_json,
                        normalized_category,
                    ),
                )
                if cursor.rowcount:
                    audit_id = int(cursor.lastrowid)
                elif normalized_decision_id:
                    existing = conn.execute(
                        "select id from motion_audits where decision_id = ?",
                        (normalized_decision_id,),
                    ).fetchone()
                    if existing is not None:
                        audit_id = int(existing["id"])
                if audit_id is None and event_id is not None:
                    existing = conn.execute(
                        "select id from motion_audits where event_id = ?",
                        (int(event_id),),
                    ).fetchone()
                    if existing is not None:
                        audit_id = int(existing["id"])
            if audit_id is None:
                raise RuntimeError("motion audit could not be persisted or resolved")
        if replaced_snapshot and replaced_snapshot != snapshot_path:
            self._delete_snapshot_if_unreferenced(replaced_snapshot)
        return self.get_motion_audit(audit_id) or {}

    def motion_audits(
        self,
        *,
        limit: int = 24,
        offset: int = 0,
        camera_id: str = "",
        outcome: str = "all",
        category: str = "all",
        include_incident_activity: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        values: list[Any] = []
        if not include_incident_activity:
            clauses.append("reason not in ('event_state_active', 'event_state_cooldown')")
        if camera_id:
            clauses.append("camera_id = ?")
            values.append(camera_id)
        if category != "all":
            clauses.append("category = ?")
            values.append(category)
        if outcome == "object":
            clauses.append("object_detected = 1")
        elif outcome == "clear":
            clauses.append("object_detected = 0")
        elif outcome == "not_run":
            clauses.append("object_detected is null")
        where = f"where {' and '.join(clauses)}" if clauses else ""
        bounded_limit = max(1, min(int(limit), 100))
        bounded_offset = max(0, int(offset))
        with self._connect() as conn:
            total = int(conn.execute(
                f"select count(*) from motion_audits {where}",
                values,
            ).fetchone()[0])
            rows = conn.execute(
                f"""
                select * from motion_audits
                {where}
                order by created_at desc, id desc
                limit ? offset ?
                """,
                [*values, bounded_limit, bounded_offset],
            ).fetchall()
        return [dict(row) for row in rows], total

    def create_motion_ai_review(self, camera_id: str, audits_considered: int) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert into motion_ai_reviews (
                    camera_id, status, audits_considered, created_at, updated_at
                ) values (?, 'queued', ?, ?, ?)
                """,
                (camera_id, max(0, int(audits_considered)), now, now),
            )
            review_id = int(cursor.lastrowid)
        return self.get_motion_ai_review(review_id) or {}

    def update_motion_ai_review(
        self,
        review_id: int,
        *,
        status: str,
        images_available: int | None = None,
        analyzed: int | None = None,
        failed: int | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        allowed_statuses = {"queued", "running", "completed", "failed", "interrupted"}
        if status not in allowed_statuses:
            raise ValueError("invalid motion AI review status")
        result_json = None if result is None else json.dumps(
            result,
            separators=(",", ":"),
            allow_nan=False,
        )
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            current = conn.execute(
                "select status from motion_ai_reviews where id = ?",
                (int(review_id),),
            ).fetchone()
            if current is None:
                raise KeyError("motion AI review not found")
            current_status = str(current["status"])
            if current_status in {"completed", "failed", "interrupted"}:
                return self._result_json_row(
                    conn.execute(
                        "select * from motion_ai_reviews where id = ?",
                        (int(review_id),),
                    ).fetchone()
                ) or {}
            cursor = conn.execute(
                """
                update motion_ai_reviews
                set status = ?,
                    images_available = coalesce(?, images_available),
                    analyzed = coalesce(?, analyzed),
                    failed = coalesce(?, failed),
                    result_json = coalesce(?, result_json),
                    error = coalesce(?, error),
                    updated_at = ?
                where id = ?
                """,
                (
                    status,
                    images_available,
                    analyzed,
                    failed,
                    result_json,
                    error,
                    now,
                    int(review_id),
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError("motion AI review not found")
        return self.get_motion_ai_review(review_id) or {}

    def get_motion_ai_review(self, review_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from motion_ai_reviews where id = ?",
                (int(review_id),),
            ).fetchone()
        return self._result_json_row(row)

    def latest_motion_ai_review(self, camera_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                select * from motion_ai_reviews
                where camera_id = ?
                order by created_at desc, id desc limit 1
                """,
                (camera_id,),
            ).fetchone()
        return self._result_json_row(row)

    @staticmethod
    def _camera_intelligence_evaluation_row(
        row: sqlite3.Row | None,
    ) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        for column, target in (
            ("applied_changes_json", "applied_changes"),
            ("baseline_result_json", "baseline_result"),
            ("followup_result_json", "followup_result"),
            ("comparison_json", "comparison"),
        ):
            try:
                decoded = json.loads(str(payload.pop(column) or "{}"))
            except (json.JSONDecodeError, TypeError):
                decoded = [] if target == "applied_changes" else {}
            payload[target] = decoded
        try:
            applied_at = datetime.fromisoformat(
                str(payload.get("applied_at") or "").replace("Z", "+00:00")
            )
            if applied_at.tzinfo is None:
                applied_at = applied_at.replace(tzinfo=timezone.utc)
            ready_at = applied_at + timedelta(
                hours=float(payload.get("evaluation_hours") or 24)
            )
            payload["ready_at"] = ready_at.isoformat()
            remaining = (ready_at - datetime.now(timezone.utc)).total_seconds()
            payload["seconds_until_ready"] = max(0, round(remaining))
            if payload.get("status") == "collecting" and remaining <= 0:
                payload["status"] = "ready"
        except (TypeError, ValueError):
            payload["ready_at"] = ""
            payload["seconds_until_ready"] = 0
        return payload

    def create_camera_intelligence_evaluation(
        self,
        *,
        camera_id: str,
        baseline_review_id: int,
        evaluation_hours: float,
        applied_changes: list[dict[str, Any]],
        baseline_result: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert into camera_intelligence_evaluations (
                    camera_id, baseline_review_id, evaluation_hours,
                    applied_changes_json, baseline_result_json, applied_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    int(baseline_review_id),
                    max(24.0, min(float(evaluation_hours), 168.0)),
                    json.dumps(applied_changes, separators=(",", ":"), allow_nan=False),
                    json.dumps(baseline_result, separators=(",", ":"), allow_nan=False),
                    now,
                    now,
                ),
            )
            evaluation_id = int(cursor.lastrowid)
        return self.get_camera_intelligence_evaluation(evaluation_id) or {}

    def get_camera_intelligence_evaluation(
        self,
        evaluation_id: int,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from camera_intelligence_evaluations where id = ?",
                (int(evaluation_id),),
            ).fetchone()
        return self._camera_intelligence_evaluation_row(row)

    def latest_camera_intelligence_evaluation(
        self,
        camera_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                select * from camera_intelligence_evaluations
                where camera_id = ?
                order by applied_at desc, id desc limit 1
                """,
                (camera_id,),
            ).fetchone()
        return self._camera_intelligence_evaluation_row(row)

    def start_camera_intelligence_followup(
        self,
        evaluation_id: int,
        followup_review_id: int,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                update camera_intelligence_evaluations
                set status = 'reviewing', followup_review_id = ?, error = '', updated_at = ?
                where id = ? and status = 'collecting'
                """,
                (int(followup_review_id), now, int(evaluation_id)),
            )
            if cursor.rowcount != 1:
                raise ValueError("effectiveness follow-up is already running or complete")
        return self.get_camera_intelligence_evaluation(evaluation_id) or {}

    def complete_camera_intelligence_evaluation(
        self,
        evaluation_id: int,
        *,
        followup_result: dict[str, Any],
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                update camera_intelligence_evaluations
                set status = 'completed', followup_result_json = ?, comparison_json = ?,
                    error = '', updated_at = ?, completed_at = ?
                where id = ? and status = 'reviewing'
                """,
                (
                    json.dumps(followup_result, separators=(",", ":"), allow_nan=False),
                    json.dumps(comparison, separators=(",", ":"), allow_nan=False),
                    now,
                    now,
                    int(evaluation_id),
                ),
            )
        return self.get_camera_intelligence_evaluation(evaluation_id) or {}

    def reset_camera_intelligence_followup(
        self,
        evaluation_id: int,
        error: str,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                update camera_intelligence_evaluations
                set status = 'collecting', followup_review_id = null,
                    error = ?, updated_at = ?
                where id = ? and status = 'reviewing'
                """,
                (str(error), now, int(evaluation_id)),
            )
        return self.get_camera_intelligence_evaluation(evaluation_id) or {}

    def motion_effectiveness(self, *, days: float = 7.0) -> dict[str, Any]:
        """Summarize durable motion decisions without conflating visual filters and deduplication."""
        bounded_days = min(90.0, max(1.0 / 24.0, float(days)))
        since = (datetime.now(timezone.utc) - timedelta(days=bounded_days)).isoformat()
        with self._connect() as conn:
            event_rows = conn.execute(
                """
                select camera_id, objects_json
                from events
                where kind = 'motion' and created_at >= ?
                  and objects_json like '%motion_qualification%'
                """,
                (since,),
            ).fetchall()
            audit_rows = conn.execute(
                """
                select camera_id, mode, reason, event_id, object_detected, features_json, category
                from motion_audits
                where created_at >= ?
                """,
                (since,),
            ).fetchall()

        summaries: dict[tuple[str, str], dict[str, Any]] = {}

        def summary_for(camera_id: str, mode: str) -> dict[str, Any]:
            return summaries.setdefault((camera_id, mode), {
                "allowed_events": 0,
                "object_events": 0,
                "no_object_events": 0,
                "borderline_rescued": 0,
                "suppression_verification_checks": 0,
                "suppression_verification_rescues": 0,
                "visual_filtered": 0,
                "state_deduplicated": 0,
                "unreviewed_visual_filters": 0,
                "visual_backup_attempts": 0,
                "visual_backup_objects": 0,
                "visual_backup_no_object": 0,
                "visual_backup_incomplete": 0,
                "visual_backup_not_ready": 0,
            })

        for row in event_rows:
            raw_objects = str(row["objects_json"] or "[]")
            qualification = self._qualification_from_objects(raw_objects)
            if not qualification:
                continue
            mode = str(qualification.get("mode") or "unknown")
            summary = summary_for(str(row["camera_id"]), mode)
            try:
                objects = json.loads(raw_objects)
            except (json.JSONDecodeError, TypeError):
                objects = []
            object_detected = bool(
                isinstance(objects, list)
                and any(
                    isinstance(item, dict)
                    and item.get("label")
                    and item.get("incident_eligible") is not False
                    for item in objects
                )
            )
            summary["allowed_events"] += 1
            summary["object_events" if object_detected else "no_object_events"] += 1
            if qualification.get("trigger_source") == "visual_backup":
                summary["visual_backup_attempts"] += 1
                summary["visual_backup_objects"] += int(object_detected)
            summary["borderline_rescued"] += int(
                bool(qualification.get("borderline_candidate"))
            )
            summary["suppression_verification_checks"] += int(
                bool(qualification.get("suppression_verification_candidate"))
            )
            summary["suppression_verification_rescues"] += int(
                bool(qualification.get("suppression_verification_rescued"))
            )

        for row in audit_rows:
            if row["event_id"] is not None:
                continue
            summary = summary_for(str(row["camera_id"]), str(row["mode"] or "unknown"))
            if str(row["category"] or "qualification") == "visual_backup":
                if str(row["reason"] or "") == "startup_not_ready":
                    summary["visual_backup_not_ready"] += 1
                    continue
                summary["visual_backup_attempts"] += 1
                if row["object_detected"] is None:
                    summary["visual_backup_incomplete"] += 1
                elif bool(row["object_detected"]):
                    summary["visual_backup_objects"] += 1
                else:
                    summary["visual_backup_no_object"] += 1
                continue
            reason = str(row["reason"] or "")
            if reason.startswith("event_state_"):
                summary["state_deduplicated"] += 1
            else:
                summary["visual_filtered"] += 1
                try:
                    features = json.loads(str(row["features_json"] or "{}"))
                except (json.JSONDecodeError, TypeError):
                    features = {}
                summary["suppression_verification_checks"] += int(
                    bool(features.get("suppression_verification"))
                )
                summary["unreviewed_visual_filters"] += int(
                    row["object_detected"] is None
                )

        by_camera: dict[str, dict[str, dict[str, Any]]] = {}
        for (camera_id, mode), summary in summaries.items():
            decisions = (
                summary["allowed_events"]
                + summary["visual_filtered"]
                + summary["state_deduplicated"]
            )
            visual_opportunities = summary["allowed_events"] + summary["visual_filtered"]
            summary.update({
                "total_decisions": decisions,
                "visual_rejection_rate": round(
                    summary["visual_filtered"] / max(1, visual_opportunities),
                    4,
                ),
                "object_yield_rate": round(
                    summary["object_events"] / max(1, summary["allowed_events"]),
                    4,
                ),
            })
            by_camera.setdefault(camera_id, {})[mode] = summary
        return {
            "days": bounded_days,
            "since": since,
            "by_camera": by_camera,
        }

    def get_motion_audit(self, audit_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from motion_audits where id = ?",
                (int(audit_id),),
            ).fetchone()
        return dict(row) if row is not None else None

    def motion_audits_for_related_events(self, event_ids: list[int]) -> list[dict[str, Any]]:
        unique_ids = sorted({int(event_id) for event_id in event_ids if int(event_id) > 0})
        if not unique_ids:
            return []
        audits: list[dict[str, Any]] = []
        with self._connect() as conn:
            for offset in range(0, len(unique_ids), 500):
                chunk = unique_ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    select * from motion_audits
                    where related_event_id in ({placeholders})
                    order by created_at asc, id asc
                    """,
                    chunk,
                ).fetchall()
                audits.extend(dict(row) for row in rows)
        return audits

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 10000))
        with self._connect() as conn:
            rows = conn.execute(
                "select * from events order by id desc limit ?",
                (bounded_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_compact(
        self,
        limit: int = 500,
        before_created_at: str | None = None,
        before_id: int | None = None,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 10000))
        with self._connect() as conn:
            if before_created_at is None or before_id is None:
                rows = conn.execute(
                    f"select {self.COMPACT_COLUMNS} from events order by created_at desc, id desc limit ?",
                    (bounded_limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    select {self.COMPACT_COLUMNS} from events
                    where created_at < ? or (created_at = ? and id < ?)
                    order by created_at desc, id desc
                    limit ?
                    """,
                    (before_created_at, before_created_at, int(before_id), bounded_limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def between_compact(self, start_at: str, end_at: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select {self.COMPACT_COLUMNS} from events
                where created_at >= ? and created_at < ?
                order by created_at desc, id desc
                """,
                (start_at, end_at),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_many(self, event_ids: list[int]) -> list[dict[str, Any]]:
        unique_ids = sorted({int(event_id) for event_id in event_ids if int(event_id) > 0})
        if not unique_ids:
            return []
        events: list[dict[str, Any]] = []
        with self._connect() as conn:
            for offset in range(0, len(unique_ids), 500):
                chunk = unique_ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"select * from events where id in ({placeholders})",
                    chunk,
                ).fetchall()
                events.extend(dict(row) for row in rows)
        return events

    def between(self, start_at: str, end_at: str, limit: int = 50000) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select * from events
                where created_at >= ? and created_at < ?
                order by created_at desc, id desc
                limit ?
                """,
                (start_at, end_at, max(1, min(int(limit), 200000))),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, event_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from events where id = ?",
                (event_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def update_objects(self, event_id: int, objects_json: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "update events set objects_json = ? where id = ?",
                (objects_json, event_id),
            )
            row = conn.execute(
                "select * from events where id = ?",
                (event_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def replace_detected_objects(
        self,
        event_id: int,
        detected_objects_json: str,
    ) -> dict[str, Any] | None:
        """Replace detections while preserving metadata added by concurrent workers."""
        try:
            detected_objects = json.loads(detected_objects_json or "[]")
        except (TypeError, ValueError):
            detected_objects = []
        if not isinstance(detected_objects, list):
            detected_objects = []
        detected_objects = [item for item in detected_objects if isinstance(item, dict)]
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "select objects_json from events where id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                existing = json.loads(str(row["objects_json"] or "[]"))
            except (TypeError, ValueError):
                existing = []
            preserved = [
                item
                for item in existing
                if isinstance(item, dict)
                and item.get("status") in {"motion_qualification", "object_tracking"}
            ] if isinstance(existing, list) else []
            objects_json = json.dumps([*detected_objects, *preserved], separators=(",", ":"))
            conn.execute(
                "update events set objects_json = ? where id = ?",
                (objects_json, event_id),
            )
            updated = conn.execute(
                "select * from events where id = ?",
                (event_id,),
            ).fetchone()
        return dict(updated) if updated is not None else None

    def refine_event_evidence(
        self,
        event_id: int,
        *,
        snapshot_path: str,
        recording_path: str,
        objects_json: str,
    ) -> dict[str, Any] | None:
        """Atomically replace delayed evidence without losing tracking state."""
        try:
            replacement = json.loads(objects_json or "[]")
        except (TypeError, ValueError):
            replacement = []
        if not isinstance(replacement, list):
            replacement = []
        replacement = [item for item in replacement if isinstance(item, dict)]
        portable_snapshot = portable_media_path(self.storage_dir, snapshot_path)
        portable_recording = portable_media_path(self.storage_dir, recording_path)
        replaced_snapshot = ""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "select objects_json, snapshot_path, recording_path from events where id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            replaced_snapshot = str(row["snapshot_path"] or "")
            try:
                existing = json.loads(str(row["objects_json"] or "[]"))
            except (TypeError, ValueError):
                existing = []
            preserved = [
                item
                for item in existing
                if isinstance(item, dict) and item.get("status") == "object_tracking"
            ] if isinstance(existing, list) else []
            merged_json = json.dumps([*replacement, *preserved], separators=(",", ":"))
            conn.execute(
                """
                update events
                set snapshot_path = ?, recording_path = ?, objects_json = ?
                where id = ?
                """,
                (
                    portable_snapshot or str(row["snapshot_path"] or ""),
                    portable_recording or str(row["recording_path"] or ""),
                    merged_json,
                    event_id,
                ),
            )
            updated = conn.execute(
                "select * from events where id = ?",
                (event_id,),
            ).fetchone()
        if replaced_snapshot and replaced_snapshot != portable_snapshot:
            self._delete_snapshot_if_unreferenced(replaced_snapshot)
        return dict(updated) if updated is not None else None

    def _delete_snapshot_if_unreferenced(self, raw_path: str) -> None:
        """Remove a replaced snapshot only after every durable reference moved."""
        portable = portable_media_path(self.storage_dir, raw_path)
        if not portable:
            return
        with self._lock, self._connect() as conn:
            referenced = bool(conn.execute(
                """
                select exists(select 1 from events where snapshot_path = ?)
                    or exists(select 1 from motion_audits where snapshot_path = ?)
                """,
                (portable, portable),
            ).fetchone()[0])
            if not referenced:
                has_faces = conn.execute(
                    "select 1 from sqlite_master where type = 'table' and name = 'face_observations'"
                ).fetchone() is not None
                referenced = bool(
                    has_faces
                    and conn.execute(
                        "select 1 from face_observations where snapshot_path = ? limit 1",
                        (portable,),
                    ).fetchone() is not None
                )
        if referenced:
            return
        try:
            path = event_snapshot_path(
                self.storage_dir,
                {"snapshot_path": portable},
            )
            path.relative_to((self.storage_dir / "snapshots").resolve())
            path.unlink(missing_ok=True)
        except (FileNotFoundError, PermissionError, OSError, RuntimeError, ValueError):
            return

    def update_object_tracking(
        self,
        event_id: int,
        tracking: dict[str, Any],
        tracked_objects: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Atomically replace tracking metadata without losing concurrent event data."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "select objects_json from events where id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                objects = json.loads(str(row["objects_json"] or "[]"))
            except (TypeError, ValueError):
                objects = []
            if not isinstance(objects, list):
                objects = []
            had_tracking = any(
                isinstance(item, dict) and item.get("status") == "object_tracking"
                for item in objects
            )
            objects = [
                item
                for item in objects
                if not (isinstance(item, dict) and item.get("status") == "object_tracking")
            ]
            if tracked_objects and not had_tracking:
                assignments = {
                    (
                        str(item.get("label") or ""),
                        json.dumps(item.get("box"), sort_keys=True, separators=(",", ":")),
                    ): item
                    for item in tracked_objects
                    if item.get("track_id") is not None
                }
                for item in objects:
                    if not isinstance(item, dict) or not item.get("label"):
                        continue
                    assigned = assignments.get((
                        str(item.get("label") or ""),
                        json.dumps(item.get("box"), sort_keys=True, separators=(",", ":")),
                    ))
                    if assigned is not None:
                        item["track_id"] = assigned["track_id"]
                        item["track_state"] = assigned.get("track_state")
                        item["track_observations"] = assigned.get("track_observations")
            objects.append({"status": "object_tracking", "object_tracking": tracking})
            objects_json = json.dumps(objects, separators=(",", ":"))
            conn.execute(
                "update events set objects_json = ? where id = ?",
                (objects_json, event_id),
            )
            updated = conn.execute(
                "select * from events where id = ?",
                (event_id,),
            ).fetchone()
        return dict(updated) if updated is not None else None

    def for_camera_range(
        self,
        camera_id: str,
        start_at: str,
        end_at: str,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 200000))
        with self._connect() as conn:
            rows = conn.execute(
                """
                select * from events
                where camera_id = ?
                    and created_at >= ?
                    and created_at < ?
                order by created_at asc
                limit ?
                """,
                (camera_id, start_at, end_at, bounded_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_for_camera_range(
        self,
        camera_id: str,
        start_at: str,
        end_at: str,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return the newest camera events without scanning unrelated cameras."""
        bounded_limit = max(1, min(int(limit), 200000))
        with self._connect() as conn:
            rows = conn.execute(
                """
                select * from events
                where camera_id = ?
                    and created_at >= ?
                    and created_at < ?
                order by created_at desc, id desc
                limit ?
                """,
                (camera_id, start_at, end_at, bounded_limit),
            ).fetchall()
        return [dict(row) for row in rows]

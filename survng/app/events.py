from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .durable_payload import durable_json_dumps
from .incident_utils import event_snapshot_path, portable_media_path
from .media_storage import MediaStorageRegistry


def _detection_job_occurrence(payload: dict[str, Any]) -> dict[str, Any]:
    qualification = payload.get("qualification")
    qualification = qualification if isinstance(qualification, dict) else {}
    return {
        "topic": str(payload.get("topic") or ""),
        "event_at": str(payload.get("event_at") or ""),
        "existing_event_id": payload.get("existing_event_id"),
        "detection_intent_id": str(
            qualification.get("detection_intent_id") or ""
        ),
        "require_eligible_object": bool(payload.get("require_eligible_object")),
        "require_motion_correlation": bool(
            payload.get("require_motion_correlation")
        ),
    }


def _motion_trigger_occurrence(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "topic": str(payload.get("topic") or ""),
        "event_at": str(payload.get("event_at") or ""),
        "episode_id": str(payload.get("episode_id") or ""),
        "detection_intent_id": str(payload.get("detection_intent_id") or ""),
        "lifecycle_generation": int(payload.get("lifecycle_generation") or 0),
    }


class EventStore:
    SNAPSHOT_SIZE_WRITE_BATCH = 50
    SNAPSHOT_REFERENCE_WRITE_BATCH = 50
    SNAPSHOT_SIZE_BACKFILL_CURSOR_KEY = "snapshot_size_backfill_cursor"
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

    def __init__(
        self,
        storage_dir: Path,
        database_dir: Path | None = None,
        media_storage: MediaStorageRegistry | None = None,
    ) -> None:
        self.storage_dir = storage_dir
        self.media_storage = media_storage
        self.db_path = (database_dir or storage_dir) / "survng.sqlite3"
        self.jobs_db_path = self.db_path.parent / "detection-jobs.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs_maintenance_lock = threading.Lock()
        self._last_detection_job_prune_monotonic = 0.0
        self._init_db()
        self._init_jobs_db()
        self._migrate_legacy_jobs()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 10000")
        conn.execute("pragma foreign_keys = on")
        return conn

    def _connect_jobs(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.jobs_db_path, timeout=2.0)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 2000")
        conn.execute("pragma synchronous = full")
        return conn

    def _connect_jobs_low_priority(self) -> sqlite3.Connection:
        """Open a fail-fast connection for optional bounded evidence caching."""
        conn = sqlite3.connect(self.jobs_db_path, timeout=0.025)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 25")
        conn.execute("pragma synchronous = full")
        return conn

    def _init_jobs_db(self) -> None:
        """Initialize the small local security-work ledger independently."""
        with self._connect_jobs() as conn:
            conn.execute("pragma journal_mode = wal")
            conn.execute("pragma synchronous = full")
            conn.execute(
                """
                create table if not exists detection_jobs (
                    id text primary key,
                    camera_id text not null,
                    dedupe_key text not null,
                    payload_json text not null,
                    state text not null,
                    attempts integer not null default 0,
                    available_at real not null,
                    lease_expires_at real,
                    lease_owner text not null default '',
                    event_id integer,
                    last_error text not null default '',
                    created_at text not null,
                    updated_at text not null,
                    unique(camera_id, dedupe_key)
                )
                """
            )
            conn.execute(
                "create index if not exists idx_detection_jobs_claim "
                "on detection_jobs(camera_id, state, available_at, created_at)"
            )
            detection_columns = {
                str(row["name"])
                for row in conn.execute("pragma table_info(detection_jobs)").fetchall()
            }
            if "lease_owner" not in detection_columns:
                conn.execute(
                    "alter table detection_jobs "
                    "add column lease_owner text not null default ''"
                )
            conn.execute(
                """
                create table if not exists motion_trigger_jobs (
                    id text primary key,
                    camera_id text not null,
                    payload_json text not null,
                    state text not null,
                    attempts integer not null default 0,
                    available_at real not null default 0,
                    lease_expires_at real,
                    lease_owner text not null default '',
                    last_error text not null default '',
                    created_at text not null,
                    updated_at text not null
                )
                """
            )
            columns = {
                str(row["name"])
                for row in conn.execute(
                    "pragma table_info(motion_trigger_jobs)"
                ).fetchall()
            }
            if "lease_owner" not in columns:
                conn.execute(
                    "alter table motion_trigger_jobs "
                    "add column lease_owner text not null default ''"
                )
            conn.execute(
                "create index if not exists idx_motion_trigger_jobs_claim "
                "on motion_trigger_jobs(camera_id, state, available_at, created_at)"
            )
            conn.execute(
                """
                create table if not exists route_watch_consumptions (
                    target_camera_id text not null,
                    source_event_id integer not null,
                    consumed_at text not null,
                    primary key(target_camera_id, source_event_id)
                )
                """
            )
            conn.execute(
                """
                create table if not exists ema_route_candidates (
                    camera_id text not null,
                    captured_at real not null,
                    payload_json text not null,
                    created_at text not null,
                    primary key(camera_id, captured_at)
                )
                """
            )
            conn.execute(
                "create index if not exists idx_ema_route_candidates_window "
                "on ema_route_candidates(camera_id, captured_at)"
            )

    def _migrate_legacy_jobs(self) -> None:
        """Move upgrade-era work tables out of the general event database once."""
        with self._lock, self._connect() as conn:
            tables = {
                str(row[0])
                for row in conn.execute(
                    "select name from sqlite_master where type = 'table' "
                    "and name in ('detection_jobs', 'motion_trigger_jobs')"
                ).fetchall()
            }
            if not tables:
                return
            conn.execute("attach database ? as durable_jobs", (str(self.jobs_db_path),))
            try:
                if "detection_jobs" in tables:
                    legacy_detection_columns = {
                        str(row["name"])
                        for row in conn.execute(
                            "pragma main.table_info(detection_jobs)"
                        ).fetchall()
                    }
                    detection_owner = (
                        "lease_owner"
                        if "lease_owner" in legacy_detection_columns
                        else "''"
                    )
                    conn.execute(
                        "insert or ignore into durable_jobs.detection_jobs "
                        "(id, camera_id, dedupe_key, payload_json, state, attempts, "
                        "available_at, lease_expires_at, lease_owner, event_id, last_error, "
                        "created_at, updated_at) select id, camera_id, dedupe_key, "
                        "payload_json, state, attempts, available_at, lease_expires_at, "
                        f"{detection_owner}, event_id, last_error, created_at, updated_at "
                        "from main.detection_jobs"
                    )
                if "motion_trigger_jobs" in tables:
                    legacy_columns = {
                        str(row["name"])
                        for row in conn.execute(
                            "pragma main.table_info(motion_trigger_jobs)"
                        ).fetchall()
                    }
                    owner_value = "lease_owner" if "lease_owner" in legacy_columns else "''"
                    conn.execute(
                        "insert or ignore into durable_jobs.motion_trigger_jobs "
                        "(id, camera_id, payload_json, state, attempts, available_at, "
                        "lease_expires_at, lease_owner, last_error, created_at, updated_at) "
                        "select id, camera_id, payload_json, state, attempts, available_at, "
                        f"lease_expires_at, {owner_value}, last_error, created_at, updated_at "
                        "from main.motion_trigger_jobs"
                    )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.execute("detach database durable_jobs")
            # Copy and source removal are intentionally separate commits. A
            # process or power failure can leave harmless duplicate legacy
            # rows, but can never drop the only durable copy of admitted work.
            conn.execute("begin immediate")
            try:
                if "detection_jobs" in tables:
                    conn.execute("drop table main.detection_jobs")
                if "motion_trigger_jobs" in tables:
                    conn.execute("drop table main.motion_trigger_jobs")
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
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
                    snapshot_size_bytes integer not null default 0,
                    recording_path text,
                    objects_json text not null default '[]',
                    created_at text not null
                )
                """
            )
            event_columns = {
                str(row["name"])
                for row in conn.execute("pragma table_info(events)").fetchall()
            }
            if "snapshot_size_bytes" not in event_columns:
                conn.execute(
                    "alter table events add column snapshot_size_bytes integer not null default 0"
                )
            if "detection_intent_id" not in event_columns:
                conn.execute("alter table events add column detection_intent_id text")
            conn.execute(
                "create unique index if not exists idx_events_detection_intent "
                "on events(detection_intent_id) where detection_intent_id is not null"
            )
            conn.execute(
                "create index if not exists idx_events_created_at on events(created_at desc)"
            )
            conn.execute(
                "create index if not exists idx_events_camera_created_at on events(camera_id, created_at desc)"
            )
            conn.execute(
                "create index if not exists idx_events_snapshot_retention "
                "on events(snapshot_path, created_at desc, id desc) where snapshot_path != ''"
            )
            conn.execute(
                "create table if not exists survng_metadata (key text primary key, value text not null)"
            )
            conn.execute(
                "create table if not exists snapshot_size_migration ("
                "snapshot_path text primary key, checked_at text not null)"
            )
            conn.execute(
                "create table if not exists media_deletion_claims ("
                "path text primary key, role text not null, claimed_at text not null)"
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
                create table if not exists motion_audit_pipeline_configs (
                    fingerprint text primary key,
                    configuration_json text not null,
                    created_at text not null
                )
                """
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

    def enqueue_detection_job(
        self,
        *,
        job_id: str,
        camera_id: str,
        dedupe_key: str,
        payload: dict[str, Any],
    ) -> str:
        """Durably admit mandatory delayed object discovery."""
        now_iso = datetime.now(timezone.utc).isoformat()
        payload_json = durable_json_dumps(payload, sort_keys=True)
        with self._connect_jobs() as conn:
            cursor = conn.execute(
                "insert or ignore into detection_jobs "
                "(id, camera_id, dedupe_key, payload_json, state, available_at, "
                "created_at, updated_at) values (?, ?, ?, ?, 'queued', ?, ?, ?)",
                (
                    job_id,
                    camera_id,
                    dedupe_key,
                    payload_json,
                    time.time(),
                    now_iso,
                    now_iso,
                ),
            )
            if cursor.rowcount:
                return "queued"
            existing = conn.execute(
                "select id, camera_id, dedupe_key, payload_json "
                "from detection_jobs where id = ? "
                "or (camera_id = ? and dedupe_key = ?)",
                (job_id, camera_id, dedupe_key),
            ).fetchone()
            if existing is None:
                raise RuntimeError("detection job insert collision was not recoverable")
            if (
                str(existing["id"]) != job_id
                or str(existing["camera_id"]) != camera_id
                or str(existing["dedupe_key"]) != dedupe_key
            ):
                raise RuntimeError(
                    "detection job identity collision with different occurrence"
                )
            try:
                existing_payload = json.loads(str(existing["payload_json"]))
            except (TypeError, ValueError):
                existing_payload = {}
            if _detection_job_occurrence(existing_payload) != _detection_job_occurrence(
                payload
            ):
                raise RuntimeError(
                    "detection job identity collision with different occurrence"
                )
            return "coalesced"

    def claim_detection_job(
        self,
        camera_id: str,
        *,
        lease_seconds: float = 60.0,
        lease_owner: str = "",
    ) -> dict[str, Any] | None:
        """Claim one due job, reclaiming an expired worker lease atomically."""
        now = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect_jobs() as conn:
            conn.execute("begin immediate")
            row = conn.execute(
                "select * from detection_jobs where camera_id = ? and "
                "((state = 'queued' and available_at <= ?) or "
                "(state = 'running' and (lease_expires_at <= ? or lease_owner = ?))) "
                "order by created_at, id limit 1",
                (camera_id, now, now, lease_owner),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "update detection_jobs set state = 'running', attempts = attempts + 1, "
                "lease_expires_at = ?, lease_owner = ?, updated_at = ? where id = ?",
                (
                    now + max(1.0, lease_seconds),
                    lease_owner,
                    now_iso,
                    str(row["id"]),
                ),
            )
            result = dict(row)
            result["attempts"] = int(row["attempts"]) + 1
            result["payload"] = json.loads(str(row["payload_json"]))
            return result

    def complete_detection_job(
        self,
        job_id: str,
        event_id: int | None,
        *,
        lease_owner: str = "",
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect_jobs() as conn:
            conn.execute(
                "update detection_jobs set state = 'completed', event_id = ?, "
                "lease_expires_at = null, lease_owner = '', last_error = '', updated_at = ? "
                "where id = ? and (lease_owner = ? or ? = '')",
                (event_id, now_iso, job_id, lease_owner, lease_owner),
            )

    def retry_detection_job(
        self,
        job_id: str,
        error: str,
        *,
        retry_delay_seconds: float = 2.0,
        maximum_attempts: int = 5,
        lease_owner: str = "",
    ) -> bool:
        """Release a failed lease for retry; return False once terminal."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect_jobs() as conn:
            row = conn.execute(
                "select attempts from detection_jobs where id = ? "
                "and (lease_owner = ? or ? = '')",
                (job_id, lease_owner, lease_owner),
            ).fetchone()
            if row is None:
                return False
            retry = int(row["attempts"]) < maximum_attempts
            conn.execute(
                "update detection_jobs set state = ?, available_at = ?, "
                "lease_expires_at = null, lease_owner = '', last_error = ?, updated_at = ? "
                "where id = ? and (lease_owner = ? or ? = '')",
                (
                    "queued" if retry else "failed",
                    time.time() + max(0.0, retry_delay_seconds),
                    str(error)[:1000],
                    now_iso,
                    job_id,
                    lease_owner,
                    lease_owner,
                ),
            )
            return retry

    def detection_job_status(self, camera_id: str) -> dict[str, int]:
        with self._connect_jobs() as conn:
            rows = conn.execute(
                "select state, count(*) as count from detection_jobs "
                "where camera_id = ? group by state",
                (camera_id,),
            ).fetchall()
        return {str(row["state"]): int(row["count"]) for row in rows}

    def prune_detection_jobs(
        self,
        *,
        retention_seconds: float = 7 * 24 * 60 * 60,
        limit: int = 250,
        minimum_interval_seconds: float = 60 * 60,
        force: bool = False,
    ) -> int:
        """Bound terminal security-job history without touching active work."""
        now_monotonic = time.monotonic()
        with self._jobs_maintenance_lock:
            if (
                not force
                and now_monotonic - self._last_detection_job_prune_monotonic
                < max(1.0, float(minimum_interval_seconds))
            ):
                return 0
            self._last_detection_job_prune_monotonic = now_monotonic
            cutoff = (
                datetime.now(timezone.utc)
                - timedelta(seconds=max(60.0, float(retention_seconds)))
            ).isoformat()
            with self._connect_jobs() as conn:
                rows = conn.execute(
                    "select id from detection_jobs "
                    "where state in ('completed','failed') and updated_at < ? "
                    "order by updated_at asc limit ?",
                    (cutoff, max(1, min(int(limit), 1000))),
                ).fetchall()
                if not rows:
                    return 0
                conn.executemany(
                    "delete from detection_jobs "
                    "where id = ? and state in ('completed','failed')",
                    [(str(row["id"]),) for row in rows],
                )
                return len(rows)

    def enqueue_motion_trigger(
        self,
        *,
        job_id: str,
        camera_id: str,
        payload: dict[str, Any],
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        payload_json = durable_json_dumps(payload, sort_keys=True)
        with self._connect_jobs() as conn:
            cursor = conn.execute(
                "insert or ignore into motion_trigger_jobs "
                "(id, camera_id, payload_json, state, available_at, created_at, updated_at) "
                "values (?, ?, ?, 'queued', ?, ?, ?)",
                (job_id, camera_id, payload_json, time.time(), now, now),
            )
            if cursor.rowcount:
                return True
            existing = conn.execute(
                "select camera_id, payload_json from motion_trigger_jobs where id = ?",
                (job_id,),
            ).fetchone()
            if existing is None:
                raise RuntimeError("motion trigger collision was not recoverable")
            try:
                existing_payload = json.loads(str(existing["payload_json"]))
            except (TypeError, ValueError):
                existing_payload = {}
            if (
                str(existing["camera_id"]) != camera_id
                or _motion_trigger_occurrence(existing_payload)
                != _motion_trigger_occurrence(payload)
            ):
                raise RuntimeError(
                    "motion trigger identity collision with different occurrence"
                )
            return False

    def claim_motion_trigger(
        self,
        camera_id: str,
        job_id: str | None = None,
        *,
        lease_seconds: float = 60.0,
        lease_owner: str = "",
    ) -> dict[str, Any] | None:
        now_epoch = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect_jobs() as conn:
            conn.execute("begin immediate")
            if job_id:
                row = conn.execute(
                    "select * from motion_trigger_jobs where id = ? and camera_id = ? "
                    "and ((state = 'queued' and available_at <= ?) or "
                    "(state = 'running' and (lease_expires_at <= ? or lease_owner = ?)))",
                    (job_id, camera_id, now_epoch, now_epoch, lease_owner),
                ).fetchone()
            else:
                row = conn.execute(
                    "select * from motion_trigger_jobs where camera_id = ? and "
                    "((state = 'queued' and available_at <= ?) or "
                    "(state = 'running' and lease_expires_at <= ?)) "
                    "order by created_at, id limit 1",
                    (camera_id, now_epoch, now_epoch),
                ).fetchone()
            if row is None:
                return None
            conn.execute(
                "update motion_trigger_jobs set state = 'running', attempts = attempts + 1, "
                "lease_expires_at = ?, lease_owner = ?, updated_at = ? where id = ?",
                (now_epoch + lease_seconds, lease_owner, now_iso, str(row["id"])),
            )
            return {**dict(row), "payload": json.loads(str(row["payload_json"]))}

    def complete_motion_trigger(self, job_id: str, *, lease_owner: str = "") -> None:
        with self._connect_jobs() as conn:
            conn.execute(
                "delete from motion_trigger_jobs where id = ? "
                "and (lease_owner = ? or ? = '')",
                (job_id, lease_owner, lease_owner),
            )

    def release_motion_trigger(self, job_id: str, *, lease_owner: str = "") -> None:
        """Return a graceful-shutdown lease without disturbing another owner."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect_jobs() as conn:
            conn.execute(
                "update motion_trigger_jobs set state = 'queued', "
                "lease_expires_at = null, lease_owner = '', updated_at = ? "
                "where id = ? and state = 'running' "
                "and (lease_owner = ? or ? = '')",
                (now, job_id, lease_owner, lease_owner),
            )

    def fail_motion_trigger(
        self,
        job_id: str,
        error: str,
        *,
        maximum_attempts: int = 5,
        lease_owner: str = "",
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect_jobs() as conn:
            row = conn.execute(
                "select attempts from motion_trigger_jobs where id = ? "
                "and (lease_owner = ? or ? = '')",
                (job_id, lease_owner, lease_owner),
            ).fetchone()
            if row is None:
                return False
            retry = int(row["attempts"]) < maximum_attempts
            conn.execute(
                "update motion_trigger_jobs set state = ?, available_at = ?, "
                "lease_expires_at = null, lease_owner = '', last_error = ?, updated_at = ? "
                "where id = ? and (lease_owner = ? or ? = '')",
                (
                    "queued" if retry else "failed",
                    time.time() + min(30.0, max(1, int(row["attempts"])) * 2.0),
                    str(error)[:1000],
                    now,
                    job_id,
                    lease_owner,
                    lease_owner,
                ),
            )
            return retry

    def motion_trigger_status(self, camera_id: str) -> dict[str, int]:
        with self._connect_jobs() as conn:
            rows = conn.execute(
                "select state, count(*) as count from motion_trigger_jobs "
                "where camera_id = ? group by state",
                (camera_id,),
            ).fetchall()
        return {str(row["state"]): int(row["count"]) for row in rows}

    def mark_route_watch_consumed(
        self,
        target_camera_id: str,
        source_event_id: int,
    ) -> None:
        """Persist route admission so restart recovery cannot replay it."""
        with self._connect_jobs() as conn:
            conn.execute(
                "insert or ignore into route_watch_consumptions "
                "(target_camera_id, source_event_id, consumed_at) values (?, ?, ?)",
                (
                    str(target_camera_id),
                    int(source_event_id),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            cutoff = (
                datetime.now(timezone.utc) - timedelta(hours=24)
            ).isoformat()
            conn.execute(
                "delete from route_watch_consumptions where consumed_at < ?",
                (cutoff,),
            )

    def route_watch_consumed(
        self,
        target_camera_id: str,
        source_event_id: int,
    ) -> bool:
        with self._connect_jobs() as conn:
            row = conn.execute(
                "select 1 from route_watch_consumptions "
                "where target_camera_id = ? and source_event_id = ?",
                (str(target_camera_id), int(source_event_id)),
            ).fetchone()
        return row is not None

    def record_ema_route_candidate(
        self,
        camera_id: str,
        captured_at: float,
        payload: dict[str, Any],
        *,
        retention_seconds: float = 600.0,
    ) -> None:
        """Retain compact accepted EMA evidence across a short restart gap."""
        payload_json = durable_json_dumps(payload, sort_keys=True)
        with self._connect_jobs_low_priority() as conn:
            conn.execute(
                "insert or replace into ema_route_candidates "
                "(camera_id, captured_at, payload_json, created_at) values (?, ?, ?, ?)",
                (
                    str(camera_id),
                    float(captured_at),
                    payload_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.execute(
                "delete from ema_route_candidates where camera_id = ? and captured_at < ?",
                (
                    str(camera_id),
                    float(captured_at) - max(300.0, float(retention_seconds)),
                ),
            )

    def ema_route_candidates_between(
        self,
        camera_id: str,
        start_at: float,
        end_at: float,
        *,
        limit: int = 4096,
    ) -> list[tuple[float, dict[str, Any]]]:
        with self._connect_jobs() as conn:
            rows = conn.execute(
                "select captured_at, payload_json from ema_route_candidates "
                "where camera_id = ? and captured_at >= ? and captured_at <= ? "
                "order by captured_at desc limit ?",
                (
                    str(camera_id),
                    float(start_at),
                    float(end_at),
                    max(1, min(int(limit), 4096)),
                ),
            ).fetchall()
        candidates: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                candidates.append((float(row["captured_at"]), payload))
        return candidates

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
        completed_at = now if status in {"completed", "failed", "interrupted", "cancelled"} else None
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
            # Protection is a string-key lookup against the recording index.
            # Lexical normalization avoids an NFS metadata round trip for every
            # retained incident while remaining harmless for an out-of-pool
            # value: protection can only prevent deletion, never authorize it.
            protected.add(os.path.normpath(os.path.abspath(str(path))))
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
        detection_intent_id: str | None = None,
    ) -> dict[str, Any]:
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()
        snapshot_path = portable_media_path(self.storage_dir, snapshot_path)
        snapshot_size_bytes = self._snapshot_file_size(snapshot_path)
        recording_path = portable_media_path(self.storage_dir, recording_path)
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert or ignore into events (
                    camera_id, kind, topic, message, snapshot_path, snapshot_size_bytes,
                    recording_path, objects_json, created_at, detection_intent_id
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    kind,
                    topic,
                    message,
                    snapshot_path,
                    snapshot_size_bytes,
                    recording_path,
                    objects_json,
                    created_at,
                    detection_intent_id,
                ),
            )
            created = bool(cursor.rowcount)
            if created:
                event_id = cursor.lastrowid
            elif detection_intent_id:
                row = conn.execute(
                    "select id, camera_id, kind, topic, message, created_at "
                    "from events where detection_intent_id = ?",
                    (detection_intent_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("detection intent event insert was not recoverable")
                if (
                    str(row["camera_id"]) != camera_id
                    or str(row["kind"]) != kind
                    or str(row["topic"]) != topic
                    or str(row["message"]) != message
                    or str(row["created_at"]) != created_at
                ):
                    raise RuntimeError(
                        "detection intent identity collision with different occurrence"
                    )
                event_id = int(row["id"])
            else:
                raise RuntimeError("event insert failed")
        return {
            "id": event_id,
            "camera_id": camera_id,
            "kind": kind,
            "topic": topic,
            "message": message,
            "snapshot_path": snapshot_path,
            "snapshot_size_bytes": snapshot_size_bytes,
            "recording_path": recording_path,
            "objects_json": objects_json,
            "created_at": created_at,
            "detection_intent_id": detection_intent_id,
            "created": created,
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
        if normalized_category not in {
            "qualification",
            "visual_backup",
            "active_followup",
        }:
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
        compact_features, pipeline_configurations = self._compact_audit_features(
            features or {}
        )
        features_json = json.dumps(
            compact_features,
            separators=(",", ":"),
            allow_nan=False,
        )
        replaced_snapshot = ""
        persisted_snapshot = snapshot_path
        with self._lock, self._connect() as conn:
            for fingerprint, configuration_json in pipeline_configurations.items():
                conn.execute(
                    """
                    insert or ignore into motion_audit_pipeline_configs (
                        fingerprint, configuration_json, created_at
                    ) values (?, ?, ?)
                    """,
                    (fingerprint, configuration_json, created_at),
                )
            audit_id: int | None = None
            if (
                reason in {"event_state_active", "event_state_cooldown"}
                and normalized_related_event_id is not None
                and event_id is None
            ):
                existing = conn.execute(
                    """
                    select id, trigger_count, features_json, snapshot_path
                    from motion_audits
                    where event_id is null and related_event_id = ?
                      and camera_id = ? and mode = ? and sensitivity = ?
                      and reason = ? and category = ?
                    order by id asc limit 1
                    """,
                    (
                        normalized_related_event_id,
                        camera_id,
                        mode,
                        sensitivity,
                        reason,
                        normalized_category,
                    ),
                ).fetchone()
                if existing is not None:
                    audit_id = int(existing["id"])
                    replaced_snapshot = str(existing["snapshot_path"] or "")
                    persisted_snapshot = snapshot_path or replaced_snapshot
                    compact_features["episode_observation_count"] = (
                        int(existing["trigger_count"] or 1) + normalized_trigger_count
                    )
                    compact_features["episode_last_observed_at"] = created_at
                    features_json = json.dumps(
                        compact_features,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    conn.execute(
                        """
                        update motion_audits
                        set snapshot_path = ?, created_at = ?, score = max(score, ?),
                            threshold = ?, trigger_count = trigger_count + ?,
                            features_json = ?
                        where id = ?
                        """,
                        (
                            persisted_snapshot,
                            created_at,
                            normalized_score,
                            normalized_threshold,
                            normalized_trigger_count,
                            features_json,
                            audit_id,
                        ),
                    )
            if audit_id is None and normalized_decision_id:
                existing = conn.execute(
                    "select id, snapshot_path from motion_audits where decision_id = ?",
                    (normalized_decision_id,),
                ).fetchone()
                if existing is not None:
                    audit_id = int(existing["id"])
                    replaced_snapshot = str(existing["snapshot_path"] or "")
                    persisted_snapshot = snapshot_path or replaced_snapshot
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
                            persisted_snapshot,
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
            elif audit_id is None and event_id is None:
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
        if replaced_snapshot and replaced_snapshot != persisted_snapshot:
            self._delete_snapshot_if_unreferenced(replaced_snapshot)
        return self.get_motion_audit(audit_id) or {}

    @staticmethod
    def _compact_audit_features(
        features: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        compact = copy.deepcopy(features)
        configurations: dict[str, str] = {}
        telemetry = compact.get("pipeline_telemetry")
        graphs = telemetry.get("graphs") if isinstance(telemetry, dict) else None
        if isinstance(graphs, dict):
            for graph in graphs.values():
                if not isinstance(graph, dict) or "configuration" not in graph:
                    continue
                configuration = graph.pop("configuration")
                serialized = json.dumps(
                    configuration,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
                graph["configuration_fingerprint"] = fingerprint
                configurations[fingerprint] = serialized
        return compact, configurations

    @staticmethod
    def _hydrate_audit_rows(
        conn: sqlite3.Connection,
        rows: list[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        decoded: list[tuple[dict[str, Any], dict[str, Any]]] = []
        fingerprints: set[str] = set()
        for row in rows:
            payload = dict(row)
            try:
                features = json.loads(str(payload.get("features_json") or "{}"))
            except (json.JSONDecodeError, TypeError):
                features = {}
            telemetry = features.get("pipeline_telemetry") if isinstance(features, dict) else None
            graphs = telemetry.get("graphs") if isinstance(telemetry, dict) else None
            if isinstance(graphs, dict):
                for graph in graphs.values():
                    if not isinstance(graph, dict):
                        continue
                    fingerprint = graph.get("configuration_fingerprint")
                    if isinstance(fingerprint, str) and fingerprint:
                        fingerprints.add(fingerprint)
            decoded.append((payload, features))

        configurations: dict[str, Any] = {}
        if fingerprints:
            placeholders = ",".join("?" for _ in fingerprints)
            config_rows = conn.execute(
                f"select fingerprint, configuration_json "
                f"from motion_audit_pipeline_configs where fingerprint in ({placeholders})",
                sorted(fingerprints),
            ).fetchall()
            for config_row in config_rows:
                try:
                    configurations[str(config_row["fingerprint"])] = json.loads(
                        str(config_row["configuration_json"])
                    )
                except (json.JSONDecodeError, TypeError):
                    continue

        hydrated: list[dict[str, Any]] = []
        for payload, features in decoded:
            telemetry = features.get("pipeline_telemetry") if isinstance(features, dict) else None
            graphs = telemetry.get("graphs") if isinstance(telemetry, dict) else None
            if isinstance(graphs, dict):
                for graph in graphs.values():
                    if not isinstance(graph, dict):
                        continue
                    configuration = configurations.get(graph.get("configuration_fingerprint"))
                    if configuration is not None:
                        graph["configuration"] = configuration
            payload["features_json"] = json.dumps(
                features,
                separators=(",", ":"),
                allow_nan=False,
            )
            hydrated.append(payload)
        return hydrated

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
            hydrated = self._hydrate_audit_rows(conn, list(rows))
        return hydrated, total

    def motion_audits_page_between(
        self,
        start_at: str,
        end_at: str,
        *,
        limit: int = 500,
        before_created_at: str | None = None,
        before_id: int | None = None,
        camera_ids: tuple[str, ...] = (),
        require_snapshot: bool = True,
        exclude_confirmed_objects: bool = True,
    ) -> list[dict[str, Any]]:
        """Return stable, newest-first motion-audit training candidates."""
        bounded_limit = max(1, min(int(limit), 5000))
        normalized_cameras = tuple(dict.fromkeys(
            str(camera_id).strip()
            for camera_id in camera_ids
            if str(camera_id).strip()
        ))
        clauses = [
            "created_at >= ?",
            "created_at < ?",
            "reason not in ('event_state_active', 'event_state_cooldown')",
        ]
        parameters: list[Any] = [start_at, end_at]
        if before_created_at is not None and before_id is not None:
            clauses.append("(created_at < ? or (created_at = ? and id < ?))")
            parameters.extend([before_created_at, before_created_at, int(before_id)])
        if normalized_cameras:
            placeholders = ",".join("?" for _ in normalized_cameras)
            clauses.append(f"camera_id in ({placeholders})")
            parameters.extend(normalized_cameras)
        if require_snapshot:
            clauses.append("snapshot_path != ''")
        if exclude_confirmed_objects:
            clauses.append("object_detected is not 1")
        parameters.append(bounded_limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select * from motion_audits
                where {' and '.join(clauses)}
                order by created_at desc, id desc
                limit ?
                """,
                parameters,
            ).fetchall()
            return self._hydrate_audit_rows(conn, list(rows))

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
                select camera_id, mode, reason, event_id, object_detected,
                       trigger_count, features_json, category
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
                "visual_backup_below_threshold": 0,
                "active_followup_attempts": 0,
                "active_followup_objects": 0,
                "active_followup_no_object": 0,
                "active_followup_incomplete": 0,
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
            category = str(row["category"] or "qualification")
            if category == "visual_backup":
                visual_reason = str(row["reason"] or "")
                if visual_reason == "startup_not_ready":
                    summary["visual_backup_not_ready"] += 1
                    continue
                if visual_reason == "visual_backup_below_threshold":
                    summary["visual_backup_below_threshold"] += 1
                    continue
                summary["visual_backup_attempts"] += 1
                if row["object_detected"] is None:
                    summary["visual_backup_incomplete"] += 1
                elif bool(row["object_detected"]):
                    summary["visual_backup_objects"] += 1
                else:
                    summary["visual_backup_no_object"] += 1
                continue
            if category == "active_followup":
                summary["active_followup_attempts"] += 1
                if row["object_detected"] is None:
                    summary["active_followup_incomplete"] += 1
                elif bool(row["object_detected"]):
                    summary["active_followup_objects"] += 1
                else:
                    summary["active_followup_no_object"] += 1
                continue
            reason = str(row["reason"] or "")
            if reason.startswith("event_state_"):
                summary["state_deduplicated"] += max(
                    1, int(row["trigger_count"] or 1)
                )
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
            hydrated = self._hydrate_audit_rows(
                conn,
                [row] if row is not None else [],
            )
        return hydrated[0] if hydrated else None

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
                audits.extend(self._hydrate_audit_rows(conn, list(rows)))
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

    def page_between(
        self,
        start_at: str,
        end_at: str,
        *,
        limit: int = 500,
        before_created_at: str | None = None,
        before_id: int | None = None,
        camera_ids: tuple[str, ...] = (),
        require_snapshot: bool = False,
    ) -> list[dict[str, Any]]:
        """Return a stable newest-first event page inside a half-open time range."""
        bounded_limit = max(1, min(int(limit), 5000))
        normalized_cameras = tuple(dict.fromkeys(
            str(camera_id).strip()
            for camera_id in camera_ids
            if str(camera_id).strip()
        ))
        clauses = ["created_at >= ?", "created_at < ?"]
        parameters: list[Any] = [start_at, end_at]
        if before_created_at is not None and before_id is not None:
            clauses.append("(created_at < ? or (created_at = ? and id < ?))")
            parameters.extend([
                before_created_at,
                before_created_at,
                int(before_id),
            ])
        if normalized_cameras:
            placeholders = ",".join("?" for _ in normalized_cameras)
            clauses.append(f"camera_id in ({placeholders})")
            parameters.extend(normalized_cameras)
        if require_snapshot:
            clauses.append("snapshot_path != ''")
        parameters.append(bounded_limit)
        query = f"""
            select * from events
            where {' and '.join(clauses)}
            order by created_at desc, id desc
            limit ?
        """
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
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
        snapshot_size_bytes = self._snapshot_file_size(portable_snapshot)
        portable_recording = portable_media_path(self.storage_dir, recording_path)
        replaced_snapshot = ""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "select objects_json, snapshot_path, snapshot_size_bytes, recording_path from events where id = ?",
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
                set snapshot_path = ?, snapshot_size_bytes = ?, recording_path = ?, objects_json = ?
                where id = ?
                """,
                (
                    portable_snapshot or str(row["snapshot_path"] or ""),
                    snapshot_size_bytes if portable_snapshot else int(row["snapshot_size_bytes"] or 0),
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

    def promote_tracking_cover(
        self,
        event_id: int,
        *,
        snapshot_path: str,
        captured_at: float,
        frame_width: int,
        frame_height: int,
        tracked_objects: list[dict[str, Any]],
        cover_metrics: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Replace only presentation evidence with a better tracked frame.

        Detection/admission facts remain intact. Boxes are refreshed for the
        promoted frame, and objects not visible there are explicitly excluded
        from image annotations and crop-based indexing.
        """
        portable_snapshot = portable_media_path(self.storage_dir, snapshot_path)
        if not portable_snapshot or frame_width <= 0 or frame_height <= 0:
            return None
        snapshot_size_bytes = self._snapshot_file_size(portable_snapshot)
        candidates = {
            str(item.get("track_id")): item
            for item in tracked_objects
            if isinstance(item, dict)
            and item.get("track_id") is not None
            and isinstance(item.get("box"), dict)
        }
        if not candidates:
            self._delete_snapshot_if_unreferenced(portable_snapshot)
            return None
        replaced_snapshot = ""
        matched = 0
        updated = None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "select objects_json, snapshot_path from events where id = ?",
                (event_id,),
            ).fetchone()
            if row is not None:
                try:
                    objects = json.loads(str(row["objects_json"] or "[]"))
                except (TypeError, ValueError):
                    objects = []
                if not isinstance(objects, list):
                    objects = []
                for item in objects:
                    if not isinstance(item, dict) or not item.get("label"):
                        continue
                    candidate = candidates.get(str(item.get("track_id")))
                    if candidate is None:
                        item["snapshot_visible"] = False
                        continue
                    item["box"] = dict(candidate["box"])
                    item["detection_frame_width"] = int(frame_width)
                    item["detection_frame_height"] = int(frame_height)
                    item["snapshot_visible"] = True
                    item["snapshot_source"] = "object_tracking"
                    item["snapshot_captured_at"] = datetime.fromtimestamp(
                        captured_at,
                        timezone.utc,
                    ).isoformat()
                    item["snapshot_detection_confidence"] = float(
                        candidate.get("confidence") or 0.0
                    )
                    if item.get("snapshot_primary_subject") is True:
                        for key, value in cover_metrics.items():
                            if key.startswith("snapshot_"):
                                item[key] = value
                    matched += 1
                if matched > 0:
                    replaced_snapshot = str(row["snapshot_path"] or "")
                    conn.execute(
                        """
                        update events
                        set snapshot_path = ?, snapshot_size_bytes = ?, objects_json = ?
                        where id = ?
                        """,
                        (
                            portable_snapshot,
                            snapshot_size_bytes,
                            json.dumps(objects, separators=(",", ":")),
                            event_id,
                        ),
                    )
                    updated = conn.execute(
                        "select * from events where id = ?",
                        (event_id,),
                    ).fetchone()
        if updated is None:
            self._delete_snapshot_if_unreferenced(portable_snapshot)
            return None
        if replaced_snapshot and replaced_snapshot != portable_snapshot:
            self._delete_snapshot_if_unreferenced(replaced_snapshot)
        return dict(updated) if updated is not None else None

    def promote_refinement_cover(
        self,
        event_id: int,
        *,
        snapshot_path: str,
        recording_path: str,
        captured_at: float,
        frame_width: int,
        frame_height: int,
        cover_objects: list[dict[str, Any]],
        source: str,
        timestamp_exact: bool,
    ) -> dict[str, Any] | None:
        """Promote verified main evidence without changing admission facts.

        Fast detection, causal motion correlation, and cover selection answer
        different questions.  A main-stream object may fail to *explain* an
        EMA region (notably when main/sub geometry is untrusted) while still
        being a materially better view of the already admitted provisional
        subject.  This transaction updates presentation coordinates only.

        Matching is deliberately conservative: one provisional subject and
        one temporally confirmed same-label subject, close in time, with
        materially more subject pixels. This is compatibility evidence, not
        identity proof. Ambiguous same-label scenes remain on the original
        cover and can later be promoted by tracked identity.
        """
        if (
            frame_width <= 0
            or frame_height <= 0
            or not math.isfinite(float(captured_at))
        ):
            return None
        portable_snapshot = portable_media_path(self.storage_dir, snapshot_path)
        if not portable_snapshot:
            return None
        portable_recording = portable_media_path(self.storage_dir, recording_path)

        def valid_box(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
            box = item.get("box")
            if not isinstance(box, dict):
                return None
            try:
                x1 = float(box["x1"])
                y1 = float(box["y1"])
                x2 = float(box["x2"])
                y2 = float(box["y2"])
            except (KeyError, TypeError, ValueError, OverflowError):
                return None
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                return None
            if x2 <= x1 or y2 <= y1:
                return None
            return x1, y1, x2, y2

        candidates = [
            item
            for item in cover_objects
            if isinstance(item, dict)
            and item.get("label")
            and not item.get("auxiliary_detection")
            and item.get("temporal_consensus") is True
            and item.get("confidence_eligible") is not False
            and item.get("zone_eligible") is not False
            and valid_box(item) is not None
        ]
        if not candidates:
            return None

        replaced_snapshot = ""
        snapshot_size_bytes = self._snapshot_file_size(portable_snapshot)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "select objects_json, snapshot_path, recording_path from events where id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                objects = json.loads(str(row["objects_json"] or "[]"))
            except (TypeError, ValueError):
                objects = []
            if not isinstance(objects, list):
                return None
            provisional = [
                item
                for item in objects
                if isinstance(item, dict)
                and item.get("label")
                and item.get("incident_eligible") is not False
                and item.get("provisional_detection") is True
            ]
            if len(provisional) != 1:
                return None
            existing = provisional[0]
            existing_label = str(existing.get("label") or "").strip().lower()
            compatible = [
                item
                for item in candidates
                if str(item.get("label") or "").strip().lower() == existing_label
            ]
            if not existing_label or len(compatible) != 1:
                return None
            candidate = compatible[0]
            existing_box = valid_box(existing)
            candidate_box = valid_box(candidate)
            if existing_box is None or candidate_box is None:
                return None
            try:
                existing_width = int(existing.get("detection_frame_width") or 0)
                existing_height = int(existing.get("detection_frame_height") or 0)
                existing_captured_at = float(existing.get("frame_captured_at_epoch"))
            except (TypeError, ValueError, OverflowError):
                return None
            if (
                existing_width <= 0
                or existing_height <= 0
                or not math.isfinite(existing_captured_at)
                or abs(float(captured_at) - existing_captured_at) > 15.0
            ):
                return None
            existing_subject_pixels = (
                (existing_box[2] - existing_box[0])
                * (existing_box[3] - existing_box[1])
            )
            candidate_subject_pixels = (
                (candidate_box[2] - candidate_box[0])
                * (candidate_box[3] - candidate_box[1])
            )
            candidate_clearance = min(
                candidate_box[0] / frame_width,
                candidate_box[1] / frame_height,
                (frame_width - candidate_box[2]) / frame_width,
                (frame_height - candidate_box[3]) / frame_height,
            )
            if (
                frame_width * frame_height <= existing_width * existing_height
                or candidate_subject_pixels < max(64.0, existing_subject_pixels * 1.5)
                or candidate_clearance < 0.005
            ):
                return None

            for item in objects:
                if not isinstance(item, dict) or not item.get("label"):
                    continue
                if item is not existing:
                    item["snapshot_visible"] = False
                    continue
                item["box"] = dict(candidate["box"])
                item["detection_frame_width"] = int(frame_width)
                item["detection_frame_height"] = int(frame_height)
                item["snapshot_visible"] = True
                item["snapshot_source"] = str(source or "recorded_refinement")
                item["snapshot_captured_at"] = datetime.fromtimestamp(
                    float(captured_at),
                    timezone.utc,
                ).isoformat()
                item["snapshot_detection_confidence"] = float(
                    candidate.get("confidence") or 0.0
                )
                item["snapshot_presentation_only"] = True
                item["snapshot_timestamp_exact"] = bool(timestamp_exact)
                for key in (
                    "snapshot_quality_score",
                    "snapshot_sharpness_score",
                    "snapshot_exposure_score",
                    "snapshot_contrast_score",
                    "snapshot_edge_detail_score",
                    "snapshot_primary_subject",
                    "snapshot_edge_clearance_ratio",
                    "snapshot_subject_area_ratio",
                ):
                    if key in candidate:
                        item[key] = candidate[key]
            objects = [
                item
                for item in objects
                if not (
                    isinstance(item, dict)
                    and item.get("status") == "cover_promotion"
                )
            ]
            objects.append({
                "status": "cover_promotion",
                "cover_promotion": {
                    "source": str(source or "recorded_refinement"),
                    "captured_at": datetime.fromtimestamp(
                        float(captured_at),
                        timezone.utc,
                    ).isoformat(),
                    "timestamp_exact": bool(timestamp_exact),
                    "reason": "compatible_recorded_refinement",
                    "admission_preserved": True,
                },
            })
            replaced_snapshot = str(row["snapshot_path"] or "")
            conn.execute(
                """
                update events
                set snapshot_path = ?, snapshot_size_bytes = ?, recording_path = ?, objects_json = ?
                where id = ?
                """,
                (
                    portable_snapshot,
                    snapshot_size_bytes,
                    portable_recording or str(row["recording_path"] or ""),
                    json.dumps(objects, separators=(",", ":")),
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

    def _snapshot_file_size(self, raw_path: str) -> int:
        """Return an incident snapshot's size without allowing arbitrary paths."""
        if not raw_path:
            return 0
        try:
            path = event_snapshot_path(
                self.storage_dir,
                {"snapshot_path": raw_path},
                self.media_storage,
            )
            if self.media_storage is None:
                path.relative_to((self.storage_dir / "snapshots").resolve())
            elif self.media_storage.location_id_for(path, role="snapshots") is None:
                return 0
            return max(0, int(path.stat().st_size))
        except (FileNotFoundError, PermissionError, OSError, RuntimeError, ValueError):
            return 0

    def migrate_snapshot_sizes(
        self,
        *,
        limit: int = 250,
        write_batch_size: int | None = None,
    ) -> int:
        """Index one bounded legacy cohort outside ordinary retention planning.

        This is intentionally an explicit, low-priority migration operation:
        retention status reads must never perform media filesystem I/O or take
        an SQLite writer lock merely because an older row has no stored size.
        """
        with self._lock, self._connect() as conn:
            try:
                cursor_id = max(
                    0,
                    int(
                        self._metadata_value(
                            conn,
                            self.SNAPSHOT_SIZE_BACKFILL_CURSOR_KEY,
                        )
                        or 0
                    ),
                )
            except ValueError:
                cursor_id = 0
            rows = conn.execute(
                """
                select min(id) as cursor_id, snapshot_path from events
                where snapshot_path != '' and snapshot_size_bytes <= 0
                  and not exists (
                      select 1 from snapshot_size_migration
                      where snapshot_size_migration.snapshot_path = events.snapshot_path
                        and snapshot_size_migration.checked_at > ?
                  )
                group by snapshot_path
                having min(id) > ?
                order by cursor_id asc limit ?
                """,
                (
                    (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
                    cursor_id,
                    max(1, int(limit)),
                ),
            ).fetchall()
        if not rows:
            if cursor_id > 0:
                with self._lock, self._connect() as conn:
                    self._set_metadata_value(
                        conn,
                        self.SNAPSHOT_SIZE_BACKFILL_CURSOR_KEY,
                        "0",
                    )
            return 0
        next_cursor_id = max(int(row["cursor_id"]) for row in rows)
        updates = [
            (size, str(row["snapshot_path"]))
            for row in rows
            if (size := self._snapshot_file_size(str(row["snapshot_path"]))) > 0
        ]
        batch_size = max(
            1,
            min(
                250,
                int(write_batch_size or self.SNAPSHOT_SIZE_WRITE_BATCH),
            ),
        )
        for offset in range(0, len(updates), batch_size):
            batch = updates[offset : offset + batch_size]
            with self._lock, self._connect() as conn:
                conn.executemany(
                    "update events set snapshot_size_bytes = ? where snapshot_path = ?",
                    batch,
                )
                if offset + batch_size >= len(updates):
                    self._set_metadata_value(
                        conn,
                        self.SNAPSHOT_SIZE_BACKFILL_CURSOR_KEY,
                        str(next_cursor_id),
                    )
        checked_paths = [str(row["snapshot_path"]) for row in rows]
        if checked_paths:
            with self._lock, self._connect() as conn:
                conn.executemany(
                    "insert or replace into snapshot_size_migration "
                    "(snapshot_path, checked_at) values (?, ?)",
                    (
                        (path, datetime.now(timezone.utc).isoformat())
                        for path in checked_paths
                    ),
                )
                self._set_metadata_value(
                    conn,
                    self.SNAPSHOT_SIZE_BACKFILL_CURSOR_KEY,
                    str(next_cursor_id),
                )
        return len(updates)

    def snapshot_retention_plan(self, cutoff_epoch: float) -> dict[str, Any]:
        """Report database-indexed incident snapshot use and age expiry.

        Planning is a pure database read.  In WAL mode it can safely run beside
        incident writers, so it must not acquire the EventStore writer mutex.
        """
        cutoff = datetime.fromtimestamp(float(cutoff_epoch), timezone.utc).isoformat()
        ranked = """
            with ranked as (
                select id, camera_id, snapshot_path, snapshot_size_bytes, created_at,
                       row_number() over (
                           partition by snapshot_path order by created_at desc, id desc
                       ) as snapshot_rank
                from events where snapshot_path != ''
            )
        """
        with self._connect() as conn:
            has_faces = conn.execute(
                "select 1 from sqlite_master where type = 'table' and name = 'face_observations'"
            ).fetchone() is not None
            protected_join = (
                "left join (select distinct snapshot_path from face_observations "
                "where reference_pinned = 1) as pinned "
                "on pinned.snapshot_path = ranked.snapshot_path"
                if has_faces
                else ""
            )
            unprotected_predicate = (
                "and pinned.snapshot_path is null" if has_faces else ""
            )
            cameras = conn.execute(
                ranked
                + f"""
                select camera_id,
                       count(*) as file_count,
                       coalesce(sum(snapshot_size_bytes), 0) as bytes,
                       coalesce(sum(case when snapshot_size_bytes <= 0
                                         and sized.snapshot_path is null
                                         then 1 else 0 end), 0)
                           as unindexed_files,
                       coalesce(sum(case when created_at < ? {unprotected_predicate}
                                         then 1 else 0 end), 0) as expired_files,
                       coalesce(sum(case when created_at < ? {unprotected_predicate}
                                         then snapshot_size_bytes else 0 end), 0)
                           as expired_bytes
                from ranked
                left join snapshot_size_migration as sized
                  on sized.snapshot_path = ranked.snapshot_path
                {protected_join}
                where snapshot_rank = 1
                group by camera_id order by camera_id
                """,
                (cutoff, cutoff),
            ).fetchall()
        file_count = sum(int(row["file_count"] or 0) for row in cameras)
        total_bytes = sum(int(row["bytes"] or 0) for row in cameras)
        unindexed_files = sum(int(row["unindexed_files"] or 0) for row in cameras)
        expired_files = sum(int(row["expired_files"] or 0) for row in cameras)
        expired_bytes = sum(int(row["expired_bytes"] or 0) for row in cameras)
        return {
            "file_count": file_count,
            "bytes": total_bytes,
            "unindexed_files": unindexed_files,
            "expired_files": expired_files,
            "expired_bytes": expired_bytes,
            "per_camera": [
                {
                    "camera_id": str(row["camera_id"]),
                    "file_count": int(row["file_count"] or 0),
                    "bytes": int(row["bytes"] or 0),
                }
                for row in cameras
            ],
        }

    def apply_snapshot_retention(self, cutoff_epoch: float, limit: int) -> dict[str, Any]:
        """Delete age-expired incident images and clear every stale reference."""
        cutoff = datetime.fromtimestamp(float(cutoff_epoch), timezone.utc).isoformat()
        bounded_limit = max(1, min(2000, int(limit)))
        ranked = """
            with ranked as (
                select id, snapshot_path, snapshot_size_bytes, created_at,
                       row_number() over (
                           partition by snapshot_path order by created_at desc, id desc
                       ) as snapshot_rank
                from events where snapshot_path != ''
            )
        """
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            conn.execute(
                "delete from media_deletion_claims "
                "where role = 'snapshot' and claimed_at < ?",
                ((datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),),
            )
            has_faces = conn.execute(
                "select 1 from sqlite_master where type = 'table' and name = 'face_observations'"
            ).fetchone() is not None
            face_clause = (
                "and not exists (select 1 from face_observations "
                "where face_observations.snapshot_path = ranked.snapshot_path "
                "and face_observations.reference_pinned = 1)"
                if has_faces
                else ""
            )
            rows = conn.execute(
                ranked
                + f"""
                select snapshot_path, snapshot_size_bytes from ranked
                where snapshot_rank = 1 and created_at < ? {face_clause}
                order by created_at asc limit ?
                """,
                (cutoff, bounded_limit),
            ).fetchall()
            claimed_at = datetime.now(timezone.utc).isoformat()
            claimed_rows: list[sqlite3.Row] = []
            for row in rows:
                claimed = conn.execute(
                    "insert or ignore into media_deletion_claims "
                    "(path, role, claimed_at) values (?, 'snapshot', ?)",
                    (str(row["snapshot_path"]), claimed_at),
                )
                if claimed.rowcount:
                    claimed_rows.append(row)
            rows = claimed_rows
        removed: list[str] = []
        deleted_files = 0
        missing_files = 0
        deleted_bytes = 0
        failed_files = 0
        for row in rows:
            raw_path = str(row["snapshot_path"] or "")
            try:
                path = event_snapshot_path(
                    self.storage_dir,
                    {"snapshot_path": raw_path},
                    self.media_storage,
                )
                if self.media_storage is None:
                    path.relative_to((self.storage_dir / "snapshots").resolve())
                elif self.media_storage.location_id_for(
                    path, role="snapshots"
                ) is None:
                    raise PermissionError(
                        "snapshot is outside configured snapshot storage"
                    )
                actual_size = max(
                    int(row["snapshot_size_bytes"] or 0), int(path.stat().st_size)
                )
                path.unlink()
                deleted_files += 1
                deleted_bytes += actual_size
                removed.append(raw_path)
            except FileNotFoundError:
                missing_files += 1
                removed.append(raw_path)
            except (PermissionError, OSError, RuntimeError, ValueError):
                failed_files += 1
        for offset in range(0, len(removed), self.SNAPSHOT_REFERENCE_WRITE_BATCH):
            batch = removed[offset : offset + self.SNAPSHOT_REFERENCE_WRITE_BATCH]
            with self._lock, self._connect() as conn:
                conn.executemany(
                    "update events set snapshot_path = '', snapshot_size_bytes = 0 where snapshot_path = ?",
                    ((path,) for path in batch),
                )
                conn.executemany(
                    "update motion_audits set snapshot_path = '' where snapshot_path = ?",
                    ((path,) for path in batch),
                )
                has_faces = conn.execute(
                    "select 1 from sqlite_master where type = 'table' and name = 'face_observations'"
                ).fetchone() is not None
                if has_faces:
                    conn.executemany(
                        "update face_observations set snapshot_path = '' "
                        "where snapshot_path = ? and reference_pinned = 0",
                        ((path,) for path in batch),
                    )
        if rows:
            with self._lock, self._connect() as conn:
                conn.executemany(
                    "delete from media_deletion_claims "
                    "where path = ? and role = 'snapshot'",
                    ((str(row["snapshot_path"]),) for row in rows),
                )
        return {
            "selected_files": len(rows),
            "deleted_files": deleted_files,
            "missing_files": missing_files,
            "deleted_bytes": deleted_bytes,
            "failed_files": failed_files,
            "batch_saturated": len(rows) >= bounded_limit,
        }

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
                self.media_storage,
            )
            if self.media_storage is None:
                path.relative_to((self.storage_dir / "snapshots").resolve())
            elif self.media_storage.location_id_for(path, role="snapshots") is None:
                return
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

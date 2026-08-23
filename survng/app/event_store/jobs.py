from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from ..durable_payload import durable_json_dumps


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


def _route_motion_trigger_identity(
    job_id: str,
    payload: dict[str, Any],
) -> str | None:
    """Return the stable route key when this trigger is route-derived.

    Route delivery jobs are keyed by target/source/source-event identity, for
    example ``route:<target>:<source>:<event_id>``. Future prefixes such as
    ``route:v2:...`` remain compatible because comparison uses the full
    identity string carried by the job/intent/episode fields.
    """
    for candidate in (
        str(payload.get("detection_intent_id") or "").strip(),
        str(payload.get("episode_id") or "").strip(),
        str(job_id or "").strip(),
    ):
        if candidate.startswith("route:"):
            return candidate
    return None


def _route_motion_trigger_equivalent(
    job_id: str,
    existing_payload: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    """Whether two payloads are the same route occurrence despite capture drift.

    Repeated Gate EMA replay for one Lower-Garage source event can rebuild the
    same ``route:...`` job with a different captured ``event_at`` or worker
    ``lifecycle_generation``. Those fields are not part of the durable route
    identity, so treat them as an idempotent no-op while still rejecting
    different cameras, topics, or route identities.
    """
    route_identity = _route_motion_trigger_identity(job_id, payload)
    existing_identity = _route_motion_trigger_identity(job_id, existing_payload)
    if (
        route_identity is None
        or existing_identity is None
        or route_identity != existing_identity
    ):
        return False
    return (
        str(existing_payload.get("topic") or "") == str(payload.get("topic") or "")
        and str(existing_payload.get("episode_id") or "")
        == str(payload.get("episode_id") or "")
        and str(existing_payload.get("detection_intent_id") or "")
        == str(payload.get("detection_intent_id") or "")
    )


class EventStoreJobsMixin:
    """Durable detection-job and motion-trigger ledger on detection-jobs.sqlite3."""

    def _connect_jobs(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.jobs_db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 10000")
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
        with self._jobs_lock, self._connect_jobs() as conn:
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
        # Idle refiners poll frequently.  Do not acquire SQLite's exclusive
        # writer reservation until a due job is actually present.
        with self._jobs_lock, self._connect_jobs() as conn:
            due = conn.execute(
                "select 1 from detection_jobs where camera_id = ? and "
                "((state = 'queued' and available_at <= ?) or "
                "(state = 'running' and (lease_expires_at <= ? or lease_owner = ?))) "
                "limit 1",
                (camera_id, now, now, lease_owner),
            ).fetchone()
            if due is None:
                return None

        now = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._jobs_lock, self._connect_jobs() as conn:
            conn.execute("begin immediate")
            row = conn.execute(
                "select * from detection_jobs where camera_id = ? and "
                "((state = 'queued' and available_at <= ?) or "
                "(state = 'running' and (lease_expires_at <= ? or lease_owner = ?))) "
                # Refinement is time-sensitive evidence, not a FIFO batch.
                # After restart, processing an old backlog before a current
                # incident makes the current event unrecoverably stale.
                "order by created_at desc, id desc limit 1",
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

    def expire_stale_detection_jobs(
        self,
        camera_id: str,
        *,
        maximum_age_seconds: float,
    ) -> int:
        """Terminally mark queued evidence that can no longer be time-accurate."""
        cutoff = datetime.fromtimestamp(
            time.time() - max(0.0, float(maximum_age_seconds)),
            timezone.utc,
        ).isoformat()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._jobs_lock, self._connect_jobs() as conn:
            cursor = conn.execute(
                "update detection_jobs set state = 'failed', lease_expires_at = null, "
                "lease_owner = '', last_error = 'stale_refinement', updated_at = ? "
                "where camera_id = ? and state = 'queued' and created_at <= ?",
                (now_iso, camera_id, cutoff),
            )
            return max(0, int(cursor.rowcount))

    def complete_detection_job(
        self,
        job_id: str,
        event_id: int | None,
        *,
        lease_owner: str = "",
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._jobs_lock, self._connect_jobs() as conn:
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
        with self._jobs_lock, self._connect_jobs() as conn:
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

    def detection_job_status(self, camera_id: str) -> dict[str, int | float]:
        with self._connect_jobs() as conn:
            rows = conn.execute(
                "select state, count(*) as count from detection_jobs "
                "where camera_id = ? group by state",
                (camera_id,),
            ).fetchall()
            oldest = conn.execute(
                "select min(created_at) as created_at from detection_jobs "
                "where camera_id = ? and state in ('queued', 'running')",
                (camera_id,),
            ).fetchone()
        result: dict[str, int | float] = {
            str(row["state"]): int(row["count"])
            for row in rows
        }
        oldest_age_ms = 0.0
        created_at = oldest["created_at"] if oldest is not None else None
        if created_at:
            try:
                created = datetime.fromisoformat(str(created_at))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                oldest_age_ms = max(
                    0.0,
                    (datetime.now(timezone.utc) - created.astimezone(timezone.utc))
                    .total_seconds()
                    * 1000.0,
                )
            except (TypeError, ValueError):
                oldest_age_ms = 0.0
        result["oldest_age_ms"] = round(oldest_age_ms, 3)
        return result

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
            with self._jobs_lock, self._connect_jobs() as conn:
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
        with self._jobs_lock, self._connect_jobs() as conn:
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
                or (
                    _motion_trigger_occurrence(existing_payload)
                    != _motion_trigger_occurrence(payload)
                    and not _route_motion_trigger_equivalent(
                        job_id,
                        existing_payload,
                        payload,
                    )
                )
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
        with self._jobs_lock, self._connect_jobs() as conn:
            if job_id:
                due = conn.execute(
                    "select 1 from motion_trigger_jobs where id = ? and camera_id = ? "
                    "and ((state = 'queued' and available_at <= ?) or "
                    "(state = 'running' and (lease_expires_at <= ? or lease_owner = ?))) "
                    "limit 1",
                    (job_id, camera_id, now_epoch, now_epoch, lease_owner),
                ).fetchone()
            else:
                due = conn.execute(
                    "select 1 from motion_trigger_jobs where camera_id = ? and "
                    "((state = 'queued' and available_at <= ?) or "
                    "(state = 'running' and lease_expires_at <= ?)) limit 1",
                    (camera_id, now_epoch, now_epoch),
                ).fetchone()
            if due is None:
                return None

        now_epoch = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._jobs_lock, self._connect_jobs() as conn:
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
        with self._jobs_lock, self._connect_jobs() as conn:
            conn.execute(
                "delete from motion_trigger_jobs where id = ? "
                "and (lease_owner = ? or ? = '')",
                (job_id, lease_owner, lease_owner),
            )

    def release_motion_trigger(self, job_id: str, *, lease_owner: str = "") -> None:
        """Return a graceful-shutdown lease without disturbing another owner."""
        now = datetime.now(timezone.utc).isoformat()
        with self._jobs_lock, self._connect_jobs() as conn:
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
        with self._jobs_lock, self._connect_jobs() as conn:
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
        with self._jobs_lock, self._connect_jobs() as conn:
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

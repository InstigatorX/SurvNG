from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .incident_utils import portable_media_path


class EventStore:
    COMPACT_COLUMNS = (
        "id, camera_id, kind, snapshot_path, recording_path, objects_json, created_at"
    )
    TRACKING_COMPARISON_HISTORY_PER_CAMERA = 100
    TRACKING_COMPARISON_VERDICTS = {
        "survng_hybrid",
        "ultralytics_botsort",
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
    def _tracking_comparison_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
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
        comparison = self._tracking_comparison_row(row)
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
        return self._tracking_comparison_row(row)

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
        return [self._tracking_comparison_row(row) or {} for row in rows]

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
            if not qualification or not qualification.get("would_suppress"):
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
            inserts.append((
                int(row["id"]),
                str(row["camera_id"]),
                str(row["snapshot_path"] or ""),
                str(row["created_at"]),
                str(qualification.get("mode") or "audit"),
                str(qualification.get("sensitivity") or "balanced"),
                float(qualification.get("score") or 0.0),
                float(qualification.get("threshold") or 0.0),
                str(qualification.get("reason") or "rejected"),
                int(object_detected),
                max(1, int(qualification.get("trigger_count") or 1)),
                json.dumps(qualification.get("features") or {}, separators=(",", ":")),
            ))
        if inserts:
            conn.executemany(
                """
                insert or ignore into motion_audits (
                    event_id, camera_id, snapshot_path, created_at, mode,
                    sensitivity, score, threshold, reason, object_detected,
                    trigger_count, features_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        with self._lock, self._connect() as conn:
            audit_id: int | None = None
            if normalized_decision_id:
                existing = conn.execute(
                    "select id from motion_audits where decision_id = ?",
                    (normalized_decision_id,),
                ).fetchone()
                if existing is not None:
                    audit_id = int(existing["id"])
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
                return self._motion_ai_review_row(
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

    @staticmethod
    def _motion_ai_review_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        try:
            result = json.loads(str(payload.pop("result_json") or "{}"))
        except (json.JSONDecodeError, TypeError):
            result = {}
        payload["result"] = result if isinstance(result, dict) else {}
        return payload

    def get_motion_ai_review(self, review_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from motion_ai_reviews where id = ?",
                (int(review_id),),
            ).fetchone()
        return self._motion_ai_review_row(row)

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
        return self._motion_ai_review_row(row)

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
                select camera_id, mode, reason, event_id, object_detected, features_json
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

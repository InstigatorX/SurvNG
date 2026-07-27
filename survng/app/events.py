from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class EventStore:
    COMPACT_COLUMNS = "id, camera_id, kind, objects_json, created_at"

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
                    features_json text not null default '{}'
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
            conn.execute(
                "create index if not exists idx_motion_audits_created_at on motion_audits(created_at desc, id desc)"
            )
            conn.execute(
                "create index if not exists idx_motion_audits_camera_created_at on motion_audits(camera_id, created_at desc)"
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
            if self._metadata_value(conn, "event_storage_root") != storage_root:
                self._rebase_media_paths(conn)
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
        storage_root = self.storage_dir.resolve()
        rows = conn.execute(
            "select id, snapshot_path, recording_path from events where snapshot_path != '' or recording_path != ''"
        ).fetchall()
        updates: list[tuple[str, str, int]] = []
        for row in rows:
            snapshot_path = self._rebased_path(str(row["snapshot_path"] or ""), storage_root)
            recording_path = self._rebased_path(str(row["recording_path"] or ""), storage_root)
            if snapshot_path != str(row["snapshot_path"] or "") or recording_path != str(row["recording_path"] or ""):
                updates.append((snapshot_path, recording_path, int(row["id"])))
        if updates:
            conn.executemany(
                "update events set snapshot_path = ?, recording_path = ? where id = ?",
                updates,
            )

    @staticmethod
    def _rebased_path(raw_path: str, storage_root: Path) -> str:
        if not raw_path:
            return raw_path
        path = Path(raw_path)
        try:
            path.resolve().relative_to(storage_root)
            return raw_path
        except ValueError:
            pass
        parts = path.parts
        for directory in ("snapshots", "recordings"):
            if directory not in parts:
                continue
            candidate = storage_root.joinpath(*parts[parts.index(directory):])
            if candidate.is_file():
                return str(candidate)
        return raw_path

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
        event_id: int | None = None,
        related_event_id: int | None = None,
        decision_id: str = "",
    ) -> dict[str, Any]:
        normalized_object_detected = (
            None if object_detected is None else int(object_detected)
        )
        normalized_trigger_count = max(1, int(trigger_count))
        normalized_decision_id = str(decision_id or "").strip()
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
                            object_detected = ?, trigger_count = ?, features_json = ?
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
                      and features_json = ?
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
                        trigger_count, features_json
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    ) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        values: list[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            values.append(camera_id)
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
                    and created_at <= ?
                order by created_at asc
                limit ?
                """,
                (camera_id, start_at, end_at, bounded_limit),
            ).fetchall()
        return [dict(row) for row in rows]

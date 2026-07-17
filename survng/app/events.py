from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventStore:
    COMPACT_COLUMNS = "id, camera_id, kind, objects_json, created_at"

    def __init__(self, storage_dir: Path) -> None:
        self.db_path = storage_dir / "survng.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
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
                """
                create table if not exists motion_audits (
                    id integer primary key autoincrement,
                    event_id integer,
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
            conn.execute(
                "create index if not exists idx_motion_audits_created_at on motion_audits(created_at desc, id desc)"
            )
            conn.execute(
                "create index if not exists idx_motion_audits_camera_created_at on motion_audits(camera_id, created_at desc)"
            )
            conn.execute(
                "create unique index if not exists idx_motion_audits_event on motion_audits(event_id) where event_id is not null"
            )
            self._rebase_media_paths(conn)
            self._backfill_motion_audits(conn)

    @staticmethod
    def _qualification_from_objects(objects_json: str) -> dict[str, Any] | None:
        try:
            objects = json.loads(objects_json or "[]")
        except json.JSONDecodeError:
            return None
        return next(
            (
                item.get("motion_qualification")
                for item in objects
                if item.get("status") == "motion_qualification"
                and isinstance(item.get("motion_qualification"), dict)
            ),
            None,
        )

    def _backfill_motion_audits(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            select id, camera_id, snapshot_path, created_at, objects_json
            from events
            where objects_json like '%\"status\":\"motion_qualification\"%'
              and id not in (select event_id from motion_audits where event_id is not null)
            order by id asc
            """
        ).fetchall()
        inserts: list[tuple[Any, ...]] = []
        for row in rows:
            qualification = self._qualification_from_objects(str(row["objects_json"] or ""))
            if not qualification or not qualification.get("would_suppress"):
                continue
            try:
                objects = json.loads(str(row["objects_json"] or "[]"))
            except json.JSONDecodeError:
                objects = []
            object_detected = any(
                item.get("label") and item.get("incident_eligible") is not False
                for item in objects
            )
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
        storage_root = self.db_path.parent.resolve()
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
    ) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert into motion_audits (
                    event_id, camera_id, snapshot_path, created_at, mode,
                    sensitivity, score, threshold, reason, object_detected,
                    trigger_count, features_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    camera_id,
                    snapshot_path,
                    created_at,
                    mode,
                    sensitivity,
                    float(score),
                    float(threshold),
                    reason,
                    None if object_detected is None else int(object_detected),
                    max(1, int(trigger_count)),
                    json.dumps(features or {}, separators=(",", ":")),
                ),
            )
            audit_id = int(cursor.lastrowid)
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

    def get_motion_audit(self, audit_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from motion_audits where id = ?",
                (int(audit_id),),
            ).fetchone()
        return dict(row) if row is not None else None

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from events order by id desc limit ?",
                (limit,),
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
                (camera_id, start_at, end_at, limit),
            ).fetchall()
        return [dict(row) for row in rows]

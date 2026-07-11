from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventStore:
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

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from events order by id desc limit ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, event_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
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

"""Durable speech segment index (``speech.sqlite3``).

Speech rows live in their own database so the listener never contends for the
event store's writer lock. Audio is not stored: rows point at a time range that
playback resolves against the camera's recording.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..main_database import ClosingDatabaseConnection

LOGGER = logging.getLogger("uvicorn.error")

STATUSES = frozenset({
    "detected",       # speech heard; transcription not requested
    "queued",         # waiting for the transcriber
    "transcribing",
    "done",
    "no_text",        # transcriber returned nothing
    "low_confidence",
    "failed",
    "skipped",        # dropped by backpressure, shutdown, or unavailable transcriber
})
PENDING_STATUSES = ("queued", "transcribing")
MAX_LIST_LIMIT = 500


def _iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat()


class SpeechStore:
    SCHEMA_VERSION = 2

    def __init__(self, database_dir: Path | str) -> None:
        self.path = Path(database_dir) / "speech.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.fts_enabled = False
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=2.0, factory=ClosingDatabaseConnection)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma journal_mode=wal")
        conn.execute("pragma synchronous=normal")
        conn.execute("pragma busy_timeout=2000")
        return conn

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                create table if not exists speech_metadata (
                    key text primary key,
                    value text not null
                );
                create table if not exists speech_segments (
                    id integer primary key autoincrement,
                    camera_id text not null,
                    started_at real not null,
                    ended_at real not null,
                    peak_probability real not null default 0,
                    status text not null,
                    text text not null default '',
                    language text not null default '',
                    score real,
                    model text not null default '',
                    device text not null default '',
                    reason text not null default '',
                    created_at real not null,
                    updated_at real not null
                );
                create index if not exists speech_segments_camera_time
                    on speech_segments (camera_id, started_at);
                create index if not exists speech_segments_time
                    on speech_segments (started_at);
                create index if not exists speech_segments_status
                    on speech_segments (status);
                """
            )
            columns = {str(row[1]) for row in conn.execute("pragma table_info(speech_segments)").fetchall()}
            if "content_expired" not in columns:
                # Set when an attached transcript's recording is gone: the
                # text is removed, the timing row stays for incident counts.
                conn.execute(
                    "alter table speech_segments add column content_expired integer not null default 0"
                )
            conn.execute(
                "create index if not exists speech_segments_retention "
                "on speech_segments (content_expired, started_at)"
            )
            conn.execute(
                "insert or replace into speech_metadata (key, value) values ('schema_version', ?)",
                (str(self.SCHEMA_VERSION),),
            )
            try:
                conn.executescript(
                    """
                    create virtual table if not exists speech_segments_fts using fts5(
                        text, content='speech_segments', content_rowid='id'
                    );
                    create trigger if not exists speech_segments_ai after insert on speech_segments begin
                        insert into speech_segments_fts (rowid, text) values (new.id, new.text);
                    end;
                    create trigger if not exists speech_segments_ad after delete on speech_segments begin
                        insert into speech_segments_fts (speech_segments_fts, rowid, text)
                            values ('delete', old.id, old.text);
                    end;
                    create trigger if not exists speech_segments_au after update of text on speech_segments begin
                        insert into speech_segments_fts (speech_segments_fts, rowid, text)
                            values ('delete', old.id, old.text);
                        insert into speech_segments_fts (rowid, text) values (new.id, new.text);
                    end;
                    """
                )
                self.fts_enabled = True
            except sqlite3.OperationalError as error:
                LOGGER.warning("Speech transcript search falls back to LIKE: %s", error)

    # Writes ------------------------------------------------------------

    def insert_segment(
        self,
        *,
        camera_id: str,
        started_at: float,
        ended_at: float,
        peak_probability: float,
        status: str,
        reason: str = "",
    ) -> int:
        if status not in STATUSES:
            raise ValueError(f"unknown speech segment status: {status}")
        now = time.time()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert into speech_segments
                    (camera_id, started_at, ended_at, peak_probability, status, reason,
                     created_at, updated_at)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (camera_id, started_at, ended_at, peak_probability, status, reason, now, now),
            )
            return int(cursor.lastrowid)

    def set_status(self, segment_id: int, status: str, *, reason: str = "") -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown speech segment status: {status}")
        with self._lock, self._connect() as conn:
            conn.execute(
                "update speech_segments set status = ?, reason = ?, updated_at = ? where id = ?",
                (status, reason, time.time(), segment_id),
            )

    def set_transcript(
        self,
        segment_id: int,
        *,
        status: str,
        text: str,
        language: str,
        score: float | None,
        model: str,
        device: str,
        reason: str = "",
    ) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown speech segment status: {status}")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                update speech_segments
                set status = ?, text = ?, language = ?, score = ?, model = ?, device = ?,
                    reason = ?, updated_at = ?
                where id = ?
                """,
                (status, text, language, score, model, device, reason, time.time(), segment_id),
            )

    def abandon_pending(self, reason: str) -> int:
        """Audio is held only in memory, so pending work cannot survive a restart."""
        placeholders = ",".join("?" for _ in PENDING_STATUSES)
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                f"update speech_segments set status = 'skipped', reason = ?, updated_at = ? "
                f"where status in ({placeholders})",
                (reason, time.time(), *PENDING_STATUSES),
            )
            return int(cursor.rowcount or 0)

    def purge_before(self, cutoff: float) -> int:
        """Delete every segment older than ``cutoff`` (no incident awareness)."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute("delete from speech_segments where started_at < ?", (cutoff,))
            return int(cursor.rowcount or 0)

    def retention_candidates(self, cutoff: float, *, after_id: int = 0, limit: int = 1000) -> list[tuple[int, str, float, float]]:
        """(id, camera_id, started_at, ended_at) past retention, not yet finalized."""
        with self._connect() as conn:
            rows = conn.execute(
                "select id, camera_id, started_at, ended_at from speech_segments "
                "where content_expired = 0 and started_at < ? and id > ? order by id limit ?",
                (cutoff, after_id, max(1, int(limit))),
            ).fetchall()
        return [(int(row[0]), str(row[1]), float(row[2]), float(row[3])) for row in rows]

    def delete_segments(self, ids: list[int]) -> int:
        if not ids:
            return 0
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                f"delete from speech_segments where id in ({','.join('?' for _ in ids)})", ids,
            )
            return int(cursor.rowcount or 0)

    def expire_content(self, ids: list[int]) -> int:
        """Remove transcript text (and its search entry) but keep the timing row."""
        if not ids:
            return 0
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "update speech_segments set text = '', content_expired = 1, updated_at = ? "
                f"where id in ({','.join('?' for _ in ids)})",
                (time.time(), *ids),
            )
            return int(cursor.rowcount or 0)

    # Reads -------------------------------------------------------------

    def get(self, segment_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("select * from speech_segments where id = ?", (segment_id,)).fetchone()
        return self._row(row) if row is not None else None

    def list_segments(
        self,
        *,
        camera_ids: list[str] | None = None,
        start: float | None = None,
        end: float | None = None,
        query: str = "",
        statuses: list[str] | None = None,
        before_id: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if camera_ids:
            clauses.append(f"s.camera_id in ({','.join('?' for _ in camera_ids)})")
            params.extend(camera_ids)
        if start is not None:
            clauses.append("s.ended_at >= ?")
            params.append(start)
        if end is not None:
            clauses.append("s.started_at <= ?")
            params.append(end)
        if statuses:
            clauses.append(f"s.status in ({','.join('?' for _ in statuses)})")
            params.extend(statuses)
        if before_id is not None:
            clauses.append("s.id < ?")
            params.append(before_id)
        source = "speech_segments s"
        query = " ".join(str(query or "").split())[:200]
        if query:
            if self.fts_enabled:
                source += " join speech_segments_fts f on f.rowid = s.id"
                clauses.append("speech_segments_fts match ?")
                params.append(self._fts_query(query))
            else:
                clauses.append("s.text like ? escape '\\'")
                escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                params.append(f"%{escaped}%")
        where = f"where {' and '.join(clauses)}" if clauses else ""
        limit = max(1, min(int(limit), MAX_LIST_LIMIT))
        with self._connect() as conn:
            rows = conn.execute(
                f"select s.* from {source} {where} order by s.id desc limit ?",
                (*params, limit),
            ).fetchall()
        return [self._row(row) for row in rows]

    def timing_rows(self, camera_id: str, start: float, end: float) -> list[tuple[int, float, float, str, bool]]:
        """(id, started_at, ended_at, status, has_text) overlapping a span; never text."""
        with self._connect() as conn:
            rows = conn.execute(
                "select id, started_at, ended_at, status, text != '' as has_text "
                "from speech_segments where camera_id = ? and ended_at >= ? and started_at <= ? "
                "order by started_at",
                (camera_id, start, end),
            ).fetchall()
        return [
            (int(row[0]), float(row[1]), float(row[2]), str(row[3]), bool(row[4]))
            for row in rows
        ]

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "select status, count(*) as total from speech_segments group by status"
            ).fetchall()
        return {str(row["status"]): int(row["total"]) for row in rows}

    @staticmethod
    def _fts_query(query: str) -> str:
        # Quote every term so user input cannot inject FTS operators.
        terms = [term.replace('"', '""') for term in query.split() if term]
        return " ".join(f'"{term}"' for term in terms)

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "camera_id": row["camera_id"],
            "started_at": _iso(row["started_at"]),
            "ended_at": _iso(row["ended_at"]),
            "duration_seconds": round(float(row["ended_at"]) - float(row["started_at"]), 3),
            "peak_probability": float(row["peak_probability"]),
            "status": row["status"],
            "text": row["text"],
            "language": row["language"],
            "score": row["score"],
            "model": row["model"],
            "device": row["device"],
            "reason": row["reason"],
            "content_expired": bool(row["content_expired"]),
        }

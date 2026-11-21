"""Durable speech segment index (``speech.sqlite3``).

Speech rows live in their own database so the listener never contends for the
event store's writer lock. Audio is not stored: rows point at a time range that
playback resolves against the camera's recording.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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
MAX_OVERLAP_WORDS = 6
FINISHED = 1
MISSING_TAIL = 2
GAP_STATUSES = frozenset({"failed", "skipped"})


def _iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat()


class SpeechStore:
    SCHEMA_VERSION = 4

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
            for column in ("voiced_dbfs", "noise_dbfs"):
                if column not in columns:
                    conn.execute(f"alter table speech_segments add column {column} real")
            for column, definition in (
                ("conversation_id", "text"),
                ("chunk_index", "integer"),
                ("conversation_final", "integer not null default 0"),
                ("conversation_interrupted", "integer not null default 0"),
                ("transcript_in_conversation", "integer not null default 0"),
            ):
                if column not in columns:
                    conn.execute(f"alter table speech_segments add column {column} {definition}")
            conn.execute(
                "create unique index if not exists speech_segments_conversation_chunk "
                "on speech_segments (conversation_id, chunk_index) where conversation_id is not null"
            )
            conn.execute(
                "create index if not exists speech_segments_conversation_id "
                "on speech_segments (conversation_id) where conversation_id is not null"
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
        voiced_dbfs: float | None = None,
        noise_dbfs: float | None = None,
        conversation_id: str | None = None,
        chunk_index: int | None = None,
        final: bool = False,
        interrupted: bool = False,
        transcript_in_conversation: bool = False,
    ) -> int:
        if status not in STATUSES:
            raise ValueError(f"unknown speech segment status: {status}")
        now = time.time()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert into speech_segments
                    (camera_id, started_at, ended_at, peak_probability, status, reason,
                     voiced_dbfs, noise_dbfs, created_at, updated_at, conversation_id, chunk_index,
                     conversation_final, conversation_interrupted, transcript_in_conversation)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (camera_id, started_at, ended_at, peak_probability, status, reason,
                 voiced_dbfs, noise_dbfs, now, now, conversation_id, chunk_index,
                 int(final), int(interrupted), int(transcript_in_conversation)),
            )
            return int(cursor.lastrowid)

    def set_status(self, segment_id: int, status: str, *, reason: str = "") -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown speech segment status: {status}")
        with self._lock, self._connect() as conn:
            conn.execute(
                "update speech_segments set status = ?, reason = ?, updated_at = ? "
                "where id = ? and content_expired = 0",
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
    ) -> bool:
        """Write a worker result only while its segment still retains content."""
        if status not in STATUSES:
            raise ValueError(f"unknown speech segment status: {status}")
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                update speech_segments
                set status = ?, text = ?, language = ?, score = ?, model = ?, device = ?,
                    reason = ?, updated_at = ?
                where id = ? and content_expired = 0
                """,
                (status, text, language, score, model, device, reason, time.time(), segment_id),
            )
            return cursor.rowcount > 0

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

    def close_unfinished_conversations(self) -> None:
        """Mark sessions whose listener lost its buffered tail on restart."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                update speech_segments
                set conversation_final = ?, conversation_interrupted = 1
                where id in (
                    select max(id) from speech_segments where conversation_id is not null
                    group by conversation_id having max(conversation_final) = 0
                )
                """,
                (FINISHED,),
            )

    def purge_before(self, cutoff: float) -> int:
        """Delete every segment older than ``cutoff`` (no incident awareness)."""
        return self._delete("started_at < ?", [cutoff])

    def retention_candidates(
        self, cutoff: float, *, after_id: int = 0, limit: int = 1000,
    ) -> list[tuple[int, str, float, float, bool]]:
        """(id, camera_id, started_at, ended_at, is_chunk) past retention."""
        with self._connect() as conn:
            rows = conn.execute(
                "select id, camera_id, started_at, ended_at, conversation_id is not null from speech_segments "
                "where content_expired = 0 and started_at < ? and id > ? order by id limit ?",
                (cutoff, after_id, max(1, int(limit))),
            ).fetchall()
        return [(int(row[0]), str(row[1]), float(row[2]), float(row[3]), bool(row[4])) for row in rows]

    def delete_segments(self, ids: list[int]) -> int:
        if not ids:
            return 0
        return self._delete(f"id in ({','.join('?' for _ in ids)})", ids)

    def _delete(self, predicate: str, params: Sequence[int | float]) -> int:
        with self._lock, self._connect() as conn:
            # A retained prefix of a closed conversation is still closed.
            conn.execute("begin immediate")
            tails = conn.execute(
                f"select conversation_id from speech_segments where conversation_final != 0 and {predicate}",
                params,
            ).fetchall()
            cursor = conn.execute(f"delete from speech_segments where {predicate}", params)
            conn.executemany(
                """
                update speech_segments set conversation_final = ?
                where id = (
                    select id from speech_segments where conversation_id = ?
                    order by chunk_index desc limit 1
                )
                """,
                [(MISSING_TAIL, row["conversation_id"]) for row in tails],
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
        view: Literal["segments", "transcript", "conversation"] = "segments",
    ) -> list[dict[str, Any]]:
        view_predicates = {
            "segments": "s.conversation_id is null",
            "transcript": "s.transcript_in_conversation = 0",
            "conversation": "s.transcript_in_conversation = 0",
        }
        clauses = [view_predicates[view]]
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
        query = " ".join(str(query or "").split())[:200]
        limit = max(1, min(int(limit), MAX_LIST_LIMIT))
        if view == "conversation":
            return self._list_conversations(clauses, params, query, before_id, limit)
        if before_id is not None:
            clauses.append("s.id < ?")
            params.append(before_id)
        source = "speech_segments s"
        if query:
            if self.fts_enabled:
                source += " join speech_segments_fts f on f.rowid = s.id"
                clauses.append("speech_segments_fts match ?")
                params.append(self._fts_query(query))
            else:
                clauses.append("s.text like ? escape '\\'")
                params.append(self._like_query(query))
        where = " and ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(
                f"select s.* from {source} where {where} order by s.id desc limit ?",
                (*params, limit),
            ).fetchall()
        return [self._row(row) for row in rows]

    def _list_conversations(
        self, clauses: list[str], params: list[Any], query: str, before_id: int | None, limit: int,
    ) -> list[dict[str, Any]]:
        matching = " and ".join(clauses)
        cursor = "and h.id < ?" if before_id is not None else ""
        cursor_params = (before_id,) if before_id is not None else ()
        terms_sql = ""
        if query:
            terms = query.split()
            # Count each matched term per conversation, including when two
            # terms occur on opposite sides of a transcription boundary.
            if self.fts_enabled:
                term_values = [self._fts_query(term) for term in terms]
                match_source = (
                    "terms cross join speech_segments_fts(terms.token) f join speech_segments s on s.id = f.rowid"
                )
            else:
                term_values = [self._like_query(term) for term in terms]
                match_source = "terms cross join speech_segments s on s.text like terms.token escape '\\'"
            term_values = list(dict.fromkeys(term_values))
            terms_sql = f"terms(token) as (values {','.join('(?)' for _ in term_values)}),"
            page = f"""
                select h.id
                from (
                    select coalesce((
                        select min(a.id) from speech_segments a where a.conversation_id = s.conversation_id
                    ), s.id) as group_id
                    from {match_source}
                    where {matching}
                    and (s.conversation_id is null or s.status = 'done')
                    group by group_id having count(distinct terms.token) = ?
                ) matches
                join speech_segments h on h.id = matches.group_id
                where 1 = 1 {cursor}
                order by h.id desc limit ?
            """
            page_params = (*term_values, *params, len(term_values), *cursor_params, limit)
        else:
            # Keep the cursor fixed while a listener appends more chunks.
            page = f"""
                select h.id from speech_segments h
                where h.transcript_in_conversation = 0 {cursor}
                and (h.conversation_id is null or h.id = (
                    select min(a.id) from speech_segments a where a.conversation_id = h.conversation_id
                ))
                and exists (
                    select 1 from speech_segments s
                    where (s.id = h.id or s.conversation_id = h.conversation_id) and {matching}
                )
                order by h.id desc limit ?
            """
            page_params = (*cursor_params, *params, limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                with {terms_sql} page as ({page})
                select page.id as group_id, c.* from page
                join speech_segments h on h.id = page.id
                join speech_segments c on c.id = h.id or c.conversation_id = h.conversation_id
                order by page.id desc, c.chunk_index
                """,
                page_params,
            )
            grouped: dict[int, list[sqlite3.Row]] = {}
            for row in rows:
                grouped.setdefault(row["group_id"], []).append(row)
        return [
            self._conversation(group_id, chunks) if chunks[0]["conversation_id"] else self._row(chunks[0])
            for group_id, chunks in grouped.items()
        ]

    def timing_rows(self, camera_id: str, start: float, end: float) -> list[tuple[int, float, float, str, bool]]:
        """(id, started_at, ended_at, status, has_text) overlapping a span; never text."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                select s.id, s.started_at, s.ended_at, s.status,
                    s.text != '' or (s.transcript_in_conversation = 1 and exists (
                        select 1 from speech_segments c
                        where c.camera_id = s.camera_id and c.conversation_id is not null
                        and c.started_at < s.ended_at and c.ended_at > s.started_at
                        and c.status = 'done' and c.text != ''
                    )) as has_text
                from speech_segments s
                where s.camera_id = ? and s.ended_at >= ? and s.started_at <= ?
                    and s.conversation_id is null
                order by s.started_at
                """,
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
    def _like_query(query: str) -> str:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"%{escaped}%"

    @staticmethod
    def _overlap_words(previous: list[str], current: list[str]) -> int:
        """Remove an exact multiword seam match, preserving uncertain speech."""
        tail = [re.sub(r"\W", "", word.casefold()) for word in previous[-MAX_OVERLAP_WORDS:]]
        head = [re.sub(r"\W", "", word.casefold()) for word in current[:MAX_OVERLAP_WORDS]]
        for count in range(min(len(tail), len(head)), 1, -1):
            if all(head[:count]) and tail[-count:] == head[:count]:
                return count
        return 0

    @classmethod
    def _conversation(cls, group_id: int, chunks: list[sqlite3.Row]) -> dict[str, Any]:
        first, last = chunks[0], chunks[-1]
        result = cls._row(first)
        start, end = first["started_at"], last["ended_at"]
        gaps = []
        words: list[str] = []
        previous_words: list[str] = []
        previous_end = None
        previous_index = -1
        needs_gap_separator = False
        statuses = set()
        model_chunk = first
        for chunk in chunks:
            index = chunk["chunk_index"]
            status = chunk["status"]
            chunk_start, chunk_end = chunk["started_at"], chunk["ended_at"]
            missing = index != previous_index + 1 or (
                previous_end is not None and chunk_start - previous_end > 0.001
            )
            if missing:
                gaps.append({
                    "started_at": _iso(previous_end), "ended_at": _iso(chunk_start), "reason": "missing",
                })
            expired = chunk["content_expired"]
            if expired or status in GAP_STATUSES:
                gaps.append({
                    "started_at": _iso(chunk_start), "ended_at": _iso(chunk_end),
                    "reason": "expired" if expired else status,
                })
            needs_gap_separator |= bool(missing or expired or status in GAP_STATUSES)
            if not expired:
                statuses.add(status)
            current_words = chunk["text"].split() if status == "done" and not expired else []
            if current_words and not words:
                model_chunk = chunk
            duplicate = 0
            if previous_words and current_words and not missing and chunk_start < previous_end:
                duplicate = cls._overlap_words(previous_words, current_words)
            if current_words and needs_gap_separator:
                if words:
                    words.append("[…]")
                needs_gap_separator = False
            words.extend(current_words[duplicate:])
            previous_words = current_words
            previous_index, previous_end = index, chunk_end
        if last["conversation_final"] == MISSING_TAIL or last["conversation_interrupted"]:
            gaps.append({
                "started_at": _iso(end), "ended_at": None,
                "reason": "missing" if last["conversation_final"] == MISSING_TAIL else "interrupted",
            })
        status = next(
            (status for status in ("transcribing", "queued", "done", "failed", "skipped", "low_confidence")
             if status in statuses),
            "no_text",
        )
        result.update(
            id=group_id,
            ended_at=_iso(end),
            duration_seconds=round(end - start, 3),
            peak_probability=max(chunk["peak_probability"] for chunk in chunks),
            status=status,
            text=" ".join(words),
            language=model_chunk["language"],
            model=model_chunk["model"],
            device=model_chunk["device"],
            score=None,
            reason="",
            content_expired=not statuses,
            chunk_count=len(chunks),
            conversation_open=not bool(last["conversation_final"]),
            transcript_gaps=gaps,
        )
        # These fields describe one chunk rather than the combined transcript.
        for key in ("chunk_index", "final", "interrupted"):
            del result[key]
        return result

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        result = {
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
            "voiced_dbfs": row["voiced_dbfs"],
            "noise_dbfs": row["noise_dbfs"],
            "snr_db": (
                round(float(row["voiced_dbfs"]) - float(row["noise_dbfs"]), 1)
                if row["voiced_dbfs"] is not None and row["noise_dbfs"] is not None
                else None
            ),
        }
        if row["conversation_id"] is not None:
            result.update(
                conversation_id=row["conversation_id"],
                chunk_index=row["chunk_index"],
                final=bool(row["conversation_final"]),
                interrupted=bool(row["conversation_interrupted"]),
            )
        return result

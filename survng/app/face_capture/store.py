"""Captured faces: one database and one crop directory beside the event store."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .groups import drop_empty_groups, ensure_group_schema, recompute_group
from .pipeline import CapturedFace, CaptureResult

LOGGER = logging.getLogger(__name__)

# Bumped when what a capture keeps changes; events captured ("done") by an
# older version are captured again.
CAPTURE_VERSION = 1
CROP_JPEG_QUALITY = 92

SCHEMA = """
create table if not exists face_capture_events (
    event_id integer primary key,
    camera_id text not null,
    event_epoch real not null,
    capture_version integer not null,
    state text not null,
    attempts integer not null default 0,
    faces integer not null default 0,
    candidates integer not null default 0,
    moment_source text not null default '',
    sub_frames integer not null default 0,
    main_frames integer not null default 0,
    seconds real not null default 0,
    error text not null default '',
    updated_at text not null
);
create table if not exists face_captures (
    id integer primary key autoincrement,
    event_id integer not null,
    camera_id text not null,
    track integer not null,
    offset_seconds real not null,
    captured_epoch real not null,
    box_json text not null,
    landmarks_json text not null,
    detector text not null,
    confidence real not null,
    norm real not null,
    embedding blob not null,
    embedding_model text not null,
    crop_path text not null,
    capture_version integer not null,
    created_at text not null
);
create index if not exists face_captures_event on face_captures(event_id);
create table if not exists face_capture_seeds (
    id integer primary key autoincrement,
    start_epoch real not null,
    end_epoch real not null,
    cameras_json text not null,
    state text not null,
    total integer not null,
    processed integer not null default 0,
    faces integer not null default 0,
    failed integer not null default 0,
    created_at text not null,
    updated_at text not null,
    finished_at text not null default ''
);
-- The moments an event's capture decoded (absolute epochs): another event of
-- the same visit skips a moment one of these already covers.
create table if not exists face_capture_moments (
    event_id integer not null,
    camera_id text not null,
    epoch real not null
);
create index if not exists face_capture_moments_time on face_capture_moments(camera_id, epoch);
create index if not exists face_capture_moments_event on face_capture_moments(event_id);
-- Which event's decoded moments an event skipped: when that event is removed,
-- the one that skipped them is captured again.
create table if not exists face_capture_covered (
    event_id integer not null,
    source_event_id integer not null,
    primary key (event_id, source_event_id)
);
create index if not exists face_capture_covered_source on face_capture_covered(source_event_id);
-- Counters other processes (the offline seed) raise for the service to see.
create table if not exists face_capture_state (
    key text primary key,
    value integer not null
);
"""


def _count_reset(connection) -> None:
    connection.execute(
        "insert into face_capture_state (key, value) values ('dependents_reset', 1)"
        " on conflict(key) do update set value = value + 1")


class ClosingConnection(sqlite3.Connection):
    """A connection that also closes when its ``with`` block ends."""

    def __exit__(self, *exc_info):
        try:
            return super().__exit__(*exc_info)
        finally:
            self.close()


class FaceCaptureStore:
    def __init__(self, database_dir: Path) -> None:
        self.path = Path(database_dir) / "face-captures.sqlite3"
        self.crop_dir = Path(database_dir) / "face-captures"
        self._lock = threading.Lock()
        # The reset counter last seen (see take_dependents_reset).
        self._resets_seen: int | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            ensure_group_schema(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0, factory=ClosingConnection)
        connection.execute("pragma journal_mode = wal")
        connection.row_factory = sqlite3.Row
        return connection

    def attempts(self, event_ids: Iterable[int]) -> dict[int, tuple[str, int, int]]:
        """event id -> (state, attempts, capture version) for events seen before."""
        ids = [int(i) for i in event_ids]
        found: dict[int, tuple[str, int, int]] = {}
        with self._connect() as connection:
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                for row in connection.execute(
                    f"select event_id, state, attempts, capture_version from face_capture_events "
                    f"where event_id in ({','.join('?' * len(chunk))})", chunk,
                ):
                    found[int(row[0])] = (str(row[1]), int(row[2]), int(row[3]))
        return found

    def failed_at(self, event_id: int) -> float | None:
        """Epoch of the event's last failed attempt, or None when it is not failed."""
        with self._connect() as connection:
            row = connection.execute(
                "select updated_at from face_capture_events where event_id = ? and state = 'failed'",
                (int(event_id),),
            ).fetchone()
        if row is None:
            return None
        try:
            return datetime.fromisoformat(str(row[0])).timestamp()
        except ValueError:
            return None

    def record(
        self,
        *,
        event_id: int,
        camera_id: str,
        event_epoch: float,
        result: CaptureResult,
        embedding_model: str,
        seconds: float,
        covered_by: Iterable[int] = (),
    ) -> int:
        """Replace the event's captured faces with ``result``'s; returns the number kept.

        ``covered_by``: the visit's events whose decoded moments this capture skipped.
        """
        now = datetime.now(timezone.utc).isoformat()
        written: list[tuple[CapturedFace, str]] = []
        day = datetime.fromtimestamp(event_epoch, timezone.utc).strftime("%Y-%m-%d")
        folder = self.crop_dir / camera_id / day
        # Names unique to this capture: a rerun never overwrites the crops the
        # event's current rows point at, so a failed rerun leaves them intact.
        stamp = time.time_ns()
        attempted: list[Path] = []
        try:
            for index, face in enumerate(result.faces):
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{event_id}-{stamp}-{index}.jpg"
                attempted.append(path)
                if not cv2.imwrite(str(path), face.crop, [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY]):
                    raise OSError(f"could not write {path}")
                written.append((face, str(path)))
            old = self._replace_faces(
                event_id=event_id, camera_id=camera_id, event_epoch=event_epoch, result=result,
                embedding_model=embedding_model, seconds=seconds, written=written, now=now,
                covered_by=sorted({int(i) for i in covered_by if int(i) != int(event_id)}),
            )
        except BaseException:
            # Any failure (a cv2.error on a bad crop too): no row points at
            # these crops, so nothing would ever remove them otherwise.
            for path in attempted:
                path.unlink(missing_ok=True)
            raise
        kept = {path for _, path in written}
        for path in old:
            if path not in kept:
                Path(path).unlink(missing_ok=True)
        return len(written)

    def _replace_faces(
        self, *, event_id, camera_id, event_epoch, result, embedding_model, seconds, written, now, covered_by=(),
    ) -> list[str]:
        """Swap the event's rows for ``written`` in one transaction; the replaced crop paths."""
        with self._lock, self._connect() as connection:
            old = [row[0] for row in connection.execute(
                "select crop_path from face_captures where event_id = ?", (event_id,))]
            old_groups = _groups_of(connection, "event_id = ?", (event_id,))
            connection.execute("delete from face_captures where event_id = ?", (event_id,))
            for group_id in old_groups:
                # Kept when emptied, name and all, so the replacement faces
                # can rejoin it; grouping drops it if they do not.
                recompute_group(connection, group_id, drop_empty=False)
            connection.executemany(
                "insert into face_captures (event_id, camera_id, track, offset_seconds, captured_epoch, box_json,"
                " landmarks_json, detector, confidence, norm, embedding, embedding_model, crop_path,"
                " capture_version, created_at, frame_width, frame_height)"
                " values (?, ?, ?, ?, ?, ?, ?, 'yunet', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        event_id, camera_id, face.track, face.offset, event_epoch + face.offset,
                        json.dumps(list(face.box)), json.dumps(face.landmarks), face.confidence, face.norm,
                        np.asarray(face.embedding, dtype=np.float32).tobytes(), embedding_model, path,
                        CAPTURE_VERSION, now,
                        *(face.frame_size or (None, None)),
                    )
                    for face, path in written
                ],
            )
            previous = sorted(round(float(row[0]), 3) for row in connection.execute(
                "select epoch from face_capture_moments where event_id = ?", (event_id,)))
            current = sorted(round(event_epoch + float(moment), 3) for moment in result.decoded_moments)
            if previous != current:
                # Events that skipped moments this event no longer decodes
                # would keep no faces there: capture them again.
                dependents = [int(row[0]) for row in connection.execute(
                    "select event_id from face_capture_covered where source_event_id = ?", (event_id,))]
                connection.executemany(
                    "update face_capture_events set capture_version = 0, attempts = 0 where event_id = ?",
                    [(dependent,) for dependent in dependents])
                if dependents:
                    _count_reset(connection)
                connection.execute("delete from face_capture_covered where source_event_id = ?", (event_id,))
            connection.execute("delete from face_capture_moments where event_id = ?", (event_id,))
            connection.executemany(
                "insert into face_capture_moments (event_id, camera_id, epoch) values (?, ?, ?)",
                [(event_id, camera_id, event_epoch + float(moment)) for moment in result.decoded_moments],
            )
            connection.execute("delete from face_capture_covered where event_id = ?", (event_id,))
            connection.executemany(
                "insert or ignore into face_capture_covered (event_id, source_event_id) values (?, ?)",
                [(event_id, source) for source in covered_by],
            )
            self._upsert_event(
                connection, event_id=event_id, camera_id=camera_id, event_epoch=event_epoch, state="done",
                faces=len(written), result=result, seconds=seconds, error="; ".join(result.errors)[:500], now=now,
            )
        return old

    def record_failure(self, *, event_id: int, camera_id: str, event_epoch: float, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            self._upsert_event(
                connection, event_id=event_id, camera_id=camera_id, event_epoch=event_epoch, state="failed",
                faces=0, result=None, seconds=0.0, error=error[:500], now=now,
            )

    @staticmethod
    def _upsert_event(connection, *, event_id, camera_id, event_epoch, state, faces, result, seconds, error, now):
        connection.execute(
            "insert into face_capture_events (event_id, camera_id, event_epoch, capture_version, state, attempts,"
            " faces, candidates, moment_source, sub_frames, main_frames, seconds, error, updated_at)"
            " values (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)"
            " on conflict(event_id) do update set state = excluded.state,"
            # Attempts count per capture version, so a version bump retries afresh.
            " attempts = case when face_capture_events.capture_version = excluded.capture_version"
            " then face_capture_events.attempts + 1 else 1 end, faces = excluded.faces,"
            " candidates = excluded.candidates, moment_source = excluded.moment_source,"
            " sub_frames = excluded.sub_frames, main_frames = excluded.main_frames, seconds = excluded.seconds,"
            " error = excluded.error, capture_version = excluded.capture_version, updated_at = excluded.updated_at",
            (
                event_id, camera_id, event_epoch, CAPTURE_VERSION, state, faces,
                result.candidates if result else 0, result.moment_source if result else "",
                result.sub_frames if result else 0, result.main_frames if result else 0,
                round(seconds, 2), error, now,
            ),
        )

    def take_dependents_reset(self) -> bool:
        """Whether events were reset to be captured again since the last call
        (by this process or another one, such as the offline seed)."""
        with self._connect() as connection:
            row = connection.execute("select value from face_capture_state where key = 'dependents_reset'").fetchone()
        count = int(row[0]) if row else 0
        with self._lock:
            changed = self._resets_seen is not None and count != self._resets_seen
            self._resets_seen = count
        return changed

    def captured_event_ids(self) -> list[int]:
        with self._connect() as connection:
            return [int(row[0]) for row in connection.execute("select event_id from face_capture_events")]

    def forget_events(self, event_ids: Iterable[int]) -> int:
        """Remove events (gone from the event store) and their faces and crops."""
        ids = [int(i) for i in event_ids]
        removed = 0
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            with self._lock, self._connect() as connection:
                paths = [row[0] for row in connection.execute(
                    f"select crop_path from face_captures where event_id in ({marks})", chunk)]
                old_groups = _groups_of(connection, f"event_id in ({marks})", tuple(chunk))
                connection.execute(f"delete from face_captures where event_id in ({marks})", chunk)
                for group_id in old_groups:
                    recompute_group(connection, group_id, drop_empty=False)
                drop_empty_groups(connection)  # and the names only those groups had
                connection.execute(f"delete from face_capture_moments where event_id in ({marks})", chunk)
                # Events that skipped these events' moments have no faces
                # there now: capture them again (as after a version change).
                dependents = [int(row[0]) for row in connection.execute(
                    f"select event_id from face_capture_covered where source_event_id in ({marks})", chunk)]
                if dependents:
                    connection.execute(
                        f"update face_capture_events set capture_version = 0, attempts = 0"
                        f" where event_id in ({','.join('?' * len(dependents))})", dependents)
                    _count_reset(connection)
                connection.execute(
                    f"delete from face_capture_covered where event_id in ({marks}) or source_event_id in ({marks})",
                    (*chunk, *chunk))
                removed += connection.execute(
                    f"delete from face_capture_events where event_id in ({marks})", chunk).rowcount
            for path in paths:
                Path(path).unlink(missing_ok=True)
        return removed

    def decoded_moments(
        self, camera_id: str, event_ids: Iterable[int], low: float, high: float,
    ) -> list[tuple[int, float]]:
        """(event, epoch) of the moments these events decoded on a camera within [low, high]."""
        ids = sorted({int(event_id) for event_id in event_ids})
        if not ids:
            return []
        with self._connect() as connection:
            return [(int(row[0]), float(row[1])) for row in connection.execute(
                "select event_id, epoch from face_capture_moments where camera_id = ? and epoch between ? and ?"
                f" and event_id in ({','.join('?' * len(ids))})",
                (camera_id, float(low), float(high), *ids))]

    def faces(self, event_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                "select * from face_captures where event_id = ? order by track, offset_seconds", (event_id,))]

    # Seeds: capturing a range of past events on request -------------------

    def create_seed(self, *, start_epoch: float, end_epoch: float, cameras: list[str], total: int) -> int:
        """A running seed over ``[start, end]`` (all cameras when ``cameras`` is
        empty). Only one seed may be running or paused at a time."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("begin immediate")
            if connection.execute(
                "select 1 from face_capture_seeds where state in ('running', 'paused')"
            ).fetchone() is not None:
                raise SeedConflict("another seed is still running or paused")
            cursor = connection.execute(
                "insert into face_capture_seeds (start_epoch, end_epoch, cameras_json, state, total,"
                " created_at, updated_at) values (?, ?, ?, 'running', ?, ?, ?)",
                (float(start_epoch), float(end_epoch), json.dumps(sorted(cameras)), int(total), now, now),
            )
            return int(cursor.lastrowid)

    def seeds(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(
                "select * from face_capture_seeds order by id desc limit ?", (int(limit),))]
        for row in rows:
            row["cameras"] = json.loads(row.pop("cameras_json") or "[]")
        return rows

    def active_seed(self) -> dict[str, Any] | None:
        return next((seed for seed in self.seeds(5) if seed["state"] in ("running", "paused")), None)

    def set_seed_state(self, seed_id: int, state: str) -> dict[str, Any]:
        """pause (running), resume (paused), cancel (running or paused), done (running)."""
        allowed = {"paused": ("running",), "running": ("paused",), "cancelled": ("running", "paused"),
                   "done": ("running",)}
        if state not in allowed:
            raise ValueError(f"unknown seed state {state}")
        now = datetime.now(timezone.utc).isoformat()
        finished = now if state in ("cancelled", "done") else ""
        with self._lock, self._connect() as connection:
            row = connection.execute("select state from face_capture_seeds where id = ?", (int(seed_id),)).fetchone()
            if row is None:
                raise KeyError(seed_id)
            if row[0] not in allowed[state]:
                raise SeedConflict(f"a {row[0]} seed cannot become {state}")
            connection.execute(
                "update face_capture_seeds set state = ?, updated_at = ?, finished_at = ? where id = ?",
                (state, now, finished, int(seed_id)),
            )
            if state == "done":
                # Every event in the range is finished by then, including those
                # the regular capture loop took; faces stay this seed's own.
                connection.execute(
                    "update face_capture_seeds set processed = max(processed, total) where id = ?",
                    (int(seed_id),),
                )
        return next(seed for seed in self.seeds(50) if seed["id"] == int(seed_id))

    def seed_progress(self, seed_id: int, *, faces: int = 0, failed: bool = False) -> None:
        """Count one seed event as finished (captured, or out of attempts)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                "update face_capture_seeds set processed = processed + 1, faces = faces + ?,"
                " failed = failed + ?, updated_at = ? where id = ?",
                (int(faces), 1 if failed else 0, now, int(seed_id)),
            )

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            states = dict(connection.execute(
                "select state, count(*) from face_capture_events group by state").fetchall())
            faces = connection.execute("select count(*) from face_captures").fetchone()[0]
        return {"events": states, "faces": int(faces)}


class SeedConflict(RuntimeError):
    """A seed request that conflicts with the seed already running or paused."""


def _groups_of(connection, where: str, args: tuple) -> list[int]:
    return [int(row[0]) for row in connection.execute(
        f"select distinct group_id from face_captures where {where} and group_id is not null", args)]

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
"""


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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

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
    ) -> int:
        """Replace the event's captured faces with ``result``'s; returns the number kept."""
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
        self, *, event_id, camera_id, event_epoch, result, embedding_model, seconds, written, now,
    ) -> list[str]:
        """Swap the event's rows for ``written`` in one transaction; the replaced crop paths."""
        with self._lock, self._connect() as connection:
            old = [row[0] for row in connection.execute(
                "select crop_path from face_captures where event_id = ?", (event_id,))]
            connection.execute("delete from face_captures where event_id = ?", (event_id,))
            connection.executemany(
                "insert into face_captures (event_id, camera_id, track, offset_seconds, captured_epoch, box_json,"
                " landmarks_json, detector, confidence, norm, embedding, embedding_model, crop_path,"
                " capture_version, created_at) values (?, ?, ?, ?, ?, ?, ?, 'yunet', ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        event_id, camera_id, face.track, face.offset, event_epoch + face.offset,
                        json.dumps(list(face.box)), json.dumps(face.landmarks), face.confidence, face.norm,
                        np.asarray(face.embedding, dtype=np.float32).tobytes(), embedding_model, path,
                        CAPTURE_VERSION, now,
                    )
                    for face, path in written
                ],
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
                connection.execute(f"delete from face_captures where event_id in ({marks})", chunk)
                removed += connection.execute(
                    f"delete from face_capture_events where event_id in ({marks})", chunk).rowcount
            for path in paths:
                Path(path).unlink(missing_ok=True)
        return removed

    def faces(self, event_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                "select * from face_captures where event_id = ? order by track, offset_seconds", (event_id,))]

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            states = dict(connection.execute(
                "select state, count(*) from face_capture_events group by state").fetchall())
            faces = connection.execute("select count(*) from face_captures").fetchone()[0]
        return {"events": states, "faces": int(faces)}


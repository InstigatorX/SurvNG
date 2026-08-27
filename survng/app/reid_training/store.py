"""Durable metadata for optional ReID training crops.

Uses an isolated SQLite database so production appearance indexing and face
identity tables stay untouched. Crop files live under storage_dir/reid_training/.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReidTrainingStore:
    """Sidecar store for environment-adaptation ReID samples."""

    def __init__(self, database_dir: Path, storage_dir: Path) -> None:
        self.database_dir = Path(database_dir)
        self.storage_dir = Path(storage_dir)
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.crops_root = self.storage_dir / "reid_training"
        self.db_path = self.database_dir / "reid-training.sqlite3"
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma busy_timeout = 10000")
        connection.execute("pragma foreign_keys = on")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute("pragma journal_mode = wal")
            connection.execute("pragma synchronous = normal")
            connection.execute(
                """
                create table if not exists reid_identities (
                    id integer primary key autoincrement,
                    display_name text not null default '',
                    created_at text not null
                )
                """
            )
            connection.execute(
                """
                create table if not exists reid_samples (
                    id integer primary key autoincrement,
                    sample_id text not null unique,
                    event_id integer not null,
                    camera_id text not null,
                    track_id integer not null,
                    captured_at text not null,
                    bounding_box_json text not null,
                    detection_confidence real not null,
                    crop_path text not null,
                    embedding_blob blob,
                    embedding_size integer not null default 0,
                    model_kind text not null default '',
                    model_fingerprint text not null default '',
                    assigned_person_id integer,
                    assignment_source text not null default 'track',
                    assignment_confidence real,
                    review_status text not null default 'auto',
                    selection_reason text not null default '',
                    quality_score real not null default 0,
                    created_at text not null,
                    foreign key(assigned_person_id) references reid_identities(id)
                        on delete set null
                )
                """
            )
            connection.execute(
                "create index if not exists idx_reid_samples_event "
                "on reid_samples(event_id, track_id)"
            )
            connection.execute(
                "create index if not exists idx_reid_samples_camera "
                "on reid_samples(camera_id, captured_at desc)"
            )
            connection.execute(
                "create index if not exists idx_reid_samples_person "
                "on reid_samples(assigned_person_id)"
            )

    def create_identity(self, *, display_name: str = "") -> int:
        created_at = _utc_now()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "insert into reid_identities (display_name, created_at) values (?, ?)",
                (str(display_name or ""), created_at),
            )
            identity_id = int(cursor.lastrowid)
            if not display_name:
                connection.execute(
                    "update reid_identities set display_name = ? where id = ?",
                    (f"person_{identity_id:06d}", identity_id),
                )
            return identity_id

    def insert_sample(self, record: dict[str, Any]) -> int | None:
        embedding = record.get("embedding")
        embedding_blob = None
        embedding_size = 0
        if embedding is not None:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if vector.size and np.all(np.isfinite(vector)):
                embedding_blob = vector.tobytes()
                embedding_size = int(vector.size)
        box = record.get("bounding_box") or {}
        with self._lock, self._connect() as connection:
            try:
                cursor = connection.execute(
                    """
                    insert into reid_samples (
                        sample_id, event_id, camera_id, track_id, captured_at,
                        bounding_box_json, detection_confidence, crop_path,
                        embedding_blob, embedding_size, model_kind, model_fingerprint,
                        assigned_person_id, assignment_source, assignment_confidence,
                        review_status, selection_reason, quality_score, created_at
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(record["sample_id"]),
                        int(record["event_id"]),
                        str(record["camera_id"]),
                        int(record["track_id"]),
                        str(record["captured_at"]),
                        json.dumps(box, separators=(",", ":")),
                        float(record.get("detection_confidence") or 0.0),
                        str(record["crop_path"]),
                        embedding_blob,
                        embedding_size,
                        str(record.get("model_kind") or ""),
                        str(record.get("model_fingerprint") or ""),
                        record.get("assigned_person_id"),
                        str(record.get("assignment_source") or "track"),
                        record.get("assignment_confidence"),
                        str(record.get("review_status") or "auto"),
                        str(record.get("selection_reason") or ""),
                        float(record.get("quality_score") or 0.0),
                        str(record.get("created_at") or _utc_now()),
                    ),
                )
            except sqlite3.IntegrityError:
                return None
            return int(cursor.lastrowid)

    def count_samples(self, *, event_id: int | None = None) -> int:
        with self._lock, self._connect() as connection:
            if event_id is None:
                row = connection.execute(
                    "select count(*) as count from reid_samples"
                ).fetchone()
            else:
                row = connection.execute(
                    "select count(*) as count from reid_samples where event_id = ?",
                    (int(event_id),),
                ).fetchone()
        return int(row["count"] if row else 0)

    def status(self) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            samples = connection.execute(
                "select count(*) as count from reid_samples"
            ).fetchone()
            identities = connection.execute(
                "select count(*) as count from reid_identities"
            ).fetchone()
            newest = connection.execute(
                "select max(created_at) as newest_at from reid_samples"
            ).fetchone()
        return {
            "database_path": str(self.db_path),
            "crops_root": str(self.crops_root),
            "samples": int(samples["count"] if samples else 0),
            "identities": int(identities["count"] if identities else 0),
            "newest_at": str(newest["newest_at"] or "") if newest else "",
        }

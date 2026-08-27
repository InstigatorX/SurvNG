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

from ..incident_utils import stored_media_path


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
                """
                create table if not exists reid_pair_reviews (
                    id integer primary key autoincrement,
                    left_event_id integer not null,
                    left_track_id integer not null,
                    right_event_id integer not null,
                    right_track_id integer not null,
                    left_sample_id text not null default '',
                    right_sample_id text not null default '',
                    decision text not null,
                    similarity real,
                    created_at text not null,
                    unique(
                        left_event_id, left_track_id,
                        right_event_id, right_track_id
                    )
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
            connection.execute(
                "create index if not exists idx_reid_samples_review "
                "on reid_samples(review_status, created_at desc)"
            )

    @staticmethod
    def _row_to_sample(row: sqlite3.Row) -> dict[str, Any]:
        try:
            box = json.loads(row["bounding_box_json"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            box = {}
        return {
            "id": int(row["id"]),
            "sample_id": str(row["sample_id"]),
            "event_id": int(row["event_id"]),
            "camera_id": str(row["camera_id"]),
            "track_id": int(row["track_id"]),
            "captured_at": str(row["captured_at"]),
            "bounding_box": box if isinstance(box, dict) else {},
            "detection_confidence": float(row["detection_confidence"] or 0.0),
            "crop_path": str(row["crop_path"]),
            "crop_url": f"/api/reid-training/samples/{row['sample_id']}/crop.jpg",
            "embedding_size": int(row["embedding_size"] or 0),
            "model_kind": str(row["model_kind"] or ""),
            "model_fingerprint": str(row["model_fingerprint"] or ""),
            "assigned_person_id": (
                int(row["assigned_person_id"])
                if row["assigned_person_id"] is not None
                else None
            ),
            "assignment_source": str(row["assignment_source"] or ""),
            "assignment_confidence": (
                float(row["assignment_confidence"])
                if row["assignment_confidence"] is not None
                else None
            ),
            "review_status": str(row["review_status"] or ""),
            "selection_reason": str(row["selection_reason"] or ""),
            "quality_score": float(row["quality_score"] or 0.0),
            "created_at": str(row["created_at"] or ""),
        }

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

    def get_sample(self, sample_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "select * from reid_samples where sample_id = ?",
                (str(sample_id),),
            ).fetchone()
        return None if row is None else self._row_to_sample(row)

    def list_samples(
        self,
        *,
        limit: int = 50,
        event_id: int | None = None,
        camera_id: str | None = None,
        review_status: str | None = None,
        person_id: int | None = None,
    ) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 500))
        clauses: list[str] = []
        parameters: list[Any] = []
        if event_id is not None:
            clauses.append("event_id = ?")
            parameters.append(int(event_id))
        if camera_id:
            clauses.append("camera_id = ?")
            parameters.append(str(camera_id))
        if review_status:
            clauses.append("review_status = ?")
            parameters.append(str(review_status))
        if person_id is not None:
            clauses.append("assigned_person_id = ?")
            parameters.append(int(person_id))
        where = f"where {' and '.join(clauses)}" if clauses else ""
        parameters.append(bounded)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                select * from reid_samples
                {where}
                order by created_at desc, id desc
                limit ?
                """,
                parameters,
            ).fetchall()
        return [self._row_to_sample(row) for row in rows]

    def samples_for_track(
        self,
        event_id: int,
        track_id: int,
        *,
        exclude_rejected: bool = True,
    ) -> list[dict[str, Any]]:
        clauses = ["event_id = ?", "track_id = ?"]
        parameters: list[Any] = [int(event_id), int(track_id)]
        if exclude_rejected:
            clauses.append("review_status != 'rejected'")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                select * from reid_samples
                where {' and '.join(clauses)}
                order by captured_at asc, id asc
                """,
                parameters,
            ).fetchall()
        return [self._row_to_sample(row) for row in rows]

    def recent_event_ids(self, *, limit: int = 100) -> list[int]:
        bounded = max(1, min(int(limit), 500))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                select event_id
                from reid_samples
                where review_status != 'rejected'
                group by event_id
                order by max(created_at) desc
                limit ?
                """,
                (bounded,),
            ).fetchall()
        return [int(row["event_id"]) for row in rows]

    def resolve_crop_path(self, sample_id: str) -> Path:
        sample = self.get_sample(sample_id)
        if sample is None:
            raise FileNotFoundError("ReID training sample is unavailable")
        return stored_media_path(self.storage_dir, sample["crop_path"])

    def pair_reviewed(
        self,
        left_event_id: int,
        left_track_id: int,
        right_event_id: int,
        right_track_id: int,
    ) -> bool:
        left = (int(left_event_id), int(left_track_id))
        right = (int(right_event_id), int(right_track_id))
        if left > right:
            left, right = right, left
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                select 1 from reid_pair_reviews
                where left_event_id = ? and left_track_id = ?
                  and right_event_id = ? and right_track_id = ?
                """,
                (*left, *right),
            ).fetchone()
        return row is not None

    def record_pair_review(
        self,
        *,
        left_event_id: int,
        left_track_id: int,
        right_event_id: int,
        right_track_id: int,
        decision: str,
        similarity: float | None = None,
        left_sample_id: str = "",
        right_sample_id: str = "",
    ) -> None:
        left = (int(left_event_id), int(left_track_id), str(left_sample_id or ""))
        right = (int(right_event_id), int(right_track_id), str(right_sample_id or ""))
        if left[:2] > right[:2]:
            left, right = right, left
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                insert into reid_pair_reviews (
                    left_event_id, left_track_id, right_event_id, right_track_id,
                    left_sample_id, right_sample_id, decision, similarity, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(left_event_id, left_track_id, right_event_id, right_track_id)
                do update set
                    decision=excluded.decision,
                    similarity=excluded.similarity,
                    left_sample_id=excluded.left_sample_id,
                    right_sample_id=excluded.right_sample_id,
                    created_at=excluded.created_at
                """,
                (
                    left[0],
                    left[1],
                    right[0],
                    right[1],
                    left[2],
                    right[2],
                    str(decision),
                    None if similarity is None else float(similarity),
                    _utc_now(),
                ),
            )

    def merge_track_identities(
        self,
        *,
        keep_person_id: int,
        absorb_person_id: int,
    ) -> int:
        if int(keep_person_id) == int(absorb_person_id):
            return 0
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                update reid_samples
                set assigned_person_id = ?,
                    assignment_source = 'manual',
                    assignment_confidence = 1.0,
                    review_status = case
                        when review_status = 'rejected' then review_status
                        else 'confirmed'
                    end
                where assigned_person_id = ?
                """,
                (int(keep_person_id), int(absorb_person_id)),
            )
            return int(cursor.rowcount or 0)

    def assign_track_identity(
        self,
        event_id: int,
        track_id: int,
        person_id: int,
        *,
        source: str = "manual",
        review_status: str = "confirmed",
    ) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                update reid_samples
                set assigned_person_id = ?,
                    assignment_source = ?,
                    assignment_confidence = 1.0,
                    review_status = ?
                where event_id = ? and track_id = ?
                  and review_status != 'rejected'
                """,
                (
                    int(person_id),
                    str(source),
                    str(review_status),
                    int(event_id),
                    int(track_id),
                ),
            )
            return int(cursor.rowcount or 0)

    def set_sample_review_status(
        self,
        sample_id: str,
        review_status: str,
        *,
        assignment_source: str | None = None,
    ) -> bool:
        assignments = ["review_status = ?"]
        parameters: list[Any] = [str(review_status)]
        if assignment_source is not None:
            assignments.append("assignment_source = ?")
            parameters.append(str(assignment_source))
        parameters.append(str(sample_id))
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"""
                update reid_samples
                set {', '.join(assignments)}
                where sample_id = ?
                """,
                parameters,
            )
            return int(cursor.rowcount or 0) > 0

    def reject_track(self, event_id: int, track_id: int) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                update reid_samples
                set review_status = 'rejected',
                    assignment_source = 'manual'
                where event_id = ? and track_id = ?
                """,
                (int(event_id), int(track_id)),
            )
            return int(cursor.rowcount or 0)

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
            pending_pairs = connection.execute(
                "select count(*) as count from reid_pair_reviews"
            ).fetchone()
            auto_samples = connection.execute(
                "select count(*) as count from reid_samples where review_status = 'auto'"
            ).fetchone()
        return {
            "database_path": str(self.db_path),
            "crops_root": str(self.crops_root),
            "samples": int(samples["count"] if samples else 0),
            "identities": int(identities["count"] if identities else 0),
            "auto_samples": int(auto_samples["count"] if auto_samples else 0),
            "pair_reviews": int(pending_pairs["count"] if pending_pairs else 0),
            "newest_at": str(newest["newest_at"] or "") if newest else "",
        }

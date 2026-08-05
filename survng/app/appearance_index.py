from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class AppearanceIndex:
    """Durable, model-versioned appearance vectors for incident investigation."""

    MAX_EMBEDDING_DIMENSIONS = 8192
    MAX_CANDIDATE_ROWS = 10000

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
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
                create table if not exists appearance_embeddings (
                    id integer primary key autoincrement,
                    event_id integer not null,
                    camera_id text not null,
                    track_id integer not null,
                    label text not null,
                    model_kind text not null,
                    model_fingerprint text not null,
                    embedding_size integer not null,
                    embedding_blob blob not null,
                    match_threshold real not null,
                    observation_count integer not null default 0,
                    quality real not null default 0,
                    first_seen text not null default '',
                    last_seen text not null default '',
                    source text not null default 'tracking_multiframe',
                    created_at text not null,
                    foreign key(event_id) references events(id) on delete cascade,
                    unique(event_id, track_id, model_kind, model_fingerprint)
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("pragma table_info(appearance_embeddings)")
            }
            if "source" not in columns:
                connection.execute(
                    "alter table appearance_embeddings add column source text not null default 'tracking_multiframe'"
                )
            connection.execute(
                "create index if not exists idx_appearance_event on appearance_embeddings(event_id)"
            )
            connection.execute(
                """
                create index if not exists idx_appearance_model_created
                on appearance_embeddings(model_kind, model_fingerprint, created_at desc)
                """
            )

    @staticmethod
    def _normalized_embedding(value: object) -> np.ndarray | None:
        try:
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if not 0 < vector.size <= AppearanceIndex.MAX_EMBEDDING_DIMENSIONS:
            return None
        norm = float(np.linalg.norm(vector))
        if not np.all(np.isfinite(vector)) or not np.isfinite(norm) or norm <= 1e-9:
            return None
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    def _prepare_records(
        self,
        event_id: int,
        camera_id: str,
        records: Iterable[dict[str, Any]],
    ) -> list[tuple[Any, ...]]:
        prepared: list[tuple[Any, ...]] = []
        for record in records:
            vector = self._normalized_embedding(record.get("embedding"))
            fingerprint = str(record.get("model_fingerprint") or "").strip()
            model_kind = str(record.get("model_kind") or "").strip().lower()
            label = str(record.get("label") or "").strip().lower()
            if vector is None or not fingerprint or model_kind not in {"person", "vehicle"} or not label:
                continue
            default_threshold = 0.7 if model_kind == "person" else 0.8
            match_threshold = min(
                1.0,
                max(
                    0.1,
                    float(record.get("match_threshold") or default_threshold),
                ),
            )
            prepared.append((
                int(event_id),
                str(camera_id),
                int(record.get("track_id") or 0),
                label,
                model_kind,
                fingerprint,
                int(vector.size),
                vector.tobytes(),
                match_threshold,
                max(0, int(record.get("observation_count") or 0)),
                min(1.0, max(0.0, float(record.get("quality") or 0.0))),
                str(record.get("first_seen") or ""),
                str(record.get("last_seen") or ""),
                str(record.get("source") or "tracking_multiframe"),
                str(record.get("created_at") or record.get("last_seen") or ""),
            ))
        return prepared

    def replace_event(
        self,
        event_id: int,
        camera_id: str,
        records: Iterable[dict[str, Any]],
    ) -> int:
        """Atomically replace an event's valid vectors without exposing them in event JSON."""
        prepared = self._prepare_records(event_id, camera_id, records)
        if not prepared:
            return 0
        with self._lock, self._connect() as connection:
            connection.execute(
                "delete from appearance_embeddings where event_id = ?",
                (int(event_id),),
            )
            connection.executemany(
                """
                insert into appearance_embeddings (
                    event_id, camera_id, track_id, label, model_kind,
                    model_fingerprint, embedding_size, embedding_blob,
                    match_threshold, observation_count, quality,
                    first_seen, last_seen, source, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
        return len(prepared)

    def append_event(
        self,
        event_id: int,
        camera_id: str,
        records: Iterable[dict[str, Any]],
    ) -> int:
        """Add fallback vectors without replacing stronger multi-frame evidence."""
        prepared = self._prepare_records(event_id, camera_id, records)
        if not prepared:
            return 0
        with self._lock, self._connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                insert or ignore into appearance_embeddings (
                    event_id, camera_id, track_id, label, model_kind,
                    model_fingerprint, embedding_size, embedding_blob,
                    match_threshold, observation_count, quality,
                    first_seen, last_seen, source, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
            return connection.total_changes - before

    def has_event(self, event_id: int) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "select 1 from appearance_embeddings where event_id = ? limit 1",
                (int(event_id),),
            ).fetchone()
        return row is not None

    @staticmethod
    def _vector(row: sqlite3.Row) -> np.ndarray | None:
        size = int(row["embedding_size"] or 0)
        raw = row["embedding_blob"]
        if size <= 0 or not isinstance(raw, bytes) or len(raw) != size * 4:
            return None
        vector = np.frombuffer(raw, dtype=np.float32)
        if vector.size != size or not np.all(np.isfinite(vector)):
            return None
        return vector

    def matches(
        self,
        event_id: int,
        *,
        start_at: str | None = None,
        end_at: str | None = None,
        cross_camera_only: bool = True,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Return strongest compatible incident matches; embeddings never leave this class."""
        bounded_limit = max(1, min(int(limit), 100))
        with self._connect() as connection:
            anchors = connection.execute(
                "select * from appearance_embeddings where event_id = ?",
                (int(event_id),),
            ).fetchall()
            if not anchors:
                return []
            clauses = ["candidate.event_id != ?"]
            parameters: list[Any] = [int(event_id)]
            if cross_camera_only:
                clauses.append("candidate.camera_id != ?")
                parameters.append(str(anchors[0]["camera_id"]))
            if start_at:
                clauses.append("candidate.created_at >= ?")
                parameters.append(str(start_at))
            if end_at:
                clauses.append("candidate.created_at <= ?")
                parameters.append(str(end_at))
            model_pairs = sorted({
                (str(row["model_kind"]), str(row["model_fingerprint"]))
                for row in anchors
            })
            model_clause = " or ".join(
                "(candidate.model_kind = ? and candidate.model_fingerprint = ?)"
                for _pair in model_pairs
            )
            for kind, fingerprint in model_pairs:
                parameters.extend((kind, fingerprint))
            parameters.append(self.MAX_CANDIDATE_ROWS)
            candidates = connection.execute(
                f"""
                select candidate.*, event.created_at as event_created_at
                from appearance_embeddings as candidate
                join events as event on event.id = candidate.event_id
                where {' and '.join(clauses)} and ({model_clause})
                order by candidate.created_at desc, candidate.id desc
                limit ?
                """,
                parameters,
            ).fetchall()

        grouped_anchors: dict[tuple[str, str, int], list[tuple[sqlite3.Row, np.ndarray]]] = {}
        grouped_candidates: dict[tuple[str, str, int], list[tuple[sqlite3.Row, np.ndarray]]] = {}
        for rows, target in (
            (anchors, grouped_anchors),
            (candidates, grouped_candidates),
        ):
            for row in rows:
                vector = self._vector(row)
                if vector is None:
                    continue
                key = (
                    str(row["model_kind"]),
                    str(row["model_fingerprint"]),
                    int(row["embedding_size"]),
                )
                target.setdefault(key, []).append((row, vector))

        best_by_event: dict[int, dict[str, Any]] = {}
        for key, anchor_group in grouped_anchors.items():
            candidate_group = grouped_candidates.get(key, [])
            if not candidate_group:
                continue
            anchor_matrix = np.stack([item[1] for item in anchor_group])
            candidate_matrix = np.stack([item[1] for item in candidate_group])
            similarities = anchor_matrix @ candidate_matrix.T
            best_anchor_indices = np.argmax(similarities, axis=0)
            for candidate_index, (candidate, _vector) in enumerate(candidate_group):
                anchor = anchor_group[int(best_anchor_indices[candidate_index])][0]
                similarity = min(
                    1.0,
                    max(
                        -1.0,
                        float(similarities[int(best_anchor_indices[candidate_index]), candidate_index]),
                    ),
                )
                threshold = max(
                    float(anchor["match_threshold"] or 0.0),
                    float(candidate["match_threshold"] or 0.0),
                )
                candidate_event_id = int(candidate["event_id"])
                current = best_by_event.get(candidate_event_id)
                if current is not None and float(current["similarity"]) >= similarity:
                    continue
                best_by_event[candidate_event_id] = {
                    "event_id": candidate_event_id,
                    "camera_id": str(candidate["camera_id"]),
                    "created_at": str(candidate["event_created_at"] or candidate["created_at"]),
                    "anchor_track_id": int(anchor["track_id"]),
                    "candidate_track_id": int(candidate["track_id"]),
                    "anchor_label": str(anchor["label"]),
                    "candidate_label": str(candidate["label"]),
                    "model_kind": str(candidate["model_kind"]),
                    "similarity": round(similarity, 4),
                    "threshold": round(threshold, 4),
                    "visually_similar": similarity >= threshold,
                }
        return sorted(
            best_by_event.values(),
            key=lambda item: (-float(item["similarity"]), str(item["created_at"])),
        )[:bounded_limit]

    def status(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                select count(*) as vectors, count(distinct event_id) as events,
                       min(created_at) as oldest_at, max(created_at) as newest_at
                from appearance_embeddings
                """
            ).fetchone()
        return {
            "vectors": int(row["vectors"] or 0),
            "events": int(row["events"] or 0),
            "oldest_at": str(row["oldest_at"] or ""),
            "newest_at": str(row["newest_at"] or ""),
        }

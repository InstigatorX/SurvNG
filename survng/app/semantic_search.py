from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from .config import SemanticSearchConfig


@dataclass(frozen=True)
class SemanticModelIdentity:
    implementation: str
    model_fingerprint: str
    preprocessing_fingerprint: str
    dimensions: int

    @property
    def generation(self) -> str:
        return f"{self.model_fingerprint}:{self.preprocessing_fingerprint}"


@dataclass(frozen=True)
class SemanticEvidence:
    event_id: int
    camera_id: str
    captured_at: str
    source_kind: str
    source_key: str
    image_path: str
    object_label: str = ""
    bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class SemanticSearchHit:
    event_id: int
    camera_id: str
    captured_at: str
    source_kind: str
    source_key: str
    image_path: str
    object_label: str
    bbox: tuple[int, int, int, int] | None
    score: float


class SemanticEncoder(Protocol):
    @property
    def identity(self) -> SemanticModelIdentity: ...

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray: ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...

    def close(self) -> None: ...


def normalized_matrix(value: object, *, max_dimensions: int = 8192) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or not 0 < matrix.shape[1] <= max_dimensions:
        raise ValueError("semantic embeddings must be a non-empty 2D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("semantic embeddings must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-9):
        raise ValueError("semantic embeddings must have non-zero magnitude")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class SemanticIndex:
    """Durable multi-generation semantic evidence index.

    Embeddings remain local and are stored as normalized float16 vectors. Old
    generations remain searchable during background migration to a new model.
    """

    MAX_CANDIDATE_ROWS = 250_000

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
                create table if not exists semantic_embeddings (
                    id integer primary key autoincrement,
                    event_id integer not null,
                    camera_id text not null,
                    captured_at text not null,
                    source_kind text not null,
                    source_key text not null,
                    image_path text not null default '',
                    object_label text not null default '',
                    bbox_json text not null default '',
                    implementation text not null,
                    model_fingerprint text not null,
                    preprocessing_fingerprint text not null,
                    embedding_size integer not null,
                    embedding_blob blob not null,
                    created_at text not null,
                    foreign key(event_id) references events(id) on delete cascade,
                    unique(
                        event_id, source_kind, source_key,
                        model_fingerprint, preprocessing_fingerprint
                    )
                )
                """
            )
            connection.execute(
                """
                create index if not exists idx_semantic_generation_time
                on semantic_embeddings(
                    model_fingerprint, preprocessing_fingerprint, captured_at desc
                )
                """
            )
            connection.execute(
                """
                create index if not exists idx_semantic_event
                on semantic_embeddings(event_id)
                """
            )

    def upsert(
        self,
        evidence: Iterable[SemanticEvidence],
        embeddings: object,
        identity: SemanticModelIdentity,
    ) -> int:
        records = list(evidence)
        if not records:
            return 0
        vectors = normalized_matrix(embeddings)
        if vectors.shape != (len(records), identity.dimensions):
            raise ValueError("semantic evidence and embedding dimensions do not match")
        now = datetime.now(timezone.utc).isoformat()
        prepared: list[tuple[Any, ...]] = []
        for record, vector in zip(records, vectors, strict=True):
            bbox_json = json.dumps(record.bbox, separators=(",", ":")) if record.bbox else ""
            prepared.append((
                int(record.event_id), str(record.camera_id), str(record.captured_at),
                str(record.source_kind), str(record.source_key), str(record.image_path),
                str(record.object_label).strip().lower(), bbox_json,
                identity.implementation, identity.model_fingerprint,
                identity.preprocessing_fingerprint, identity.dimensions,
                np.ascontiguousarray(vector, dtype=np.float16).tobytes(), now,
            ))
        with self._lock, self._connect() as connection:
            connection.executemany(
                """
                insert into semantic_embeddings (
                    event_id, camera_id, captured_at, source_kind, source_key,
                    image_path, object_label, bbox_json, implementation,
                    model_fingerprint, preprocessing_fingerprint, embedding_size,
                    embedding_blob, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(
                    event_id, source_kind, source_key,
                    model_fingerprint, preprocessing_fingerprint
                ) do update set
                    camera_id=excluded.camera_id,
                    captured_at=excluded.captured_at,
                    image_path=excluded.image_path,
                    object_label=excluded.object_label,
                    bbox_json=excluded.bbox_json,
                    embedding_size=excluded.embedding_size,
                    embedding_blob=excluded.embedding_blob,
                    created_at=excluded.created_at
                """,
                prepared,
            )
        return len(prepared)

    def search(
        self,
        query_embedding: object,
        identity: SemanticModelIdentity,
        *,
        camera_ids: Sequence[str] = (),
        object_labels: Sequence[str] = (),
        start_at: str = "",
        end_at: str = "",
        limit: int = 100,
        minimum_score: float = -1.0,
    ) -> list[SemanticSearchHit]:
        query = normalized_matrix(query_embedding)
        if query.shape != (1, identity.dimensions):
            raise ValueError("semantic query dimensions do not match the model")
        clauses = ["model_fingerprint = ?", "preprocessing_fingerprint = ?"]
        parameters: list[Any] = [
            identity.model_fingerprint,
            identity.preprocessing_fingerprint,
        ]
        if camera_ids:
            normalized_cameras = sorted({str(value) for value in camera_ids if str(value)})
            clauses.append(f"camera_id in ({','.join('?' for _ in normalized_cameras)})")
            parameters.extend(normalized_cameras)
        if object_labels:
            normalized_labels = sorted({str(value).strip().lower() for value in object_labels if str(value).strip()})
            clauses.append(f"object_label in ({','.join('?' for _ in normalized_labels)})")
            parameters.extend(normalized_labels)
        if start_at:
            clauses.append("captured_at >= ?")
            parameters.append(str(start_at))
        if end_at:
            clauses.append("captured_at <= ?")
            parameters.append(str(end_at))
        parameters.append(self.MAX_CANDIDATE_ROWS)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                select * from semantic_embeddings
                where {' and '.join(clauses)}
                order by captured_at desc, id desc
                limit ?
                """,
                parameters,
            ).fetchall()
        candidates: list[tuple[sqlite3.Row, np.ndarray]] = []
        for row in rows:
            size = int(row["embedding_size"] or 0)
            raw = row["embedding_blob"]
            if size != identity.dimensions or not isinstance(raw, bytes) or len(raw) != size * 2:
                continue
            vector = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            if np.all(np.isfinite(vector)):
                candidates.append((row, vector))
        if not candidates:
            return []
        scores = np.stack([item[1] for item in candidates]) @ query[0]
        ordered = np.argsort(scores)[::-1]
        hits: list[SemanticSearchHit] = []
        for candidate_index in ordered:
            score = float(scores[candidate_index])
            if score < minimum_score:
                continue
            row = candidates[int(candidate_index)][0]
            bbox: tuple[int, int, int, int] | None = None
            try:
                values = json.loads(str(row["bbox_json"] or ""))
                if isinstance(values, list) and len(values) == 4:
                    bbox = tuple(int(value) for value in values)  # type: ignore[assignment]
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            hits.append(SemanticSearchHit(
                event_id=int(row["event_id"]), camera_id=str(row["camera_id"]),
                captured_at=str(row["captured_at"]), source_kind=str(row["source_kind"]),
                source_key=str(row["source_key"]), image_path=str(row["image_path"]),
                object_label=str(row["object_label"]), bbox=bbox, score=score,
            ))
            if len(hits) >= max(1, min(int(limit), 500)):
                break
        return hits

    def coverage(self, identity: SemanticModelIdentity | None = None) -> dict[str, int]:
        clauses = ""
        parameters: tuple[Any, ...] = ()
        if identity is not None:
            clauses = "where model_fingerprint = ? and preprocessing_fingerprint = ?"
            parameters = (identity.model_fingerprint, identity.preprocessing_fingerprint)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                select count(*) as evidence_count, count(distinct event_id) as event_count
                from semantic_embeddings {clauses}
                """,
                parameters,
            ).fetchone()
        return {
            "evidence_count": int(row["evidence_count"] if row else 0),
            "event_count": int(row["event_count"] if row else 0),
        }


def fingerprint_model_package(model_dir: Path) -> str:
    """Fingerprint local model artifacts without reading multi-gigabyte weights at startup."""
    digest = hashlib.sha256()
    for path in sorted(item for item in Path(model_dir).rglob("*") if item.is_file()):
        relative = path.relative_to(model_dir).as_posix()
        stat = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        with path.open("rb") as handle:
            digest.update(handle.read(65536))
    return digest.hexdigest()[:24]


def load_semantic_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = Path(model_dir) / "semantic_model.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"semantic model manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid semantic model manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("semantic model manifest must contain a JSON object")
    return payload


class DisabledSemanticSearch:
    """Stable no-op service used when semantic search is disabled."""

    def __init__(self, config: SemanticSearchConfig, index: SemanticIndex) -> None:
        self.config = config.model_copy(deep=True)
        self.index = index

    def status(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "state": "disabled",
            "implementation": self.config.implementation,
            **self.index.coverage(),
        }

    def close(self) -> None:
        return


class UnavailableSemanticSearch(DisabledSemanticSearch):
    """Failure-isolated placeholder for an enabled but unusable model package."""

    def __init__(
        self,
        config: SemanticSearchConfig,
        index: SemanticIndex,
        reason: str,
    ) -> None:
        super().__init__(config, index)
        self.reason = str(reason)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "state": "unavailable",
            "implementation": self.config.implementation,
            "reason": self.reason,
            **self.index.coverage(),
        }


def build_semantic_search(
    config: SemanticSearchConfig,
    index: SemanticIndex,
) -> DisabledSemanticSearch:
    """Build safely; semantic failures must never prevent camera startup."""
    if not config.enabled:
        return DisabledSemanticSearch(config, index)
    model_dir = Path(config.model_dir)
    if not config.model_dir:
        return UnavailableSemanticSearch(config, index, "model directory is not configured")
    if not model_dir.is_dir():
        return UnavailableSemanticSearch(config, index, f"model directory does not exist: {model_dir}")
    try:
        load_semantic_manifest(model_dir)
    except RuntimeError as exc:
        return UnavailableSemanticSearch(config, index, str(exc))
    return UnavailableSemanticSearch(
        config,
        index,
        "semantic inference runtime has not started",
    )

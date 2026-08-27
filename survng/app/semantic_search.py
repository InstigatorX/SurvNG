from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import logging
import queue
import itertools
import multiprocessing
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import cv2

from .config import SemanticSearchConfig
from .incident_utils import event_snapshot_path
from .media_storage import MediaStorageRegistry
from .openclip_tokenizer import OpenClipBpeTokenizer

LOGGER = logging.getLogger("uvicorn.error")
SEMANTIC_WORKER_START_TIMEOUT_SECONDS = 60.0
SEMANTIC_WORKER_REQUEST_TIMEOUT_SECONDS = 60.0
SEMANTIC_WORKER_RETRY_INITIAL_SECONDS = 15.0
SEMANTIC_WORKER_RETRY_MAX_SECONDS = 300.0
SEMANTIC_WORKER_CONFIGURED_DEVICE_ATTEMPTS = 2
SEMANTIC_WORKER_FALLBACK_DELAY_SECONDS = 1.0
MODEL_FINGERPRINT_CHUNK_SIZE = 1024 * 1024
SEMANTIC_BACKFILL_RETRY_SECONDS = 5.0


class SemanticInferenceError(RuntimeError):
    """A valid worker response reporting a deterministic inference failure."""


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
    rank_score: float | None = None
    match_strength: str = "visual_similarity"
    component_scores: Mapping[str, float] | None = None


@dataclass(frozen=True)
class SemanticQueryPlan:
    """A bounded set of prompts used to evaluate a compound visual query."""

    original: str
    prompts: Mapping[str, str]
    required: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()

    @property
    def composed(self) -> bool:
        return bool(self.required)


SEMANTIC_COLORS = (
    "black", "blue", "brown", "gold", "gray", "green", "grey",
    "orange", "red", "silver", "white", "yellow",
)
SEMANTIC_VEHICLE_WORDS = frozenset({
    "bus", "car", "cars", "motorcycle", "pickup", "suv", "truck",
    "trucks", "van", "vehicle", "vehicles",
})


def semantic_query_plan(query: str) -> SemanticQueryPlan:
    """Decompose color-qualified vehicle searches without pretending to parse NLP.

    This intentionally handles only a high-confidence grammar. Unknown queries
    retain the original single-vector behavior instead of receiving guessed
    semantics.
    """
    original = " ".join(str(query).strip().split())
    lowered = original.lower()
    words = lowered.replace("-", " ").split()
    colors = [color for color in SEMANTIC_COLORS if color in words]
    if len(colors) != 1 or not SEMANTIC_VEHICLE_WORDS.intersection(words):
        return SemanticQueryPlan(original, {"full": original})
    color = colors[0]
    subject_words = [word for word in words if word != color]
    while subject_words and subject_words[0] in {"a", "an", "the"}:
        subject_words.pop(0)
    subject = " ".join(subject_words).strip()
    if not subject:
        return SemanticQueryPlan(original, {"full": original})
    prompts: dict[str, str] = {
        "full": original,
        "subject": f"a {subject}",
        "attribute": f"a {color} vehicle",
    }
    contradictions: list[str] = []
    for other in SEMANTIC_COLORS:
        if other in {color, "grey" if color == "gray" else "gray" if color == "grey" else ""}:
            continue
        name = f"not_{other}"
        prompts[name] = f"a {other} vehicle"
        contradictions.append(name)
    return SemanticQueryPlan(
        original,
        prompts,
        required=("subject", "attribute"),
        contradictions=tuple(contradictions),
    )


def semantic_event_objects(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return actual labeled detections, excluding motion/audit metadata."""
    raw_objects: object = event.get("objects")
    if not isinstance(raw_objects, list):
        try:
            raw_objects = json.loads(str(event.get("objects_json") or "[]"))
        except (TypeError, json.JSONDecodeError):
            raw_objects = []
    if not isinstance(raw_objects, list):
        return []
    return [
        item for item in raw_objects
        if isinstance(item, dict)
        and str(item.get("label") or "").strip()
        and item.get("snapshot_visible") is not False
    ]


def semantic_object_bbox(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Return a finite, positive detection box without assuming snapshot dimensions."""
    raw_bbox = item.get("bbox") or item.get("box")
    if isinstance(raw_bbox, dict):
        raw_coordinates = tuple(raw_bbox.get(key) for key in ("x1", "y1", "x2", "y2"))
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        raw_coordinates = tuple(raw_bbox)
    else:
        return None
    try:
        coordinates = tuple(float(value) for value in raw_coordinates)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(np.isfinite(value) for value in coordinates):
        return None
    x1, y1, x2, y2 = coordinates
    return coordinates if x2 > x1 and y2 > y1 else None


def semantic_object_crop_candidates(
    objects: Sequence[dict[str, Any]],
    limit: int,
) -> list[tuple[int, dict[str, Any], tuple[float, float, float, float]]]:
    """Choose stable, highest-confidence crop candidates within the configured cap."""
    candidates: list[
        tuple[float, int, dict[str, Any], tuple[float, float, float, float]]
    ] = []
    for index, item in enumerate(objects):
        coordinates = semantic_object_bbox(item)
        if coordinates is None:
            continue
        try:
            confidence = float(item.get("confidence") or 0)
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0
        candidates.append(
            (confidence if np.isfinite(confidence) else 0.0, index, item, coordinates)
        )
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    return [
        (index, item, coordinates)
        for _confidence, index, item, coordinates in candidates[:max(0, int(limit))]
    ]


def semantic_crop_source_key(index: int, item: dict[str, Any]) -> str:
    label = str(item.get("label") or "").strip().lower()
    signature = json.dumps(
        {
            "bbox": semantic_object_bbox(item),
            "width": item.get("detection_frame_width"),
            "height": item.get("detection_frame_height"),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"{label}:{index}:{hashlib.sha256(signature).hexdigest()[:12]}"


def positive_dimension(value: object, fallback: int) -> float:
    try:
        number = float(value or fallback)
    except (TypeError, ValueError, OverflowError):
        return float(fallback)
    return number if np.isfinite(number) and number > 0 else float(fallback)


class SemanticEncoder(Protocol):
    @property
    def identity(self) -> SemanticModelIdentity: ...

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray: ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...

    def close(self) -> None: ...


class SemanticTokenizer(Protocol):
    def __call__(self, texts: Sequence[str]) -> dict[str, np.ndarray]: ...


class OpenClipTokenizerAdapter:
    """Expose the legacy OpenCLIP tokenizer through the manifest input contract."""

    def __init__(self, path: Path, max_length: int) -> None:
        self._tokenizer = OpenClipBpeTokenizer(path, context_length=max_length)

    def __call__(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        return {"input_ids": np.asarray(self._tokenizer(list(texts)), dtype=np.int64)}


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

    Embeddings remain local and are stored as normalized float16 vectors. Model
    generations remain isolated so incompatible vectors are never compared.
    """

    MAX_CANDIDATE_ROWS = 50_000

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
        query_plan: SemanticQueryPlan | None = None,
        camera_ids: Sequence[str] = (),
        object_labels: Sequence[str] = (),
        source_kinds: Sequence[str] = (),
        start_at: str = "",
        end_at: str = "",
        limit: int = 100,
        minimum_score: float = -1.0,
    ) -> list[SemanticSearchHit]:
        query = normalized_matrix(query_embedding)
        plan = query_plan or SemanticQueryPlan("", {"full": ""})
        component_names = tuple(plan.prompts)
        if query.shape != (len(component_names), identity.dimensions):
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
        if source_kinds:
            normalized_source_kinds = sorted({
                str(value).strip().lower()
                for value in source_kinds
                if str(value).strip()
            })
            if normalized_source_kinds:
                clauses.append(
                    f"source_kind in ({','.join('?' for _ in normalized_source_kinds)})"
                )
                parameters.extend(normalized_source_kinds)
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
        component_matrix = np.stack([item[1] for item in candidates]) @ query.T
        component_indexes = {name: index for index, name in enumerate(component_names)}
        full_scores = component_matrix[:, component_indexes["full"]]
        if plan.composed:
            required_scores = np.stack([
                component_matrix[:, component_indexes[name]] for name in plan.required
            ], axis=1)
            weakest_required = required_scores.min(axis=1)
            contradiction_scores = np.stack([
                component_matrix[:, component_indexes[name]]
                for name in plan.contradictions
            ], axis=1)
            strongest_contradiction = contradiction_scores.max(axis=1)
            attribute_scores = component_matrix[:, component_indexes["attribute"]]
            contradiction_margin = attribute_scores - strongest_contradiction
            rank_scores = (
                full_scores * 0.45
                + required_scores.mean(axis=1) * 0.55
                + np.minimum(0.04, contradiction_margin) * 0.35
            )
            eligible = contradiction_margin >= -0.005
            if np.any(eligible):
                best_rank = float(rank_scores[eligible].max())
                best_required = required_scores[eligible].max(axis=0)
                eligible &= rank_scores >= best_rank - 0.05
                eligible &= np.all(required_scores >= best_required - 0.07, axis=1)
            else:
                best_rank = 1.0
        else:
            weakest_required = full_scores
            rank_scores = full_scores
            eligible = np.ones(len(candidates), dtype=bool)
            best_rank = float(rank_scores.max())
        ordered = np.argsort(rank_scores)[::-1]
        hits: list[SemanticSearchHit] = []
        for candidate_index in ordered:
            if not bool(eligible[candidate_index]):
                continue
            score = float(full_scores[candidate_index])
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
            rank_score = float(rank_scores[candidate_index])
            distance = best_rank - rank_score
            match_strength = (
                "strong_match" if plan.composed and distance <= 0.02
                else "possible_match" if plan.composed
                else "visual_similarity"
            )
            component_scores = {
                name: round(float(component_matrix[candidate_index, index]), 6)
                for name, index in component_indexes.items()
                if name == "full" or name in plan.required
            }
            hits.append(SemanticSearchHit(
                event_id=int(row["event_id"]), camera_id=str(row["camera_id"]),
                captured_at=str(row["captured_at"]), source_kind=str(row["source_kind"]),
                source_key=str(row["source_key"]), image_path=str(row["image_path"]),
                object_label=str(row["object_label"]), bbox=bbox, score=score,
                rank_score=rank_score,
                match_strength=match_strength,
                component_scores=component_scores,
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

    def clone_image_generation(
        self,
        source: SemanticModelIdentity,
        target: SemanticModelIdentity,
    ) -> int:
        """Reuse stored image vectors after external image-tower validation.

        Semantic embeddings contain image evidence only. Callers must first
        prove that image model artifacts and preprocessing are identical.
        """
        if source.dimensions != target.dimensions:
            raise ValueError("semantic generation dimensions do not match")
        if source.generation == target.generation:
            return 0
        with self._lock, self._connect() as connection:
            before = connection.total_changes
            connection.execute(
                """
                insert into semantic_embeddings (
                    event_id, camera_id, captured_at, source_kind, source_key,
                    image_path, object_label, bbox_json, implementation,
                    model_fingerprint, preprocessing_fingerprint, embedding_size,
                    embedding_blob, created_at
                )
                select event_id, camera_id, captured_at, source_kind, source_key,
                    image_path, object_label, bbox_json, ?, ?, ?, ?,
                    embedding_blob, ?
                from semantic_embeddings
                where model_fingerprint = ? and preprocessing_fingerprint = ?
                on conflict(
                    event_id, source_kind, source_key,
                    model_fingerprint, preprocessing_fingerprint
                ) do nothing
                """,
                (
                    target.implementation,
                    target.model_fingerprint,
                    target.preprocessing_fingerprint,
                    target.dimensions,
                    datetime.now(timezone.utc).isoformat(),
                    source.model_fingerprint,
                    source.preprocessing_fingerprint,
                ),
            )
            return connection.total_changes - before

    def event_indexed(self, event_id: int, identity: SemanticModelIdentity) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                select 1 from semantic_embeddings
                where event_id = ? and model_fingerprint = ?
                    and preprocessing_fingerprint = ? limit 1
                """,
                (int(event_id), identity.model_fingerprint, identity.preprocessing_fingerprint),
            ).fetchone()
        return row is not None

    def event_source_indexed(
        self,
        event_id: int,
        identity: SemanticModelIdentity,
        source_kind: str,
    ) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                select 1 from semantic_embeddings
                where event_id = ? and model_fingerprint = ?
                    and preprocessing_fingerprint = ? and source_kind = ? limit 1
                """,
                (int(event_id), identity.model_fingerprint,
                 identity.preprocessing_fingerprint, str(source_kind)),
            ).fetchone()
        return row is not None

    def event_source_keys(
        self,
        event_id: int,
        identity: SemanticModelIdentity,
        source_kind: str,
    ) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select source_key from semantic_embeddings
                where event_id = ? and model_fingerprint = ?
                    and preprocessing_fingerprint = ? and source_kind = ?
                """,
                (
                    int(event_id),
                    identity.model_fingerprint,
                    identity.preprocessing_fingerprint,
                    str(source_kind),
                ),
            ).fetchall()
        return {str(row["source_key"]) for row in rows}

    def reconcile_event_source_keys(
        self,
        event_id: int,
        identity: SemanticModelIdentity,
        source_kind: str,
        desired_keys: set[str],
    ) -> int:
        """Delete stale evidence after an event's objects or crop cap changes."""
        clauses = [
            "event_id = ?",
            "model_fingerprint = ?",
            "preprocessing_fingerprint = ?",
            "source_kind = ?",
        ]
        parameters: list[Any] = [
            int(event_id),
            identity.model_fingerprint,
            identity.preprocessing_fingerprint,
            str(source_kind),
        ]
        if desired_keys:
            clauses.append(f"source_key not in ({','.join('?' for _ in desired_keys)})")
            parameters.extend(sorted(desired_keys))
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"delete from semantic_embeddings where {' and '.join(clauses)}",
                parameters,
            )
        return max(0, int(cursor.rowcount or 0))

    def delete_generation_source(
        self,
        identity: SemanticModelIdentity,
        source_kind: str,
    ) -> int:
        """Remove a disabled evidence source for the active model generation."""
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                delete from semantic_embeddings
                where model_fingerprint = ? and preprocessing_fingerprint = ?
                    and source_kind = ?
                """,
                (
                    identity.model_fingerprint,
                    identity.preprocessing_fingerprint,
                    str(source_kind),
                ),
            )
        return max(0, int(cursor.rowcount or 0))

    def delete_event(self, event_id: int) -> int:
        """Remove semantic evidence for an event that is not object-searchable."""
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "delete from semantic_embeddings where event_id = ?",
                (int(event_id),),
            )
        return max(0, int(cursor.rowcount or 0))

    def indexed_event_ids(self) -> set[int]:
        """Return event IDs with any semantic evidence, across model generations."""
        with self._connect() as connection:
            rows = connection.execute(
                "select distinct event_id from semantic_embeddings"
            ).fetchall()
        return {int(row["event_id"]) for row in rows}


def fingerprint_model_package(model_dir: Path) -> str:
    """Fingerprint every byte of local model artifacts to prevent stale generations."""
    digest = hashlib.sha256()
    for path in sorted(item for item in Path(model_dir).rglob("*") if item.is_file()):
        relative = path.relative_to(model_dir).as_posix()
        stat = path.stat()
        encoded_relative = relative.encode("utf-8")
        digest.update(len(encoded_relative).to_bytes(8, "big"))
        digest.update(encoded_relative)
        digest.update(stat.st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(MODEL_FINGERPRINT_CHUNK_SIZE):
                digest.update(chunk)
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

    def start(self, event_store: Any, storage_dir: Path, media_storage: MediaStorageRegistry | None = None) -> None:
        return

    def queue_event(self, event: dict[str, Any]) -> bool:
        return False

    def search_text(self, query: str, **filters: Any) -> list[SemanticSearchHit]:
        raise RuntimeError("semantic search is disabled")

    def search_image(
        self,
        image: np.ndarray,
        **filters: Any,
    ) -> list[SemanticSearchHit]:
        raise RuntimeError("semantic search is disabled")

    def search_event_object(
        self,
        event: dict[str, Any],
        object_index: int,
        **filters: Any,
    ) -> list[SemanticSearchHit]:
        raise RuntimeError("semantic search is disabled")


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
        manifest = load_semantic_manifest(model_dir)
    except RuntimeError as exc:
        return UnavailableSemanticSearch(config, index, str(exc))
    return SemanticSearchService(config, index, model_dir, manifest)


def _semantic_package_path(model_dir: Path, value: object, default: str) -> Path:
    package_root = Path(model_dir).resolve()
    path = (package_root / str(value or default)).resolve()
    try:
        path.relative_to(package_root)
    except ValueError as exc:
        raise RuntimeError("semantic manifest path escapes the model package") from exc
    return path


def _semantic_model_identity(
    model_dir: Path,
    manifest: dict[str, Any],
) -> SemanticModelIdentity:
    dimensions = int(manifest.get("dimensions") or 0)
    if not 0 < dimensions <= 8192:
        raise RuntimeError("semantic manifest dimensions must be between 1 and 8192")
    preprocessing = json.dumps(
        {
            "image": dict(manifest.get("image") or {}),
            "text": dict(manifest.get("text") or {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return SemanticModelIdentity(
        str(manifest.get("implementation") or "openvino_manifest"),
        fingerprint_model_package(model_dir),
        hashlib.sha256(preprocessing).hexdigest()[:24],
        dimensions,
    )


def _semantic_tokenizer(
    model_dir: Path,
    text_spec: Mapping[str, Any],
) -> SemanticTokenizer:
    tokenizer_kind = str(text_spec.get("tokenizer_kind") or "openclip_bpe")
    if tokenizer_kind == "openclip_bpe":
        path = _semantic_package_path(
            model_dir,
            text_spec.get("tokenizer_path"),
            "tokenizer/bpe_simple_vocab_16e6.txt.gz",
        )
        if not path.is_file():
            raise RuntimeError("semantic tokenizer file is missing")
        return OpenClipTokenizerAdapter(
            path,
            max_length=int(text_spec.get("max_length") or 77),
        )
    raise RuntimeError(f"unsupported semantic tokenizer: {tokenizer_kind}")


def _semantic_text_inputs(
    text_spec: Mapping[str, Any],
    tokenized: Mapping[str, np.ndarray],
    compiled_model: Any,
) -> dict[str, np.ndarray]:
    configured = text_spec.get("inputs")
    if isinstance(configured, dict) and configured:
        input_names = {str(key): str(value) for key, value in configured.items()}
    else:
        input_names = {
            "input_ids": str(
                text_spec.get("input") or compiled_model.input(0).get_any_name()
            )
        }
    missing = sorted(set(input_names) - set(tokenized))
    if missing:
        raise RuntimeError(
            "semantic tokenizer did not produce required inputs: " + ", ".join(missing)
        )
    return {
        model_input: np.asarray(tokenized[logical_name], dtype=np.int64)
        for logical_name, model_input in input_names.items()
    }


def _semantic_named_inputs(
    spec: Mapping[str, Any],
    prepared: Mapping[str, np.ndarray],
    compiled_model: Any,
) -> dict[str, np.ndarray]:
    configured = spec.get("inputs")
    if not isinstance(configured, dict) or not configured:
        raise RuntimeError("semantic manifest is missing its input mapping")
    names = {str(key): str(value) for key, value in configured.items()}
    missing = sorted(set(names) - set(prepared))
    if missing:
        raise RuntimeError(
            "semantic preprocessor did not produce required inputs: " + ", ".join(missing)
        )
    return {model_name: np.asarray(prepared[key]) for key, model_name in names.items()}


class OpenVinoManifestEncoder:
    """OpenVINO dual-encoder loaded entirely from a local model package."""

    def __init__(
        self,
        model_dir: Path,
        manifest: dict[str, Any],
        device: str,
        identity: SemanticModelIdentity | None = None,
    ) -> None:
        from openvino import Core
        self.model_dir = Path(model_dir)
        image_spec = dict(manifest.get("image") or {})
        text_spec = dict(manifest.get("text") or {})
        image_model = _semantic_package_path(
            self.model_dir, manifest.get("image_model"), "image_encoder.xml"
        )
        text_model = _semantic_package_path(
            self.model_dir, manifest.get("text_model"), "text_encoder.xml"
        )
        if not image_model.is_file() or not text_model.is_file():
            raise RuntimeError("semantic image or text OpenVINO model is missing")
        core = Core()
        self._image_model = core.compile_model(str(image_model), device)
        self._text_model = core.compile_model(str(text_model), device)
        self._tokenizer = _semantic_tokenizer(self.model_dir, text_spec)
        self._image_spec = image_spec
        self._text_spec = text_spec
        self._identity = identity or _semantic_model_identity(self.model_dir, manifest)

    @property
    def identity(self) -> SemanticModelIdentity:
        return self._identity

    @staticmethod
    def _output(compiled: Any, result: Any, name: str) -> np.ndarray:
        if name:
            return np.asarray(result[compiled.output(name)])
        return np.asarray(result[compiled.output(0)])

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        return self.encode_prepared_images(self.prepare_images(images, self._image_spec))

    @staticmethod
    def prepare_images(
        images: Sequence[np.ndarray],
        spec: dict[str, Any],
    ) -> np.ndarray | dict[str, np.ndarray]:
        size = int(spec.get("size") or 256)
        mean = np.asarray(spec.get("mean") or [0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.asarray(spec.get("std") or [0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        prepared = []
        for image in images:
            height, width = image.shape[:2]
            interpolation = (
                cv2.INTER_CUBIC
                if str(spec.get("interpolation") or "bicubic").lower() == "bicubic"
                else cv2.INTER_AREA
            )
            resize_mode = str(spec.get("resize_mode") or "shortest_center_crop")
            if resize_mode == "fixed":
                resized = cv2.resize(image, (size, size), interpolation=interpolation)
            elif resize_mode == "shortest_center_crop":
                scale = size / max(1, min(height, width))
                resized = cv2.resize(
                    image,
                    (max(size, round(width * scale)), max(size, round(height * scale))),
                    interpolation=interpolation,
                )
                top = max(0, (resized.shape[0] - size) // 2)
                left = max(0, (resized.shape[1] - size) // 2)
                resized = resized[top:top + size, left:left + size]
            else:
                raise RuntimeError(f"unsupported semantic resize mode: {resize_mode}")
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            prepared.append(np.transpose((rgb - mean) / std, (2, 0, 1)))
        return np.stack(prepared).astype(np.float32)

    def encode_prepared_images(
        self, batch: np.ndarray | Mapping[str, np.ndarray]
    ) -> np.ndarray:
        spec = self._image_spec
        batch_size = int(spec.get("batch_size") or 0)
        row_count = (
            len(next(iter(batch.values())))
            if isinstance(batch, Mapping)
            else len(batch)
        )
        if batch_size == 1 and row_count > 1:
            return np.concatenate([
                self.encode_prepared_images(
                    {name: value[index:index + 1] for name, value in batch.items()}
                    if isinstance(batch, Mapping)
                    else batch[index:index + 1]
                )
                for index in range(row_count)
            ])
        if isinstance(batch, Mapping):
            inputs = _semantic_named_inputs(spec, batch, self._image_model)
        else:
            input_name = str(spec.get("input") or self._image_model.input(0).get_any_name())
            inputs = {input_name: batch}
        result = self._image_model(inputs)
        return normalized_matrix(self._output(self._image_model, result, str(spec.get("output") or "")))

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_tokens(self._tokenizer(list(texts)))

    def encode_tokens(self, tokens: Mapping[str, np.ndarray]) -> np.ndarray:
        inputs = _semantic_text_inputs(self._text_spec, tokens, self._text_model)
        result = self._text_model(inputs)
        return normalized_matrix(self._output(self._text_model, result, str(self._text_spec.get("output") or "")))

    def close(self) -> None:
        self._image_model = None
        self._text_model = None


def _semantic_encoder_worker_main(
    connection: Any,
    model_dir: str,
    manifest: dict[str, Any],
    device: str,
    identity: SemanticModelIdentity | None = None,
) -> None:
    from .inference import _disable_worker_core_dumps, _set_worker_process_name

    _set_worker_process_name("semantic")
    _disable_worker_core_dumps()
    encoder: OpenVinoManifestEncoder | None = None
    try:
        encoder = OpenVinoManifestEncoder(
            Path(model_dir), manifest, device, identity=identity
        )
        connection.send({"type": "ready", "pid": multiprocessing.current_process().pid})
        while True:
            request = connection.recv()
            request_id = int(request.get("id") or 0)
            operation = str(request.get("op") or "")
            if operation == "shutdown":
                connection.send({"id": request_id, "type": "stopped"})
                return
            try:
                if operation == "images":
                    raw_batch = request["batch"]
                    if isinstance(raw_batch, dict):
                        batch = {
                            str(name): np.asarray(value)
                            for name, value in raw_batch.items()
                        }
                    else:
                        batch = np.asarray(raw_batch, dtype=np.float32)
                    result = encoder.encode_prepared_images(batch)
                elif operation == "text":
                    result = encoder.encode_tokens({
                        str(name): np.asarray(value, dtype=np.int64)
                        for name, value in dict(request["inputs"]).items()
                    })
                else:
                    raise ValueError(f"unknown semantic inference operation: {operation}")
                connection.send({"id": request_id, "type": "result", "value": result})
            except Exception as exc:
                connection.send({"id": request_id, "type": "error", "error": str(exc)})
    except (EOFError, BrokenPipeError):
        return
    except BaseException as exc:
        try:
            connection.send({"type": "fatal", "error": str(exc)})
        except (BrokenPipeError, OSError):
            pass
    finally:
        if encoder is not None:
            encoder.close()
        connection.close()


class IsolatedOpenVinoManifestEncoder:
    """OpenVINO encoder proxy backed by a named, failure-isolated process."""

    def __init__(self, model_dir: Path, manifest: dict[str, Any], device: str) -> None:
        self.model_dir = Path(model_dir)
        self.manifest = dict(manifest)
        self.device = str(device)
        self._image_spec = dict(manifest.get("image") or {})
        self._text_spec = dict(manifest.get("text") or {})
        self._tokenizer = _semantic_tokenizer(self.model_dir, self._text_spec)
        self._identity = _semantic_model_identity(self.model_dir, manifest)
        self._context = multiprocessing.get_context("spawn")
        self._connection: Any = None
        self._process: Any = None
        self._lock = threading.RLock()
        self._request_id = 0
        self._closed = False
        self._start_locked()

    @property
    def identity(self) -> SemanticModelIdentity:
        return self._identity

    @property
    def worker_pid(self) -> int | None:
        process = self._process
        return int(process.pid) if process is not None and process.is_alive() else None

    def _start_locked(self) -> None:
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_semantic_encoder_worker_main,
            args=(child, str(self.model_dir), self.manifest, self.device, self._identity),
            name="survng-semantic-inference",
            daemon=False,
        )
        try:
            process.start()
            child.close()
            if not parent.poll(SEMANTIC_WORKER_START_TIMEOUT_SECONDS):
                raise RuntimeError("semantic inference worker startup timed out")
            try:
                message = parent.recv()
            except EOFError as exc:
                process.join(timeout=0.25)
                exitcode = process.exitcode
                if exitcode is None:
                    detail = ""
                elif exitcode < 0:
                    detail = f" (signal {-exitcode})"
                else:
                    detail = f" (exit code {exitcode})"
                raise RuntimeError(
                    f"semantic inference worker exited during startup{detail}"
                ) from exc
            if message.get("type") != "ready":
                raise RuntimeError(
                    str(message.get("error") or "semantic inference worker failed to start")
                )
        except BaseException:
            parent.close()
            try:
                child.close()
            except OSError:
                pass
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=3.0)
            raise
        self._connection = parent
        self._process = process
        LOGGER.info(
            "Semantic inference worker ready pid=%s device=%s",
            process.pid,
            self.device,
        )

    def _stop_locked(self) -> None:
        process = self._process
        connection = self._connection
        stubborn = False
        if process is not None and process.is_alive() and connection is not None:
            try:
                self._request_id += 1
                connection.send({"id": self._request_id, "op": "shutdown"})
                if connection.poll(5.0):
                    connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        if process is not None:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            stubborn = process.is_alive()
        if connection is not None:
            connection.close()
        self._process = None
        self._connection = None
        if stubborn:
            raise RuntimeError("semantic inference worker did not stop")

    def _request(self, operation: str, field: str, value: Any) -> np.ndarray:
        with self._lock:
            if self._closed:
                raise RuntimeError("semantic inference worker is closed")
            for attempt in range(2):
                try:
                    if self._closed:
                        raise RuntimeError("semantic inference worker is closed")
                    if self._process is None or not self._process.is_alive():
                        self._stop_locked()
                        self._start_locked()
                    self._request_id += 1
                    request_id = self._request_id
                    self._connection.send({
                        "id": request_id,
                        "op": operation,
                        field: value,
                    })
                    if not self._connection.poll(SEMANTIC_WORKER_REQUEST_TIMEOUT_SECONDS):
                        raise RuntimeError("semantic inference worker request timed out")
                    message = self._connection.recv()
                    if int(message.get("id") or 0) != request_id:
                        raise RuntimeError("semantic inference worker response was out of sequence")
                    if message.get("type") != "result":
                        raise SemanticInferenceError(
                            str(message.get("error") or "semantic inference worker failed")
                        )
                    return normalized_matrix(message["value"])
                except SemanticInferenceError:
                    raise
                except (BrokenPipeError, EOFError, OSError, RuntimeError):
                    self._stop_locked()
                    if attempt:
                        raise
            raise RuntimeError("semantic inference worker failed")

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        batch = OpenVinoManifestEncoder.prepare_images(images, self._image_spec)
        return self._request("images", "batch", batch)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return self._request("text", "inputs", self._tokenizer(list(texts)))

    def close(self) -> None:
        self.abort()
        with self._lock:
            self._stop_locked()

    def abort(self) -> None:
        """Interrupt in-flight inference so application shutdown stays bounded."""
        self._closed = True
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()


class SemanticSearchService(DisabledSemanticSearch):
    """Low-priority asynchronous incident indexer and text search service."""

    def __init__(self, config: SemanticSearchConfig, index: SemanticIndex, model_dir: Path, manifest: dict[str, Any]) -> None:
        super().__init__(config, index)
        self.model_dir = model_dir
        self.manifest = manifest
        self.encoder: SemanticEncoder | None = None
        self._queue: queue.PriorityQueue[tuple[int, int, dict[str, Any] | None]] = (
            queue.PriorityQueue(config.worker_queue_size)
        )
        self._queue_sequence = itertools.count()
        self._live_queue_reserve = max(1, min(16, config.worker_queue_size // 4))
        self._thread: threading.Thread | None = None
        self._backfill_thread: threading.Thread | None = None
        self._bootstrap_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._storage_dir = Path()
        self._media_storage: MediaStorageRegistry | None = None
        self._error = ""
        self._indexed = 0
        self._skipped_missing = 0
        self._state = "stopped"
        self._initialization_attempts = 0
        self._next_retry_at = 0.0
        self._active_device = str(config.device)
        self._fallback_active = False
        self._encoder_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()

    def start(self, event_store: Any, storage_dir: Path, media_storage: MediaStorageRegistry | None = None) -> None:
        with self._lifecycle_lock:
            if self._state in {"initializing", "recovering", "ready", "stopping"}:
                return
            # A stop during bootstrap has no worker available to consume the
            # sentinel. Never let that stale sentinel terminate the next
            # generation as soon as it starts.
            self._drain_queue()
            self._storage_dir = Path(storage_dir)
            self._media_storage = media_storage
            self._stop.clear()
            self._state = "initializing"
            self._bootstrap_thread = threading.Thread(
                target=self._initialize,
                args=(event_store,),
                name="survng-semantic-loader",
                daemon=True,
            )
            self._bootstrap_thread.start()

    def _initialize(self, event_store: Any) -> None:
        retry_seconds = SEMANTIC_WORKER_RETRY_INITIAL_SECONDS
        configured_device = str(self.config.device)
        target_device = configured_device
        configured_device_failures = 0
        while not self._stop.is_set():
            try:
                encoder = IsolatedOpenVinoManifestEncoder(
                    self.model_dir,
                    self.manifest,
                    target_device,
                )
            except Exception as exc:
                reason = str(exc).strip() or "semantic inference worker exited during startup"
                if target_device == configured_device:
                    configured_device_failures += 1
                use_cpu_fallback = bool(
                    target_device == configured_device
                    and configured_device.strip().upper() != "CPU"
                    and configured_device_failures
                    >= SEMANTIC_WORKER_CONFIGURED_DEVICE_ATTEMPTS
                )
                wait_seconds = (
                    SEMANTIC_WORKER_FALLBACK_DELAY_SECONDS
                    if use_cpu_fallback
                    else retry_seconds
                )
                if use_cpu_fallback:
                    target_device = "CPU"
                with self._lifecycle_lock:
                    if self._stop.is_set() or self._state not in {"initializing", "recovering"}:
                        return
                    self._initialization_attempts += 1
                    self._error = reason
                    self._state = "recovering"
                    self._active_device = target_device
                    self._fallback_active = use_cpu_fallback or target_device != configured_device
                    self._next_retry_at = time.monotonic() + wait_seconds
                    attempt = self._initialization_attempts
                if use_cpu_fallback:
                    LOGGER.warning(
                        "semantic search %s worker startup failed after %d attempts: %s; "
                        "falling back to CPU in %.0fs",
                        configured_device,
                        configured_device_failures,
                        reason,
                        wait_seconds,
                    )
                else:
                    LOGGER.warning(
                        "semantic search %s worker startup failed (attempt %d): %s; "
                        "retrying in %.0fs",
                        target_device,
                        attempt,
                        reason,
                        wait_seconds,
                    )
                if self._stop.wait(wait_seconds):
                    return
                if not use_cpu_fallback:
                    retry_seconds = min(
                        SEMANTIC_WORKER_RETRY_MAX_SECONDS,
                        retry_seconds * 2.0,
                    )
                continue
            break
        else:
            return

        if self._stop.is_set():
            encoder.close()
            return

        with self._lifecycle_lock:
            if self._stop.is_set() or self._state not in {"initializing", "recovering"}:
                should_close = True
            else:
                should_close = False
                self.encoder = encoder
                self._error = ""
                self._next_retry_at = 0.0
                self._active_device = target_device
                self._fallback_active = target_device != configured_device
                self._state = "ready"
                self._thread = threading.Thread(
                    target=self._run, name="survng-semantic", daemon=True
                )
                self._backfill_thread = threading.Thread(
                    target=self._run_backfill,
                    args=(event_store,),
                    name="survng-semantic-backfill",
                    daemon=True,
                )
                # Start while lifecycle publication is locked so close() can never
                # observe an assigned but not-yet-started thread and attempt to join it.
                self._thread.start()
                self._backfill_thread.start()
        if should_close:
            encoder.close()

    def _history_queue_has_capacity(self) -> bool:
        return (
            self._queue.qsize()
            < self.config.worker_queue_size - self._live_queue_reserve
        )

    def _run_backfill(self, event_store: Any) -> None:
        """Retry transient index/event-store failures without losing backfill forever."""
        while not self._stop.is_set():
            try:
                self._backfill(event_store)
                return
            except Exception as exc:
                self._error = str(exc)
                LOGGER.warning("semantic historical indexing interrupted: %s", exc)
                if self._stop.wait(SEMANTIC_BACKFILL_RETRY_SECONDS):
                    return

    def _backfill(self, event_store: Any) -> None:
        before_created_at: str | None = None
        before_id: int | None = None
        encoder = self.encoder
        if encoder is None:
            return
        if not self.config.index_full_frame:
            self.index.delete_generation_source(encoder.identity, "full_frame")
        if not self.config.index_object_crops:
            self.index.delete_generation_source(encoder.identity, "object_crop")
        indexed_event_ids = self.index.indexed_event_ids()
        while not self._stop.is_set():
            rows = event_store.recent_compact(
                self.config.backfill_batch_size,
                before_created_at,
                before_id,
            )
            if not rows:
                return
            for event in reversed(rows):
                if self._stop.is_set():
                    return
                event_id = int(event.get("id") or 0)
                objects = semantic_event_objects(event)
                if not objects:
                    if event_id > 0 and event_id in indexed_event_ids:
                        self.index.delete_event(event_id)
                        indexed_event_ids.discard(event_id)
                    continue
                if self.encoder:
                    full_ready = not self.config.index_full_frame or self.index.event_source_indexed(
                        event_id, self.encoder.identity, "full_frame"
                    )
                    crop_candidates = semantic_object_crop_candidates(
                        objects, self.config.max_object_crops_per_event
                    )
                    desired_crop_keys = {
                        semantic_crop_source_key(index, item)
                        for index, item, _coordinates in crop_candidates
                    }
                    existing_crop_keys = self.index.event_source_keys(
                        event_id, self.encoder.identity, "object_crop"
                    )
                    crops_ready = (
                        not self.config.index_object_crops
                        or existing_crop_keys == desired_crop_keys
                    )
                    if full_ready and crops_ready:
                        continue
                while not self._stop.is_set():
                    if not self._history_queue_has_capacity():
                        self._stop.wait(0.1)
                        continue
                    try:
                        self._queue.put(
                            (1, next(self._queue_sequence), dict(event)),
                            timeout=0.5,
                        )
                        break
                    except queue.Full:
                        continue
            last = rows[-1]
            before_created_at = str(last.get("created_at") or "")
            before_id = int(last.get("id") or 0)
            if len(rows) < self.config.backfill_batch_size:
                return

    def queue_event(self, event: dict[str, Any]) -> bool:
        if (
            self.encoder is None
            or not event.get("snapshot_path")
            or not event.get("id")
            or not semantic_event_objects(event)
        ):
            return False
        try:
            self._queue.put_nowait((0, next(self._queue_sequence), dict(event)))
            return True
        except queue.Full:
            self._error = "index queue is full"
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                priority, _sequence, event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if event is None:
                break
            try:
                self.index_event(event)
                self._error = ""
                if priority > 0:
                    self._stop.wait(self.config.backfill_pause_seconds)
            except Exception as exc:
                self._error = str(exc)
                LOGGER.warning("semantic indexing failed for event %s: %s", event.get("id"), exc)

    def index_event(self, event: dict[str, Any]) -> int:
        """Synchronously index one event for tooling and the worker loop.

        New evidence is generation-isolated and idempotent. Encoder use is
        serialized by the service.
        """
        if self.encoder is None:
            return 0
        event_id = int(event.get("id") or 0)
        if event_id <= 0:
            return 0
        objects = semantic_event_objects(event)
        if not objects:
            self.index.delete_event(event_id)
            return 0
        identity = self.encoder.identity
        full_frame_needed = self.config.index_full_frame and not self.index.event_source_indexed(
            event_id, identity, "full_frame"
        )
        crop_candidates = semantic_object_crop_candidates(
            objects, self.config.max_object_crops_per_event
        )
        desired_crop_keys = {
            semantic_crop_source_key(index, item)
            for index, item, _coordinates in crop_candidates
        }
        existing_crop_keys = self.index.event_source_keys(
            event_id, identity, "object_crop"
        )
        object_crops_needed = (
            self.config.index_object_crops
            and bool(desired_crop_keys - existing_crop_keys)
        )
        if not full_frame_needed and not object_crops_needed:
            if self.config.index_object_crops and existing_crop_keys != desired_crop_keys:
                self.index.reconcile_event_source_keys(
                    event_id, identity, "object_crop", desired_crop_keys
                )
            return 0
        try:
            path = event_snapshot_path(self._storage_dir, event, self._media_storage)
        except FileNotFoundError:
            self._skipped_missing += 1
            return 0
        frame = cv2.imread(str(path))
        if frame is None:
            self._skipped_missing += 1
            return 0
        evidence: list[SemanticEvidence] = []
        images: list[np.ndarray] = []
        if full_frame_needed:
            evidence.append(SemanticEvidence(event_id, str(event.get("camera_id") or ""), str(event.get("created_at") or ""), "full_frame", "frame", str(event["snapshot_path"])))
            images.append(frame)
        if object_crops_needed:
            height, width = frame.shape[:2]
            for index, item, coordinates in crop_candidates:
                source_key = semantic_crop_source_key(index, item)
                if source_key in existing_crop_keys:
                    continue
                source_width = positive_dimension(item.get("detection_frame_width"), width)
                source_height = positive_dimension(item.get("detection_frame_height"), height)
                x1, y1, x2, y2 = (
                    int(round(coordinates[0] * width / source_width)),
                    int(round(coordinates[1] * height / source_height)),
                    int(round(coordinates[2] * width / source_width)),
                    int(round(coordinates[3] * height / source_height)),
                )
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                label = str(item.get("label") or "").strip().lower()
                evidence.append(SemanticEvidence(event_id, str(event.get("camera_id") or ""), str(event.get("created_at") or ""), "object_crop", source_key, str(event["snapshot_path"]), label, (x1, y1, x2, y2)))
                images.append(frame[y1:y2, x1:x2])
        if images:
            with self._encoder_lock:
                if self.encoder is None:
                    return 0
                embeddings = self.encoder.encode_images(images)
            written = self.index.upsert(evidence, embeddings, self.encoder.identity)
            self._indexed += written
            if self.config.index_object_crops and existing_crop_keys != desired_crop_keys:
                # Preserve prior searchable evidence until replacements have
                # encoded successfully, then remove only stale source keys.
                self.index.reconcile_event_source_keys(
                    event_id, identity, "object_crop", desired_crop_keys
                )
            return written
        return 0

    def _index_event(self, event: dict[str, Any]) -> int:
        """Backward-compatible internal alias for existing integrations."""
        return self.index_event(event)

    def _object_crop_from_event(
        self,
        event: dict[str, Any],
        object_index: int,
    ) -> np.ndarray:
        objects = semantic_event_objects(event)
        try:
            item = objects[int(object_index)]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("semantic object index is out of range") from exc
        if int(object_index) < 0:
            raise ValueError("semantic object index is out of range")
        coordinates = semantic_object_bbox(item)
        if coordinates is None:
            raise ValueError("selected semantic object has no valid crop")
        try:
            path = event_snapshot_path(
                self._storage_dir,
                event,
                self._media_storage,
            )
        except FileNotFoundError as exc:
            raise ValueError("event snapshot is unavailable") from exc
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError("event snapshot is unavailable")
        height, width = frame.shape[:2]
        source_width = positive_dimension(item.get("detection_frame_width"), width)
        source_height = positive_dimension(item.get("detection_frame_height"), height)
        x1, y1, x2, y2 = (
            int(round(coordinates[0] * width / source_width)),
            int(round(coordinates[1] * height / source_height)),
            int(round(coordinates[2] * width / source_width)),
            int(round(coordinates[3] * height / source_height)),
        )
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("selected semantic object crop is empty")
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("selected semantic object crop is empty")
        return crop

    def search_image(
        self,
        image: np.ndarray,
        **filters: Any,
    ) -> list[SemanticSearchHit]:
        if self.encoder is None:
            raise RuntimeError(self._error or "semantic search is unavailable")
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("semantic search image cannot be empty")
        plan = SemanticQueryPlan("visual", {"full": "visual"})
        with self._encoder_lock:
            if self.encoder is None:
                raise RuntimeError(self._error or "semantic search is unavailable")
            embedding = self.encoder.encode_images([image])
            identity = self.encoder.identity
        return self.index.search(
            embedding,
            identity,
            query_plan=plan,
            **filters,
        )

    def search_event_object(
        self,
        event: dict[str, Any],
        object_index: int,
        **filters: Any,
    ) -> list[SemanticSearchHit]:
        return self.search_image(
            self._object_crop_from_event(event, object_index),
            **filters,
        )

    def search_text(self, query: str, **filters: Any) -> list[SemanticSearchHit]:
        if self.encoder is None:
            raise RuntimeError(self._error or "semantic search is unavailable")
        text = str(query).strip()
        if not text:
            raise ValueError("semantic search query cannot be empty")
        plan = semantic_query_plan(text)
        with self._encoder_lock:
            if self.encoder is None:
                raise RuntimeError(self._error or "semantic search is unavailable")
            embedding = self.encoder.encode_text(list(plan.prompts.values()))
            identity = self.encoder.identity
        return self.index.search(
            embedding,
            identity,
            query_plan=plan,
            **filters,
        )

    def status(self) -> dict[str, Any]:
        identity = self.encoder.identity if self.encoder else None
        retry_in_seconds = max(0.0, self._next_retry_at - time.monotonic())
        return {
            "enabled": True, "state": self._state,
            "implementation": self.config.implementation,
            "device": self._active_device,
            "configured_device": self.config.device,
            "fallback_active": self._fallback_active,
            "generation": identity.generation if identity else "", "error": self._error,
            "initialization_attempts": self._initialization_attempts,
            "retry_in_seconds": round(retry_in_seconds, 1),
            "queue_depth": self._queue.qsize(), "indexed_since_start": self._indexed,
            "skipped_missing_since_start": self._skipped_missing,
            "worker_pid": getattr(self.encoder, "worker_pid", None),
            **self.index.coverage(identity),
        }

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._state == "stopped":
                return
            self._stop.set()
            self._state = "stopping"
            bootstrap_thread = self._bootstrap_thread
            backfill_thread = self._backfill_thread
            worker_thread = self._thread
            encoder = self.encoder
        try:
            self._queue.put_nowait((-1, next(self._queue_sequence), None))
        except queue.Full:
            pass
        abort = getattr(encoder, "abort", None)
        if callable(abort):
            abort()
        for thread in (bootstrap_thread, backfill_thread, worker_thread):
            if thread and thread is not threading.current_thread():
                thread.join(timeout=5.0)
        alive = [
            thread.name
            for thread in (bootstrap_thread, backfill_thread, worker_thread)
            if thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ]
        if alive:
            raise RuntimeError(
                "semantic search workers did not stop: " + ", ".join(alive)
            )
        with self._encoder_lock:
            if self.encoder:
                self.encoder.close()
            self.encoder = None
        with self._lifecycle_lock:
            self._state = "stopped"
            self._next_retry_at = 0.0
            self._bootstrap_thread = None
            self._backfill_thread = None
            self._thread = None

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._queue.task_done()

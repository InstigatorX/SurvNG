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
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import cv2

from .config import SemanticSearchConfig
from .incident_utils import event_snapshot_path
from .main_database import connect_main_database
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
SEMANTIC_MEDIA_QUARANTINE_ATTEMPTS = 3
SEMANTIC_MEDIA_QUARANTINE_RETRY_SECONDS = 15 * 60.0


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
class SemanticBackfillState:
    """Durable cursor for one image generation's historical walk."""

    before_created_at: str | None = None
    before_id: int | None = None
    head_created_at: str | None = None
    head_id: int | None = None
    complete: bool = False


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
    evidence_revision: int = 0
    observation_id: str = ""


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
    observation_id: str = ""


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


@dataclass
class _SemanticVectorCache:
    """Exact vectors for one model generation plus a bounded live delta."""

    generation: str
    dimensions: int
    row_ids: np.ndarray
    vectors: np.ndarray
    positions: dict[int, int]
    signature: tuple[int, int, str]
    expected_signature: tuple[int, int, str]
    built_monotonic: float
    overrides: dict[int, np.ndarray] = field(default_factory=dict)
    dirty_event_ids: set[int] = field(default_factory=set)
    force_rebuild: bool = False

    @property
    def bytes(self) -> int:
        return int(self.row_ids.nbytes + self.vectors.nbytes) + sum(
            int(vector.nbytes) for vector in self.overrides.values()
        )


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


def semantic_event_searchable(event: dict[str, Any]) -> bool:
    """Search observed evidence independently of recording and alert decisions."""
    return bool(event.get("snapshot_path")) and bool(semantic_event_objects(event))


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

    VECTOR_CACHE_MAX_BYTES = 512 * 1024 * 1024
    VECTOR_CACHE_MAX_DELTA_ROWS = 4_096
    VECTOR_CACHE_MAX_AGE_SECONDS = 6 * 60 * 60.0
    VECTOR_CACHE_RETRY_SECONDS = 5 * 60.0
    VECTOR_LOAD_BATCH_ROWS = 2_000

    @staticmethod
    def _event_matches(connection: sqlite3.Connection, event: dict[str, Any]) -> bool:
        columns = {row[1] for row in connection.execute("pragma table_info(events)")}
        guarded = [
            name for name in
            ("evidence_revision", "scene_media_revision", "snapshot_path")
            if name in columns
        ]
        row = connection.execute(
            f"select {','.join(['id', *guarded])} from events where id = ?", (int(event["id"]),),
        ).fetchone()
        return row is not None and all(
            row[name] == event.get(
                name, 0 if name in {"evidence_revision", "scene_media_revision"} else "",
            )
            for name in guarded
        )

    def __init__(
        self,
        database_path: Path,
        database_write_lock: threading.RLock | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self._lock = threading.Lock()
        self._database_write_lock = database_write_lock or threading.RLock()
        self._cache_lock = threading.RLock()
        self._vector_cache: _SemanticVectorCache | None = None
        self._vector_cache_disabled_until = 0.0
        self._vector_cache_error = ""
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = connect_main_database(
            self.database_path, timeout=10.0, write_lock=self._database_write_lock
        )
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
                    evidence_revision integer not null default 0,
                    foreign key(event_id) references events(id) on delete cascade,
                    unique(
                        event_id, source_kind, source_key,
                        model_fingerprint, preprocessing_fingerprint
                    )
                )
                """
            )
            columns = {row[1] for row in connection.execute("pragma table_info(semantic_embeddings)")}
            if "evidence_revision" not in columns:
                connection.execute(
                    "alter table semantic_embeddings add column evidence_revision integer not null default 0"
                )
            if "observation_id" not in columns:
                connection.execute("alter table semantic_embeddings add column observation_id text not null default ''")
            connection.execute(
                """
                create index if not exists idx_semantic_generation_search
                on semantic_embeddings(
                    model_fingerprint, preprocessing_fingerprint,
                    captured_at desc, id desc
                )
                """
            )
            connection.execute("drop index if exists idx_semantic_generation_time")
            connection.execute(
                """
                create index if not exists idx_semantic_event
                on semantic_embeddings(event_id)
                """
            )
            # Point lookups by observation must not scan a model generation.
            connection.execute(
                """
                create index if not exists idx_semantic_observation
                on semantic_embeddings(
                    observation_id, image_path, model_fingerprint, preprocessing_fingerprint
                )
                """
            )
            # Search suppresses an event-cover crop when independently retained
            # scene evidence represents the same box.  This partial index keeps
            # that correlated existence check bounded to one event instead of
            # walking a model generation for every candidate row.
            connection.execute(
                """
                create index if not exists idx_semantic_retained_cover
                on semantic_embeddings(
                    event_id, image_path, bbox_json,
                    model_fingerprint, preprocessing_fingerprint
                ) where observation_id <> ''
                """
            )
            connection.execute("""
                create table if not exists semantic_projection_receipts (
                    event_id integer not null references events(id) on delete cascade,
                    model_fingerprint text not null, preprocessing_fingerprint text not null,
                    evidence_revision integer not null, image_path text not null,
                    plan_key text not null, outcome_json text not null,
                    primary key(event_id,model_fingerprint,preprocessing_fingerprint)
                )
            """)
            connection.execute("""
                create table if not exists semantic_generation_sources (
                    image_contract text primary key,
                    implementation text not null,
                    model_fingerprint text not null,
                    preprocessing_fingerprint text not null,
                    dimensions integer not null
                )
            """)
            connection.execute("""
                create table if not exists semantic_backfill_state (
                    model_fingerprint text not null,
                    preprocessing_fingerprint text not null,
                    before_created_at text,
                    before_id integer,
                    head_created_at text,
                    head_id integer,
                    complete integer not null default 0,
                    primary key (model_fingerprint, preprocessing_fingerprint)
                )
            """)
            connection.execute("""
                create table if not exists semantic_media_failures (
                    event_id integer not null references events(id) on delete cascade,
                    model_fingerprint text not null,
                    preprocessing_fingerprint text not null,
                    failure_count integer not null default 0,
                    last_error text not null default '',
                    last_attempt_at real not null,
                    next_attempt_at real not null,
                    primary key(event_id,model_fingerprint,preprocessing_fingerprint)
                )
            """)

    def upsert(
        self,
        evidence: Iterable[SemanticEvidence],
        embeddings: object,
        identity: SemanticModelIdentity,
        *,
        expected_event: dict[str, Any] | None = None,
        reconcile_sources: dict[str, set[str]] | None = None,
        projection_receipt: dict[str, Any] | None = None,
        expected_observation: dict[str, Any] | None = None,
    ) -> int:
        records = list(evidence)
        if projection_receipt is not None and expected_event is None:
            raise ValueError("projection completion requires an authoritative event guard")
        if not records and not reconcile_sources and projection_receipt is None:
            return 0
        vectors = normalized_matrix(embeddings) if records else np.empty((0, identity.dimensions))
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
                int(record.evidence_revision),
                str(record.observation_id),
            ))
        mutation_event_ids = {int(record.event_id) for record in records}
        if expected_event is not None:
            mutation_event_ids.add(int(expected_event["id"]))
        with self._lock, self._connect() as connection:
            connection.execute("begin immediate")
            if expected_event is not None:
                # Guard against an event commit that has not yet delivered its
                # outbox notification. In-memory tokens alone cannot see it.
                if not self._event_matches(connection, expected_event):
                    return 0
            if expected_observation is not None:
                row = connection.execute("select snapshot_path,payload_json from scene_observations where id=?",
                                         (expected_observation["id"],)).fetchone()
                if row is None or any(row[key] != expected_observation[key] for key in ("snapshot_path", "payload_json")):
                    return 0
            connection.executemany(
                """
                insert into semantic_embeddings (
                    event_id, camera_id, captured_at, source_kind, source_key,
                    image_path, object_label, bbox_json, implementation,
                    model_fingerprint, preprocessing_fingerprint, embedding_size,
                    embedding_blob, created_at, evidence_revision, observation_id
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    created_at=excluded.created_at,
                    evidence_revision=excluded.evidence_revision,
                    observation_id=excluded.observation_id
                """,
                prepared,
            )
            if expected_event is not None:
                for source_kind, desired_keys in (reconcile_sources or {}).items():
                    parameters: list[Any] = [
                        int(expected_event["id"]), identity.model_fingerprint,
                        identity.preprocessing_fingerprint, source_kind,
                    ]
                    clause = ""
                    if desired_keys:
                        clause = f" and source_key not in ({','.join('?' for _ in desired_keys)})"
                        parameters.extend(sorted(desired_keys))
                    connection.execute(
                        "delete from semantic_embeddings where event_id = ? "
                        "and model_fingerprint = ? and preprocessing_fingerprint = ? "
                        f"and source_kind = ? and observation_id=''{clause}", parameters,
                    )
            if projection_receipt is not None:
                connection.execute(
                    "insert or replace into semantic_projection_receipts values(?,?,?,?,?,?,?)",
                    (int(expected_event["id"]), identity.model_fingerprint, identity.preprocessing_fingerprint,
                     int(expected_event.get("evidence_revision") or 0), str(expected_event.get("snapshot_path") or ""),
                     projection_receipt["plan_key"], json.dumps(projection_receipt)),
                )
        self._record_cache_mutation(identity, mutation_event_ids)
        return len(prepared)

    def projection_receipt(self, event: dict[str, Any], identity: SemanticModelIdentity, plan_key: str):
        with self._connect() as connection:
            row = connection.execute(
                "select outcome_json from semantic_projection_receipts where event_id=? "
                "and model_fingerprint=? and preprocessing_fingerprint=? and evidence_revision=? "
                "and image_path=? and plan_key=?",
                (int(event["id"]), identity.model_fingerprint, identity.preprocessing_fingerprint,
                 int(event.get("evidence_revision") or 0), str(event.get("snapshot_path") or ""), plan_key),
            ).fetchone()
        return json.loads(row[0]) if row else None

    @staticmethod
    def _generation_signature(
        connection: sqlite3.Connection,
        identity: SemanticModelIdentity,
    ) -> tuple[int, int, str]:
        row = connection.execute(
            """
            select count(*) as row_count, coalesce(max(id), 0) as maximum_id,
                coalesce(max(created_at), '') as newest_write
            from semantic_embeddings
            where model_fingerprint = ? and preprocessing_fingerprint = ?
            """,
            (identity.model_fingerprint, identity.preprocessing_fingerprint),
        ).fetchone()
        return (
            int(row["row_count"] if row else 0),
            int(row["maximum_id"] if row else 0),
            str(row["newest_write"] if row else ""),
        )

    def _record_cache_mutation(
        self,
        identity: SemanticModelIdentity | None,
        event_ids: Iterable[int] = (),
        *,
        force_rebuild: bool = False,
    ) -> None:
        """Publish an in-process semantic write to an already-loaded cache."""
        with self._cache_lock:
            cache = self._vector_cache
            if cache is None or (identity is not None and cache.generation != identity.generation):
                return
        if identity is None:
            with self._cache_lock:
                if self._vector_cache is cache:
                    cache.force_rebuild = True
            return
        with self._connect() as connection:
            signature = self._generation_signature(connection, identity)
        with self._cache_lock:
            if self._vector_cache is not cache:
                return
            cache.expected_signature = signature
            cache.dirty_event_ids.update(
                int(event_id) for event_id in event_ids if int(event_id) > 0
            )
            cache.force_rebuild = cache.force_rebuild or force_rebuild

    def _build_vector_cache_locked(
        self,
        connection: sqlite3.Connection,
        identity: SemanticModelIdentity,
        signature: tuple[int, int, str],
    ) -> _SemanticVectorCache:
        row_count = signature[0]
        vector_bytes = row_count * identity.dimensions * np.dtype(np.float32).itemsize
        id_bytes = row_count * np.dtype(np.int64).itemsize
        if vector_bytes + id_bytes > self.VECTOR_CACHE_MAX_BYTES:
            raise MemoryError(
                "semantic generation exceeds the exact vector cache memory budget"
            )
        row_ids = np.empty(row_count, dtype=np.int64)
        vectors = np.empty((row_count, identity.dimensions), dtype=np.float32)
        cursor = connection.execute(
            """
            select id, embedding_size, embedding_blob
            from semantic_embeddings
            where model_fingerprint = ? and preprocessing_fingerprint = ?
            order by id
            """,
            (identity.model_fingerprint, identity.preprocessing_fingerprint),
        )
        used = 0
        while True:
            batch = cursor.fetchmany(self.VECTOR_LOAD_BATCH_ROWS)
            if not batch:
                break
            for row in batch:
                size = int(row["embedding_size"] or 0)
                raw = row["embedding_blob"]
                if (
                    size != identity.dimensions
                    or not isinstance(raw, bytes)
                    or len(raw) != size * 2
                ):
                    continue
                vector = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
                if not np.all(np.isfinite(vector)):
                    continue
                row_ids[used] = int(row["id"])
                vectors[used] = vector
                used += 1
        if used != row_count:
            row_ids = np.ascontiguousarray(row_ids[:used])
            vectors = np.ascontiguousarray(vectors[:used])
        positions = {int(row_id): index for index, row_id in enumerate(row_ids)}
        cache = _SemanticVectorCache(
            generation=identity.generation,
            dimensions=identity.dimensions,
            row_ids=row_ids,
            vectors=vectors,
            positions=positions,
            signature=signature,
            expected_signature=signature,
            built_monotonic=time.monotonic(),
        )
        self._vector_cache = cache
        self._vector_cache_error = ""
        return cache

    def _refresh_cache_events_locked(
        self,
        connection: sqlite3.Connection,
        cache: _SemanticVectorCache,
        identity: SemanticModelIdentity,
    ) -> None:
        event_ids = sorted(cache.dirty_event_ids)
        cache.dirty_event_ids.clear()
        for offset in range(0, len(event_ids), 250):
            chunk = event_ids[offset:offset + 250]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                "select id,embedding_size,embedding_blob from semantic_embeddings "
                f"where event_id in ({placeholders}) and model_fingerprint=? "
                "and preprocessing_fingerprint=?",
                (*chunk, identity.model_fingerprint, identity.preprocessing_fingerprint),
            ).fetchall()
            for row in rows:
                size = int(row["embedding_size"] or 0)
                raw = row["embedding_blob"]
                if (
                    size != identity.dimensions
                    or not isinstance(raw, bytes)
                    or len(raw) != size * 2
                ):
                    continue
                vector = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
                if np.all(np.isfinite(vector)):
                    cache.overrides[int(row["id"])] = vector

    def _vector_cache_locked(
        self,
        connection: sqlite3.Connection,
        identity: SemanticModelIdentity,
    ) -> _SemanticVectorCache | None:
        now = time.monotonic()
        if now < self._vector_cache_disabled_until:
            return None
        signature = self._generation_signature(connection, identity)
        cache = self._vector_cache
        rebuild = bool(
            cache is None
            or cache.generation != identity.generation
            or cache.dimensions != identity.dimensions
            or cache.force_rebuild
            or signature != cache.expected_signature
            or now - cache.built_monotonic >= self.VECTOR_CACHE_MAX_AGE_SECONDS
        )
        try:
            if rebuild:
                cache = self._build_vector_cache_locked(connection, identity, signature)
            else:
                if cache.dirty_event_ids:
                    self._refresh_cache_events_locked(connection, cache, identity)
                cache.signature = signature
                cache.expected_signature = signature
                if len(cache.overrides) > self.VECTOR_CACHE_MAX_DELTA_ROWS:
                    cache = self._build_vector_cache_locked(connection, identity, signature)
        except (MemoryError, sqlite3.Error) as exc:
            self._vector_cache = None
            self._vector_cache_error = str(exc)
            self._vector_cache_disabled_until = now + self.VECTOR_CACHE_RETRY_SECONDS
            LOGGER.warning(
                "semantic exact vector cache unavailable; using chunked full scan: %s",
                exc,
            )
            return None
        return cache

    def warm_search_cache(self, identity: SemanticModelIdentity) -> None:
        """Load the complete active generation without delaying application startup."""
        with self._cache_lock, self._connect() as connection:
            connection.execute("begin")
            self._vector_cache_locked(connection, identity)

    def search_cache_status(
        self, identity: SemanticModelIdentity | None = None
    ) -> dict[str, Any]:
        with self._cache_lock:
            cache = self._vector_cache
            matching = bool(
                cache is not None
                and (identity is None or cache.generation == identity.generation)
            )
            return {
                "mode": "exact_full_generation",
                "state": (
                    "ready" if matching
                    else "fallback" if time.monotonic() < self._vector_cache_disabled_until
                    else "cold"
                ),
                "generation": cache.generation if matching and cache else "",
                "rows": len(cache.row_ids) if matching and cache else 0,
                "delta_rows": len(cache.overrides) if matching and cache else 0,
                "bytes": cache.bytes if matching and cache else 0,
                "error": self._vector_cache_error,
                "candidate_cap": None,
            }

    def search(
        self,
        query_embedding: object,
        identity: SemanticModelIdentity,
        *,
        query_plan: SemanticQueryPlan | None = None,
        camera_ids: Sequence[str] = (),
        object_labels: Sequence[str] = (),
        source_kinds: Sequence[str] = (),
        exclude_event_ids: Sequence[int] = (),
        unique_events: bool = False,
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
        if exclude_event_ids:
            normalized_exclusions = sorted({
                int(value) for value in exclude_event_ids if int(value) > 0
            })
            if normalized_exclusions:
                clauses.append(
                    f"event_id not in ({','.join('?' for _ in normalized_exclusions)})"
                )
                parameters.extend(normalized_exclusions)
        if start_at:
            clauses.append("captured_at >= ?")
            parameters.append(str(start_at))
        if end_at:
            clauses.append("captured_at <= ?")
            parameters.append(str(end_at))
        with self._cache_lock, self._connect() as connection:
            connection.execute("begin")
            columns = {row[1] for row in connection.execute("pragma table_info(events)")}
            has_scene_observations = connection.execute(
                "select 1 from sqlite_master where type='table' and name='scene_observations'"
            ).fetchone() is not None
            effective_label = "object_label"
            if has_scene_observations:
                effective_label = ("coalesce((select s.label_override from scene_observations o "
                                   "join scene_objects s on s.id=o.object_id where "
                                   "o.id=semantic_embeddings.observation_id),object_label)")
                clauses = [clause.replace("object_label in", f"{effective_label} in") for clause in clauses]
                # A retained observation supersedes the identical cover crop,
                # including its corrected label, without duplicating a hit.
                clauses.append("not(observation_id='' and source_kind='object_crop' and exists("
                               "select 1 from semantic_embeddings observed where observed.observation_id<>'' "
                               "and observed.event_id=semantic_embeddings.event_id "
                               "and observed.image_path=semantic_embeddings.image_path "
                               "and observed.bbox_json=semantic_embeddings.bbox_json "
                               "and observed.model_fingerprint=semantic_embeddings.model_fingerprint "
                               "and observed.preprocessing_fingerprint=semantic_embeddings.preprocessing_fingerprint))")
            if "evidence_revision" in columns and "snapshot_path" in columns:
                # Retain prior embeddings until replacement succeeds, but never
                # present their scores/crops as evidence for the current image.
                current = ("(observation_id='' and exists(select 1 from events e where e.id=semantic_embeddings.event_id "
                           "and e.evidence_revision=semantic_embeddings.evidence_revision "
                           "and e.snapshot_path=semantic_embeddings.image_path))")
                retained = ("(observation_id<>'' and exists(select 1 from scene_observations o "
                            "where o.id=semantic_embeddings.observation_id and o.snapshot_path=semantic_embeddings.image_path))")
                clauses.append(f"({current} or {retained})" if has_scene_observations else current)
            selected_columns = (
                "id,event_id,camera_id,captured_at,source_kind,source_key,"
                "image_path,bbox_json,observation_id,"
                f"{effective_label} as search_object_label"
            )
            where = " and ".join(clauses)
            cache = self._vector_cache_locked(connection, identity)
            candidates: list[sqlite3.Row | dict[str, Any]] = []
            if cache is not None:
                base_components = cache.vectors @ query.T
                extra_ids: list[int] = []
                extra_components: list[np.ndarray] = []
                for row_id, override in cache.overrides.items():
                    components = override @ query.T
                    position = cache.positions.get(row_id)
                    if position is None:
                        extra_ids.append(row_id)
                        extra_components.append(components)
                    else:
                        base_components[position] = components
                scored_ids = cache.row_ids
                all_components = base_components
                if extra_ids:
                    scored_ids = np.concatenate((
                        scored_ids, np.asarray(extra_ids, dtype=np.int64),
                    ))
                    all_components = np.concatenate((
                        all_components, np.stack(extra_components),
                    ))
                scored_positions = {
                    **cache.positions,
                    **{
                        row_id: len(cache.row_ids) + index
                        for index, row_id in enumerate(extra_ids)
                    },
                }
                if plan.composed:
                    rows = connection.execute(
                        f"select {selected_columns} from semantic_embeddings "
                        f"where {where} order by captured_at desc,id desc",
                        parameters,
                    ).fetchall()
                    component_matrix = np.empty(
                        (len(rows), len(component_names)), dtype=np.float32
                    )
                    for row in rows:
                        position = scored_positions.get(int(row["id"]))
                        if position is None:
                            continue
                        component_matrix[len(candidates)] = all_components[position]
                        candidates.append(row)
                    component_matrix = component_matrix[:len(candidates)]
                else:
                    # Score every retained-generation vector first, then ask
                    # SQLite to validate ranked IDs in batches. This proves the
                    # exact top-k without transferring metadata for the whole
                    # generation on the common single-vector query path.
                    scores = all_components[:, component_names.index("full")]
                    ranked_positions = np.argsort(scores)[::-1]
                    selected_components: list[np.ndarray] = []
                    selected_event_ids: set[int] = set()
                    target = max(1, int(limit))
                    available = 0
                    for offset in range(0, len(ranked_positions), self.VECTOR_LOAD_BATCH_ROWS):
                        positions = ranked_positions[
                            offset:offset + self.VECTOR_LOAD_BATCH_ROWS
                        ]
                        if len(positions) == 0 or float(scores[positions[0]]) < minimum_score:
                            break
                        positions = positions[scores[positions] >= minimum_score]
                        if len(positions) == 0:
                            break
                        ids = [int(scored_ids[position]) for position in positions]
                        placeholders = ",".join("?" for _ in ids)
                        rows = connection.execute(
                            f"select {selected_columns} from semantic_embeddings "
                            f"where {where} and id in ({placeholders})",
                            (*parameters, *ids),
                        ).fetchall()
                        rows_by_id = {int(row["id"]): row for row in rows}
                        for position, row_id in zip(positions, ids, strict=True):
                            row = rows_by_id.get(row_id)
                            if row is None:
                                continue
                            candidates.append(row)
                            selected_components.append(all_components[position])
                            selected_event_ids.add(int(row["event_id"]))
                            available = (
                                len(selected_event_ids) if unique_events
                                else len(candidates)
                            )
                            if available >= target:
                                break
                        if available >= target:
                            break
                    component_matrix = (
                        np.stack(selected_components)
                        if selected_components
                        else np.empty((0, len(component_names)), dtype=np.float32)
                    )
            else:
                cursor = connection.execute(
                    f"select {selected_columns},embedding_size,embedding_blob "
                    f"from semantic_embeddings where {where} "
                    "order by captured_at desc,id desc",
                    parameters,
                )
                component_batches: list[np.ndarray] = []
                while True:
                    batch = cursor.fetchmany(self.VECTOR_LOAD_BATCH_ROWS)
                    if not batch:
                        break
                    vectors: list[np.ndarray] = []
                    metadata: list[dict[str, Any]] = []
                    for row in batch:
                        size = int(row["embedding_size"] or 0)
                        raw = row["embedding_blob"]
                        if (
                            size != identity.dimensions
                            or not isinstance(raw, bytes)
                            or len(raw) != size * 2
                        ):
                            continue
                        vector = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
                        if not np.all(np.isfinite(vector)):
                            continue
                        vectors.append(vector)
                        metadata.append({
                            key: row[key]
                            for key in (
                                "id", "event_id", "camera_id", "captured_at",
                                "source_kind", "source_key", "image_path",
                                "bbox_json", "observation_id", "search_object_label",
                            )
                        })
                    if vectors:
                        component_batches.append(np.stack(vectors) @ query.T)
                        candidates.extend(metadata)
                component_matrix = (
                    np.concatenate(component_batches)
                    if component_batches
                    else np.empty((0, len(component_names)), dtype=np.float32)
                )
        if not candidates:
            return []
        component_indexes = {name: index for index, name in enumerate(component_names)}
        full_scores = component_matrix[:, component_indexes["full"]]
        if plan.composed:
            required_scores = np.stack([
                component_matrix[:, component_indexes[name]] for name in plan.required
            ], axis=1)
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
            rank_scores = full_scores
            eligible = np.ones(len(candidates), dtype=bool)
            best_rank = float(rank_scores.max())
        ordered = np.argsort(rank_scores)[::-1]
        hits: list[SemanticSearchHit] = []
        seen_event_ids: set[int] = set()
        for candidate_index in ordered:
            if not bool(eligible[candidate_index]):
                continue
            score = float(full_scores[candidate_index])
            if score < minimum_score:
                continue
            row = candidates[int(candidate_index)]
            event_id = int(row["event_id"])
            if unique_events and event_id in seen_event_ids:
                continue
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
                event_id=event_id, camera_id=str(row["camera_id"]),
                captured_at=str(row["captured_at"]), source_kind=str(row["source_kind"]),
                source_key=str(row["source_key"]), image_path=str(row["image_path"]),
                object_label=str(row["search_object_label"]), bbox=bbox, score=score,
                rank_score=rank_score,
                match_strength=match_strength,
                component_scores=component_scores,
                observation_id=str(row["observation_id"]),
            ))
            seen_event_ids.add(event_id)
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
                    embedding_blob, created_at, evidence_revision, observation_id
                )
                select event_id, camera_id, captured_at, source_kind, source_key,
                    image_path, object_label, bbox_json, ?, ?, ?, ?,
                    embedding_blob, ?, evidence_revision, observation_id
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
            written = connection.total_changes - before
        if written:
            self._record_cache_mutation(target, force_rebuild=True)
        return written

    def event_indexed(self, event_id: int, identity: SemanticModelIdentity) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                select 1 from semantic_embeddings
                where event_id = ? and model_fingerprint = ?
                    and preprocessing_fingerprint = ? and observation_id='' limit 1
                """,
                (int(event_id), identity.model_fingerprint, identity.preprocessing_fingerprint),
            ).fetchone()
        return row is not None

    def event_source_indexed(
        self,
        event_id: int,
        identity: SemanticModelIdentity,
        source_kind: str,
        *,
        event: dict[str, Any] | None = None,
    ) -> bool:
        return bool(self.event_source_keys(event_id, identity, source_kind, event=event))

    def event_source_keys(
        self,
        event_id: int,
        identity: SemanticModelIdentity,
        source_kind: str,
        *,
        event: dict[str, Any] | None = None,
    ) -> set[str]:
        parameters: list[Any] = [
            int(event_id), identity.model_fingerprint,
            identity.preprocessing_fingerprint, str(source_kind),
        ]
        current_evidence = ""
        if event is not None:
            current_evidence = " and evidence_revision = ? and image_path = ?"
            parameters.extend([
                int(event.get("evidence_revision") or 0), str(event.get("snapshot_path") or ""),
            ])
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                select source_key from semantic_embeddings
                where event_id = ? and model_fingerprint = ?
                    and preprocessing_fingerprint = ? and source_kind = ?
                    and observation_id=''
                    {current_evidence}
                """,
                parameters,
            ).fetchall()
        return {str(row["source_key"]) for row in rows}

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
        deleted = max(0, int(cursor.rowcount or 0))
        if deleted:
            self._record_cache_mutation(identity, force_rebuild=True)
        return deleted

    def delete_event(self, event_id: int, *, expected_event: dict[str, Any] | None = None) -> int:
        """Remove semantic evidence for an event that is not object-searchable."""
        with self._lock, self._connect() as connection:
            if expected_event is not None:
                connection.execute("begin immediate")
                if not self._event_matches(connection, expected_event):
                    return 0
            cursor = connection.execute(
                "delete from semantic_embeddings where event_id = ?" + (" and observation_id=''" if expected_event is not None else ""),
                (int(event_id),),
            )
            connection.execute("delete from semantic_projection_receipts where event_id=?", (int(event_id),))
        deleted = max(0, int(cursor.rowcount or 0))
        if deleted:
            self._record_cache_mutation(None, force_rebuild=True)
        return deleted

    def indexed_observation_keys(self, event_id: int, identity: SemanticModelIdentity) -> set[tuple[str, str]]:
        """Observation/image pairs already stored for this event and model generation."""
        if event_id <= 0:
            return set()
        with self._connect() as connection:
            rows = connection.execute(
                "select observation_id, image_path from semantic_embeddings "
                "where event_id=? and model_fingerprint=? and preprocessing_fingerprint=? "
                "and observation_id!=''",
                (int(event_id), identity.model_fingerprint, identity.preprocessing_fingerprint),
            ).fetchall()
        return {(str(row["observation_id"]), str(row["image_path"])) for row in rows}

    def reconcile_observation_keys(
        self, event: dict[str, Any], identity: SemanticModelIdentity,
        desired: set[tuple[str, str]],
    ) -> int:
        """Delete retired observation vectors under the event revision guard."""
        event_id = int(event.get("id") or 0)
        with self._lock, self._connect() as connection:
            connection.execute("begin immediate")
            if not self._event_matches(connection, event):
                return 0
            rows = connection.execute(
                "select id,observation_id,image_path from semantic_embeddings "
                "where event_id=? and model_fingerprint=? and preprocessing_fingerprint=? "
                "and observation_id!=''",
                (event_id, identity.model_fingerprint, identity.preprocessing_fingerprint),
            ).fetchall()
            retired = [
                int(row["id"]) for row in rows
                if (str(row["observation_id"]), str(row["image_path"])) not in desired
            ]
            connection.executemany(
                "delete from semantic_embeddings where id=?", ((row_id,) for row_id in retired),
            )
        if retired:
            self._record_cache_mutation(identity, [event_id])
        return len(retired)

    def indexed_event_ids(self) -> set[int]:
        """Return event IDs with any semantic evidence, across model generations."""
        with self._connect() as connection:
            rows = connection.execute(
                "select distinct event_id from semantic_embeddings"
            ).fetchall()
        return {int(row["event_id"]) for row in rows}

    def resolve_model_identity(self, model_dir: Path, manifest: dict[str, Any]) -> SemanticModelIdentity:
        """Reuse the generation whose declared joint embedding space matches.

        Packages may explicitly declare a stable ``embedding_space_id``. When
        they do not, both towers and the tokenizer are part of the contract so
        text vectors are never compared with incompatible stored image vectors.
        """
        contract = semantic_embedding_contract(model_dir, manifest)
        bound = self._generation_source(contract)
        if bound is not None:
            return bound
        fresh = SemanticModelIdentity(
            _semantic_implementation(manifest),
            contract,
            _preprocessing_fingerprint(
                manifest, include_text=not bool(str(manifest.get("embedding_space_id") or "").strip())
            ),
            _semantic_dimensions(manifest),
        )
        chosen = fresh
        # Hash the legacy whole-directory key only when an older generation
        # might already hold these image vectors.
        if not self._generation_recorded(fresh) and self._has_recorded_generation():
            legacy = legacy_semantic_model_identity(model_dir, manifest)
            if self._generation_recorded(legacy):
                chosen = legacy
        with self._lock, self._connect() as connection:
            connection.execute("begin immediate")
            row = self._generation_source_row(connection, contract)
            if row is not None:
                return self._identity_from_source(row)
            connection.execute(
                """
                insert or ignore into semantic_generation_sources (
                    image_contract, implementation, model_fingerprint,
                    preprocessing_fingerprint, dimensions
                ) values (?, ?, ?, ?, ?)
                """,
                (
                    contract,
                    chosen.implementation,
                    chosen.model_fingerprint,
                    chosen.preprocessing_fingerprint,
                    chosen.dimensions,
                ),
            )
            row = self._generation_source_row(connection, contract)
        if row is None:
            raise RuntimeError("semantic generation source was not recorded")
        return self._identity_from_source(row)

    def _generation_source(self, contract: str) -> SemanticModelIdentity | None:
        with self._connect() as connection:
            row = self._generation_source_row(connection, contract)
        return self._identity_from_source(row) if row is not None else None

    @staticmethod
    def _generation_source_row(connection: sqlite3.Connection, contract: str) -> sqlite3.Row | None:
        return connection.execute(
            """
            select implementation, model_fingerprint, preprocessing_fingerprint, dimensions
            from semantic_generation_sources where image_contract = ?
            """,
            (contract,),
        ).fetchone()

    @staticmethod
    def _identity_from_source(row: sqlite3.Row) -> SemanticModelIdentity:
        return SemanticModelIdentity(
            str(row["implementation"]),
            str(row["model_fingerprint"]),
            str(row["preprocessing_fingerprint"]),
            int(row["dimensions"]),
        )

    def _generation_recorded(self, identity: SemanticModelIdentity) -> bool:
        with self._connect() as connection:
            for table in ("semantic_embeddings", "semantic_projection_receipts"):
                row = connection.execute(
                    f"""
                    select 1 from {table}
                    where model_fingerprint = ? and preprocessing_fingerprint = ?
                    limit 1
                    """,
                    (identity.model_fingerprint, identity.preprocessing_fingerprint),
                ).fetchone()
                if row is not None:
                    return True
        return False

    def _has_recorded_generation(self) -> bool:
        with self._connect() as connection:
            for table in ("semantic_embeddings", "semantic_projection_receipts"):
                if connection.execute(f"select 1 from {table} limit 1").fetchone() is not None:
                    return True
        return False

    def backfill_state(self, identity: SemanticModelIdentity) -> SemanticBackfillState:
        with self._connect() as connection:
            row = connection.execute(
                """
                select before_created_at, before_id, head_created_at, head_id, complete
                from semantic_backfill_state
                where model_fingerprint = ? and preprocessing_fingerprint = ?
                """,
                (identity.model_fingerprint, identity.preprocessing_fingerprint),
            ).fetchone()
        if row is None:
            return SemanticBackfillState()
        before_id = row["before_id"]
        head_id = row["head_id"]
        return SemanticBackfillState(
            before_created_at=row["before_created_at"],
            before_id=None if before_id is None else int(before_id),
            head_created_at=row["head_created_at"],
            head_id=None if head_id is None else int(head_id),
            complete=bool(row["complete"]),
        )

    def save_backfill_state(
        self,
        identity: SemanticModelIdentity,
        state: SemanticBackfillState,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                insert into semantic_backfill_state (
                    model_fingerprint, preprocessing_fingerprint,
                    before_created_at, before_id, head_created_at, head_id, complete
                ) values (?, ?, ?, ?, ?, ?, ?)
                on conflict (model_fingerprint, preprocessing_fingerprint) do update set
                    before_created_at = excluded.before_created_at,
                    before_id = excluded.before_id,
                    head_created_at = excluded.head_created_at,
                    head_id = excluded.head_id,
                    complete = excluded.complete
                """,
                (
                    identity.model_fingerprint,
                    identity.preprocessing_fingerprint,
                    state.before_created_at,
                    state.before_id,
                    state.head_created_at,
                    state.head_id,
                    int(state.complete),
                ),
            )

    def record_media_failure(
        self, identity: SemanticModelIdentity, event_id: int, error: BaseException,
    ) -> bool:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                insert into semantic_media_failures(
                    event_id,model_fingerprint,preprocessing_fingerprint,
                    failure_count,last_error,last_attempt_at,next_attempt_at
                ) values(?,?,?,?,?,?,?)
                on conflict(event_id,model_fingerprint,preprocessing_fingerprint)
                do update set failure_count=failure_count+1,
                    last_error=excluded.last_error,last_attempt_at=excluded.last_attempt_at,
                    next_attempt_at=excluded.next_attempt_at
                """,
                (int(event_id), identity.model_fingerprint, identity.preprocessing_fingerprint,
                 1, str(error)[:500], now, now + SEMANTIC_MEDIA_QUARANTINE_RETRY_SECONDS),
            )
            count = connection.execute(
                "select failure_count from semantic_media_failures where event_id=? "
                "and model_fingerprint=? and preprocessing_fingerprint=?",
                (int(event_id), identity.model_fingerprint, identity.preprocessing_fingerprint),
            ).fetchone()[0]
        return int(count) >= SEMANTIC_MEDIA_QUARANTINE_ATTEMPTS

    def clear_media_failure(self, identity: SemanticModelIdentity, event_id: int) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "delete from semantic_media_failures where event_id=? and model_fingerprint=? "
                "and preprocessing_fingerprint=?",
                (int(event_id), identity.model_fingerprint, identity.preprocessing_fingerprint),
            )

    def due_media_failures(
        self, identity: SemanticModelIdentity, *, limit: int = 10,
    ) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute(
                "select event_id from semantic_media_failures where model_fingerprint=? "
                "and preprocessing_fingerprint=? and failure_count>=? and next_attempt_at<=? "
                "order by next_attempt_at,event_id limit ?",
                (identity.model_fingerprint, identity.preprocessing_fingerprint,
                 SEMANTIC_MEDIA_QUARANTINE_ATTEMPTS, time.time(), max(1, min(100, int(limit)))),
            ).fetchall()
        return [int(row[0]) for row in rows]

    def has_media_failures(self, identity: SemanticModelIdentity) -> bool:
        with self._connect() as connection:
            return connection.execute(
                "select 1 from semantic_media_failures where model_fingerprint=? "
                "and preprocessing_fingerprint=? limit 1",
                (identity.model_fingerprint, identity.preprocessing_fingerprint),
            ).fetchone() is not None


def fingerprint_model_package(model_dir: Path) -> str:
    """Fingerprint every byte under a model directory.

    This is the legacy generation key. A current index binds that key to the
    image-encoder contract so an unrelated file cannot start a new generation.
    """
    digest = hashlib.sha256()
    for path in sorted(item for item in Path(model_dir).rglob("*") if item.is_file()):
        relative = path.relative_to(model_dir).as_posix()
        _update_fingerprint(digest, relative, path)
    return digest.hexdigest()[:24]


def _update_fingerprint(digest: Any, name: str, path: Path) -> None:
    encoded_name = name.encode("utf-8")
    digest.update(len(encoded_name).to_bytes(8, "big"))
    digest.update(encoded_name)
    digest.update(path.stat().st_size.to_bytes(8, "big"))
    with path.open("rb") as handle:
        while chunk := handle.read(MODEL_FINGERPRINT_CHUNK_SIZE):
            digest.update(chunk)


def _semantic_dimensions(manifest: dict[str, Any]) -> int:
    dimensions = int(manifest.get("dimensions") or 0)
    if not 0 < dimensions <= 8192:
        raise RuntimeError("semantic manifest dimensions must be between 1 and 8192")
    return dimensions


def _preprocessing_fingerprint(manifest: dict[str, Any], *, include_text: bool) -> str:
    payload: dict[str, Any] = {"image": dict(manifest.get("image") or {})}
    if include_text:
        payload["text"] = dict(manifest.get("text") or {})
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def semantic_image_artifacts(model_dir: Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
    """Return the image encoder IR pair declared by the manifest."""
    xml_path = _semantic_package_path(
        model_dir, manifest.get("image_model"), "image_encoder.xml"
    )
    binary_path = xml_path.with_suffix(".bin")
    return xml_path, binary_path


def semantic_image_contract(model_dir: Path, manifest: dict[str, Any]) -> str:
    """Identity of the image tower that produced stored vectors.

    Text-encoder, tokenizer, license, and cache files are query or package
    metadata. They are not part of an image vector and must not invalidate one.
    """
    xml_path, binary_path = semantic_image_artifacts(model_dir, manifest)
    relative_xml = xml_path.relative_to(Path(model_dir).resolve()).as_posix()
    digest = hashlib.sha256()
    for name, path in ((relative_xml, xml_path), (str(Path(relative_xml).with_suffix(".bin")), binary_path)):
        if not path.is_file():
            raise RuntimeError(f"semantic image model artifact is missing: {path}")
        _update_fingerprint(digest, name, path)
    material = (
        f"{digest.hexdigest()}:{_preprocessing_fingerprint(manifest, include_text=False)}:"
        f"{_semantic_dimensions(manifest)}"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:24]


def semantic_embedding_contract(model_dir: Path, manifest: dict[str, Any]) -> str:
    """Identity of the shared image/text vector space used for retrieval."""
    image_contract = semantic_image_contract(model_dir, manifest)
    declared = str(manifest.get("embedding_space_id") or "").strip()
    if declared:
        material = f"declared:{declared}:{image_contract}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()[:24]

    text_spec = dict(manifest.get("text") or {})
    text_xml = _semantic_package_path(
        model_dir, manifest.get("text_model"), "text_encoder.xml"
    )
    text_bin = text_xml.with_suffix(".bin")
    tokenizer = _semantic_package_path(
        model_dir,
        text_spec.get("tokenizer_path"),
        "tokenizer/bpe_simple_vocab_16e6.txt.gz",
    )
    digest = hashlib.sha256()
    for path in (text_xml, text_bin, tokenizer):
        if not path.is_file():
            raise RuntimeError(f"semantic text model artifact is missing: {path}")
        _update_fingerprint(
            digest, path.relative_to(Path(model_dir).resolve()).as_posix(), path
        )
    material = (
        f"derived:{image_contract}:{digest.hexdigest()}:"
        f"{_preprocessing_fingerprint(manifest, include_text=True)}"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:24]


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

    def refresh_event(self, event: dict[str, Any]) -> bool:
        self.index.delete_event(int(event.get("id") or 0))
        return self.queue_event(event)

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


def _semantic_implementation(manifest: dict[str, Any]) -> str:
    return str(manifest.get("implementation") or "openvino_manifest")


def legacy_semantic_model_identity(
    model_dir: Path,
    manifest: dict[str, Any],
) -> SemanticModelIdentity:
    """Generation key used before image vectors were bound to the image tower."""
    return SemanticModelIdentity(
        _semantic_implementation(manifest),
        fingerprint_model_package(model_dir),
        _preprocessing_fingerprint(manifest, include_text=True),
        _semantic_dimensions(manifest),
    )


def _semantic_model_identity(
    model_dir: Path,
    manifest: dict[str, Any],
) -> SemanticModelIdentity:
    """Generation key for a compatible joint image/text embedding space."""
    return SemanticModelIdentity(
        _semantic_implementation(manifest),
        semantic_embedding_contract(model_dir, manifest),
        _preprocessing_fingerprint(
            manifest, include_text=not bool(str(manifest.get("embedding_space_id") or "").strip())
        ),
        _semantic_dimensions(manifest),
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

    def __init__(
        self,
        model_dir: Path,
        manifest: dict[str, Any],
        device: str,
        identity: SemanticModelIdentity | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.manifest = dict(manifest)
        self.device = str(device)
        self._image_spec = dict(manifest.get("image") or {})
        self._text_spec = dict(manifest.get("text") or {})
        self._tokenizer = _semantic_tokenizer(self.model_dir, self._text_spec)
        self._identity = identity or _semantic_model_identity(self.model_dir, manifest)
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


@dataclass
class _SemanticEventRevision:
    event: dict[str, Any]
    valid: bool = True
    pending: bool = False
    historical: bool = False
    outcome: str = ""


class SemanticMediaUnavailable(RuntimeError):
    """A retained historical image is temporarily unavailable for projection."""


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
        self._event_revision_lock = threading.RLock()
        # Tokens are retained by queued/in-flight work, not by process lifetime.
        self._event_revisions: weakref.WeakValueDictionary[int, _SemanticEventRevision] = weakref.WeakValueDictionary()
        self._event_store: Any = None

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
            self._event_store = event_store
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
        identity: SemanticModelIdentity | None = None
        while not self._stop.is_set():
            try:
                # A package without its image encoder fails in the worker
                # constructor. Resolve only after the files exist so a test
                # double, or a retry once the files appear, still runs.
                resolved = identity
                if resolved is None:
                    try:
                        resolved = self.index.resolve_model_identity(
                            self.model_dir, self.manifest
                        )
                    except RuntimeError as exc:
                        if "artifact is missing" not in str(exc):
                            raise
                    else:
                        identity = resolved
                encoder = IsolatedOpenVinoManifestEncoder(
                    self.model_dir,
                    self.manifest,
                    target_device,
                    resolved,
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
        """Retry transient failures and periodically revisit quarantined media."""
        while not self._stop.is_set():
            try:
                self._retry_quarantined_media(event_store)
                self._backfill(event_store)
                encoder = self.encoder
                if encoder is not None:
                    self.index.warm_search_cache(encoder.identity)
                if encoder is None or not self.index.has_media_failures(encoder.identity):
                    return
                if self._stop.wait(SEMANTIC_MEDIA_QUARANTINE_RETRY_SECONDS):
                    return
            except Exception as exc:
                self._error = str(exc)
                LOGGER.warning("semantic historical indexing interrupted: %s", exc)
                if self._stop.wait(SEMANTIC_BACKFILL_RETRY_SECONDS):
                    return

    def _retry_quarantined_media(self, event_store: Any) -> None:
        encoder = self.encoder
        getter = getattr(event_store, "get", None)
        if encoder is None or not callable(getter):
            return
        for event_id in self.index.due_media_failures(encoder.identity):
            if self._stop.is_set():
                return
            event = getter(event_id)
            if event is None:
                self.index.clear_media_failure(encoder.identity, event_id)
                continue
            queued = self._revision_event(event, refresh=True)
            revision = queued.get("_semantic_revision")
            if isinstance(revision, _SemanticEventRevision):
                revision.historical = True
            try:
                self.index_event(queued)
            except SemanticMediaUnavailable as exc:
                self.index.record_media_failure(encoder.identity, event_id, exc)
            else:
                if self.projection_current(queued):
                    self.index.clear_media_failure(encoder.identity, event_id)

    def _history_pending(self) -> bool:
        with self._event_revision_lock:
            return any(
                token.historical and token.pending
                for token in self._event_revisions.values()
            )

    def _wait_for_queued_history(self) -> None:
        """Let queued history finish before the cursor moves past it.

        Direct backfill calls used by tests have no worker thread. They keep
        the previous queue-and-return behavior.
        """
        worker = self._thread
        if worker is None or not worker.is_alive():
            return
        while not self._stop.is_set() and self._history_pending():
            self._stop.wait(0.05)

    def _finish_history_page(self, tokens: list[_SemanticEventRevision]) -> bool:
        self._wait_for_queued_history()
        if self._stop.is_set():
            return False
        if any(token.outcome == "failed" for token in tokens):
            raise RuntimeError("semantic historical indexing failed")
        return True

    def _backfill(self, event_store: Any) -> None:
        encoder = self.encoder
        if encoder is None:
            return
        identity = encoder.identity
        if not self.config.index_full_frame:
            self.index.delete_generation_source(identity, "full_frame")
        if not self.config.index_object_crops:
            self.index.delete_generation_source(identity, "object_crop")
        state = self.index.backfill_state(identity)
        indexed_event_ids = self.index.indexed_event_ids()
        if state.complete and state.head_created_at is not None and state.head_id is not None:
            head = self._backfill_newer(
                event_store,
                state.head_created_at,
                state.head_id,
                indexed_event_ids,
            )
            if head is None:
                return
            if head != (state.head_created_at, state.head_id):
                self.index.save_backfill_state(identity, SemanticBackfillState(
                    before_created_at=state.before_created_at,
                    before_id=state.before_id,
                    head_created_at=head[0],
                    head_id=head[1],
                    complete=True,
                ))
            return
        before_created_at = state.before_created_at
        before_id = state.before_id
        head_created_at = state.head_created_at
        head_id = state.head_id
        if head_created_at is not None and head_id is not None:
            head = self._backfill_newer(
                event_store, head_created_at, head_id, indexed_event_ids,
            )
            if head is None:
                return
            head_created_at, head_id = head
        while not self._stop.is_set():
            rows = event_store.recent_compact(
                self.config.backfill_batch_size,
                before_created_at,
                before_id,
            )
            if not rows:
                self.index.save_backfill_state(identity, SemanticBackfillState(
                    before_created_at=before_created_at,
                    before_id=before_id,
                    head_created_at=head_created_at,
                    head_id=head_id,
                    complete=True,
                ))
                return
            if head_created_at is None or head_id is None:
                head_created_at = str(rows[0].get("created_at") or "")
                head_id = int(rows[0].get("id") or 0)
            tokens = self._consume_backfill_rows(event_store, rows, indexed_event_ids)
            if tokens is None or not self._finish_history_page(tokens):
                return
            last = rows[-1]
            before_created_at = str(last.get("created_at") or "")
            before_id = int(last.get("id") or 0)
            finished = len(rows) < self.config.backfill_batch_size
            self.index.save_backfill_state(identity, SemanticBackfillState(
                before_created_at=before_created_at,
                before_id=before_id,
                head_created_at=head_created_at,
                head_id=head_id,
                complete=finished,
            ))
            if finished:
                return

    def _backfill_newer(
        self,
        event_store: Any,
        after_created_at: str,
        after_id: int,
        indexed_event_ids: set[int],
    ) -> tuple[str, int] | None:
        """Index incidents admitted after the saved head. Return the new head."""
        newest = (after_created_at, after_id)
        before_created_at: str | None = None
        before_id: int | None = None
        first_page = True
        since = getattr(event_store, "recent_compact_since", None)
        if not callable(since):
            raise RuntimeError("event store cannot page incidents newer than the semantic cursor")
        while not self._stop.is_set():
            rows = since(
                self.config.backfill_batch_size,
                after_created_at,
                after_id,
                before_created_at,
                before_id,
            )
            if not rows:
                return newest
            if first_page:
                newest = (str(rows[0].get("created_at") or ""), int(rows[0].get("id") or 0))
                first_page = False
            tokens = self._consume_backfill_rows(event_store, rows, indexed_event_ids)
            if tokens is None or not self._finish_history_page(tokens):
                return None
            if len(rows) < self.config.backfill_batch_size:
                return newest
            last = rows[-1]
            before_created_at = str(last.get("created_at") or "")
            before_id = int(last.get("id") or 0)
        return None

    def _consume_backfill_rows(
        self,
        event_store: Any,
        rows: list[dict[str, Any]],
        indexed_event_ids: set[int],
    ) -> list[_SemanticEventRevision] | None:
        tokens: list[_SemanticEventRevision] = []
        for event in reversed(rows):
            if self._stop.is_set():
                return None
            event_id = int(event.get("id") or 0)
            has_scene = getattr(event_store, "scene_has_search_observation", None)
            scene_present = has_scene(event_id) if callable(has_scene) else next(self._scene_observations(event_id), None) is not None
            if not semantic_event_searchable(event) and not scene_present:
                if event_id > 0 and event_id in indexed_event_ids:
                    # Resolve current evidence before acting on a historical snapshot.
                    self.index_event(event)
                    indexed_event_ids.discard(event_id)
                continue
            if self.encoder and self.projection_current(event):
                continue
            while not self._stop.is_set():
                if not self._history_queue_has_capacity():
                    self._stop.wait(0.1)
                    continue
                queued = self._revision_event(event)
                revision = queued.get("_semantic_revision")
                if isinstance(revision, _SemanticEventRevision):
                    revision.outcome = ""
                    revision.historical = True
                    revision.pending = True
                try:
                    self._queue.put(
                        (1, next(self._queue_sequence), queued),
                        timeout=0.5,
                    )
                except queue.Full:
                    if isinstance(revision, _SemanticEventRevision):
                        revision.pending = False
                    continue
                if isinstance(revision, _SemanticEventRevision):
                    tokens.append(revision)
                break
            else:
                return None
        return tokens

    def _revision_event(self, event: dict[str, Any], *, refresh: bool = False) -> dict[str, Any]:
        with self._event_revision_lock:
            event_id = int(event.get("id") or 0)
            # Notifications and history pages can arrive out of order. The event
            # store owns the latest snapshot even after prior tokens expire.
            getter = getattr(self._event_store, "get", None)
            latest = getter(event_id) if callable(getter) else event
            if latest is None:
                latest = {"id": event_id}
            current = self._event_revisions.get(event_id)
            payload = {key: value for key, value in dict(latest).items() if key != "_semantic_revision"}
            if current is None or refresh or current.event != payload:
                if current is not None:
                    current.valid = False
                current = _SemanticEventRevision(payload)
                self._event_revisions[event_id] = current
            return {**current.event, "_semantic_revision": current}

    def refresh_event(self, event: dict[str, Any]) -> bool:
        """Invalidate queued and in-flight evidence before submitting its replacement."""
        with self._event_revision_lock:
            event = self._revision_event(event, refresh=True)
            return self._queue_event_revision(event)

    def queue_event(self, event: dict[str, Any]) -> bool:
        with self._event_revision_lock:
            return self._queue_event_revision(self._revision_event(event))

    def semantic_searchable(self, event: dict[str, Any]) -> bool:
        return semantic_event_searchable(event) or next(self._scene_observations(int(event.get("id") or 0)), None) is not None

    def _queue_event_revision(self, event: dict[str, Any]) -> bool:
        if not event.get("id"):
            return False
        if not semantic_event_searchable(event):
            # A negative correction is complete without an encoder. Queued
            # work checks the same policy again before any image/model work.
            self.index.delete_event(int(event["id"]), expected_event=event)
            if not next(self._scene_observations(int(event["id"])), None):
                return True
        if self.encoder is None:
            return False
        revision = event["_semantic_revision"]
        if revision.pending:
            return True
        try:
            self._queue.put_nowait((0, next(self._queue_sequence), dict(event)))
            revision.pending = True
            return True
        except queue.Full:
            self._error = "index queue is full"
            return False

    def projection_pending(self, event: dict[str, Any]) -> bool:
        with self._event_revision_lock:
            token = self._event_revisions.get(int(event.get("id") or 0))
            return bool(token and token.valid and token.pending
                        and token.event.get("evidence_revision", 0) == event.get("evidence_revision", 0)
                        and token.event.get("scene_media_revision", 0) == event.get("scene_media_revision", 0)
                        and token.event.get("snapshot_path") == event.get("snapshot_path"))

    def _projection_plan_key(self, objects: list[dict[str, Any]]) -> str:
        plan = {
            "full_frame": self.config.index_full_frame,
            "object_crops": self.config.index_object_crops,
            "crops": [semantic_crop_source_key(index, item) for index, item, _
                      in semantic_object_crop_candidates(objects, self.config.max_object_crops_per_event)],
        }
        return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()

    def projection_current(self, event: dict[str, Any]) -> bool:
        encoder = self.encoder
        if encoder is None:
            return False
        event_id = int(event.get("id") or 0)
        if self.config.index_object_crops and self._scene_projection_missing(event_id, encoder.identity):
            return False
        objects = semantic_event_objects(event)
        if not semantic_event_searchable(event):
            return not self.index.event_indexed(event_id, encoder.identity)
        if self.config.index_full_frame and not self.index.event_source_indexed(
            event_id, encoder.identity, "full_frame", event=event
        ):
            return False
        if not self.config.index_object_crops:
            return True
        desired = {
            semantic_crop_source_key(index, item)
            for index, item, _ in semantic_object_crop_candidates(objects, self.config.max_object_crops_per_event)
        }
        receipt = self.index.projection_receipt(event, encoder.identity, self._projection_plan_key(objects))
        if receipt is not None:
            # Explicitly skipped crops are terminal for this image/revision and
            # plan. Missing files and failed inference never produce receipts.
            desired = set(receipt["indexed_crop_keys"])
        return (
            self.index.event_source_keys(event_id, encoder.identity, "object_crop", event=event) == desired
            and self.index.event_source_keys(event_id, encoder.identity, "object_crop") == desired
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                priority, _sequence, event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if event is None:
                break
            revision = event.get("_semantic_revision")
            try:
                self.index_event(event)
                self._error = ""
                if isinstance(revision, _SemanticEventRevision):
                    revision.outcome = "done"
                if self.encoder is not None and self.projection_current(event):
                    self.index.clear_media_failure(self.encoder.identity, int(event.get("id") or 0))
                if priority > 0:
                    self._stop.wait(self.config.backfill_pause_seconds)
            except Exception as exc:
                self._error = str(exc)
                if isinstance(revision, _SemanticEventRevision):
                    quarantined = bool(
                        revision.historical
                        and isinstance(exc, SemanticMediaUnavailable)
                        and self.encoder is not None
                        and self.index.record_media_failure(
                            self.encoder.identity, int(event.get("id") or 0), exc,
                        )
                    )
                    revision.outcome = "quarantined" if quarantined else "failed"
                LOGGER.warning("semantic indexing failed for event %s: %s", event.get("id"), exc)
            finally:
                if isinstance(revision, _SemanticEventRevision):
                    revision.pending = False

    def _scene_projection_missing(self, event_id: int, identity) -> bool:
        """True when an indexable observation has no embedding for this generation.

        Already indexed events are decided from ids and paths. Payloads are read
        only for observations that are not in the index yet.
        """
        media = getattr(self._event_store, "scene_observation_media", None)
        payloads = getattr(self._event_store, "scene_observation_payloads", None)
        indexed = self.index.indexed_observation_keys(event_id, identity)
        if not callable(media) or not callable(payloads):
            return any(
                (str(observation["id"]), str(observation.get("snapshot_path") or "")) not in indexed
                for observation in self._scene_observations(event_id)
            )
        current_media = {
            (str(row["id"]), str(row.get("snapshot_path") or ""))
            for row in media(event_id)
        }
        if indexed - current_media:
            return True
        missing = [
            observation_id for observation_id, path in current_media
            if (observation_id, path) not in indexed
        ]
        if not missing:
            return False
        for observation in payloads(missing):
            if self._indexable_scene_observation(observation):
                return True
        return False

    @staticmethod
    def _indexable_scene_observation(observation: dict[str, Any]) -> bool:
        try:
            item = json.loads(observation["payload_json"])
        except (TypeError, ValueError):
            return False
        return bool(
            isinstance(item, dict) and item.get("label") and observation.get("snapshot_path")
            and item.get("snapshot_visible") is not False and semantic_object_bbox(item) is not None
        )

    def _scene_observations(self, event_id: int):
        getter = getattr(self._event_store, "scene_search_observations", None)
        if not callable(getter) or not self.config.index_object_crops:
            return
        after_id = ""
        while True:
            rows = getter(event_id=event_id, after_id=after_id, limit=100)
            for observation in rows:
                try:
                    item = json.loads(observation["payload_json"])
                except (TypeError, ValueError):
                    continue
                if (isinstance(item, dict) and item.get("label") and observation.get("snapshot_path")
                        and item.get("snapshot_visible") is not False and semantic_object_bbox(item) is not None):
                    yield observation
            if len(rows) < 100:
                return
            after_id = str(rows[-1]["id"])

    def _index_scene_observations(self, event: dict[str, Any], *, historical: bool = False) -> int:
        """Retain independently addressable crops within the existing worker budget."""
        encoder = self.encoder
        if encoder is None:
            return 0
        event_id = int(event.get("id") or 0)
        identity = encoder.identity
        written = 0
        indexed = self.index.indexed_observation_keys(event_id, identity)
        observations = list(self._scene_observations(event_id))
        desired = {
            (str(observation["id"]), str(observation.get("snapshot_path") or ""))
            for observation in observations
        }
        completed = True
        for observation in observations:
            if self._stop.is_set():
                completed = False
                break
            identity_key = (str(observation["id"]), str(observation.get("snapshot_path") or ""))
            if identity_key in indexed:
                continue
            indexed.add(identity_key)
            item = json.loads(observation["payload_json"])
            image_event = {"snapshot_path": observation["snapshot_path"], "objects": [item]}
            try:
                # Load only retained evidence. A missing frame never produces
                # a vector or a successful projection receipt.
                path = event_snapshot_path(self._storage_dir, image_event, self._media_storage)
            except (FileNotFoundError, PermissionError):
                self._skipped_missing += 1
                if historical:
                    raise SemanticMediaUnavailable(
                        f"semantic observation media is unavailable: {observation['id']}"
                    )
                continue
            frame = cv2.imread(str(path))
            if frame is None:
                self._skipped_missing += 1
                if historical:
                    raise SemanticMediaUnavailable(
                        f"semantic observation media is unreadable: {observation['id']}"
                    )
                continue
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = semantic_object_bbox(item)
            source_width = positive_dimension(item.get("detection_frame_width"), width)
            source_height = positive_dimension(item.get("detection_frame_height"), height)
            box = (max(0, round(x1 * width / source_width)), max(0, round(y1 * height / source_height)),
                   min(width, round(x2 * width / source_width)), min(height, round(y2 * height / source_height)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            evidence = SemanticEvidence(
                event_id, str(observation["camera_id"]),
                datetime.fromtimestamp(float(observation["captured_epoch"]), timezone.utc).isoformat(),
                "object_crop", f"scene:{observation['id']}", str(observation["snapshot_path"]),
                str(item["label"]), box, observation_id=str(observation["id"]),
            )
            with self._encoder_lock:
                if self.encoder is not encoder:
                    completed = False
                    break
                embeddings = encoder.encode_images([frame[box[1]:box[3], box[0]:box[2]]])
            count = self.index.upsert([evidence], embeddings, identity, expected_observation=observation)
            written += count
            self._indexed += count
            # Historical scene evidence shares the configured pacing and the
            # one semantic worker, never spawning unbounded inference work.
            if self._stop.wait(self.config.backfill_pause_seconds):
                completed = False
                break
        if completed:
            self.index.reconcile_observation_keys(event, identity, desired)
        return written

    def index_event(self, event: dict[str, Any]) -> int:
        written = self._index_current_event(event)
        revision = event.get("_semantic_revision")
        return written + self._index_scene_observations(
            event,
            historical=isinstance(revision, _SemanticEventRevision) and revision.historical,
        )

    def _index_current_event(self, event: dict[str, Any]) -> int:
        """Synchronously index one event for tooling and the worker loop.

        New evidence is generation-isolated and idempotent. Encoder use is
        serialized by the service.
        """
        event_id = int(event.get("id") or 0)
        if event_id <= 0:
            return 0
        event = event if "_semantic_revision" in event else self._revision_event(event)
        revision = event["_semantic_revision"]
        with self._event_revision_lock:
            if not revision.valid:
                return 0
            objects = semantic_event_objects(event)
            if not semantic_event_searchable(event):
                self.index.delete_event(event_id, expected_event=event)
                return 0
            if self.encoder is None:
                return 0
            if self.projection_current(event):
                return 0
            plan_key = self._projection_plan_key(objects)
            identity = self.encoder.identity
            full_frame_needed = self.config.index_full_frame and not self.index.event_source_indexed(
                event_id, identity, "full_frame", event=event
            )
            crop_candidates = semantic_object_crop_candidates(
                objects, self.config.max_object_crops_per_event
            )
            desired_crop_keys = {
                semantic_crop_source_key(index, item)
                for index, item, _coordinates in crop_candidates
            }
            existing_crop_keys = self.index.event_source_keys(
                event_id, identity, "object_crop", event=event
            )
            all_crop_keys = self.index.event_source_keys(event_id, identity, "object_crop")
            object_crops_needed = (
                self.config.index_object_crops
                and bool(desired_crop_keys - existing_crop_keys)
            )
            if not full_frame_needed and not object_crops_needed:
                if self.config.index_object_crops and all_crop_keys != desired_crop_keys:
                    self.index.upsert(
                        [], [], identity, expected_event=event,
                        reconcile_sources={"object_crop": desired_crop_keys},
                    )
                return 0
        try:
            path = event_snapshot_path(self._storage_dir, event, self._media_storage)
        except (FileNotFoundError, PermissionError):
            self._skipped_missing += 1
            if revision.historical:
                raise SemanticMediaUnavailable(
                    f"semantic event media is unavailable: {event_id}"
                )
            return 0
        frame = cv2.imread(str(path))
        if frame is None:
            self._skipped_missing += 1
            if revision.historical:
                raise SemanticMediaUnavailable(
                    f"semantic event media is unreadable: {event_id}"
                )
            return 0
        skipped_crops: dict[str, str] = {}
        indexed_crop_keys: set[str] = set(existing_crop_keys & desired_crop_keys)
        evidence: list[SemanticEvidence] = []
        images: list[np.ndarray] = []
        if full_frame_needed:
            evidence.append(SemanticEvidence(event_id, str(event.get("camera_id") or ""), str(event.get("created_at") or ""), "full_frame", "frame", str(event["snapshot_path"]), evidence_revision=int(event.get("evidence_revision") or 0)))
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
                    skipped_crops[source_key] = "empty_after_clipping"
                    continue
                indexed_crop_keys.add(source_key)
                label = str(item.get("label") or "").strip().lower()
                evidence.append(SemanticEvidence(event_id, str(event.get("camera_id") or ""), str(event.get("created_at") or ""), "object_crop", source_key, str(event["snapshot_path"]), label, (x1, y1, x2, y2), int(event.get("evidence_revision") or 0)))
                images.append(frame[y1:y2, x1:x2])
        embeddings = []
        if images:
            with self._encoder_lock:
                if self.encoder is None:
                    return 0
                embeddings = self.encoder.encode_images(images)
        with self._event_revision_lock:
            if not revision.valid:
                return 0
            written = self.index.upsert(
                evidence, embeddings, identity, expected_event=event,
                reconcile_sources=(
                    {"object_crop": indexed_crop_keys} if self.config.index_object_crops else {}
                ),
                projection_receipt={"plan_key": plan_key,
                    "indexed_crop_keys": sorted(indexed_crop_keys), "skipped_crops": skipped_crops},
            )
            self._indexed += written
            return written

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
        with self._lifecycle_lock:
            state = self._state
            backfill_active = bool(
                self._backfill_thread and self._backfill_thread.is_alive()
            )
        identity = self.encoder.identity if self.encoder else None
        retry_in_seconds = max(0.0, self._next_retry_at - time.monotonic())
        return {
            "enabled": True, "state": state,
            "backfill_active": backfill_active,
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
            "search_cache": self.index.search_cache_status(identity),
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

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import logging
import queue
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import cv2

from .config import SemanticSearchConfig
from .incident_utils import event_snapshot_path

LOGGER = logging.getLogger("uvicorn.error")


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

    def start(self, event_store: Any, storage_dir: Path) -> None:
        return

    def queue_event(self, event: dict[str, Any]) -> bool:
        return False

    def search_text(self, query: str, **filters: Any) -> list[SemanticSearchHit]:
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


class OpenVinoManifestEncoder:
    """OpenVINO dual-encoder loaded entirely from a local model package."""

    def __init__(self, model_dir: Path, manifest: dict[str, Any], device: str) -> None:
        from openvino import Core
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("semantic tokenizer runtime is not installed") from exc
        self.model_dir = Path(model_dir)
        image_spec = dict(manifest.get("image") or {})
        text_spec = dict(manifest.get("text") or {})
        dimensions = int(manifest.get("dimensions") or 0)
        if not 0 < dimensions <= 8192:
            raise RuntimeError("semantic manifest dimensions must be between 1 and 8192")
        image_model = self.model_dir / str(manifest.get("image_model") or "image_encoder.xml")
        text_model = self.model_dir / str(manifest.get("text_model") or "text_encoder.xml")
        if not image_model.is_file() or not text_model.is_file():
            raise RuntimeError("semantic image or text OpenVINO model is missing")
        core = Core()
        self._image_model = core.compile_model(str(image_model), device)
        self._text_model = core.compile_model(str(text_model), device)
        tokenizer_path = self.model_dir / str(text_spec.get("tokenizer_path") or "tokenizer")
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self._image_spec = image_spec
        self._text_spec = text_spec
        preprocessing = json.dumps(
            {"image": image_spec, "text": text_spec}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._identity = SemanticModelIdentity(
            str(manifest.get("implementation") or "openvino_manifest"),
            fingerprint_model_package(self.model_dir),
            hashlib.sha256(preprocessing).hexdigest()[:24],
            dimensions,
        )

    @property
    def identity(self) -> SemanticModelIdentity:
        return self._identity

    @staticmethod
    def _output(compiled: Any, result: Any, name: str) -> np.ndarray:
        if name:
            return np.asarray(result[compiled.output(name)])
        return np.asarray(result[compiled.output(0)])

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        spec = self._image_spec
        size = int(spec.get("size") or 256)
        mean = np.asarray(spec.get("mean") or [0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.asarray(spec.get("std") or [0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        prepared = []
        for image in images:
            resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            prepared.append(np.transpose((rgb - mean) / std, (2, 0, 1)))
        batch = np.stack(prepared).astype(np.float32)
        input_name = str(spec.get("input") or self._image_model.input(0).get_any_name())
        result = self._image_model({input_name: batch})
        return normalized_matrix(self._output(self._image_model, result, str(spec.get("output") or "")))

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        max_length = int(self._text_spec.get("max_length") or 77)
        tokens = self._tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=max_length, return_tensors="np",
        )
        mapping = dict(self._text_spec.get("inputs") or {})
        inputs: dict[str, np.ndarray] = {}
        for model_input in self._text_model.inputs:
            name = model_input.get_any_name()
            token_name = str(mapping.get(name) or name)
            if token_name in tokens:
                inputs[name] = np.asarray(tokens[token_name])
        result = self._text_model(inputs)
        return normalized_matrix(self._output(self._text_model, result, str(self._text_spec.get("output") or "")))

    def close(self) -> None:
        self._image_model = None
        self._text_model = None


class SemanticSearchService(DisabledSemanticSearch):
    """Low-priority asynchronous incident indexer and text search service."""

    def __init__(self, config: SemanticSearchConfig, index: SemanticIndex, model_dir: Path, manifest: dict[str, Any]) -> None:
        super().__init__(config, index)
        self.model_dir = model_dir
        self.manifest = manifest
        self.encoder: SemanticEncoder | None = None
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(config.worker_queue_size)
        self._thread: threading.Thread | None = None
        self._backfill_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._storage_dir = Path()
        self._error = ""
        self._indexed = 0

    def start(self, event_store: Any, storage_dir: Path) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._storage_dir = Path(storage_dir)
        try:
            self.encoder = OpenVinoManifestEncoder(self.model_dir, self.manifest, self.config.device)
        except Exception as exc:
            self._error = str(exc)
            LOGGER.warning("semantic search unavailable: %s", exc)
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="survng-semantic", daemon=True)
        self._thread.start()
        self._backfill_thread = threading.Thread(
            target=self._backfill,
            args=(event_store,),
            name="survng-semantic-backfill",
            daemon=True,
        )
        self._backfill_thread.start()

    def _backfill(self, event_store: Any) -> None:
        before_created_at: str | None = None
        before_id: int | None = None
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
                if self.encoder and self.index.event_indexed(int(event.get("id") or 0), self.encoder.identity):
                    continue
                try:
                    self._queue.put(dict(event), timeout=0.5)
                except queue.Full:
                    if self._stop.is_set():
                        return
            last = rows[-1]
            before_created_at = str(last.get("created_at") or "")
            before_id = int(last.get("id") or 0)
            if len(rows) < self.config.backfill_batch_size:
                return

    def queue_event(self, event: dict[str, Any]) -> bool:
        if self.encoder is None or not event.get("snapshot_path") or not event.get("id"):
            return False
        try:
            self._queue.put_nowait(dict(event))
            return True
        except queue.Full:
            self._error = "index queue is full"
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if event is None:
                break
            try:
                self._index_event(event)
            except Exception as exc:
                self._error = str(exc)
                LOGGER.warning("semantic indexing failed for event %s: %s", event.get("id"), exc)

    def _index_event(self, event: dict[str, Any]) -> None:
        if self.encoder is None:
            return
        path = event_snapshot_path(self._storage_dir, event)
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError("incident snapshot could not be decoded")
        try:
            objects = json.loads(str(event.get("objects_json") or "[]"))
        except json.JSONDecodeError:
            objects = []
        if not isinstance(objects, list) or not objects:
            return
        evidence: list[SemanticEvidence] = []
        images: list[np.ndarray] = []
        if self.config.index_full_frame:
            evidence.append(SemanticEvidence(int(event["id"]), str(event["camera_id"]), str(event["created_at"]), "full_frame", "frame", str(event["snapshot_path"])))
            images.append(frame)
        if self.config.index_object_crops:
            height, width = frame.shape[:2]
            for index, item in enumerate(objects):
                bbox = item.get("bbox") if isinstance(item, dict) else None
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = (int(value) for value in bbox)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                label = str(item.get("label") or "").strip().lower()
                evidence.append(SemanticEvidence(int(event["id"]), str(event["camera_id"]), str(event["created_at"]), "object_crop", f"{label}:{index}", str(event["snapshot_path"]), label, (x1, y1, x2, y2)))
                images.append(frame[y1:y2, x1:x2])
        if images:
            embeddings = self.encoder.encode_images(images)
            self._indexed += self.index.upsert(evidence, embeddings, self.encoder.identity)

    def search_text(self, query: str, **filters: Any) -> list[SemanticSearchHit]:
        if self.encoder is None:
            raise RuntimeError(self._error or "semantic search is unavailable")
        text = str(query).strip()
        if not text:
            raise ValueError("semantic search query cannot be empty")
        embedding = self.encoder.encode_text([text])
        return self.index.search(embedding, self.encoder.identity, **filters)

    def status(self) -> dict[str, Any]:
        identity = self.encoder.identity if self.encoder else None
        return {
            "enabled": True, "state": "ready" if identity else "unavailable",
            "implementation": self.config.implementation, "device": self.config.device,
            "generation": identity.generation if identity else "", "error": self._error,
            "queue_depth": self._queue.qsize(), "indexed_since_start": self._indexed,
            **self.index.coverage(identity),
        }

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._backfill_thread:
            self._backfill_thread.join(timeout=5.0)
        if self.encoder:
            self.encoder.close()
        self.encoder = None

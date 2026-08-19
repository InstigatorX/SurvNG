from __future__ import annotations

import json
import logging
import math
from queue import Empty, Full, Queue
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .face_recognition import OpenVinoFaceRecognizer
from .inference import INFERENCE_REQUEST_TIMEOUT_SECONDS, InferenceUnavailable
from .incident_utils import event_snapshot_path, portable_media_path
from .media_storage import MediaStorageRegistry
from .visual_quality import image_quality
from .unknown_identity import cluster_unknown_embeddings, unknown_cluster_cohesion
from .unknown_identity import DEFAULT_UNKNOWN_CLUSTER_THRESHOLD


LOGGER = logging.getLogger(__name__)
FACE_QUALITY_VERSION = 2
FACE_OUTCOME_PENDING = "pending"
FACE_OUTCOME_EMBEDDED = "embedded"
FACE_OUTCOME_TOO_SMALL = "too_small"
FACE_OUTCOME_FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FaceQuality:
    score: float
    sharpness: float
    exposure: float
    contrast: float
    size: float
    edge_detail: float


@dataclass(frozen=True, slots=True)
class FaceMatch:
    person_id: int | None
    score: float | None
    runner_up_score: float | None
    margin: float | None
    reference_ids: tuple[int, ...]
    reference_scores: tuple[float, ...]


class FaceTooSmallError(ValueError):
    """The detected crop cannot produce a reliable face embedding."""


def _face_crop(
    frame: np.ndarray,
    box: dict[str, float],
    *,
    padding: float = 0.12,
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1 = float(box["x1"]), float(box["y1"])
    x2, y2 = float(box["x2"]), float(box["y2"])
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def _face_quality(face: np.ndarray, detector_confidence: float) -> FaceQuality:
    if face.size == 0:
        return FaceQuality(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    height, width = face.shape[:2]
    visual = image_quality(face, max_dimension=256)
    size = max(0.0, min(1.0, min(height, width) / 160.0))
    confidence = max(0.0, min(1.0, float(detector_confidence)))
    score = (
        0.35 * visual.sharpness
        + 0.20 * visual.exposure
        + 0.15 * visual.contrast
        + 0.20 * size
        + 0.10 * confidence
    )
    return FaceQuality(
        round(score, 4),
        round(visual.sharpness, 4),
        round(visual.exposure, 4),
        round(visual.contrast, 4),
        round(size, 4),
        round(visual.edge_detail, 4),
    )


def parse_face_box(value: object) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        box = {name: float(value[name]) for name in ("x1", "y1", "x2", "y2")}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(coordinate) for coordinate in box.values()):
        return None
    if box["x1"] < 0 or box["y1"] < 0:
        return None
    if box["x2"] <= box["x1"] or box["y2"] <= box["y1"]:
        return None
    return box


class FaceStore:
    def __init__(
        self,
        storage_dir: Path,
        max_observations: int = 1000,
        recognizer: OpenVinoFaceRecognizer | None = None,
        start_recognition: bool = True,
        database_dir: Path | None = None,
        media_storage: MediaStorageRegistry | None = None,
    ) -> None:
        self.storage_dir = storage_dir.resolve()
        self.media_storage = media_storage
        resolved_database_dir = (database_dir or self.storage_dir).resolve()
        resolved_database_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = resolved_database_dir / "survng.sqlite3"
        self.max_observations = max(100, int(max_observations))
        self.recognizer = recognizer
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._recognition_queue: Queue[int | None] = Queue(maxsize=self.max_observations + 1)
        self._recognition_pending: set[int] = set()
        self._recognition_pending_lock = threading.Lock()
        self._gallery_lock = threading.RLock()
        self._gallery_generation = 0
        self._gallery_cache_key: tuple[str, tuple[int, ...], int, int] | None = None
        self._gallery_cache: list[dict[str, Any]] = []
        self._recognition_refill_needed = threading.Event()
        self._match_refresh_needed = threading.Event()
        self._recognition_stop = threading.Event()
        self._recognition_thread: threading.Thread | None = None
        self._init_db()
        if start_recognition:
            self.start()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._recognition_thread is not None and self._recognition_thread.is_alive():
                return
            if self.recognizer is None or not self.recognizer.enabled:
                return
            self._recognition_stop.clear()
            self._recognition_thread = threading.Thread(
                target=self._recognition_loop,
                name="survng-face-recognition",
                daemon=False,
            )
            self._recognition_thread.start()
        try:
            self._queue_pending_recognition()
        except Exception:
            LOGGER.exception("Could not restore pending face recognition work")

    def reconfigure_max_observations(self, max_observations: int) -> int:
        """Apply a new history limit and prune transactionally without stopping recognition."""
        next_limit = max(100, int(max_observations))
        with self._lock, self._connect() as connection:
            previous_limit = self.max_observations
            self.max_observations = next_limit
            try:
                return self._prune_locked(connection)
            except BaseException:
                self.max_observations = previous_limit
                raise

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma busy_timeout = 10000")
        connection.execute("pragma foreign_keys = on")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute("pragma journal_mode = wal")
            connection.execute("pragma synchronous = normal")
            connection.execute(
                "create table if not exists media_deletion_claims ("
                "path text primary key, role text not null, claimed_at text not null)"
            )
            connection.execute(
                "create table if not exists survng_metadata (key text primary key, value text not null)"
            )
            connection.executescript(
                """
                create table if not exists face_people (
                    id integer primary key autoincrement,
                    name text not null,
                    notes text not null default '',
                    created_at text not null,
                    updated_at text not null
                );
                create table if not exists face_observations (
                    id integer primary key autoincrement,
                    event_id integer not null,
                    object_index integer not null,
                    person_id integer,
                    camera_id text not null,
                    snapshot_path text not null,
                    box_json text not null,
                    confidence real not null default 0,
                    observed_at text not null,
                    match_confidence real,
                    review_status text not null default 'unknown',
                    created_at text not null,
                    unique(event_id, object_index),
                    foreign key(person_id) references face_people(id) on delete set null
                );
                create index if not exists idx_face_observations_observed_at
                    on face_observations(observed_at desc);
                create index if not exists idx_face_observations_person
                    on face_observations(person_id, observed_at desc);
                create table if not exists face_unknown_members (
                    observation_id integer primary key,
                    cluster_id integer not null,
                    updated_at text not null,
                    foreign key(observation_id) references face_observations(id) on delete cascade
                );
                create index if not exists idx_face_unknown_members_cluster
                    on face_unknown_members(cluster_id, observation_id);
                create table if not exists face_rejections (
                    observation_id integer not null,
                    person_id integer not null,
                    created_at text not null,
                    primary key(observation_id, person_id),
                    foreign key(observation_id) references face_observations(id) on delete cascade,
                    foreign key(person_id) references face_people(id) on delete cascade
                );
                """
            )
            columns = {str(row[1]) for row in connection.execute("pragma table_info(face_observations)")}
            migrations = {
                "embedding_blob": "alter table face_observations add column embedding_blob blob",
                "embedding_model": "alter table face_observations add column embedding_model text not null default ''",
                "candidate_person_id": "alter table face_observations add column candidate_person_id integer",
                "candidate_confidence": "alter table face_observations add column candidate_confidence real",
                "rejected_person_id": "alter table face_observations add column rejected_person_id integer",
                "recognition_error": "alter table face_observations add column recognition_error text not null default ''",
                "recognized_at": "alter table face_observations add column recognized_at text not null default ''",
                "recognition_pending": "alter table face_observations add column recognition_pending integer not null default 1",
                "recognition_outcome": "alter table face_observations add column recognition_outcome text not null default 'pending'",
                "quality_score": "alter table face_observations add column quality_score real",
                "quality_json": "alter table face_observations add column quality_json text not null default '{}'",
                "quality_version": "alter table face_observations add column quality_version integer not null default 0",
                "reference_pinned": "alter table face_observations add column reference_pinned integer not null default 0",
                "reference_auto_pinned": "alter table face_observations add column reference_auto_pinned integer not null default 0",
                "auto_identified": "alter table face_observations add column auto_identified integer not null default 0",
                "match_details_json": "alter table face_observations add column match_details_json text not null default '{}'",
                "candidate_track_id": "alter table face_observations add column candidate_track_id text not null default ''",
                "candidate_rank": "alter table face_observations add column candidate_rank integer not null default 1",
                "candidate_offset_seconds": "alter table face_observations add column candidate_offset_seconds real not null default 0",
                "canonical": "alter table face_observations add column canonical integer not null default 1",
                "duplicate_of_observation_id": "alter table face_observations add column duplicate_of_observation_id integer",
                "consensus_json": "alter table face_observations add column consensus_json text not null default '{}'",
            }
            for name, statement in migrations.items():
                if name not in columns:
                    connection.execute(statement)
            connection.execute(
                "create index if not exists idx_face_duplicate_of "
                "on face_observations(duplicate_of_observation_id)"
            )
            connection.execute(
                """
                update face_observations
                set recognition_outcome = case
                    when recognition_pending = 1 then ?
                    when recognition_error like 'Face is smaller than %' then ?
                    when recognition_error != '' then ?
                    when embedding_blob is not null then ?
                    else ?
                end
                where recognition_outcome = ''
                    or recognition_outcome not in (?, ?, ?, ?)
                    or (recognition_outcome = ? and recognition_pending = 0)
                """,
                (
                    FACE_OUTCOME_PENDING,
                    FACE_OUTCOME_TOO_SMALL,
                    FACE_OUTCOME_FAILED,
                    FACE_OUTCOME_EMBEDDED,
                    FACE_OUTCOME_FAILED,
                    FACE_OUTCOME_PENDING,
                    FACE_OUTCOME_EMBEDDED,
                    FACE_OUTCOME_TOO_SMALL,
                    FACE_OUTCOME_FAILED,
                    FACE_OUTCOME_PENDING,
                ),
            )
            connection.execute(
                """
                create unique index if not exists idx_face_candidate_identity
                on face_observations(event_id, candidate_track_id, candidate_offset_seconds)
                where candidate_track_id != ''
                """
            )
            connection.execute(
                """
                update face_observations
                set recognition_pending = 1, recognition_outcome = ?
                where quality_version != ? and embedding_blob is not null
                    and recognition_error = ''
                """,
                (FACE_OUTCOME_PENDING, FACE_QUALITY_VERSION),
            )
            connection.execute(
                """
                insert or ignore into face_rejections (observation_id, person_id, created_at)
                select o.id, o.rejected_person_id, coalesce(nullif(o.recognized_at, ''), o.created_at)
                from face_observations o
                join face_people p on p.id = o.rejected_person_id
                where o.rejected_person_id is not null
                """
            )
            connection.execute(
                """
                update face_observations
                set candidate_person_id = null, candidate_confidence = null
                where candidate_person_id is not null
                    and not exists (
                        select 1 from face_people p where p.id = candidate_person_id
                    )
                """
            )
            connection.execute(
                """
                update face_observations
                set rejected_person_id = null
                where rejected_person_id is not null
                    and not exists (
                        select 1 from face_people p where p.id = rejected_person_id
                    )
                """
            )
            connection.execute(
                """
                update face_observations
                set review_status = 'unknown', match_confidence = null
                where person_id is null and review_status = 'confirmed'
                """
            )
            storage_root = str(self.storage_dir.resolve())
            metadata = connection.execute(
                "select value from survng_metadata where key = 'face_storage_root'"
            ).fetchone()
            if metadata is None or str(metadata[0]) != storage_root:
                rows = connection.execute("select id, snapshot_path from face_observations").fetchall()
                updates = []
                for row in rows:
                    raw_path = str(row["snapshot_path"] or "")
                    portable_path = portable_media_path(self.storage_dir, raw_path)
                    if portable_path != raw_path:
                        updates.append((portable_path, int(row["id"])))
                if updates:
                    connection.executemany(
                        "update face_observations set snapshot_path = ? where id = ?",
                        updates,
                    )
                connection.execute(
                    """
                    insert into survng_metadata (key, value) values ('face_storage_root', ?)
                    on conflict(key) do update set value = excluded.value
                    """,
                    (storage_root,),
                )
            self._prune_locked(connection)

    def ingest_candidates(
        self,
        event_id: int,
        camera_id: str,
        observed_at: str,
        candidates: list[dict[str, Any]],
    ) -> int:
        """Persist bounded face crops produced by temporal object analysis."""
        if event_id <= 0:
            return 0
        inserted = 0
        recognition_ids: list[int] = []
        touched_tracks: set[str] = set()
        discarded_paths: list[Path] = []
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            next_index = int(connection.execute(
                "select coalesce(max(object_index), -1) + 1 from face_observations where event_id = ?",
                (event_id,),
            ).fetchone()[0])
            for candidate in candidates:
                box = parse_face_box(candidate.get("box"))
                track_id = str(candidate.get("track_id") or "").strip()
                if box is None or not track_id:
                    continue
                try:
                    confidence = float(candidate.get("confidence") or 0.0)
                    rank = max(1, int(candidate.get("rank") or 1))
                    offset_seconds = float(candidate.get("offset_seconds") or 0.0)
                    quality_score = float(candidate.get("quality_score") or 0.0)
                    resolved_snapshot = event_snapshot_path(
                        self.storage_dir,
                        {"snapshot_path": str(candidate.get("snapshot_path") or "")},
                        self.media_storage,
                    )
                    snapshot_path = portable_media_path(self.storage_dir, resolved_snapshot)
                except (FileNotFoundError, PermissionError, OSError, RuntimeError, TypeError, ValueError):
                    continue
                if not all(math.isfinite(value) for value in (confidence, offset_seconds, quality_score)):
                    continue
                quality_payload = dict(candidate.get("quality") or {})
                quality_payload["collector_score"] = max(0.0, min(1.0, quality_score))
                cursor = connection.execute(
                    """
                    insert or ignore into face_observations (
                        event_id, object_index, camera_id, snapshot_path, box_json,
                        confidence, observed_at, created_at, candidate_track_id,
                        candidate_rank, candidate_offset_seconds, canonical,
                        quality_score, quality_json
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id, next_index, camera_id, snapshot_path,
                        json.dumps(box, separators=(",", ":")),
                        max(0.0, min(1.0, confidence)), observed_at, now,
                        track_id, rank, offset_seconds, int(rank == 1),
                        max(0.0, min(1.0, quality_score)),
                        json.dumps(quality_payload, separators=(",", ":")),
                    ),
                )
                next_index += 1
                if cursor.rowcount > 0:
                    inserted += 1
                    recognition_ids.append(int(cursor.lastrowid))
                    touched_tracks.add(track_id)
                else:
                    try:
                        resolved_snapshot.unlink(missing_ok=True)
                    except OSError:
                        LOGGER.debug("could not remove duplicate face candidate %s", resolved_snapshot)
            for track_id in touched_tracks:
                protected_count = int(connection.execute(
                    """
                    select count(*) from face_observations
                    where event_id = ? and candidate_track_id = ?
                        and (person_id is not null or review_status != 'unknown' or reference_pinned = 1)
                    """,
                    (event_id, track_id),
                ).fetchone()[0])
                unreviewed_limit = max(0, 4 - protected_count)
                excess_rows = connection.execute(
                    """
                    select id, snapshot_path from face_observations
                    where event_id = ? and candidate_track_id = ?
                        and person_id is null and review_status = 'unknown'
                        and reference_pinned = 0
                    order by quality_score desc, candidate_rank asc, id asc
                    limit -1 offset ?
                    """,
                    (event_id, track_id, unreviewed_limit),
                ).fetchall()
                if excess_rows:
                    connection.executemany(
                        "delete from face_observations where id = ?",
                        ((int(row["id"]),) for row in excess_rows),
                    )
                    for row in excess_rows:
                        try:
                            discarded_paths.append(event_snapshot_path(
                                self.storage_dir,
                                {"snapshot_path": str(row["snapshot_path"] or "")},
                                self.media_storage,
                            ))
                        except (FileNotFoundError, PermissionError, OSError, RuntimeError):
                            continue
                self._reconcile_candidate_track(connection, event_id, track_id)
            self._prune_locked(connection)
            if recognition_ids:
                retained_ids = {
                    int(row["id"])
                    for row in connection.execute(
                        f"select id from face_observations where id in ({','.join('?' for _ in recognition_ids)})",
                        recognition_ids,
                    ).fetchall()
                }
                recognition_ids = [
                    observation_id
                    for observation_id in recognition_ids
                    if observation_id in retained_ids
                ]
        for discarded_path in discarded_paths:
            try:
                discarded_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.debug("could not remove superseded face candidate %s", discarded_path)
        for observation_id in recognition_ids:
            self._queue_recognition(observation_id)
        return inserted

    def _invalidate_reference_gallery(self) -> None:
        with self._gallery_lock:
            self._gallery_generation += 1
            self._gallery_cache_key = None
            self._gallery_cache = []

    def ingest_events(self, events: list[dict[str, Any]]) -> int:
        inserted = 0
        recognition_ids: list[int] = []
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            for event in events:
                try:
                    event_id = int(event["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if event_id <= 0:
                    continue
                try:
                    snapshot_path = portable_media_path(
                        self.storage_dir,
                        event_snapshot_path(self.storage_dir, event, self.media_storage),
                    )
                except (FileNotFoundError, PermissionError, OSError, RuntimeError):
                    continue
                try:
                    objects = json.loads(event.get("objects_json", "[]") or "[]")
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(objects, list):
                    continue
                if any(
                    isinstance(item, dict)
                    and item.get("status") == "face_evidence_pending"
                    for item in objects
                ):
                    continue
                for object_index, detected in enumerate(objects):
                    if not isinstance(detected, dict):
                        continue
                    if str(detected.get("label") or "").lower() != "face":
                        continue
                    if connection.execute(
                        "select 1 from face_observations where event_id = ? and candidate_track_id != '' limit 1",
                        (event_id,),
                    ).fetchone() is not None:
                        continue
                    box = parse_face_box(detected.get("box"))
                    if box is None:
                        continue
                    try:
                        confidence = float(detected.get("confidence") or 0)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                        continue
                    cursor = connection.execute(
                        """
                        insert or ignore into face_observations (
                            event_id, object_index, camera_id, snapshot_path,
                            box_json, confidence, observed_at, created_at
                        ) values (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event_id, object_index, str(event.get("camera_id") or ""),
                            snapshot_path, json.dumps(box, separators=(",", ":")),
                            confidence,
                            str(event.get("created_at") or now), now,
                        ),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                        recognition_ids.append(int(cursor.lastrowid))
            self._prune_locked(connection)
            if recognition_ids:
                inserted_ids = set(recognition_ids)
                recognition_ids = [
                    int(row["id"])
                    for row in connection.execute(
                        "select id from face_observations order by observed_at desc, id desc limit ?",
                        (self.max_observations,),
                    ).fetchall()
                    if int(row["id"]) in inserted_ids
                ]
        for observation_id in recognition_ids:
            self._queue_recognition(observation_id)
        return inserted

    def close(self) -> None:
        with self._lifecycle_lock:
            self._recognition_stop.set()
            thread = self._recognition_thread
            if thread is not None:
                try:
                    self._recognition_queue.put_nowait(None)
                except Full:
                    pass
        if thread is not None:
            thread.join(timeout=INFERENCE_REQUEST_TIMEOUT_SECONDS + 5.0)
            if thread.is_alive():
                LOGGER.error("face recognition worker did not stop")
                raise RuntimeError("face recognition worker did not stop")
            with self._lifecycle_lock:
                if self._recognition_thread is thread:
                    self._recognition_thread = None

    def recognition_status(self) -> dict[str, Any]:
        recognizer_status = self.recognizer.status() if self.recognizer is not None else {
            "enabled": False,
            "ready": False,
            "error": "Face recognition is not configured.",
        }
        with self._connect() as connection:
            row = connection.execute(
                """
                select sum(case when embedding_model = ? and embedding_blob is not null then 1 else 0 end) as embedded_current,
                    sum(case when canonical = 1 and candidate_person_id is not null
                        and person_id is null and recognition_pending = 0
                        and recognition_outcome = ? then 1 else 0 end) as suggested,
                    sum(case when recognition_outcome = ? then 1 else 0 end) as too_small,
                    sum(case when recognition_outcome = ? then 1 else 0 end) as failed,
                    sum(case when recognition_pending = 1 then 1 else 0 end) as pending
                from face_observations
                """,
                (
                    str(recognizer_status.get("model_fingerprint") or ""),
                    FACE_OUTCOME_EMBEDDED,
                    FACE_OUTCOME_TOO_SMALL,
                    FACE_OUTCOME_FAILED,
                ),
            ).fetchone()
        return {
            **recognizer_status,
            "queue_depth": self._recognition_queue.qsize(),
            "embedded": int(row["embedded_current"] or 0),
            "suggested": int(row["suggested"] or 0),
            "too_small": int(row["too_small"] or 0),
            "failed": int(row["failed"] or 0),
            "pending": int(row["pending"] or 0),
        }

    def _queue_recognition(self, observation_id: int) -> None:
        if self.recognizer is None or self._recognition_stop.is_set():
            return
        observation_id = int(observation_id)
        with self._recognition_pending_lock:
            if observation_id in self._recognition_pending:
                return
            try:
                self._recognition_queue.put_nowait(observation_id)
            except Full:
                LOGGER.warning("face recognition queue is full; deferred observation %s", observation_id)
                self._recognition_refill_needed.set()
                return
            self._recognition_pending.add(observation_id)

    def _queue_pending_recognition(self) -> None:
        if self.recognizer is None or not self.recognizer.enabled:
            return
        recognizer_status = self.recognizer.status()
        model_fingerprint = str(recognizer_status.get("model_fingerprint") or "")
        with self._connect() as connection:
            if model_fingerprint:
                connection.execute(
                    """
                    update face_observations
                    set recognition_pending = 1, recognition_outcome = ?
                    where embedding_blob is not null and embedding_model != ?
                    """,
                    (FACE_OUTCOME_PENDING, model_fingerprint),
                )
            rows = connection.execute(
                """
                select id from face_observations
                where recognition_pending = 1
                    or (embedding_blob is not null and ? != '' and embedding_model != ?)
                order by case when person_id is not null then 0 else 1 end,
                    observed_at desc limit ?
                """,
                (model_fingerprint, model_fingerprint, self.max_observations),
            ).fetchall()
        for row in rows:
            self._queue_recognition(int(row["id"]))

    def _refresh_unknown_recognition(self) -> None:
        if self.recognizer is None or not self.recognizer.enabled:
            return
        recognizer_status = self.recognizer.status()
        model_fingerprint = str(recognizer_status.get("model_fingerprint") or "")
        if not model_fingerprint:
            return
        with self._lock, self._connect() as connection:
            embedded_rows = connection.execute(
                """
                select id, embedding_blob from face_observations
                where person_id is null and recognition_pending = 0
                    and embedding_model = ? and embedding_blob is not null
                order by observed_at desc limit ?
                """,
                (model_fingerprint, self.max_observations),
            ).fetchall()
            for row in embedded_rows:
                try:
                    embedding = np.frombuffer(row["embedding_blob"], dtype=np.float32)
                except (TypeError, ValueError):
                    continue
                norm = float(np.linalg.norm(embedding))
                if (
                    embedding.size == 0
                    or not np.all(np.isfinite(embedding))
                    or not math.isfinite(norm)
                    or norm <= 1e-9
                ):
                    continue
                match = self._match_result(
                    connection,
                    int(row["id"]),
                    embedding / norm,
                    model_fingerprint,
                )
                connection.execute(
                    """
                    update face_observations
                    set candidate_person_id = ?, candidate_confidence = ?,
                        match_details_json = ?
                    where id = ? and person_id is null
                    """,
                    (
                        match.person_id,
                        match.score,
                        json.dumps(
                            {
                                "score": match.score,
                                "runner_up_score": match.runner_up_score,
                                "margin": match.margin,
                                "reference_ids": list(match.reference_ids),
                                "reference_scores": list(match.reference_scores),
                            },
                            separators=(",", ":"),
                        ),
                        int(row["id"]),
                    ),
                )
            connection.execute(
                """
                update face_observations
                set recognition_pending = 1, recognition_outcome = ?
                where person_id is null and embedding_blob is not null
                    and embedding_model != ?
                """,
                (FACE_OUTCOME_PENDING, model_fingerprint),
            )
            pending_rows = connection.execute(
                """
                select id from face_observations
                where person_id is null and (
                    recognition_pending = 1
                    or (embedding_blob is not null and embedding_model != ?)
                )
                order by observed_at desc limit ?
                """,
                (model_fingerprint, self.max_observations),
            ).fetchall()
        for row in pending_rows:
            self._queue_recognition(int(row["id"]))

    def _try_refresh_unknown_recognition(self) -> bool:
        try:
            self._refresh_unknown_recognition()
            return True
        except Exception:
            LOGGER.exception("Could not refresh unknown face matches")
            return False

    def request_match_refresh(self) -> None:
        """Refresh saved suggestions asynchronously when the worker is active."""
        thread = self._recognition_thread
        if thread is not None and thread.is_alive() and not self._recognition_stop.is_set():
            self._match_refresh_needed.set()
            return
        self._try_refresh_unknown_recognition()

    def _recognition_loop(self) -> None:
        references_changed = False
        while True:
            try:
                observation_id = self._recognition_queue.get(timeout=1)
            except Empty:
                if self._recognition_stop.is_set():
                    break
                if self._recognition_refill_needed.is_set():
                    self._recognition_refill_needed.clear()
                    self._queue_pending_recognition()
                if references_changed or self._match_refresh_needed.is_set():
                    self._match_refresh_needed.clear()
                    if self._try_refresh_unknown_recognition():
                        references_changed = False
                    else:
                        self._match_refresh_needed.set()
                continue
            if observation_id is None:
                self._recognition_queue.task_done()
                break
            if self._recognition_stop.is_set():
                self._recognition_queue.task_done()
                with self._recognition_pending_lock:
                    self._recognition_pending.discard(observation_id)
                continue
            retry = False
            try:
                references_changed = (
                    self._recognize_observation(observation_id) or references_changed
                )
            except InferenceUnavailable as exc:
                if not self._recognition_stop.is_set():
                    retry = True
                    LOGGER.warning("Face recognition deferred while inference recovers: %s", exc)
            except Exception:
                LOGGER.exception("Face recognition failed for observation %s", observation_id)
            finally:
                self._recognition_queue.task_done()
                with self._recognition_pending_lock:
                    self._recognition_pending.discard(observation_id)
            if retry and not self._recognition_stop.wait(1.0):
                self._queue_recognition(observation_id)
            if self._recognition_refill_needed.is_set() and not self._recognition_stop.is_set():
                self._recognition_refill_needed.clear()
                self._queue_pending_recognition()

    def _recognize_observation(self, observation_id: int) -> bool:
        recognizer = self.recognizer
        if recognizer is None:
            return False
        recognizer_status = recognizer.status()
        if not recognizer_status.get("ready"):
            isolation = recognizer_status.get("isolation") or {}
            if recognizer.enabled:
                raise InferenceUnavailable(
                    str(isolation.get("last_error") or "face inference is still starting")
                )
            return False
        model_fingerprint = str(recognizer_status.get("model_fingerprint") or "")
        if not model_fingerprint:
            return False
        with self._connect() as connection:
            row = connection.execute(
                "select * from face_observations where id = ?", (observation_id,)
            ).fetchone()
        if row is None:
            return False
        try:
            box = parse_face_box(json.loads(row["box_json"] or "{}"))
            if box is None:
                raise ValueError("Face box is invalid.")
            snapshot_path = event_snapshot_path(
                self.storage_dir,
                {"snapshot_path": str(row["snapshot_path"] or "")},
                self.media_storage,
            )
            frame = cv2.imread(str(snapshot_path))
            if frame is None:
                raise ValueError("Snapshot is unavailable.")
            x1, y1 = float(box.get("x1", 0)), float(box.get("y1", 0))
            x2, y2 = float(box.get("x2", 0)), float(box.get("y2", 0))
            face_width, face_height = x2 - x1, y2 - y1
            if min(face_width, face_height) < recognizer.config.face_min_size:
                raise FaceTooSmallError(
                    f"Face is smaller than {recognizer.config.face_min_size}px."
                )
            face = _face_crop(frame, box)
            if face is None:
                raise ValueError("Face crop is invalid.")
            quality = _face_quality(face, float(row["confidence"] or 0.0))
            embedding = np.asarray(
                recognizer.embed(face),
                dtype=np.float32,
            ).reshape(-1)
            expected_size = int(recognizer_status.get("embedding_size") or 0)
            if embedding.size == 0 or embedding.size > 16384:
                raise ValueError("Face embedding size was invalid.")
            if expected_size and embedding.size != expected_size:
                raise ValueError(
                    f"Face embedding had {embedding.size} values; expected {expected_size}."
                )
            norm = float(np.linalg.norm(embedding))
            if not math.isfinite(norm) or norm <= 1e-9 or not np.all(np.isfinite(embedding)):
                raise ValueError("Face embedding was empty or invalid.")
            embedding = embedding / norm
            now = datetime.now(timezone.utc).isoformat()
            with self._lock, self._connect() as connection:
                match = self._match_result(
                    connection,
                    observation_id,
                    embedding,
                    model_fingerprint,
                )
                candidate_id = match.person_id
                candidate_confidence = match.score
                auto_identified = bool(
                    getattr(recognizer.config, "face_auto_identify_enabled", False)
                    and candidate_id is not None
                    and candidate_confidence is not None
                    and candidate_confidence
                    >= getattr(recognizer.config, "face_auto_identify_threshold", 1.0)
                    and match.runner_up_score is not None
                    and match.margin is not None
                    and match.margin
                    >= getattr(recognizer.config, "face_auto_identify_margin", 1.0)
                    and len(match.reference_ids) >= 3
                    and quality.score >= 0.45
                    and not str(row["candidate_track_id"] or "")
                )
                details = json.dumps(
                    {
                        "score": match.score,
                        "runner_up_score": match.runner_up_score,
                        "margin": match.margin,
                        "reference_ids": list(match.reference_ids),
                        "reference_scores": list(match.reference_scores),
                        "quality_score": quality.score,
                    },
                    separators=(",", ":"),
                )
                quality_payload = json.dumps(
                    {
                        "sharpness": quality.sharpness,
                        "exposure": quality.exposure,
                        "contrast": quality.contrast,
                        "size": quality.size,
                        "edge_detail": quality.edge_detail,
                    },
                    separators=(",", ":"),
                )
                connection.execute(
                    """
                    update face_observations
                        set embedding_blob = ?, embedding_model = ?,
                        person_id = case when person_id is null and ? then ? else person_id end,
                        review_status = case when person_id is null and ? then 'auto_identified' else review_status end,
                        match_confidence = case when person_id is null and ? then ? else match_confidence end,
                        auto_identified = case when person_id is null and ? then 1 else auto_identified end,
                        candidate_person_id = case when person_id is null and not ? then ? else null end,
                        candidate_confidence = case when person_id is null and not ? then ? else null end,
                        quality_score = ?, quality_json = ?, quality_version = ?,
                        match_details_json = ?,
                        recognition_error = '', recognized_at = ?, recognition_pending = 0,
                        recognition_outcome = ?
                    where id = ?
                    """,
                    (
                        embedding.astype(np.float32).tobytes(),
                        model_fingerprint,
                        auto_identified,
                        candidate_id,
                        auto_identified,
                        auto_identified,
                        candidate_confidence,
                        auto_identified,
                        auto_identified,
                        candidate_id,
                        auto_identified,
                        candidate_confidence,
                        quality.score,
                        quality_payload,
                        FACE_QUALITY_VERSION,
                        details,
                        now,
                        FACE_OUTCOME_EMBEDDED,
                        observation_id,
                    ),
                )
                current = connection.execute(
                    "select person_id from face_observations where id = ?",
                    (observation_id,),
                ).fetchone()
                track_id = str(row["candidate_track_id"] or "")
                if track_id:
                    self._reconcile_candidate_track(
                        connection,
                        int(row["event_id"]),
                        track_id,
                    )
            if row["person_id"] is not None:
                self._invalidate_reference_gallery()
            return current is not None and current["person_id"] is not None
        except InferenceUnavailable:
            raise
        except Exception as exc:
            outcome = (
                FACE_OUTCOME_TOO_SMALL
                if isinstance(exc, FaceTooSmallError)
                else FACE_OUTCOME_FAILED
            )
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    update face_observations
                    set recognition_error = ?, recognized_at = ?, recognition_pending = 0,
                        recognition_outcome = ?
                    where id = ?
                    """,
                    (
                        str(exc)[:500],
                        datetime.now(timezone.utc).isoformat(),
                        outcome,
                        observation_id,
                    ),
                )
            return False


    def _mark_exact_embedding_duplicate_locked(
        self,
        connection: sqlite3.Connection,
        observation_id: int,
        *,
        window_seconds: float = 60.0,
    ) -> int | None:
        row = connection.execute(
            """
            select id, camera_id, observed_at, embedding_blob, embedding_model,
                person_id, review_status, reference_pinned
            from face_observations
            where id = ?
            """,
            (observation_id,),
        ).fetchone()
        if (
            row is None
            or row["embedding_blob"] is None
            or row["person_id"] is not None
            or str(row["review_status"] or "") == "confirmed"
            or bool(row["reference_pinned"])
        ):
            return None

        duplicate = connection.execute(
            """
            select id from face_observations
            where id != ?
                and canonical = 1
                and camera_id = ?
                and embedding_model = ?
                and embedding_blob = ?
                and person_id is null
                and review_status != 'confirmed'
                and reference_pinned = 0
                and abs((julianday(observed_at) - julianday(?)) * 86400.0) <= ?
            order by observed_at asc, id asc
            limit 1
            """,
            (
                observation_id,
                str(row["camera_id"] or ""),
                str(row["embedding_model"] or ""),
                row["embedding_blob"],
                str(row["observed_at"] or ""),
                max(0.0, float(window_seconds)),
            ),
        ).fetchone()
        if duplicate is None:
            return None

        duplicate_id = int(duplicate["id"])
        connection.execute(
            """
            update face_observations
            set canonical = 0,
                duplicate_of_observation_id = ?,
                candidate_person_id = null,
                candidate_confidence = null
            where id = ?
                and person_id is null
                and review_status != 'confirmed'
                and reference_pinned = 0
            """,
            (duplicate_id, observation_id),
        )
        return duplicate_id

    def dedupe_exact_embeddings(
        self,
        *,
        window_seconds: float = 60.0,
    ) -> dict[str, Any]:
        window = max(0.0, min(float(window_seconds), 3600.0))
        marked = 0
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                select id from face_observations
                where canonical = 1
                    and embedding_blob is not null
                    and person_id is null
                    and review_status != 'confirmed'
                    and reference_pinned = 0
                order by observed_at asc, id asc
                """
            ).fetchall()
            for row in rows:
                marked += int(
                    self._mark_exact_embedding_duplicate_locked(
                        connection,
                        int(row["id"]),
                        window_seconds=window,
                    )
                    is not None
                )
        if marked:
            self.request_match_refresh()
        return {
            "marked_duplicates": marked,
            "window_seconds": window,
            **self.duplicate_stats(),
        }

    def duplicate_stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            summary = connection.execute(
                """
                select count(*) as total_rows,
                    sum(case when duplicate_of_observation_id is not null then 1 else 0 end)
                        as duplicate_rows,
                    sum(case when canonical = 1 then 1 else 0 end) as canonical_rows
                from face_observations
                """
            ).fetchone()
            groups = connection.execute(
                """
                select original.id as observation_id,
                    original.camera_id,
                    original.observed_at,
                    count(duplicate.id) as duplicate_count
                from face_observations original
                join face_observations duplicate
                    on duplicate.duplicate_of_observation_id = original.id
                group by original.id
                order by duplicate_count desc, original.id
                limit 20
                """
            ).fetchall()
        return {
            "total_rows": int(summary["total_rows"] or 0),
            "canonical_rows": int(summary["canonical_rows"] or 0),
            "duplicate_rows": int(summary["duplicate_rows"] or 0),
            "top_duplicate_groups": [dict(row) for row in groups],
        }

    def unknown_cluster_members(
        self,
        cluster_id: int,
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select o.*, p.name as person_name,
                    candidate.name as candidate_person_name,
                    m.cluster_id as unknown_cluster_id
                from face_unknown_members m
                join face_observations o on o.id = m.observation_id
                left join face_people p on p.id = o.person_id
                left join face_people candidate on candidate.id = o.candidate_person_id
                where m.cluster_id = ?
                    and o.canonical = 1
                order by o.observed_at desc, o.id desc
                limit ?
                """,
                (int(cluster_id), max(1, min(int(limit), 500))),
            ).fetchall()
        return [self._observation_row(row) for row in rows]

    def confirmed_quality_issues(
        self,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select o.*, p.name as person_name,
                    candidate.name as candidate_person_name
                from face_observations o
                join face_people p on p.id = o.person_id
                left join face_people candidate on candidate.id = o.candidate_person_id
                where o.canonical = 1
                    and o.person_id is not null
                    and o.review_status = 'confirmed'
                order by coalesce(o.quality_score, 0) asc,
                    coalesce(o.match_confidence, 0) asc,
                    o.observed_at desc
                limit ?
                """,
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        result = []
        for row in rows:
            item = self._observation_row(row)
            flags = []
            if float(item.get("quality_score") or 0.0) < 0.45:
                flags.append("low_quality")
            if float(item.get("match_confidence") or 0.0) < 0.30:
                flags.append("weak_match")
            if item.get("reference_pinned"):
                flags.append("reference")
            item["diagnostic_flags"] = flags
            result.append(item)
        return result

    def _reconcile_candidate_track(
        self,
        connection: sqlite3.Connection,
        event_id: int,
        track_id: str,
    ) -> None:
        rows = connection.execute(
            """
            select id, person_id, candidate_person_id, candidate_confidence,
                match_confidence, review_status,
                quality_score, recognition_pending, recognition_error,
                match_details_json
            from face_observations
            where event_id = ? and candidate_track_id = ?
            order by candidate_rank, id
            """,
            (event_id, track_id),
        ).fetchall()
        if not rows:
            return
        completed = [
            row for row in rows
            if not bool(row["recognition_pending"]) and not str(row["recognition_error"] or "")
        ]
        votes: dict[int, list[sqlite3.Row]] = {}
        for row in completed:
            person_id = row["person_id"] or row["candidate_person_id"]
            confidence = row["match_confidence"] if row["person_id"] is not None else row["candidate_confidence"]
            if person_id is not None and confidence is not None:
                votes.setdefault(int(person_id), []).append(row)
        winner_id: int | None = None
        support: list[sqlite3.Row] = []
        if votes:
            winner_id, support = max(
                votes.items(),
                key=lambda item: (
                    len(item[1]),
                    sum(self._row_identity_confidence(row) for row in item[1]) / len(item[1]),
                ),
            )
        consensus_score = (
            sum(self._row_identity_confidence(row) for row in support) / len(support)
            if support else None
        )
        canonical = max(
            support or completed or rows,
            key=lambda row: (
                0.55 * float(row["quality_score"] or 0.0)
                + 0.45 * self._row_identity_confidence(row),
                -int(row["id"]),
            ),
        )
        consensus = {
            "candidate_count": len(rows),
            "processed_count": len(completed),
            "agreement_count": len(support),
            "person_id": winner_id,
            "score": round(consensus_score, 4) if consensus_score is not None else None,
        }
        connection.execute(
            "update face_observations set canonical = 0 where event_id = ? and candidate_track_id = ?",
            (event_id, track_id),
        )
        connection.execute(
            "update face_observations set canonical = 1, consensus_json = ? where id = ?",
            (json.dumps(consensus, separators=(",", ":")), int(canonical["id"])),
        )
        connection.execute(
            """
            update face_observations
            set candidate_person_id = null, candidate_confidence = null
            where event_id = ? and candidate_track_id = ? and id != ?
                and person_id is null
            """,
            (event_id, track_id, int(canonical["id"])),
        )
        recognizer = self.recognizer
        auto_identify = bool(
            recognizer is not None
            and getattr(recognizer.config, "face_auto_identify_enabled", False)
            and winner_id is not None
            and len(support) >= 2
            and len(support) > len(completed) / 2
            and consensus_score is not None
            and consensus_score >= getattr(recognizer.config, "face_auto_identify_threshold", 1.0)
            and all(self._candidate_auto_eligible(row, recognizer) for row in support)
        )
        if auto_identify:
            connection.execute(
                """
                update face_observations
                set person_id = ?, review_status = 'auto_identified',
                    match_confidence = ?, auto_identified = 1,
                    candidate_person_id = null, candidate_confidence = null
                where id = ? and person_id is null
                """,
                (winner_id, consensus_score, int(canonical["id"])),
            )

    @staticmethod
    def _row_identity_confidence(row: sqlite3.Row) -> float:
        value = row["match_confidence"] if row["person_id"] is not None else row["candidate_confidence"]
        return float(value or 0.0)

    @staticmethod
    def _candidate_auto_eligible(
        row: sqlite3.Row,
        recognizer: OpenVinoFaceRecognizer,
    ) -> bool:
        try:
            details = json.loads(row["match_details_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            return False
        return bool(
            float(details.get("margin") or 0.0)
            >= getattr(recognizer.config, "face_auto_identify_margin", 1.0)
            and len(details.get("reference_ids") or ()) >= 3
            and float(row["quality_score"] or 0.0) >= 0.45
        )

    def _best_match(
        self,
        connection: sqlite3.Connection,
        observation_id: int,
        embedding: np.ndarray,
        model_fingerprint: str,
    ) -> tuple[int | None, float | None]:
        match = self._match_result(
            connection,
            observation_id,
            embedding,
            model_fingerprint,
        )
        return match.person_id, match.score

    def _match_result(
        self,
        connection: sqlite3.Connection,
        observation_id: int,
        embedding: np.ndarray,
        model_fingerprint: str,
    ) -> FaceMatch:
        recognizer = self.recognizer
        if recognizer is None:
            return FaceMatch(None, None, None, None, (), ())
        if embedding.ndim != 1 or embedding.size == 0 or not np.all(np.isfinite(embedding)):
            return FaceMatch(None, None, None, None, (), ())
        rows = self._reference_gallery(
            connection,
            model_fingerprint,
            max(1, int(recognizer.config.face_max_references)),
            embedding.shape,
        )
        rejected_people = {
            int(row["person_id"])
            for row in connection.execute(
                "select person_id from face_rejections where observation_id = ?",
                (observation_id,),
            ).fetchall()
        }
        scores: dict[int, list[tuple[float, int]]] = {}
        for row in rows:
            if int(row["id"]) == observation_id:
                continue
            reference = row["_embedding"]
            score = float(np.dot(embedding, reference))
            if math.isfinite(score):
                scores.setdefault(int(row["person_id"]), []).append((score, int(row["id"])))
        ranked: list[tuple[float, int, list[tuple[float, int]]]] = []
        for person_id, values in scores.items():
            if person_id in rejected_people:
                continue
            top = sorted(values, reverse=True)[:3]
            ranked.append((float(sum(item[0] for item in top) / len(top)), person_id, top))
        if not ranked:
            return FaceMatch(None, None, None, None, (), ())
        ranked.sort(reverse=True)
        score, person_id, top = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else None
        score = max(0.0, min(1.0, score))
        margin = score - runner_up if runner_up is not None else score
        result = FaceMatch(
            person_id,
            round(score, 4),
            round(runner_up, 4) if runner_up is not None else None,
            round(margin, 4),
            tuple(item[1] for item in top),
            tuple(round(item[0], 4) for item in top),
        )
        if score < recognizer.config.face_match_threshold:
            return FaceMatch(
                None,
                result.score,
                result.runner_up_score,
                result.margin,
                result.reference_ids,
                result.reference_scores,
            )
        return result

    def _reference_gallery(
        self,
        connection: sqlite3.Connection,
        model_fingerprint: str,
        limit: int,
        embedding_shape: tuple[int, ...],
    ) -> list[dict[str, Any]]:
        with self._gallery_lock:
            key = (
                model_fingerprint,
                embedding_shape,
                limit,
                self._gallery_generation,
            )
            if key == self._gallery_cache_key:
                return self._gallery_cache
            rows = connection.execute(
                """
                select id, person_id, camera_id, confidence, quality_score, box_json,
                    reference_pinned, observed_at, embedding_blob
                from face_observations
                where person_id is not null and embedding_blob is not null
                    and embedding_model = ? and review_status = 'confirmed'
                order by observed_at desc, id desc
                """,
                (model_fingerprint,),
            ).fetchall()
            selected = self._select_reference_gallery(rows, limit, embedding_shape)
            self._gallery_cache_key = key
            self._gallery_cache = selected
            return selected

    @staticmethod
    def _select_reference_gallery(
        rows: list[sqlite3.Row],
        limit: int,
        embedding_shape: tuple[int, ...],
    ) -> list[dict[str, Any]]:
        """Choose central, high-quality, non-redundant references per identity."""
        grouped: dict[int, list[dict[str, Any]]] = {}
        for raw in rows:
            try:
                embedding = np.frombuffer(raw["embedding_blob"], dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if embedding.shape != embedding_shape or not np.all(np.isfinite(embedding)):
                continue
            norm = float(np.linalg.norm(embedding))
            if not math.isfinite(norm) or norm <= 1e-9:
                continue
            row = dict(raw)
            row["_embedding"] = embedding / norm
            grouped.setdefault(int(row["person_id"]), []).append(row)

        selected_all: list[dict[str, Any]] = []
        for candidates in grouped.values():
            for candidate in candidates:
                peers = [
                    float(np.dot(candidate["_embedding"], peer["_embedding"]))
                    for peer in candidates
                    if peer["id"] != candidate["id"]
                ]
                nearest = sorted(peers, reverse=True)[:4]
                centrality = sum(nearest) / len(nearest) if nearest else 1.0
                candidate["_nearest_peer"] = max(peers, default=1.0)
                quality = candidate.get("quality_score")
                if quality is None or not math.isfinite(float(quality)):
                    size_score = 0.0
                    try:
                        box = parse_face_box(json.loads(candidate.get("box_json") or "{}"))
                        if box is not None:
                            size_score = min(
                                1.0,
                                min(box["x2"] - box["x1"], box["y2"] - box["y1"]) / 160.0,
                            )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                    quality = 0.70 * float(candidate.get("confidence") or 0.0) + 0.30 * size_score
                quality = max(0.0, min(1.0, float(quality)))
                candidate["_base_score"] = 0.65 * quality + 0.35 * max(
                    0.0, min(1.0, (centrality + 1.0) / 2.0)
                )

            pinned = sorted(
                (item for item in candidates if bool(item.get("reference_pinned"))),
                key=lambda item: (item["_base_score"], item["observed_at"], item["id"]),
                reverse=True,
            )
            person_limit = max(limit, len(pinned))
            selected = list(pinned)
            pinned_ids = {int(item["id"]) for item in pinned}
            remaining = [
                item
                for item in candidates
                if int(item["id"]) not in pinned_ids
                and (len(candidates) < 4 or float(item["_nearest_peer"]) >= 0.15)
            ]
            if not selected and not remaining and candidates:
                remaining = list(candidates)
            if not selected and remaining:
                first = max(
                    remaining,
                    key=lambda item: (item["_base_score"], item["observed_at"], item["id"]),
                )
                selected.append(first)
                remaining.remove(first)
            while remaining and len(selected) < person_limit:
                selected_cameras = {str(item.get("camera_id") or "") for item in selected}

                def utility(item: dict[str, Any]) -> tuple[float, str, int]:
                    nearest = max(
                        float(np.dot(item["_embedding"], chosen["_embedding"]))
                        for chosen in selected
                    )
                    diversity = max(0.0, min(1.0, 1.0 - nearest))
                    camera_novelty = 1.0 if str(item.get("camera_id") or "") not in selected_cameras else 0.0
                    score = 0.55 * item["_base_score"] + 0.35 * diversity + 0.10 * camera_novelty
                    return score, str(item["observed_at"]), int(item["id"])

                chosen = max(remaining, key=utility)
                selected.append(chosen)
                remaining.remove(chosen)
            selected_all.extend(selected)
        return selected_all

    def _prune_locked(self, connection: sqlite3.Connection) -> int:
        total = int(connection.execute(
            "select count(*) from face_observations where canonical = 1"
        ).fetchone()[0])
        excess = total - self.max_observations
        if excess <= 0:
            return 0
        selected = connection.execute(
            """
            select id, event_id, candidate_track_id from face_observations
            where canonical = 1 and reference_pinned = 0 and id not in (
                select id from (
                    select id, row_number() over (
                        partition by person_id order by observed_at desc, id desc
                    ) as position
                    from face_observations
                    where person_id is not null and canonical = 1
                ) where position = 1
            )
            order by observed_at asc, id asc
            limit ?
            """,
            (excess,),
        ).fetchall()
        remove_ids: set[int] = set()
        candidate_paths: list[str] = []
        for row in selected:
            track_id = str(row["candidate_track_id"] or "")
            if not track_id:
                remove_ids.add(int(row["id"]))
                continue
            group = connection.execute(
                """
                select id, snapshot_path, reference_pinned
                from face_observations
                where event_id = ? and candidate_track_id = ?
                """,
                (int(row["event_id"]), track_id),
            ).fetchall()
            if any(bool(item["reference_pinned"]) for item in group):
                continue
            remove_ids.update(int(item["id"]) for item in group)
            candidate_paths.extend(str(item["snapshot_path"] or "") for item in group)
        if remove_ids:
            connection.executemany("delete from face_observations where id = ?", ((item,) for item in remove_ids))
            # Commit the authoritative database removal before deleting the
            # uniquely owned crop files. A later SQLite commit failure must
            # never leave retained rows pointing at already deleted images.
            connection.commit()
            self._invalidate_reference_gallery()
            for raw_path in candidate_paths:
                try:
                    event_snapshot_path(
                        self.storage_dir,
                        {"snapshot_path": raw_path},
                        self.media_storage,
                    ).unlink(missing_ok=True)
                except (FileNotFoundError, PermissionError, OSError, RuntimeError):
                    LOGGER.debug("could not remove pruned face candidate %s", raw_path)
        return len(remove_ids)

    def people(self) -> list[dict[str, Any]]:
        status_reader = getattr(self.recognizer, "status", None)
        recognizer_status = status_reader() if callable(status_reader) else {}
        model_fingerprint = str(recognizer_status.get("model_fingerprint") or "")
        with self._connect() as connection:
            rows = connection.execute(
                """
                select p.*, count(o.id) as observation_count,
                    sum(case when o.review_status = 'confirmed' then 1 else 0 end) as reference_count,
                    sum(case when o.review_status = 'confirmed' and o.embedding_blob is not null
                        and o.embedding_model = ? then 1 else 0 end) as usable_reference_count,
                    sum(case when o.review_status = 'confirmed' and o.reference_pinned = 1 then 1 else 0 end) as pinned_reference_count,
                    avg(case when o.review_status = 'confirmed' then o.quality_score end) as average_reference_quality,
                    max(o.observed_at) as last_seen_at,
                    (select id from face_observations latest
                     where latest.person_id = p.id
                     order by latest.observed_at desc limit 1) as preview_observation_id
                from face_people p
                left join face_observations o on o.person_id = p.id and o.canonical = 1
                group by p.id
                order by lower(p.name)
                """,
                (model_fingerprint,),
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                select count(*) as total_observations,
                    sum(case when person_id is null and recognition_pending = 0 and recognition_outcome = ? then 1 else 0 end) as unknown,
                    sum(case when person_id is not null then 1 else 0 end) as known,
                    sum(case when candidate_person_id is not null and person_id is null and recognition_pending = 0 and recognition_outcome = ? then 1 else 0 end) as suggested,
                    sum(case when person_id is null and recognition_outcome = ? then 1 else 0 end) as too_small,
                    sum(case when person_id is null and recognition_outcome = ? then 1 else 0 end) as processing_failed,
                    sum(case when person_id is null and recognition_pending = 0 and recognition_outcome = ? and embedding_blob is not null then 1 else 0 end) as embedded_unknown,
                    sum(case when person_id is null and recognition_pending = 1 then 1 else 0 end) as pending
                from face_observations
                where canonical = 1
                """,
                (
                    FACE_OUTCOME_EMBEDDED,
                    FACE_OUTCOME_EMBEDDED,
                    FACE_OUTCOME_TOO_SMALL,
                    FACE_OUTCOME_FAILED,
                    FACE_OUTCOME_EMBEDDED,
                ),
            ).fetchone()
            people = connection.execute("select count(*) from face_people").fetchone()[0]
            per_camera_rows = connection.execute(
                """
                select camera_id, count(*) as total,
                    sum(case when person_id is not null then 1 else 0 end) as known,
                    sum(case when person_id is null and recognition_pending = 0 and recognition_outcome = ? then 1 else 0 end) as unknown,
                    sum(case when person_id is null and recognition_outcome = ? then 1 else 0 end) as too_small,
                    sum(case when person_id is null and recognition_outcome = ? then 1 else 0 end) as processing_failed,
                    sum(case when person_id is null and recognition_pending = 1 then 1 else 0 end) as pending
                from face_observations
                where canonical = 1
                group by camera_id
                order by camera_id
                """,
                (FACE_OUTCOME_EMBEDDED, FACE_OUTCOME_TOO_SMALL, FACE_OUTCOME_FAILED),
            ).fetchall()
            candidate_rows = connection.execute(
                "select count(*) from face_observations where candidate_track_id != ''"
            ).fetchone()[0]
            track_rows = connection.execute(
                """
                select consensus_json from face_observations
                where canonical = 1 and candidate_track_id != ''
                """
            ).fetchall()
        consensus_tracks = 0
        multi_frame_tracks = 0
        candidate_total = 0
        for track_row in track_rows:
            try:
                consensus = json.loads(track_row["consensus_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                consensus = {}
            count = max(1, int(consensus.get("candidate_count") or 1))
            candidate_total += count
            multi_frame_tracks += int(count > 1)
            consensus_tracks += int(int(consensus.get("agreement_count") or 0) >= 2)
        total_observations = int(row["total_observations"] or 0)
        known = int(row["known"] or 0)
        unknown = int(row["unknown"] or 0)
        recognizable = known + unknown
        return {
            "people": int(people or 0),
            "observations": total_observations,
            "actionable_observations": recognizable,
            "unknown": unknown,
            "known": known,
            "suggested": int(row["suggested"] or 0),
            "too_small": int(row["too_small"] or 0),
            "processing_failed": int(row["processing_failed"] or 0),
            "embedded_unknown": int(row["embedded_unknown"] or 0),
            "pending": int(row["pending"] or 0),
            "identified_percent": round(100.0 * known / recognizable, 1) if recognizable else 0.0,
            "by_camera": [dict(camera_row) for camera_row in per_camera_rows],
            "candidate_frames": int(candidate_rows or 0),
            "temporal_tracks": len(track_rows),
            "multi_frame_tracks": multi_frame_tracks,
            "consensus_tracks": consensus_tracks,
            "average_candidates_per_track": round(
                candidate_total / len(track_rows), 2
            ) if track_rows else 0.0,
        }

    def camera_suitability(self) -> list[dict[str, Any]]:
        """Score how useful each camera is for face recognition."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                select camera_id, recognition_outcome, person_id,
                    quality_score, quality_json
                from face_observations
                where canonical = 1
                order by camera_id, observed_at desc
                """
            ).fetchall()

        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(str(row["camera_id"] or ""), []).append(row)

        result: list[dict[str, Any]] = []
        for camera_id, items in grouped.items():
            total = len(items)
            embedded = sum(
                1 for row in items
                if row["recognition_outcome"] == FACE_OUTCOME_EMBEDDED
            )
            too_small = sum(
                1 for row in items
                if row["recognition_outcome"] == FACE_OUTCOME_TOO_SMALL
            )
            failed = sum(
                1 for row in items
                if row["recognition_outcome"] == FACE_OUTCOME_FAILED
            )
            known = sum(1 for row in items if row["person_id"] is not None)
            qualities = [
                float(row["quality_score"])
                for row in items
                if row["quality_score"] is not None
            ]
            sizes: list[float] = []
            for row in items:
                try:
                    quality = json.loads(row["quality_json"] or "{}")
                    value = float(quality.get("size"))
                    if math.isfinite(value):
                        sizes.append(max(0.0, min(1.0, value)))
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass

            usable_rate = embedded / total if total else 0.0
            avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
            avg_size = sum(sizes) / len(sizes) if sizes else 0.0
            known_rate = known / embedded if embedded else 0.0
            score = max(
                0.0,
                min(
                    1.0,
                    0.50 * usable_rate
                    + 0.25 * avg_quality
                    + 0.15 * avg_size
                    + 0.10 * known_rate,
                ),
            )
            if total < 5:
                grade = "insufficient_data"
            elif score >= 0.78:
                grade = "excellent"
            elif score >= 0.62:
                grade = "good"
            elif score >= 0.45:
                grade = "marginal"
            else:
                grade = "poor"

            result.append({
                "camera_id": camera_id,
                "score": round(score, 4),
                "grade": grade,
                "observations": total,
                "embedded": embedded,
                "known": known,
                "too_small": too_small,
                "processing_failed": failed,
                "usable_rate": round(usable_rate, 4),
                "too_small_rate": round(too_small / total, 4) if total else 0.0,
                "failure_rate": round(failed / total, 4) if total else 0.0,
                "average_quality": round(avg_quality, 4),
                "average_face_size": round(avg_size, 4),
                "identified_rate": round(known_rate, 4),
            })

        return sorted(
            result,
            key=lambda item: (-item["score"], item["camera_id"]),
        )

    def benchmark_production_matcher(self) -> dict[str, Any]:
        """Emulate production gallery matching with leave-one-out evaluation."""
        recognizer = self.recognizer
        if recognizer is None:
            return {"ready": False, "message": "Face recognition is not configured."}

        status = recognizer.status()
        model_fingerprint = str(status.get("model_fingerprint") or "")
        if not model_fingerprint:
            return {"ready": False, "message": "The face model is not ready."}

        max_refs = max(
            1,
            int(getattr(recognizer.config, "face_max_references", 20)),
        )
        current_threshold = float(recognizer.config.face_match_threshold)

        with self._connect() as connection:
            rows = connection.execute(
                """
                select o.id, o.person_id, p.name as person_name,
                    o.camera_id, o.quality_score, o.confidence,
                    o.box_json, o.reference_pinned, o.observed_at,
                    o.embedding_blob
                from face_observations o
                join face_people p on p.id = o.person_id
                where o.canonical = 1
                    and o.person_id is not null
                    and o.review_status = 'confirmed'
                    and o.embedding_model = ?
                    and o.embedding_blob is not null
                order by lower(p.name), o.observed_at desc, o.id desc
                """,
                (model_fingerprint,),
            ).fetchall()

        samples: list[dict[str, Any]] = []
        for raw in rows:
            vector = np.frombuffer(raw["embedding_blob"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if (
                vector.size
                and math.isfinite(norm)
                and norm > 1e-9
                and np.all(np.isfinite(vector))
            ):
                item = dict(raw)
                item["_embedding"] = vector / norm
                samples.append(item)

        grouped: dict[int, list[dict[str, Any]]] = {}
        names: dict[int, str] = {}
        for sample in samples:
            person_id = int(sample["person_id"])
            grouped.setdefault(person_id, []).append(sample)
            names[person_id] = str(sample["person_name"] or "")

        if len(grouped) < 2 or len(samples) < 6:
            return {
                "ready": False,
                "message": (
                    "Benchmarking needs at least two identities and six "
                    "confirmed embedded observations."
                ),
                "identities": len(grouped),
                "samples": len(samples),
            }

        per_identity: dict[int, dict[str, Any]] = {}
        trials: list[dict[str, Any]] = []

        for held_out in samples:
            held_out_id = int(held_out["id"])
            held_out_person_id = int(held_out["person_id"])
            candidate_rows = [
                row for row in samples
                if int(row["id"]) != held_out_id
            ]

            gallery = self._select_reference_gallery(
                candidate_rows,
                max_refs,
                held_out["_embedding"].shape,
            )
            if not gallery:
                continue

            by_person: dict[int, list[tuple[float, int]]] = {}
            for reference in gallery:
                reference_person_id = int(reference["person_id"])
                score = float(
                    np.dot(
                        held_out["_embedding"],
                        reference["_embedding"],
                    )
                )
                by_person.setdefault(reference_person_id, []).append(
                    (score, int(reference["id"]))
                )

            if held_out_person_id not in by_person or len(by_person) < 2:
                continue

            ranked: list[tuple[float, int, list[tuple[float, int]]]] = []
            for person_id, values in by_person.items():
                top = sorted(values, reverse=True)[:3]
                aggregate = sum(item[0] for item in top) / len(top)
                ranked.append((aggregate, person_id, top))
            ranked.sort(reverse=True)

            top_score, top_person_id, top_refs = ranked[0]
            true_entry = next(
                entry for entry in ranked
                if entry[1] == held_out_person_id
            )
            true_score = float(true_entry[0])
            best_wrong = max(
                score for score, person_id, _refs in ranked
                if person_id != held_out_person_id
            )
            rank_one = top_person_id == held_out_person_id

            trials.append({
                "observation_id": held_out_id,
                "person_id": held_out_person_id,
                "person_name": names[held_out_person_id],
                "camera_id": str(held_out["camera_id"] or ""),
                "true_score": true_score,
                "best_wrong_score": float(best_wrong),
                "rank_one": rank_one,
                "predicted_person_id": int(top_person_id),
                "predicted_person_name": names.get(int(top_person_id), ""),
                "predicted_score": float(top_score),
                "accepted_at_current": bool(
                    rank_one and top_score >= current_threshold
                ),
                "top_reference_ids": [int(item[1]) for item in top_refs],
            })

        if not trials:
            return {
                "ready": False,
                "message": "No leave-one-out trials could be formed.",
            }

        def q(values: list[float], quantile: float) -> float | None:
            if not values:
                return None
            return round(float(np.quantile(values, quantile)), 4)

        true_scores = [float(item["true_score"]) for item in trials]
        wrong_scores = [float(item["best_wrong_score"]) for item in trials]
        rank_one_accuracy = sum(
            1 for item in trials if item["rank_one"]
        ) / len(trials)

        threshold_sweep: list[tuple[float, float, float, float]] = []
        for raw in range(20, 71):
            threshold = raw / 100.0
            accepted_correct = sum(
                1
                for item in trials
                if item["rank_one"]
                and item["predicted_score"] >= threshold
            )
            accepted_wrong = sum(
                1
                for item in trials
                if (not item["rank_one"])
                and item["predicted_score"] >= threshold
            )
            true_accept_rate = accepted_correct / len(trials)
            false_accept_rate = accepted_wrong / len(trials)
            miss_rate = 1.0 - true_accept_rate
            objective = false_accept_rate * 8.0 + miss_rate
            threshold_sweep.append(
                (
                    threshold,
                    true_accept_rate,
                    false_accept_rate,
                    objective,
                )
            )

        constrained = [
            item
            for item in threshold_sweep
            if item[2] <= 0.005
        ]
        best = min(
            constrained or threshold_sweep,
            key=lambda item: (
                item[3],
                -item[1],
                item[2],
                item[0],
            ),
        )

        by_identity_trials: dict[int, list[dict[str, Any]]] = {}
        for trial in trials:
            by_identity_trials.setdefault(
                int(trial["person_id"]),
                [],
            ).append(trial)

        identity_rows: list[dict[str, Any]] = []
        for person_id, person_trials in by_identity_trials.items():
            person_true = [
                float(item["true_score"])
                for item in person_trials
            ]
            person_wrong = [
                float(item["best_wrong_score"])
                for item in person_trials
            ]
            person_rank_one = sum(
                1 for item in person_trials if item["rank_one"]
            ) / len(person_trials)
            person_current_accept = sum(
                1
                for item in person_trials
                if item["accepted_at_current"]
            ) / len(person_trials)
            worst_trial = min(
                person_trials,
                key=lambda item: (
                    float(item["true_score"])
                    - float(item["best_wrong_score"])
                ),
            )
            identity_rows.append({
                "person_id": person_id,
                "name": names.get(person_id, ""),
                "trials": len(person_trials),
                "rank_one_accuracy": round(person_rank_one, 4),
                "accepted_at_current_threshold": round(
                    person_current_accept,
                    4,
                ),
                "true_score": {
                    "p05": q(person_true, 0.05),
                    "median": q(person_true, 0.50),
                    "p95": q(person_true, 0.95),
                },
                "best_wrong_score": {
                    "p95": q(person_wrong, 0.95),
                    "maximum": round(max(person_wrong), 4),
                },
                "worst_margin": round(
                    float(worst_trial["true_score"])
                    - float(worst_trial["best_wrong_score"]),
                    4,
                ),
                "worst_case": {
                    "observation_id": int(worst_trial["observation_id"]),
                    "camera_id": str(worst_trial["camera_id"]),
                    "true_score": round(
                        float(worst_trial["true_score"]),
                        4,
                    ),
                    "predicted_person_id": int(
                        worst_trial["predicted_person_id"]
                    ),
                    "predicted_person_name": str(
                        worst_trial["predicted_person_name"]
                    ),
                    "predicted_score": round(
                        float(worst_trial["predicted_score"]),
                        4,
                    ),
                },
            })

        identity_rows.sort(
            key=lambda item: (
                item["rank_one_accuracy"],
                item["accepted_at_current_threshold"],
                item["name"].lower(),
            )
        )

        return {
            "ready": True,
            "model_fingerprint": model_fingerprint,
            "identities": len(grouped),
            "samples": len(samples),
            "trials": len(trials),
            "gallery_limit": max_refs,
            "rank_one_accuracy": round(rank_one_accuracy, 4),
            "true_score": {
                "p05": q(true_scores, 0.05),
                "median": q(true_scores, 0.50),
                "p95": q(true_scores, 0.95),
            },
            "best_wrong_score": {
                "p95": q(wrong_scores, 0.95),
                "p99": q(wrong_scores, 0.99),
                "maximum": round(max(wrong_scores), 4),
            },
            "current": {
                "match_threshold": current_threshold,
                "true_accept_rate": round(
                    sum(
                        1
                        for item in trials
                        if item["rank_one"]
                        and item["predicted_score"] >= current_threshold
                    )
                    / len(trials),
                    4,
                ),
                "false_accept_rate": round(
                    sum(
                        1
                        for item in trials
                        if (not item["rank_one"])
                        and item["predicted_score"] >= current_threshold
                    )
                    / len(trials),
                    4,
                ),
            },
            "recommended": {
                "match_threshold": round(best[0], 2),
                "true_accept_rate": round(best[1], 4),
                "false_accept_rate": round(best[2], 4),
            },
            "results": identity_rows,
            "message": (
                "This benchmark mirrors SurvNG gallery selection and top-three "
                "reference aggregation more closely than raw pairwise similarity."
            ),
        }

    def benchmark_camera_pairs(self) -> dict[str, Any]:
        """Report same-person similarity by camera pair for confirmed identities."""
        recognizer = self.recognizer
        if recognizer is None:
            return {"ready": False, "message": "Face recognition is not configured."}

        status = recognizer.status()
        model_fingerprint = str(status.get("model_fingerprint") or "")
        if not model_fingerprint:
            return {"ready": False, "message": "The face model is not ready."}

        with self._connect() as connection:
            rows = connection.execute(
                """
                select o.id, o.person_id, p.name as person_name,
                    o.camera_id, o.embedding_blob
                from face_observations o
                join face_people p on p.id = o.person_id
                where o.canonical = 1
                    and o.person_id is not null
                    and o.review_status = 'confirmed'
                    and o.embedding_model = ?
                    and o.embedding_blob is not null
                    and o.camera_id != ''
                order by lower(p.name), o.id
                """,
                (model_fingerprint,),
            ).fetchall()

        samples: list[dict[str, Any]] = []
        for row in rows:
            vector = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if (
                vector.size
                and math.isfinite(norm)
                and norm > 1e-9
                and np.all(np.isfinite(vector))
            ):
                samples.append({
                    "id": int(row["id"]),
                    "person_id": int(row["person_id"]),
                    "person_name": str(row["person_name"] or ""),
                    "camera_id": str(row["camera_id"] or ""),
                    "embedding": vector / norm,
                })

        grouped: dict[int, list[dict[str, Any]]] = {}
        for sample in samples:
            grouped.setdefault(sample["person_id"], []).append(sample)

        pair_scores: dict[tuple[str, str], list[float]] = {}
        person_pair_scores: dict[
            tuple[int, str, str],
            list[float],
        ] = {}

        for person_id, person_samples in grouped.items():
            for index, left in enumerate(person_samples):
                for right in person_samples[index + 1:]:
                    camera_a, camera_b = sorted(
                        [left["camera_id"], right["camera_id"]]
                    )
                    key = (camera_a, camera_b)
                    score = float(
                        np.dot(
                            left["embedding"],
                            right["embedding"],
                        )
                    )
                    pair_scores.setdefault(key, []).append(score)
                    person_pair_scores.setdefault(
                        (person_id, camera_a, camera_b),
                        [],
                    ).append(score)

        def summarize(values: list[float]) -> dict[str, Any]:
            return {
                "pairs": len(values),
                "p05": round(float(np.quantile(values, 0.05)), 4),
                "median": round(float(np.quantile(values, 0.50)), 4),
                "p95": round(float(np.quantile(values, 0.95)), 4),
            }

        global_rows = [
            {
                "camera_a": camera_a,
                "camera_b": camera_b,
                **summarize(values),
            }
            for (camera_a, camera_b), values in pair_scores.items()
            if values
        ]
        global_rows.sort(
            key=lambda item: (
                item["median"],
                -item["pairs"],
                item["camera_a"],
                item["camera_b"],
            )
        )

        names = {
            int(sample["person_id"]): str(sample["person_name"])
            for sample in samples
        }
        per_identity = [
            {
                "person_id": person_id,
                "name": names.get(person_id, ""),
                "camera_a": camera_a,
                "camera_b": camera_b,
                **summarize(values),
            }
            for (
                person_id,
                camera_a,
                camera_b,
            ), values in person_pair_scores.items()
            if values
        ]
        per_identity.sort(
            key=lambda item: (
                item["median"],
                -item["pairs"],
                item["name"].lower(),
                item["camera_a"],
                item["camera_b"],
            )
        )

        return {
            "ready": True,
            "model_fingerprint": model_fingerprint,
            "samples": len(samples),
            "camera_pairs": global_rows,
            "identity_camera_pairs": per_identity,
            "message": (
                "Low medians identify camera transitions that produce weak "
                "same-person embedding consistency."
            ),
        }

    def benchmark_by_identity(self) -> dict[str, Any]:
        """Return per-identity embedding cohesion and separation diagnostics."""
        recognizer = self.recognizer
        if recognizer is None:
            return {"ready": False, "message": "Face recognition is not configured."}

        status = recognizer.status()
        model_fingerprint = str(status.get("model_fingerprint") or "")
        if not model_fingerprint:
            return {"ready": False, "message": "The face model is not ready."}

        with self._connect() as connection:
            rows = connection.execute(
                """
                select o.id, o.person_id, p.name as person_name,
                    o.camera_id, o.quality_score, o.embedding_blob
                from face_observations o
                join face_people p on p.id = o.person_id
                where o.canonical = 1
                    and o.person_id is not null
                    and o.review_status = 'confirmed'
                    and o.embedding_model = ?
                    and o.embedding_blob is not null
                order by lower(p.name), o.id
                """,
                (model_fingerprint,),
            ).fetchall()

        samples: list[dict[str, Any]] = []
        for row in rows:
            vector = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if (
                vector.size
                and math.isfinite(norm)
                and norm > 1e-9
                and np.all(np.isfinite(vector))
            ):
                samples.append({
                    "id": int(row["id"]),
                    "person_id": int(row["person_id"]),
                    "person_name": str(row["person_name"] or ""),
                    "camera_id": str(row["camera_id"] or ""),
                    "quality_score": (
                        float(row["quality_score"])
                        if row["quality_score"] is not None
                        else None
                    ),
                    "embedding": vector / norm,
                })

        grouped: dict[int, list[dict[str, Any]]] = {}
        for sample in samples:
            grouped.setdefault(sample["person_id"], []).append(sample)

        if len(grouped) < 2:
            return {
                "ready": False,
                "message": "At least two confirmed identities are required.",
                "identities": len(grouped),
                "samples": len(samples),
            }

        identities: list[dict[str, Any]] = []
        for person_id, person_samples in grouped.items():
            person_name = person_samples[0]["person_name"]
            cameras = sorted({
                str(sample["camera_id"])
                for sample in person_samples
                if sample["camera_id"]
            })
            qualities = [
                float(sample["quality_score"])
                for sample in person_samples
                if sample["quality_score"] is not None
                and math.isfinite(float(sample["quality_score"]))
            ]

            genuine_scores: list[float] = []
            for index, sample_a in enumerate(person_samples):
                for sample_b in person_samples[index + 1:]:
                    genuine_scores.append(
                        float(np.dot(sample_a["embedding"], sample_b["embedding"]))
                    )

            other_samples = [
                sample
                for other_id, items in grouped.items()
                if other_id != person_id
                for sample in items
            ]
            impostor_scores: list[float] = []
            nearest_other_by_sample: list[float] = []
            nearest_other_identity_by_sample: list[tuple[float, int, str]] = []

            for sample in person_samples:
                best_score = -1.0
                best_id = 0
                best_name = ""
                for other in other_samples:
                    score = float(np.dot(sample["embedding"], other["embedding"]))
                    impostor_scores.append(score)
                    if score > best_score:
                        best_score = score
                        best_id = int(other["person_id"])
                        best_name = str(other["person_name"])
                if best_score >= -1.0:
                    nearest_other_by_sample.append(best_score)
                    nearest_other_identity_by_sample.append(
                        (best_score, best_id, best_name)
                    )

            def q(values: list[float], quantile: float) -> float | None:
                if not values:
                    return None
                return round(float(np.quantile(values, quantile)), 4)

            median_genuine = q(genuine_scores, 0.50)
            p05_genuine = q(genuine_scores, 0.05)
            p95_genuine = q(genuine_scores, 0.95)
            maximum_impostor = (
                round(max(impostor_scores), 4)
                if impostor_scores
                else None
            )
            p99_impostor = q(impostor_scores, 0.99)
            nearest_other = (
                max(nearest_other_identity_by_sample, key=lambda item: item[0])
                if nearest_other_identity_by_sample
                else None
            )
            nearest_other_score = (
                round(float(nearest_other[0]), 4)
                if nearest_other
                else None
            )

            # A simple risk score intended for operator triage, not classification.
            overlap_margin = None
            if median_genuine is not None and nearest_other_score is not None:
                overlap_margin = round(
                    float(median_genuine) - float(nearest_other_score),
                    4,
                )

            flags: list[str] = []
            if len(person_samples) < 3:
                flags.append("low_sample_count")
            if median_genuine is not None and median_genuine < 0.20:
                flags.append("weak_identity_cohesion")
            if p05_genuine is not None and p05_genuine < 0.05:
                flags.append("very_low_tail_similarity")
            if maximum_impostor is not None and maximum_impostor >= 0.30:
                flags.append("high_impostor_overlap")
            if overlap_margin is not None and overlap_margin < 0.05:
                flags.append("poor_separation")
            if len(cameras) <= 1 and len(person_samples) >= 3:
                flags.append("single_camera_gallery")

            identities.append({
                "person_id": person_id,
                "name": person_name,
                "samples": len(person_samples),
                "camera_count": len(cameras),
                "cameras": cameras,
                "average_quality": (
                    round(sum(qualities) / len(qualities), 4)
                    if qualities
                    else None
                ),
                "genuine_pairs": len(genuine_scores),
                "same_person": {
                    "p05": p05_genuine,
                    "median": median_genuine,
                    "p95": p95_genuine,
                },
                "different_person": {
                    "p99": p99_impostor,
                    "maximum": maximum_impostor,
                    "nearest_identity_id": (
                        int(nearest_other[1]) if nearest_other else None
                    ),
                    "nearest_identity_name": (
                        str(nearest_other[2]) if nearest_other else None
                    ),
                    "nearest_identity_score": nearest_other_score,
                },
                "separation_margin": overlap_margin,
                "flags": flags,
            })

        identities.sort(
            key=lambda item: (
                -len(item["flags"]),
                item["separation_margin"]
                if item["separation_margin"] is not None
                else 999.0,
                item["name"].lower(),
            )
        )

        flagged = sum(1 for item in identities if item["flags"])
        return {
            "ready": True,
            "model_fingerprint": model_fingerprint,
            "identities": len(identities),
            "samples": len(samples),
            "flagged_identities": flagged,
            "results": identities,
            "message": (
                "Flags are diagnostic only. Review weak identities and gallery "
                "coverage before changing production thresholds."
            ),
        }

    def benchmark(self) -> dict[str, Any]:
        """Benchmark identity and clustering thresholds on reviewed embeddings."""
        recognizer = self.recognizer
        if recognizer is None:
            return {"ready": False, "message": "Face recognition is not configured."}
        status = recognizer.status()
        model_fingerprint = str(status.get("model_fingerprint") or "")
        if not model_fingerprint:
            return {"ready": False, "message": "The face model is not ready."}

        with self._connect() as connection:
            rows = connection.execute(
                """
                select id, person_id, embedding_blob
                from face_observations
                where canonical = 1
                    and person_id is not null
                    and review_status = 'confirmed'
                    and embedding_model = ?
                    and embedding_blob is not null
                order by person_id, id
                """,
                (model_fingerprint,),
            ).fetchall()

        samples: list[tuple[int, int, np.ndarray]] = []
        for row in rows:
            vector = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if (
                vector.size
                and math.isfinite(norm)
                and norm > 1e-9
                and np.all(np.isfinite(vector))
            ):
                samples.append(
                    (int(row["id"]), int(row["person_id"]), vector / norm)
                )

        identities = {person_id for _, person_id, _ in samples}
        if len(identities) < 2 or len(samples) < 6:
            return {
                "ready": False,
                "message": (
                    "Benchmarking needs at least two identities and six "
                    "confirmed embedded observations."
                ),
                "identities": len(identities),
                "samples": len(samples),
            }

        genuine: list[float] = []
        impostor: list[float] = []
        for index, (_id_a, person_a, emb_a) in enumerate(samples):
            for _id_b, person_b, emb_b in samples[index + 1:]:
                score = float(np.dot(emb_a, emb_b))
                if person_a == person_b:
                    genuine.append(score)
                else:
                    impostor.append(score)

        if not genuine or not impostor:
            return {
                "ready": False,
                "message": "More varied confirmed samples are required.",
            }

        sweep = []
        for raw in range(30, 91):
            threshold = raw / 100.0
            tar = sum(score >= threshold for score in genuine) / len(genuine)
            far = sum(score >= threshold for score in impostor) / len(impostor)
            balanced_error = ((1.0 - tar) + far) / 2.0
            sweep.append((threshold, tar, far, balanced_error))

        constrained = [item for item in sweep if item[2] <= 0.01]
        best_match = min(
            constrained or sweep,
            key=lambda item: (item[3], -item[1], item[2], item[0]),
        )
        cluster_candidates = [
            item
            for item in sweep
            if item[2] <= 0.005 and item[1] >= 0.80
        ]
        best_cluster = min(
            cluster_candidates or constrained or sweep,
            key=lambda item: (item[3], -item[1], item[2], item[0]),
        )

        def quantile(values, q):
            return round(float(np.quantile(values, q)), 4)

        return {
            "ready": True,
            "model_fingerprint": model_fingerprint,
            "identities": len(identities),
            "samples": len(samples),
            "genuine_pairs": len(genuine),
            "impostor_pairs": len(impostor),
            "same_person": {
                "p05": quantile(genuine, 0.05),
                "median": quantile(genuine, 0.50),
                "p95": quantile(genuine, 0.95),
            },
            "different_person": {
                "p95": quantile(impostor, 0.95),
                "p99": quantile(impostor, 0.99),
                "maximum": round(max(impostor), 4),
            },
            "recommended": {
                "match_threshold": round(best_match[0], 2),
                "match_true_accept_rate": round(best_match[1], 4),
                "match_false_accept_rate": round(best_match[2], 4),
                "unknown_cluster_threshold": round(best_cluster[0], 2),
                "cluster_true_link_rate": round(best_cluster[1], 4),
                "cluster_false_link_rate": round(best_cluster[2], 4),
            },
            "current": {
                "match_threshold": float(recognizer.config.face_match_threshold),
                "unknown_cluster_threshold": 0.62,
            },
            "message": (
                "Recommendations are empirical and should be reviewed before "
                "changing production thresholds."
            ),
        }

    def calibration(self) -> dict[str, Any]:
        """Measure gallery separation using reviewed identities and rejections."""
        recognizer = self.recognizer
        if recognizer is None:
            return {"ready": False, "message": "Face recognition is not configured."}
        status = recognizer.status()
        model_fingerprint = str(status.get("model_fingerprint") or "")
        if not model_fingerprint:
            return {"ready": False, "message": "The face model is not ready."}
        with self._connect() as connection:
            rows = connection.execute(
                """
                select id, person_id, embedding_blob from face_observations
                where person_id is not null and review_status = 'confirmed'
                    and embedding_model = ? and embedding_blob is not null
                """,
                (model_fingerprint,),
            ).fetchall()
            rejected_rows = connection.execute(
                """
                select o.id, o.embedding_blob, r.person_id
                from face_rejections r
                join face_observations o on o.id = r.observation_id
                where o.embedding_model = ? and o.embedding_blob is not null
                """,
                (model_fingerprint,),
            ).fetchall()
        embeddings: list[tuple[int, int, np.ndarray]] = []
        for row in rows:
            vector = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.size and math.isfinite(norm) and norm > 1e-9 and np.all(np.isfinite(vector)):
                embeddings.append((int(row["id"]), int(row["person_id"]), vector / norm))
        if not embeddings:
            return {
                "ready": False,
                "message": "Confirm at least two people with multiple face observations to calibrate matching.",
                "confirmed_samples": 0,
                "rejected_samples": 0,
            }
        with self._connect() as connection:
            gallery = self._reference_gallery(
                connection,
                model_fingerprint,
                max(1, int(getattr(recognizer.config, "face_max_references", 20))),
                embeddings[0][2].shape,
            )
        gallery_embeddings = [
            (int(row["id"]), int(row["person_id"]), row["_embedding"])
            for row in gallery
        ]
        true_scores: list[float] = []
        impostor_scores: list[float] = []
        margins: list[float] = []
        rank_one = 0
        for observation_id, person_id, target in embeddings:
            grouped: dict[int, list[float]] = {}
            for reference_id, reference_person_id, reference in gallery_embeddings:
                if reference_id == observation_id:
                    continue
                grouped.setdefault(reference_person_id, []).append(float(np.dot(target, reference)))
            if person_id not in grouped or len(grouped) < 2:
                continue
            ranked = sorted(
                (
                    (sum(sorted(values, reverse=True)[:3]) / min(3, len(values)), candidate_id)
                    for candidate_id, values in grouped.items()
                ),
                reverse=True,
            )
            true_score = next(score for score, candidate_id in ranked if candidate_id == person_id)
            best_other = max(score for score, candidate_id in ranked if candidate_id != person_id)
            true_scores.append(true_score)
            impostor_scores.append(best_other)
            margins.append(true_score - best_other)
            rank_one += int(ranked[0][1] == person_id)

        rejected_scores: list[float] = []
        references_by_person: dict[int, list[np.ndarray]] = {}
        for _observation_id, person_id, vector in gallery_embeddings:
            references_by_person.setdefault(person_id, []).append(vector)
        for row in rejected_rows:
            target = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            norm = float(np.linalg.norm(target))
            references = references_by_person.get(int(row["person_id"]), [])
            if not references or not math.isfinite(norm) or norm <= 1e-9:
                continue
            scores = sorted(
                (float(np.dot(target / norm, reference)) for reference in references),
                reverse=True,
            )[:3]
            rejected_scores.append(sum(scores) / len(scores))
        negative_scores = impostor_scores + rejected_scores
        if not true_scores or not negative_scores:
            return {
                "ready": False,
                "message": "Confirm at least two people with multiple face observations to calibrate matching.",
                "confirmed_samples": len(true_scores),
                "rejected_samples": len(rejected_scores),
            }
        negative_p99 = float(np.quantile(negative_scores, 0.99))
        true_p75 = float(np.quantile(true_scores, 0.75))
        suggestion = round(max(0.40, min(0.60, negative_p99 + 0.06)), 2)
        automatic = round(max(0.55, min(0.80, negative_p99 + 0.20, true_p75)), 2)
        automatic = max(automatic, suggestion + 0.10)
        margin = 0.12
        return {
            "ready": True,
            "confirmed_samples": len(true_scores),
            "rejected_samples": len(rejected_scores),
            "rank_one_accuracy": round(rank_one / len(true_scores), 4),
            "median_same_person_score": round(float(np.median(true_scores)), 4),
            "maximum_impostor_score": round(max(negative_scores), 4),
            "recommended": {
                "suggestion_threshold": suggestion,
                "automatic_threshold": round(automatic, 2),
                "automatic_margin": margin,
            },
            "current": {
                "suggestion_threshold": recognizer.config.face_match_threshold,
                "automatic_threshold": recognizer.config.face_auto_identify_threshold,
                "automatic_margin": recognizer.config.face_auto_identify_margin,
            },
            "message": (
                "Recommendations are based on leave-one-out comparisons of confirmed faces"
                + (
                    " and explicit rejected matches."
                    if rejected_scores
                    else "; additional varied confirmations will strengthen calibration."
                )
            ),
        }

    def observations(
        self,
        *,
        person_id: int | None = None,
        camera_id: str = "",
        status: str = "all",
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["o.canonical = 1"]
        values: list[Any] = []
        if person_id is not None:
            clauses.append("o.person_id = ?")
            values.append(person_id)
        elif status == "unknown":
            clauses.append("o.person_id is null and o.recognition_pending = 0 and o.recognition_outcome = ?")
            values.append(FACE_OUTCOME_EMBEDDED)
        elif status == "known":
            clauses.append("o.person_id is not null")
        elif status == "suggested":
            clauses.append("o.person_id is null and o.candidate_person_id is not null and o.recognition_pending = 0 and o.recognition_outcome = ?")
            values.append(FACE_OUTCOME_EMBEDDED)
        elif status == "unusable":
            clauses.append("o.person_id is null and o.recognition_outcome in (?, ?)")
            values.extend((FACE_OUTCOME_TOO_SMALL, FACE_OUTCOME_FAILED))
        elif status == "pending":
            clauses.append("o.person_id is null and o.recognition_pending = 1")
        if camera_id:
            clauses.append("o.camera_id = ?")
            values.append(camera_id)
        where = f"where {' and '.join(clauses)}" if clauses else ""
        values.extend([max(1, min(limit, 500)), max(0, offset)])
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                select o.*, p.name as person_name, candidate.name as candidate_person_name
                from face_observations o
                left join face_people p on p.id = o.person_id
                left join face_people candidate on candidate.id = o.candidate_person_id
                {where}
                order by o.observed_at desc, o.id desc
                limit ? offset ?
                """,
                values,
            ).fetchall()
        return [self._observation_row(row) for row in rows]

    def observation_count(
        self,
        *,
        person_id: int | None = None,
        camera_id: str = "",
        status: str = "all",
    ) -> int:
        clauses: list[str] = ["canonical = 1"]
        values: list[Any] = []
        if person_id is not None:
            clauses.append("person_id = ?")
            values.append(person_id)
        elif status == "unknown":
            clauses.append("person_id is null and recognition_pending = 0 and recognition_outcome = ?")
            values.append(FACE_OUTCOME_EMBEDDED)
        elif status == "known":
            clauses.append("person_id is not null")
        elif status == "suggested":
            clauses.append("person_id is null and candidate_person_id is not null and recognition_pending = 0 and recognition_outcome = ?")
            values.append(FACE_OUTCOME_EMBEDDED)
        elif status == "unusable":
            clauses.append("person_id is null and recognition_outcome in (?, ?)")
            values.extend((FACE_OUTCOME_TOO_SMALL, FACE_OUTCOME_FAILED))
        elif status == "pending":
            clauses.append("person_id is null and recognition_pending = 1")
        if camera_id:
            clauses.append("camera_id = ?")
            values.append(camera_id)
        where = f"where {' and '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            row = connection.execute(
                f"select count(*) from face_observations {where}",
                values,
            ).fetchone()
        return int(row[0] or 0)

    def observation(self, observation_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                select o.*, p.name as person_name, candidate.name as candidate_person_name
                from face_observations o
                left join face_people p on p.id = o.person_id
                left join face_people candidate on candidate.id = o.candidate_person_id
                where o.id = ?
                """,
                (observation_id,),
            ).fetchone()
        return self._observation_row(row) if row else None

    def refresh_unknown_clusters(self, threshold: float | None = None) -> int:
        if threshold is None:
            config = getattr(self.recognizer, "config", None)
            threshold = float(
                getattr(
                    config,
                    "face_unknown_cluster_threshold",
                    DEFAULT_UNKNOWN_CLUSTER_THRESHOLD,
                )
            )
        recognizer_status = self.recognizer.status() if self.recognizer is not None else {}
        fingerprint = str(recognizer_status.get("model_fingerprint") or "")
        if not fingerprint:
            return 0
        with self._lock, self._connect() as connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    """
                    select id, embedding_blob, quality_score
                    from face_observations
                    where canonical = 1
                        and person_id is null
                        and recognition_pending = 0
                        and recognition_outcome = ?
                        and embedding_model = ?
                        and embedding_blob is not null
                    order by id
                    """,
                    (FACE_OUTCOME_EMBEDDED, fingerprint),
                ).fetchall()
            ]
            membership = cluster_unknown_embeddings(rows, threshold=threshold)
            now = datetime.now(timezone.utc).isoformat()
            connection.execute("delete from face_unknown_members")
            connection.executemany(
                "insert into face_unknown_members (observation_id, cluster_id, updated_at) values (?, ?, ?)",
                [(observation_id, cluster_id, now) for observation_id, cluster_id in membership.items()],
            )
        return len(set(membership.values()))

    def unknown_cluster_health(self) -> dict[str, Any]:
        """Summarize recurring-unknown clustering and effective thresholds."""
        recognizer = self.recognizer
        config = getattr(recognizer, "config", None)
        match_threshold = float(getattr(config, "face_match_threshold", 0.30))
        cluster_threshold = float(
            getattr(
                config,
                "face_unknown_cluster_threshold",
                DEFAULT_UNKNOWN_CLUSTER_THRESHOLD,
            )
        )

        clusters = self.unknown_clusters()
        counts = sorted(
            (int(cluster.get("observation_count") or 0) for cluster in clusters),
            reverse=True,
        )
        total_members = sum(counts)
        singletons = sum(1 for count in counts if count == 1)
        multi = sum(1 for count in counts if count > 1)

        top = sorted(
            clusters,
            key=lambda item: (
                int(item.get("observation_count") or 0),
                int(item.get("camera_count") or 0),
                str(item.get("last_seen") or ""),
            ),
            reverse=True,
        )[:20]

        with self._connect() as connection:
            diagnostic_rows = [
                dict(row)
                for row in connection.execute(
                    """
                    select o.id, o.embedding_blob
                    from face_observations o
                    join face_unknown_members m on m.observation_id = o.id
                    where o.embedding_blob is not null
                    """
                ).fetchall()
            ]
            membership = {
                int(row["observation_id"]): int(row["cluster_id"])
                for row in connection.execute(
                    "select observation_id, cluster_id from face_unknown_members"
                ).fetchall()
            }

        cohesion = unknown_cluster_cohesion(diagnostic_rows, membership)
        enriched_top = []
        suspicious_clusters = 0
        radius_floor = max(0.0, cluster_threshold - 0.08)
        for cluster in top:
            item = dict(cluster)
            metrics = cohesion.get(int(item.get("cluster_id") or 0), {})
            item.update(metrics)
            minimum_similarity = metrics.get("centroid_min_similarity")
            suspicious = bool(
                int(item.get("observation_count") or 0) >= 50
                or (
                    minimum_similarity is not None
                    and float(minimum_similarity) < radius_floor
                )
            )
            item["suspicious"] = suspicious
            suspicious_clusters += int(suspicious)
            enriched_top.append(item)

        return {
            "match_threshold": round(match_threshold, 4),
            "unknown_cluster_threshold": round(cluster_threshold, 4),
            "cluster_count": len(clusters),
            "clustered_observations": total_members,
            "singleton_clusters": singletons,
            "multi_observation_clusters": multi,
            "largest_cluster_size": counts[0] if counts else 0,
            "median_cluster_size": float(np.median(counts)) if counts else 0.0,
            "suspicious_top_clusters": suspicious_clusters,
            "cohesion": {
                "centroid_support_margin": 0.03,
                "radius_margin": 0.08,
                "large_cluster_growth_bonus": 0.03,
            },
            "top_clusters": enriched_top,
        }

    def unknown_clusters(self) -> list[dict[str, Any]]:
        self.refresh_unknown_clusters()
        with self._connect() as connection:
            rows = connection.execute(
                """
                select m.cluster_id, count(*) as observation_count,
                    min(o.observed_at) as first_seen,
                    max(o.observed_at) as last_seen,
                    count(distinct o.camera_id) as camera_count
                from face_unknown_members m
                join face_observations o on o.id = m.observation_id
                group by m.cluster_id
                order by last_seen desc, m.cluster_id
                """
            ).fetchall()
        return [
            {
                "cluster_id": int(row["cluster_id"]),
                "name": f"Unknown Person {int(row['cluster_id'])}",
                "observation_count": int(row["observation_count"] or 0),
                "camera_count": int(row["camera_count"] or 0),
                "first_seen": str(row["first_seen"] or ""),
                "last_seen": str(row["last_seen"] or ""),
            }
            for row in rows
        ]

    def for_event_ids(self, event_ids: list[int]) -> list[dict[str, Any]]:
        self.refresh_unknown_clusters()
        unique_ids = sorted({int(event_id) for event_id in event_ids if int(event_id) > 0})
        if not unique_ids:
            return []
        observations: list[dict[str, Any]] = []
        with self._connect() as connection:
            for offset in range(0, len(unique_ids), 500):
                chunk = unique_ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"""
                    select o.id as observation_id, o.event_id, o.person_id, o.confidence, o.match_confidence,
                        o.candidate_person_id, o.candidate_confidence,
                        o.candidate_track_id, o.consensus_json,
                        unknowns.cluster_id as unknown_cluster_id,
                        p.name as person_name, candidate.name as candidate_person_name
                    from face_observations o
                    left join face_people p on p.id = o.person_id
                    left join face_people candidate on candidate.id = o.candidate_person_id
                    left join face_unknown_members unknowns on unknowns.observation_id = o.id
                    where o.event_id in ({placeholders})
                        and o.canonical = 1
                    """,
                    chunk,
                ).fetchall()
                for row in rows:
                    item = dict(row)
                    try:
                        item["consensus"] = json.loads(item.pop("consensus_json") or "{}")
                    except (TypeError, json.JSONDecodeError):
                        item["consensus"] = {}
                    observations.append(item)
        return observations

    def bootstrap_person_references(
        self,
        person_id: int,
        *,
        seed_observation_id: int | None = None,
        target_count: int = 4,
    ) -> list[int]:
        """Auto-pin a small high-quality gallery from already confirmed faces."""
        target = max(1, min(int(target_count), 8))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                select id, camera_id, quality_score, observed_at,
                    reference_pinned, coalesce(reference_auto_pinned, 0) as reference_auto_pinned
                from face_observations
                where person_id = ? and review_status = 'confirmed' and canonical = 1
                order by observed_at desc, id desc
                """,
                (person_id,),
            ).fetchall()
            if not rows:
                return []

            explicit = [
                row for row in rows
                if bool(row["reference_pinned"]) and not bool(row["reference_auto_pinned"])
            ]
            selected_ids = {int(row["id"]) for row in explicit}
            seed = next(
                (
                    row for row in rows
                    if seed_observation_id is not None
                    and int(row["id"]) == int(seed_observation_id)
                ),
                None,
            )
            if seed is not None:
                selected_ids.add(int(seed["id"]))

            remaining = [row for row in rows if int(row["id"]) not in selected_ids]
            selected_cameras = {
                str(row["camera_id"] or "")
                for row in rows
                if int(row["id"]) in selected_ids
            }
            while remaining and len(selected_ids) < target:
                def utility(row):
                    quality = max(
                        0.0,
                        min(1.0, float(row["quality_score"] or 0.0)),
                    )
                    camera = str(row["camera_id"] or "")
                    novelty = 1.0 if camera and camera not in selected_cameras else 0.0
                    return (
                        0.85 * quality + 0.15 * novelty,
                        str(row["observed_at"] or ""),
                        int(row["id"]),
                    )

                chosen = max(remaining, key=utility)
                remaining.remove(chosen)
                selected_ids.add(int(chosen["id"]))
                selected_cameras.add(str(chosen["camera_id"] or ""))

            connection.execute(
                """
                update face_observations
                set reference_pinned = 0, reference_auto_pinned = 0
                where person_id = ? and reference_auto_pinned = 1
                """,
                (person_id,),
            )
            explicit_ids = {int(row["id"]) for row in explicit}
            for observation_id in sorted(selected_ids - explicit_ids):
                connection.execute(
                    """
                    update face_observations
                    set reference_pinned = 1, reference_auto_pinned = 1
                    where id = ? and person_id = ? and review_status = 'confirmed'
                    """,
                    (observation_id, person_id),
                )

        self._invalidate_reference_gallery()
        self.request_match_refresh()
        return sorted(selected_ids)

    def create_person(self, name: str, observation_id: int | None = None, notes: str = "") -> dict[str, Any]:
        name = name.strip()
        notes = notes.strip()
        if not name:
            raise ValueError("person name is required")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            if connection.execute(
                "select 1 from face_people where lower(name) = lower(?)",
                (name,),
            ).fetchone() is not None:
                raise ValueError("person name already exists")
            if observation_id is not None:
                observation = connection.execute(
                    "select person_id from face_observations where id = ?",
                    (observation_id,),
                ).fetchone()
                if observation is None:
                    raise ValueError("face observation not found")
                if observation["person_id"] is not None:
                    raise ValueError("face observation is already assigned")
            cursor = connection.execute(
                "insert into face_people (name, notes, created_at, updated_at) values (?, ?, ?, ?)",
                (name, notes, now, now),
            )
            person_id = int(cursor.lastrowid)
            if observation_id is not None:
                connection.execute(
                    """update face_observations set person_id = ?, review_status = 'confirmed',
                        match_confidence = 1, candidate_person_id = null, candidate_confidence = null,
                        rejected_person_id = null
                        where id = ?""",
                    (person_id, observation_id),
                )
                connection.execute(
                    "delete from face_rejections where observation_id = ?",
                    (observation_id,),
                )
        if observation_id is not None:
            self._invalidate_reference_gallery()
            self._queue_recognition(observation_id)
            self.bootstrap_person_references(
                person_id,
                seed_observation_id=observation_id,
            )
        return next(person for person in self.people() if person["id"] == person_id)

    def assign(self, observation_id: int, person_id: int | None) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            if person_id is not None and connection.execute("select 1 from face_people where id = ?", (person_id,)).fetchone() is None:
                raise ValueError("person not found")
            current = connection.execute(
                "select person_id, candidate_person_id from face_observations where id = ?",
                (observation_id,),
            ).fetchone()
            if current is None:
                return None
            rejected_person_id = (
                int(current["candidate_person_id"])
                if person_id is None and current["candidate_person_id"] is not None
                else None
            )
            if person_id is not None:
                connection.execute(
                    "delete from face_rejections where observation_id = ?",
                    (observation_id,),
                )
            elif rejected_person_id is not None:
                connection.execute(
                    """
                    insert or ignore into face_rejections (observation_id, person_id, created_at)
                    values (?, ?, ?)
                    """,
                    (observation_id, rejected_person_id, datetime.now(timezone.utc).isoformat()),
                )
            connection.execute(
                """update face_observations set person_id = ?, review_status = ?, match_confidence = ?,
                    candidate_person_id = null, candidate_confidence = null, rejected_person_id = ?,
                    auto_identified = 0, reference_pinned = 0
                    where id = ?""",
                (
                    person_id,
                    "confirmed" if person_id is not None else "rejected" if rejected_person_id is not None else "unknown",
                    1 if person_id is not None else None,
                    rejected_person_id,
                    observation_id,
                ),
            )
        if current["person_id"] is not None or person_id is not None:
            self._invalidate_reference_gallery()
        if person_id is not None:
            self._queue_recognition(observation_id)
            self.bootstrap_person_references(
                person_id,
                seed_observation_id=observation_id,
            )
        else:
            self.request_match_refresh()
        return self.observation(observation_id)

    def set_reference_pinned(
        self,
        observation_id: int,
        pinned: bool,
    ) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            connection.execute("begin immediate")
            row = connection.execute(
                "select person_id, review_status, snapshot_path "
                "from face_observations where id = ?",
                (observation_id,),
            ).fetchone()
            if row is None:
                return None
            if row["person_id"] is None or row["review_status"] != "confirmed":
                raise ValueError("only manually confirmed faces can be pinned as references")
            if pinned and row["snapshot_path"]:
                deleting = connection.execute(
                    "select 1 from media_deletion_claims where path = ?",
                    (str(row["snapshot_path"]),),
                ).fetchone()
                if deleting is not None:
                    raise RuntimeError("face snapshot is currently being removed")
            connection.execute(
                "update face_observations set reference_pinned = ?, reference_auto_pinned = 0 where id = ?",
                (1 if pinned else 0, observation_id),
            )
        self._invalidate_reference_gallery()
        self.request_match_refresh()
        return self.observation(observation_id)

    def delete_person(self, person_id: int) -> bool:
        with self._lock, self._connect() as connection:
            connection.execute(
                "update face_observations set candidate_person_id = null, candidate_confidence = null where candidate_person_id = ?",
                (person_id,),
            )
            connection.execute(
                "update face_observations set rejected_person_id = null where rejected_person_id = ?",
                (person_id,),
            )
            connection.execute(
                """
                update face_observations
                set person_id = null, review_status = 'unknown', match_confidence = null,
                    auto_identified = 0, reference_pinned = 0
                where person_id = ?
                """,
                (person_id,),
            )
            cursor = connection.execute("delete from face_people where id = ?", (person_id,))
        deleted = cursor.rowcount > 0
        if deleted:
            self._invalidate_reference_gallery()
            self.request_match_refresh()
        return deleted

    def snapshot_path(self, observation_id: int) -> tuple[Path, dict[str, float]] | None:
        observation = self.observation(observation_id)
        if not observation:
            return None
        try:
            path = event_snapshot_path(self.storage_dir, observation, self.media_storage)
        except (FileNotFoundError, PermissionError, OSError, RuntimeError):
            return None
        box = parse_face_box(observation["box"])
        if box is None:
            return None
        return path, box

    @staticmethod
    def _observation_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item.pop("embedding_blob", None)
        try:
            item["box"] = parse_face_box(json.loads(item.pop("box_json"))) or {}
        except (TypeError, json.JSONDecodeError):
            item["box"] = {}
        for source, target in (
            ("quality_json", "quality"),
            ("match_details_json", "match_details"),
            ("consensus_json", "consensus"),
        ):
            try:
                item[target] = json.loads(item.pop(source) or "{}")
            except (TypeError, json.JSONDecodeError):
                item[target] = {}
        item["reference_pinned"] = bool(item.get("reference_pinned"))
        item["reference_auto_pinned"] = bool(item.get("reference_auto_pinned"))
        item["auto_identified"] = bool(item.get("auto_identified"))
        return item

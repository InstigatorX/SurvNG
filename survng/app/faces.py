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


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceQuality:
    score: float
    sharpness: float
    exposure: float
    contrast: float
    size: float


@dataclass(frozen=True, slots=True)
class FaceMatch:
    person_id: int | None
    score: float | None
    runner_up_score: float | None
    margin: float | None
    reference_ids: tuple[int, ...]
    reference_scores: tuple[float, ...]


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
        return FaceQuality(0.0, 0.0, 0.0, 0.0, 0.0)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face
    height, width = gray.shape[:2]
    if max(height, width) > 256:
        scale = 256.0 / max(height, width)
        gray = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    pixels = gray.astype(np.float32, copy=False)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness = max(0.0, min(1.0, math.log1p(laplacian_variance) / math.log1p(1200.0)))
    clipped = float(np.mean((pixels <= 4.0) | (pixels >= 251.0)))
    exposure = max(0.0, min(1.0, 1.0 - clipped))
    contrast = max(0.0, min(1.0, float(pixels.std()) / 64.0))
    size = max(0.0, min(1.0, min(height, width) / 160.0))
    confidence = max(0.0, min(1.0, float(detector_confidence)))
    score = (
        0.35 * sharpness
        + 0.20 * exposure
        + 0.15 * contrast
        + 0.20 * size
        + 0.10 * confidence
    )
    return FaceQuality(
        round(score, 4),
        round(sharpness, 4),
        round(exposure, 4),
        round(contrast, 4),
        round(size, 4),
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
    ) -> None:
        self.storage_dir = storage_dir.resolve()
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
                "quality_score": "alter table face_observations add column quality_score real",
                "quality_json": "alter table face_observations add column quality_json text not null default '{}'",
                "reference_pinned": "alter table face_observations add column reference_pinned integer not null default 0",
                "auto_identified": "alter table face_observations add column auto_identified integer not null default 0",
                "match_details_json": "alter table face_observations add column match_details_json text not null default '{}'",
            }
            for name, statement in migrations.items():
                if name not in columns:
                    connection.execute(statement)
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
                        event_snapshot_path(self.storage_dir, event),
                    )
                except (FileNotFoundError, PermissionError, OSError, RuntimeError):
                    continue
                try:
                    objects = json.loads(event.get("objects_json", "[]") or "[]")
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(objects, list):
                    continue
                for object_index, detected in enumerate(objects):
                    if not isinstance(detected, dict):
                        continue
                    if str(detected.get("label") or "").lower() != "face":
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
                    sum(case when candidate_person_id is not null and person_id is null then 1 else 0 end) as suggested,
                    sum(case when recognition_error != '' then 1 else 0 end) as failed,
                    sum(case when recognition_pending = 1 then 1 else 0 end) as pending
                from face_observations
                """,
                (str(recognizer_status.get("model_fingerprint") or ""),),
            ).fetchone()
        return {
            **recognizer_status,
            "queue_depth": self._recognition_queue.qsize(),
            "embedded": int(row["embedded_current"] or 0),
            "suggested": int(row["suggested"] or 0),
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
                if references_changed and self._try_refresh_unknown_recognition():
                    references_changed = False
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
            )
            frame = cv2.imread(str(snapshot_path))
            if frame is None:
                raise ValueError("Snapshot is unavailable.")
            x1, y1 = float(box.get("x1", 0)), float(box.get("y1", 0))
            x2, y2 = float(box.get("x2", 0)), float(box.get("y2", 0))
            face_width, face_height = x2 - x1, y2 - y1
            if min(face_width, face_height) < recognizer.config.face_min_size:
                raise ValueError(f"Face is smaller than {recognizer.config.face_min_size}px.")
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
                        quality_score = ?, quality_json = ?, match_details_json = ?,
                        recognition_error = '', recognized_at = ?, recognition_pending = 0
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
                        details,
                        now,
                        observation_id,
                    ),
                )
                current = connection.execute(
                    "select person_id from face_observations where id = ?",
                    (observation_id,),
                ).fetchone()
            if row["person_id"] is not None:
                self._invalidate_reference_gallery()
            return current is not None and current["person_id"] is not None
        except InferenceUnavailable:
            raise
        except Exception as exc:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    update face_observations
                    set recognition_error = ?, recognized_at = ?, recognition_pending = 0
                    where id = ?
                    """,
                    (str(exc)[:500], datetime.now(timezone.utc).isoformat(), observation_id),
                )
            return False

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
        total = int(connection.execute("select count(*) from face_observations").fetchone()[0])
        excess = total - self.max_observations
        if excess <= 0:
            return 0
        remove_ids = [
            int(row["id"])
            for row in connection.execute(
                """
                select id from face_observations
                where reference_pinned = 0 and id not in (
                    select id from (
                        select id, row_number() over (
                            partition by person_id order by observed_at desc, id desc
                        ) as position
                        from face_observations
                        where person_id is not null
                    ) where position = 1
                )
                order by observed_at asc, id asc
                limit ?
                """,
                (excess,),
            ).fetchall()
        ]
        if remove_ids:
            connection.executemany("delete from face_observations where id = ?", ((item,) for item in remove_ids))
            self._invalidate_reference_gallery()
        return len(remove_ids)

    def people(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select p.*, count(o.id) as observation_count,
                    sum(case when o.review_status = 'confirmed' then 1 else 0 end) as reference_count,
                    sum(case when o.review_status = 'confirmed' and o.reference_pinned = 1 then 1 else 0 end) as pinned_reference_count,
                    avg(case when o.review_status = 'confirmed' then o.quality_score end) as average_reference_quality,
                    max(o.observed_at) as last_seen_at,
                    (select id from face_observations latest
                     where latest.person_id = p.id
                     order by latest.observed_at desc limit 1) as preview_observation_id
                from face_people p
                left join face_observations o on o.person_id = p.id
                group by p.id
                order by lower(p.name)
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            row = connection.execute(
                """
                select count(*) as observations,
                    sum(case when person_id is null then 1 else 0 end) as unknown,
                    sum(case when person_id is not null then 1 else 0 end) as known,
                    sum(case when candidate_person_id is not null and person_id is null then 1 else 0 end) as suggested
                from face_observations
                """
            ).fetchone()
            people = connection.execute("select count(*) from face_people").fetchone()[0]
        return {
            "people": int(people or 0),
            "observations": int(row["observations"] or 0),
            "unknown": int(row["unknown"] or 0),
            "known": int(row["known"] or 0),
            "suggested": int(row["suggested"] or 0),
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
        true_scores: list[float] = []
        impostor_scores: list[float] = []
        margins: list[float] = []
        rank_one = 0
        for observation_id, person_id, target in embeddings:
            grouped: dict[int, list[float]] = {}
            for reference_id, reference_person_id, reference in embeddings:
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
        for _observation_id, person_id, vector in embeddings:
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
                + (" and explicit rejected matches." if rejected_scores else "; confirm or reject more suggestions to strengthen calibration.")
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
        clauses: list[str] = []
        values: list[Any] = []
        if person_id is not None:
            clauses.append("o.person_id = ?")
            values.append(person_id)
        elif status == "unknown":
            clauses.append("o.person_id is null")
        elif status == "known":
            clauses.append("o.person_id is not null")
        elif status == "suggested":
            clauses.append("o.person_id is null and o.candidate_person_id is not null")
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
        clauses: list[str] = []
        values: list[Any] = []
        if person_id is not None:
            clauses.append("person_id = ?")
            values.append(person_id)
        elif status == "unknown":
            clauses.append("person_id is null")
        elif status == "known":
            clauses.append("person_id is not null")
        elif status == "suggested":
            clauses.append("person_id is null and candidate_person_id is not null")
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

    def for_event_ids(self, event_ids: list[int]) -> list[dict[str, Any]]:
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
                        p.name as person_name, candidate.name as candidate_person_name
                    from face_observations o
                    left join face_people p on p.id = o.person_id
                    left join face_people candidate on candidate.id = o.candidate_person_id
                    where o.event_id in ({placeholders})
                    """,
                    chunk,
                ).fetchall()
                observations.extend(dict(row) for row in rows)
        return observations

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
            self._try_refresh_unknown_recognition()
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
        self._try_refresh_unknown_recognition()
        return self.observation(observation_id)

    def set_reference_pinned(
        self,
        observation_id: int,
        pinned: bool,
    ) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "select person_id, review_status from face_observations where id = ?",
                (observation_id,),
            ).fetchone()
            if row is None:
                return None
            if row["person_id"] is None or row["review_status"] != "confirmed":
                raise ValueError("only manually confirmed faces can be pinned as references")
            connection.execute(
                "update face_observations set reference_pinned = ? where id = ?",
                (1 if pinned else 0, observation_id),
            )
        self._invalidate_reference_gallery()
        self._try_refresh_unknown_recognition()
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
            self._try_refresh_unknown_recognition()
        return deleted

    def snapshot_path(self, observation_id: int) -> tuple[Path, dict[str, float]] | None:
        observation = self.observation(observation_id)
        if not observation:
            return None
        try:
            path = event_snapshot_path(self.storage_dir, observation)
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
        ):
            try:
                item[target] = json.loads(item.pop(source) or "{}")
            except (TypeError, json.JSONDecodeError):
                item[target] = {}
        item["reference_pinned"] = bool(item.get("reference_pinned"))
        item["auto_identified"] = bool(item.get("auto_identified"))
        return item

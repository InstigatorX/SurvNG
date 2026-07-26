from __future__ import annotations

import json
import logging
import math
from queue import Empty, Full, Queue
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .face_recognition import OpenVinoFaceRecognizer
from .inference import INFERENCE_REQUEST_TIMEOUT_SECONDS, InferenceUnavailable
from .incident_utils import event_snapshot_path


LOGGER = logging.getLogger(__name__)


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
    ) -> None:
        self.storage_dir = storage_dir.resolve()
        self.db_path = self.storage_dir / "survng.sqlite3"
        self.max_observations = max(100, int(max_observations))
        self.recognizer = recognizer
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._recognition_queue: Queue[int | None] = Queue(maxsize=self.max_observations + 1)
        self._recognition_pending: set[int] = set()
        self._recognition_pending_lock = threading.Lock()
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
            storage_root = str(self.storage_dir)
            metadata = connection.execute(
                "select value from survng_metadata where key = 'face_storage_root'"
            ).fetchone()
            if metadata is None or str(metadata[0]) != storage_root:
                rows = connection.execute("select id, snapshot_path from face_observations").fetchall()
                updates = []
                for row in rows:
                    raw_path = str(row["snapshot_path"] or "")
                    parts = Path(raw_path).parts
                    if "snapshots" not in parts:
                        continue
                    candidate = self.storage_dir.joinpath(*parts[parts.index("snapshots"):])
                    if str(candidate) != raw_path and candidate.is_file():
                        updates.append((str(candidate), int(row["id"])))
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
                    snapshot_path = str(event_snapshot_path(self.storage_dir, event))
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
                candidate_id, candidate_confidence = self._best_match(
                    connection,
                    int(row["id"]),
                    embedding / norm,
                    model_fingerprint,
                )
                connection.execute(
                    """
                    update face_observations
                    set candidate_person_id = ?, candidate_confidence = ?
                    where id = ? and person_id is null
                    """,
                    (candidate_id, candidate_confidence, int(row["id"])),
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

    def _recognize_observation(self, observation_id: int) -> bool:
        recognizer = self.recognizer
        if recognizer is None:
            return False
        recognizer_status = recognizer.status()
        if not recognizer_status.get("ready"):
            isolation = recognizer_status.get("isolation") or {}
            if recognizer.enabled and not isolation.get("worker_alive", True):
                raise InferenceUnavailable(
                    str(isolation.get("last_error") or "face inference worker is unavailable")
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
            height, width = frame.shape[:2]
            x1, y1 = float(box.get("x1", 0)), float(box.get("y1", 0))
            x2, y2 = float(box.get("x2", 0)), float(box.get("y2", 0))
            face_width, face_height = x2 - x1, y2 - y1
            if min(face_width, face_height) < recognizer.config.face_min_size:
                raise ValueError(f"Face is smaller than {recognizer.config.face_min_size}px.")
            pad_x, pad_y = face_width * 0.12, face_height * 0.12
            left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
            right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
            if right <= left or bottom <= top:
                raise ValueError("Face crop is invalid.")
            embedding = np.asarray(
                recognizer.embed(frame[top:bottom, left:right]),
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
                candidate_id, candidate_confidence = self._best_match(
                    connection,
                    observation_id,
                    embedding,
                    model_fingerprint,
                )
                connection.execute(
                    """
                    update face_observations
                    set embedding_blob = ?, embedding_model = ?,
                        candidate_person_id = case when person_id is null then ? else null end,
                        candidate_confidence = case when person_id is null then ? else null end,
                        recognition_error = '', recognized_at = ?, recognition_pending = 0
                    where id = ?
                    """,
                    (
                        embedding.astype(np.float32).tobytes(),
                        model_fingerprint,
                        candidate_id,
                        candidate_confidence,
                        now,
                        observation_id,
                    ),
                )
                current = connection.execute(
                    "select person_id from face_observations where id = ?",
                    (observation_id,),
                ).fetchone()
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
        recognizer = self.recognizer
        if recognizer is None:
            return None, None
        if embedding.ndim != 1 or embedding.size == 0 or not np.all(np.isfinite(embedding)):
            return None, None
        rows = connection.execute(
            """
            select id, person_id, embedding_blob from (
                select id, person_id, embedding_blob, observed_at,
                    row_number() over (
                        partition by person_id order by observed_at desc, id desc
                    ) as reference_position
                from face_observations
                where person_id is not null and embedding_blob is not null
                    and embedding_model = ? and id != ?
            ) where reference_position <= ?
            order by observed_at desc, id desc
            """,
            (
                model_fingerprint,
                observation_id,
                recognizer.config.face_max_references,
            ),
        ).fetchall()
        rejected_people = {
            int(row["person_id"])
            for row in connection.execute(
                "select person_id from face_rejections where observation_id = ?",
                (observation_id,),
            ).fetchall()
        }
        scores: dict[int, list[float]] = {}
        for row in rows:
            person_scores = scores.setdefault(int(row["person_id"]), [])
            try:
                reference = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if reference.shape != embedding.shape or not np.all(np.isfinite(reference)):
                continue
            reference_norm = float(np.linalg.norm(reference))
            if not math.isfinite(reference_norm) or reference_norm <= 1e-9:
                continue
            score = float(np.dot(embedding, reference / reference_norm))
            if math.isfinite(score):
                person_scores.append(score)
        ranked: list[tuple[float, int]] = []
        for person_id, values in scores.items():
            if person_id in rejected_people:
                continue
            top = sorted(values, reverse=True)[:3]
            ranked.append((float(sum(top) / len(top)), person_id))
        if not ranked:
            return None, None
        score, person_id = max(ranked)
        score = max(0.0, min(1.0, score))
        if score < recognizer.config.face_match_threshold:
            return None, round(score, 4)
        return person_id, round(score, 4)

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
                where id not in (
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
        return len(remove_ids)

    def people(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select p.*, count(o.id) as observation_count,
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
            self._queue_recognition(observation_id)
            self._try_refresh_unknown_recognition()
        return next(person for person in self.people() if person["id"] == person_id)

    def assign(self, observation_id: int, person_id: int | None) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            if person_id is not None and connection.execute("select 1 from face_people where id = ?", (person_id,)).fetchone() is None:
                raise ValueError("person not found")
            current = connection.execute(
                "select candidate_person_id from face_observations where id = ?", (observation_id,)
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
                    candidate_person_id = null, candidate_confidence = null, rejected_person_id = ?
                    where id = ?""",
                (
                    person_id,
                    "confirmed" if person_id is not None else "rejected" if rejected_person_id is not None else "unknown",
                    1 if person_id is not None else None,
                    rejected_person_id,
                    observation_id,
                ),
            )
        if person_id is not None:
            self._queue_recognition(observation_id)
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
                set person_id = null, review_status = 'unknown', match_confidence = null
                where person_id = ?
                """,
                (person_id,),
            )
            cursor = connection.execute("delete from face_people where id = ?", (person_id,))
        deleted = cursor.rowcount > 0
        if deleted:
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
        return item

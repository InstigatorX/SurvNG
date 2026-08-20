from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from queue import Empty, Full
from typing import Any

import cv2
import numpy as np

from ..face_recognition import OpenVinoFaceRecognizer
from ..inference import InferenceUnavailable
from ..incident_utils import event_snapshot_path
from .quality import (
    FACE_OUTCOME_EMBEDDED,
    FACE_OUTCOME_FAILED,
    FACE_OUTCOME_PENDING,
    FACE_OUTCOME_TOO_SMALL,
    FACE_QUALITY_VERSION,
    FaceMatch,
    FaceTooSmallError,
    LOGGER,
    _face_crop,
    _face_quality,
    parse_face_box,
)


class FaceStoreRecognitionMixin:
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
                    "select person_id, review_status from face_observations where id = ?",
                    (observation_id,),
                ).fetchone()
                track_id = str(row["candidate_track_id"] or "")
                if track_id:
                    self._reconcile_candidate_track(
                        connection,
                        int(row["event_id"]),
                        track_id,
                    )
            if (
                row["person_id"] is None
                and current is not None
                and current["person_id"] is not None
            ):
                source = (
                    "auto_recognition"
                    if str(current["review_status"] or "") == "auto_identified"
                    else "recognition"
                )
                self._emit_identity_update(observation_id, source=source)
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

    def _invalidate_reference_gallery(self) -> None:
        with self._gallery_lock:
            self._gallery_generation += 1
            self._gallery_cache_key = None
            self._gallery_cache = []

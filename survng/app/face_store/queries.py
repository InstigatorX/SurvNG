from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..incident_utils import event_snapshot_path
from .quality import (
    FACE_OUTCOME_EMBEDDED,
    FACE_OUTCOME_FAILED,
    FACE_OUTCOME_TOO_SMALL,
    parse_face_box,
)


class FaceStoreQueryMixin:
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

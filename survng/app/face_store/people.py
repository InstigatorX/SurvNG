from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from .quality import LOGGER


class FaceStorePeopleMixin:
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

    def set_identity_event_publisher(
        self,
        publisher: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        self._identity_event_publisher = publisher

    def _emit_identity_update(
        self,
        observation_id: int,
        *,
        source: str,
    ) -> None:
        publisher = getattr(self, "_identity_event_publisher", None)
        if not callable(publisher):
            return
        observation = self.observation(observation_id)
        if not observation or observation.get("person_id") is None:
            return
        payload = {
            "type": "identity_update",
            "source": str(source or "recognition"),
            "observation_id": int(observation_id),
            "event_id": int(observation.get("event_id") or 0),
            "camera_id": str(observation.get("camera_id") or ""),
            "person_id": int(observation["person_id"]),
            "name": str(observation.get("person_name") or ""),
            "confidence": (
                round(float(observation["match_confidence"]), 4)
                if observation.get("match_confidence") is not None
                else None
            ),
            "review_status": str(observation.get("review_status") or ""),
            "observed_at": str(observation.get("observed_at") or ""),
        }
        try:
            publisher(payload)
        except Exception:
            LOGGER.exception(
                "identity update publisher failed for observation %s",
                observation_id,
            )

    def optimize_person_gallery(
        self,
        person_id: int,
        *,
        max_references: int = 8,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Greedily choose confirmed references that improve held-out rank/margin."""
        recognizer = self.recognizer
        if recognizer is None:
            raise RuntimeError("face recognizer is not configured")
        status = recognizer.status()
        model_fingerprint = str(status.get("model_fingerprint") or "")
        if not model_fingerprint:
            raise RuntimeError("face embedding model is not ready")

        limit = max(2, min(int(max_references), 12))
        with self._connect() as connection:
            person = connection.execute(
                "select id, name from face_people where id = ?",
                (int(person_id),),
            ).fetchone()
            if person is None:
                raise ValueError("person not found")

            rows = connection.execute(
                """
                select o.id, o.camera_id, o.quality_score, o.embedding_blob,
                    o.reference_pinned, o.reference_auto_pinned
                from face_observations o
                where o.canonical = 1
                    and o.person_id = ?
                    and o.review_status = 'confirmed'
                    and o.embedding_model = ?
                    and o.embedding_blob is not null
                order by o.quality_score desc, o.id
                """,
                (int(person_id), model_fingerprint),
            ).fetchall()

            competitors = connection.execute(
                """
                select o.id, o.person_id, o.embedding_blob
                from face_observations o
                where o.canonical = 1
                    and o.person_id is not null
                    and o.person_id != ?
                    and o.review_status = 'confirmed'
                    and o.embedding_model = ?
                    and o.embedding_blob is not null
                """,
                (int(person_id), model_fingerprint),
            ).fetchall()

        def normalized(row):
            vector = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if not vector.size or not math.isfinite(norm) or norm <= 1e-9:
                return None
            return vector / norm

        samples = []
        for row in rows:
            vector = normalized(row)
            if vector is not None:
                samples.append((row, vector))
        others = []
        for row in competitors:
            vector = normalized(row)
            if vector is not None:
                others.append((row, vector))

        if len(samples) < 2:
            return {
                "person_id": int(person["id"]),
                "name": str(person["name"] or ""),
                "sample_count": len(samples),
                "applied": False,
                "reason": "not_enough_samples",
            }

        def aggregate(query, refs):
            scores = sorted(
                (float(np.dot(query, ref_vector)) for _ref_row, ref_vector in refs),
                reverse=True,
            )
            if not scores:
                return float("-inf")
            top = scores[:3]
            return sum(top) / len(top)

        competitor_by_person = {}
        for row, vector in others:
            competitor_by_person.setdefault(int(row["person_id"]), []).append((row, vector))

        def evaluate(reference_ids):
            ref_set = set(reference_ids)
            refs = [(row, vector) for row, vector in samples if int(row["id"]) in ref_set]
            held_out = [(row, vector) for row, vector in samples if int(row["id"]) not in ref_set]
            if not refs or not held_out:
                return {
                    "trials": 0,
                    "rank_one_accuracy": 0.0,
                    "median_margin": None,
                    "median_true_score": None,
                }

            rank_one = 0
            margins = []
            true_scores = []
            for _row, query in held_out:
                true_score = aggregate(query, refs)
                wrong_score = max(
                    (
                        aggregate(query, competitor_refs)
                        for competitor_refs in competitor_by_person.values()
                        if competitor_refs
                    ),
                    default=float("-inf"),
                )
                true_scores.append(true_score)
                margins.append(true_score - wrong_score)
                rank_one += int(true_score > wrong_score)

            return {
                "trials": len(held_out),
                "rank_one_accuracy": round(rank_one / len(held_out), 4),
                "median_margin": round(float(np.median(margins)), 4),
                "median_true_score": round(float(np.median(true_scores)), 4),
            }

        current_ids = [
            int(row["id"])
            for row, _vector in samples
            if bool(row["reference_pinned"])
        ]
        if not current_ids:
            current_ids = [int(samples[0][0]["id"])]

        current_ids = current_ids[:limit]
        baseline = evaluate(current_ids)

        selected = list(current_ids)
        candidate_ids = [
            int(row["id"])
            for row, _vector in samples
            if int(row["id"]) not in selected
        ]

        def objective(metrics):
            return (
                float(metrics["rank_one_accuracy"]),
                float(metrics["median_margin"] if metrics["median_margin"] is not None else -999),
                float(metrics["median_true_score"] if metrics["median_true_score"] is not None else -999),
            )

        best_metrics = baseline
        while len(selected) < limit and candidate_ids:
            best_candidate = None
            best_candidate_metrics = None
            for candidate_id in candidate_ids:
                metrics = evaluate([*selected, candidate_id])
                if best_candidate_metrics is None or objective(metrics) > objective(best_candidate_metrics):
                    best_candidate = candidate_id
                    best_candidate_metrics = metrics

            if best_candidate is None or best_candidate_metrics is None:
                break
            if objective(best_candidate_metrics) <= objective(best_metrics):
                break

            selected.append(best_candidate)
            candidate_ids.remove(best_candidate)
            best_metrics = best_candidate_metrics

        improved = objective(best_metrics) > objective(baseline)

        if apply and improved:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    update face_observations
                    set reference_pinned = 0,
                        reference_auto_pinned = 0
                    where person_id = ?
                        and canonical = 1
                        and review_status = 'confirmed'
                        and reference_auto_pinned = 1
                    """,
                    (int(person_id),),
                )
                connection.executemany(
                    """
                    update face_observations
                    set reference_pinned = 1,
                        reference_auto_pinned = 1
                    where id = ?
                        and person_id = ?
                        and canonical = 1
                        and review_status = 'confirmed'
                    """,
                    ((observation_id, int(person_id)) for observation_id in selected),
                )
            self._invalidate_reference_gallery()
            self.request_match_refresh()

        return {
            "person_id": int(person["id"]),
            "name": str(person["name"] or ""),
            "sample_count": len(samples),
            "baseline_reference_ids": current_ids,
            "optimized_reference_ids": selected,
            "baseline": baseline,
            "optimized": best_metrics,
            "improved": improved,
            "applied": bool(apply and improved),
        }

    def optimize_all_galleries(
        self,
        *,
        max_references: int = 8,
        apply: bool = False,
    ) -> list[dict[str, Any]]:
        return [
            self.optimize_person_gallery(
                int(person["id"]),
                max_references=max_references,
                apply=apply,
            )
            for person in self.people()
        ]

    def person_representation(
        self,
        person_id: int,
    ) -> dict[str, Any]:
        recognizer = self.recognizer
        if recognizer is None:
            raise RuntimeError("face recognizer is not configured")
        status = recognizer.status()
        model_fingerprint = str(status.get("model_fingerprint") or "")
        if not model_fingerprint:
            raise RuntimeError("face embedding model is not ready")

        with self._connect() as connection:
            person = connection.execute(
                "select id, name from face_people where id = ?",
                (int(person_id),),
            ).fetchone()
            if person is None:
                raise ValueError("person not found")
            rows = connection.execute(
                """
                select id, camera_id, observed_at, quality_score,
                    reference_pinned, reference_auto_pinned,
                    embedding_blob, match_details_json
                from face_observations
                where canonical = 1
                    and person_id = ?
                    and review_status = 'confirmed'
                    and embedding_model = ?
                    and embedding_blob is not null
                order by observed_at asc, id asc
                """,
                (int(person_id), model_fingerprint),
            ).fetchall()
            other_rows = connection.execute(
                """
                select o.id, o.person_id, p.name as person_name,
                    o.embedding_blob
                from face_observations o
                join face_people p on p.id = o.person_id
                where o.canonical = 1
                    and o.person_id is not null
                    and o.person_id != ?
                    and o.review_status = 'confirmed'
                    and o.embedding_model = ?
                    and o.embedding_blob is not null
                """,
                (int(person_id), model_fingerprint),
            ).fetchall()

        vectors = []
        for row in rows:
            vector = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.size and math.isfinite(norm) and norm > 1e-9:
                vectors.append((row, vector / norm))

        other_vectors = []
        for row in other_rows:
            vector = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.size and math.isfinite(norm) and norm > 1e-9:
                other_vectors.append((row, vector / norm))

        same_scores = []
        for index, (_row, vector) in enumerate(vectors):
            for _other_row, other in vectors[index + 1:]:
                same_scores.append(float(np.dot(vector, other)))

        competitor_scores = {}
        competitor_names = {}
        for _row, vector in vectors:
            for other_row, other in other_vectors:
                competitor_id = int(other_row["person_id"])
                competitor_names[competitor_id] = str(other_row["person_name"] or "")
                competitor_scores.setdefault(competitor_id, []).append(
                    float(np.dot(vector, other))
                )

        def percentile(values, q):
            if not values:
                return None
            return round(float(np.percentile(np.asarray(values, dtype=np.float32), q)), 4)

        nearest_id = None
        nearest_name = ""
        nearest_score = None
        if competitor_scores:
            nearest_id, nearest_values = max(
                competitor_scores.items(),
                key=lambda item: max(item[1]),
            )
            nearest_name = competitor_names.get(nearest_id, "")
            nearest_score = round(max(nearest_values), 4)

        camera_counts = {}
        pinned = 0
        auto_pinned = 0
        model_scores = []
        for row, _vector in vectors:
            camera = str(row["camera_id"] or "")
            camera_counts[camera] = camera_counts.get(camera, 0) + 1
            pinned += int(bool(row["reference_pinned"]))
            auto_pinned += int(bool(row["reference_auto_pinned"]))
            try:
                details = json.loads(str(row["match_details_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                details = {}
            if isinstance(details, dict):
                try:
                    score = float(details.get("score"))
                except (TypeError, ValueError):
                    score = float("nan")
                if math.isfinite(score):
                    model_scores.append(score)

        same_median = percentile(same_scores, 50)
        separation = None
        if same_median is not None and nearest_score is not None:
            separation = round(same_median - nearest_score, 4)

        flags = []
        if len(vectors) < 4:
            flags.append("low_sample_count")
        if len(camera_counts) < 2 and len(vectors) >= 4:
            flags.append("single_camera_gallery")
        if same_median is not None and same_median < 0.20:
            flags.append("weak_within_identity_cohesion")
        if separation is not None and separation < 0.05:
            flags.append("weak_identity_separation")
        if pinned < min(4, len(vectors)):
            flags.append("under_pinned_gallery")

        return {
            "person_id": int(person["id"]),
            "name": str(person["name"] or ""),
            "sample_count": len(vectors),
            "camera_count": len(camera_counts),
            "camera_counts": dict(sorted(camera_counts.items())),
            "pinned_references": pinned,
            "auto_pinned_references": auto_pinned,
            "same_person": {
                "pairs": len(same_scores),
                "p05": percentile(same_scores, 5),
                "median": same_median,
                "p95": percentile(same_scores, 95),
            },
            "nearest_competitor": (
                {
                    "person_id": int(nearest_id),
                    "name": nearest_name,
                    "maximum_similarity": nearest_score,
                }
                if nearest_id is not None
                else None
            ),
            "separation": separation,
            "model_score": {
                "count": len(model_scores),
                "p05": percentile(model_scores, 5),
                "median": percentile(model_scores, 50),
                "p95": percentile(model_scores, 95),
            },
            "flags": flags,
        }

    def gallery_candidates(self, person_id: int, *, limit: int = 20) -> list[dict[str, Any]]:
        resolved_limit = max(1, min(int(limit), 100))
        with self._connect() as connection:
            person = connection.execute(
                "select id from face_people where id = ?", (int(person_id),)
            ).fetchone()
            if person is None:
                raise ValueError("person not found")
            rows = connection.execute(
                """
                select o.*, p.name as person_name,
                    candidate.name as candidate_person_name
                from face_observations o
                join face_people p on p.id = o.person_id
                left join face_people candidate on candidate.id = o.candidate_person_id
                where o.canonical = 1
                    and o.person_id = ?
                    and o.review_status = 'confirmed'
                    and o.embedding_blob is not null
                order by o.quality_score desc, o.observed_at desc, o.id desc
                """,
                (int(person_id),),
            ).fetchall()

        normalized = []
        for row in rows:
            vector = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.size and math.isfinite(norm) and norm > 1e-9:
                normalized.append((row, vector / norm))

        pinned = [(row, vector) for row, vector in normalized if bool(row["reference_pinned"])]
        pinned_cameras = {str(row["camera_id"] or "") for row, _vector in pinned}
        candidates = []
        for row, vector in normalized:
            if bool(row["reference_pinned"]):
                continue
            quality = max(0.0, min(1.0, float(row["quality_score"] or 0.0)))
            camera = str(row["camera_id"] or "")
            camera_novelty = 1.0 if camera and camera not in pinned_cameras else 0.0
            max_similarity = 0.0
            if pinned:
                max_similarity = max(
                    float(np.dot(vector, ref_vector))
                    for _ref_row, ref_vector in pinned
                )
            diversity = 1.0 - max(-1.0, min(1.0, max_similarity)) if pinned else 1.0
            diversity = max(0.0, min(1.0, diversity))
            score = 0.55 * quality + 0.30 * camera_novelty + 0.15 * diversity
            item = self._observation_row(row)
            item["gallery_candidate_score"] = round(score, 4)
            item["gallery_candidate_reasons"] = [
                reason for reason, enabled in (
                    ("high_quality", quality >= 0.70),
                    ("new_camera", camera_novelty > 0),
                    ("diverse_embedding", diversity >= 0.30),
                ) if enabled
            ]
            item["max_similarity_to_pinned"] = round(max_similarity, 4) if pinned else None
            candidates.append(item)

        candidates.sort(
            key=lambda item: (
                float(item.get("gallery_candidate_score") or 0.0),
                float(item.get("quality_score") or 0.0),
                int(item.get("id") or 0),
            ),
            reverse=True,
        )
        return candidates[:resolved_limit]

    def enrich_person_gallery(self, person_id: int, *, target_count: int = 8) -> dict[str, Any]:
        target = max(1, min(int(target_count), 20))
        with self._connect() as connection:
            exists = connection.execute(
                "select 1 from face_people where id = ?", (int(person_id),)
            ).fetchone()
            if exists is None:
                raise ValueError("person not found")
            before = int(connection.execute(
                """
                select count(*) from face_observations
                where person_id = ? and canonical = 1 and reference_pinned = 1
                """,
                (int(person_id),),
            ).fetchone()[0])

        if before >= target:
            return {"person_id": int(person_id), "target_count": target, "before": before, "after": before, "added": 0}

        candidates = self.gallery_candidates(int(person_id), limit=100)
        selected_ids = [int(item["id"]) for item in candidates[: target - before]]
        if selected_ids:
            with self._lock, self._connect() as connection:
                connection.executemany(
                    """
                    update face_observations
                    set reference_pinned = 1, reference_auto_pinned = 1
                    where id = ? and person_id = ? and canonical = 1
                        and review_status = 'confirmed'
                    """,
                    ((observation_id, int(person_id)) for observation_id in selected_ids),
                )
            self._invalidate_reference_gallery()
            self.request_match_refresh()

        with self._connect() as connection:
            after = int(connection.execute(
                """
                select count(*) from face_observations
                where person_id = ? and canonical = 1 and reference_pinned = 1
                """,
                (int(person_id),),
            ).fetchone()[0])
        return {
            "person_id": int(person_id),
            "target_count": target,
            "before": before,
            "after": after,
            "added": max(0, after - before),
            "selected_observation_ids": selected_ids,
        }

    def people_representation_health(self) -> list[dict[str, Any]]:
        results = []
        for person in self.people():
            results.append(self.person_representation(int(person["id"])))
        results.sort(
            key=lambda item: (
                0 if item.get("flags") else 1,
                int(item.get("sample_count") or 0),
                str(item.get("name") or ""),
            )
        )
        return results

    def person_history(
        self,
        person_id: int,
        *,
        limit: int = 100,
    ) -> dict[str, Any]:
        resolved_limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            person = connection.execute(
                "select * from face_people where id = ?",
                (int(person_id),),
            ).fetchone()
            if person is None:
                raise ValueError("person not found")
            summary = connection.execute(
                """
                select count(*) as observations,
                    count(distinct camera_id) as camera_count,
                    min(observed_at) as first_seen,
                    max(observed_at) as last_seen,
                    avg(quality_score) as average_quality,
                    avg(match_confidence) as average_match_confidence,
                    sum(case when reference_pinned = 1 then 1 else 0 end)
                        as pinned_references,
                    sum(case when auto_identified = 1 then 1 else 0 end)
                        as auto_identified
                from face_observations
                where canonical = 1 and person_id = ?
                """,
                (int(person_id),),
            ).fetchone()
            cameras = connection.execute(
                """
                select camera_id, count(*) as observations,
                    min(observed_at) as first_seen,
                    max(observed_at) as last_seen,
                    avg(quality_score) as average_quality,
                    avg(match_confidence) as average_match_confidence
                from face_observations
                where canonical = 1 and person_id = ?
                group by camera_id
                order by observations desc, camera_id
                """,
                (int(person_id),),
            ).fetchall()
            recent = connection.execute(
                """
                select o.*, p.name as person_name,
                    candidate.name as candidate_person_name
                from face_observations o
                left join face_people p on p.id = o.person_id
                left join face_people candidate on candidate.id = o.candidate_person_id
                where o.canonical = 1 and o.person_id = ?
                order by o.observed_at desc, o.id desc
                limit ?
                """,
                (int(person_id), resolved_limit),
            ).fetchall()
        model_scores = []
        for row in recent:
            try:
                details = json.loads(str(row["match_details_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                details = {}
            if not isinstance(details, dict):
                continue
            try:
                score = float(details.get("score"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                model_scores.append(score)
        model_score = {
            "count": len(model_scores),
            "p05": round(float(np.percentile(model_scores, 5)), 4) if model_scores else None,
            "median": round(float(np.percentile(model_scores, 50)), 4) if model_scores else None,
            "p95": round(float(np.percentile(model_scores, 95)), 4) if model_scores else None,
        }
        return {
            "person": dict(person),
            "summary": dict(summary),
            "model_score": model_score,
            "cameras": [dict(row) for row in cameras],
            "recent": [self._observation_row(row) for row in recent],
        }

    def review_queue(
        self,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Prioritize actionable unknowns and ambiguous suggestions."""
        resolved_limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            rows = connection.execute(
                """
                select o.*, candidate.name as candidate_person_name,
                    m.cluster_id as unknown_cluster_id,
                    coalesce(cluster_stats.cluster_size, 1) as cluster_size
                from face_observations o
                left join face_people candidate on candidate.id = o.candidate_person_id
                left join face_unknown_members m on m.observation_id = o.id
                left join (
                    select cluster_id, count(*) as cluster_size
                    from face_unknown_members
                    group by cluster_id
                ) cluster_stats on cluster_stats.cluster_id = m.cluster_id
                where o.canonical = 1
                    and o.person_id is null
                    and o.recognition_pending = 0
                    and o.recognition_outcome = ?
                order by o.observed_at desc, o.id desc
                limit ?
                """,
                (FACE_OUTCOME_EMBEDDED, max(resolved_limit * 4, 200)),
            ).fetchall()

        queue_rows = []
        for row in rows:
            item = self._observation_row(row)
            quality = max(0.0, min(1.0, float(item.get("quality_score") or 0.0)))
            candidate_confidence = max(
                0.0,
                min(1.0, float(item.get("candidate_confidence") or 0.0)),
            )
            cluster_size = max(1, int(item.get("cluster_size") or 1))
            details = item.get("match_details") or {}
            margin = float(details.get("margin") or 0.0)
            ambiguity = (
                max(0.0, min(1.0, (0.12 - margin) / 0.12))
                if item.get("candidate_person_id") is not None
                else 0.0
            )
            recurring = min(1.0, max(0, cluster_size - 1) / 4.0)
            priority = (
                0.40 * quality
                + 0.25 * candidate_confidence
                + 0.20 * recurring
                + 0.15 * ambiguity
            )
            reasons = []
            if cluster_size > 1:
                reasons.append("recurring_unknown")
            if item.get("candidate_person_id") is not None:
                reasons.append("identity_suggestion")
            if ambiguity >= 0.5:
                reasons.append("ambiguous_match")
            if quality >= 0.70:
                reasons.append("high_quality")
            item["review_priority"] = round(priority, 4)
            item["review_reasons"] = reasons
            queue_rows.append(item)

        queue_rows.sort(
            key=lambda item: (
                float(item.get("review_priority") or 0.0),
                str(item.get("observed_at") or ""),
                int(item.get("id") or 0),
            ),
            reverse=True,
        )
        return queue_rows[:resolved_limit]

    def confirmed_match_diagnostics(
        self,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Find confirmed faces that fit another identity unusually well."""
        recognizer = self.recognizer
        if recognizer is None:
            return []
        status = recognizer.status()
        model_fingerprint = str(status.get("model_fingerprint") or "")
        if not model_fingerprint:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                select o.id, o.person_id, p.name as person_name, o.camera_id,
                    o.observed_at, o.quality_score, o.embedding_blob
                from face_observations o
                join face_people p on p.id = o.person_id
                where o.canonical = 1
                    and o.person_id is not null
                    and o.review_status = 'confirmed'
                    and o.embedding_model = ?
                    and o.embedding_blob is not null
                order by o.observed_at desc, o.id desc
                """,
                (model_fingerprint,),
            ).fetchall()
            if not rows:
                return []
            first_vector = np.frombuffer(rows[0]["embedding_blob"], dtype=np.float32)
            gallery = self._reference_gallery(
                connection,
                model_fingerprint,
                max(1, int(getattr(recognizer.config, "face_max_references", 20))),
                first_vector.shape,
            )

        names = {
            int(row["person_id"]): str(row["person_name"] or "")
            for row in rows
        }
        diagnostics = []
        for row in rows:
            target = np.frombuffer(row["embedding_blob"], dtype=np.float32)
            norm = float(np.linalg.norm(target))
            if not target.size or not math.isfinite(norm) or norm <= 1e-9:
                continue
            target = target / norm
            grouped = {}
            for reference in gallery:
                if int(reference["id"]) == int(row["id"]):
                    continue
                score = float(np.dot(target, reference["_embedding"]))
                grouped.setdefault(int(reference["person_id"]), []).append(score)
            true_person = int(row["person_id"])
            if true_person not in grouped or len(grouped) < 2:
                continue

            scores = {
                person_id: sum(sorted(values, reverse=True)[:3])
                / min(3, len(values))
                for person_id, values in grouped.items()
            }
            true_score = float(scores[true_person])
            wrong_person, wrong_score = max(
                (
                    (person_id, score)
                    for person_id, score in scores.items()
                    if person_id != true_person
                ),
                key=lambda item: item[1],
            )
            margin = true_score - float(wrong_score)
            flags = []
            if margin < 0:
                flags.append("assigned_identity_not_rank1")
            elif margin < 0.05:
                flags.append("poor_separation")
            if true_score < float(recognizer.config.face_match_threshold):
                flags.append("below_match_threshold")
            diagnostics.append({
                "observation_id": int(row["id"]),
                "person_id": true_person,
                "name": str(row["person_name"] or ""),
                "camera_id": str(row["camera_id"] or ""),
                "observed_at": str(row["observed_at"] or ""),
                "quality_score": (
                    round(float(row["quality_score"]), 4)
                    if row["quality_score"] is not None
                    else None
                ),
                "true_score": round(true_score, 4),
                "nearest_competing_person_id": int(wrong_person),
                "nearest_competing_name": names.get(int(wrong_person), ""),
                "nearest_competing_score": round(float(wrong_score), 4),
                "margin": round(margin, 4),
                "flags": flags,
            })

        diagnostics.sort(
            key=lambda item: (
                float(item["margin"]),
                float(item["true_score"]),
                int(item["observation_id"]),
            )
        )
        return diagnostics[: max(1, min(int(limit), 500))]

    def bulk_review(
        self,
        observation_ids: list[int],
        *,
        action: str,
        person_id: int | None = None,
    ) -> dict[str, Any]:
        ids = sorted({int(value) for value in observation_ids if int(value) > 0})
        if not ids:
            raise ValueError("at least one observation is required")
        if len(ids) > 500:
            raise ValueError("bulk review is limited to 500 observations")
        normalized_action = str(action or "").strip().lower()
        if normalized_action not in {"unassign", "assign"}:
            raise ValueError("bulk review action must be assign or unassign")
        if normalized_action == "assign" and person_id is None:
            raise ValueError("person_id is required for assign")

        changed = []
        with self._lock, self._connect() as connection:
            if person_id is not None:
                exists = connection.execute(
                    "select 1 from face_people where id = ?",
                    (int(person_id),),
                ).fetchone()
                if exists is None:
                    raise ValueError("person not found")

            for observation_id in ids:
                row = connection.execute(
                    "select id from face_observations where id = ?",
                    (observation_id,),
                ).fetchone()
                if row is None:
                    continue
                if normalized_action == "assign":
                    connection.execute(
                        """
                        update face_observations
                        set person_id = ?, review_status = 'confirmed',
                            match_confidence = 1,
                            candidate_person_id = null,
                            candidate_confidence = null,
                            rejected_person_id = null,
                            auto_identified = 0,
                            canonical = 1
                        where id = ?
                        """,
                        (int(person_id), observation_id),
                    )
                else:
                    connection.execute(
                        """
                        update face_observations
                        set person_id = null, review_status = 'unknown',
                            match_confidence = null,
                            candidate_person_id = null,
                            candidate_confidence = null,
                            rejected_person_id = null,
                            auto_identified = 0,
                            reference_pinned = 0,
                            reference_auto_pinned = 0
                        where id = ?
                        """,
                        (observation_id,),
                    )
                changed.append(observation_id)

        self._invalidate_reference_gallery()
        self.request_match_refresh()
        if normalized_action == "assign":
            for observation_id in changed:
                self._emit_identity_update(observation_id, source="manual_bulk")
        return {
            "action": normalized_action,
            "person_id": person_id,
            "changed": len(changed),
            "observation_ids": changed,
        }

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
        if observation_id is not None:
            self._emit_identity_update(observation_id, source="manual")
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
        if person_id is not None:
            self._emit_identity_update(observation_id, source="manual")
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

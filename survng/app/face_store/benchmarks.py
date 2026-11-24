from __future__ import annotations

import math
from typing import Any

import numpy as np


class FaceStoreBenchmarkMixin:
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
                "unknown_cluster_threshold": float(
                    getattr(
                        recognizer.config,
                        "face_unknown_cluster_threshold",
                        0.55,
                    )
                ),
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

from __future__ import annotations

import math
from typing import Any

import numpy as np


class FaceStoreBenchmarkMixin:
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

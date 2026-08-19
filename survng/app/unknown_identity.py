"""Deterministic recurring-unknown clustering for face embeddings."""
from __future__ import annotations
import numpy as np

DEFAULT_UNKNOWN_CLUSTER_THRESHOLD = 0.55
MIN_UNKNOWN_CLUSTER_QUALITY = 0.35

def normalized_embedding(blob):
    if blob is None:
        return None
    try:
        vector = np.frombuffer(blob, dtype=np.float32).copy()
    except (TypeError, ValueError):
        return None
    if vector.size == 0 or vector.size > 16384 or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-9:
        return None
    return vector / norm

def unknown_cluster_cohesion(rows, membership):
    """Return centroid-cohesion diagnostics for an existing membership map."""
    grouped = {}
    for row in rows:
        observation_id = int(row.get("id") or 0)
        cluster_id = int(membership.get(observation_id) or 0)
        if observation_id <= 0 or cluster_id <= 0:
            continue
        embedding = normalized_embedding(row.get("embedding_blob"))
        if embedding is None:
            continue
        grouped.setdefault(cluster_id, []).append(embedding)

    result = {}
    for cluster_id, members in grouped.items():
        if not members:
            continue
        aggregate = np.sum(np.vstack(members), axis=0)
        norm = float(np.linalg.norm(aggregate))
        if not np.isfinite(norm) or norm <= 1e-9:
            continue
        centroid = aggregate / norm
        scores = sorted(float(np.dot(centroid, member)) for member in members)
        result[cluster_id] = {
            "centroid_min_similarity": round(scores[0], 4),
            "centroid_median_similarity": round(float(np.median(scores)), 4),
            "centroid_p05_similarity": round(
                float(np.quantile(scores, 0.05)),
                4,
            ),
        }
    return result

def cluster_unknown_embeddings(
    rows,
    threshold=DEFAULT_UNKNOWN_CLUSTER_THRESHOLD,
):
    """Cluster unknown faces conservatively without single-link chain growth."""
    clusters = []
    membership = {}
    minimum = max(0.0, min(1.0, float(threshold)))
    support_floor = max(0.0, minimum - 0.03)
    radius_floor = max(0.0, minimum - 0.08)

    for row in sorted(rows, key=lambda item: int(item.get("id") or 0)):
        observation_id = int(row.get("id") or 0)
        if observation_id <= 0:
            continue
        if float(row.get("quality_score") or 0.0) < MIN_UNKNOWN_CLUSTER_QUALITY:
            continue
        embedding = normalized_embedding(row.get("embedding_blob"))
        if embedding is None:
            continue

        best = None
        for cluster in clusters:
            centroid = cluster["centroid"]
            members = cluster["members"]
            if centroid.shape != embedding.shape:
                continue

            growth_bonus = 0.03 if len(members) >= 12 else 0.0
            centroid_floor = min(0.95, minimum + growth_bonus)
            centroid_score = float(np.dot(centroid, embedding))
            if centroid_score < centroid_floor:
                continue

            member_scores = sorted(
                (
                    float(np.dot(member, embedding))
                    for member in members
                    if member.shape == embedding.shape
                ),
                reverse=True,
            )
            if not member_scores:
                continue

            if len(members) == 1:
                if member_scores[0] < minimum:
                    continue
            else:
                if len(member_scores) < 2 or member_scores[1] < support_floor:
                    continue

            proposed_members = members + [embedding]
            aggregate = cluster["aggregate"] + embedding
            norm = float(np.linalg.norm(aggregate))
            if not np.isfinite(norm) or norm <= 1e-9:
                continue
            proposed_centroid = aggregate / norm

            radius_scores = [
                float(np.dot(proposed_centroid, member))
                for member in proposed_members
            ]
            minimum_radius = min(radius_scores)
            median_radius = float(np.median(radius_scores))
            if minimum_radius < radius_floor:
                continue

            candidate = (
                median_radius,
                centroid_score,
                minimum_radius,
                -int(cluster["id"]),
                cluster,
                aggregate,
                proposed_centroid,
            )
            if best is None or candidate[:4] > best[:4]:
                best = candidate

        if best is None:
            cluster = {
                "id": observation_id,
                "aggregate": embedding.copy(),
                "centroid": embedding.copy(),
                "members": [embedding],
            }
            clusters.append(cluster)
        else:
            cluster = best[4]
            cluster["aggregate"] = best[5]
            cluster["centroid"] = best[6]
            cluster["members"].append(embedding)

        membership[observation_id] = int(cluster["id"])

    return membership


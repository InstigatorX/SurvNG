from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..unknown_identity import (
    DEFAULT_UNKNOWN_CLUSTER_THRESHOLD,
    cluster_unknown_embeddings,
    unknown_cluster_cohesion,
)
from .quality import LOGGER


class FaceStoreUnknownMixin:
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

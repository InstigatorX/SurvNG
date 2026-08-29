from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..incident_utils import event_snapshot_path, portable_media_path
from .quality import parse_face_box


class FaceStoreIngestMixin:
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
                    discarded_paths.append(resolved_snapshot)
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
            self._prune_locked(connection, discarded_paths)
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
            self._delete_face_snapshots([discarded_path], "superseded")
        for observation_id in recognition_ids:
            self._queue_recognition(observation_id)
        return inserted

    def ingest_events(self, events: list[dict[str, Any]]) -> int:
        inserted = 0
        recognition_ids: list[int] = []
        discarded_paths: list[Path] = []
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
            self._prune_locked(connection, discarded_paths)
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
        self._delete_face_snapshots(discarded_paths, "pruned")
        for observation_id in recognition_ids:
            self._queue_recognition(observation_id)
        return inserted

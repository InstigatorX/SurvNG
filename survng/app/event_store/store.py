from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..durable_payload import durable_json_dumps
from ..incident_utils import event_snapshot_path, portable_media_path
from ..media_storage import MediaStorageRegistry
from .calibration import EventStoreCalibrationMixin
from .jobs import EventStoreJobsMixin
from .motion_intelligence import EventStoreMotionIntelligenceMixin
from .tracking import EventStoreTrackingMixin


class EventStore(
    EventStoreJobsMixin,
    EventStoreCalibrationMixin,
    EventStoreTrackingMixin,
    EventStoreMotionIntelligenceMixin,
):
    SNAPSHOT_SIZE_WRITE_BATCH = 50
    SNAPSHOT_REFERENCE_WRITE_BATCH = 50
    SNAPSHOT_SIZE_BACKFILL_CURSOR_KEY = "snapshot_size_backfill_cursor"
    COMPACT_COLUMNS = (
        "id, camera_id, kind, snapshot_path, recording_path, objects_json, created_at"
    )
    TRACKING_COMPARISON_HISTORY_PER_CAMERA = 100
    TRACKING_COMPARISON_VERDICTS = {
        "survng_hybrid",
        "ultralytics_botsort",
        "ultralytics_deepocsort",
        "ultralytics_fasttrack",
        "inconclusive",
    }

    def __init__(
        self,
        storage_dir: Path,
        database_dir: Path | None = None,
        media_storage: MediaStorageRegistry | None = None,
    ) -> None:
        self.storage_dir = storage_dir
        self.media_storage = media_storage
        self.db_path = (database_dir or storage_dir) / "survng.sqlite3"
        self.jobs_db_path = self.db_path.parent / "detection-jobs.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # SQLite permits a single writer.  Camera event/refinement workers share
        # the jobs ledger, so serialize their short local transactions instead
        # of making them contend through SQLite's busy timeout.
        self._jobs_lock = threading.RLock()
        self._jobs_maintenance_lock = threading.Lock()
        self._last_detection_job_prune_monotonic = 0.0
        self._init_db()
        self._recover_snapshot_deletion_claims()
        self._init_jobs_db()
        self._migrate_legacy_jobs()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma busy_timeout = 10000")
        conn.execute("pragma foreign_keys = on")
        return conn

    def protected_recording_paths(self) -> set[str]:
        """Return continuous segments still referenced by incident history."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT recording_path FROM events WHERE recording_path != ''"
            ).fetchall()
        protected: set[str] = set()
        for row in rows:
            raw_path = str(row["recording_path"] or "")
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.storage_dir / path
            # Protection is a string-key lookup against the recording index.
            # Lexical normalization avoids an NFS metadata round trip for every
            # retained incident while remaining harmless for an out-of-pool
            # value: protection can only prevent deletion, never authorize it.
            protected.add(os.path.normpath(os.path.abspath(str(path))))
        return protected

    @staticmethod
    def _metadata_value(conn: sqlite3.Connection, key: str) -> str:
        row = conn.execute(
            "select value from survng_metadata where key = ?",
            (key,),
        ).fetchone()
        return str(row[0]) if row is not None else ""

    @staticmethod
    def _set_metadata_value(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            """
            insert into survng_metadata (key, value) values (?, ?)
            on conflict(key) do update set value = excluded.value
            """,
            (key, value),
        )

    @staticmethod
    def _qualification_from_objects(objects_json: str) -> dict[str, Any] | None:
        try:
            objects = json.loads(objects_json or "[]")
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(objects, list):
            return None
        return next(
            (
                item.get("motion_qualification")
                for item in objects
                if isinstance(item, dict)
                if item.get("status") == "motion_qualification"
                and isinstance(item.get("motion_qualification"), dict)
            ),
            None,
        )

    def _backfill_motion_audits(
        self,
        conn: sqlite3.Connection,
        *,
        after_event_id: int = 0,
    ) -> None:
        rows = conn.execute(
            """
            select id, camera_id, snapshot_path, created_at, objects_json
            from events
            where id > ?
              and objects_json like '%motion_qualification%'
              and id not in (select event_id from motion_audits where event_id is not null)
            order by id asc
            """,
            (max(0, int(after_event_id)),),
        ).fetchall()
        inserts: list[tuple[Any, ...]] = []
        for row in rows:
            qualification = self._qualification_from_objects(str(row["objects_json"] or ""))
            if not qualification:
                continue
            visual_backup = qualification.get("trigger_source") == "visual_backup"
            if not qualification.get("would_suppress") and not visual_backup:
                continue
            try:
                objects = json.loads(str(row["objects_json"] or "[]"))
            except (json.JSONDecodeError, TypeError):
                objects = []
            object_detected = any(
                isinstance(item, dict)
                and item.get("label")
                and item.get("incident_eligible") is not False
                for item in objects
            ) if isinstance(objects, list) else False
            raw_features = qualification.get("features")
            features = dict(raw_features) if isinstance(raw_features, dict) else {}
            reason = str(qualification.get("reason") or "rejected")
            category = "qualification"
            if visual_backup:
                features["visual_backup_original_reason"] = reason
                reason = "visual_backup_trigger"
                category = "visual_backup"
            inserts.append((
                int(row["id"]),
                str(row["camera_id"]),
                str(row["snapshot_path"] or ""),
                str(row["created_at"]),
                str(qualification.get("mode") or "audit"),
                str(qualification.get("sensitivity") or "balanced"),
                float(qualification.get("score") or 0.0),
                float(qualification.get("threshold") or 0.0),
                reason,
                int(object_detected),
                max(1, int(qualification.get("trigger_count") or 1)),
                json.dumps(features, separators=(",", ":")),
                category,
            ))
        if inserts:
            conn.executemany(
                """
                insert or ignore into motion_audits (
                    event_id, camera_id, snapshot_path, created_at, mode,
                    sensitivity, score, threshold, reason, object_detected,
                    trigger_count, features_json, category
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                inserts,
            )

    def _rebase_media_paths(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "select id, snapshot_path, recording_path from events where snapshot_path != '' or recording_path != ''"
        ).fetchall()
        updates: list[tuple[str, str, int]] = []
        for row in rows:
            snapshot_path = portable_media_path(self.storage_dir, row["snapshot_path"])
            recording_path = portable_media_path(self.storage_dir, row["recording_path"])
            if snapshot_path != str(row["snapshot_path"] or "") or recording_path != str(row["recording_path"] or ""):
                updates.append((snapshot_path, recording_path, int(row["id"])))
        if updates:
            conn.executemany(
                "update events set snapshot_path = ?, recording_path = ? where id = ?",
                updates,
            )

        audit_rows = conn.execute(
            "select id, snapshot_path from motion_audits where snapshot_path != ''"
        ).fetchall()
        audit_updates: list[tuple[str, int]] = []
        for row in audit_rows:
            raw_path = str(row["snapshot_path"] or "")
            portable_path = portable_media_path(self.storage_dir, raw_path)
            if portable_path != raw_path:
                audit_updates.append((portable_path, int(row["id"])))
        if audit_updates:
            conn.executemany(
                "update motion_audits set snapshot_path = ? where id = ?",
                audit_updates,
            )

        face_table = conn.execute(
            "select 1 from sqlite_master where type = 'table' and name = 'face_observations'"
        ).fetchone()
        if face_table:
            face_rows = conn.execute(
                "select id, snapshot_path from face_observations where snapshot_path != ''"
            ).fetchall()
            face_updates: list[tuple[str, int]] = []
            for row in face_rows:
                raw_path = str(row["snapshot_path"] or "")
                portable_path = portable_media_path(self.storage_dir, raw_path)
                if portable_path != raw_path:
                    face_updates.append((portable_path, int(row["id"])))
            if face_updates:
                conn.executemany(
                    "update face_observations set snapshot_path = ? where id = ?",
                    face_updates,
                )

    def add_event(
        self,
        camera_id: str,
        kind: str,
        topic: str = "",
        message: str = "",
        snapshot_path: str = "",
        recording_path: str = "",
        objects_json: str = "[]",
        created_at: str | None = None,
        detection_intent_id: str | None = None,
        route_origin_camera_id: str | None = None,
        route_origin_event_id: int | None = None,
    ) -> dict[str, Any]:
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()
        snapshot_path = portable_media_path(self.storage_dir, snapshot_path)
        snapshot_size_bytes = self._snapshot_file_size(snapshot_path)
        recording_path = portable_media_path(self.storage_dir, recording_path)
        route_origin_camera = str(route_origin_camera_id or "").strip()
        route_origin_event = int(route_origin_event_id or 0)
        route_admission = bool(route_origin_camera and route_origin_event > 0)
        discarded_snapshot_path = ""
        duplicate_route_admission: dict[str, Any] | None = None
        with self._lock, self._connect() as conn:
            # A route admission is the authoritative identity for a routed
            # occurrence.  Resolve it before checking the intent ID: a
            # coalesced camera-primary batch can carry the route job's stable
            # ID while selecting a later camera timestamp for its event.
            if route_admission:
                admitted = conn.execute(
                    "select event_id from route_incident_admissions "
                    "where origin_camera_id = ? and origin_event_id = ? "
                    "and target_camera_id = ?",
                    (route_origin_camera, route_origin_event, camera_id),
                ).fetchone()
                if admitted is not None:
                    duplicate_route_admission = {
                        "id": int(admitted["event_id"]),
                        "camera_id": camera_id,
                        "kind": kind,
                        "topic": topic,
                        "message": message,
                        "snapshot_path": snapshot_path,
                        "snapshot_size_bytes": snapshot_size_bytes,
                        "recording_path": recording_path,
                        "objects_json": objects_json,
                        "created_at": created_at,
                        "detection_intent_id": detection_intent_id,
                        "created": False,
                        "route_admission_created": False,
                    }
            if duplicate_route_admission is None:
                cursor = conn.execute(
                    """
                    insert or ignore into events (
                        camera_id, kind, topic, message, snapshot_path, snapshot_size_bytes,
                        recording_path, objects_json, created_at, detection_intent_id
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        camera_id,
                        kind,
                        topic,
                        message,
                        snapshot_path,
                        snapshot_size_bytes,
                        recording_path,
                        objects_json,
                        created_at,
                        detection_intent_id,
                    ),
                )
                created = bool(cursor.rowcount)
                if created:
                    event_id = cursor.lastrowid
                elif detection_intent_id:
                    row = conn.execute(
                        "select id, camera_id, kind, topic, message, created_at "
                        "from events where detection_intent_id = ?",
                        (detection_intent_id,),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("detection intent event insert was not recoverable")
                    if (
                        str(row["camera_id"]) != camera_id
                        or str(row["kind"]) != kind
                        or str(row["topic"]) != topic
                        or str(row["message"]) != message
                        or str(row["created_at"]) != created_at
                    ):
                        raise RuntimeError(
                            "detection intent identity collision with different occurrence"
                        )
                    event_id = int(row["id"])
                else:
                    raise RuntimeError("event insert failed")
                route_admission_created = False
                if route_admission:
                    admission = conn.execute(
                        "insert or ignore into route_incident_admissions "
                        "(origin_camera_id, origin_event_id, target_camera_id, "
                        "event_id, admitted_at) values (?, ?, ?, ?, ?)",
                        (
                            route_origin_camera,
                            route_origin_event,
                            camera_id,
                            event_id,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    route_admission_created = bool(admission.rowcount)
                    if not route_admission_created:
                        admitted = conn.execute(
                            "select event_id from route_incident_admissions "
                            "where origin_camera_id = ? and origin_event_id = ? "
                            "and target_camera_id = ?",
                            (route_origin_camera, route_origin_event, camera_id),
                        ).fetchone()
                        if admitted is None:
                            raise RuntimeError(
                                "route target admission conflict was not recoverable"
                            )
                        admitted_event_id = int(admitted["event_id"])
                        if created and admitted_event_id != int(event_id):
                            conn.execute("delete from events where id = ?", (event_id,))
                            discarded_snapshot_path = snapshot_path
                        event_id = admitted_event_id
                        created = False
        if duplicate_route_admission is not None:
            self._delete_snapshot_if_unreferenced(snapshot_path)
            return duplicate_route_admission
        if discarded_snapshot_path:
            self._delete_snapshot_if_unreferenced(discarded_snapshot_path)
        return {
            "id": event_id,
            "camera_id": camera_id,
            "kind": kind,
            "topic": topic,
            "message": message,
            "snapshot_path": snapshot_path,
            "snapshot_size_bytes": snapshot_size_bytes,
            "recording_path": recording_path,
            "objects_json": objects_json,
            "created_at": created_at,
            "detection_intent_id": detection_intent_id,
            "created": created,
            "route_admission_created": route_admission_created,
        }

    def route_target_admitted(
        self,
        origin_camera_id: str,
        origin_event_id: int,
        target_camera_id: str,
    ) -> bool:
        """Whether an origin occurrence already produced this target incident."""
        with self._connect() as conn:
            row = conn.execute(
                "select 1 from route_incident_admissions "
                "where origin_camera_id = ? and origin_event_id = ? "
                "and target_camera_id = ?",
                (
                    str(origin_camera_id),
                    int(origin_event_id),
                    str(target_camera_id),
                ),
            ).fetchone()
        return row is not None

    def telemetry_activity(
        self,
        *,
        hours: int = 24,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Aggregate bounded event/object history into UTC hourly buckets."""
        bounded_hours = max(1, min(int(hours), 168))
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        current_hour = current.replace(minute=0, second=0, microsecond=0)
        first_hour = current_hour - timedelta(hours=bounded_hours - 1)
        one_hour_ago = current - timedelta(hours=1)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                select camera_id, created_at, objects_json
                from events
                where created_at >= ?
                order by created_at, id
                """,
                (first_hour.isoformat(),),
            ).fetchall()

        def empty_counts() -> dict[str, Any]:
            return {"events": 0, "object_incidents": 0, "objects": 0, "labels": {}}

        def add_counts(target: dict[str, Any], detected: list[dict[str, Any]]) -> None:
            target["events"] += 1
            if detected:
                target["object_incidents"] += 1
            target["objects"] += len(detected)
            labels = target["labels"]
            for item in detected:
                label = str(item.get("label") or "unknown")
                labels[label] = int(labels.get(label, 0)) + 1

        hourly = [
            {"started_at": (first_hour + timedelta(hours=index)).isoformat(), **empty_counts()}
            for index in range(bounded_hours)
        ]

        def empty_hourly() -> list[dict[str, Any]]:
            return [
                {"started_at": item["started_at"], **empty_counts()}
                for item in hourly
            ]

        overall_24h = empty_counts()
        overall_1h = empty_counts()
        cameras: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                created_at = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                created_at = created_at.astimezone(timezone.utc)
            except (TypeError, ValueError):
                continue
            bucket_index = int((created_at.replace(minute=0, second=0, microsecond=0) - first_hour).total_seconds() // 3600)
            if bucket_index < 0 or bucket_index >= bounded_hours:
                continue
            try:
                objects = json.loads(str(row["objects_json"] or "[]"))
            except (TypeError, ValueError):
                objects = []
            detected = [
                item for item in objects
                if isinstance(item, dict)
                and item.get("label")
                and item.get("incident_eligible") is not False
                and not item.get("status")
            ] if isinstance(objects, list) else []
            camera_id = str(row["camera_id"])
            camera = cameras.setdefault(
                camera_id,
                {"last_24h": empty_counts(), "last_hour": empty_counts(), "hourly": empty_hourly()},
            )
            add_counts(hourly[bucket_index], detected)
            add_counts(camera["hourly"][bucket_index], detected)
            add_counts(overall_24h, detected)
            add_counts(camera["last_24h"], detected)
            if created_at >= one_hour_ago:
                add_counts(overall_1h, detected)
                add_counts(camera["last_hour"], detected)

        return {
            "hours": bounded_hours,
            "started_at": first_hour.isoformat(),
            "last_hour": overall_1h,
            "last_24h": overall_24h,
            "hourly": hourly,
            "by_camera": cameras,
        }




    def tracking_capacity_activity(
        self,
        *,
        hours: int,
        bucket_minutes: int,
        camera_id: str = "",
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        bounded_hours = max(1, min(int(hours), 24 * 31))
        bucket_seconds = max(60, min(int(bucket_minutes), 60) * 60)
        start = current - timedelta(hours=bounded_hours)
        query = "select camera_id, created_at, objects_json from events where created_at >= ?"
        parameters: list[Any] = [start.isoformat()]
        if camera_id:
            query += " and camera_id = ?"
            parameters.append(camera_id)
        query += " order by created_at, id"
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        buckets: dict[int, dict[str, Any]] = {}
        for row in rows:
            try:
                created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
                objects = json.loads(str(row["objects_json"] or "[]"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            created = created.astimezone(timezone.utc)
            tracking = next((
                item.get("object_tracking")
                for item in objects
                if isinstance(item, dict)
                and item.get("status") == "object_tracking"
                and isinstance(item.get("object_tracking"), dict)
            ), None) if isinstance(objects, list) else None
            if not tracking:
                continue
            bucket_epoch = int(created.timestamp() // bucket_seconds) * bucket_seconds
            bucket = buckets.setdefault(bucket_epoch, {
                "sampled_at": datetime.fromtimestamp(bucket_epoch, timezone.utc).isoformat(),
                "attempts": 0,
                "waited": 0,
                "skipped": 0,
                "wait_seconds_total": 0.0,
                "wait_seconds_max": 0.0,
            })
            wait_seconds = max(0.0, float(tracking.get("capacity_wait_seconds") or 0.0))
            bucket["attempts"] += 1
            bucket["waited"] += int(wait_seconds >= 0.01)
            bucket["skipped"] += int(tracking.get("state") == "skipped_capacity")
            bucket["wait_seconds_total"] += wait_seconds
            bucket["wait_seconds_max"] = max(bucket["wait_seconds_max"], wait_seconds)
        first_bucket = int(start.timestamp() // bucket_seconds) * bucket_seconds
        last_bucket = int(current.timestamp() // bucket_seconds) * bucket_seconds
        result: list[dict[str, Any]] = []
        for bucket_epoch in range(first_bucket, last_bucket + 1, bucket_seconds):
            bucket = buckets.get(bucket_epoch, {
                "sampled_at": datetime.fromtimestamp(bucket_epoch, timezone.utc).isoformat(),
                "attempts": 0,
                "waited": 0,
                "skipped": 0,
                "wait_seconds_total": 0.0,
                "wait_seconds_max": 0.0,
            })
            wait_total = float(bucket["wait_seconds_total"])
            result.append({
                "sampled_at": bucket["sampled_at"],
                "attempts": int(bucket["attempts"]),
                "waited": int(bucket["waited"]),
                "skipped": int(bucket["skipped"]),
                "wait_seconds_average": round(
                    wait_total / max(1, int(bucket["waited"])),
                    3,
                ),
                "wait_seconds_max": round(float(bucket["wait_seconds_max"]), 3),
            })
        return result

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 10000))
        with self._connect() as conn:
            rows = conn.execute(
                "select * from events order by id desc limit ?",
                (bounded_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_compact(
        self,
        limit: int = 500,
        before_created_at: str | None = None,
        before_id: int | None = None,
        camera_id: str = "",
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 10000))
        with self._connect() as conn:
            camera_clause = " and camera_id = ?" if camera_id else ""
            if before_created_at is None or before_id is None:
                rows = conn.execute(
                    f"select {self.COMPACT_COLUMNS} from events where 1 = 1{camera_clause} order by created_at desc, id desc limit ?",
                    ((camera_id,) if camera_id else ()) + (bounded_limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    select {self.COMPACT_COLUMNS} from events
                    where (created_at < ? or (created_at = ? and id < ?)){camera_clause}
                    order by created_at desc, id desc
                    limit ?
                    """,
                    (before_created_at, before_created_at, int(before_id))
                    + ((camera_id,) if camera_id else ())
                    + (bounded_limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def between_compact(
        self,
        start_at: str,
        end_at: str,
        camera_id: str = "",
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select {self.COMPACT_COLUMNS} from events
                where created_at >= ? and created_at < ?
                    {"and camera_id = ?" if camera_id else ""}
                order by created_at desc, id desc
                """,
                (start_at, end_at) + ((camera_id,) if camera_id else ()),
            ).fetchall()
        return [dict(row) for row in rows]

    def page_between(
        self,
        start_at: str,
        end_at: str,
        *,
        limit: int = 500,
        before_created_at: str | None = None,
        before_id: int | None = None,
        camera_ids: tuple[str, ...] = (),
        require_snapshot: bool = False,
    ) -> list[dict[str, Any]]:
        """Return a stable newest-first event page inside a half-open time range."""
        bounded_limit = max(1, min(int(limit), 5000))
        normalized_cameras = tuple(dict.fromkeys(
            str(camera_id).strip()
            for camera_id in camera_ids
            if str(camera_id).strip()
        ))
        clauses = ["created_at >= ?", "created_at < ?"]
        parameters: list[Any] = [start_at, end_at]
        if before_created_at is not None and before_id is not None:
            clauses.append("(created_at < ? or (created_at = ? and id < ?))")
            parameters.extend([
                before_created_at,
                before_created_at,
                int(before_id),
            ])
        if normalized_cameras:
            placeholders = ",".join("?" for _ in normalized_cameras)
            clauses.append(f"camera_id in ({placeholders})")
            parameters.extend(normalized_cameras)
        if require_snapshot:
            clauses.append("snapshot_path != ''")
        parameters.append(bounded_limit)
        query = f"""
            select * from events
            where {' and '.join(clauses)}
            order by created_at desc, id desc
            limit ?
        """
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def get_many(self, event_ids: list[int]) -> list[dict[str, Any]]:
        unique_ids = sorted({int(event_id) for event_id in event_ids if int(event_id) > 0})
        if not unique_ids:
            return []
        events: list[dict[str, Any]] = []
        with self._connect() as conn:
            for offset in range(0, len(unique_ids), 500):
                chunk = unique_ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"select * from events where id in ({placeholders})",
                    chunk,
                ).fetchall()
                events.extend(dict(row) for row in rows)
        return events

    def between(self, start_at: str, end_at: str, limit: int = 50000) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select * from events
                where created_at >= ? and created_at < ?
                order by created_at desc, id desc
                limit ?
                """,
                (start_at, end_at, max(1, min(int(limit), 200000))),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, event_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from events where id = ?",
                (event_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def update_objects(self, event_id: int, objects_json: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "update events set objects_json = ? where id = ?",
                (objects_json, event_id),
            )
            row = conn.execute(
                "select * from events where id = ?",
                (event_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def replace_detected_objects(
        self,
        event_id: int,
        detected_objects_json: str,
    ) -> dict[str, Any] | None:
        """Replace detections while preserving metadata added by concurrent workers."""
        try:
            detected_objects = json.loads(detected_objects_json or "[]")
        except (TypeError, ValueError):
            detected_objects = []
        if not isinstance(detected_objects, list):
            detected_objects = []
        detected_objects = [item for item in detected_objects if isinstance(item, dict)]
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "select objects_json from events where id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                existing = json.loads(str(row["objects_json"] or "[]"))
            except (TypeError, ValueError):
                existing = []
            preserved = [
                item
                for item in existing
                if isinstance(item, dict)
                and item.get("status") in {"motion_qualification", "object_tracking"}
            ] if isinstance(existing, list) else []
            objects_json = json.dumps([*detected_objects, *preserved], separators=(",", ":"))
            conn.execute(
                "update events set objects_json = ? where id = ?",
                (objects_json, event_id),
            )
            updated = conn.execute(
                "select * from events where id = ?",
                (event_id,),
            ).fetchone()
        return dict(updated) if updated is not None else None

    def refine_event_evidence(
        self,
        event_id: int,
        *,
        snapshot_path: str,
        recording_path: str,
        objects_json: str,
    ) -> dict[str, Any] | None:
        """Atomically replace delayed evidence without losing tracking state."""
        try:
            replacement = json.loads(objects_json or "[]")
        except (TypeError, ValueError):
            replacement = []
        if not isinstance(replacement, list):
            replacement = []
        replacement = [item for item in replacement if isinstance(item, dict)]
        portable_snapshot = portable_media_path(self.storage_dir, snapshot_path)
        snapshot_size_bytes = self._snapshot_file_size(portable_snapshot)
        portable_recording = portable_media_path(self.storage_dir, recording_path)
        replaced_snapshot = ""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "select objects_json, snapshot_path, snapshot_size_bytes, recording_path from events where id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            replaced_snapshot = str(row["snapshot_path"] or "")
            try:
                existing = json.loads(str(row["objects_json"] or "[]"))
            except (TypeError, ValueError):
                existing = []
            preserved = [
                item
                for item in existing
                if isinstance(item, dict) and item.get("status") == "object_tracking"
            ] if isinstance(existing, list) else []
            merged_json = json.dumps([*replacement, *preserved], separators=(",", ":"))
            conn.execute(
                """
                update events
                set snapshot_path = ?, snapshot_size_bytes = ?, recording_path = ?, objects_json = ?
                where id = ?
                """,
                (
                    portable_snapshot or str(row["snapshot_path"] or ""),
                    snapshot_size_bytes if portable_snapshot else int(row["snapshot_size_bytes"] or 0),
                    portable_recording or str(row["recording_path"] or ""),
                    merged_json,
                    event_id,
                ),
            )
            updated = conn.execute(
                "select * from events where id = ?",
                (event_id,),
            ).fetchone()
        if replaced_snapshot and replaced_snapshot != portable_snapshot:
            self._delete_snapshot_if_unreferenced(replaced_snapshot)
        return dict(updated) if updated is not None else None

    def promote_tracking_cover(
        self,
        event_id: int,
        *,
        snapshot_path: str,
        captured_at: float,
        frame_width: int,
        frame_height: int,
        tracked_objects: list[dict[str, Any]],
        cover_metrics: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Replace only presentation evidence with a better tracked frame.

        Detection/admission facts remain intact. Boxes are refreshed for the
        promoted frame, and objects not visible there are explicitly excluded
        from image annotations and crop-based indexing.
        """
        portable_snapshot = portable_media_path(self.storage_dir, snapshot_path)
        if not portable_snapshot or frame_width <= 0 or frame_height <= 0:
            return None
        snapshot_size_bytes = self._snapshot_file_size(portable_snapshot)
        candidates = {
            str(item.get("track_id")): item
            for item in tracked_objects
            if isinstance(item, dict)
            and item.get("track_id") is not None
            and isinstance(item.get("box"), dict)
        }
        if not candidates:
            self._delete_snapshot_if_unreferenced(portable_snapshot)
            return None
        replaced_snapshot = ""
        matched = 0
        updated = None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "select objects_json, snapshot_path from events where id = ?",
                (event_id,),
            ).fetchone()
            if row is not None:
                try:
                    objects = json.loads(str(row["objects_json"] or "[]"))
                except (TypeError, ValueError):
                    objects = []
                if not isinstance(objects, list):
                    objects = []
                for item in objects:
                    if not isinstance(item, dict) or not item.get("label"):
                        continue
                    candidate = candidates.get(str(item.get("track_id")))
                    if candidate is None:
                        item["snapshot_visible"] = False
                        continue
                    item["box"] = dict(candidate["box"])
                    item["detection_frame_width"] = int(frame_width)
                    item["detection_frame_height"] = int(frame_height)
                    item["snapshot_visible"] = True
                    item["snapshot_source"] = "object_tracking"
                    item["snapshot_captured_at"] = datetime.fromtimestamp(
                        captured_at,
                        timezone.utc,
                    ).isoformat()
                    item["snapshot_detection_confidence"] = float(
                        candidate.get("confidence") or 0.0
                    )
                    if item.get("snapshot_primary_subject") is True:
                        for key, value in cover_metrics.items():
                            if key.startswith("snapshot_"):
                                item[key] = value
                    matched += 1
                if matched > 0:
                    replaced_snapshot = str(row["snapshot_path"] or "")
                    conn.execute(
                        """
                        update events
                        set snapshot_path = ?, snapshot_size_bytes = ?, objects_json = ?
                        where id = ?
                        """,
                        (
                            portable_snapshot,
                            snapshot_size_bytes,
                            json.dumps(objects, separators=(",", ":")),
                            event_id,
                        ),
                    )
                    updated = conn.execute(
                        "select * from events where id = ?",
                        (event_id,),
                    ).fetchone()
        if updated is None:
            self._delete_snapshot_if_unreferenced(portable_snapshot)
            return None
        if replaced_snapshot and replaced_snapshot != portable_snapshot:
            self._delete_snapshot_if_unreferenced(replaced_snapshot)
        return dict(updated) if updated is not None else None

    def promote_refinement_cover(
        self,
        event_id: int,
        *,
        snapshot_path: str,
        recording_path: str,
        captured_at: float,
        frame_width: int,
        frame_height: int,
        cover_objects: list[dict[str, Any]],
        source: str,
        timestamp_exact: bool,
    ) -> dict[str, Any] | None:
        """Promote verified main evidence without changing admission facts.

        Fast detection, causal motion correlation, and cover selection answer
        different questions.  A main-stream object may fail to *explain* an
        EMA region (notably when main/sub geometry is untrusted) while still
        being a materially better view of the already admitted provisional
        subject.  This transaction updates presentation coordinates only.

        Matching is deliberately conservative: one provisional subject and
        one temporally confirmed same-label subject, close in time, with
        materially more subject pixels. This is compatibility evidence, not
        identity proof. Ambiguous same-label scenes remain on the original
        cover and can later be promoted by tracked identity.
        """
        if (
            frame_width <= 0
            or frame_height <= 0
            or not math.isfinite(float(captured_at))
        ):
            return None
        portable_snapshot = portable_media_path(self.storage_dir, snapshot_path)
        if not portable_snapshot:
            return None
        portable_recording = portable_media_path(self.storage_dir, recording_path)

        def valid_box(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
            box = item.get("box")
            if not isinstance(box, dict):
                return None
            try:
                x1 = float(box["x1"])
                y1 = float(box["y1"])
                x2 = float(box["x2"])
                y2 = float(box["y2"])
            except (KeyError, TypeError, ValueError, OverflowError):
                return None
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                return None
            if x2 <= x1 or y2 <= y1:
                return None
            return x1, y1, x2, y2

        candidates = [
            item
            for item in cover_objects
            if isinstance(item, dict)
            and item.get("label")
            and not item.get("auxiliary_detection")
            and item.get("temporal_consensus") is True
            and item.get("confidence_eligible") is not False
            and item.get("zone_eligible") is not False
            and valid_box(item) is not None
        ]
        if not candidates:
            return None

        replaced_snapshot = ""
        snapshot_size_bytes = self._snapshot_file_size(portable_snapshot)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "select objects_json, snapshot_path, recording_path from events where id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                objects = json.loads(str(row["objects_json"] or "[]"))
            except (TypeError, ValueError):
                objects = []
            if not isinstance(objects, list):
                return None
            provisional = [
                item
                for item in objects
                if isinstance(item, dict)
                and item.get("label")
                and item.get("incident_eligible") is not False
                and item.get("provisional_detection") is True
            ]
            if len(provisional) != 1:
                return None
            existing = provisional[0]
            existing_label = str(existing.get("label") or "").strip().lower()
            compatible = [
                item
                for item in candidates
                if str(item.get("label") or "").strip().lower() == existing_label
            ]
            if not existing_label or len(compatible) != 1:
                return None
            candidate = compatible[0]
            existing_box = valid_box(existing)
            candidate_box = valid_box(candidate)
            if existing_box is None or candidate_box is None:
                return None
            try:
                existing_width = int(existing.get("detection_frame_width") or 0)
                existing_height = int(existing.get("detection_frame_height") or 0)
                existing_captured_at = float(existing.get("frame_captured_at_epoch"))
            except (TypeError, ValueError, OverflowError):
                return None
            if (
                existing_width <= 0
                or existing_height <= 0
                or not math.isfinite(existing_captured_at)
                or abs(float(captured_at) - existing_captured_at) > 15.0
            ):
                return None
            existing_subject_pixels = (
                (existing_box[2] - existing_box[0])
                * (existing_box[3] - existing_box[1])
            )
            candidate_subject_pixels = (
                (candidate_box[2] - candidate_box[0])
                * (candidate_box[3] - candidate_box[1])
            )
            candidate_clearance = min(
                candidate_box[0] / frame_width,
                candidate_box[1] / frame_height,
                (frame_width - candidate_box[2]) / frame_width,
                (frame_height - candidate_box[3]) / frame_height,
            )
            if (
                frame_width * frame_height <= existing_width * existing_height
                or candidate_subject_pixels < max(64.0, existing_subject_pixels * 1.5)
                or candidate_clearance < 0.005
            ):
                return None

            for item in objects:
                if not isinstance(item, dict) or not item.get("label"):
                    continue
                if item is not existing:
                    item["snapshot_visible"] = False
                    continue
                item["box"] = dict(candidate["box"])
                item["detection_frame_width"] = int(frame_width)
                item["detection_frame_height"] = int(frame_height)
                item["snapshot_visible"] = True
                item["snapshot_source"] = str(source or "recorded_refinement")
                item["snapshot_captured_at"] = datetime.fromtimestamp(
                    float(captured_at),
                    timezone.utc,
                ).isoformat()
                item["snapshot_detection_confidence"] = float(
                    candidate.get("confidence") or 0.0
                )
                item["snapshot_presentation_only"] = True
                item["snapshot_timestamp_exact"] = bool(timestamp_exact)
                for key in (
                    "snapshot_quality_score",
                    "snapshot_sharpness_score",
                    "snapshot_exposure_score",
                    "snapshot_contrast_score",
                    "snapshot_edge_detail_score",
                    "snapshot_primary_subject",
                    "snapshot_edge_clearance_ratio",
                    "snapshot_subject_area_ratio",
                ):
                    if key in candidate:
                        item[key] = candidate[key]
            objects = [
                item
                for item in objects
                if not (
                    isinstance(item, dict)
                    and item.get("status") == "cover_promotion"
                )
            ]
            objects.append({
                "status": "cover_promotion",
                "cover_promotion": {
                    "source": str(source or "recorded_refinement"),
                    "captured_at": datetime.fromtimestamp(
                        float(captured_at),
                        timezone.utc,
                    ).isoformat(),
                    "timestamp_exact": bool(timestamp_exact),
                    "reason": "compatible_recorded_refinement",
                    "admission_preserved": True,
                },
            })
            replaced_snapshot = str(row["snapshot_path"] or "")
            conn.execute(
                """
                update events
                set snapshot_path = ?, snapshot_size_bytes = ?, recording_path = ?, objects_json = ?
                where id = ?
                """,
                (
                    portable_snapshot,
                    snapshot_size_bytes,
                    portable_recording or str(row["recording_path"] or ""),
                    json.dumps(objects, separators=(",", ":")),
                    event_id,
                ),
            )
            updated = conn.execute(
                "select * from events where id = ?",
                (event_id,),
            ).fetchone()
        if replaced_snapshot and replaced_snapshot != portable_snapshot:
            self._delete_snapshot_if_unreferenced(replaced_snapshot)
        return dict(updated) if updated is not None else None

    def _snapshot_file_size(self, raw_path: str) -> int:
        """Return an incident snapshot's size without allowing arbitrary paths."""
        if not raw_path:
            return 0
        try:
            path = event_snapshot_path(
                self.storage_dir,
                {"snapshot_path": raw_path},
                self.media_storage,
            )
            if self.media_storage is None:
                path.relative_to((self.storage_dir / "snapshots").resolve())
            elif self.media_storage.location_id_for(path, role="snapshots") is None:
                return 0
            return max(0, int(path.stat().st_size))
        except (FileNotFoundError, PermissionError, OSError, RuntimeError, ValueError):
            return 0

    def migrate_snapshot_sizes(
        self,
        *,
        limit: int = 250,
        write_batch_size: int | None = None,
    ) -> int:
        """Index one bounded legacy cohort outside ordinary retention planning.

        This is intentionally an explicit, low-priority migration operation:
        retention status reads must never perform media filesystem I/O or take
        an SQLite writer lock merely because an older row has no stored size.
        """
        with self._lock, self._connect() as conn:
            try:
                cursor_id = max(
                    0,
                    int(
                        self._metadata_value(
                            conn,
                            self.SNAPSHOT_SIZE_BACKFILL_CURSOR_KEY,
                        )
                        or 0
                    ),
                )
            except ValueError:
                cursor_id = 0
            rows = conn.execute(
                """
                select min(id) as cursor_id, snapshot_path from events
                where snapshot_path != '' and snapshot_size_bytes <= 0
                  and not exists (
                      select 1 from snapshot_size_migration
                      where snapshot_size_migration.snapshot_path = events.snapshot_path
                        and snapshot_size_migration.checked_at > ?
                  )
                group by snapshot_path
                having min(id) > ?
                order by cursor_id asc limit ?
                """,
                (
                    (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
                    cursor_id,
                    max(1, int(limit)),
                ),
            ).fetchall()
        if not rows:
            if cursor_id > 0:
                with self._lock, self._connect() as conn:
                    self._set_metadata_value(
                        conn,
                        self.SNAPSHOT_SIZE_BACKFILL_CURSOR_KEY,
                        "0",
                    )
            return 0
        next_cursor_id = max(int(row["cursor_id"]) for row in rows)
        updates = [
            (size, str(row["snapshot_path"]))
            for row in rows
            if (size := self._snapshot_file_size(str(row["snapshot_path"]))) > 0
        ]
        batch_size = max(
            1,
            min(
                250,
                int(write_batch_size or self.SNAPSHOT_SIZE_WRITE_BATCH),
            ),
        )
        for offset in range(0, len(updates), batch_size):
            batch = updates[offset : offset + batch_size]
            with self._lock, self._connect() as conn:
                conn.executemany(
                    "update events set snapshot_size_bytes = ? where snapshot_path = ?",
                    batch,
                )
                if offset + batch_size >= len(updates):
                    self._set_metadata_value(
                        conn,
                        self.SNAPSHOT_SIZE_BACKFILL_CURSOR_KEY,
                        str(next_cursor_id),
                    )
        checked_paths = [str(row["snapshot_path"]) for row in rows]
        if checked_paths:
            with self._lock, self._connect() as conn:
                conn.executemany(
                    "insert or replace into snapshot_size_migration "
                    "(snapshot_path, checked_at) values (?, ?)",
                    (
                        (path, datetime.now(timezone.utc).isoformat())
                        for path in checked_paths
                    ),
                )
                self._set_metadata_value(
                    conn,
                    self.SNAPSHOT_SIZE_BACKFILL_CURSOR_KEY,
                    str(next_cursor_id),
                )
        return len(updates)

    def snapshot_retention_plan(self, cutoff_epoch: float) -> dict[str, Any]:
        """Report database-indexed incident snapshot use and age expiry.

        Planning is a pure database read.  In WAL mode it can safely run beside
        incident writers, so it must not acquire the EventStore writer mutex.
        """
        cutoff = datetime.fromtimestamp(float(cutoff_epoch), timezone.utc).isoformat()
        ranked = """
            with ranked as (
                select id, camera_id, snapshot_path, snapshot_size_bytes, created_at,
                       row_number() over (
                           partition by snapshot_path order by created_at desc, id desc
                       ) as snapshot_rank
                from events where snapshot_path != ''
            )
        """
        with self._connect() as conn:
            has_faces = conn.execute(
                "select 1 from sqlite_master where type = 'table' and name = 'face_observations'"
            ).fetchone() is not None
            protected_join = (
                "left join (select distinct snapshot_path from face_observations "
                "where reference_pinned = 1) as pinned "
                "on pinned.snapshot_path = ranked.snapshot_path"
                if has_faces
                else ""
            )
            unprotected_predicate = (
                "and pinned.snapshot_path is null" if has_faces else ""
            )
            cameras = conn.execute(
                ranked
                + f"""
                select camera_id,
                       count(*) as file_count,
                       coalesce(sum(snapshot_size_bytes), 0) as bytes,
                       coalesce(sum(case when snapshot_size_bytes <= 0
                                         and sized.snapshot_path is null
                                         then 1 else 0 end), 0)
                           as unindexed_files,
                       coalesce(sum(case when created_at < ? {unprotected_predicate}
                                         then 1 else 0 end), 0) as expired_files,
                       coalesce(sum(case when created_at < ? {unprotected_predicate}
                                         then snapshot_size_bytes else 0 end), 0)
                           as expired_bytes
                from ranked
                left join snapshot_size_migration as sized
                  on sized.snapshot_path = ranked.snapshot_path
                {protected_join}
                where snapshot_rank = 1
                group by camera_id order by camera_id
                """,
                (cutoff, cutoff),
            ).fetchall()
        file_count = sum(int(row["file_count"] or 0) for row in cameras)
        total_bytes = sum(int(row["bytes"] or 0) for row in cameras)
        unindexed_files = sum(int(row["unindexed_files"] or 0) for row in cameras)
        expired_files = sum(int(row["expired_files"] or 0) for row in cameras)
        expired_bytes = sum(int(row["expired_bytes"] or 0) for row in cameras)
        return {
            "file_count": file_count,
            "bytes": total_bytes,
            "unindexed_files": unindexed_files,
            "expired_files": expired_files,
            "expired_bytes": expired_bytes,
            "per_camera": [
                {
                    "camera_id": str(row["camera_id"]),
                    "file_count": int(row["file_count"] or 0),
                    "bytes": int(row["bytes"] or 0),
                }
                for row in cameras
            ],
        }

    def _snapshot_path_for_retention(self, raw_path: str) -> Path:
        """Resolve a snapshot retention path even when its file is gone."""
        path = Path(raw_path)
        resolved = (
            path if path.is_absolute() else self.storage_dir / path
        ).resolve(strict=False)
        if resolved.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise PermissionError("snapshot is not an image")
        if self.media_storage is None:
            resolved.relative_to((self.storage_dir / "snapshots").resolve())
        elif self.media_storage.location_id_for(resolved, role="snapshots") is None:
            raise PermissionError("snapshot is outside configured snapshot storage")
        return resolved

    @staticmethod
    def _clear_snapshot_references(
        conn: sqlite3.Connection,
        paths: list[str],
    ) -> None:
        if not paths:
            return
        conn.executemany(
            "update events set snapshot_path = '', snapshot_size_bytes = 0 where snapshot_path = ?",
            ((path,) for path in paths),
        )
        conn.executemany(
            "update motion_audits set snapshot_path = '' where snapshot_path = ?",
            ((path,) for path in paths),
        )
        has_faces = conn.execute(
            "select 1 from sqlite_master where type = 'table' and name = 'face_observations'"
        ).fetchone() is not None
        if has_faces:
            conn.executemany(
                "update face_observations set snapshot_path = '' where snapshot_path = ?",
                ((path,) for path in paths),
            )

    def _recover_snapshot_deletion_claims(self) -> None:
        """Finish or release snapshot deletion claims left by a process crash."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "select path from media_deletion_claims where role = 'snapshot'"
            ).fetchall()
            missing: list[str] = []
            releasable: list[str] = []
            for row in rows:
                raw_path = str(row["path"] or "")
                try:
                    path = self._snapshot_path_for_retention(raw_path)
                except (OSError, RuntimeError, ValueError, PermissionError):
                    # A malformed claim must never authorize a reference wipe.
                    continue
                releasable.append(raw_path)
                if not path.exists():
                    missing.append(raw_path)
            self._clear_snapshot_references(conn, missing)
            conn.executemany(
                "delete from media_deletion_claims where path = ? and role = 'snapshot'",
                ((path,) for path in releasable),
            )

    def apply_snapshot_retention(self, cutoff_epoch: float, limit: int) -> dict[str, Any]:
        """Delete age-expired incident images and clear every stale reference."""
        cutoff = datetime.fromtimestamp(float(cutoff_epoch), timezone.utc).isoformat()
        bounded_limit = max(1, min(2000, int(limit)))
        ranked = """
            with ranked as (
                select id, snapshot_path, snapshot_size_bytes, created_at,
                       row_number() over (
                           partition by snapshot_path order by created_at desc, id desc
                       ) as snapshot_rank
                from events where snapshot_path != ''
            )
        """
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            conn.execute(
                "delete from media_deletion_claims "
                "where role = 'snapshot' and claimed_at < ?",
                ((datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),),
            )
            has_faces = conn.execute(
                "select 1 from sqlite_master where type = 'table' and name = 'face_observations'"
            ).fetchone() is not None
            face_clause = (
                "and not exists (select 1 from face_observations "
                "where face_observations.snapshot_path = ranked.snapshot_path "
                "and face_observations.reference_pinned = 1)"
                if has_faces
                else ""
            )
            rows = conn.execute(
                ranked
                + f"""
                select snapshot_path, snapshot_size_bytes from ranked
                where snapshot_rank = 1 and created_at < ? {face_clause}
                order by created_at asc limit ?
                """,
                (cutoff, bounded_limit),
            ).fetchall()
            claimed_at = datetime.now(timezone.utc).isoformat()
            claimed_rows: list[sqlite3.Row] = []
            for row in rows:
                claimed = conn.execute(
                    "insert or ignore into media_deletion_claims "
                    "(path, role, claimed_at) values (?, 'snapshot', ?)",
                    (str(row["snapshot_path"]), claimed_at),
                )
                if claimed.rowcount:
                    claimed_rows.append(row)
            rows = claimed_rows
        removed: list[str] = []
        deleted_files = 0
        missing_files = 0
        deleted_bytes = 0
        failed_files = 0
        for row in rows:
            raw_path = str(row["snapshot_path"] or "")
            try:
                path = self._snapshot_path_for_retention(raw_path)
                actual_size = max(
                    int(row["snapshot_size_bytes"] or 0), int(path.stat().st_size)
                )
                path.unlink()
                deleted_files += 1
                deleted_bytes += actual_size
                removed.append(raw_path)
            except FileNotFoundError:
                missing_files += 1
                removed.append(raw_path)
            except (PermissionError, OSError, RuntimeError, ValueError):
                failed_files += 1
        for offset in range(0, len(removed), self.SNAPSHOT_REFERENCE_WRITE_BATCH):
            batch = removed[offset : offset + self.SNAPSHOT_REFERENCE_WRITE_BATCH]
            with self._lock, self._connect() as conn:
                self._clear_snapshot_references(conn, batch)
        if rows:
            with self._lock, self._connect() as conn:
                conn.executemany(
                    "delete from media_deletion_claims "
                    "where path = ? and role = 'snapshot'",
                    ((str(row["snapshot_path"]),) for row in rows),
                )
        return {
            "selected_files": len(rows),
            "deleted_files": deleted_files,
            "missing_files": missing_files,
            "deleted_bytes": deleted_bytes,
            "failed_files": failed_files,
            "batch_saturated": len(rows) >= bounded_limit,
        }

    def _delete_snapshot_if_unreferenced(self, raw_path: str) -> None:
        """Remove a replaced snapshot only after every durable reference moved."""
        portable = portable_media_path(self.storage_dir, raw_path)
        if not portable:
            return
        with self._lock, self._connect() as conn:
            referenced = bool(conn.execute(
                """
                select exists(select 1 from events where snapshot_path = ?)
                    or exists(select 1 from motion_audits where snapshot_path = ?)
                """,
                (portable, portable),
            ).fetchone()[0])
            if not referenced:
                has_faces = conn.execute(
                    "select 1 from sqlite_master where type = 'table' and name = 'face_observations'"
                ).fetchone() is not None
                referenced = bool(
                    has_faces
                    and conn.execute(
                        "select 1 from face_observations where snapshot_path = ? limit 1",
                        (portable,),
                    ).fetchone() is not None
                )
        if referenced:
            return
        try:
            path = event_snapshot_path(
                self.storage_dir,
                {"snapshot_path": portable},
                self.media_storage,
            )
            if self.media_storage is None:
                path.relative_to((self.storage_dir / "snapshots").resolve())
            elif self.media_storage.location_id_for(path, role="snapshots") is None:
                return
            path.unlink(missing_ok=True)
        except (FileNotFoundError, PermissionError, OSError, RuntimeError, ValueError):
            return

    def update_object_tracking(
        self,
        event_id: int,
        tracking: dict[str, Any],
        tracked_objects: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Atomically replace tracking metadata without losing concurrent event data."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "select objects_json from events where id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                objects = json.loads(str(row["objects_json"] or "[]"))
            except (TypeError, ValueError):
                objects = []
            if not isinstance(objects, list):
                objects = []
            had_tracking = any(
                isinstance(item, dict) and item.get("status") == "object_tracking"
                for item in objects
            )
            objects = [
                item
                for item in objects
                if not (isinstance(item, dict) and item.get("status") == "object_tracking")
            ]
            if tracked_objects and not had_tracking:
                assignments = {
                    (
                        str(item.get("label") or ""),
                        json.dumps(item.get("box"), sort_keys=True, separators=(",", ":")),
                    ): item
                    for item in tracked_objects
                    if item.get("track_id") is not None
                }
                for item in objects:
                    if not isinstance(item, dict) or not item.get("label"):
                        continue
                    assigned = assignments.get((
                        str(item.get("label") or ""),
                        json.dumps(item.get("box"), sort_keys=True, separators=(",", ":")),
                    ))
                    if assigned is not None:
                        item["track_id"] = assigned["track_id"]
                        item["track_state"] = assigned.get("track_state")
                        item["track_observations"] = assigned.get("track_observations")
            objects.append({"status": "object_tracking", "object_tracking": tracking})
            objects_json = json.dumps(objects, separators=(",", ":"))
            conn.execute(
                "update events set objects_json = ? where id = ?",
                (objects_json, event_id),
            )
            updated = conn.execute(
                "select * from events where id = ?",
                (event_id,),
            ).fetchone()
        return dict(updated) if updated is not None else None

    def for_camera_range(
        self,
        camera_id: str,
        start_at: str,
        end_at: str,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 200000))
        with self._connect() as conn:
            rows = conn.execute(
                """
                select * from events
                where camera_id = ?
                    and created_at >= ?
                    and created_at < ?
                order by created_at asc
                limit ?
                """,
                (camera_id, start_at, end_at, bounded_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_for_camera_range(
        self,
        camera_id: str,
        start_at: str,
        end_at: str,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return the newest camera events without scanning unrelated cameras."""
        bounded_limit = max(1, min(int(limit), 200000))
        with self._connect() as conn:
            rows = conn.execute(
                """
                select * from events
                where camera_id = ?
                    and created_at >= ?
                    and created_at < ?
                order by created_at desc, id desc
                limit ?
                """,
                (camera_id, start_at, end_at, bounded_limit),
            ).fetchall()
        return [dict(row) for row in rows]

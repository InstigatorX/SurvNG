from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from ..incident_utils import portable_media_path


class EventStoreMotionIntelligenceMixin:
    def add_motion_audit(
        self,
        *,
        camera_id: str,
        snapshot_path: str,
        created_at: str,
        mode: str,
        sensitivity: str,
        score: float,
        threshold: float,
        reason: str,
        object_detected: bool | None,
        trigger_count: int,
        features: dict[str, Any],
        category: str = "qualification",
        event_id: int | None = None,
        related_event_id: int | None = None,
        decision_id: str = "",
    ) -> dict[str, Any]:
        normalized_object_detected = (
            None if object_detected is None else int(object_detected)
        )
        normalized_trigger_count = max(1, int(trigger_count))
        snapshot_path = portable_media_path(self.storage_dir, snapshot_path)
        normalized_decision_id = str(decision_id or "").strip()
        normalized_category = str(category or "qualification").strip().lower()
        if normalized_category not in {
            "qualification",
            "visual_backup",
            "active_followup",
        }:
            raise ValueError("invalid motion audit category")
        normalized_related_event_id = (
            int(related_event_id) if related_event_id is not None else None
        )
        if normalized_related_event_id is not None and normalized_related_event_id <= 0:
            raise ValueError("related motion event id must be positive")
        if len(normalized_decision_id) > 128:
            raise ValueError("motion audit decision_id must be at most 128 characters")
        normalized_score = float(score)
        normalized_threshold = float(threshold)
        if not math.isfinite(normalized_score) or not math.isfinite(normalized_threshold):
            raise ValueError("motion audit score and threshold must be finite")
        compact_features, pipeline_configurations = self._compact_audit_features(
            features or {}
        )
        features_json = json.dumps(
            compact_features,
            separators=(",", ":"),
            allow_nan=False,
        )
        replaced_snapshot = ""
        persisted_snapshot = snapshot_path
        with self._lock, self._connect() as conn:
            for fingerprint, configuration_json in pipeline_configurations.items():
                conn.execute(
                    """
                    insert or ignore into motion_audit_pipeline_configs (
                        fingerprint, configuration_json, created_at
                    ) values (?, ?, ?)
                    """,
                    (fingerprint, configuration_json, created_at),
                )
            audit_id: int | None = None
            if (
                reason in {"event_state_active", "event_state_cooldown"}
                and normalized_related_event_id is not None
                and event_id is None
            ):
                existing = conn.execute(
                    """
                    select id, trigger_count, features_json, snapshot_path
                    from motion_audits
                    where event_id is null and related_event_id = ?
                      and camera_id = ? and mode = ? and sensitivity = ?
                      and reason = ? and category = ?
                    order by id asc limit 1
                    """,
                    (
                        normalized_related_event_id,
                        camera_id,
                        mode,
                        sensitivity,
                        reason,
                        normalized_category,
                    ),
                ).fetchone()
                if existing is not None:
                    audit_id = int(existing["id"])
                    replaced_snapshot = str(existing["snapshot_path"] or "")
                    persisted_snapshot = snapshot_path or replaced_snapshot
                    compact_features["episode_observation_count"] = (
                        int(existing["trigger_count"] or 1) + normalized_trigger_count
                    )
                    compact_features["episode_last_observed_at"] = created_at
                    features_json = json.dumps(
                        compact_features,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    conn.execute(
                        """
                        update motion_audits
                        set snapshot_path = ?, created_at = ?, score = max(score, ?),
                            threshold = ?, trigger_count = trigger_count + ?,
                            features_json = ?
                        where id = ?
                        """,
                        (
                            persisted_snapshot,
                            created_at,
                            normalized_score,
                            normalized_threshold,
                            normalized_trigger_count,
                            features_json,
                            audit_id,
                        ),
                    )
            if audit_id is None and normalized_decision_id:
                existing = conn.execute(
                    "select id, snapshot_path from motion_audits where decision_id = ?",
                    (normalized_decision_id,),
                ).fetchone()
                if existing is not None:
                    audit_id = int(existing["id"])
                    replaced_snapshot = str(existing["snapshot_path"] or "")
                    persisted_snapshot = snapshot_path or replaced_snapshot
                    conn.execute(
                        """
                        update motion_audits
                        set event_id = coalesce(?, event_id),
                            related_event_id = coalesce(?, related_event_id), camera_id = ?,
                            snapshot_path = ?, created_at = ?, mode = ?,
                            sensitivity = ?, score = ?, threshold = ?, reason = ?,
                            object_detected = ?, trigger_count = ?, features_json = ?,
                            category = ?
                        where id = ?
                        """,
                        (
                            event_id,
                            normalized_related_event_id,
                            camera_id,
                            persisted_snapshot,
                            created_at,
                            mode,
                            sensitivity,
                            normalized_score,
                            normalized_threshold,
                            reason,
                            normalized_object_detected,
                            normalized_trigger_count,
                            features_json,
                            normalized_category,
                            audit_id,
                        ),
                    )
            elif audit_id is None and event_id is None:
                existing = conn.execute(
                    """
                    select id from motion_audits
                    where event_id is null and camera_id = ? and created_at = ?
                      and mode = ? and sensitivity = ? and reason = ?
                      and snapshot_path = ? and score = ? and threshold = ?
                      and object_detected is ? and trigger_count = ?
                      and related_event_id is ?
                      and features_json = ? and category = ?
                    order by id asc limit 1
                    """,
                    (
                        camera_id,
                        created_at,
                        mode,
                        sensitivity,
                        reason,
                        snapshot_path,
                        normalized_score,
                        normalized_threshold,
                        normalized_object_detected,
                        normalized_trigger_count,
                        normalized_related_event_id,
                        features_json,
                        normalized_category,
                    ),
                ).fetchone()
                if existing is not None:
                    audit_id = int(existing["id"])
            if audit_id is None:
                cursor = conn.execute(
                    """
                    insert or ignore into motion_audits (
                        event_id, related_event_id, decision_id, camera_id, snapshot_path, created_at, mode,
                        sensitivity, score, threshold, reason, object_detected,
                        trigger_count, features_json, category
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        normalized_related_event_id,
                        normalized_decision_id or None,
                        camera_id,
                        snapshot_path,
                        created_at,
                        mode,
                        sensitivity,
                        normalized_score,
                        normalized_threshold,
                        reason,
                        normalized_object_detected,
                        normalized_trigger_count,
                        features_json,
                        normalized_category,
                    ),
                )
                if cursor.rowcount:
                    audit_id = int(cursor.lastrowid)
                elif normalized_decision_id:
                    existing = conn.execute(
                        "select id from motion_audits where decision_id = ?",
                        (normalized_decision_id,),
                    ).fetchone()
                    if existing is not None:
                        audit_id = int(existing["id"])
                if audit_id is None and event_id is not None:
                    existing = conn.execute(
                        "select id from motion_audits where event_id = ?",
                        (int(event_id),),
                    ).fetchone()
                    if existing is not None:
                        audit_id = int(existing["id"])
            if audit_id is None:
                raise RuntimeError("motion audit could not be persisted or resolved")
        if replaced_snapshot and replaced_snapshot != persisted_snapshot:
            self._delete_snapshot_if_unreferenced(replaced_snapshot)
        return self.get_motion_audit(audit_id) or {}

    @staticmethod
    def _compact_audit_features(
        features: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        compact = copy.deepcopy(features)
        configurations: dict[str, str] = {}
        telemetry = compact.get("pipeline_telemetry")
        graphs = telemetry.get("graphs") if isinstance(telemetry, dict) else None
        if isinstance(graphs, dict):
            for graph in graphs.values():
                if not isinstance(graph, dict) or "configuration" not in graph:
                    continue
                configuration = graph.pop("configuration")
                serialized = json.dumps(
                    configuration,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
                graph["configuration_fingerprint"] = fingerprint
                configurations[fingerprint] = serialized
        return compact, configurations

    @staticmethod
    def _hydrate_audit_rows(
        conn: sqlite3.Connection,
        rows: list[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        decoded: list[tuple[dict[str, Any], dict[str, Any]]] = []
        fingerprints: set[str] = set()
        for row in rows:
            payload = dict(row)
            try:
                features = json.loads(str(payload.get("features_json") or "{}"))
            except (json.JSONDecodeError, TypeError):
                features = {}
            telemetry = features.get("pipeline_telemetry") if isinstance(features, dict) else None
            graphs = telemetry.get("graphs") if isinstance(telemetry, dict) else None
            if isinstance(graphs, dict):
                for graph in graphs.values():
                    if not isinstance(graph, dict):
                        continue
                    fingerprint = graph.get("configuration_fingerprint")
                    if isinstance(fingerprint, str) and fingerprint:
                        fingerprints.add(fingerprint)
            decoded.append((payload, features))

        configurations: dict[str, Any] = {}
        if fingerprints:
            placeholders = ",".join("?" for _ in fingerprints)
            config_rows = conn.execute(
                f"select fingerprint, configuration_json "
                f"from motion_audit_pipeline_configs where fingerprint in ({placeholders})",
                sorted(fingerprints),
            ).fetchall()
            for config_row in config_rows:
                try:
                    configurations[str(config_row["fingerprint"])] = json.loads(
                        str(config_row["configuration_json"])
                    )
                except (json.JSONDecodeError, TypeError):
                    continue

        hydrated: list[dict[str, Any]] = []
        for payload, features in decoded:
            telemetry = features.get("pipeline_telemetry") if isinstance(features, dict) else None
            graphs = telemetry.get("graphs") if isinstance(telemetry, dict) else None
            if isinstance(graphs, dict):
                for graph in graphs.values():
                    if not isinstance(graph, dict):
                        continue
                    configuration = configurations.get(graph.get("configuration_fingerprint"))
                    if configuration is not None:
                        graph["configuration"] = configuration
            payload["features_json"] = json.dumps(
                features,
                separators=(",", ":"),
                allow_nan=False,
            )
            hydrated.append(payload)
        return hydrated

    def motion_audits(
        self,
        *,
        limit: int = 24,
        offset: int = 0,
        camera_id: str = "",
        outcome: str = "all",
        category: str = "all",
        include_incident_activity: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        values: list[Any] = []
        if not include_incident_activity:
            clauses.append("reason not in ('event_state_active', 'event_state_cooldown')")
        if camera_id:
            clauses.append("camera_id = ?")
            values.append(camera_id)
        if category != "all":
            clauses.append("category = ?")
            values.append(category)
        if outcome == "object":
            clauses.append("object_detected = 1")
        elif outcome == "clear":
            clauses.append("object_detected = 0")
        elif outcome == "not_run":
            clauses.append("object_detected is null")
        where = f"where {' and '.join(clauses)}" if clauses else ""
        bounded_limit = max(1, min(int(limit), 100))
        bounded_offset = max(0, int(offset))
        with self._connect() as conn:
            total = int(conn.execute(
                f"select count(*) from motion_audits {where}",
                values,
            ).fetchone()[0])
            rows = conn.execute(
                f"""
                select * from motion_audits
                {where}
                order by created_at desc, id desc
                limit ? offset ?
                """,
                [*values, bounded_limit, bounded_offset],
            ).fetchall()
            hydrated = self._hydrate_audit_rows(conn, list(rows))
        return hydrated, total

    def motion_audits_page_between(
        self,
        start_at: str,
        end_at: str,
        *,
        limit: int = 500,
        before_created_at: str | None = None,
        before_id: int | None = None,
        camera_ids: tuple[str, ...] = (),
        require_snapshot: bool = True,
        exclude_confirmed_objects: bool = True,
    ) -> list[dict[str, Any]]:
        """Return stable, newest-first motion-audit training candidates."""
        bounded_limit = max(1, min(int(limit), 5000))
        normalized_cameras = tuple(dict.fromkeys(
            str(camera_id).strip()
            for camera_id in camera_ids
            if str(camera_id).strip()
        ))
        clauses = [
            "created_at >= ?",
            "created_at < ?",
            "reason not in ('event_state_active', 'event_state_cooldown')",
        ]
        parameters: list[Any] = [start_at, end_at]
        if before_created_at is not None and before_id is not None:
            clauses.append("(created_at < ? or (created_at = ? and id < ?))")
            parameters.extend([before_created_at, before_created_at, int(before_id)])
        if normalized_cameras:
            placeholders = ",".join("?" for _ in normalized_cameras)
            clauses.append(f"camera_id in ({placeholders})")
            parameters.extend(normalized_cameras)
        if require_snapshot:
            clauses.append("snapshot_path != ''")
        if exclude_confirmed_objects:
            clauses.append("object_detected is not 1")
        parameters.append(bounded_limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select * from motion_audits
                where {' and '.join(clauses)}
                order by created_at desc, id desc
                limit ?
                """,
                parameters,
            ).fetchall()
            return self._hydrate_audit_rows(conn, list(rows))

    def create_motion_ai_review(self, camera_id: str, audits_considered: int) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert into motion_ai_reviews (
                    camera_id, status, audits_considered, created_at, updated_at
                ) values (?, 'queued', ?, ?, ?)
                """,
                (camera_id, max(0, int(audits_considered)), now, now),
            )
            review_id = int(cursor.lastrowid)
        return self.get_motion_ai_review(review_id) or {}

    def update_motion_ai_review(
        self,
        review_id: int,
        *,
        status: str,
        images_available: int | None = None,
        analyzed: int | None = None,
        failed: int | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        allowed_statuses = {"queued", "running", "completed", "failed", "interrupted"}
        if status not in allowed_statuses:
            raise ValueError("invalid motion AI review status")
        result_json = None if result is None else json.dumps(
            result,
            separators=(",", ":"),
            allow_nan=False,
        )
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            current = conn.execute(
                "select status from motion_ai_reviews where id = ?",
                (int(review_id),),
            ).fetchone()
            if current is None:
                raise KeyError("motion AI review not found")
            current_status = str(current["status"])
            if current_status in {"completed", "failed", "interrupted"}:
                return self._result_json_row(
                    conn.execute(
                        "select * from motion_ai_reviews where id = ?",
                        (int(review_id),),
                    ).fetchone()
                ) or {}
            cursor = conn.execute(
                """
                update motion_ai_reviews
                set status = ?,
                    images_available = coalesce(?, images_available),
                    analyzed = coalesce(?, analyzed),
                    failed = coalesce(?, failed),
                    result_json = coalesce(?, result_json),
                    error = coalesce(?, error),
                    updated_at = ?
                where id = ?
                """,
                (
                    status,
                    images_available,
                    analyzed,
                    failed,
                    result_json,
                    error,
                    now,
                    int(review_id),
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError("motion AI review not found")
        return self.get_motion_ai_review(review_id) or {}

    def get_motion_ai_review(self, review_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from motion_ai_reviews where id = ?",
                (int(review_id),),
            ).fetchone()
        return self._result_json_row(row)

    def latest_motion_ai_review(self, camera_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                select * from motion_ai_reviews
                where camera_id = ?
                order by created_at desc, id desc limit 1
                """,
                (camera_id,),
            ).fetchone()
        return self._result_json_row(row)

    @staticmethod
    def _camera_intelligence_evaluation_row(
        row: sqlite3.Row | None,
    ) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        for column, target in (
            ("applied_changes_json", "applied_changes"),
            ("baseline_result_json", "baseline_result"),
            ("followup_result_json", "followup_result"),
            ("comparison_json", "comparison"),
        ):
            try:
                decoded = json.loads(str(payload.pop(column) or "{}"))
            except (json.JSONDecodeError, TypeError):
                decoded = [] if target == "applied_changes" else {}
            payload[target] = decoded
        try:
            applied_at = datetime.fromisoformat(
                str(payload.get("applied_at") or "").replace("Z", "+00:00")
            )
            if applied_at.tzinfo is None:
                applied_at = applied_at.replace(tzinfo=timezone.utc)
            ready_at = applied_at + timedelta(
                hours=float(payload.get("evaluation_hours") or 24)
            )
            payload["ready_at"] = ready_at.isoformat()
            remaining = (ready_at - datetime.now(timezone.utc)).total_seconds()
            payload["seconds_until_ready"] = max(0, round(remaining))
            if payload.get("status") == "collecting" and remaining <= 0:
                payload["status"] = "ready"
        except (TypeError, ValueError):
            payload["ready_at"] = ""
            payload["seconds_until_ready"] = 0
        return payload

    def create_camera_intelligence_evaluation(
        self,
        *,
        camera_id: str,
        baseline_review_id: int,
        evaluation_hours: float,
        applied_changes: list[dict[str, Any]],
        baseline_result: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                insert into camera_intelligence_evaluations (
                    camera_id, baseline_review_id, evaluation_hours,
                    applied_changes_json, baseline_result_json, applied_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    int(baseline_review_id),
                    max(24.0, min(float(evaluation_hours), 168.0)),
                    json.dumps(applied_changes, separators=(",", ":"), allow_nan=False),
                    json.dumps(baseline_result, separators=(",", ":"), allow_nan=False),
                    now,
                    now,
                ),
            )
            evaluation_id = int(cursor.lastrowid)
        return self.get_camera_intelligence_evaluation(evaluation_id) or {}

    def get_camera_intelligence_evaluation(
        self,
        evaluation_id: int,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from camera_intelligence_evaluations where id = ?",
                (int(evaluation_id),),
            ).fetchone()
        return self._camera_intelligence_evaluation_row(row)

    def latest_camera_intelligence_evaluation(
        self,
        camera_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                select * from camera_intelligence_evaluations
                where camera_id = ?
                order by applied_at desc, id desc limit 1
                """,
                (camera_id,),
            ).fetchone()
        return self._camera_intelligence_evaluation_row(row)

    def start_camera_intelligence_followup(
        self,
        evaluation_id: int,
        followup_review_id: int,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                update camera_intelligence_evaluations
                set status = 'reviewing', followup_review_id = ?, error = '', updated_at = ?
                where id = ? and status = 'collecting'
                """,
                (int(followup_review_id), now, int(evaluation_id)),
            )
            if cursor.rowcount != 1:
                raise ValueError("effectiveness follow-up is already running or complete")
        return self.get_camera_intelligence_evaluation(evaluation_id) or {}

    def complete_camera_intelligence_evaluation(
        self,
        evaluation_id: int,
        *,
        followup_result: dict[str, Any],
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                update camera_intelligence_evaluations
                set status = 'completed', followup_result_json = ?, comparison_json = ?,
                    error = '', updated_at = ?, completed_at = ?
                where id = ? and status = 'reviewing'
                """,
                (
                    json.dumps(followup_result, separators=(",", ":"), allow_nan=False),
                    json.dumps(comparison, separators=(",", ":"), allow_nan=False),
                    now,
                    now,
                    int(evaluation_id),
                ),
            )
        return self.get_camera_intelligence_evaluation(evaluation_id) or {}

    def reset_camera_intelligence_followup(
        self,
        evaluation_id: int,
        error: str,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                update camera_intelligence_evaluations
                set status = 'collecting', followup_review_id = null,
                    error = ?, updated_at = ?
                where id = ? and status = 'reviewing'
                """,
                (str(error), now, int(evaluation_id)),
            )
        return self.get_camera_intelligence_evaluation(evaluation_id) or {}

    def motion_effectiveness(self, *, days: float = 7.0) -> dict[str, Any]:
        """Summarize durable motion decisions without conflating visual filters and deduplication."""
        bounded_days = min(90.0, max(1.0 / 24.0, float(days)))
        since = (datetime.now(timezone.utc) - timedelta(days=bounded_days)).isoformat()
        with self._connect() as conn:
            event_rows = conn.execute(
                """
                select camera_id, objects_json
                from events
                where kind = 'motion' and created_at >= ?
                  and objects_json like '%motion_qualification%'
                """,
                (since,),
            ).fetchall()
            audit_rows = conn.execute(
                """
                select camera_id, mode, reason, event_id, object_detected,
                       trigger_count, features_json, category
                from motion_audits
                where created_at >= ?
                """,
                (since,),
            ).fetchall()

        summaries: dict[tuple[str, str], dict[str, Any]] = {}

        def summary_for(camera_id: str, mode: str) -> dict[str, Any]:
            return summaries.setdefault((camera_id, mode), {
                "allowed_events": 0,
                "object_events": 0,
                "no_object_events": 0,
                "borderline_rescued": 0,
                "suppression_verification_checks": 0,
                "suppression_verification_rescues": 0,
                "visual_filtered": 0,
                "state_deduplicated": 0,
                "unreviewed_visual_filters": 0,
                "visual_backup_attempts": 0,
                "visual_backup_objects": 0,
                "visual_backup_no_object": 0,
                "visual_backup_incomplete": 0,
                "visual_backup_not_ready": 0,
                "visual_backup_below_threshold": 0,
                "active_followup_attempts": 0,
                "active_followup_objects": 0,
                "active_followup_no_object": 0,
                "active_followup_incomplete": 0,
                "depth_shadow": {
                    "decisions": 0,
                    "objects_evaluated": 0,
                    "valid_depth": 0,
                    "near_depth": 0,
                    "would_admit": 0,
                    "alignment_reliable": 0,
                    "spatial_match": 0,
                    "stable_geometry": 0,
                    "correlation_accepted": 0,
                    "correlation_rejected": 0,
                },
            })

        for row in event_rows:
            raw_objects = str(row["objects_json"] or "[]")
            qualification = self._qualification_from_objects(raw_objects)
            if not qualification:
                continue
            mode = str(qualification.get("mode") or "unknown")
            summary = summary_for(str(row["camera_id"]), mode)
            try:
                objects = json.loads(raw_objects)
            except (json.JSONDecodeError, TypeError):
                objects = []
            object_detected = bool(
                isinstance(objects, list)
                and any(
                    isinstance(item, dict)
                    and item.get("label")
                    and item.get("incident_eligible") is not False
                    for item in objects
                )
            )
            summary["allowed_events"] += 1
            summary["object_events" if object_detected else "no_object_events"] += 1
            if qualification.get("trigger_source") == "visual_backup":
                summary["visual_backup_attempts"] += 1
                summary["visual_backup_objects"] += int(object_detected)
            summary["borderline_rescued"] += int(
                bool(qualification.get("borderline_candidate"))
            )
            summary["suppression_verification_checks"] += int(
                bool(qualification.get("suppression_verification_candidate"))
            )
            summary["suppression_verification_rescues"] += int(
                bool(qualification.get("suppression_verification_rescued"))
            )

        for row in audit_rows:
            try:
                features = json.loads(str(row["features_json"] or "{}"))
            except (json.JSONDecodeError, TypeError):
                features = {}
            depth_attribution = (
                features.get("depth_attribution")
                if isinstance(features, dict)
                else None
            )
            if isinstance(depth_attribution, dict):
                summary = summary_for(
                    str(row["camera_id"]), str(row["mode"] or "unknown")
                )
                depth_shadow = summary["depth_shadow"]
                depth_shadow["decisions"] += 1
                if row["object_detected"] is not None:
                    outcome = (
                        "correlation_accepted"
                        if bool(row["object_detected"])
                        else "correlation_rejected"
                    )
                    depth_shadow[outcome] += 1
                for key, field in (
                    ("evaluated_count", "objects_evaluated"),
                    ("valid_depth_count", "valid_depth"),
                    ("near_depth_count", "near_depth"),
                    ("would_admit_count", "would_admit"),
                    ("alignment_reliable_count", "alignment_reliable"),
                    ("spatial_match_count", "spatial_match"),
                    ("stable_geometry_count", "stable_geometry"),
                ):
                    depth_shadow[field] += max(0, int(depth_attribution.get(key) or 0))
            if row["event_id"] is not None:
                continue
            summary = summary_for(str(row["camera_id"]), str(row["mode"] or "unknown"))
            category = str(row["category"] or "qualification")
            if category == "visual_backup":
                visual_reason = str(row["reason"] or "")
                if visual_reason == "startup_not_ready":
                    summary["visual_backup_not_ready"] += 1
                    continue
                if visual_reason == "visual_backup_below_threshold":
                    summary["visual_backup_below_threshold"] += 1
                    continue
                summary["visual_backup_attempts"] += 1
                if row["object_detected"] is None:
                    summary["visual_backup_incomplete"] += 1
                elif bool(row["object_detected"]):
                    summary["visual_backup_objects"] += 1
                else:
                    summary["visual_backup_no_object"] += 1
                continue
            if category == "active_followup":
                summary["active_followup_attempts"] += 1
                if row["object_detected"] is None:
                    summary["active_followup_incomplete"] += 1
                elif bool(row["object_detected"]):
                    summary["active_followup_objects"] += 1
                else:
                    summary["active_followup_no_object"] += 1
                continue
            reason = str(row["reason"] or "")
            if reason.startswith("event_state_"):
                summary["state_deduplicated"] += max(
                    1, int(row["trigger_count"] or 1)
                )
            else:
                summary["visual_filtered"] += 1
                summary["suppression_verification_checks"] += int(
                    bool(features.get("suppression_verification"))
                )
                summary["unreviewed_visual_filters"] += int(
                    row["object_detected"] is None
                )

        by_camera: dict[str, dict[str, dict[str, Any]]] = {}
        for (camera_id, mode), summary in summaries.items():
            decisions = (
                summary["allowed_events"]
                + summary["visual_filtered"]
                + summary["state_deduplicated"]
            )
            visual_opportunities = summary["allowed_events"] + summary["visual_filtered"]
            summary.update({
                "total_decisions": decisions,
                "visual_rejection_rate": round(
                    summary["visual_filtered"] / max(1, visual_opportunities),
                    4,
                ),
                "object_yield_rate": round(
                    summary["object_events"] / max(1, summary["allowed_events"]),
                    4,
                ),
            })
            by_camera.setdefault(camera_id, {})[mode] = summary
        return {
            "days": bounded_days,
            "since": since,
            "by_camera": by_camera,
        }

    def get_motion_audit(self, audit_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from motion_audits where id = ?",
                (int(audit_id),),
            ).fetchone()
            hydrated = self._hydrate_audit_rows(
                conn,
                [row] if row is not None else [],
            )
        return hydrated[0] if hydrated else None

    def motion_audits_for_related_events(self, event_ids: list[int]) -> list[dict[str, Any]]:
        unique_ids = sorted({int(event_id) for event_id in event_ids if int(event_id) > 0})
        if not unique_ids:
            return []
        audits: list[dict[str, Any]] = []
        with self._connect() as conn:
            for offset in range(0, len(unique_ids), 500):
                chunk = unique_ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    select * from motion_audits
                    where related_event_id in ({placeholders})
                    order by created_at asc, id asc
                    """,
                    chunk,
                ).fetchall()
                audits.extend(self._hydrate_audit_rows(conn, list(rows)))
        return audits

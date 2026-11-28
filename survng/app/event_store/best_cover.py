"""Durable whole-episode cover jobs ("best cover" pass).

One job per closed camera episode, owned by the episode's first event (the
event whose cover the incident shows). The job is presentation work only: it
never changes admission, establishment, activity clocks, or alert decisions,
and its promotion commits without an external notification. The lifecycle
mirrors ``event_cover_requirements``: lease, bounded attempts, a deadline, and
a deferral that does not charge an attempt when security work preempts it.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Any

from ..database_polling import polling_connection
from ..incident_utils import portable_media_path, snapshot_deletion_claimed

BEST_COVER_TERMINAL_STATES = frozenset({"done", "skipped", "exhausted"})
# Matches the fixed episode inactivity grace in scene admission: an episode
# whose last activity is older than this cannot be joined or extended by
# live activity any more, so its window is known.
BEST_COVER_SETTLE_SECONDS = 45.0
# Episodes that closed longer ago than this are never discovered, so enabling
# the pass does not backfill history.
BEST_COVER_MAX_AGE_SECONDS = 3600.0
BEST_COVER_JOB_DEADLINE_SECONDS = 1800.0
BEST_COVER_MAX_ATTEMPTS = 3
# A fully framed cover is replaced only by a materially larger, still sharp
# view. Same rule as tracking's cover upgrade (object_track/session.py).
BEST_COVER_MIN_EDGE_CLEARANCE = 0.01


def best_cover_replaces(
    *,
    baseline_fully_framed: bool,
    baseline_area: float,
    baseline_pixels: float,
    baseline_quality: float,
    candidate_area: float,
    candidate_pixels: float,
    candidate_clearance: float,
    candidate_quality: float,
) -> bool:
    """Whether a candidate view is materially better than the stored cover."""
    if candidate_clearance < BEST_COVER_MIN_EDGE_CLEARANCE:
        return False
    if candidate_quality < max(0.12, 0.55 * max(0.0, baseline_quality)):
        return False
    if baseline_fully_framed:
        return candidate_area >= max(baseline_area * 1.5, baseline_area + 0.0025)
    return candidate_pixels > baseline_pixels


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _box(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = item.get("box")
    if not isinstance(box, dict):
        return None
    try:
        values = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in values) or values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values  # type: ignore[return-value]


def best_cover_target(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The single admitted subject the cover depicts, or None when ambiguous.

    Label compatibility is not identity proof, so an event with more than one
    candidate subject keeps its cover for tracking to decide.
    """
    labelled = [item for item in objects if isinstance(item, dict) and item.get("label")
                and not item.get("status") and not item.get("auxiliary_detection")
                and item.get("incident_eligible") is not False]
    for key in ("snapshot_primary_subject", "provisional_detection"):
        chosen = [item for item in labelled if item.get(key) is True]
        if chosen:
            return chosen[0] if len(chosen) == 1 else None
    return labelled[0] if len(labelled) == 1 else None


def best_cover_baseline(target: dict[str, Any]) -> dict[str, float | bool]:
    """Stored cover metrics for the target, derived from geometry when absent."""
    width = int(_float(target.get("detection_frame_width")))
    height = int(_float(target.get("detection_frame_height")))
    box = _box(target)
    visible = target.get("snapshot_visible") is not False and box is not None and width > 0 and height > 0
    if not visible:
        return {"area": 0.0, "clearance": 0.0, "quality": 0.0, "pixels": 0.0, "fully_framed": False}
    x1, y1, x2, y2 = box  # type: ignore[misc]
    area = _float(target.get("snapshot_subject_area_ratio"), (x2 - x1) * (y2 - y1) / float(width * height))
    clearance = _float(target.get("snapshot_edge_clearance_ratio"), max(0.0, min(
        x1 / width, y1 / height, (width - x2) / width, (height - y2) / height)))
    return {
        "area": max(0.0, area), "clearance": max(0.0, clearance),
        "quality": max(0.0, _float(target.get("snapshot_quality_score"))),
        "pixels": max(0.0, area) * width * height,
        "fully_framed": clearance >= BEST_COVER_MIN_EDGE_CLEARANCE,
    }


_FRAME_SPECIFIC_COVER_FIELDS = (
    "snapshot_frame_reference", "snapshot_timestamp_exact", "snapshot_timestamp_uncertainty_seconds",
    "snapshot_sharpness_score", "snapshot_exposure_score", "snapshot_edge_detail_score",
    "snapshot_contrast_score",
)


class EventStoreBestCoverMixin:
    def _init_best_cover_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript("""
                create table if not exists event_best_cover_jobs (
                    event_id integer primary key references events(id) on delete cascade,
                    episode_id text not null, camera_id text not null,
                    state text not null check(state in ('pending','done','skipped','exhausted')),
                    available_at_epoch real not null, deadline_epoch real not null,
                    attempts integer not null default 0, reason text not null default '',
                    lease_owner text not null default '', lease_expires_at_epoch real,
                    result_json text not null default '{}',
                    created_at real not null, updated_at real not null
                );
                create unique index if not exists idx_best_cover_episode on event_best_cover_jobs(episode_id);
                create index if not exists idx_best_cover_due
                    on event_best_cover_jobs(camera_id, state, available_at_epoch);
            """)

    # Tracking that can still run, or a pending cover requirement, owns the
    # cover first; the pass waits rather than racing either writer. Demand
    # work whose viewer left is not waiting to run and does not block.
    _BEST_COVER_BLOCKED_SQL = (
        "exists(select 1 from scene_analysis_jobs j where j.episode_id=b.episode_id "
        "and (j.state='running' or (j.state='queued' and (j.admission='automatic' or j.requested_until>:now)))) "
        "or exists(select 1 from event_cover_requirements r where r.event_id=b.event_id and r.state='pending')"
    )
    _BEST_COVER_DISCOVERY_SQL = (
        "select p.id as episode_id, min(m.event_id) as event_id from scene_episodes p "
        "join scene_incidents i on i.id=p.incident_id "
        "join scene_event_membership m on m.episode_id=p.id "
        "where p.camera_id=:camera and p.last_activity_epoch<=:closed_before and p.last_activity_epoch>=:oldest "
        "and i.state not in ('unconfirmed','merged') "
        "and not exists(select 1 from event_best_cover_jobs b where b.episode_id=p.id) "
        "group by p.id order by p.last_activity_epoch limit :limit"
    )

    def claim_best_cover(
        self, camera_id: str, *, lease_owner: str, lease_seconds: float = 120.0,
        now: float | None = None, settle_seconds: float = BEST_COVER_SETTLE_SECONDS,
        max_age_seconds: float = BEST_COVER_MAX_AGE_SECONDS,
        maximum_attempts: int = BEST_COVER_MAX_ATTEMPTS,
    ) -> dict[str, Any] | None:
        """Discover closed episodes, then lease one unblocked due job."""
        if not lease_owner:
            raise ValueError("best cover lease owner is required")
        now = time.time() if now is None else float(now)
        params = {"camera": camera_id, "now": now, "closed_before": now - settle_seconds,
                  "oldest": now - max_age_seconds, "attempts": int(maximum_attempts)}
        # Expired budgets are settled even while blocked, so a job waiting on
        # tracking cannot stay pending past its deadline.
        due_sql = (
            "select 1 from event_best_cover_jobs b where b.camera_id=:camera and b.state='pending' "
            "and (b.lease_expires_at_epoch is null or b.lease_expires_at_epoch<=:now) "
            "and (b.deadline_epoch<=:now or b.attempts>=:attempts "
            f"or (b.available_at_epoch<=:now and not ({self._BEST_COVER_BLOCKED_SQL}))) limit 1"
        )
        # Idle per-camera workers must not reserve the shared main-DB writer.
        with polling_connection(self.db_path) as conn:
            discover = conn.execute(self._BEST_COVER_DISCOVERY_SQL, {**params, "limit": 1}).fetchone() is not None
            due = discover or conn.execute(due_sql, params).fetchone() is not None
        if not due:
            return None
        with self._lock, self._connect() as conn:
            conn.execute("begin immediate")
            for row in conn.execute(self._BEST_COVER_DISCOVERY_SQL, {**params, "limit": 8}).fetchall():
                conn.execute(
                    "insert or ignore into event_best_cover_jobs (event_id,episode_id,camera_id,state,"
                    "available_at_epoch,deadline_epoch,created_at,updated_at) values(?,?,?,'pending',?,?,?,?)",
                    (row["event_id"], row["episode_id"], camera_id, now, now + BEST_COVER_JOB_DEADLINE_SECONDS, now, now),
                )
            conn.execute(
                "update event_best_cover_jobs set state='exhausted',reason=case when deadline_epoch<=:now "
                "then 'deadline_expired' else 'attempts_exhausted' end,lease_owner='',lease_expires_at_epoch=null,"
                "updated_at=:now where camera_id=:camera and state='pending' "
                "and (deadline_epoch<=:now or attempts>=:attempts) "
                "and (lease_expires_at_epoch is null or lease_expires_at_epoch<=:now)",
                params,
            )
            row = conn.execute(
                "select b.* from event_best_cover_jobs b where b.camera_id=:camera and b.state='pending' "
                "and b.available_at_epoch<=:now and (b.lease_expires_at_epoch is null or b.lease_expires_at_epoch<=:now) "
                f"and not ({self._BEST_COVER_BLOCKED_SQL}) order by b.available_at_epoch,b.event_id limit 1",
                params,
            ).fetchone()
            if row is None:
                return None
            event = conn.execute("select * from events where id=?", (row["event_id"],)).fetchone()
            episode = conn.execute("select * from scene_episodes where id=?", (row["episode_id"],)).fetchone()
            if event is None or episode is None:
                conn.execute("update event_best_cover_jobs set state='skipped',reason='episode_unavailable',"
                             "updated_at=? where event_id=?", (now, row["event_id"]))
                return None
            conn.execute("update event_best_cover_jobs set attempts=attempts+1,lease_owner=?,"
                         "lease_expires_at_epoch=?,updated_at=? where event_id=?",
                         (lease_owner, now + max(1.0, lease_seconds), now, row["event_id"]))
            job = self._decode_best_cover(conn.execute(
                "select * from event_best_cover_jobs where event_id=?", (row["event_id"],)).fetchone())
            job["event"] = dict(event)
            job["episode"] = dict(episode)
            # Subjects of the episode's other events share its window; the
            # pass must not present one of theirs as this event's subject.
            labels: set[str] = set()
            for other in conn.execute("select e.objects_json from scene_event_membership m join events e "
                                      "on e.id=m.event_id where m.episode_id=? and m.event_id!=?",
                                      (row["episode_id"], row["event_id"])):
                try:
                    items = json.loads(other["objects_json"] or "[]")
                except (TypeError, ValueError):
                    items = []
                labels.update(str(item["label"]) for item in items if isinstance(item, dict) and item.get("label")
                              and not item.get("status") and item.get("incident_eligible") is not False)
            job["episode_member_labels"] = sorted(labels)
            return job

    @staticmethod
    def _decode_best_cover(row) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["result"] = json.loads(result.pop("result_json") or "{}")
        return result

    def best_cover_job(self, event_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            return self._decode_best_cover(conn.execute(
                "select * from event_best_cover_jobs where event_id=?", (int(event_id),)).fetchone())

    def finish_best_cover(
        self, event_id: int, *, lease_owner: str, state: str, reason: str,
        result: dict[str, Any] | None = None, retry_delay_seconds: float = 60.0,
        maximum_attempts: int = BEST_COVER_MAX_ATTEMPTS,
    ) -> bool:
        """Settle a leased job; ``retry`` keeps it pending until its budget ends."""
        if state not in {"done", "skipped", "retry"}:
            raise ValueError("best cover result must be done, skipped or retry")
        now = time.time()
        raw = json.dumps(result or {}, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(raw) > 16384:
            # Keep the stored summary valid JSON rather than cutting it.
            raw = json.dumps({"truncated": True, "bytes": len(raw)})
        with self._lock, self._connect() as conn:
            if state == "retry":
                sql = ("update event_best_cover_jobs set state=case when attempts>=? or deadline_epoch<=? "
                       "then 'exhausted' else 'pending' end,available_at_epoch=?,")
                params: tuple[Any, ...] = (maximum_attempts, now, now + max(0.0, retry_delay_seconds))
            else:
                sql, params = "update event_best_cover_jobs set state=?,", (state,)
            return conn.execute(
                sql + "reason=?,result_json=?,lease_owner='',lease_expires_at_epoch=null,updated_at=? "
                "where event_id=? and state='pending' and lease_owner=?",
                (*params, reason[:200], raw, now, int(event_id), lease_owner),
            ).rowcount == 1

    def defer_best_cover(self, event_id: int, *, lease_owner: str, reason: str,
                         delay_seconds: float = 5.0) -> bool:
        """Yield a live lease to security work without charging an attempt."""
        now = time.time()
        with self._lock, self._connect() as conn:
            return conn.execute(
                "update event_best_cover_jobs set attempts=max(0,attempts-1),available_at_epoch=?,reason=?,"
                "lease_owner='',lease_expires_at_epoch=null,updated_at=? "
                "where event_id=? and state='pending' and lease_owner=? and lease_expires_at_epoch>?",
                (now + max(0.0, delay_seconds), reason[:200], now, int(event_id), lease_owner, now),
            ).rowcount == 1

    def promote_best_cover(
        self, event_id: int, *, snapshot_path: str, captured_at: float, frame_width: int,
        frame_height: int, subject: dict[str, Any], cover_metrics: dict[str, Any],
        expected_revision: int, lease_owner: str, diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Replace presentation evidence only; admission facts stay as recorded.

        The transaction re-checks everything the pass decided on: its lease,
        the evidence revision it read, that tracking has not taken the cover,
        that the same single subject is still the target, and that the new
        view is still materially better than what is stored now.
        """

        portable_snapshot = portable_media_path(self.storage_dir, snapshot_path)
        reason = ""
        updated = None
        replaced_snapshot = ""
        if not portable_snapshot or frame_width <= 0 or frame_height <= 0 or not math.isfinite(float(captured_at)):
            reason = "invalid_snapshot_or_geometry"
        elif _box(subject) is None:
            reason = "invalid_subject_geometry"
        else:
            snapshot_size_bytes = self._snapshot_file_size(portable_snapshot)
            try:
                with self._lock, self._connect() as conn:
                    conn.execute("begin immediate")
                    reason, updated, replaced_snapshot = self._promote_best_cover_locked(
                        conn, event_id, portable_snapshot=portable_snapshot,
                        snapshot_size_bytes=snapshot_size_bytes, captured_at=captured_at,
                        frame_width=frame_width, frame_height=frame_height, subject=subject,
                        cover_metrics=cover_metrics, expected_revision=expected_revision,
                        lease_owner=lease_owner,
                    )
            except Exception:
                # Revision races and database errors alike: nothing references
                # the new file, and the retry writes its own.
                self._delete_snapshot_if_unreferenced(portable_snapshot)
                raise
        if updated is None:
            if diagnostics is not None:
                diagnostics["reason"] = reason or "event_unavailable"
            if portable_snapshot:
                self._delete_snapshot_if_unreferenced(portable_snapshot)
            return None
        if replaced_snapshot and replaced_snapshot != portable_snapshot:
            self._delete_snapshot_if_unreferenced(replaced_snapshot)
        return dict(updated)

    def _promote_best_cover_locked(
        self, conn, event_id, *, portable_snapshot, snapshot_size_bytes, captured_at,
        frame_width, frame_height, subject, cover_metrics, expected_revision, lease_owner,
    ) -> tuple[str, Any, str]:
        """Return (decline reason, updated row, replaced snapshot) inside one transaction."""
        from .evidence import EventSnapshotChangedError

        if conn.execute("select 1 from event_best_cover_jobs where event_id=? and state='pending' "
                        "and lease_owner=? and lease_expires_at_epoch>?",
                        (event_id, lease_owner, time.time())).fetchone() is None:
            raise EventSnapshotChangedError("best cover lease changed or expired")
        if snapshot_deletion_claimed(conn, self.storage_dir, portable_snapshot):
            return "snapshot_deletion_claimed", None, ""
        row = conn.execute("select * from events where id=?", (event_id,)).fetchone()
        if row is None:
            return "event_unavailable", None, ""
        self._check_evidence_revision(row, expected_revision)
        try:
            objects = json.loads(str(row["objects_json"] or "[]"))
        except (TypeError, ValueError):
            objects = []
        if not isinstance(objects, list):
            return "invalid_event_objects", None, ""
        if any(isinstance(item, dict) and item.get("snapshot_source") == "object_tracking" for item in objects):
            return "tracking_cover_present", None, ""
        if frame_width * frame_height < self._cover_frame_area(objects):
            return "lower_resolution_than_cover", None, ""
        target = best_cover_target(objects)
        if target is None or str(target.get("label")) != str(subject.get("label")):
            return "ambiguous_cover_subject", None, ""
        baseline = best_cover_baseline(target)
        area = _float(cover_metrics.get("snapshot_subject_area_ratio"))
        if not best_cover_replaces(
            baseline_fully_framed=bool(baseline["fully_framed"]), baseline_area=float(baseline["area"]),
            baseline_pixels=float(baseline["pixels"]), baseline_quality=float(baseline["quality"]),
            candidate_area=area, candidate_pixels=area * frame_width * frame_height,
            candidate_clearance=_float(cover_metrics.get("snapshot_edge_clearance_ratio")),
            candidate_quality=_float(cover_metrics.get("snapshot_quality_score")),
        ):
            return "not_materially_better", None, ""
        captured_iso = datetime.fromtimestamp(float(captured_at), timezone.utc).isoformat()
        for item in objects:
            if not isinstance(item, dict) or not item.get("label") or item.get("status"):
                continue
            item.pop("mask_polygon", None)
            if item is not target:
                # No identity is claimed for other subjects in this frame.
                item["snapshot_visible"] = False
                continue
            for stale in _FRAME_SPECIFIC_COVER_FIELDS:
                # These describe the previous cover frame, not this one.
                item.pop(stale, None)
            item["box"] = {key: float(subject["box"][key]) for key in ("x1", "y1", "x2", "y2")}
            item["detection_frame_width"] = int(frame_width)
            item["detection_frame_height"] = int(frame_height)
            item["snapshot_visible"] = True
            item["snapshot_primary_subject"] = True
            item["snapshot_source"] = "best_cover"
            item["snapshot_captured_at"] = captured_iso
            item["snapshot_detection_confidence"] = _float(subject.get("confidence"))
            for key, value in cover_metrics.items():
                if key.startswith("snapshot_"):
                    item[key] = value
        conn.execute("update events set snapshot_path=?,snapshot_size_bytes=?,objects_json=? where id=?",
                     (portable_snapshot, snapshot_size_bytes, json.dumps(objects, separators=(",", ":")), event_id))
        # After-the-fact presentation: browser and indexes refresh, external
        # notifications and the activity clock do not.
        updated = self._finish_evidence_commit(conn, event_id, row, reason="best_cover",
                                               notify=False, activity=False)
        return ("", updated, str(row["snapshot_path"] or "")) if updated is not None else ("event_unavailable", None, "")

    def _best_cover_blocks_tracking_cover(self, objects: list[dict[str, Any]], frame_width: int,
                                          frame_height: int, cover_metrics: dict[str, Any]) -> bool:
        """Tracking may replace a best-cover image only with a materially better view."""
        target = next((item for item in objects if isinstance(item, dict)
                       and item.get("snapshot_source") == "best_cover"), None)
        if target is None:
            return False
        baseline = best_cover_baseline(target)
        area = _float(cover_metrics.get("snapshot_subject_area_ratio"))
        return not best_cover_replaces(
            baseline_fully_framed=bool(baseline["fully_framed"]), baseline_area=float(baseline["area"]),
            baseline_pixels=float(baseline["pixels"]), baseline_quality=float(baseline["quality"]),
            candidate_area=area, candidate_pixels=area * frame_width * frame_height,
            candidate_clearance=_float(cover_metrics.get("snapshot_edge_clearance_ratio")),
            candidate_quality=_float(cover_metrics.get("snapshot_quality_score")),
        )

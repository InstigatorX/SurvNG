"""Scheduled site review: rank bounded suggestions against the whole system.

Daily evidence is read from stored rows. The weekly pass reuses calibration
recommendations and drops any change that fights site load or zone policy.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any


# One registry: a weekly pass may propose these settings, and only at this scope.
# Classes and the admin catalog are derived from it.
_CAMERA_OR_GLOBAL = frozenset({"camera", "global"})
AUTOMATIC_SETTING_SPECS: dict[str, dict[str, Any]] = {
    "motion.sensitivity": {"class_id": "motion_sensitivity", "scopes": _CAMERA_OR_GLOBAL},
    "motion.stationary_object_tolerance": {"class_id": "motion_sensitivity", "scopes": _CAMERA_OR_GLOBAL},
    "motion.frame_width": {"class_id": "motion_analysis", "scopes": _CAMERA_OR_GLOBAL},
    "motion.visual_backup_grace_seconds": {"class_id": "visual_backup", "scopes": _CAMERA_OR_GLOBAL},
    "motion.visual_backup_min_score": {"class_id": "visual_backup", "scopes": _CAMERA_OR_GLOBAL},
    "motion.visual_backup_min_consecutive": {"class_id": "visual_backup", "scopes": _CAMERA_OR_GLOBAL},
    "motion.visual_backup_cooldown_seconds": {"class_id": "visual_backup", "scopes": _CAMERA_OR_GLOBAL},
    "motion.visual_backup_max_triggers_5m": {"class_id": "visual_backup", "scopes": _CAMERA_OR_GLOBAL},
    "motion.borderline_rescue_enabled": {"class_id": "borderline_rescue", "scopes": _CAMERA_OR_GLOBAL},
    "motion.borderline_margin": {"class_id": "borderline_rescue", "scopes": _CAMERA_OR_GLOBAL},
    "detector.tracking.sample_fps": {"class_id": "tracking", "scopes": frozenset({"global"})},
}
_CLASS_COPY: dict[str, tuple[str, str]] = {
    "motion_sensitivity": (
        "Motion sensitivity",
        "How readily motion qualifies, including stationary-object tolerance.",
    ),
    "motion_analysis": (
        "Motion analysis",
        "Frame size used for visual motion analysis.",
    ),
    "visual_backup": (
        "Visual backup",
        "Rescue score, persistence, cooldown, and how often a rescue may fire.",
    ),
    "borderline_rescue": (
        "Borderline rescue",
        "Whether near-threshold motion is checked again, and how close it must be.",
    ),
    "restore_inherit": (
        "Restore inheritance",
        "Put a camera back on the system value when that does not change what is in effect.",
    ),
}
# Applying these reloads the camera manager, which refuses while any AI work,
# including the weekly review itself, is registered. They stay suggestions.
SUGGESTION_ONLY_CLASSES = frozenset({"tracking"})
_CLASS_SETTINGS: dict[str, set[str]] = {class_id: set() for class_id in _CLASS_COPY}
for _setting, _spec in AUTOMATIC_SETTING_SPECS.items():
    if _spec["class_id"] in _CLASS_SETTINGS:
        _CLASS_SETTINGS[_spec["class_id"]].add(_setting)
AUTOMATIC_CLASSES: dict[str, dict[str, Any]] = {
    class_id: {
        "label": label,
        "detail": detail,
        "settings": set(_CLASS_SETTINGS[class_id]),
    }
    for class_id, (label, detail) in _CLASS_COPY.items()
}
AUTOMATIC_CLASS_IDS = frozenset(AUTOMATIC_CLASSES)
AUTOMATIC_CLASS_ALIASES = {"visual_backup_cooldown": "visual_backup"}
# A single in-flight frame is ordinary. Backlog starts at the next queued item.
QUEUE_BACKLOG_DEPTH = 2
_POLICY_REASONS = ("ignored_zone", "depth_ignore_zone", "ineligible_zone", "below_confidence")

NEVER_AUTOMATIC_SETTINGS = frozenset({
    "detector.confidence_threshold",
    "detector.event_confirmation_frames",
    "detector.require_incident_zone",
    "detector.event_class_confidence_thresholds",
    "detector.event_class_confirmation_frames",
})

_LOAD_INCREASE_SETTINGS = frozenset({
    "detector.tracking.sample_fps",
    "motion.sample_fps",
    "motion.frame_width",
    "detector.event_confirmation_frames",
})


def suggestion_class(recommendation: dict[str, Any]) -> str | None:
    """Return the automatic class for a recommendation the weekly pass can apply."""
    setting = str(recommendation.get("setting") or "")
    if not setting or setting in NEVER_AUTOMATIC_SETTINGS or "confidence" in setting or "require_incident_zone" in setting:
        return None
    spec = AUTOMATIC_SETTING_SPECS.get(setting)
    proposed = recommendation.get("proposed")
    scope = str(recommendation.get("scope") or "")
    scopes = spec["scopes"] if spec else frozenset()
    if proposed in (None, "inherit"):
        if "camera" in scopes and scope in {"", "camera"}:
            return "restore_inherit"
        return None
    if spec is None:
        return None
    if scope and scope not in scopes:
        return None
    return spec["class_id"]


def normalize_automatic_classes(classes: list[str]) -> list[str]:
    """Map retired class ids onto the current breakdown and drop suggestion-only ones."""
    return list(dict.fromkeys(
        AUTOMATIC_CLASS_ALIASES.get(item, item)
        for item in classes
        if item not in SUGGESTION_ONLY_CLASSES
    ))


def automatic_eligible(recommendation: dict[str, Any], allowed: set[str] | frozenset[str]) -> bool:
    class_id = suggestion_class(recommendation)
    if class_id is None:
        return False
    return class_id in set(normalize_automatic_classes(list(allowed)))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _increases_load(recommendation: dict[str, Any]) -> bool:
    setting = str(recommendation.get("setting") or "")
    if setting not in _LOAD_INCREASE_SETTINGS:
        return False
    current = _number(recommendation.get("current", recommendation.get("current_effective")))
    proposed = _number(recommendation.get("proposed"))
    return current is not None and proposed is not None and proposed > current


def _loosens_confidence(recommendation: dict[str, Any]) -> bool:
    setting = str(recommendation.get("setting") or "")
    if "confidence" not in setting:
        return False
    current = _number(recommendation.get("current", recommendation.get("current_effective")))
    proposed = _number(recommendation.get("proposed"))
    return current is not None and proposed is not None and proposed < current


def daily_notes(signals: dict[str, Any]) -> list[dict[str, str]]:
    """Explain site patterns that are not setting changes."""
    reasons = signals.get("establishment_reasons") or {}
    notes: list[dict[str, str]] = []
    ignored = (
        int(reasons.get("ignored_zone") or 0)
        + int(reasons.get("depth_ignore_zone") or 0)
        + int(reasons.get("ineligible_zone") or 0)
    )
    if ignored:
        notes.append({
            "kind": "policy",
            "text": (
                f"{ignored} establishment decisions were outside an eligible zone, past the depth limit, "
                "or in an Ignore zone. Those observations stay in the incident. "
                "Loosening the detector would not make them eligible."
            ),
        })
    below = int(reasons.get("below_confidence") or 0)
    if below:
        notes.append({
            "kind": "policy",
            "text": (
                f"{below} decisions were below the class confidence threshold. "
                "That threshold decides what can open an incident and is not an automatic change."
            ),
        })
    reclaims = int(signals.get("reclaims") or 0)
    if reclaims:
        notes.append({
            "kind": "tracking",
            "text": f"{reclaims} scene tracking jobs were claimed more than once. Playback keeps the earlier samples.",
        })
    return notes


def rank_site_recommendations(
    recommendations: list[dict[str, Any]],
    signals: dict[str, Any],
) -> dict[str, Any]:
    """Drop suggestions that raise load on a busy site or contradict zone policy."""
    notes = list(daily_notes(signals))
    kept: list[dict[str, Any]] = []
    tracking_saturated = bool(signals.get("tracking_saturated"))
    jobs_expiring = bool(signals.get("jobs_expiring"))
    detector_behind = bool(signals.get("detector_behind"))
    reason_counts = signals.get("establishment_reasons") or {}
    policy_block = any(int(reason_counts.get(name) or 0) for name in _POLICY_REASONS)

    for item in recommendations:
        if item.get("dismissed"):
            continue
        setting = str(item.get("setting") or "")
        if _loosens_confidence(item) and policy_block:
            notes.append({
                "kind": "dropped",
                "text": "Detector confidence was not lowered. Zone and class thresholds explain the rejected activity.",
            })
            continue
        if suggestion_class(item) is None:
            notes.append({
                "kind": "dropped",
                "text": f"{setting} stays manual. This review does not apply it.",
            })
            continue
        if setting == "detector.tracking.sample_fps" and tracking_saturated and _increases_load(item):
            notes.append({
                "kind": "dropped",
                "text": "Tracking sample rate was not raised because tracking slots are already full.",
            })
            continue
        if _increases_load(item) and (jobs_expiring or detector_behind):
            notes.append({
                "kind": "dropped",
                "text": f"{setting} was not increased because analysis is already behind or jobs are being reclaimed.",
            })
            continue
        kept.append(item)

    best: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for item in kept:
        key = (str(item.get("setting") or ""), str(item.get("camera_id") or ""))
        current = best.get(key)
        if current is None:
            best[key] = item
            order.append(key)
            continue
        if int(item.get("support_count") or 0) > int(current.get("support_count") or 0):
            notes.append({
                "kind": "dropped",
                "text": f"Kept one {key[0]} change and dropped a weaker conflicting value.",
            })
            best[key] = item
        else:
            notes.append({
                "kind": "dropped",
                "text": f"Dropped a conflicting {key[0]} change with less supporting evidence.",
            })
    ranked = [best[key] for key in order]
    ranked.sort(key=lambda item: (-int(item.get("support_count") or 0), str(item.get("setting") or "")))
    return {"recommendations": ranked, "notes": notes}


def collect_stored_signals(connection: sqlite3.Connection, since_epoch: float) -> dict[str, Any]:
    """Read incident and tracking evidence already stored. Missing tables yield zeros."""

    def rows(sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        try:
            return list(connection.execute(sql, args))
        except sqlite3.OperationalError:
            return []

    reasons: dict[str, int] = {}
    # Rejected decisions store a null activity_epoch. created_at is set for every row.
    for row in rows(
        "select reason, count(*) from scene_activity_decisions where created_at >= ? group by reason",
        (since_epoch,),
    ):
        reasons[str(row[0] or "")] = int(row[1] or 0)
    incidents = [
        {"camera_id": str(row[0] or ""), "count": int(row[1] or 0)}
        for row in rows(
            "select camera_id, count(*) from scene_episodes where start_epoch >= ? group by camera_id",
            (since_epoch,),
        )
    ]
    reclaims = rows(
        "select count(*) from scene_analysis_jobs where attempts > 1 and end_epoch >= ?",
        (since_epoch,),
    )
    running = rows("select count(*) from scene_analysis_jobs where state = 'running'")
    return {
        "establishment_reasons": reasons,
        "incidents_by_camera": incidents,
        "reclaims": int(reclaims[0][0]) if reclaims else 0,
        "tracking_running": int(running[0][0]) if running else 0,
    }


def with_runtime_signals(
    stored: dict[str, Any],
    *,
    tracking_limit: int,
    detector_queue: int = 0,
) -> dict[str, Any]:
    running = int(stored.get("tracking_running") or 0)
    limit = max(1, int(tracking_limit or 1))
    reclaims = int(stored.get("reclaims") or 0)
    return {
        **stored,
        "tracking_limit": limit,
        "tracking_saturated": running >= limit,
        "jobs_expiring": reclaims > 0,
        "detector_behind": int(detector_queue or 0) >= QUEUE_BACKLOG_DEPTH,
    }


def daily_briefing(signals: dict[str, Any]) -> dict[str, Any]:
    incident_count = sum(int(item.get("count") or 0) for item in signals.get("incidents_by_camera") or [])
    notes = daily_notes(signals)
    summary = f"Reviewed stored evidence for {incident_count} camera episodes."
    if notes:
        summary = f"{summary} {len(notes)} site notes, no new setting changes from the daily pass."
    else:
        summary = f"{summary} No site-level setting change was justified."
    return {
        "review_type": "system_review",
        "pass": "daily",
        "summary": summary,
        "recommendations": [],
        "site_notes": notes,
        "signals": {
            "incidents_by_camera": signals.get("incidents_by_camera") or [],
            "establishment_reasons": signals.get("establishment_reasons") or {},
            "reclaims": int(signals.get("reclaims") or 0),
            "tracking_saturated": bool(signals.get("tracking_saturated")),
            "detector_behind": bool(signals.get("detector_behind")),
        },
    }


def detection_camera_ids(cameras: list[Any], statuses: list[dict[str, Any]]) -> list[str]:
    by_id = {str(item.get("id") or item.get("camera_id") or ""): item for item in statuses}
    selected: list[str] = []
    for camera in cameras:
        camera_id = str(getattr(camera, "id", "") or "")
        if not camera_id:
            continue
        status = by_id.get(camera_id) or {}
        if status.get("detection_enabled", True):
            selected.append(camera_id)
    return selected


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return _aware(parsed)


def due_pass(
    now: datetime,
    cadence: str,
    last_daily: datetime | None,
    last_weekly: datetime | None,
) -> str | None:
    """Return the pass that should start, preferring the weekly sample when both are due."""
    if cadence not in {"daily", "weekly"}:
        return None
    now = _aware(now)
    covered = last_daily
    if last_weekly is not None and (covered is None or _aware(last_weekly) > _aware(covered)):
        covered = last_weekly
    daily_due = covered is None or now - _aware(covered) >= timedelta(hours=20)
    weekly_due = cadence == "weekly" and (last_weekly is None or now - _aware(last_weekly) >= timedelta(days=6))
    if weekly_due:
        return "weekly"
    if daily_due:
        return "daily"
    return None


def review_progress(
    *,
    completed: int,
    total: int,
    phase: str,
    camera_id: str = "",
    camera_name: str = "",
    images_done: int | None = None,
    images_total: int | None = None,
) -> dict[str, Any]:
    """Describe where a site review is, including the camera being sampled."""
    payload: dict[str, Any] = {
        "completed": int(completed),
        "total": int(total),
        "phase": phase,
    }
    if camera_id:
        payload["camera_id"] = camera_id
        payload["camera_name"] = camera_name or camera_id
        payload["camera_index"] = min(int(completed) + 1, int(total) or int(completed) + 1)
    if images_total:
        payload["images_done"] = int(images_done or 0)
        payload["images_total"] = int(images_total)
    return payload


def next_run_at(last: datetime | None, *, hours: float, now: datetime) -> datetime:
    now = _aware(now)
    if last is None:
        return now
    candidate = _aware(last) + timedelta(hours=hours)
    return candidate if candidate > now else now

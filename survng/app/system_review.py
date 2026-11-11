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
        "How easily motion counts",
        "How much movement it takes before a camera treats something as motion, including objects that sit still.",
    ),
    "motion_analysis": (
        "Picture size for motion checks",
        "How large a picture SurvNG uses when it looks for motion.",
    ),
    "visual_backup": (
        "Backup alerts from the video",
        "When the camera stays quiet, how sure SurvNG must be, and how often it may raise an alert from the video itself.",
    ),
    "borderline_rescue": (
        "A second look at close calls",
        "Whether motion that almost qualified gets checked again, and how close it has to be.",
    ),
    "restore_inherit": (
        "Follow the site setting",
        "Put a camera back on the site-wide value when that does not change what the camera does today.",
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


# Plain-language copy for every setting a review can suggest.
# stricter_when is internal: higher, lower, higher_choice, lower_choice, or off.
_GUIDE_FIELDS = ("label", "what", "higher", "lower", "unit", "location")
SETTING_GUIDE: dict[str, dict[str, str]] = {
    "motion.sensitivity": {
        "label": "Motion sensitivity",
        "what": "How much movement it takes before the camera treats something as motion.",
        "higher": "High notices smaller movement and raises more alerts.",
        "lower": "Low ignores small movement and raises fewer alerts.",
        "unit": "choice",
        "location": "camera_motion",
        "stricter_when": "lower_choice",
    },
    "motion.stationary_object_tolerance": {
        "label": "Still-object tolerance",
        "what": "How readily a camera ignores something that sits still, such as a parked car.",
        "higher": "High ignores objects that sit still.",
        "lower": "Low keeps alerting while something sits in view.",
        "unit": "choice",
        "location": "camera_motion",
        "stricter_when": "higher_choice",
    },
    "motion.frame_width": {
        "label": "Motion picture size",
        "what": "How wide a picture SurvNG uses when it looks for motion. Larger pictures see more detail and use more of the computer.",
        "higher": "A larger picture can catch smaller movement and uses more processing.",
        "lower": "A smaller picture uses less processing and can miss small movement.",
        "unit": "px",
        "location": "camera_motion",
        "stricter_when": "lower",
    },
    "motion.visual_backup_grace_seconds": {
        "label": "Wait before a backup alert",
        "what": "How long SurvNG waits after the camera goes quiet before it may raise an alert from the video itself.",
        "higher": "A longer wait means fewer backup alerts.",
        "lower": "A shorter wait means backup alerts can start sooner.",
        "unit": "seconds",
        "location": "camera_motion",
        "stricter_when": "higher",
    },
    "motion.visual_backup_min_score": {
        "label": "Backup alert confidence",
        "what": "How sure SurvNG must be before raising an alert from video when the camera sent no motion signal.",
        "higher": "Higher means fewer false alarms but may miss faint movement.",
        "lower": "Lower catches fainter movement and can raise more false alarms.",
        "unit": "percent",
        "location": "camera_motion",
        "stricter_when": "higher",
    },
    "motion.visual_backup_min_consecutive": {
        "label": "Backup alert persistence",
        "what": "How many checks in a row must agree before a backup alert is raised.",
        "higher": "More checks mean fewer brief false alarms.",
        "lower": "Fewer checks raise an alert sooner.",
        "unit": "count",
        "location": "camera_motion",
        "stricter_when": "higher",
    },
    "motion.visual_backup_cooldown_seconds": {
        "label": "Pause between backup alerts",
        "what": "How long SurvNG waits after a backup alert before it can raise another one.",
        "higher": "A longer pause means fewer repeated alerts.",
        "lower": "A shorter pause allows alerts to repeat sooner.",
        "unit": "seconds",
        "location": "camera_motion",
        "stricter_when": "higher",
    },
    "motion.visual_backup_max_triggers_5m": {
        "label": "Backup alerts in five minutes",
        "what": "How many backup alerts one camera may raise in five minutes.",
        "higher": "A higher cap allows more alerts in a short time.",
        "lower": "A lower cap holds alerts back once the camera has been busy.",
        "unit": "count",
        "location": "camera_motion",
        "stricter_when": "lower",
    },
    "motion.borderline_rescue_enabled": {
        "label": "Second look at close calls",
        "what": "Whether motion that almost qualified gets a second check.",
        "higher": "On gives close calls another chance to become an alert.",
        "lower": "Off leaves close calls alone.",
        "unit": "choice",
        "location": "camera_motion",
        "stricter_when": "off",
    },
    "motion.borderline_margin": {
        "label": "Close-call window",
        "what": "How close to the line motion must be before that second check runs.",
        "higher": "A wider margin checks more close calls.",
        "lower": "A narrower margin checks only the closest calls.",
        "unit": "percent",
        "location": "camera_motion",
        "stricter_when": "lower",
    },
    "detector.tracking.sample_fps": {
        "label": "Tracking sample rate",
        "what": "How many times a second SurvNG looks at an object it is already following.",
        "higher": "More samples follow the object more closely and use more of the computer.",
        "lower": "Fewer samples use less of the computer and can lose the object sooner.",
        "unit": "fps",
        "location": "detection",
        "stricter_when": "lower",
    },
}
_TITLES: dict[str, dict[str, str]] = {
    "motion.sensitivity": {
        "stricter": "Make {where} ignore more small movement",
        "looser": "Make {where} notice smaller movement",
    },
    "motion.stationary_object_tolerance": {
        "stricter": "Make {where} ignore objects that sit still",
        "looser": "Make {where} alert on objects that sit still",
    },
    "motion.frame_width": {
        "stricter": "Use a smaller picture when {where} checks for motion",
        "looser": "Use a larger picture when {where} checks for motion",
    },
    "motion.visual_backup_grace_seconds": {
        "stricter": "Give {where} longer before a backup alert",
        "looser": "Let {where} raise a backup alert sooner",
    },
    "motion.visual_backup_min_score": {
        "stricter": "Make {where} less eager to raise visual backup alerts",
        "looser": "Make {where} more willing to raise visual backup alerts",
    },
    "motion.visual_backup_min_consecutive": {
        "stricter": "Ask {where} to see motion longer before a backup alert",
        "looser": "Let {where} raise a backup alert after less persistence",
    },
    "motion.visual_backup_cooldown_seconds": {
        "stricter": "Pause longer between backup alerts on {where}",
        "looser": "Allow backup alerts more often on {where}",
    },
    "motion.visual_backup_max_triggers_5m": {
        "stricter": "Cap how often {where} can raise backup alerts",
        "looser": "Allow {where} to raise backup alerts more often",
    },
    "motion.borderline_rescue_enabled": {
        "stricter": "Stop the second look at close calls on {where}",
        "looser": "Give close calls a second look on {where}",
    },
    "motion.borderline_margin": {
        "stricter": "Narrow the close-call window on {where}",
        "looser": "Widen the close-call window on {where}",
    },
    "detector.tracking.sample_fps": {
        "stricter": "Sample active tracks less often",
        "looser": "Sample active tracks more often",
    },
}
_CHOICE_ORDER = {"low": 0, "balanced": 1, "high": 2}


def public_setting_guide() -> dict[str, dict[str, str]]:
    """Copy the guide without the internal direction rule."""
    return {
        setting: {field: spec[field] for field in _GUIDE_FIELDS}
        for setting, spec in SETTING_GUIDE.items()
    }


def automatic_level(classes: list[str] | None) -> str:
    """Return on, off, or custom for the single automation switch."""
    selected = set(normalize_automatic_classes(list(classes or [])))
    if not selected:
        return "off"
    if selected >= set(AUTOMATIC_CLASSES):
        return "on"
    return "custom"


def review_image_url(value: Any) -> str:
    """Keep sample pictures that are already served on the authenticated API."""
    url = str(value or "")
    if url.startswith("/api/") and not url.startswith("//"):
        return url
    return ""


def _camera_name(config: Any, camera_id: str) -> str:
    for camera in getattr(config, "cameras", []) or []:
        if str(getattr(camera, "id", "") or "") == camera_id:
            return str(getattr(camera, "name", "") or camera_id)
    return camera_id or "This camera"


def _guide(setting: str) -> dict[str, str]:
    spec = SETTING_GUIDE.get(setting)
    if spec:
        return spec
    label = setting.split(".")[-1].replace("_", " ").capitalize() if setting else "Setting"
    return {
        "label": label,
        "what": "SurvNG suggested a change to this setting.",
        "higher": "",
        "lower": "",
        "unit": "",
        "location": "camera_motion",
        "stricter_when": "higher",
    }


def _public_guide(setting: str) -> dict[str, str]:
    spec = _guide(setting)
    return {field: spec.get(field, "") for field in _GUIDE_FIELDS}


def _direction(setting: str, current: Any, proposed: Any) -> str:
    if proposed in (None, "inherit"):
        return "restore"
    rule = _guide(setting).get("stricter_when") or "higher"
    if rule == "off":
        current_on = current is True or current == "true"
        proposed_on = proposed is True or proposed == "true"
        return "looser" if proposed_on and not current_on else "stricter"
    if rule in {"higher", "lower"}:
        current_number = _number(current)
        proposed_number = _number(proposed)
        if current_number is None or proposed_number is None or current_number == proposed_number:
            return "looser"
        increased = proposed_number > current_number
        return "stricter" if increased == (rule == "higher") else "looser"
    current_rank = _CHOICE_ORDER.get(str(current))
    proposed_rank = _CHOICE_ORDER.get(str(proposed))
    if current_rank is None or proposed_rank is None or current_rank == proposed_rank:
        return "looser"
    toward_high = proposed_rank > current_rank
    return "stricter" if toward_high == (rule == "higher_choice") else "looser"


def _where_rows(item: dict[str, Any], config: Any) -> list[dict[str, Any]]:
    preview = item.get("effective_preview") or []
    rows: list[dict[str, Any]] = []
    if isinstance(preview, list) and preview:
        for row in preview:
            if not isinstance(row, dict):
                continue
            camera_id = str(row.get("camera_id") or "")
            current = row.get("current")
            proposed = row.get("proposed")
            rows.append({
                "camera_id": camera_id,
                "camera_name": _camera_name(config, camera_id),
                "current": current,
                "proposed": proposed,
                "inherits": bool(row.get("inherits")),
                "changes": current != proposed,
            })
        return rows
    camera_id = str(item.get("camera_id") or "")
    current = item.get("current_effective", item.get("current"))
    proposed = item.get("proposed")
    if str(item.get("scope") or "") == "global":
        affected = [str(value) for value in item.get("affected_cameras") or [] if value]
        targets = affected or [camera.id for camera in getattr(config, "cameras", []) or []]
        return [
            {
                "camera_id": target,
                "camera_name": _camera_name(config, target),
                "current": current,
                "proposed": proposed,
                "inherits": True,
                "changes": current != proposed,
            }
            for target in targets
        ]
    return [{
        "camera_id": camera_id,
        "camera_name": _camera_name(config, camera_id),
        "current": current,
        "proposed": proposed,
        "inherits": False,
        "changes": current != proposed,
    }]


def describe_recommendation(
    item: dict[str, Any],
    config: Any,
    automatic_classes: list[str] | None,
    apply_enabled: bool,
) -> dict[str, Any]:
    """Explain one suggestion: what changes, where, and whether it can apply itself."""
    setting = str(item.get("setting") or "")
    guide = _public_guide(setting)
    rows = _where_rows(item, config)
    changed = [row for row in rows if row["changes"]]
    unchanged = [row for row in rows if not row["changes"]]
    scope = "global" if str(item.get("scope") or "") == "global" else "camera"
    if scope == "camera":
        place = rows[0]["camera_name"] if rows else "This camera"
        summary = f"{place} only"
    else:
        place = "every camera"
        summary = f"Changes on {len(changed)} " + ("camera" if len(changed) == 1 else "cameras")
        if len(unchanged) == 1:
            summary += ", 1 keeps its own setting"
        elif unchanged:
            summary += f", {len(unchanged)} keep their own setting"
    current = item.get("current_effective", item.get("current"))
    proposed = item.get("proposed")
    direction = _direction(setting, current, proposed)
    if direction == "restore":
        title = f"Put {place} back on the site setting for {guide['label']}"
    else:
        template = _TITLES.get(setting, {}).get(direction) or "{where}: {label}"
        title = template.format(where=place, label=guide["label"])
    class_id = suggestion_class(item)
    allowed = set(normalize_automatic_classes(list(automatic_classes or [])))
    if not apply_enabled:
        auto_status = "disabled"
    elif class_id in AUTOMATIC_CLASSES and class_id in allowed:
        auto_status = "automatic"
    elif class_id in AUTOMATIC_CLASSES:
        auto_status = "approval"
    else:
        auto_status = "manual"
    return {
        "title": title,
        "direction": direction,
        "guide": guide,
        "where": {
            "scope": scope,
            "summary": summary,
            "changed_count": len(changed),
            "unchanged_count": len(unchanged),
            "cameras": rows,
        },
        "auto_status": auto_status,
    }


def at_a_glance(result: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize what a briefing looked at, without the raw reports."""
    payload = result or {}
    reports = payload.get("camera_reports") or {}
    analyzed = 0
    verdicts: dict[str, int] = {}
    if isinstance(reports, dict):
        for report in reports.values():
            if not isinstance(report, dict):
                continue
            analyzed += int(report.get("analyzed") or 0)
            for name, count in (report.get("verdict_counts") or {}).items():
                verdicts[str(name)] = verdicts.get(str(name), 0) + int(count or 0)
    if not analyzed:
        analyzed = int(payload.get("analyzed") or 0)
    summaries = payload.get("camera_summaries") or []
    cameras = len(summaries) if isinstance(summaries, list) and summaries else (len(reports) if isinstance(reports, dict) else 0)
    suggestions = [
        item for item in payload.get("recommendations") or []
        if isinstance(item, dict) and not item.get("dismissed")
    ]
    return {
        "analyzed": analyzed,
        "cameras": cameras,
        "suggestions": len(suggestions),
        "verdicts": verdicts,
        "pass": str(payload.get("pass") or ""),
        "summary": str(payload.get("summary") or ""),
    }


def describe_change_set(change_set: dict[str, Any], config: Any) -> dict[str, Any]:
    """Explain an applied change: who made it, where it is in the watch, and the result."""
    action = str(change_set.get("action") or "apply")
    status = str(change_set.get("status") or "")
    evaluation = change_set.get("evaluation") if isinstance(change_set.get("evaluation"), dict) else {}
    automatic = bool((change_set.get("apply_result") or {}).get("automatic"))
    if action == "rollback":
        origin = "rollback"
    elif automatic:
        origin = "automatic"
    else:
        origin = "you"
    rolled_back = status in {"rolled_back", "partially_rolled_back"} or bool(evaluation.get("rolled_back"))
    if rolled_back or action == "rollback":
        phase = "rolled_back"
    elif status == "reviewing":
        phase = "checking"
    elif status == "collecting":
        phase = "watching"
    else:
        phase = "result"
    created = parse_timestamp(change_set.get("created_at"))
    try:
        hours = float(change_set.get("evaluation_hours") or 0)
    except (TypeError, ValueError):
        hours = 0
    watch_until = (created + timedelta(hours=hours)).isoformat() if created and action == "apply" and hours else None
    cameras: list[dict[str, str]] = []
    for camera_id, payload in (evaluation.get("camera_comparisons") or {}).items():
        outcome = ""
        if isinstance(payload, dict):
            outcome = str((payload.get("comparison") or {}).get("outcome") or "")
        cameras.append({
            "camera_id": str(camera_id),
            "camera_name": _camera_name(config, str(camera_id)),
            "outcome": outcome,
        })
    changes: list[dict[str, Any]] = []
    for change in change_set.get("changes") or []:
        if not isinstance(change, dict):
            continue
        setting = str(change.get("setting") or "")
        guide = _public_guide(setting)
        camera_id = str(change.get("camera_id") or "")
        changes.append({
            "id": change.get("id"),
            "setting": setting,
            "label": guide["label"],
            "unit": guide["unit"],
            "scope": str(change.get("scope") or ("camera" if camera_id else "global")),
            "camera_id": camera_id,
            "camera_name": _camera_name(config, camera_id) if camera_id else "Every camera",
            "before": change.get("before"),
            "after": change.get("after"),
        })
    return {
        "id": change_set.get("id"),
        "run_id": change_set.get("run_id"),
        "action": action,
        "origin": origin,
        "phase": phase,
        "status": status,
        "created_at": change_set.get("created_at"),
        "watch_until": watch_until,
        "outcome": evaluation.get("outcome"),
        "summary": str(evaluation.get("summary") or ""),
        "cameras": cameras,
        "changes": changes,
        "rollback_error": str(evaluation.get("rollback_error") or ""),
        "undoes_id": change_set.get("parent_change_set_id") if action == "rollback" else None,
    }


def link_change_feed(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Point each applied change at the rollback that undid it, and the reverse."""
    undone_by: dict[Any, list[Any]] = {}
    for item in items:
        parent = item.get("undoes_id")
        if parent:
            undone_by.setdefault(parent, []).append(item.get("id"))
    return [{**item, "undone_by": undone_by.get(item.get("id"), [])} for item in items]


def present_review_result(
    result: dict[str, Any] | None,
    config: Any,
    automatic_classes: list[str] | None,
    apply_enabled: bool,
) -> dict[str, Any]:
    """Attach plain descriptions and keep only authenticated sample pictures."""
    payload = dict(result or {})
    notes = [
        note for note in payload.get("site_notes") or []
        if not (
            isinstance(note, dict)
            and (note.get("kind") == "grace" or "45-second" in str(note.get("text") or ""))
        )
    ]
    payload["site_notes"] = notes
    recommendations: list[dict[str, Any]] = []
    for item in payload.get("recommendations") or []:
        if not isinstance(item, dict):
            continue
        class_id = suggestion_class(item)
        evidence = [
            {**entry, "image_url": review_image_url(entry.get("image_url"))}
            for entry in item.get("evidence") or []
            if isinstance(entry, dict)
        ]
        described = describe_recommendation(
            {**item, "evidence": evidence},
            config,
            automatic_classes,
            apply_enabled,
        )
        recommendations.append({
            **item,
            "automatic_class": class_id if class_id in AUTOMATIC_CLASSES else None,
            "evidence": evidence,
            "plain": described,
        })
    payload["recommendations"] = recommendations
    reports = payload.get("camera_reports")
    if isinstance(reports, dict):
        presented: dict[str, Any] = {}
        for camera_id, report in reports.items():
            if not isinstance(report, dict):
                presented[camera_id] = report
                continue
            samples = []
            for sample in report.get("samples") or []:
                if isinstance(sample, dict):
                    samples.append({**sample, "image_url": review_image_url(sample.get("image_url"))})
                else:
                    samples.append(sample)
            presented[camera_id] = {**report, "samples": samples}
        payload["camera_reports"] = presented
    payload["at_a_glance"] = at_a_glance(payload)
    return payload

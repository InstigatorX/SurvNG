from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from .audit_ai import validate_tuning_value
from .config import AppConfig, CameraConfig, camera_by_id
from .manager import validate_motion_pipeline_configuration


MOTION_SETTINGS = {
    "sensitivity",
    "stationary_object_tolerance",
    "frame_width",
    "sample_fps",
    "window_seconds",
    "post_trigger_seconds",
    "burst_quiet_seconds",
    "borderline_rescue_enabled",
    "borderline_margin",
    "visual_backup_warmup_seconds",
    "visual_backup_grace_seconds",
    "visual_backup_min_score",
    "visual_backup_score_margin",
    "visual_backup_min_consecutive",
    "visual_backup_cooldown_seconds",
    "visual_backup_max_triggers_5m",
}
CAMERA_MOTION_SETTINGS = {
    "sensitivity",
    "stationary_object_tolerance",
    "frame_width",
    "visual_backup_grace_seconds",
    "visual_backup_min_score",
    "visual_backup_min_consecutive",
    "visual_backup_cooldown_seconds",
    "visual_backup_max_triggers_5m",
    "borderline_rescue_enabled",
    "borderline_margin",
}
DETECTOR_SETTINGS = {
    "confidence_threshold": (0.01, 0.99),
    "event_confirmation_frames": (1, 5),
    "require_incident_zone": None,
}
TRACKING_SETTINGS = {
    "sample_fps": (0.5, 5.0),
    "lost_timeout_seconds": (0.5, 15.0),
    "capacity_wait_seconds": (0.0, 30.0),
    "reid_match_threshold": (0.0, 1.0),
    "vehicle_reid_match_threshold": (0.0, 1.0),
    "max_active_cameras": (1, 16),
}
_CLASS_LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")


def calibration_configuration_payload(config: AppConfig) -> dict[str, Any]:
    """Return only tunable, non-secret configuration for conflict detection."""
    detector = config.detector
    tracking = detector.tracking
    return {
        "motion": config.motion_qualification.model_dump(mode="json"),
        "detector": {
            "confidence_threshold": detector.confidence_threshold,
            "event_confirmation_frames": detector.event_confirmation_frames,
            "event_class_confirmation_frames": detector.event_class_confirmation_frames,
            "event_class_confidence_thresholds": detector.event_class_confidence_thresholds,
            "require_incident_zone": detector.require_incident_zone,
        },
        "tracking": {
            name: getattr(tracking, name) for name in sorted(TRACKING_SETTINGS)
        },
        "cameras": {
            camera.id: {
                "motion": camera.motion_qualification.model_dump(mode="json"),
                "require_incident_zone": camera.require_incident_zone,
            }
            for camera in config.cameras
        },
    }


def calibration_configuration_fingerprint(config: AppConfig) -> str:
    encoded = json.dumps(
        calibration_configuration_payload(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _recommendation_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _motion_effective_value(
    config: AppConfig,
    camera: CameraConfig | None,
    setting: str,
) -> Any:
    global_value = getattr(config.motion_qualification, setting)
    if camera is None:
        return global_value
    override = getattr(camera.motion_qualification, setting)
    return global_value if override is None or override == "inherit" else override


def calibration_setting_value(
    config: AppConfig,
    *,
    scope: str,
    setting: str,
    camera_id: str = "",
) -> Any:
    camera = camera_by_id(config, camera_id) if camera_id else None
    if scope == "camera":
        if camera is None:
            raise ValueError(f"camera not found: {camera_id}")
        if setting.startswith("motion."):
            return getattr(camera.motion_qualification, setting.removeprefix("motion."))
        if setting == "detector.require_incident_zone":
            return camera.require_incident_zone
        raise ValueError(f"unsupported camera calibration setting: {setting}")
    if scope != "global":
        raise ValueError("calibration scope must be global or camera")
    if setting.startswith("motion."):
        return getattr(config.motion_qualification, setting.removeprefix("motion."))
    if setting.startswith("detector.tracking."):
        return getattr(config.detector.tracking, setting.removeprefix("detector.tracking."))
    if setting.startswith("detector.class_confidence."):
        label = setting.removeprefix("detector.class_confidence.")
        return config.detector.event_class_confidence_thresholds.get(label)
    if setting.startswith("detector.class_confirmations."):
        label = setting.removeprefix("detector.class_confirmations.")
        return config.detector.event_class_confirmation_frames.get(label)
    if setting.startswith("detector."):
        return getattr(config.detector, setting.removeprefix("detector."))
    raise ValueError(f"unsupported global calibration setting: {setting}")


def _normalized_calibration_value(
    *,
    scope: str,
    setting: str,
    value: Any,
) -> Any:
    if setting.startswith("motion."):
        name = setting.removeprefix("motion.")
        if name not in MOTION_SETTINGS:
            raise ValueError(f"unsupported motion calibration setting: {name}")
        if scope == "camera" and value in (None, "inherit"):
            return "inherit" if name in {"sensitivity", "stationary_object_tolerance"} else None
        if scope == "camera" and name not in CAMERA_MOTION_SETTINGS:
            raise ValueError(f"{name} cannot be overridden per camera")
        return validate_tuning_value(name, value)
    if setting == "detector.require_incident_zone":
        if scope == "camera" and value is None:
            return None
        if not isinstance(value, bool):
            raise ValueError("zone eligibility must be true or false")
        return value
    if setting.startswith("detector.class_confidence."):
        label = setting.removeprefix("detector.class_confidence.")
        if not _CLASS_LABEL_PATTERN.fullmatch(label):
            raise ValueError("per-class confidence requires a valid object class")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("per-class confidence must be a number") from exc
        if not math.isfinite(number):
            raise ValueError("per-class confidence must be finite")
        if not 0.01 <= number <= 0.99:
            raise ValueError("per-class confidence must be between 0.01 and 0.99")
        return number
    if setting.startswith("detector.class_confirmations."):
        label = setting.removeprefix("detector.class_confirmations.")
        if not _CLASS_LABEL_PATTERN.fullmatch(label):
            raise ValueError("per-class confirmations requires a valid object class")
        try:
            numeric_value = float(value)
            number = int(numeric_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("per-class confirmations must be an integer") from exc
        if (
            isinstance(value, bool)
            or not math.isfinite(numeric_value)
            or number != numeric_value
            or not 1 <= number <= 5
        ):
            raise ValueError("per-class confirmations must be between 1 and 5")
        return number
    if setting.startswith("detector.tracking."):
        name = setting.removeprefix("detector.tracking.")
        bounds = TRACKING_SETTINGS.get(name)
        if bounds is None:
            raise ValueError(f"unsupported tracking calibration setting: {name}")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        if not bounds[0] <= number <= bounds[1]:
            raise ValueError(f"{name} must be between {bounds[0]:g} and {bounds[1]:g}")
        return int(number) if name == "max_active_cameras" else number
    if setting.startswith("detector."):
        name = setting.removeprefix("detector.")
        bounds = DETECTOR_SETTINGS.get(name)
        if bounds is None:
            raise ValueError(f"unsupported detector calibration setting: {name}")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        if not bounds[0] <= number <= bounds[1]:
            raise ValueError(f"{name} must be between {bounds[0]:g} and {bounds[1]:g}")
        return int(number) if name == "event_confirmation_frames" else number
    raise ValueError(f"unsupported calibration setting: {setting}")


def _validate_bounded_step(setting: str, before: Any, after: Any) -> None:
    ordered = {
        "motion.sensitivity": ["low", "balanced", "high"],
        "motion.stationary_object_tolerance": ["low", "balanced", "high"],
    }
    if setting in ordered:
        values = ordered[setting]
        if before in values and after in values and abs(values.index(after) - values.index(before)) > 1:
            raise ValueError(f"{setting} may move only one level per calibration")
        return
    maximum_delta = {
        "motion.frame_width": 160.0,
        "motion.visual_backup_grace_seconds": 1.5,
        "motion.visual_backup_min_score": 0.10,
        "motion.visual_backup_min_consecutive": 2.0,
        "motion.visual_backup_cooldown_seconds": 60.0,
        "motion.visual_backup_max_triggers_5m": 5.0,
        "motion.borderline_margin": 0.04,
        "detector.confidence_threshold": 0.05,
        "detector.event_confirmation_frames": 1.0,
        "detector.tracking.sample_fps": 1.0,
        "detector.tracking.lost_timeout_seconds": 2.0,
        "detector.tracking.capacity_wait_seconds": 5.0,
        "detector.tracking.reid_match_threshold": 0.10,
        "detector.tracking.vehicle_reid_match_threshold": 0.10,
        "detector.tracking.max_active_cameras": 2.0,
    }.get(setting)
    if (
        maximum_delta is None
        or before is None
        or after in (None, "inherit")
        or isinstance(before, bool)
        or isinstance(after, bool)
    ):
        return
    if abs(float(after) - float(before)) > maximum_delta + 1e-9:
        raise ValueError(
            f"{setting} may change by at most {maximum_delta:g} per calibration"
        )


def _validate_effective_steps(
    original: AppConfig,
    candidate: AppConfig,
    applied: list[dict[str, Any]],
) -> None:
    """Bound the effective change on the finished configuration."""
    for item in applied:
        setting = str(item.get("setting") or "")
        scope = str(item.get("scope") or "")
        if setting.startswith("motion."):
            name = setting.removeprefix("motion.")
            if scope == "camera":
                before_camera = camera_by_id(original, str(item.get("camera_id") or ""))
                after_camera = camera_by_id(candidate, str(item.get("camera_id") or ""))
                before = _motion_effective_value(original, before_camera, name)
                after = _motion_effective_value(candidate, after_camera, name)
            else:
                before = _motion_effective_value(original, None, name)
                after = _motion_effective_value(candidate, None, name)
            _validate_bounded_step(setting, before, after)
            continue
        _validate_bounded_step(setting, item.get("before"), item.get("after"))


def apply_calibration_changes(
    config: AppConfig,
    changes: Iterable[dict[str, Any]],
    *,
    restore: bool = False,
) -> tuple[AppConfig, list[dict[str, Any]]]:
    """Apply calibration changes.

    ``restore`` is only for ledger rollback. It skips the forward step cap
    and is not exposed to clients. Forward applies compare effective values
    on the finished candidate, so site and camera changes are order-independent.
    """
    candidate = config.model_copy(deep=True)
    applied: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in changes:
        scope = str(raw.get("scope") or "")
        camera_id = str(raw.get("camera_id") or "")
        setting = str(raw.get("setting") or "")
        key = (scope, camera_id, setting)
        if key in seen:
            raise ValueError(f"duplicate calibration setting: {setting}")
        seen.add(key)
        proposed = _normalized_calibration_value(
            scope=scope,
            setting=setting,
            value=raw.get("proposed", raw.get("value")),
        )
        before = calibration_setting_value(
            candidate, scope=scope, camera_id=camera_id, setting=setting
        )
        if before == proposed:
            continue
        if scope == "camera":
            camera = camera_by_id(candidate, camera_id)
            if camera is None:
                raise ValueError(f"camera not found: {camera_id}")
            if setting.startswith("motion."):
                setattr(camera.motion_qualification, setting.removeprefix("motion."), proposed)
            elif setting == "detector.require_incident_zone":
                camera.require_incident_zone = proposed
        elif setting.startswith("motion."):
            setattr(candidate.motion_qualification, setting.removeprefix("motion."), proposed)
        elif setting.startswith("detector.tracking."):
            setattr(candidate.detector.tracking, setting.removeprefix("detector.tracking."), proposed)
        elif setting.startswith("detector.class_confidence."):
            label = setting.removeprefix("detector.class_confidence.")
            candidate.detector.event_class_confidence_thresholds[label] = proposed
        elif setting.startswith("detector.class_confirmations."):
            label = setting.removeprefix("detector.class_confirmations.")
            candidate.detector.event_class_confirmation_frames[label] = proposed
        elif setting.startswith("detector."):
            setattr(candidate.detector, setting.removeprefix("detector."), proposed)
        else:
            raise ValueError(f"unsupported calibration change: {setting}")
        applied.append({
            **raw,
            "scope": scope,
            "camera_id": camera_id,
            "setting": setting,
            "before": before,
            "after": proposed,
            "before_inherited": scope == "camera" and before in (None, "inherit"),
            "after_inherited": scope == "camera" and proposed in (None, "inherit"),
        })
    if not restore:
        _validate_effective_steps(config, candidate, applied)
    validate_motion_pipeline_configuration(candidate)
    candidate = AppConfig.model_validate(candidate.model_dump(mode="json"))
    return candidate, applied


def build_calibration_report(
    config: AppConfig,
    camera_reports: dict[str, dict[str, Any]],
    *,
    mode: str,
    stream_health: dict[str, dict[str, Any]] | None = None,
    eligible_camera_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Consolidate reviewed cameras without averaging away scene differences.

    eligible_camera_ids limits site-wide agreement to that population. Cameras
    outside it are reported as excluded and do not block a global change.
    """
    recommendations: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    camera_summaries: list[dict[str, Any]] = []
    for camera_id, report in camera_reports.items():
        camera = camera_by_id(config, camera_id)
        if camera is None:
            continue
        camera_summaries.append({
            "camera_id": camera_id,
            "camera_name": camera.name,
            "summary": report.get("summary") or "",
            "analyzed": int(report.get("analyzed") or 0),
            "failed": int(report.get("failed") or 0),
            "verdict_counts": report.get("verdict_counts") or {},
            "detector_assessments": report.get("detector_assessments") or {},
            "tracking_assessments": report.get("tracking_assessments") or {},
            "stream_health": (stream_health or {}).get(camera_id, {}),
        })
        for item in report.get("recommendations") or []:
            setting = str(item.get("setting") or "")
            proposed = item.get("proposed", item.get("value"))
            grouped[(setting, json.dumps(proposed, sort_keys=True))].append({
                **item,
                "camera_id": camera_id,
            })

    reviewed_camera_ids = set(camera_reports)
    configured_ids = {camera.id for camera in config.cameras}
    if eligible_camera_ids is None:
        population = set(configured_ids)
    else:
        population = {camera_id for camera_id in eligible_camera_ids if camera_id in configured_ids}
    excluded_cameras = [
        {"camera_id": camera.id, "camera_name": camera.name}
        for camera in config.cameras
        if camera.id not in population
    ]
    population_reviewed = bool(population) and population <= reviewed_camera_ids
    globally_covered: set[tuple[str, str]] = set()
    for (setting, value_key), rows in grouped.items():
        proposed = json.loads(value_key)
        supporting_camera_ids = {str(row["camera_id"]) for row in rows}
        inheriting_camera_ids = {
            camera.id
            for camera in config.cameras
            if camera.id in population
            and getattr(camera.motion_qualification, setting, "__unsupported__")
            in (None, "inherit")
        }
        # A global change is safe only when every camera in this review population
        # was reviewed and every inheriting camera in that set supported it.
        global_consensus = (
            population_reviewed
            and bool(inheriting_camera_ids)
            and inheriting_camera_ids <= supporting_camera_ids
        )
        if setting in MOTION_SETTINGS and global_consensus:
            current = _motion_effective_value(config, None, setting)
            if current != proposed:
                payload = {
                    "scope": "global",
                    "camera_id": "",
                    "setting": f"motion.{setting}",
                    "current": current,
                    "proposed": proposed,
                }
                recommendations.append({
                    **payload,
                    "id": _recommendation_id(payload),
                    "subsystem": "motion",
                    "expected_benefit": rows[0].get("reasons", ["Repeated camera evidence supports this change."])[0],
                    "downside": "A global adjustment affects cameras that inherit this value; review listed exceptions.",
                    "evidence_strength": "strong" if len(rows) >= 3 else "moderate",
                    "support_count": sum(int(row.get("support_count") or 1) for row in rows),
                    "affected_cameras": [
                        camera.id for camera in config.cameras
                        if getattr(camera.motion_qualification, setting) in (None, "inherit")
                    ],
                    "effective_preview": [
                        {
                            "camera_id": camera.id,
                            "current": _motion_effective_value(config, camera, setting),
                            "proposed": (
                                proposed
                                if getattr(camera.motion_qualification, setting) in (None, "inherit")
                                else getattr(camera.motion_qualification, setting)
                            ),
                            "inherits": getattr(camera.motion_qualification, setting) in (None, "inherit"),
                        }
                        for camera in config.cameras
                    ],
                    "evidence": [e for row in rows for e in (row.get("evidence") or [])][:12],
                    "compute_impact": "May change visual-analysis demand." if setting in {"frame_width", "sample_fps"} else "Minimal.",
                })
                globally_covered.add((setting, value_key))

    for (setting, value_key), rows in grouped.items():
        proposed = json.loads(value_key)
        covered_globally = (setting, value_key) in globally_covered
        for row in rows:
            camera = camera_by_id(config, row["camera_id"])
            if camera is None or setting not in CAMERA_MOTION_SETTINGS:
                continue
            override = getattr(camera.motion_qualification, setting)
            if covered_globally and override in (None, "inherit"):
                continue
            current = _motion_effective_value(config, camera, setting)
            if current == proposed:
                site_after = _motion_effective_value(config, None, setting)
                for existing in recommendations:
                    if (
                        existing.get("scope") == "global"
                        and existing.get("setting") == f"motion.{setting}"
                    ):
                        site_after = existing.get("proposed")
                # Automatic restore_inherit only when inheritance keeps today's
                # effective value. A manual inherit that would change behavior
                # is still accepted by apply, with the ordinary step bound.
                if override not in (None, "inherit") and current == site_after:
                    inherited_value: Any = (
                        "inherit"
                        if setting in {"sensitivity", "stationary_object_tolerance"}
                        else None
                    )
                    payload = {
                        "scope": "camera",
                        "camera_id": camera.id,
                        "setting": f"motion.{setting}",
                        "current": override,
                        "current_effective": current,
                        "proposed": inherited_value,
                    }
                    recommendations.append({
                        **payload,
                        "id": _recommendation_id(payload),
                        "subsystem": "motion",
                        "expected_benefit": "Restore inheritance without changing the camera's effective value.",
                        "downside": "Future global adjustments will also affect this camera.",
                        "evidence_strength": "strong",
                        "support_count": int(row.get("support_count") or 0),
                        "affected_cameras": [camera.id],
                        "evidence": row.get("evidence") or [],
                        "compute_impact": "None.",
                    })
                continue
            payload = {
                "scope": "camera",
                "camera_id": camera.id,
                "setting": f"motion.{setting}",
                "current": getattr(camera.motion_qualification, setting),
                "current_effective": current,
                "proposed": proposed,
            }
            recommendations.append({
                **payload,
                "id": _recommendation_id(payload),
                "subsystem": "motion",
                "expected_benefit": (row.get("reasons") or ["Repeated evidence supports this camera-specific change."])[0],
                "downside": "This camera may trade sensitivity for fewer nuisance triggers, or the reverse.",
                "evidence_strength": "strong" if int(row.get("support_count") or 0) >= 3 else "moderate",
                "support_count": int(row.get("support_count") or 0),
                "affected_cameras": [camera.id],
                "effective_preview": [{
                    "camera_id": camera.id,
                    "current": current,
                    "proposed": proposed,
                    "inherits": False,
                }],
                "evidence": row.get("evidence") or [],
                "compute_impact": "May change visual-analysis demand." if setting == "frame_width" else "Minimal.",
            })

    total_analyzed = sum(int(report.get("analyzed") or 0) for report in camera_reports.values())
    verdicts: Counter[str] = Counter()
    tracking: Counter[str] = Counter()
    for report in camera_reports.values():
        verdicts.update(report.get("verdict_counts") or {})
        tracking.update(report.get("tracking_assessments") or {})

    # Object/tracking changes require materially more evidence than motion.
    object_advisory = "No broad detector adjustment was justified."
    if total_analyzed >= 20 and verdicts["likely_misclassification"] >= 4:
        object_advisory = (
            "Repeated likely misclassifications were found, but confirmation depth is not "
            "an automatic remedy for a consistently wrong label. Review class-specific "
            "confidence evidence before changing detector policy."
        )
    tracking_issues = sum(
        count for name, count in tracking.items()
        if name in {"late", "lost", "duplicate"}
    )
    if population_reviewed and total_analyzed >= 12 and tracking_issues >= 4:
        current = config.detector.tracking.sample_fps
        proposed = min(5.0, round(current + 0.5, 1))
        if proposed != current:
            payload = {"scope": "global", "camera_id": "", "setting": "detector.tracking.sample_fps", "current": current, "proposed": proposed}
            recommendations.append({
                **payload,
                "id": _recommendation_id(payload),
                "subsystem": "tracking",
                "expected_benefit": "Reduce acquisition delay and gaps between track observations.",
                "downside": "Raises detector and tracking compute while incidents are active.",
                "evidence_strength": "strong",
                "support_count": tracking_issues,
                "affected_cameras": [camera.id for camera in config.cameras],
                "evidence": [],
                "compute_impact": "Moderate increase during active tracking sessions.",
            })

    validated_recommendations: list[dict[str, Any]] = []
    for recommendation in recommendations:
        try:
            _candidate, bounded = apply_calibration_changes(config, [recommendation])
        except ValueError:
            continue
        if bounded:
            validated_recommendations.append(recommendation)

    manual_review_settings = sorted({
        setting
        for setting, _value_key in grouped
        if setting not in CAMERA_MOTION_SETTINGS
    })

    return {
        "review_type": "calibration_lab",
        "mode": mode,
        "summary": f"Analyzed {len(camera_reports)} cameras using {total_analyzed} balanced visual samples.",
        "configuration_fingerprint": calibration_configuration_fingerprint(config),
        "camera_summaries": camera_summaries,
        "camera_reports": camera_reports,
        "recommendations": validated_recommendations,
        "excluded_cameras": excluded_cameras,
        "advisories": {
            "stream_health": "Stream recommendations are advisory and never modify camera URLs or firmware settings.",
            "object_safety": object_advisory,
            "scope_safety": (
                "Global changes require unanimous support from every inheriting camera in this review. "
                "Cameras left out of the review are named separately and do not block agreement."
                if excluded_cameras
                else "Global changes require unanimous support from every inheriting camera "
                "and are not proposed for partial-camera analyses."
            ),
            "manual_review": (
                "The following pipeline-level suggestions require manual review because "
                f"they can replace custom stage graphs: {', '.join(manual_review_settings)}."
                if manual_review_settings
                else "No pipeline-level suggestion required separate manual review."
            ),
        },
    }

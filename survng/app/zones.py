from __future__ import annotations

import math
from typing import Any

from .config import CameraConfig, DetectionZone


ZONE_BOUNDARY_EPSILON = 1e-6


def _class_applies(zone: DetectionZone, label: str) -> bool:
    classes = {item.strip().lower() for item in zone.object_classes if item.strip()}
    return not classes or label.lower() in classes


def _point_on_segment(
    x: float,
    y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> bool:
    segment_x = end_x - start_x
    segment_y = end_y - start_y
    length_squared = segment_x * segment_x + segment_y * segment_y
    if length_squared <= ZONE_BOUNDARY_EPSILON * ZONE_BOUNDARY_EPSILON:
        return (x - start_x) ** 2 + (y - start_y) ** 2 <= ZONE_BOUNDARY_EPSILON ** 2

    projection = ((x - start_x) * segment_x + (y - start_y) * segment_y) / length_squared
    if projection < -ZONE_BOUNDARY_EPSILON or projection > 1.0 + ZONE_BOUNDARY_EPSILON:
        return False
    closest_x = start_x + min(1.0, max(0.0, projection)) * segment_x
    closest_y = start_y + min(1.0, max(0.0, projection)) * segment_y
    return (x - closest_x) ** 2 + (y - closest_y) ** 2 <= ZONE_BOUNDARY_EPSILON ** 2


def _point_in_polygon(x: float, y: float, zone: DetectionZone) -> bool:
    points = zone.points
    if len(points) < 3:
        return False
    inside = False
    previous = points[-1]
    for current in points:
        if _point_on_segment(x, y, previous.x, previous.y, current.x, current.y):
            return True
        if (current.y > y) != (previous.y > y):
            crossing_x = (previous.x - current.x) * (y - current.y) / (previous.y - current.y) + current.x
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def class_confidence_threshold(
    label: str,
    default: float,
    class_thresholds: dict[str, float] | None = None,
) -> float:
    normalized_label = str(label or "").strip().lower()
    configured = (class_thresholds or {}).get(normalized_label, default)
    return max(0.01, min(0.99, float(configured)))


def detection_threshold(
    camera: CameraConfig,
    default: float,
    class_thresholds: dict[str, float] | None = None,
) -> float:
    thresholds = [default]
    thresholds.extend((class_thresholds or {}).values())
    thresholds.extend(
        zone.confidence_threshold
        for zone in camera.zones
        if (
            zone.enabled
            and zone.behavior == "incident"
            and len(zone.points) >= 3
            and zone.confidence_threshold is not None
        )
    )
    return max(0.01, min(float(value) for value in thresholds))


def apply_detection_zones(
    camera: CameraConfig,
    objects: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    default_confidence: float,
    require_incident_zone: bool = True,
    class_confidence_thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    zones = [zone for zone in camera.zones if zone.enabled and len(zone.points) >= 3]
    if not zones:
        for detected in objects:
            if not isinstance(detected, dict):
                continue
            detected["zones"] = []
            detected["zone_matches"] = []
            detected.pop("zone_point", None)
            detected["incident_eligible"] = False
            label = str(detected.get("label") or "")
            if class_confidence_thresholds is None:
                detected["incident_eligible"] = bool(label)
                continue
            threshold = class_confidence_threshold(
                label,
                default_confidence,
                class_confidence_thresholds,
            )
            try:
                confidence = float(detected.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            confidence_eligible = bool(
                label and math.isfinite(confidence) and confidence >= threshold
            )
            detected["confidence_threshold"] = threshold
            detected["confidence_eligible"] = confidence_eligible
            detected["incident_eligible"] = confidence_eligible
        return objects

    has_incident_zones = any(zone.behavior == "incident" for zone in zones)
    width = max(1, frame_width)
    height = max(1, frame_height)
    for detected in objects:
        if not isinstance(detected, dict):
            continue
        # Always replace prior annotations so re-evaluating an object for a
        # different camera or zone configuration cannot retain stale access.
        detected["zones"] = []
        detected["zone_matches"] = []
        detected.pop("zone_point", None)
        detected["incident_eligible"] = False
        label = str(detected.get("label") or "")
        label_threshold = class_confidence_threshold(
            label,
            default_confidence,
            class_confidence_thresholds,
        )
        detected["confidence_threshold"] = label_threshold
        detected["confidence_eligible"] = False
        box = detected.get("box") or {}
        if not label or not isinstance(box, dict) or not all(
            key in box for key in ("x1", "y1", "x2", "y2")
        ):
            continue
        try:
            confidence = float(detected.get("confidence") or 0.0)
            coordinates = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(confidence) or not all(math.isfinite(value) for value in coordinates):
            continue

        x1, _y1, x2, y2 = coordinates
        x = ((x1 + x2) / 2.0) / width
        y = y2 / height
        relevant = [
            zone
            for zone in zones
            if zone.behavior != "none" and _class_applies(zone, label)
        ]
        matches = []
        for zone in relevant:
            threshold = zone.confidence_threshold if zone.confidence_threshold is not None else label_threshold
            if confidence >= threshold and _point_in_polygon(x, y, zone):
                matches.append(zone)

        ignored = any(zone.behavior == "ignore" for zone in matches)
        admitted = any(zone.behavior == "incident" for zone in matches)
        meets_default = confidence >= label_threshold
        detected["confidence_eligible"] = bool(meets_default or matches)
        detected["zones"] = [zone.name for zone in matches]
        detected["zone_matches"] = [
            {"name": zone.name, "behavior": zone.behavior, "color": zone.color}
            for zone in matches
        ]
        detected["zone_point"] = {"x": round(x, 5), "y": round(y, 5)}
        zone_required = (
            require_incident_zone
            if camera.require_incident_zone is None
            else camera.require_incident_zone
        )
        if zone_required and has_incident_zones:
            eligible = admitted
        else:
            # In full-frame mode, a matching incident zone can admit an
            # object at its zone-specific threshold, while objects elsewhere
            # must still satisfy the detector's normal confidence threshold.
            eligible = admitted or meets_default
        detected["incident_eligible"] = bool(not ignored and eligible)
    return objects

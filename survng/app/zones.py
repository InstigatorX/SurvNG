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


def detection_threshold(camera: CameraConfig, default: float) -> float:
    thresholds = [default]
    thresholds.extend(
        zone.confidence_threshold
        for zone in camera.zones
        if zone.enabled and len(zone.points) >= 3 and zone.confidence_threshold is not None
    )
    return max(0.01, min(float(value) for value in thresholds))


def apply_detection_zones(
    camera: CameraConfig,
    objects: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    default_confidence: float,
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
            if detected.get("label"):
                detected["incident_eligible"] = True
        return objects

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
        relevant = [zone for zone in zones if _class_applies(zone, label)]
        incident_zones = [zone for zone in relevant if zone.behavior == "incident"]
        matches = []
        for zone in relevant:
            threshold = zone.confidence_threshold if zone.confidence_threshold is not None else default_confidence
            if confidence >= threshold and _point_in_polygon(x, y, zone):
                matches.append(zone)

        ignored = any(zone.behavior == "ignore" for zone in matches)
        admitted = any(zone.behavior == "incident" for zone in matches)
        meets_default = confidence >= default_confidence
        detected["zones"] = [zone.name for zone in matches]
        detected["zone_matches"] = [
            {"name": zone.name, "behavior": zone.behavior, "color": zone.color}
            for zone in matches
        ]
        detected["zone_point"] = {"x": round(x, 5), "y": round(y, 5)}
        detected["incident_eligible"] = bool(not ignored and (admitted if incident_zones else meets_default))
    return objects

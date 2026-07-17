from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


MOTION_SCORE_THRESHOLDS = {
    "high": 0.36,
    "balanced": 0.48,
    "low": 0.60,
}


@dataclass(frozen=True)
class MotionQualificationResult:
    accepted: bool
    score: float
    threshold: float
    reason: str
    frame_count: int
    features: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gray_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(frame, (5, 5), 0)


def qualify_motion(frames: list[np.ndarray], sensitivity: str = "balanced") -> MotionQualificationResult:
    threshold = MOTION_SCORE_THRESHOLDS.get(sensitivity, MOTION_SCORE_THRESHOLDS["balanced"])
    if len(frames) < 4:
        return MotionQualificationResult(
            accepted=True,
            score=1.0,
            threshold=threshold,
            reason="insufficient_frames",
            frame_count=len(frames),
            features={},
        )

    prepared = [_gray_frame(frame) for frame in frames]
    frame_height, frame_width = prepared[0].shape[:2]
    frame_area = max(1, frame_width * frame_height)
    kernel = np.ones((3, 3), np.uint8)
    edge_margin_x = max(1, round(frame_width * 0.06))
    edge_margin_y = max(1, round(frame_height * 0.06))

    active: list[bool] = []
    areas: list[float] = []
    centroids: list[tuple[float, float]] = []
    edge_distances: list[float] = []
    interior: list[float] = []
    fragmentation: list[float] = []
    global_changes = 0

    for previous, current in zip(prepared, prepared[1:]):
        difference = cv2.absdiff(previous, current)
        _, mask = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        changed_pixels = int(cv2.countNonZero(mask))
        changed_ratio = changed_pixels / frame_area
        if changed_ratio >= 0.45:
            global_changes += 1

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        components = [contour for contour in contours if cv2.contourArea(contour) >= frame_area * 0.0003]
        if not components:
            active.append(False)
            continue

        largest = max(components, key=cv2.contourArea)
        largest_area = float(cv2.contourArea(largest))
        area_ratio = largest_area / frame_area
        is_active = area_ratio >= 0.0008 or changed_ratio >= 0.003
        active.append(is_active)
        if not is_active:
            continue

        x, y, width, height = cv2.boundingRect(largest)
        moments = cv2.moments(largest)
        if moments["m00"]:
            center_x = float(moments["m10"] / moments["m00"]) / frame_width
            center_y = float(moments["m01"] / moments["m00"]) / frame_height
        else:
            center_x = (x + width / 2) / frame_width
            center_y = (y + height / 2) / frame_height
        touches_edge = (
            x <= edge_margin_x
            or y <= edge_margin_y
            or x + width >= frame_width - edge_margin_x
            or y + height >= frame_height - edge_margin_y
        )
        areas.append(area_ratio)
        centroids.append((center_x, center_y))
        edge_distances.append(min(center_x, center_y, 1.0 - center_x, 1.0 - center_y))
        interior.append(0.0 if touches_edge else 1.0)
        fragmentation.append(min(1.0, largest_area / max(1.0, changed_pixels)))

    transition_count = max(1, len(active))
    persistence = sum(active) / transition_count
    if len(areas) >= 2:
        median_area = float(np.median(areas))
        area_deviation = float(np.median(np.abs(np.asarray(areas) - median_area))) / max(median_area, 1e-6)
        area_stability = max(0.0, min(1.0, 1.0 - area_deviation * 2.5))
    else:
        area_stability = 0.0

    continuity = 0.0
    if len(centroids) >= 2:
        vectors = np.diff(np.asarray(centroids), axis=0)
        speeds = np.linalg.norm(vectors, axis=1)
        moving = speeds > 0.002
        if np.any(moving):
            directions = vectors[moving] / speeds[moving, None]
            direction_coherence = float(np.linalg.norm(np.sum(directions, axis=0)) / len(directions))
            median_speed = float(np.median(speeds[moving]))
            speed_deviation = float(np.median(np.abs(speeds[moving] - median_speed))) / max(median_speed, 1e-6)
            speed_stability = max(0.0, min(1.0, 1.0 - speed_deviation * 2.0))
            continuity = direction_coherence * 0.65 + speed_stability * 0.35
        else:
            continuity = 0.35

    interior_score = float(np.mean(interior)) if interior else 0.0
    fragmentation_score = float(np.mean(fragmentation)) if fragmentation else 0.0
    inward_progress = 0.0
    if len(edge_distances) >= 2:
        inward_progress = max(0.0, min(1.0, (edge_distances[-1] - edge_distances[0]) / 0.08))
    coherent_entry = inward_progress * continuity * fragmentation_score
    coherent_edge_track = continuity * area_stability * fragmentation_score
    global_change_ratio = global_changes / transition_count
    edge_relief = max(coherent_entry, coherent_edge_track)
    edge_penalty = (1.0 - interior_score) * 0.18 * (1.0 - edge_relief)
    score = (
        persistence * 0.35
        + continuity * 0.20
        + area_stability * 0.15
        + interior_score * 0.12
        + fragmentation_score * 0.18
        + coherent_entry * 0.30
        - global_change_ratio * 0.35
        - edge_penalty
    )
    score = max(0.0, min(1.0, score))
    accepted = score >= threshold
    if accepted and interior_score < 0.25 and max(coherent_entry, coherent_edge_track) >= 0.35:
        reason = "coherent_edge_track"
    elif accepted:
        reason = "qualified"
    elif global_change_ratio >= 0.5:
        reason = "global_change"
    elif persistence < 0.35:
        reason = "low_persistence"
    elif continuity < 0.2:
        reason = "erratic_motion"
    elif interior_score < 0.25:
        reason = "edge_motion"
    elif fragmentation_score < 0.25:
        reason = "fragmented_motion"
    else:
        reason = "low_score"

    return MotionQualificationResult(
        accepted=accepted,
        score=round(score, 4),
        threshold=threshold,
        reason=reason,
        frame_count=len(frames),
        features={
            "persistence": round(persistence, 4),
            "continuity": round(continuity, 4),
            "area_stability": round(area_stability, 4),
            "interior": round(interior_score, 4),
            "fragmentation": round(fragmentation_score, 4),
            "inward_progress": round(inward_progress, 4),
            "coherent_entry": round(coherent_entry, 4),
            "coherent_edge_track": round(coherent_edge_track, 4),
            "global_change": round(global_change_ratio, 4),
        },
    )

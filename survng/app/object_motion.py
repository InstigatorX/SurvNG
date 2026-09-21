"""Pure temporal object-motion evidence shared by incident policies."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping


MINIMUM_MOVEMENT_RATIO = 0.003
MAXIMUM_MOVEMENT_RATIO = 0.02
MOVEMENT_BOX_SCALE = 0.04
MINIMUM_PATH_RATIO = 0.01
PATH_MOVEMENT_SCALE = 2.5
MOTION_CORRELATION_REASON = "object_not_motion_correlated"
TRACKING_RESAMPLE_BUCKET = 8
MINIMUM_TRACKING_OBSERVATIONS = 2 * TRACKING_RESAMPLE_BUCKET
MINIMUM_TRACKING_MATCH_IOU = 0.5


@dataclass(frozen=True, slots=True)
class TemporalObjectMotionEvidence:
    """Resolution-independent evidence, without an admission decision."""

    normalized_box: tuple[float, float, float, float] | None
    displacement_ratio: float
    path_ratio: float
    movement_threshold: float
    track_observations: int
    pretrigger_observations: int
    posttrigger_observations: int
    newly_appeared: bool
    robust_new_appearance: bool
    zone_entry: bool

    @property
    def temporal_evidence_available(self) -> bool:
        return self.track_observations >= 2

    @property
    def path_threshold(self) -> float:
        return max(MINIMUM_PATH_RATIO, self.movement_threshold * PATH_MOVEMENT_SCALE)

    @property
    def credible_movement(self) -> bool:
        return bool(
            self.displacement_ratio >= self.movement_threshold
            or self.path_ratio >= self.path_threshold
        )

    def stable(
        self,
        *,
        maximum_displacement_ratio: float,
        maximum_path_ratio: float,
        require_trigger_span: bool = False,
    ) -> bool:
        return bool(
            self.temporal_evidence_available
            and self.displacement_ratio <= maximum_displacement_ratio
            and self.path_ratio <= maximum_path_ratio
            and not self.robust_new_appearance
            and not self.zone_entry
            and (
                not require_trigger_span
                or (
                    self.pretrigger_observations >= 1
                    and self.posttrigger_observations >= 1
                )
            )
        )


def temporal_object_motion_evidence(
    observation: Mapping[str, Any],
    *,
    frame_width: int | float | None = None,
    frame_height: int | float | None = None,
) -> TemporalObjectMotionEvidence:
    # An explicit frame shape is authoritative for the box being correlated.
    # Persisted dimensions are the fallback used by attribution and replay.
    width = _positive_float(frame_width)
    height = _positive_float(frame_height)
    if width is None:
        width = _positive_float(observation.get("detection_frame_width"))
    if height is None:
        height = _positive_float(observation.get("detection_frame_height"))
    normalized_box = _normalized_box(observation.get("box"), width, height)
    movement_threshold = _movement_threshold(normalized_box)
    pretrigger = _integer(observation.get("temporal_pretrigger_observations"))
    posttrigger = _integer(observation.get("temporal_posttrigger_observations"))
    first_offset = _finite_signed(
        observation.get("temporal_first_observation_offset_seconds")
    )
    last_offset = _finite_signed(
        observation.get("temporal_last_observation_offset_seconds")
    )
    if pretrigger == 0 and first_offset is not None and first_offset < 0.0:
        pretrigger = 1
    if posttrigger == 0 and last_offset is not None and last_offset >= 0.0:
        posttrigger = 1
    return TemporalObjectMotionEvidence(
        normalized_box=normalized_box,
        displacement_ratio=_finite(observation.get("temporal_center_displacement_ratio")),
        path_ratio=_finite(observation.get("temporal_center_path_ratio")),
        movement_threshold=movement_threshold,
        track_observations=_integer(observation.get("temporal_track_observations")),
        pretrigger_observations=pretrigger,
        posttrigger_observations=posttrigger,
        newly_appeared=bool(observation.get("temporal_newly_appeared")),
        robust_new_appearance=bool(
            observation.get("temporal_robust_new_appearance")
        ),
        zone_entry=bool(observation.get("temporal_zone_entry")),
    )


def tracking_motion_promotions(
    objects: Sequence[Any],
    tracking: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Re-check motion-demoted objects against whole-session tracking evidence.

    Refinement decides motion correlation from roughly one second of samples,
    so a subject that moves slowly can fail the path test and stay out of the
    incident even though it is real. Tracking follows the same subject for the
    whole incident, so apply the unchanged path threshold to that longer
    trajectory. Raw path accumulates detector-box noise, and a median filter
    does not remove a steady oscillation, so measure travel between resampled
    bucket means and the widest center separation instead. Either may satisfy
    the threshold: resampling catches a slow walker, while the separation
    catches a fast transit that bucket averaging blurs. Both stay below the
    threshold for jitter. Every other admission gate remains authoritative,
    so only an object that failed motion correlation alone can be restored.
    """
    width = _positive_float(tracking.get("frame_width"))
    height = _positive_float(tracking.get("frame_height"))
    if width is None or height is None:
        return {}
    candidates = [
        (index, item) for index, item in enumerate(objects) if _promotable(item)
    ]
    if not candidates:
        return {}
    tracks = [
        summary
        for summary in (tracking.get("tracks") or [])
        if isinstance(summary, Mapping)
        and str(summary.get("state") or "") == "confirmed"
        and _integer(summary.get("observations")) >= MINIMUM_TRACKING_OBSERVATIONS
    ]
    promotions: dict[int, dict[str, Any]] = {}
    claimed: set[int] = set()
    for index, item in candidates:
        box = _normalized_box(
            item.get("box"),
            _positive_float(item.get("detection_frame_width")),
            _positive_float(item.get("detection_frame_height")),
        )
        if box is None:
            continue
        threshold = max(
            MINIMUM_PATH_RATIO,
            _movement_threshold(box) * PATH_MOVEMENT_SCALE,
        )
        matched: tuple[float, Mapping[str, Any]] | None = None
        for summary in tracks:
            if _integer(summary.get("track_id")) in claimed:
                continue
            if str(summary.get("label") or "") != str(item.get("label") or ""):
                continue
            anchor = _track_anchor_box(summary, width, height)
            if anchor is None:
                continue
            overlap = _intersection_over_union(box, anchor)
            if overlap < MINIMUM_TRACKING_MATCH_IOU:
                continue
            if matched is None or overlap > matched[0]:
                matched = (overlap, summary)
        if matched is None:
            continue
        overlap, summary = matched
        path, span = _resampled_motion_ratios(
            summary.get("trajectory"), width, height
        )
        if path < threshold and span < threshold:
            continue
        track_id = _integer(summary.get("track_id"))
        claimed.add(track_id)
        observations = _integer(summary.get("observations"))
        promotions[index] = {
            "incident_eligible": True,
            "incident_ineligible_reasons": [],
            "motion_correlated": True,
            "motion_correlation": "tracking_path",
            "motion_correlation_eligible": True,
            "track_id": track_id,
            "track_state": "confirmed",
            "track_observations": observations,
            "tracking_motion_promotion": {
                "track_id": track_id,
                "observations": observations,
                "match_iou": round(overlap, 3),
                "resampled_path_ratio": round(path, 5),
                "trajectory_span_ratio": round(span, 5),
                "path_threshold": round(threshold, 5),
                "resample_bucket": TRACKING_RESAMPLE_BUCKET,
            },
        }
    return promotions


def _promotable(item: Any) -> bool:
    if not isinstance(item, Mapping) or not item.get("label"):
        return False
    if item.get("incident_eligible") is not False:
        return False
    if item.get("auxiliary_detection") is True:
        return False
    reasons = item.get("incident_ineligible_reasons")
    if not isinstance(reasons, list):
        return False
    if [str(value) for value in reasons] != [MOTION_CORRELATION_REASON]:
        return False
    return bool(
        item.get("confidence_eligible") is True
        and item.get("spatial_zone_eligible") is True
        and item.get("temporal_eligible") is True
        and item.get("temporal_consensus") is True
        and str(item.get("activity_role") or "") != "scene_context"
    )


def _track_anchor_box(
    summary: Mapping[str, Any],
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    """Anchor on the first tracked box, which shares the event's frame."""
    history = summary.get("box_history")
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        for entry in history:
            if (
                isinstance(entry, Sequence)
                and not isinstance(entry, (str, bytes))
                and len(entry) >= 5
            ):
                return _normalized_box(
                    {
                        "x1": entry[1],
                        "y1": entry[2],
                        "x2": entry[3],
                        "y2": entry[4],
                    },
                    width,
                    height,
                )
    return _normalized_box(summary.get("box"), width, height)


def _intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    overlap_width = min(first[2], second[2]) - max(first[0], second[0])
    overlap_height = min(first[3], second[3]) - max(first[1], second[1])
    if overlap_width <= 0.0 or overlap_height <= 0.0:
        return 0.0
    intersection = overlap_width * overlap_height
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return intersection / union if union > 0.0 else 0.0


def _resampled_motion_ratios(
    trajectory: object,
    width: float,
    height: float,
) -> tuple[float, float]:
    """Return travel between bucket means and the widest center separation."""
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        return 0.0, 0.0
    centers: list[tuple[float, float]] = []
    for entry in trajectory:
        if (
            not isinstance(entry, Sequence)
            or isinstance(entry, (str, bytes))
            or len(entry) < 3
        ):
            continue
        try:
            point = (float(entry[1]) / width, float(entry[2]) / height)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if all(math.isfinite(value) for value in point):
            centers.append(point)
    if len(centers) < 2:
        return 0.0, 0.0
    means = [
        (
            statistics.fmean(point[0] for point in bucket),
            statistics.fmean(point[1] for point in bucket),
        )
        for index in range(0, len(centers), TRACKING_RESAMPLE_BUCKET)
        if (bucket := centers[index:index + TRACKING_RESAMPLE_BUCKET])
    ]
    path = sum(
        math.dist(previous, current)
        for previous, current in zip(means, means[1:])
    )
    span = max(
        math.dist(first, second)
        for index, first in enumerate(centers)
        for second in centers[index + 1:]
    )
    return path, span


def _movement_threshold(
    box: tuple[float, float, float, float] | None,
) -> float:
    if box is None:
        return MAXIMUM_MOVEMENT_RATIO
    x1, y1, x2, y2 = box
    diagonal = math.hypot(x2 - x1, y2 - y1)
    return min(
        MAXIMUM_MOVEMENT_RATIO,
        max(MINIMUM_MOVEMENT_RATIO, diagonal * MOVEMENT_BOX_SCALE),
    )


def _normalized_box(
    raw_box: object,
    width: float | None,
    height: float | None,
) -> tuple[float, float, float, float] | None:
    if not isinstance(raw_box, Mapping) or width is None or height is None:
        return None
    try:
        normalized = (
            float(raw_box["x1"]) / width,
            float(raw_box["y1"]) / height,
            float(raw_box["x2"]) / width,
            float(raw_box["y2"]) / height,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not all(math.isfinite(value) for value in normalized):
        return None
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        return None
    return normalized


def _finite(value: object) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result >= 0.0 else 0.0


def _finite_signed(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _positive_float(value: object) -> float | None:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None

"""Pure temporal object-motion evidence shared by incident policies."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Mapping


MINIMUM_MOVEMENT_RATIO = 0.003
MAXIMUM_MOVEMENT_RATIO = 0.02
MOVEMENT_BOX_SCALE = 0.04
MINIMUM_PATH_RATIO = 0.01
PATH_MOVEMENT_SCALE = 2.5
MOTION_CORRELATION_REASON = "object_not_motion_correlated"
MINIMUM_TRACKING_OBSERVATIONS = 16
MINIMUM_TRACKING_MATCH_IOU = 0.5
ISOLATED_POINT_SEPARATION_FACTOR = 3.0


@dataclass(frozen=True, slots=True)
class ObjectMotionEstimate:
    """Bounded travel evidence; accumulated path is diagnostic only."""

    displacement_ratio: float = 0.0
    excursion_ratio: float = 0.0
    raw_displacement_ratio: float = 0.0
    raw_path_ratio: float = 0.0
    filtered_path_ratio: float = 0.0
    duration_seconds: float = 0.0
    samples: int = 0
    isolated_points_rejected: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "method": "supported_observation_excursion",
            "isolation_floor_ratio": MINIMUM_MOVEMENT_RATIO,
            "isolation_separation_factor": ISOLATED_POINT_SEPARATION_FACTOR,
            **asdict(self),
        }


def estimate_object_motion(
    trajectory: Sequence[Sequence[float]],
) -> ObjectMotionEstimate:
    """Measure normalized timestamped centers without accumulating box jitter.

    Retain distinct observations instead of binning away support for brief
    excursions. Reject only a point far from both temporal neighbors when those
    neighbors agree spatially and no other observation supports its location.
    Endpoint rejection also checks the timestamped
    local trend, so irregularly sampled steady travel is not trimmed away.
    Decisions use the original points in one pass: rejection cannot cascade.
    Two-point evidence remains ambiguous between real movement and box error.
    """
    by_time: dict[float, list[tuple[float, float]]] = {}
    for entry in trajectory:
        if len(entry) < 3:
            continue
        try:
            timestamp, x, y = (float(value) for value in entry[:3])
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in (timestamp, x, y)):
            by_time.setdefault(timestamp, []).append((x, y))
    if not by_time:
        return ObjectMotionEstimate()

    def center(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
        return (
            statistics.median(p[0] for p in points),
            statistics.median(p[1] for p in points),
        )

    times = sorted(by_time)
    raw = [center(by_time[timestamp]) for timestamp in times]
    filtered = []
    for index, point in enumerate(raw):
        if len(raw) < 3:
            filtered.append(point)
            continue
        if index == 0:
            first, second = 1, 2
        elif index == len(raw) - 1:
            first, second = index - 1, index - 2
        else:
            first, second = index - 1, index + 1
        neighbor_span = math.dist(raw[first], raw[second])
        tolerance = max(
            MINIMUM_MOVEMENT_RATIO,
            ISOLATED_POINT_SEPARATION_FACTOR * neighbor_span,
        )
        isolated = min(
            math.dist(point, raw[first]), math.dist(point, raw[second]),
        ) > tolerance
        if isolated and index in (0, len(raw) - 1):
            # Extrapolate only to test support, never to manufacture a center.
            # A long gap before an endpoint can explain a large real step.
            fraction = (times[index] - times[first]) / (times[second] - times[first])
            expected = tuple(
                raw[first][axis] + fraction * (raw[second][axis] - raw[first][axis])
                for axis in (0, 1)
            )
            isolated = math.dist(point, expected) > tolerance
        if isolated:
            # Repeated excursions can alternate between distant positions.
            # Support need not be adjacent, but must be a distinct timestamp.
            isolated = not any(
                other_index != index
                and math.dist(point, other) <= MINIMUM_MOVEMENT_RATIO
                for other_index, other in enumerate(raw)
            )
        if not isolated:
            filtered.append(point)
    # With three inconsistent observations there may be no supported point.
    # Retain raw diagnostics, but no affirmative movement evidence.
    return ObjectMotionEstimate(
        displacement_ratio=math.dist(filtered[0], filtered[-1]) if len(filtered) >= 2 else 0.0,
        excursion_ratio=max(
            (math.dist(first, second) for index, first in enumerate(filtered)
             for second in filtered[index + 1:]), default=0.0,
        ),
        raw_displacement_ratio=math.dist(raw[0], raw[-1]),
        raw_path_ratio=sum(math.dist(a, b) for a, b in zip(raw, raw[1:])),
        filtered_path_ratio=sum(math.dist(a, b) for a, b in zip(filtered, filtered[1:])),
        duration_seconds=times[-1] - times[0],
        samples=len(times),
        isolated_points_rejected=len(raw) - len(filtered),
    )


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
    excursion_ratio: float | None = None

    @property
    def movement_extent_ratio(self) -> float:
        # Legacy records have only aggregate path; do not reinterpret their
        # historical decisions as if the original trajectory had been retained.
        return self.path_ratio if self.excursion_ratio is None else self.excursion_ratio

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
            or self.movement_extent_ratio >= self.path_threshold
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
            and self.movement_extent_ratio <= maximum_path_ratio
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
    estimate = observation.get("temporal_motion")
    has_estimate = isinstance(estimate, Mapping) and estimate.get("version") in (1, 2)
    return TemporalObjectMotionEvidence(
        normalized_box=normalized_box,
        displacement_ratio=_finite(
            estimate.get("displacement_ratio") if has_estimate
            else observation.get("temporal_center_displacement_ratio")
        ),
        path_ratio=_finite(observation.get("temporal_center_path_ratio")),
        excursion_ratio=_finite(estimate.get("excursion_ratio")) if has_estimate else None,
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

    Use the same bounded excursion estimator as initial admission. A longer
    observation window can establish travel that sparse refinement missed, but
    accumulating stationary box noise cannot. All other eligibility gates and
    track identity matching remain authoritative.
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
        estimate = _tracking_motion_estimate(summary.get("trajectory"), width, height)
        if estimate.excursion_ratio < threshold:
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
                "resampled_path_ratio": round(estimate.filtered_path_ratio, 5),
                "trajectory_span_ratio": round(estimate.excursion_ratio, 5),
                "path_threshold": round(threshold, 5),
                "motion_estimate": estimate.as_dict(),
                "qualification_metric": "supported_excursion",
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


def _tracking_motion_estimate(
    trajectory: object, width: float, height: float,
) -> ObjectMotionEstimate:
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        return ObjectMotionEstimate()
    normalized = []
    for entry in trajectory:
        if (
            not isinstance(entry, Sequence)
            or isinstance(entry, (str, bytes))
            or len(entry) < 3
        ):
            continue
        try:
            normalized.append((
                float(entry[0]), float(entry[1]) / width, float(entry[2]) / height,
            ))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return estimate_object_motion(normalized)


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

"""Pure temporal object-motion evidence shared by incident policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


MINIMUM_MOVEMENT_RATIO = 0.003
MAXIMUM_MOVEMENT_RATIO = 0.02
MOVEMENT_BOX_SCALE = 0.04
MINIMUM_PATH_RATIO = 0.01
PATH_MOVEMENT_SCALE = 2.5


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

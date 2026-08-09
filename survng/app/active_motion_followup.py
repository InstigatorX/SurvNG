from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from .motion import MotionQualificationResult


MotionRegion = tuple[float, float, float, float]


class ActiveMotionFollowupAction(StrEnum):
    INACTIVE = "inactive"
    BASELINE = "baseline"
    NOT_CREDIBLE = "not_credible"
    DUPLICATE = "duplicate"
    RATE_LIMITED = "rate_limited"
    EPISODE_LIMIT = "episode_limit"
    CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class ActiveMotionFollowupDecision:
    action: ActiveMotionFollowupAction
    event_key: str = ""
    track_id: int | None = None
    region: MotionRegion | None = None
    anchor_index: int = 0
    maximum_overlap: float = 0.0
    nearest_center_distance: float = 0.0

    @property
    def admitted(self) -> bool:
        return self.action is ActiveMotionFollowupAction.CANDIDATE


class ActiveMotionFollowupPolicy:
    """Recognize a new credible motion track during an active EMA episode.

    The policy is deliberately bounded and synchronous. It owns no worker or
    queue; the camera's existing analysis worker calls it after EMA fusion and
    commits a candidate only after the normal motion queue accepts the trigger.
    """

    def __init__(
        self,
        *,
        maximum_anchors: int = 2,
        minimum_anchor_interval_seconds: float = 1.5,
        maximum_overlap: float = 0.10,
        minimum_center_distance: float = 0.10,
        maximum_coverage_regions: int = 32,
    ) -> None:
        self.maximum_anchors = max(1, int(maximum_anchors))
        self.minimum_anchor_interval_seconds = max(
            0.0,
            float(minimum_anchor_interval_seconds),
        )
        self.maximum_overlap = min(1.0, max(0.0, float(maximum_overlap)))
        self.minimum_center_distance = min(
            1.0,
            max(0.0, float(minimum_center_distance)),
        )
        self.maximum_coverage_regions = max(4, int(maximum_coverage_regions))
        self.reset()

    def reset(self) -> None:
        self._event_key = ""
        self._coverage: list[MotionRegion] = []
        self._known_track_ids: set[int] = set()
        self._anchor_count = 0
        self._last_anchor_at = 0.0

    def consider(
        self,
        result: MotionQualificationResult,
        captured_at: float,
        *,
        credible_motion: bool,
    ) -> ActiveMotionFollowupDecision:
        features = result.features
        phase = str(features.get("event_state_phase") or "")
        transition = str(features.get("event_state_transition") or "")
        event_key = str(features.get("event_state_key") or "")
        regions = self._regions(features.get("motion_regions"))
        track_id = self._track_id(features.get("motion_region_track_id"))

        if phase != "active" or not event_key:
            self.reset()
            return ActiveMotionFollowupDecision(ActiveMotionFollowupAction.INACTIVE)

        if transition == "activation_threshold" or event_key != self._event_key:
            self.reset()
            self._event_key = event_key
            self._last_anchor_at = captured_at
            self._remember(track_id, regions)
            return ActiveMotionFollowupDecision(
                ActiveMotionFollowupAction.BASELINE,
                event_key=event_key,
                track_id=track_id,
                region=regions[-1] if regions else None,
            )

        if (
            not credible_motion
            or not regions
            or result.score < result.threshold
            or not math.isfinite(captured_at)
        ):
            return ActiveMotionFollowupDecision(
                ActiveMotionFollowupAction.NOT_CREDIBLE,
                event_key=event_key,
                track_id=track_id,
            )

        if track_id is not None and track_id in self._known_track_ids:
            self._append_coverage(regions)
            return ActiveMotionFollowupDecision(
                ActiveMotionFollowupAction.DUPLICATE,
                event_key=event_key,
                track_id=track_id,
                region=regions[-1],
            )

        region = regions[-1]
        overlap = max(
            (self._intersection_over_union(region, known) for known in self._coverage),
            default=0.0,
        )
        center_distance = min(
            (self._center_distance(region, known) for known in self._coverage),
            default=1.0,
        )
        if (
            self._coverage
            and (
                overlap > self.maximum_overlap
                or center_distance < self.minimum_center_distance
            )
        ):
            # A tracker ID can change after a brief miss. Remember the new ID
            # when its geometry still describes the existing motion so churn
            # cannot repeatedly present the same subject as a novel candidate.
            self._remember(track_id, regions)
            return ActiveMotionFollowupDecision(
                ActiveMotionFollowupAction.DUPLICATE,
                event_key=event_key,
                track_id=track_id,
                region=region,
                maximum_overlap=overlap,
                nearest_center_distance=center_distance,
            )

        if self._anchor_count >= self.maximum_anchors:
            self._remember(track_id, regions)
            return ActiveMotionFollowupDecision(
                ActiveMotionFollowupAction.EPISODE_LIMIT,
                event_key=event_key,
                track_id=track_id,
                region=region,
                anchor_index=self._anchor_count + 1,
                maximum_overlap=overlap,
                nearest_center_distance=center_distance,
            )
        if (
            self._last_anchor_at > 0.0
            and captured_at - self._last_anchor_at
            < self.minimum_anchor_interval_seconds
        ):
            return ActiveMotionFollowupDecision(
                ActiveMotionFollowupAction.RATE_LIMITED,
                event_key=event_key,
                track_id=track_id,
                region=region,
                anchor_index=self._anchor_count + 1,
                maximum_overlap=overlap,
                nearest_center_distance=center_distance,
            )
        return ActiveMotionFollowupDecision(
            ActiveMotionFollowupAction.CANDIDATE,
            event_key=event_key,
            track_id=track_id,
            region=region,
            anchor_index=self._anchor_count + 1,
            maximum_overlap=overlap,
            nearest_center_distance=center_distance,
        )

    def commit(
        self,
        decision: ActiveMotionFollowupDecision,
        captured_at: float,
        regions: object,
    ) -> bool:
        if (
            not decision.admitted
            or decision.event_key != self._event_key
            or self._anchor_count >= self.maximum_anchors
        ):
            return False
        normalized = self._regions(regions)
        self._anchor_count += 1
        self._last_anchor_at = captured_at
        self._remember(
            decision.track_id,
            normalized or ([decision.region] if decision.region is not None else []),
        )
        return True

    def _remember(
        self,
        track_id: int | None,
        regions: Iterable[MotionRegion],
    ) -> None:
        if track_id is not None:
            self._known_track_ids.add(track_id)
        self._append_coverage(regions)

    def _append_coverage(self, regions: Iterable[MotionRegion]) -> None:
        self._coverage.extend(regions)
        if len(self._coverage) > self.maximum_coverage_regions:
            del self._coverage[: len(self._coverage) - self.maximum_coverage_regions]

    @staticmethod
    def _track_id(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            track_id = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return None
        return track_id if track_id > 0 else None

    @staticmethod
    def _regions(value: object) -> list[MotionRegion]:
        if not isinstance(value, (list, tuple)):
            return []
        normalized: list[MotionRegion] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(component) for component in item)
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(component) for component in (x1, y1, x2, y2)):
                continue
            x1, x2 = sorted((min(1.0, max(0.0, x1)), min(1.0, max(0.0, x2))))
            y1, y2 = sorted((min(1.0, max(0.0, y1)), min(1.0, max(0.0, y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            normalized.append((x1, y1, x2, y2))
        return normalized

    @staticmethod
    def _intersection_over_union(left: MotionRegion, right: MotionRegion) -> float:
        intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
            0.0,
            min(left[3], right[3]) - max(left[1], right[1]),
        )
        if intersection <= 0.0:
            return 0.0
        left_area = (left[2] - left[0]) * (left[3] - left[1])
        right_area = (right[2] - right[0]) * (right[3] - right[1])
        return intersection / max(1e-9, left_area + right_area - intersection)

    @staticmethod
    def _center_distance(left: MotionRegion, right: MotionRegion) -> float:
        left_center = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
        right_center = ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0)
        return math.dist(left_center, right_center)

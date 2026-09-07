"""Offline TrackTrack-inspired association, not a replacement TrackTrack runtime.

HMIoU and confidence projection follow the concepts documented at
https://docs.ultralytics.com/reference/trackers/track_tracker/ . Direction uses
SurvNG's center history rather than detector-jitter-sensitive corner velocities.
No upstream code, encoder, Kalman state, or track-initialization policy is used.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any

from ..config import ObjectTrackingConfig
from .bytetrack import ObjectTrack
from .geometry import _confidence, _iou
from .hybrid import DetectionBatch, HybridObjectTracker
from .types import Box

IMPLEMENTATION = "survng_hybrid_multicue"
CUE_VERSION = "survng_association_cues_v1"
HORIZON_SECONDS = 2.0
TRACE_LIMIT = 64


def _height_modulated_iou(left: Box, right: Box) -> tuple[float, float]:
    """Return IoU and IoU multiplied by vertical intersection/enclosing span."""
    overlap = _iou(left, right)
    height_overlap = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    height_union = max(left[3], right[3]) - min(left[1], right[1])
    return overlap, overlap * height_overlap / max(height_union, 1e-9)


def _direction_distance(track: ObjectTrack, box: Box, captured_at: float) -> tuple[float | None, str]:
    """Soft center-direction distance; unavailable history is neutral, not bad."""
    gap = captured_at - track.last_seen
    if not 0.0 < gap <= HORIZON_SECONDS:
        return None, "timestamp_gap"
    if len(track.trajectory) < 3:
        return None, "short_history"
    points = track.trajectory[-3:]
    if not all(math.isfinite(value) for point in points for value in point):
        return None, "invalid_history"
    if not points[0][0] < points[1][0] < points[2][0]:
        return None, "timestamp_history"
    if any(b[0] - a[0] > HORIZON_SECONDS for a, b in zip(points, points[1:])):
        return None, "stale_history"
    noise = max(2.0, 0.02 * math.hypot(track.box[2] - track.box[0], track.box[3] - track.box[1]))
    steps = [(b[1] - a[1], b[2] - a[2]) for a, b in zip(points, points[1:])]
    lengths = [math.hypot(*step) for step in steps]
    if min(lengths) < noise:
        return None, "stationary_or_jitter"
    # Do not extrapolate a direction through a recent sharp turn.
    if sum(a * b for a, b in zip(*steps)) / (lengths[0] * lengths[1]) < 0.5:
        return None, "recent_turn"
    vx = (track.velocity[0] + track.velocity[2]) / 2.0
    vy = (track.velocity[1] + track.velocity[3]) / 2.0
    dx = (box[0] + box[2] - track.box[0] - track.box[2]) / 2.0
    dy = (box[1] + box[3] - track.box[1] - track.box[3]) / 2.0
    speed, movement = math.hypot(vx, vy), math.hypot(dx, dy)
    if not all(math.isfinite(value) for value in (speed, movement)) or movement < noise or speed * gap < noise * 0.5:
        return None, "stationary_or_jitter"
    cosine = (vx * dx + vy * dy) / (speed * movement)
    return math.acos(max(-1.0, min(1.0, cosine))) / math.pi, "available"


class HybridMultiCueObjectTracker(HybridObjectTracker):
    """Refine ambiguous strict geometry without changing recovery or new births.

    Only already-admissible pairs can be adjusted. Uncontested continuations
    keep their exact production scores. Cue penalties cannot create new edges
    or reward maximizing match count. No appearance embedding is requested here.
    Optional weights are for offline ablation, not production configuration.
    """

    def __init__(
        self, config: ObjectTrackingConfig, high_confidence_threshold: float, *,
        overlap_weight: float = 0.25, confidence_weight: float = 0.10,
        direction_weight: float = 0.05,
    ) -> None:
        super().__init__(config, high_confidence_threshold)
        self._weights = {"overlap": overlap_weight, "confidence": confidence_weight, "direction": direction_weight}
        if any(not math.isfinite(value) or value < 0.0 for value in self._weights.values()) or sum(self._weights.values()) > 0.5:
            raise ValueError("cue weights must be finite, nonnegative, and sum to at most 0.5")
        # At most two actual observations per allocated ID; bounded by the
        # inherited max_tracks_per_session, including completed identities.
        self._confidence_samples: dict[int, tuple[tuple[float, float], ...]] = {}
        self._counts = {"valid_pairs": 0, "ambiguous_pairs": 0, "adjusted_pairs": 0,
                        "confidence_available": 0, "direction_available": 0}
        self._totals = {key: 0.0 for key in (
            "base_score", "overlap_penalty", "confidence_penalty", "direction_penalty", "final_score",
        )}
        self._pair_samples: deque[dict[str, Any]] = deque(maxlen=TRACE_LIMIT)

    def update(
        self, detections: list[dict[str, Any]], captured_at: float, *, confirm_new: bool = False,
    ) -> list[dict[str, Any]]:
        tracked = super().update(detections, captured_at, confirm_new=confirm_new)
        for item in tracked:
            track_id = item["track_id"]
            track = self._tracks[track_id]
            previous = self._confidence_samples.get(track_id, ())
            # A backwards/equal clock must not invent a confidence slope.
            if previous and captured_at <= previous[-1][0]:
                previous = ()
            self._confidence_samples[track_id] = (*previous[-1:], (track.last_seen, track.confidence))
        return tracked

    def _projected_confidence(self, track_id: int, captured_at: float) -> tuple[float | None, str]:
        samples = self._confidence_samples.get(track_id, ())
        if len(samples) < 2:
            return None, "short_history"
        (previous_at, previous), (last_at, last) = samples
        interval, gap = last_at - previous_at, captured_at - last_at
        if not 1e-3 < interval <= HORIZON_SECONDS or not 0.0 < gap <= HORIZON_SECONDS:
            return None, "timestamp_gap"
        if not all(math.isfinite(value) for sample in samples for value in sample):
            return None, "invalid_history"
        # Wall-clock slope, extrapolated by no more than one observed interval.
        projected = last + (last - previous) * min(gap / interval, 1.0)
        return max(0.0, min(1.0, projected)), "available"

    def _geometry_scores(
        self, track_ids: list[int], detections: DetectionBatch, captured_at: float,
    ) -> list[list[float]]:
        scores = super()._geometry_scores(track_ids, detections, captured_at)
        if not track_ids or not detections:
            return scores
        row_counts = [sum(score > 0.0 for score in row) for row in scores]
        column_counts = [sum(row[column] > 0.0 for row in scores) for column in range(len(detections))]
        for row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            predicted = track.predicted_box(captured_at)
            projected, confidence_reason = self._projected_confidence(track_id, captured_at)
            for column, (index, detection, box) in enumerate(detections):
                base = scores[row][column]
                if base <= 0.0:
                    continue  # Class, age, scale and geometry gates remain authoritative.
                self._counts["valid_pairs"] += 1
                self._totals["base_score"] += base
                if row_counts[row] == 1 and column_counts[column] == 1:
                    self._totals["final_score"] += base
                    continue
                self._counts["ambiguous_pairs"] += 1
                overlap, hmiou = _height_modulated_iou(predicted, box)
                direction, direction_reason = _direction_distance(track, box, captured_at)
                confidence = abs(projected - _confidence(detection)) if projected is not None else None
                # Replace a fraction of IoU with HMIoU, not the entire existing
                # containment/center fallback. Zero-IoU recovery stays possible.
                penalties = {
                    "overlap_penalty": self._weights["overlap"] * max(0.0, overlap - hmiou),
                    "confidence_penalty": self._weights["confidence"] * (confidence or 0.0),
                    "direction_penalty": self._weights["direction"] * (direction or 0.0),
                }
                score = max(0.0, base - sum(penalties.values()))
                scores[row][column] = score
                self._counts["adjusted_pairs"] += int(score != base)
                self._counts["confidence_available"] += int(confidence is not None)
                self._counts["direction_available"] += int(direction is not None)
                for key, value in penalties.items():
                    self._totals[key] += value
                self._totals["final_score"] += score
                self._pair_samples.append({
                    "captured_at": captured_at, "track_id": track_id, "detection_index": index,
                    "label": track.label, "detection_confidence": _confidence(detection),
                    "base_score": base, "iou": overlap, "hmiou": hmiou,
                    "projected_confidence": projected, "confidence_distance": confidence,
                    "confidence_status": confidence_reason, "direction_distance": direction,
                    "direction_status": direction_reason, **penalties, "final_score": score,
                })
        return scores

    def diagnostics(self) -> dict[str, Any]:
        return {**super().diagnostics(), "association_cues": {
            "version": CUE_VERSION, "experimental": True, "weights": dict(self._weights),
            "scope": "ambiguous_strict_geometry_only", "motion_timebase": "elapsed_seconds",
            "horizon_seconds": HORIZON_SECONDS, "new_track_policy": "unchanged",
            "appearance_policy": "unchanged_selective_reid", "counts": dict(self._counts),
            "score_totals": dict(self._totals),
            "pair_samples_note": "Last 64 evaluated ambiguous pairs, not necessarily selected assignments.",
            "pair_samples_truncated": max(0, self._counts["ambiguous_pairs"] - len(self._pair_samples)),
            "pair_samples": copy.deepcopy(list(self._pair_samples)),
        }}

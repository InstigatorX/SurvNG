"""Offline-only hybrid candidate; deliberately absent from the live registry."""

from __future__ import annotations

from typing import Any

from ..config import ObjectTrackingConfig
from .assignment import maximum_weight_assignment
from .bytetrack import ByteTrackObjectTracker
from .types import Box

DetectionBatch = list[tuple[int, dict[str, Any], Box]]


class HybridCandidateObjectTracker(ByteTrackObjectTracker):
    """Evaluate three bounded association fixes against the production hybrid.

    Reuse track creation, retention, appearance recovery, histories and output.
    Only motion-size prediction and the geometry association schedule differ.
    Appearance-based crossing disambiguation is intentionally not added here.
    """

    def __init__(self, config: ObjectTrackingConfig, high_confidence_threshold: float) -> None:
        super().__init__(config, high_confidence_threshold)
        self._pending_high: DetectionBatch | None = None

    def update(
        self,
        detections: list[dict[str, Any]],
        captured_at: float,
        *,
        confirm_new: bool = False,
    ) -> list[dict[str, Any]]:
        self._pending_high = None
        try:
            tracked = super().update(detections, captured_at, confirm_new=confirm_new)
        finally:
            self._pending_high = None
        # Predict center translation using the existing smoothed velocity, but
        # retain the last measured dimensions. Detector size jitter must not
        # extrapolate a valid box into a collapsed or inverted one. Normalizing
        # after every update also covers tracks observed by appearance recovery.
        for track in self._tracks.values():
            vx = (track.velocity[0] + track.velocity[2]) / 2.0
            vy = (track.velocity[1] + track.velocity[3]) / 2.0
            track.velocity = (vx, vy, vx, vy)
        return tracked

    def _associate(
        self,
        detections: DetectionBatch,
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        # The base update calls this hook exactly twice: high, then low. Delay
        # both passes until low candidates are visible without duplicating its
        # detection filtering, seeding, creation or expiry implementation.
        if self._pending_high is None:
            self._pending_high = detections
            return
        high = self._pending_high
        self._associate_geometry(high, captured_at, unmatched_tracks, assignments)
        self._associate_geometry(detections, captured_at, unmatched_tracks, assignments)
        self._associate_unambiguous(
            [*high, *detections], captured_at, unmatched_tracks, assignments,
        )
        self._associate_appearance(high, captured_at, unmatched_tracks, assignments)
        self._associate_appearance(detections, captured_at, unmatched_tracks, assignments)

    def _associate_geometry(
        self,
        detections: DetectionBatch,
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        track_ids = sorted(unmatched_tracks)
        if not track_ids or not detections:
            return
        scores = [[0.0] * len(detections) for _ in track_ids]
        for row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            if captured_at - track.last_seen > self._association_stale_limit(track):
                continue
            for column, (_index, detection, box) in enumerate(detections):
                if track.label != str(detection.get("label")):
                    continue
                score = self._geometry_score(
                    track.predicted_box(captured_at), box, allow_scale_jump=track.seeded,
                )
                if score is not None:
                    scores[row][column] = score
        # One additional valid edge outweighs all possible score differences.
        # Within that cardinality, retain the maximum total geometry score.
        bonus = min(len(track_ids), len(detections)) * max(map(max, scores)) + 1.0
        weights = [[score + bonus if score > 0 else 0.0 for score in row] for row in scores]
        for row, column in maximum_weight_assignment(weights):
            track_id = track_ids[row]
            index, detection, box = detections[column]
            self._observe_geometry(self._tracks[track_id], detection, captured_at, box)
            unmatched_tracks.remove(track_id)
            assignments[index] = track_id
            self._association_counts["geometry"] += 1

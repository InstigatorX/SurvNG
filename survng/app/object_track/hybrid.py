from __future__ import annotations

from typing import Any

from ..config import ObjectTrackingConfig
from .assignment import maximum_weight_assignment
from .bytetrack import ByteTrackObjectTracker
from .types import Box

DetectionBatch = list[tuple[int, dict[str, Any], Box]]


class HybridObjectTracker(ByteTrackObjectTracker):
    """SurvNG production tracker with wall-clock lifecycle and global association.

    Reuse the established track creation, retention, appearance recovery,
    histories, depth metadata, and output contract from ByteTrackObjectTracker.
    Production association differs in three bounded ways:

    * predict center translation while retaining the last measured box size;
    * recover high-confidence appearance before competing low-confidence
      geometry, while deferring relaxed matches for contested labels; and
    * use maximum-weight one-to-one assignment instead of greedy edge selection.

    These changes preserve SurvNG's timestamp-based lifecycle and selective ReID
    behavior while reducing avoidable fragmentation in ambiguous scenes.
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
        # The base update calls this hook exactly twice: high, then low. Wait
        # for low candidates before deciding which relaxed matches are safe.
        if self._pending_high is None:
            self._pending_high = detections
            return

        high = self._pending_high
        self._associate_geometry(high, captured_at, unmatched_tracks, assignments)
        # Keep cheap single-candidate recovery when no low-confidence detection
        # of that label could be displaced. Contested labels must wait until
        # low geometry is evaluated; unrelated labels need no extra ReID work.
        low_labels = {str(item[1].get("label") or "") for item in detections}
        self._associate_unambiguous(
            [item for item in high if str(item[1].get("label") or "") not in low_labels],
            captured_at, unmatched_tracks, assignments,
        )
        # Resolve strong high-confidence appearance recovery before a weak
        # low-confidence box can consume the same identity. The existing ReID
        # path applies per-label thresholds and requests each embedding lazily.
        self._associate_appearance(high, captured_at, unmatched_tracks, assignments)
        self._associate_geometry(detections, captured_at, unmatched_tracks, assignments)
        self._associate_unambiguous(
            [*high, *detections],
            captured_at,
            unmatched_tracks,
            assignments,
        )
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
                    track.predicted_box(captured_at),
                    box,
                    allow_scale_jump=track.seeded,
                )
                if score is not None:
                    scores[row][column] = score

        maximum_score = max(map(max, scores), default=0.0)
        if maximum_score <= 0.0:
            return

        # Maximize evidence, not match count. Zero-weight dummy assignments
        # let tracks remain unmatched; a strong continuation must not be traded
        # for weaker pairs solely to avoid allocating another identity.
        for row, column in maximum_weight_assignment(scores):
            track_id = track_ids[row]
            index, detection, box = detections[column]
            self._observe_geometry(self._tracks[track_id], detection, captured_at, box)
            unmatched_tracks.remove(track_id)
            assignments[index] = track_id
            self._association_counts["geometry"] += 1

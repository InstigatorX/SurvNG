from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any

import numpy as np

from ..config import ObjectTrackingConfig
from .geometry import _appearance, _box, _confidence, _ensure_detection_appearance, _iou
from .types import Box, ObjectTrackerBackend


@dataclass(slots=True)
class ObjectTrack:
    track_id: int
    label: str
    box: Box
    first_seen: float
    last_seen: float
    confidence: float
    max_confidence: float
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    velocity: Box = (0.0, 0.0, 0.0, 0.0)
    zones: set[str] = field(default_factory=set)
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)
    box_history: list[tuple[float, float, float, float, float]] = field(default_factory=list)
    appearance: np.ndarray | None = field(default=None, repr=False)
    reid_matches: int = 0
    reid_recovery_history: list[dict[str, Any]] = field(default_factory=list)
    seeded: bool = False

    def predicted_box(self, captured_at: float) -> Box:
        elapsed = max(0.0, min(captured_at - self.last_seen, 2.0))
        return tuple(
            coordinate + speed * elapsed
            for coordinate, speed in zip(self.box, self.velocity, strict=True)
        )  # type: ignore[return-value]

    def observe(self, detection: dict[str, Any], captured_at: float, box: Box) -> None:
        # Capture timestamps are wall-clock values and can briefly move backwards
        # after a clock correction. Keep track chronology monotonic even though
        # frame freshness is independently enforced with a monotonic token.
        elapsed = captured_at - self.last_seen
        captured_at = max(captured_at, self.last_seen)
        if elapsed > 1e-3:
            instantaneous = tuple(
                (coordinate - previous) / elapsed
                for coordinate, previous in zip(box, self.box, strict=True)
            )
            self.velocity = tuple(
                prior * 0.65 + current * 0.35
                for prior, current in zip(self.velocity, instantaneous, strict=True)
            )  # type: ignore[assignment]
        self.box = box
        self.last_seen = captured_at
        self.confidence = _confidence(detection)
        self.max_confidence = max(self.max_confidence, self.confidence)
        self.hits += 1
        self.missed = 0
        self.seeded = False
        self.zones.update(str(zone) for zone in detection.get("zones", []) if zone)
        next_appearance = _appearance(detection.get("_tracking_embedding"))
        if next_appearance is not None:
            if self.appearance is not None and self.appearance.shape == next_appearance.shape:
                next_appearance = _appearance(self.appearance * 0.8 + next_appearance * 0.2)
            self.appearance = next_appearance
        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[1] + box[3]) / 2.0
        self.trajectory.append((round(captured_at, 3), round(center_x, 1), round(center_y, 1)))
        self.box_history.append((
            round(captured_at, 3),
            round(box[0], 1),
            round(box[1], 1),
            round(box[2], 1),
            round(box[3], 1),
        ))
        if len(self.trajectory) > 60:
            del self.trajectory[:-60]
        if len(self.box_history) > 60:
            del self.box_history[:-60]

    def summary(self, *, active: bool) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "state": "confirmed" if self.confirmed and active else "lost" if self.confirmed else "tentative",
            "first_seen": datetime.fromtimestamp(self.first_seen, timezone.utc).isoformat(),
            "last_seen": datetime.fromtimestamp(self.last_seen, timezone.utc).isoformat(),
            "duration_seconds": round(max(0.0, self.last_seen - self.first_seen), 3),
            "observations": self.hits,
            "max_confidence": round(self.max_confidence, 4),
            "box": {
                "x1": round(self.box[0], 1),
                "y1": round(self.box[1], 1),
                "x2": round(self.box[2], 1),
                "y2": round(self.box[3], 1),
            },
            "zones": sorted(self.zones),
            "trajectory": [list(point) for point in self.trajectory],
            "box_history": [list(sample) for sample in self.box_history],
            "reid_matches": self.reid_matches,
            "reid_recovery_history": [dict(item) for item in self.reid_recovery_history],
        }

class ByteTrackObjectTracker:
    """Small tracking-by-detection engine using ByteTrack's two-pass association."""

    def __init__(self, config: ObjectTrackingConfig, high_confidence_threshold: float) -> None:
        self.config = config
        self.high_confidence_threshold = max(
            config.low_confidence_threshold,
            float(high_confidence_threshold),
        )
        self._tracks: dict[int, ObjectTrack] = {}
        self._completed: dict[int, ObjectTrack] = {}
        self._next_track_id = 1
        self._association_counts = {
            "new_track": 0,
            "geometry": 0,
            "appearance_recovery": 0,
        }
        self._reid_avoided_geometry_matches = 0
        self._reid_avoided_by_label: dict[str, int] = {}

    def update(
        self,
        detections: list[dict[str, Any]],
        captured_at: float,
        *,
        confirm_new: bool = False,
    ) -> list[dict[str, Any]]:
        usable = [
            (index, detection, parsed)
            for index, detection in enumerate(detections)
            if self.config.tracks_label(detection.get("label"))
            and (parsed := _box(detection.get("box"))) is not None
        ]
        high = [
            item
            for item in usable
            if confirm_new
            or _confidence(item[1]) >= self.high_confidence_threshold
        ]
        low = [
            item
            for item in usable
            if not confirm_new
            and self.config.low_confidence_threshold
            <= _confidence(item[1])
            < self.high_confidence_threshold
        ]
        unmatched_tracks = set(self._tracks)
        assignments: dict[int, int] = {}
        self._associate(high, captured_at, unmatched_tracks, assignments)
        self._associate(low, captured_at, unmatched_tracks, assignments)

        for index, detection, box in high:
            if index in assignments or detection.get("incident_eligible") is False:
                continue
            if len(self._tracks) + len(self._completed) >= self.config.max_tracks_per_session:
                break
            confidence = _confidence(detection)
            try:
                candidate_first_seen = float(detection.get("_tracking_first_seen_at"))
                first_seen = (
                    min(captured_at, candidate_first_seen)
                    if np.isfinite(candidate_first_seen) and candidate_first_seen >= 0.0
                    else captured_at
                )
            except (TypeError, ValueError):
                first_seen = captured_at
            track = ObjectTrack(
                track_id=self._next_track_id,
                label=str(detection["label"]),
                box=box,
                first_seen=first_seen,
                last_seen=captured_at,
                confidence=confidence,
                max_confidence=confidence,
                confirmed=confirm_new or self.config.min_confirmations <= 1,
                zones={str(zone) for zone in detection.get("zones", []) if zone},
                trajectory=[(
                    round(captured_at, 3),
                    round((box[0] + box[2]) / 2.0, 1),
                    round((box[1] + box[3]) / 2.0, 1),
                )],
                box_history=[(
                    round(captured_at, 3),
                    round(box[0], 1),
                    round(box[1], 1),
                    round(box[2], 1),
                    round(box[3], 1),
                )],
                appearance=_ensure_detection_appearance(detection, "track_seed"),
                seeded=confirm_new,
            )
            self._tracks[track.track_id] = track
            assignments[index] = track.track_id
            self._next_track_id += 1
            self._association_counts["new_track"] += 1

        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.missed += 1
        self._expire(captured_at)

        tracked: list[dict[str, Any]] = []
        for index, detection, _box_value in usable:
            track_id = assignments.get(index)
            if track_id is None:
                continue
            track = self._tracks.get(track_id)
            if track is None:
                continue
            tracked.append({
                **{
                    key: value
                    for key, value in detection.items()
                    if not key.startswith("_tracking_")
                },
                "track_id": track_id,
                "track_state": "confirmed" if track.confirmed else "tentative",
                "track_observations": track.hits,
            })
        return tracked

    def _associate(
        self,
        detections: list[tuple[int, dict[str, Any], Box]],
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        candidates: list[tuple[float, int, int, dict[str, Any], Box]] = []
        for index, detection, box in detections:
            for track_id in unmatched_tracks:
                track = self._tracks[track_id]
                if track.label != str(detection.get("label")):
                    continue
                if captured_at - track.last_seen > self._association_stale_limit(track):
                    continue
                score = self._geometry_score(
                    track.predicted_box(captured_at),
                    box,
                    allow_scale_jump=track.seeded,
                )
                if score is not None:
                    candidates.append((score, track_id, index, detection, box))
        used_detections: set[int] = set()
        for _score, track_id, index, detection, box in sorted(
            candidates,
            key=lambda item: item[0],
            reverse=True,
        ):
            if track_id not in unmatched_tracks or index in used_detections:
                continue
            track = self._tracks[track_id]
            self._observe_geometry(track, detection, captured_at, box)
            unmatched_tracks.remove(track_id)
            used_detections.add(index)
            assignments[index] = track_id
            self._association_counts["geometry"] += 1

        self._associate_unambiguous(
            detections,
            captured_at,
            unmatched_tracks,
            assignments,
        )
        self._associate_appearance(detections, captured_at, unmatched_tracks, assignments)

    def _associate_unambiguous(
        self,
        detections: list[tuple[int, dict[str, Any], Box]],
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        """Bridge a short geometry jump when one identity is the only candidate.

        Perspective changes and detector box jitter can move a nearby person's
        center by more than the normal association threshold.  When exactly one
        confirmed track and one detection remain for a label, a bounded relaxed
        geometry check is safer than manufacturing a second identity.
        """
        labels = {
            str(detection.get("label") or "")
            for index, detection, _box_value in detections
            if index not in assignments
        }
        maximum_gap = min(
            self.config.lost_timeout_seconds,
            max(1.0, 2.0 / max(0.1, self.config.sample_fps)),
        )
        for label in labels:
            remaining_detections = [
                item
                for item in detections
                if item[0] not in assignments
                and str(item[1].get("label") or "") == label
            ]
            remaining_tracks = [
                track_id
                for track_id in unmatched_tracks
                if self._tracks[track_id].label == label
                and self._tracks[track_id].confirmed
                and captured_at - self._tracks[track_id].last_seen <= (
                    self._association_stale_limit(self._tracks[track_id])
                    if self._tracks[track_id].seeded
                    else maximum_gap
                )
            ]
            if len(remaining_detections) != 1 or len(remaining_tracks) != 1:
                continue
            index, detection, box = remaining_detections[0]
            track_id = remaining_tracks[0]
            track = self._tracks[track_id]
            score = self._geometry_score(
                track.predicted_box(captured_at),
                box,
                center_distance_multiplier=2.0 if track.seeded else 1.75,
                allow_scale_jump=track.seeded,
            )
            if score is None:
                continue
            self._observe_geometry(track, detection, captured_at, box)
            unmatched_tracks.remove(track_id)
            assignments[index] = track_id
            self._association_counts["geometry"] += 1

    def _observe_geometry(
        self,
        track: ObjectTrack,
        detection: dict[str, Any],
        captured_at: float,
        box: Box,
    ) -> None:
        supports_reid = (
            self.config.appearance_reid_enabled
            and self.config.reid_enabled_for_label(track.label)
        )
        refresh_due = (
            track.appearance is None
            or track.hits % self.config.reid_refresh_interval_frames == 0
        )
        if supports_reid and refresh_due:
            _ensure_detection_appearance(
                detection,
                "track_seed" if track.appearance is None else "periodic_refresh",
            )
        elif supports_reid and callable(detection.get("_tracking_embedding_provider")):
            self._reid_avoided_geometry_matches += 1
            label = track.label.lower()
            self._reid_avoided_by_label[label] = (
                self._reid_avoided_by_label.get(label, 0) + 1
            )
        track.observe(detection, captured_at, box)
        track.confirmed = track.confirmed or track.hits >= self.config.min_confirmations

    def _geometry_score(
        self,
        left: Box,
        right: Box,
        *,
        center_distance_multiplier: float = 1.0,
        allow_scale_jump: bool = False,
    ) -> float | None:
        left_width = left[2] - left[0]
        left_height = left[3] - left[1]
        right_width = right[2] - right[0]
        right_height = right[3] - right[1]
        left_area = left_width * left_height
        right_area = right_width * right_height
        intersection = (
            max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
            * max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        )
        containment = intersection / max(1.0, min(left_area, right_area))
        area_ratio = min(left_area, right_area) / max(1.0, max(left_area, right_area))
        left_center = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
        right_center = ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0)
        distance = float(np.hypot(
            left_center[0] - right_center[0],
            left_center[1] - right_center[1],
        ))
        scale = max(1.0, float(np.hypot(
            max(left_width, right_width),
            max(left_height, right_height),
        )))
        distance_ratio = distance / scale
        # A detector can occasionally emit a scene-sized box around an object
        # for one frame.  Its overlap is deceptively high because it contains
        # the prior box, but accepting it produces a physically impossible
        # track jump.  Permit major scale changes only while the center remains
        # close; normal perspective growth and edge entry still satisfy this.
        if (
            not allow_scale_jump
            and right_area > left_area * 2.0
            and distance_ratio > self.config.match_center_distance_ratio * 0.55
        ):
            return None
        overlap = _iou(left, right)
        if overlap >= self.config.match_iou_threshold:
            return 1.0 + overlap
        if containment >= 0.55 and area_ratio >= 0.10:
            return 0.9 + containment
        center_limit = (
            self.config.match_center_distance_ratio
            * max(1.0, min(2.0, center_distance_multiplier))
        )
        if area_ratio >= 0.20 and distance_ratio <= center_limit:
            return 0.5 + (1.0 - distance_ratio / center_limit)
        return None

    def _associate_appearance(
        self,
        detections: list[tuple[int, dict[str, Any], Box]],
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        if not self.config.appearance_reid_enabled:
            return
        candidates: list[tuple[float, int, bool, int, dict[str, Any], Box]] = []
        for index, detection, box in detections:
            label = str(detection.get("label") or "").lower()
            if index in assignments or not self.config.reid_enabled_for_label(label):
                continue
            active_candidates = [
                (track_id, track)
                for track_id in unmatched_tracks
                if (
                    (track := self._tracks[track_id]).label
                    == str(detection.get("label"))
                    and track.appearance is not None
                    and captured_at - track.last_seen <= self.config.reid_max_age_seconds
                )
            ]
            completed_candidates = [
                (track_id, track)
                for track_id, track in self._completed.items()
                if (
                    track.label == str(detection.get("label"))
                    and track.appearance is not None
                    and captured_at - track.last_seen <= self.config.reid_max_age_seconds
                )
            ]
            if not active_candidates and not completed_candidates:
                continue
            embedding = _ensure_detection_appearance(detection, "geometry_recovery")
            if embedding is None:
                continue
            for track_id, track in active_candidates:
                if track.appearance.shape == embedding.shape:
                    candidates.append((
                        float(np.dot(track.appearance, embedding)),
                        track_id,
                        False,
                        index,
                        detection,
                        box,
                    ))
            for track_id, track in completed_candidates:
                if track.appearance.shape != embedding.shape:
                    continue
                candidates.append((
                    float(np.dot(track.appearance, embedding)),
                    track_id,
                    True,
                    index,
                    detection,
                    box,
                ))
        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        for score, track_id, completed, index, detection, box in sorted(
            candidates,
            key=lambda item: item[0],
            reverse=True,
        ):
            if (
                score < self.config.reid_threshold_for_label(
                    str(detection.get("label") or "")
                )
                or index in used_detections
                or track_id in used_tracks
            ):
                continue
            if completed:
                track = self._completed.pop(track_id, None)
                if track is None:
                    continue
                self._tracks[track_id] = track
            else:
                if track_id not in unmatched_tracks:
                    continue
                track = self._tracks[track_id]
                unmatched_tracks.remove(track_id)
            track.observe(detection, captured_at, box)
            track.reid_matches += 1
            track.reid_recovery_history.append({
                "captured_at": round(captured_at, 3),
                "similarity": round(score, 4),
                "resumed_completed_track": completed,
                "box": [round(value, 1) for value in box],
            })
            if len(track.reid_recovery_history) > 60:
                del track.reid_recovery_history[:-60]
            track.confirmed = track.confirmed or track.hits >= self.config.min_confirmations
            assignments[index] = track_id
            used_detections.add(index)
            used_tracks.add(track_id)
            self._association_counts["appearance_recovery"] += 1

    def _expire(self, captured_at: float) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if captured_at - track.last_seen > self.config.lost_timeout_seconds
        ]
        for track_id in expired:
            self._completed[track_id] = self._tracks.pop(track_id)

    def has_live_tracks(self, captured_at: float) -> bool:
        return any(
            captured_at - track.last_seen <= self.config.lost_timeout_seconds
            for track in self._tracks.values()
        )

    def summaries(self, captured_at: float) -> list[dict[str, Any]]:
        completed = [track.summary(active=False) for track in self._completed.values() if track.confirmed]
        active = [
            track.summary(
                active=captured_at - track.last_seen <= self.config.lost_timeout_seconds,
            )
            for track in self._tracks.values()
            if track.confirmed
        ]
        return sorted([*completed, *active], key=lambda item: int(item["track_id"]))

    def appearance_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for track in [*self._completed.values(), *self._tracks.values()]:
            if not track.confirmed or track.appearance is None:
                continue
            records.append({
                "track_id": track.track_id,
                "label": track.label,
                "embedding": track.appearance.copy(),
                "observation_count": track.hits,
                "quality": track.max_confidence,
                "first_seen": datetime.fromtimestamp(
                    track.first_seen,
                    timezone.utc,
                ).isoformat(),
                "last_seen": datetime.fromtimestamp(
                    track.last_seen,
                    timezone.utc,
                ).isoformat(),
            })
        return records

    def _association_stale_limit(self, track: ObjectTrack) -> float:
        if track.seeded:
            return max(
                self.config.lost_timeout_seconds,
                min(8.0, self.config.max_session_seconds),
            )
        return self.config.lost_timeout_seconds

    def diagnostics(self) -> dict[str, Any]:
        return {
            "association_counts": dict(self._association_counts),
            "reid_avoided_geometry_matches": self._reid_avoided_geometry_matches,
            "reid_avoided_by_label": dict(self._reid_avoided_by_label),
        }

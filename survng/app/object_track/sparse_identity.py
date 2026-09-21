"""Offline Sparse Identity tracker for SurvNG Compare / evaluation.

Keep Hybrid's wall-clock lifecycle and lazy ReID while adding cascaded
active-before-lost association, buffered IoU fallback, observation-centric
velocity reset, contested appearance veto, quality-gated multi-gallery memory,
provisional recovery, and co-occlusion freezes.

Production configuration normalizes this implementation name back to Hybrid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import ObjectTrackingConfig
from .assignment import maximum_weight_assignment
from .geometry import _appearance, _ensure_detection_appearance, _iou
from .hybrid import HybridObjectTracker
from .types import Box

DetectionBatch = list[tuple[int, dict[str, Any], Box]]


def _normalize_label(label: object) -> str:
    return str(label or "").strip().lower()


def _expand_box(box: Box, factor: float) -> Box:
    width = max(1.0, box[2] - box[0])
    height = max(1.0, box[3] - box[1])
    pad_x = width * factor * 0.5
    pad_y = height * factor * 0.5
    return (box[0] - pad_x, box[1] - pad_y, box[2] + pad_x, box[3] + pad_y)


def _center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _center_distance_ratio(left: Box, right: Box) -> float:
    left_center = _center(left)
    right_center = _center(right)
    distance = float(np.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]))
    scale = max(
        1.0,
        float(np.hypot(max(left[2] - left[0], right[2] - right[0]), max(left[3] - left[1], right[3] - right[1]))),
    )
    return distance / scale


def _crop_quality(detection: dict[str, Any], box: Box) -> float:
    confidence = float(detection.get("confidence") or 0.0)
    if not np.isfinite(confidence):
        confidence = 0.0
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    area_score = min(1.0, (width * height) / 12000.0)
    aspect = width / max(1.0, height)
    aspect_score = 1.0 if 0.2 <= aspect <= 1.2 else 0.55
    return max(0.0, min(1.0, 0.55 * confidence + 0.35 * area_score + 0.10 * aspect_score))


class SparseIdentityObjectTracker(HybridObjectTracker):
    """Compare-only tracker oriented at sparse CCTV identity retention."""

    def __init__(self, config: ObjectTrackingConfig, high_confidence_threshold: float) -> None:
        super().__init__(config, high_confidence_threshold)
        self._entity_aliases: dict[int, int] = {}
        self._co_occluded_until: dict[int, float] = {}
        self._provisional_hits: dict[int, int] = {}
        self._galleries: dict[int, list[np.ndarray]] = {}
        self._association_counts.update({
            "buffered_geometry": 0,
            "contested_appearance": 0,
            "entity_relink": 0,
            "co_occlusion_freeze": 0,
        })

    def update(
        self,
        detections: list[dict[str, Any]],
        captured_at: float,
        *,
        confirm_new: bool = False,
    ) -> list[dict[str, Any]]:
        for detection in detections:
            if "label" in detection:
                detection["label"] = _normalize_label(detection.get("label"))
        tracked = super().update(detections, captured_at, confirm_new=confirm_new)
        self._prune_completed(captured_at)
        self._mark_co_occlusions(captured_at)
        for item in tracked:
            track = self._tracks.get(int(item["track_id"]))
            if track is None:
                continue
            item["entity_id"] = int(getattr(track, "entity_id", track.track_id))
        return tracked

    def _associate(
        self,
        detections: DetectionBatch,
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        if self._pending_high is None:
            self._pending_high = detections
            return

        high = self._pending_high
        # Stage 1: strict geometry on high-confidence detections.
        self._associate_geometry(high, captured_at, unmatched_tracks, assignments)
        self._associate_contested(high, captured_at, unmatched_tracks, assignments)
        # Stage 2 leftovers: buffered geometry before appearance.
        self._associate_buffered_geometry(high, captured_at, unmatched_tracks, assignments, small=True)
        low_labels = {_normalize_label(item[1].get("label")) for item in detections}
        self._associate_unambiguous(
            [item for item in high if _normalize_label(item[1].get("label")) not in low_labels],
            captured_at,
            unmatched_tracks,
            assignments,
        )
        # Stage 3: appearance on remaining high (active before completed).
        self._associate_appearance(high, captured_at, unmatched_tracks, assignments)
        # Stage 4: low-confidence geometry rescue without gallery writes.
        self._associate_geometry(detections, captured_at, unmatched_tracks, assignments, allow_gallery_update=False)
        self._associate_buffered_geometry(
            detections, captured_at, unmatched_tracks, assignments, small=True, allow_gallery_update=False,
        )
        self._associate_buffered_geometry(
            [*high, *detections],
            captured_at,
            unmatched_tracks,
            assignments,
            small=False,
            allow_gallery_update=False,
        )
        self._associate_unambiguous(
            [*high, *detections],
            captured_at,
            unmatched_tracks,
            assignments,
        )
        self._associate_appearance(detections, captured_at, unmatched_tracks, assignments)
        remaining = [
            item for item in (*high, *detections) if item[0] not in assignments
        ]
        self._associate_entity_relink(remaining, captured_at, unmatched_tracks, assignments)

    def _associate_geometry(
        self,
        detections: DetectionBatch,
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
        *,
        allow_gallery_update: bool = True,
        buffer_factor: float = 0.0,
    ) -> None:
        track_ids = sorted(unmatched_tracks)
        if not track_ids or not detections:
            return

        scores = [[0.0] * len(detections) for _ in track_ids]
        for row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            if captured_at - track.last_seen > self._association_stale_limit(track):
                continue
            predicted = track.predicted_box(captured_at)
            if buffer_factor > 0.0:
                predicted = _expand_box(predicted, buffer_factor)
            for column, (_index, detection, box) in enumerate(detections):
                if _normalize_label(track.label) != _normalize_label(detection.get("label")):
                    continue
                score = self._geometry_score(
                    predicted,
                    box,
                    allow_scale_jump=track.seeded,
                )
                same_label_tracks = sum(
                    1
                    for other_id in track_ids
                    if _normalize_label(self._tracks[other_id].label)
                    == _normalize_label(track.label)
                    and self._tracks[other_id].appearance is not None
                )
                if (
                    allow_gallery_update
                    and track.appearance is not None
                    and self.config.reid_enabled_for_label(track.label)
                    and same_label_tracks >= 2
                ):
                    if not self._spatial_reid_ok(track, box, captured_at):
                        continue
                    embedding = _ensure_detection_appearance(detection, "fused_geometry")
                    similarity = (
                        self._appearance_similarity(track, embedding)
                        if embedding is not None
                        else None
                    )
                    if similarity is None:
                        if score is None:
                            continue
                        score *= 0.5
                    elif score is None:
                        if similarity < self.config.reid_threshold_for_label(track.label):
                            continue
                        score = 0.55 * (1.0 + similarity)
                    else:
                        score = 0.45 * score + 0.55 * (1.0 + similarity)
                elif score is None:
                    continue
                scores[row][column] = score

        if max(map(max, scores), default=0.0) <= 0.0:
            return

        # Defer geometrically ambiguous columns to contested appearance only when
        # ReID can break the tie. Without appearance, keep Hybrid-like geometry.
        if self.config.appearance_reid_enabled:
            for column in range(len(detections)):
                label = _normalize_label(detections[column][1].get("label"))
                if not self.config.reid_enabled_for_label(label):
                    continue
                column_scores = sorted(
                    (
                        scores[row][column]
                        for row in range(len(track_ids))
                        if scores[row][column] > 0
                    ),
                    reverse=True,
                )
                if len(column_scores) >= 2 and column_scores[0] - column_scores[1] < 0.35:
                    for row in range(len(track_ids)):
                        scores[row][column] = 0.0

        if max(map(max, scores), default=0.0) <= 0.0:
            return

        for row, column in maximum_weight_assignment(scores):
            track_id = track_ids[row]
            index, detection, box = detections[column]
            if index in assignments or track_id not in unmatched_tracks:
                continue
            self._observe_geometry(
                self._tracks[track_id],
                detection,
                captured_at,
                box,
                allow_gallery_update=allow_gallery_update,
                observation_centric=True,
            )
            unmatched_tracks.remove(track_id)
            assignments[index] = track_id
            key = "buffered_geometry" if buffer_factor > 0.0 else "geometry"
            self._association_counts[key] += 1

    def _associate_buffered_geometry(
        self,
        detections: DetectionBatch,
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
        *,
        small: bool,
        allow_gallery_update: bool = True,
    ) -> None:
        factor = (
            self.config.sparse_buffer_iou_small
            if small
            else self.config.sparse_buffer_iou_large
        )
        remaining = [
            item for item in detections if item[0] not in assignments
        ]
        self._associate_geometry(
            remaining,
            captured_at,
            unmatched_tracks,
            assignments,
            allow_gallery_update=allow_gallery_update,
            buffer_factor=factor,
        )

    def _associate_contested(
        self,
        detections: DetectionBatch,
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        """When several tracks remain plausible, require an appearance margin."""
        if not self.config.appearance_reid_enabled:
            return
        remaining = [item for item in detections if item[0] not in assignments]
        for index, detection, box in remaining:
            label = _normalize_label(detection.get("label"))
            if not self.config.reid_enabled_for_label(label):
                continue
            candidates = []
            for track_id in list(unmatched_tracks):
                track = self._tracks[track_id]
                if _normalize_label(track.label) != label:
                    continue
                if captured_at - track.last_seen > self._association_stale_limit(track):
                    continue
                if track.appearance is None:
                    continue
                if not self._spatial_reid_ok(track, box, captured_at):
                    continue
                predicted = track.predicted_box(captured_at)
                score = self._geometry_score(predicted, box, allow_scale_jump=track.seeded)
                if score is None:
                    score = 0.1 + max(0.0, 1.0 - _center_distance_ratio(predicted, box))
                candidates.append((score, track_id, track))
            if len(candidates) < 2:
                continue
            candidates.sort(reverse=True)
            if candidates[0][0] - candidates[1][0] > 0.25:
                continue
            embedding = _ensure_detection_appearance(detection, "contested_geometry")
            if embedding is None:
                continue
            ranked = []
            for _score, track_id, track in candidates:
                similarity = self._appearance_similarity(track, embedding)
                if similarity is None:
                    continue
                ranked.append((similarity, track_id, track))
            if len(ranked) < 2:
                continue
            ranked.sort(reverse=True)
            best, second = ranked[0], ranked[1]
            if best[0] < self.config.reid_threshold_for_label(label):
                continue
            if best[0] - second[0] < self.config.reid_top_two_margin:
                continue
            track_id = best[1]
            if track_id not in unmatched_tracks or index in assignments:
                continue
            self._observe_geometry(
                self._tracks[track_id],
                detection,
                captured_at,
                box,
                allow_gallery_update=True,
                observation_centric=True,
            )
            unmatched_tracks.remove(track_id)
            assignments[index] = track_id
            self._association_counts["contested_appearance"] += 1

    def _associate_appearance(
        self,
        detections: DetectionBatch,
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        if not self.config.appearance_reid_enabled:
            return
        remaining = [item for item in detections if item[0] not in assignments]
        if not remaining:
            return

        active_rows: list[tuple[int, Any, bool]] = []
        for track_id in sorted(unmatched_tracks):
            track = self._tracks[track_id]
            if track.appearance is None:
                continue
            if captured_at - track.last_seen > self.config.reid_max_age_seconds:
                continue
            active_rows.append((track_id, track, False))
        completed_rows: list[tuple[int, Any, bool]] = []
        for track_id, track in sorted(self._completed.items()):
            if track.appearance is None:
                continue
            if captured_at - track.last_seen > self.config.reid_max_age_seconds:
                continue
            completed_rows.append((track_id, track, True))

        # Active tracks consume detections before completed / long-lost ones.
        self._assign_appearance_block(
            remaining, active_rows, captured_at, unmatched_tracks, assignments,
        )
        remaining = [item for item in remaining if item[0] not in assignments]
        # When entity relink is enabled, long-lost completed tracks become new
        # tracklets that inherit entity_id instead of zombie trajectories.
        if not self.config.entity_relink_enabled:
            self._assign_appearance_block(
                remaining, completed_rows, captured_at, unmatched_tracks, assignments,
            )

    def _assign_appearance_block(
        self,
        detections: DetectionBatch,
        rows: list[tuple[int, Any, bool]],
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        if not detections or not rows:
            return
        scores = [[0.0] * len(detections) for _ in rows]
        for column, (_index, detection, box) in enumerate(detections):
            label = _normalize_label(detection.get("label"))
            if not self.config.reid_enabled_for_label(label):
                continue
            needs_embedding = False
            for track_id, track, _completed in rows:
                if _normalize_label(track.label) != label:
                    continue
                if not self._spatial_reid_ok(track, box, captured_at):
                    continue
                needs_embedding = True
                break
            if not needs_embedding:
                continue
            embedding = _ensure_detection_appearance(detection, "geometry_recovery")
            if embedding is None:
                continue
            for row, (_track_id, track, _completed) in enumerate(rows):
                if _normalize_label(track.label) != label:
                    continue
                if not self._spatial_reid_ok(track, box, captured_at):
                    continue
                similarity = self._appearance_similarity(track, embedding)
                if similarity is None:
                    continue
                if similarity < self.config.reid_threshold_for_label(label):
                    continue
                scores[row][column] = similarity

        # Top-two margin: zero edges that are too close to a runner-up.
        margin = self.config.reid_top_two_margin
        for column in range(len(detections)):
            column_scores = sorted(
                ((scores[row][column], row) for row in range(len(rows)) if scores[row][column] > 0),
                reverse=True,
            )
            if len(column_scores) >= 2 and column_scores[0][0] - column_scores[1][0] < margin:
                for _score, row in column_scores:
                    scores[row][column] = 0.0

        if max(map(max, scores), default=0.0) <= 0.0:
            return

        for row, column in maximum_weight_assignment(scores):
            track_id, track, completed = rows[row]
            index, detection, box = detections[column]
            if index in assignments:
                continue
            if completed:
                resumed = self._completed.pop(track_id, None)
                if resumed is None:
                    continue
                self._tracks[track_id] = resumed
                track = resumed
                self._provisional_hits[track_id] = self.config.reid_provisional_hits
            else:
                if track_id not in unmatched_tracks:
                    continue
                unmatched_tracks.remove(track_id)
            self._observe_geometry(
                track,
                detection,
                captured_at,
                box,
                allow_gallery_update=False,
                observation_centric=True,
            )
            track.reid_matches += 1
            similarity = float(scores[row][column])
            track.reid_recovery_history.append({
                "captured_at": round(captured_at, 3),
                "similarity": round(similarity, 4),
                "resumed_completed_track": completed,
                "box": [round(value, 1) for value in box],
            })
            if len(track.reid_recovery_history) > 60:
                del track.reid_recovery_history[:-60]
            track.confirmed = track.confirmed or track.hits >= self.config.min_confirmations
            assignments[index] = track_id
            self._association_counts["appearance_recovery"] += 1

    def _associate_entity_relink(
        self,
        detections: DetectionBatch,
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        if not self.config.entity_relink_enabled or not self.config.appearance_reid_enabled:
            return
        remaining = [item for item in detections if item[0] not in assignments]
        if not remaining:
            return
        for index, detection, box in remaining:
            label = _normalize_label(detection.get("label"))
            if not self.config.reid_enabled_for_label(label):
                continue
            embedding = _ensure_detection_appearance(detection, "entity_relink")
            if embedding is None:
                continue
            detection_zones = {
                str(zone) for zone in detection.get("zones", []) if zone
            }
            ranked: list[tuple[float, int, Any]] = []
            for track_id, track in self._completed.items():
                if _normalize_label(track.label) != label or track.appearance is None:
                    continue
                age = captured_at - track.last_seen
                if age <= self.config.lost_timeout_seconds or age > self.config.reid_max_age_seconds:
                    continue
                if not self._spatial_reid_ok(track, box, captured_at, relaxed=True):
                    continue
                similarity = self._appearance_similarity(track, embedding)
                if similarity is None or similarity < self.config.reid_threshold_for_label(label):
                    continue
                if detection_zones and track.zones and detection_zones.isdisjoint(track.zones):
                    # Soft scene incompatiblity: keep but do not prefer.
                    similarity -= 0.03
                elif detection_zones and track.zones and detection_zones & track.zones:
                    similarity += 0.02
                ranked.append((similarity, track_id, track))
            if len(ranked) < 1:
                continue
            ranked.sort(reverse=True)
            if len(ranked) >= 2 and ranked[0][0] - ranked[1][0] < self.config.reid_top_two_margin:
                continue
            _similarity, source_id, source = ranked[0]
            # Birth a new tracklet that inherits the entity ID (not the trajectory).
            if len(self._tracks) + len(self._completed) >= self.config.max_tracks_per_session:
                continue
            confidence = float(detection.get("confidence") or 0.0)
            from .bytetrack import ObjectTrack

            track = ObjectTrack(
                track_id=self._next_track_id,
                label=str(detection["label"]),
                box=box,
                first_seen=captured_at,
                last_seen=captured_at,
                confidence=confidence,
                max_confidence=confidence,
                confirmed=True,
                appearance=_appearance(embedding),
                seeded=False,
            )
            track.entity_id = int(getattr(source, "entity_id", source_id))
            self._tracks[track.track_id] = track
            self._galleries[track.track_id] = [
                item.copy() for item in self._galleries.get(source_id, [])[-self.config.reid_gallery_size:]
            ]
            if track.appearance is not None:
                self._galleries.setdefault(track.track_id, []).append(track.appearance.copy())
            self._entity_aliases[track.track_id] = track.entity_id
            self._provisional_hits[track.track_id] = self.config.reid_provisional_hits
            self._next_track_id += 1
            assignments[index] = track.track_id
            self._association_counts["entity_relink"] += 1
            self._association_counts["new_track"] += 1

    def _observe_geometry(
        self,
        track,
        detection: dict[str, Any],
        captured_at: float,
        box: Box,
        *,
        allow_gallery_update: bool = True,
        observation_centric: bool = False,
    ) -> None:
        previous_box = track.box
        previous_seen = track.last_seen
        frozen = (
            self._co_occluded_until.get(track.track_id, 0.0) > captured_at
            or self._provisional_hits.get(track.track_id, 0) > 0
            or self._incoming_boxes_co_occlude(box, captured_at, track.track_id)
        )
        supports_reid = (
            self.config.appearance_reid_enabled
            and self.config.reid_enabled_for_label(track.label)
        )
        quality = _crop_quality(detection, box)
        refresh_due = (
            track.appearance is None
            or track.hits % self.config.reid_refresh_interval_frames == 0
        )
        if supports_reid and refresh_due and allow_gallery_update and not frozen and quality >= 0.45:
            _ensure_detection_appearance(
                detection,
                "track_seed" if track.appearance is None else "periodic_refresh",
            )
        elif supports_reid and callable(detection.get("_tracking_embedding_provider")):
            self._reid_avoided_geometry_matches += 1
            label = _normalize_label(track.label)
            self._reid_avoided_by_label[label] = self._reid_avoided_by_label.get(label, 0) + 1

        # Observation-centric velocity: rebuild from endpoints after a gap.
        saved_embedding = detection.get("_tracking_embedding")
        should_update_appearance = (
            allow_gallery_update
            and not frozen
            and quality >= 0.45
            and saved_embedding is not None
        )
        if not should_update_appearance and "_tracking_embedding" in detection:
            detection["_tracking_embedding"] = None

        if observation_centric and captured_at - previous_seen > max(0.4, 1.0 / max(0.1, self.config.sample_fps)):
            elapsed = max(1e-3, captured_at - previous_seen)
            instantaneous = tuple(
                (coordinate - previous) / elapsed
                for coordinate, previous in zip(box, previous_box, strict=True)
            )
            track.velocity = instantaneous  # type: ignore[assignment]
            track.box = box
            track.last_seen = max(captured_at, track.last_seen)
            track.confidence = float(detection.get("confidence") or track.confidence)
            track.max_confidence = max(track.max_confidence, track.confidence)
            track.hits += 1
            track.missed = 0
            track.seeded = False
            track.zones.update(str(zone) for zone in detection.get("zones", []) if zone)
            center_x = (box[0] + box[2]) / 2.0
            center_y = (box[1] + box[3]) / 2.0
            track.trajectory.append((round(captured_at, 3), round(center_x, 1), round(center_y, 1)))
            track.box_history.append((
                round(captured_at, 3),
                round(box[0], 1),
                round(box[1], 1),
                round(box[2], 1),
                round(box[3], 1),
            ))
            if len(track.trajectory) > 60:
                del track.trajectory[:-60]
            if len(track.box_history) > 60:
                del track.box_history[:-60]
        else:
            track.observe(detection, captured_at, box)

        if saved_embedding is not None:
            detection["_tracking_embedding"] = saved_embedding

        if track.entity_id is None:
            track.entity_id = track.track_id

        next_appearance = _appearance(saved_embedding)
        if should_update_appearance and next_appearance is not None:
            self._update_gallery(track, next_appearance, quality)
        elif next_appearance is not None and frozen:
            self._association_counts["co_occlusion_freeze"] += 1

        if track.track_id in self._provisional_hits and self._provisional_hits[track.track_id] > 0:
            self._provisional_hits[track.track_id] -= 1
            if self._provisional_hits[track.track_id] <= 0:
                self._provisional_hits.pop(track.track_id, None)

        track.confirmed = track.confirmed or track.hits >= self.config.min_confirmations

    def _update_gallery(self, track, embedding: np.ndarray, quality: float) -> None:
        alpha = 0.98 - (0.08 * quality)
        alpha = min(0.98, max(0.85, alpha))
        if track.appearance is not None and track.appearance.shape == embedding.shape:
            blended = _appearance(track.appearance * alpha + embedding * (1.0 - alpha))
            if blended is not None:
                track.appearance = blended
        else:
            track.appearance = embedding
        gallery = self._galleries.setdefault(track.track_id, [])
        if not gallery or float(np.dot(gallery[-1], embedding)) < 0.97:
            gallery.append(embedding.copy())
        if len(gallery) > self.config.reid_gallery_size:
            del gallery[:-self.config.reid_gallery_size]

    def _appearance_similarity(self, track, embedding: np.ndarray) -> float | None:
        scores: list[float] = []
        if track.appearance is not None and track.appearance.shape == embedding.shape:
            scores.append(float(np.dot(track.appearance, embedding)))
        for item in self._galleries.get(track.track_id, []):
            if item.shape == embedding.shape:
                scores.append(float(np.dot(item, embedding)))
        if not scores:
            return None
        return max(scores)

    def _spatial_reid_ok(
        self,
        track,
        box: Box,
        captured_at: float,
        *,
        relaxed: bool = False,
    ) -> bool:
        age = max(0.0, captured_at - track.last_seen)
        predicted = track.predicted_box(captured_at)
        ratio = _center_distance_ratio(predicted, box)
        limit = self.config.reid_spatial_gate_ratio
        if relaxed:
            limit *= 1.0 + min(2.0, age / max(1.0, self.config.lost_timeout_seconds))
        else:
            limit *= 1.0 + min(1.0, age / max(1.0, self.config.lost_timeout_seconds))
        if ratio <= limit:
            return True
        # Buffered overlap is an alternate spatial gate.
        factor = self.config.sparse_buffer_iou_large if relaxed else self.config.sparse_buffer_iou_small
        return _iou(_expand_box(predicted, factor), box) >= 0.01

    def _mark_co_occlusions(self, captured_at: float) -> None:
        active = [track for track in self._tracks.values() if track.confirmed]
        for index, left in enumerate(active):
            for right in active[index + 1:]:
                if _normalize_label(left.label) != _normalize_label(right.label):
                    continue
                overlap = _iou(left.box, right.box)
                if overlap < 0.35:
                    continue
                until = captured_at + max(1.0 / max(0.1, self.config.sample_fps), 0.75)
                self._co_occluded_until[left.track_id] = until
                self._co_occluded_until[right.track_id] = until

    def _incoming_boxes_co_occlude(self, box: Box, captured_at: float, track_id: int) -> bool:
        for other_id, other in self._tracks.items():
            if other_id == track_id or not other.confirmed:
                continue
            if _normalize_label(other.label) != _normalize_label(self._tracks[track_id].label):
                continue
            if _iou(box, other.box) >= 0.35 or _iou(box, other.predicted_box(captured_at)) >= 0.35:
                return True
        return False

    def _prune_completed(self, captured_at: float) -> None:
        expired = [
            track_id
            for track_id, track in self._completed.items()
            if captured_at - track.last_seen > self.config.reid_max_age_seconds
        ]
        for track_id in expired:
            self._completed.pop(track_id, None)
            self._galleries.pop(track_id, None)
            self._provisional_hits.pop(track_id, None)
            self._co_occluded_until.pop(track_id, None)

    def summaries(self, captured_at: float) -> list[dict[str, Any]]:
        records = super().summaries(captured_at)
        for item in records:
            track_id = int(item["track_id"])
            track = self._tracks.get(track_id) or self._completed.get(track_id)
            if track is not None:
                item["entity_id"] = int(getattr(track, "entity_id", track_id))
            else:
                item["entity_id"] = self._entity_aliases.get(track_id, track_id)
        return records

    def appearance_records(self) -> list[dict[str, Any]]:
        records = super().appearance_records()
        for record in records:
            track_id = int(record["track_id"])
            track = self._tracks.get(track_id) or self._completed.get(track_id)
            record["entity_id"] = int(
                getattr(track, "entity_id", self._entity_aliases.get(track_id, track_id))
                if track is not None
                else self._entity_aliases.get(track_id, track_id)
            )
        return records

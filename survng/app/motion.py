from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from .motion_types import MotionBlob, MotionFrameBlobs, MotionTrack


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
    features: dict[str, Any]
    telemetry: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _BlobTrack:
    track_id: int
    first_frame: int
    last_frame: int
    hits: int
    centers: list[tuple[float, float]]
    areas: list[float]
    edge_hits: int
    box: tuple[float, float, float, float]


class BackgroundMotionTracker:
    """Stateful MOG2 foreground model with lightweight centroid blob tracks."""

    def __init__(self, sample_fps: float = 5.0, history_seconds: float = 30.0) -> None:
        self.sample_fps = max(1.0, float(sample_fps))
        self.history_seconds = max(5.0, float(history_seconds))
        self._warmup_frames = max(5, round(self.sample_fps * 2.0))
        self._max_gap_frames = max(2, round(self.sample_fps * 0.8))
        self._next_track_id = 1
        self._shape: tuple[int, int] | None = None
        self._frame_index = 0
        self._tracks: list[_BlobTrack] = []
        self._subtractor = self._new_subtractor()

    def _new_subtractor(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=max(30, round(self.sample_fps * self.history_seconds)),
            varThreshold=16,
            detectShadows=True,
        )

    def reset(self) -> None:
        self._shape = None
        self._frame_index = 0
        self._tracks.clear()
        self._next_track_id = 1
        self._subtractor = self._new_subtractor()

    @staticmethod
    def _iou(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> float:
        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if intersection <= 0:
            return 0.0
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        return intersection / max(1e-9, left_area + right_area - intersection)

    @staticmethod
    def _track_metrics(track: _BlobTrack, sample_fps: float) -> dict[str, float]:
        span_frames = max(1, track.last_frame - track.first_frame + 1)
        hit_ratio = track.hits / span_frames
        maturity = min(1.0, track.hits / max(3.0, sample_fps * 0.8))
        persistence = hit_ratio * maturity

        areas = np.asarray(track.areas[-20:], dtype=np.float32)
        if len(areas) >= 2:
            median_area = float(np.median(areas))
            deviation = float(np.median(np.abs(areas - median_area))) / max(median_area, 1e-9)
            area_stability = max(0.0, min(1.0, 1.0 - deviation * 2.5))
        else:
            area_stability = 0.0

        direction_coherence = 0.0
        centers = np.asarray(track.centers[-20:], dtype=np.float32)
        if len(centers) >= 3:
            vectors = np.diff(centers, axis=0)
            speeds = np.linalg.norm(vectors, axis=1)
            moving = speeds > 0.001
            if np.any(moving):
                directions = vectors[moving] / speeds[moving, None]
                direction_coherence = float(np.linalg.norm(np.sum(directions, axis=0)) / len(directions))

        edge_fraction = track.edge_hits / max(1, track.hits)
        interior = 1.0 - edge_fraction
        largest_area = max(track.areas) if track.areas else 0.0
        size_signal = min(1.0, largest_area / 0.01)
        score = (
            persistence * 0.40
            + area_stability * 0.20
            + direction_coherence * 0.20
            + interior * 0.10
            + size_signal * 0.10
        )
        return {
            "track_id": float(track.track_id),
            "score": max(0.0, min(1.0, score)),
            "track_persistence": max(0.0, min(1.0, persistence)),
            "track_age_seconds": span_frames / sample_fps,
            "track_hits": float(track.hits),
            "area_stability": area_stability,
            "direction_coherence": direction_coherence,
            "interior": interior,
            "largest_blob_ratio": largest_area,
        }

    @classmethod
    def _track_geometry(cls, track: _BlobTrack, sample_fps: float) -> dict[str, Any]:
        metrics = cls._track_metrics(track, sample_fps)
        return {
            "id": int(track.track_id),
            "score": round(metrics["score"], 4),
            "persistence": round(metrics["track_persistence"], 4),
            "box": [round(value, 4) for value in track.box],
            "path": [
                [round(x, 4), round(y, 4)]
                for x, y in track.centers[-30:]
            ],
        }

    def update(self, frame: np.ndarray) -> dict[str, Any]:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.shape[:2] != self._shape:
            self.reset()
            self._shape = frame.shape[:2]

        self._frame_index += 1
        foreground = self._subtractor.apply(frame)
        _, mask = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        frame_height, frame_width = frame.shape[:2]
        frame_area = max(1, frame_width * frame_height)
        foreground_ratio = float(cv2.countNonZero(mask)) / frame_area
        warmed = self._frame_index > self._warmup_frames
        if not warmed:
            return {
                "warmed": 0.0,
                "foreground_ratio": foreground_ratio,
                "blob_count": 0.0,
                "score": 0.0,
            }

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[dict[str, Any]] = []
        min_area = frame_area * 0.0004
        edge_x = frame_width * 0.06
        edge_y = frame_height * 0.06
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            box = (x / frame_width, y / frame_height, (x + width) / frame_width, (y + height) / frame_height)
            detections.append({
                "center": ((x + width / 2) / frame_width, (y + height / 2) / frame_height),
                "area": area / frame_area,
                "box": box,
                "edge": x <= edge_x or y <= edge_y or x + width >= frame_width - edge_x or y + height >= frame_height - edge_y,
            })

        available = [
            track for track in self._tracks
            if self._frame_index - track.last_frame <= self._max_gap_frames
        ]
        matched_ids: set[int] = set()
        for detection in sorted(detections, key=lambda item: item["area"], reverse=True):
            best_track: _BlobTrack | None = None
            best_cost = float("inf")
            for track in available:
                if track.track_id in matched_ids:
                    continue
                distance = float(np.linalg.norm(np.asarray(track.centers[-1]) - np.asarray(detection["center"])))
                overlap = self._iou(track.box, detection["box"])
                max_distance = max(0.08, min(0.24, np.sqrt(max(track.areas[-1], detection["area"])) * 2.5))
                if distance > max_distance and overlap <= 0:
                    continue
                cost = distance - overlap * 0.20
                if cost < best_cost:
                    best_cost = cost
                    best_track = track
            if best_track is None:
                best_track = _BlobTrack(
                    track_id=self._next_track_id,
                    first_frame=self._frame_index,
                    last_frame=self._frame_index,
                    hits=0,
                    centers=[],
                    areas=[],
                    edge_hits=0,
                    box=detection["box"],
                )
                self._next_track_id += 1
                available.append(best_track)
            best_track.last_frame = self._frame_index
            best_track.hits += 1
            best_track.centers.append(detection["center"])
            best_track.areas.append(detection["area"])
            best_track.edge_hits += int(detection["edge"])
            best_track.box = detection["box"]
            best_track.centers = best_track.centers[-30:]
            best_track.areas = best_track.areas[-30:]
            matched_ids.add(best_track.track_id)

        self._tracks = [
            track for track in available
            if self._frame_index - track.last_frame <= self._max_gap_frames
        ]
        active = [track for track in self._tracks if track.last_frame == self._frame_index]
        if not active:
            return {
                "warmed": 1.0,
                "foreground_ratio": foreground_ratio,
                "blob_count": 0.0,
                "score": 0.0,
            }
        best = max(active, key=lambda track: (track.hits, max(track.areas)))
        metrics = self._track_metrics(best, self.sample_fps)
        visible_tracks = sorted(
            active,
            key=lambda track: self._track_metrics(track, self.sample_fps)["score"],
            reverse=True,
        )[:6]
        return {
            "warmed": 1.0,
            "foreground_ratio": foreground_ratio,
            "blob_count": float(len(active)),
            **{name: round(value, 4) for name, value in metrics.items()},
            "tracks": [self._track_geometry(track, self.sample_fps) for track in visible_tracks],
        }


def aggregate_mog2_evidence(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"mog2_warmed": 0.0}
    warmed = [sample for sample in samples if sample.get("warmed", 0.0) >= 1.0]
    if not warmed:
        return {
            "mog2_warmed": 0.0,
            "mog2_foreground_ratio": round(max(sample.get("foreground_ratio", 0.0) for sample in samples), 4),
        }
    best = max(warmed, key=lambda sample: sample.get("score", 0.0))
    result: dict[str, Any] = {
        "mog2_warmed": 1.0,
        "mog2_score": round(best.get("score", 0.0), 4),
        "mog2_track_persistence": round(best.get("track_persistence", 0.0), 4),
        "mog2_track_age_seconds": round(max(sample.get("track_age_seconds", 0.0) for sample in warmed), 4),
        "mog2_track_hits": round(max(sample.get("track_hits", 0.0) for sample in warmed), 4),
        "mog2_area_stability": round(best.get("area_stability", 0.0), 4),
        "mog2_direction_coherence": round(best.get("direction_coherence", 0.0), 4),
        "mog2_interior": round(best.get("interior", 0.0), 4),
        "mog2_largest_blob_ratio": round(max(sample.get("largest_blob_ratio", 0.0) for sample in warmed), 4),
        "mog2_foreground_ratio": round(max(sample.get("foreground_ratio", 0.0) for sample in warmed), 4),
        "mog2_blob_count": round(max(sample.get("blob_count", 0.0) for sample in warmed), 4),
    }
    tracks = best.get("tracks")
    if isinstance(tracks, list) and tracks:
        result["mog2_tracks"] = tracks
    return result


def preprocess_motion_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(frame, (5, 5), 0)


def preprocess_motion_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [preprocess_motion_frame(frame) for frame in frames]


def difference_motion_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.absdiff(previous, current) for previous, current in zip(frames, frames[1:])]


def threshold_motion_differences(
    differences: list[np.ndarray],
    threshold_value: int = 18,
) -> list[np.ndarray]:
    return [
        cv2.threshold(difference, threshold_value, 255, cv2.THRESH_BINARY)[1]
        for difference in differences
    ]


def morphology_motion_masks(
    masks: list[np.ndarray],
    kernel_size: int = 3,
    close_iterations: int = 2,
) -> list[np.ndarray]:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    processed: list[np.ndarray] = []
    for mask in masks:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        processed.append(
            cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
        )
    return processed


def _legacy_score_motion_masks(
    prepared_frames: list[np.ndarray],
    masks: list[np.ndarray],
    sensitivity: str = "balanced",
) -> MotionQualificationResult:
    score_threshold = MOTION_SCORE_THRESHOLDS.get(
        sensitivity,
        MOTION_SCORE_THRESHOLDS["balanced"],
    )
    if len(prepared_frames) < 4:
        return MotionQualificationResult(
            accepted=True,
            score=1.0,
            threshold=score_threshold,
            reason="insufficient_frames",
            frame_count=len(prepared_frames),
            features={},
        )
    if len(masks) != len(prepared_frames) - 1:
        raise ValueError("motion mask count must be one less than prepared frame count")

    frame_height, frame_width = prepared_frames[0].shape[:2]
    frame_area = max(1, frame_width * frame_height)
    edge_margin_x = max(1, round(frame_width * 0.06))
    edge_margin_y = max(1, round(frame_height * 0.06))

    active: list[bool] = []
    areas: list[float] = []
    centroids: list[tuple[float, float]] = []
    edge_distances: list[float] = []
    interior: list[float] = []
    fragmentation: list[float] = []
    global_changes = 0

    for mask in masks:
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
    accepted = score >= score_threshold
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
        threshold=score_threshold,
        reason=reason,
        frame_count=len(prepared_frames),
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


def extract_motion_blobs(
    prepared_frames: list[np.ndarray],
    masks: list[np.ndarray],
) -> list[MotionFrameBlobs]:
    if len(masks) != max(0, len(prepared_frames) - 1):
        raise ValueError("motion mask count must be one less than prepared frame count")
    if not prepared_frames:
        return []
    frame_height, frame_width = prepared_frames[0].shape[:2]
    frame_area = max(1, frame_width * frame_height)
    edge_margin_x = max(1, round(frame_width * 0.06))
    edge_margin_y = max(1, round(frame_height * 0.06))
    history: list[MotionFrameBlobs] = []
    for mask in masks:
        changed_pixels = int(cv2.countNonZero(mask))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs: list[MotionBlob] = []
        for contour in contours:
            area_pixels = float(cv2.contourArea(contour))
            x, y, width, height = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments["m00"]:
                center_x = float(moments["m10"] / moments["m00"]) / frame_width
                center_y = float(moments["m01"] / moments["m00"]) / frame_height
            else:
                center_x = (x + width / 2) / frame_width
                center_y = (y + height / 2) / frame_height
            blobs.append(
                MotionBlob(
                    box=(
                        x / frame_width,
                        y / frame_height,
                        (x + width) / frame_width,
                        (y + height) / frame_height,
                    ),
                    centroid=(center_x, center_y),
                    area_pixels=area_pixels,
                    area_ratio=area_pixels / frame_area,
                    touches_edge=(
                        x <= edge_margin_x
                        or y <= edge_margin_y
                        or x + width >= frame_width - edge_margin_x
                        or y + height >= frame_height - edge_margin_y
                    ),
                )
            )
        history.append(
            MotionFrameBlobs(
                frame_area=frame_area,
                changed_pixels=changed_pixels,
                changed_ratio=changed_pixels / frame_area,
                blobs=tuple(blobs),
            )
        )
    return history


def filter_motion_blobs(
    history: list[MotionFrameBlobs],
    minimum_area_ratio: float = 0.0003,
) -> list[MotionFrameBlobs]:
    return [
        MotionFrameBlobs(
            frame_area=frame.frame_area,
            changed_pixels=frame.changed_pixels,
            changed_ratio=frame.changed_ratio,
            blobs=tuple(
                blob
                for blob in frame.blobs
                if blob.area_pixels >= frame.frame_area * minimum_area_ratio
            ),
        )
        for frame in history
    ]


def track_dominant_motion(
    history: list[MotionFrameBlobs],
    minimum_active_area_ratio: float = 0.0008,
    minimum_changed_ratio: float = 0.003,
) -> MotionTrack:
    active_history: list[bool] = []
    observations: list[MotionBlob] = []
    observation_frames: list[int] = []
    for frame_index, frame in enumerate(history):
        if not frame.blobs:
            active_history.append(False)
            continue
        largest = max(frame.blobs, key=lambda blob: blob.area_pixels)
        active = (
            largest.area_ratio >= minimum_active_area_ratio
            or frame.changed_ratio >= minimum_changed_ratio
        )
        active_history.append(active)
        if active:
            observations.append(largest)
            observation_frames.append(frame_index)
    return MotionTrack(
        track_id=1,
        box=observations[-1].box if observations else (0.0, 0.0, 0.0, 0.0),
        path=tuple(blob.centroid for blob in observations),
        observations=tuple(observations),
        observation_frames=tuple(observation_frames),
        active_history=tuple(active_history),
        changed_pixels=tuple(frame.changed_pixels for frame in history),
        changed_ratios=tuple(frame.changed_ratio for frame in history),
    )


def score_motion_track(
    track: MotionTrack,
    frame_count: int,
    sensitivity: str = "balanced",
) -> MotionQualificationResult:
    score_threshold = MOTION_SCORE_THRESHOLDS.get(
        sensitivity,
        MOTION_SCORE_THRESHOLDS["balanced"],
    )
    if frame_count < 4:
        return MotionQualificationResult(
            accepted=True,
            score=1.0,
            threshold=score_threshold,
            reason="insufficient_frames",
            frame_count=frame_count,
            features={},
        )

    transition_count = max(1, len(track.active_history))
    persistence = sum(track.active_history) / transition_count
    areas = np.asarray([blob.area_ratio for blob in track.observations])
    if len(areas) >= 2:
        median_area = float(np.median(areas))
        area_deviation = float(np.median(np.abs(areas - median_area))) / max(median_area, 1e-6)
        area_stability = max(0.0, min(1.0, 1.0 - area_deviation * 2.5))
    else:
        area_stability = 0.0

    continuity = 0.0
    centroids = np.asarray(track.path)
    if len(centroids) >= 2:
        vectors = np.diff(centroids, axis=0)
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

    interior = [0.0 if blob.touches_edge else 1.0 for blob in track.observations]
    edge_distances = [
        min(x, y, 1.0 - x, 1.0 - y)
        for x, y in track.path
    ]
    fragmentation = [
        min(1.0, blob.area_pixels / max(1, track.changed_pixels[frame_index]))
        for blob, frame_index in zip(track.observations, track.observation_frames)
    ]
    interior_score = float(np.mean(interior)) if interior else 0.0
    fragmentation_score = float(np.mean(fragmentation)) if fragmentation else 0.0
    inward_progress = 0.0
    if len(edge_distances) >= 2:
        inward_progress = max(0.0, min(1.0, (edge_distances[-1] - edge_distances[0]) / 0.08))
    coherent_entry = inward_progress * continuity * fragmentation_score
    coherent_edge_track = continuity * area_stability * fragmentation_score
    global_change_ratio = (
        sum(changed_ratio >= 0.45 for changed_ratio in track.changed_ratios) / transition_count
    )
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
    accepted = score >= score_threshold
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
        threshold=score_threshold,
        reason=reason,
        frame_count=frame_count,
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


def score_motion_masks(
    prepared_frames: list[np.ndarray],
    masks: list[np.ndarray],
    sensitivity: str = "balanced",
) -> MotionQualificationResult:
    raw_blobs = extract_motion_blobs(prepared_frames, masks)
    filtered_blobs = filter_motion_blobs(raw_blobs)
    track = track_dominant_motion(filtered_blobs)
    return score_motion_track(track, len(prepared_frames), sensitivity)


def qualify_motion(frames: list[np.ndarray], sensitivity: str = "balanced") -> MotionQualificationResult:
    prepared = preprocess_motion_frames(frames)
    differences = difference_motion_frames(prepared)
    thresholded = threshold_motion_differences(differences)
    masks = morphology_motion_masks(thresholded)
    return score_motion_masks(prepared, masks, sensitivity)

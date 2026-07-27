from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import cv2
import numpy as np

from ..motion import MOTION_SCORE_THRESHOLDS
from ..motion_types import MotionBlob, MotionFrameBlobs, MotionTrack
from .context import MotionContext, MotionScoring
from .registry import (
    MotionStageDependencies,
    MotionStageOption,
    MotionStageRegistration,
    MotionStageRegistry,
)


@dataclass(slots=True)
class _BackgroundRuntime:
    background: np.ndarray | None = None
    noise_ema: float = 4.0
    brightness_ema: float = 128.0
    last_processed_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> "_BackgroundRuntime":
        with self.lock:
            return _BackgroundRuntime(
                background=None if self.background is None else self.background.copy(),
                noise_ema=self.noise_ema,
                brightness_ema=self.brightness_ema,
                last_processed_at=self.last_processed_at,
            )


@dataclass(slots=True)
class _ThresholdRuntime:
    threshold_ema: float = 12.0
    noise_ema: float = 4.0
    last_processed_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> "_ThresholdRuntime":
        with self.lock:
            return _ThresholdRuntime(
                threshold_ema=self.threshold_ema,
                noise_ema=self.noise_ema,
                last_processed_at=self.last_processed_at,
            )


@dataclass(slots=True)
class _TrackedBlob:
    track_id: int
    first_seen: float
    last_seen: float
    consecutive_started_at: float
    consecutive_frames: int
    centroids: list[tuple[float, float]]
    blobs: list[MotionBlob]
    sizes: list[float]
    velocity: tuple[float, float] = (0.0, 0.0)
    accumulated_score: float = 0.0
    missed_frames: int = 0


@dataclass(slots=True)
class _TrackerRuntime:
    next_track_id: int = 1
    tracks: dict[int, _TrackedBlob] = field(default_factory=dict)
    last_processed_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> "_TrackerRuntime":
        with self.lock:
            return _TrackerRuntime(
                next_track_id=self.next_track_id,
                tracks={key: _TrackedBlob(
                    track_id=value.track_id,
                    first_seen=value.first_seen,
                    last_seen=value.last_seen,
                    consecutive_started_at=value.consecutive_started_at,
                    consecutive_frames=value.consecutive_frames,
                    centroids=list(value.centroids),
                    blobs=list(value.blobs),
                    sizes=list(value.sizes),
                    velocity=value.velocity,
                    accumulated_score=value.accumulated_score,
                    missed_frames=value.missed_frames,
                ) for key, value in self.tracks.items()},
                last_processed_at=self.last_processed_at,
            )


@dataclass(slots=True)
class _ScoringRuntime:
    accumulated_scores: dict[int, float] = field(default_factory=dict)
    last_seen: dict[int, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> "_ScoringRuntime":
        with self.lock:
            return _ScoringRuntime(
                accumulated_scores=dict(self.accumulated_scores),
                last_seen=dict(self.last_seen),
            )


class AdaptiveEmaBackgroundStage:
    """Selective EMA background model with scene-aware learning rates."""

    def __init__(
        self,
        stage_id: str,
        *,
        learning_rate: float = 0.025,
        fast_learning_rate: float = 0.18,
        motion_learning_scale: float = 0.03,
        global_change_ratio: float = 0.55,
    ) -> None:
        self._stage_id = stage_id
        self.learning_rate = min(1.0, max(0.0001, float(learning_rate)))
        self.fast_learning_rate = min(1.0, max(self.learning_rate, float(fast_learning_rate)))
        self.motion_learning_scale = min(1.0, max(0.0, float(motion_learning_scale)))
        self.global_change_ratio = min(1.0, max(0.05, float(global_change_ratio)))

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        frames = context.processed_frame_history
        if not frames:
            return context
        state = context.runtime.state_for(self.stage_id, _BackgroundRuntime)
        differences: list[np.ndarray] = []
        learning_rates: list[float] = []
        global_changes: list[float] = []
        with state.lock:
            first = frames[0].astype(np.float32, copy=False)
            timestamps = context.frame_timestamps
            if (
                state.background is None
                or state.background.shape != first.shape
            ):
                background = first.copy()
            else:
                background = state.background.copy()
            if state.background is None or state.background.shape != first.shape:
                state.noise_ema = 4.0
                state.brightness_ema = float(np.mean(first))
            noise_ema = state.noise_ema
            brightness_ema = state.brightness_ema
            sample_fps = max(1.0, float(context.configuration.get("sample_fps", 5.0)))
            for frame_index, frame in enumerate(frames[1:], start=1):
                current = frame.astype(np.float32, copy=False)
                delta = cv2.absdiff(current, background)
                median = float(np.median(delta))
                mad = float(np.median(np.abs(delta - median)))
                robust_noise = max(1.0, median + 1.4826 * mad)
                noise_ema = noise_ema * 0.92 + robust_noise * 0.08
                stable_limit = max(6.0, noise_ema * 3.0)
                changed = delta > stable_limit
                changed_ratio = float(np.count_nonzero(changed)) / max(1, changed.size)

                elapsed = 1.0 / sample_fps
                if frame_index < len(timestamps):
                    previous_at = timestamps[frame_index - 1]
                    current_at = timestamps[frame_index]
                    if current_at > previous_at:
                        elapsed = current_at - previous_at
                elapsed_frames = min(sample_fps * 10.0, max(0.05, elapsed * sample_fps))

                if changed_ratio >= self.global_change_ratio:
                    rate = 1.0 - (1.0 - self.fast_learning_rate) ** elapsed_frames
                    cv2.accumulateWeighted(current, background, rate)
                else:
                    noise_boost = min(3.0, max(1.0, noise_ema / 6.0))
                    base_rate = min(self.fast_learning_rate, self.learning_rate * noise_boost)
                    rate = 1.0 - (1.0 - base_rate) ** elapsed_frames
                    stable = np.logical_not(changed).astype(np.uint8)
                    cv2.accumulateWeighted(current, background, rate, mask=stable)
                    if self.motion_learning_scale > 0:
                        moving = changed.astype(np.uint8)
                        moving_rate = 1.0 - (
                            1.0 - base_rate * self.motion_learning_scale
                        ) ** elapsed_frames
                        cv2.accumulateWeighted(
                            current,
                            background,
                            moving_rate,
                            mask=moving,
                        )

                differences.append(np.clip(delta, 0, 255).astype(np.uint8))
                learning_rates.append(rate)
                global_changes.append(changed_ratio)
                brightness = float(np.mean(current))
                brightness_ema = brightness_ema * 0.96 + brightness * 0.04

            if not timestamps or state.last_processed_at is None or timestamps[-1] > state.last_processed_at:
                state.background = background
                state.noise_ema = noise_ema
                state.brightness_ema = brightness_ema
                state.last_processed_at = timestamps[-1] if timestamps else context.captured_at
            context.background_image = np.clip(background, 0, 255).astype(np.uint8)
            scene_noise = noise_ema
            brightness = brightness_ema

        context.difference_history = tuple(differences)
        context.difference_image = differences[-1] if differences else None
        context.debug.values.update({
            "background_learning_rates": [round(value, 5) for value in learning_rates],
            "scene_noise": round(scene_noise, 4),
            "scene_brightness": round(brightness, 2),
            "scene_mode": "night" if brightness < 55 else "day",
            "global_change_ratios": [round(value, 4) for value in global_changes],
            "global_change_threshold": self.global_change_ratio,
        })
        return context


class AdaptiveStatisticalThresholdStage:
    """Derive a smoothed threshold from robust per-scene difference statistics."""

    def __init__(
        self,
        stage_id: str,
        *,
        sigma: float = 4.0,
        minimum: float = 7.0,
        maximum: float = 72.0,
        smoothing: float = 0.25,
    ) -> None:
        self._stage_id = stage_id
        self.sigma = max(0.5, float(sigma))
        self.minimum = max(0.0, float(minimum))
        self.maximum = max(self.minimum, float(maximum))
        self.smoothing = min(1.0, max(0.01, float(smoothing)))

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        state = context.runtime.state_for(self.stage_id, _ThresholdRuntime)
        masks: list[np.ndarray] = []
        thresholds: list[float] = []
        noises: list[float] = []
        difference_timestamps = context.frame_timestamps[1:]
        with state.lock:
            for index, difference in enumerate(context.difference_history):
                flat = difference.reshape(-1).astype(np.float32, copy=False)
                median = float(np.median(flat))
                mad = float(np.median(np.abs(flat - median)))
                noise = max(0.5, 1.4826 * mad)
                percentile = float(np.percentile(flat, 80))
                candidate = max(self.minimum, median + self.sigma * noise, percentile * 0.8)
                candidate = min(self.maximum, candidate)
                timestamp = (
                    difference_timestamps[index]
                    if index < len(difference_timestamps)
                    else None
                )
                is_new = (
                    timestamp is None
                    or state.last_processed_at is None
                    or timestamp > state.last_processed_at
                )
                if is_new:
                    state.threshold_ema += self.smoothing * (candidate - state.threshold_ema)
                    state.noise_ema = state.noise_ema * 0.9 + noise * 0.1
                    if timestamp is not None:
                        state.last_processed_at = timestamp
                threshold = min(self.maximum, max(self.minimum, state.threshold_ema))
                masks.append(cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)[1])
                thresholds.append(threshold)
                noises.append(noise)

        context.threshold_mask_history = tuple(masks)
        context.binary_motion_mask = masks[-1] if masks else None
        context.debug.values["adaptive_thresholds"] = [round(value, 3) for value in thresholds]
        context.debug.values["threshold_noise"] = round(state.noise_ema, 4)
        return context


def _motion_zones(
    configuration: Mapping[str, Any], width: int, height: int
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, np.ndarray], ...]]:
    included = np.zeros((height, width), dtype=np.uint8)
    ignored = np.zeros((height, width), dtype=np.uint8)
    named_included: list[tuple[str, np.ndarray]] = []
    raw_zones = configuration.get("motion_zones", [])
    if not isinstance(raw_zones, list):
        return included, ignored, ()
    for raw in raw_zones:
        if not isinstance(raw, Mapping) or not raw.get("enabled", True):
            continue
        points = raw.get("points", [])
        if not isinstance(points, list) or len(points) < 3:
            continue
        polygon = np.asarray(
            [
                [
                    round(float(point.get("x", 0.0)) * (width - 1)),
                    round(float(point.get("y", 0.0)) * (height - 1)),
                ]
                for point in points
                if isinstance(point, Mapping)
            ],
            dtype=np.int32,
        )
        if len(polygon) < 3:
            continue
        is_ignored = raw.get("behavior") == "ignore"
        target = ignored if is_ignored else included
        cv2.fillPoly(target, [polygon], 255)
        if not is_ignored:
            zone_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(zone_mask, [polygon], 255)
            named_included.append((str(raw.get("name") or "zone"), zone_mask))
    return included, ignored, tuple(named_included)


class ConnectedComponentBlobStage:
    def __init__(self, stage_id: str, edge_margin_ratio: float = 0.06) -> None:
        self._stage_id = stage_id
        self.edge_margin_ratio = min(0.25, max(0.0, float(edge_margin_ratio)))

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        if not context.processed_frame_history:
            context.raw_blob_history = ()
            return context
        height, width = context.processed_frame_history[0].shape[:2]
        frame_area = max(1, width * height)
        included_zone, ignored_zone, named_zones = _motion_zones(context.configuration, width, height)
        history: list[MotionFrameBlobs] = []
        for index, mask in enumerate(context.motion_mask_history):
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            intensity = (
                context.difference_history[index]
                if index < len(context.difference_history)
                else mask
            )
            blobs: list[MotionBlob] = []
            for label in range(1, count):
                x, y, box_width, box_height, pixels = stats[label]
                if pixels <= 0 or box_width <= 0 or box_height <= 0:
                    continue
                component = labels[y : y + box_height, x : x + box_width] == label
                component_pixels = int(np.count_nonzero(component))
                if component_pixels <= 0:
                    continue
                cx, cy = centroids[label]
                box_area = max(1, int(box_width) * int(box_height))
                included_overlap = float(
                    np.count_nonzero(component & (included_zone[y:y + box_height, x:x + box_width] > 0))
                ) / component_pixels
                ignored_overlap = float(
                    np.count_nonzero(component & (ignored_zone[y:y + box_height, x:x + box_width] > 0))
                ) / component_pixels
                edge_distance = min(cx / width, cy / height, (width - cx) / width, (height - cy) / height)
                overlapping_zone_names = tuple(
                    name
                    for name, zone_mask in named_zones
                    if np.count_nonzero(
                        component & (zone_mask[y:y + box_height, x:x + box_width] > 0)
                    ) > 0
                )
                blobs.append(MotionBlob(
                    box=(x / width, y / height, (x + box_width) / width, (y + box_height) / height),
                    centroid=(float(cx) / width, float(cy) / height),
                    area_pixels=float(component_pixels),
                    area_ratio=float(component_pixels) / frame_area,
                    touches_edge=edge_distance <= self.edge_margin_ratio,
                    fill_ratio=float(component_pixels) / box_area,
                    aspect_ratio=float(box_width) / max(1.0, float(box_height)),
                    average_motion_intensity=float(np.mean(intensity[y:y + box_height, x:x + box_width][component])),
                    edge_distance=float(edge_distance),
                    zone_overlap=included_overlap,
                    ignored_zone_overlap=ignored_overlap,
                    zone_names=overlapping_zone_names,
                ))
            changed = int(cv2.countNonZero(mask))
            history.append(MotionFrameBlobs(
                frame_area=frame_area,
                changed_pixels=changed,
                changed_ratio=changed / frame_area,
                blobs=tuple(blobs),
            ))
        context.raw_blob_history = tuple(history)
        return context


class AdaptiveBlobFilterStage:
    def __init__(
        self,
        stage_id: str,
        *,
        minimum_area_ratio: float = 0.00025,
        minimum_fill_ratio: float = 0.12,
        maximum_aspect_ratio: float = 12.0,
        ignored_zone_overlap: float = 0.6,
    ) -> None:
        self._stage_id = stage_id
        self.minimum_area_ratio = max(0.0, float(minimum_area_ratio))
        self.minimum_fill_ratio = min(1.0, max(0.0, float(minimum_fill_ratio)))
        self.maximum_aspect_ratio = max(1.0, float(maximum_aspect_ratio))
        self.ignored_zone_overlap = min(1.0, max(0.0, float(ignored_zone_overlap)))

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        noise = float(context.debug.values.get("threshold_noise", 1.0))
        noise_scale = min(3.0, max(1.0, noise / 4.0))
        adaptive_minimum = self.minimum_area_ratio * noise_scale
        filtered: list[MotionFrameBlobs] = []
        rejected = {"area": 0, "fill": 0, "shape": 0, "edge": 0, "zone": 0}
        for frame in context.raw_blob_history:
            kept: list[MotionBlob] = []
            for blob in frame.blobs:
                aspect = max(blob.aspect_ratio, 1.0 / max(blob.aspect_ratio, 1e-6))
                if blob.area_ratio < adaptive_minimum:
                    rejected["area"] += 1
                elif blob.fill_ratio < self.minimum_fill_ratio:
                    rejected["fill"] += 1
                elif aspect > self.maximum_aspect_ratio:
                    rejected["shape"] += 1
                elif blob.touches_edge and blob.area_ratio < adaptive_minimum * 2.0:
                    rejected["edge"] += 1
                elif blob.ignored_zone_overlap >= self.ignored_zone_overlap:
                    rejected["zone"] += 1
                else:
                    kept.append(blob)
            filtered.append(replace(frame, blobs=tuple(kept)))
        context.filtered_blob_history = tuple(filtered)
        context.blobs = [blob for frame in filtered for blob in frame.blobs]
        context.debug.values["adaptive_minimum_blob_ratio"] = round(adaptive_minimum, 7)
        context.debug.values["blob_rejections"] = rejected
        return context


def _box_iou(left: MotionBlob, right: MotionBlob) -> float:
    x1 = max(left.box[0], right.box[0])
    y1 = max(left.box[1], right.box[1])
    x2 = min(left.box[2], right.box[2])
    y2 = min(left.box[3], right.box[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left.box[2] - left.box[0]) * max(0.0, left.box[3] - left.box[1])
    right_area = max(0.0, right.box[2] - right.box[0]) * max(0.0, right.box[3] - right.box[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


class PersistentCentroidTrackerStage:
    def __init__(
        self,
        stage_id: str,
        *,
        maximum_distance: float = 0.18,
        maximum_missed_frames: int = 3,
        maximum_track_age_seconds: float = 8.0,
    ) -> None:
        self._stage_id = stage_id
        self.maximum_distance = min(1.0, max(0.01, float(maximum_distance)))
        self.maximum_missed_frames = max(0, int(maximum_missed_frames))
        self.maximum_track_age_seconds = max(0.5, float(maximum_track_age_seconds))

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        runtime = context.runtime.state_for(self.stage_id, _TrackerRuntime)
        sample_fps = max(1.0, float(context.configuration.get("sample_fps", 5.0)))
        frame_count = len(context.filtered_blob_history)
        timestamps = context.frame_timestamps
        visible_ids: set[int] = set()
        with runtime.lock:
            if (
                timestamps
                and runtime.last_processed_at is not None
                and timestamps[-1] <= runtime.last_processed_at
            ):
                runtime.tracks.clear()
            runtime.tracks = {
                key: value
                for key, value in runtime.tracks.items()
                if context.captured_at - value.last_seen <= self.maximum_track_age_seconds
            }
            for frame_index, frame in enumerate(context.filtered_blob_history):
                timestamp = (
                    timestamps[min(frame_index + 1, len(timestamps) - 1)]
                    if timestamps
                    else context.captured_at - max(0, frame_count - frame_index - 1) / sample_fps
                )
                unmatched = set(runtime.tracks)
                for blob in sorted(frame.blobs, key=lambda item: item.area_ratio, reverse=True):
                    best_id: int | None = None
                    best_cost = float("inf")
                    for track_id in unmatched:
                        track = runtime.tracks[track_id]
                        previous = track.blobs[-1]
                        distance = math.dist(previous.centroid, blob.centroid)
                        overlap = _box_iou(previous, blob)
                        dynamic_limit = min(
                            self.maximum_distance,
                            max(
                                0.015,
                                math.sqrt(max(previous.area_ratio, blob.area_ratio)) * 2.5,
                            ),
                        )
                        if distance > dynamic_limit and overlap <= 0:
                            continue
                        cost = distance - overlap * 0.2
                        if cost < best_cost:
                            best_cost = cost
                            best_id = track_id
                    if best_id is None:
                        best_id = runtime.next_track_id
                        runtime.next_track_id += 1
                        runtime.tracks[best_id] = _TrackedBlob(
                            track_id=best_id,
                            first_seen=timestamp,
                            last_seen=timestamp,
                            consecutive_started_at=timestamp,
                            consecutive_frames=0,
                            centroids=[],
                            blobs=[],
                            sizes=[],
                        )
                    track = runtime.tracks[best_id]
                    if track.missed_frames:
                        track.consecutive_frames = 0
                        track.consecutive_started_at = timestamp
                    if track.centroids:
                        elapsed = max(1.0 / sample_fps, timestamp - track.last_seen)
                        track.velocity = (
                            (blob.centroid[0] - track.centroids[-1][0]) / elapsed,
                            (blob.centroid[1] - track.centroids[-1][1]) / elapsed,
                        )
                    track.last_seen = max(track.last_seen, timestamp)
                    track.consecutive_frames += 1
                    track.missed_frames = 0
                    track.centroids.append(blob.centroid)
                    track.blobs.append(blob)
                    track.sizes.append(blob.area_ratio)
                    track.centroids = track.centroids[-40:]
                    track.blobs = track.blobs[-40:]
                    track.sizes = track.sizes[-40:]
                    visible_ids.add(best_id)
                    unmatched.discard(best_id)
                for track_id in unmatched:
                    runtime.tracks[track_id].missed_frames += 1
                expired = {
                    track_id
                    for track_id, track in runtime.tracks.items()
                    if track.missed_frames > self.maximum_missed_frames
                }
                for track_id in expired:
                    runtime.tracks.pop(track_id, None)
                    visible_ids.discard(track_id)
            tracks = [
                self._publish(track, frame_count)
                for track_id, track in runtime.tracks.items()
                if track_id in visible_ids
            ]
            runtime.last_processed_at = timestamps[-1] if timestamps else context.captured_at

        tracks.sort(key=lambda item: (item.consecutive_frames, max(item.size_history, default=0.0)), reverse=True)
        context.tracked_objects = tracks
        context.dominant_track = tracks[0] if tracks else self._empty_track(frame_count)
        context.debug.values["active_track_count"] = len(tracks)
        return context

    @staticmethod
    def _publish(track: _TrackedBlob, frame_count: int) -> MotionTrack:
        direction = math.atan2(track.velocity[1], track.velocity[0]) if track.velocity != (0.0, 0.0) else 0.0
        observations = tuple(track.blobs)
        active_count = min(frame_count, len(observations))
        return MotionTrack(
            track_id=track.track_id,
            box=observations[-1].box,
            path=tuple(track.centroids),
            observations=observations,
            observation_frames=tuple(range(len(observations))),
            active_history=tuple([True] * active_count + [False] * max(0, frame_count - active_count)),
            changed_pixels=(),
            changed_ratios=(),
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            consecutive_started_at=track.consecutive_started_at,
            consecutive_frames=track.consecutive_frames,
            velocity=track.velocity,
            direction=direction,
            size_history=tuple(track.sizes),
            accumulated_score=track.accumulated_score,
        )

    @staticmethod
    def _empty_track(frame_count: int) -> MotionTrack:
        return MotionTrack(
            track_id=0,
            box=(0.0, 0.0, 0.0, 0.0),
            path=(),
            observations=(),
            observation_frames=(),
            active_history=tuple(False for _ in range(frame_count)),
            changed_pixels=(),
            changed_ratios=(),
        )


class AdaptiveMotionScoringStage:
    def __init__(
        self,
        stage_id: str,
        minimum_persistence_frames: int = 2,
        *,
        stationary_displacement_ratio: float = 0.01,
        stationary_path_ratio: float = 0.025,
        micro_area_ratio: float = 0.001,
        micro_displacement_ratio: float = 0.04,
        credible_displacement_ratio: float = 0.03,
        maximum_credible_track_seconds: float = 6.0,
    ) -> None:
        self._stage_id = stage_id
        self.minimum_persistence_frames = max(1, int(minimum_persistence_frames))
        self.stationary_displacement_ratio = max(0.0, float(stationary_displacement_ratio))
        self.stationary_path_ratio = max(0.0, float(stationary_path_ratio))
        self.micro_area_ratio = max(0.0, float(micro_area_ratio))
        self.micro_displacement_ratio = max(0.0, float(micro_displacement_ratio))
        self.credible_displacement_ratio = max(
            0.0001,
            float(credible_displacement_ratio),
        )
        self.maximum_credible_track_seconds = max(
            0.5,
            float(maximum_credible_track_seconds),
        )

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        sensitivity = str(context.configuration.get("sensitivity") or "balanced")
        base_threshold = MOTION_SCORE_THRESHOLDS.get(sensitivity, MOTION_SCORE_THRESHOLDS["balanced"])
        scene_noise = float(context.debug.values.get("scene_noise", 4.0))
        night = context.debug.values.get("scene_mode") == "night"
        noise_adjustment = min(0.12, max(0.0, (scene_noise - 6.0) / 80.0))
        threshold = min(0.78, base_threshold + noise_adjustment + (0.025 if night else 0.0))
        scores: list[tuple[float, MotionTrack, dict[str, float]]] = []
        sample_fps = max(1.0, float(context.configuration.get("sample_fps", 5.0)))
        for track in context.tracked_objects:
            features = self._track_features(
                track,
                sample_fps,
                self.credible_displacement_ratio,
            )
            score = (
                features["persistence"] * 0.20
                + features["translation"] * 0.10
                + features["size"] * 0.15
                + features["direction_consistency"] * 0.17
                + features["speed_consistency"] * 0.12
                + features["fill"] * 0.10
                + features["interior"] * 0.08
                + features["zone_weight"] * 0.08
                - features["insect_penalty"] * 0.28
            )
            scores.append((min(1.0, max(0.0, score)), track, features))
        score_state = context.runtime.state_for(self.stage_id, _ScoringRuntime)
        if scores:
            minimum_persistence_seconds = self.minimum_persistence_frames / sample_fps
            evaluations: list[
                tuple[float, MotionTrack, dict[str, float], bool, str]
            ] = []
            for candidate_score, candidate_track, candidate_features in scores:
                persistence_ok = (
                    candidate_features["persistence_seconds"]
                    >= minimum_persistence_seconds
                )
                insect_like = (
                    candidate_features["insect_penalty"] >= 0.55
                    and candidate_features["size"] < 0.25
                )
                stationary_foreground = (
                    candidate_features["persistence"] >= 0.4
                    and candidate_features["path_length"] < self.stationary_path_ratio
                    and candidate_features["net_displacement"]
                    < self.stationary_displacement_ratio
                )
                micro_jitter = (
                    candidate_features["median_area_ratio"] < self.micro_area_ratio
                    and candidate_features["net_displacement"]
                    < self.micro_displacement_ratio
                )
                persistent_scene_motion = (
                    candidate_features["persistence_seconds"]
                    > self.maximum_credible_track_seconds
                )
                candidate_accepted = (
                    candidate_score >= threshold
                    and persistence_ok
                    and not insect_like
                    and not stationary_foreground
                    and not micro_jitter
                    and not persistent_scene_motion
                )
                candidate_reason = "qualified" if candidate_accepted else (
                    "low_persistence" if not persistence_ok else
                    "persistent_scene_motion" if persistent_scene_motion else
                    "insect_like_motion" if insect_like else
                    "stationary_foreground" if stationary_foreground else
                    "micro_jitter" if micro_jitter else
                    "low_score"
                )
                evaluations.append((
                    candidate_score,
                    candidate_track,
                    candidate_features,
                    candidate_accepted,
                    candidate_reason,
                ))
            accepted_evaluations = [item for item in evaluations if item[3]]
            score, best, features, accepted, reason = max(
                accepted_evaluations or evaluations,
                key=lambda item: item[0],
            )
            with score_state.lock:
                accumulated = score_state.accumulated_scores.get(best.track_id, 0.0) + score
                score_state.accumulated_scores[best.track_id] = accumulated
                score_state.last_seen[best.track_id] = context.captured_at
                expired = [
                    track_id for track_id, last_seen in score_state.last_seen.items()
                    if context.captured_at - last_seen > 30.0
                ]
                for track_id in expired:
                    score_state.last_seen.pop(track_id, None)
                    score_state.accumulated_scores.pop(track_id, None)
            scored = replace(best, score=round(score, 4), accumulated_score=accumulated)
            context.dominant_track = scored
            context.tracked_objects = [scored if item.track_id == scored.track_id else item for item in context.tracked_objects]
        else:
            score = 0.0
            accepted = False
            reason = "no_motion_blobs"
            features = {
                "persistence": 0.0,
                "size": 0.0,
                "direction_consistency": 0.0,
                "speed_consistency": 0.0,
                "fill": 0.0,
                "interior": 0.0,
                "zone_weight": 0.0,
                "insect_penalty": 0.0,
                "persistence_seconds": 0.0,
                "median_area_ratio": 0.0,
                "net_displacement": 0.0,
                "path_length": 0.0,
                "translation": 0.0,
            }
        global_changes = context.debug.values.get("global_change_ratios", [])
        global_change = max(global_changes, default=0.0) if isinstance(global_changes, list) else 0.0
        global_change_threshold = min(
            1.0,
            max(
                0.05,
                float(context.debug.values.get("global_change_threshold", 0.55)),
            ),
        )
        if global_change >= global_change_threshold:
            accepted = False
            score = min(score, threshold * 0.5)
            reason = "global_illumination_change"
        scoring_features: dict[str, Any] = {
            **{key: round(value, 4) for key, value in features.items()},
            "scene_noise": round(scene_noise, 4),
            "scene_mode": "night" if night else "day",
            "adaptive_threshold": round(threshold, 4),
            "global_change": round(global_change, 4),
            "global_change_threshold": round(global_change_threshold, 4),
            "track_count": len(context.tracked_objects),
        }
        context.scoring = MotionScoring(
            accepted=accepted,
            score=round(score, 4),
            threshold=round(threshold, 4),
            reason=reason,
            frame_count=len(context.processed_frame_history),
            features=scoring_features,
        )
        return context

    @staticmethod
    def _track_features(
        track: MotionTrack,
        sample_fps: float,
        credible_displacement_ratio: float,
    ) -> dict[str, float]:
        observations = track.observations[-20:]
        consecutive_started_at = track.consecutive_started_at or max(
            track.first_seen,
            track.last_seen - max(0, track.consecutive_frames - 1) / sample_fps,
        )
        persistence_seconds = max(
            track.last_seen - consecutive_started_at + 1.0 / sample_fps,
            track.consecutive_frames / sample_fps,
        )
        persistence = min(1.0, persistence_seconds / 1.0)
        median_size = float(np.median(track.size_history[-20:])) if track.size_history else 0.0
        size = min(1.0, median_size / 0.012)
        fill = float(np.mean([blob.fill_ratio for blob in observations])) if observations else 0.0
        interior = float(np.mean([not blob.touches_edge for blob in observations])) if observations else 0.0
        zone_overlap = max((blob.zone_overlap for blob in observations), default=0.0)
        zone_weight = 0.65 + min(0.35, zone_overlap * 0.35)
        path = np.asarray(track.path[-20:], dtype=np.float32)
        vectors = np.diff(path, axis=0)
        path_length = float(np.sum(np.linalg.norm(vectors, axis=1))) if len(vectors) else 0.0
        net_displacement = (
            float(np.linalg.norm(path[-1] - path[0]))
            if len(path) > 1
            else 0.0
        )
        translation = min(1.0, net_displacement / credible_displacement_ratio)
        median_speed = 0.0
        if len(vectors):
            speeds = np.linalg.norm(vectors, axis=1)
            moving = speeds > 0.001
            if np.any(moving):
                unit = vectors[moving] / speeds[moving, None]
                direction_consistency = float(np.linalg.norm(np.sum(unit, axis=0)) / len(unit))
                median_speed = float(np.median(speeds[moving]))
                deviation = float(np.median(np.abs(speeds[moving] - median_speed))) / max(median_speed, 1e-6)
                speed_consistency = max(0.0, 1.0 - deviation * 2.0)
            else:
                direction_consistency = 0.25
                speed_consistency = 0.5
        else:
            direction_consistency = 0.0
            speed_consistency = 0.0
            median_speed = 0.0
        edge_fraction = 1.0 - interior
        tiny = max(0.0, 1.0 - median_size / 0.003)
        erratic = 1.0 - direction_consistency
        fast = min(1.0, median_speed / 0.12)
        insect_penalty = min(1.0, tiny * 0.45 + erratic * 0.30 + edge_fraction * 0.15 + fast * 0.10)
        return {
            "persistence": persistence,
            "size": size,
            "direction_consistency": direction_consistency,
            "speed_consistency": speed_consistency,
            "fill": fill,
            "interior": interior,
            "zone_weight": zone_weight,
            "insect_penalty": insect_penalty,
            "persistence_seconds": persistence_seconds,
            "median_area_ratio": median_size,
            "net_displacement": net_displacement,
            "path_length": path_length,
            "translation": translation,
        }


def _build_background(stage_id: str, options: Mapping[str, Any], dependencies: MotionStageDependencies) -> AdaptiveEmaBackgroundStage:
    del dependencies
    return AdaptiveEmaBackgroundStage(
        stage_id,
        learning_rate=float(options.get("learning_rate", 0.025)),
        fast_learning_rate=float(options.get("fast_learning_rate", 0.18)),
        motion_learning_scale=float(options.get("motion_learning_scale", 0.03)),
        global_change_ratio=float(options.get("global_change_ratio", 0.55)),
    )


def _build_threshold(stage_id: str, options: Mapping[str, Any], dependencies: MotionStageDependencies) -> AdaptiveStatisticalThresholdStage:
    del dependencies
    return AdaptiveStatisticalThresholdStage(
        stage_id,
        sigma=float(options.get("sigma", 4.0)),
        minimum=float(options.get("minimum", 7.0)),
        maximum=float(options.get("maximum", 72.0)),
        smoothing=float(options.get("smoothing", 0.25)),
    )


def _build_components(stage_id: str, options: Mapping[str, Any], dependencies: MotionStageDependencies) -> ConnectedComponentBlobStage:
    del dependencies
    return ConnectedComponentBlobStage(stage_id, float(options.get("edge_margin_ratio", 0.06)))


def _build_filter(stage_id: str, options: Mapping[str, Any], dependencies: MotionStageDependencies) -> AdaptiveBlobFilterStage:
    del dependencies
    return AdaptiveBlobFilterStage(
        stage_id,
        minimum_area_ratio=float(options.get("minimum_area_ratio", 0.00025)),
        minimum_fill_ratio=float(options.get("minimum_fill_ratio", 0.12)),
        maximum_aspect_ratio=float(options.get("maximum_aspect_ratio", 12.0)),
        ignored_zone_overlap=float(options.get("ignored_zone_overlap", 0.6)),
    )


def _build_tracker(stage_id: str, options: Mapping[str, Any], dependencies: MotionStageDependencies) -> PersistentCentroidTrackerStage:
    del dependencies
    return PersistentCentroidTrackerStage(
        stage_id,
        maximum_distance=float(options.get("maximum_distance", 0.18)),
        maximum_missed_frames=int(options.get("maximum_missed_frames", 3)),
        maximum_track_age_seconds=float(options.get("maximum_track_age_seconds", 8.0)),
    )


def _build_scorer(stage_id: str, options: Mapping[str, Any], dependencies: MotionStageDependencies) -> AdaptiveMotionScoringStage:
    del dependencies
    return AdaptiveMotionScoringStage(
        stage_id,
        int(options.get("minimum_persistence_frames", 2)),
        stationary_displacement_ratio=float(
            options.get("stationary_displacement_ratio", 0.01)
        ),
        stationary_path_ratio=float(options.get("stationary_path_ratio", 0.025)),
        micro_area_ratio=float(options.get("micro_area_ratio", 0.001)),
        micro_displacement_ratio=float(
            options.get("micro_displacement_ratio", 0.04)
        ),
        credible_displacement_ratio=float(
            options.get("credible_displacement_ratio", 0.03)
        ),
        maximum_credible_track_seconds=float(
            options.get("maximum_credible_track_seconds", 6.0)
        ),
    )


def register_adaptive_motion_stages(registry: MotionStageRegistry) -> None:
    registry.register(MotionStageRegistration(
        implementation="adaptive_ema_background",
        builder=_build_background,
        requires=frozenset({"processed_frame_history", "runtime"}),
        provides=frozenset({"background_image", "difference_image", "difference_history", "debug"}),
        graph="qualification",
        category="background",
        display_name="Adaptive scene background",
        description="Learns the normal scene while protecting moving regions from immediate absorption.",
        continuous_analysis=True,
        motion_source="adaptive_background",
        options=(
            MotionStageOption("learning_rate", "Normal learning speed", "number", 0.025, minimum=0.0001, maximum=1, advanced=True),
            MotionStageOption("fast_learning_rate", "Scene-change learning speed", "number", 0.18, minimum=0.001, maximum=1, advanced=True),
            MotionStageOption("motion_learning_scale", "Moving-region learning", "number", 0.03, minimum=0, maximum=1, advanced=True),
            MotionStageOption("global_change_ratio", "Whole-scene change level", "number", 0.55, minimum=0.05, maximum=1, advanced=True),
        ),
    ))
    registry.register(MotionStageRegistration(
        implementation="adaptive_statistical_threshold",
        builder=_build_threshold,
        requires=frozenset({"difference_history", "runtime"}),
        provides=frozenset({"binary_motion_mask", "threshold_mask_history", "debug"}),
        graph="qualification",
        category="threshold",
        display_name="Scene-adaptive noise threshold",
        description="Continuously adjusts pixel sensitivity for sensor noise and lighting conditions.",
        options=(
            MotionStageOption("sigma", "Noise separation", "number", 4.0, minimum=0.5, maximum=12, advanced=True),
            MotionStageOption("minimum", "Minimum threshold", "number", 7.0, minimum=0, maximum=255, advanced=True),
            MotionStageOption("maximum", "Maximum threshold", "number", 72.0, minimum=1, maximum=255, advanced=True),
            MotionStageOption("smoothing", "Adaptation speed", "number", 0.25, minimum=0.01, maximum=1, advanced=True),
        ),
    ))
    registry.register(MotionStageRegistration(
        implementation="connected_component_blobs",
        builder=_build_components,
        requires=frozenset({"processed_frame_history", "difference_history", "motion_mask_history", "configuration"}),
        provides=frozenset({"raw_blob_history"}),
        graph="qualification",
        category="blob_extraction",
        display_name="Detailed motion regions",
        description="Measures connected regions, shape, intensity, edges, and configured-zone overlap.",
        options=(MotionStageOption("edge_margin_ratio", "Edge area", "number", 0.06, minimum=0, maximum=0.25, advanced=True),),
    ))
    registry.register(MotionStageRegistration(
        implementation="adaptive_blob_filter",
        builder=_build_filter,
        requires=frozenset({"raw_blob_history"}),
        provides=frozenset({"filtered_blob_history", "blobs", "debug"}),
        graph="qualification",
        category="blob_filtering",
        display_name="Adaptive nuisance filter",
        description="Rejects noise-sized, sparse, extreme, edge, and ignored-zone regions.",
        options=(
            MotionStageOption("minimum_area_ratio", "Base minimum area", "number", 0.00025, minimum=0, maximum=1, advanced=True),
            MotionStageOption("minimum_fill_ratio", "Minimum solidity", "number", 0.12, minimum=0, maximum=1, advanced=True),
            MotionStageOption("maximum_aspect_ratio", "Maximum shape ratio", "number", 12.0, minimum=1, maximum=100, advanced=True),
            MotionStageOption("ignored_zone_overlap", "Ignored-zone overlap", "number", 0.6, minimum=0, maximum=1, advanced=True),
        ),
    ))
    registry.register(MotionStageRegistration(
        implementation="persistent_centroid_tracker",
        builder=_build_tracker,
        requires=frozenset({"filtered_blob_history", "runtime", "configuration"}),
        provides=frozenset({"dominant_track", "tracked_objects", "debug"}),
        graph="qualification",
        category="tracking",
        display_name="Persistent multi-region tracking",
        description="Tracks multiple motion regions over time with stable IDs, velocity, direction, and size history.",
        options=(
            MotionStageOption("maximum_distance", "Maximum match distance", "number", 0.18, minimum=0.01, maximum=1, advanced=True),
            MotionStageOption("maximum_missed_frames", "Allowed missed samples", "integer", 3, minimum=0, maximum=30, advanced=True),
            MotionStageOption("maximum_track_age_seconds", "Track memory", "number", 8.0, minimum=0.5, maximum=60, advanced=True),
        ),
    ))
    registry.register(MotionStageRegistration(
        implementation="adaptive_motion_score",
        builder=_build_scorer,
        requires=frozenset({"tracked_objects", "processed_frame_history"}),
        provides=frozenset({"dominant_track", "tracked_objects", "scoring"}),
        graph="qualification",
        category="scoring",
        display_name="Adaptive credibility score",
        description="Scores persistence, size, direction, velocity, zones, scene noise, and insect-like behavior.",
        options=(
            MotionStageOption("minimum_persistence_frames", "Minimum persistent samples", "integer", 2, minimum=1, maximum=20),
            MotionStageOption("stationary_displacement_ratio", "Minimum position change", "number", 0.01, minimum=0, maximum=0.1, advanced=True),
            MotionStageOption("stationary_path_ratio", "Minimum movement path", "number", 0.025, minimum=0, maximum=0.2, advanced=True),
            MotionStageOption("micro_area_ratio", "Tiny disturbance size", "number", 0.001, minimum=0, maximum=0.1, advanced=True),
            MotionStageOption("micro_displacement_ratio", "Tiny disturbance travel", "number", 0.04, minimum=0, maximum=0.5, advanced=True),
            MotionStageOption("credible_displacement_ratio", "Full movement credit distance", "number", 0.03, minimum=0.0001, maximum=0.5, advanced=True),
            MotionStageOption("maximum_credible_track_seconds", "Maximum single motion age", "number", 6.0, minimum=0.5, maximum=60, advanced=True),
        ),
    ))

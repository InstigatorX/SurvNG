from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MotionBlob:
    box: tuple[float, float, float, float]
    centroid: tuple[float, float]
    area_pixels: float
    area_ratio: float
    touches_edge: bool = False
    fill_ratio: float = 0.0
    aspect_ratio: float = 1.0
    average_motion_intensity: float = 0.0
    edge_distance: float = 0.0
    zone_overlap: float = 0.0
    ignored_zone_overlap: float = 0.0
    zone_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MotionFrameBlobs:
    frame_area: int
    changed_pixels: int
    changed_ratio: float
    blobs: tuple[MotionBlob, ...]


@dataclass(frozen=True, slots=True)
class MotionTrack:
    track_id: int
    box: tuple[float, float, float, float]
    path: tuple[tuple[float, float], ...]
    observations: tuple[MotionBlob, ...]
    observation_frames: tuple[int, ...]
    active_history: tuple[bool, ...]
    changed_pixels: tuple[int, ...]
    changed_ratios: tuple[float, ...]
    score: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    consecutive_started_at: float = 0.0
    consecutive_frames: int = 0
    velocity: tuple[float, float] = (0.0, 0.0)
    direction: float = 0.0
    size_history: tuple[float, ...] = ()
    accumulated_score: float = 0.0

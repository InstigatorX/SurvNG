from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MotionBlob:
    box: tuple[float, float, float, float]
    centroid: tuple[float, float]
    area_pixels: float
    area_ratio: float
    touches_edge: bool = False


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

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Iterator, Protocol

import numpy as np

from ..config import ObjectTrackingConfig
from ..video_frames import DecodedVideoFrame, VideoFrameReference


Box = tuple[float, float, float, float]
FrameSample = tuple[np.ndarray, float, float]
FrameProvider = Callable[[], FrameSample | None]


@dataclass(frozen=True, slots=True)
class TrackingFrameBatch:
    """Frames plus the truthful continuity boundary for a tracking read."""

    frames: tuple[tuple[float, np.ndarray] | DecodedVideoFrame, ...]
    covered_through: float
    interruption: str | None = None

    def __iter__(self) -> Iterator[tuple[float, np.ndarray] | DecodedVideoFrame]:
        return iter(self.frames)


CatchupFrameProvider = Callable[
    [float, float, float, int],
    Iterable[tuple[float, np.ndarray] | DecodedVideoFrame],
]
TrackingUpdate = Callable[[int, dict[str, Any], list[dict[str, Any]] | None], object | None]
TrackingPublisher = Callable[[str, dict[str, Any]], None]
AppearanceIndexWriter = Callable[[int, str, Iterable[dict[str, Any]]], int]
TrackingCoverFrameProvider = Callable[
    [float, int, VideoFrameReference | None],
    np.ndarray | None,
]
TrackingSnapshotWriter = Callable[[np.ndarray, datetime], str]
TrackingCoverPromoter = Callable[..., dict[str, Any] | None]


class ObjectDetectorBackend(Protocol):
    config: Any

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        ...


class AppearanceEncoder(Protocol):
    @property
    def enabled(self) -> bool:
        ...

    def embed(self, person: np.ndarray) -> np.ndarray:
        ...

    def supports_label(self, label: str) -> bool:
        ...

    def embed_for_label(self, label: str, crop: np.ndarray) -> np.ndarray:
        ...

    def model_identity_for_label(self, label: str) -> dict[str, Any] | None:
        ...


class ObjectTrackerBackend(Protocol):
    def update(
        self,
        detections: list[dict[str, Any]],
        captured_at: float,
        *,
        confirm_new: bool = False,
    ) -> list[dict[str, Any]]:
        ...

    def has_live_tracks(self, captured_at: float) -> bool:
        ...

    def summaries(self, captured_at: float) -> list[dict[str, Any]]:
        ...

    def diagnostics(self) -> dict[str, Any]:
        ...


ObjectTrackerBuilder = Callable[[ObjectTrackingConfig, float], ObjectTrackerBackend]

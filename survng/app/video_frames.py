"""Typed decoded-frame identity shared by recording and tracking consumers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, overload

import numpy as np


@dataclass(frozen=True, slots=True)
class VideoFrameReference:
    """Stable locator and timing evidence for one decoded recording frame."""

    source_path: Path
    seek_offset_seconds: float
    pts: int
    pts_seconds: float
    time_base_num: int
    time_base_den: int
    captured_at: float
    exact: bool = True


@dataclass(frozen=True, slots=True)
class DecodedVideoFrame:
    """A frame with exact source timing while retaining two-value unpacking."""

    captured_at: float
    frame: np.ndarray
    reference: VideoFrameReference | None = None

    def __iter__(self) -> Iterator[object]:
        yield self.captured_at
        yield self.frame

    @overload
    def __getitem__(self, index: int) -> float | np.ndarray: ...

    def __getitem__(self, index: int) -> float | np.ndarray:
        if index == 0:
            return self.captured_at
        if index == 1:
            return self.frame
        raise IndexError(index)

    def __len__(self) -> int:
        return 2

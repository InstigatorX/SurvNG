"""Speech loudness telemetry (recorded only; nothing is gated on it yet).

Absolute dBFS depends on each camera's microphone gain and automatic gain
control, so it is a poor distance signal by itself. Each segment therefore
also records how far its voiced audio sits above a rolling background noise
floor. These numbers exist to calibrate a future proximity filter from real
data; they do not filter anything today.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FLOOR_ALPHA = 0.02  # ~1.6 s time constant at 31 frames/s
QUIET_PROBABILITY = 0.15
MIN_DBFS = -120.0


def dbfs(rms: float) -> float:
    return MIN_DBFS if rms <= 0 else max(MIN_DBFS, 20.0 * math.log10(rms))


@dataclass(frozen=True, slots=True)
class SpeechLevel:
    voiced_dbfs: float
    noise_dbfs: float | None

    @property
    def snr_db(self) -> float | None:
        return None if self.noise_dbfs is None else round(self.voiced_dbfs - self.noise_dbfs, 1)


class LevelTracker:
    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self._floor_db: float | None = None
        self._energy = 0.0
        self._frames = 0

    def observe(self, frame: np.ndarray, probability: float) -> None:
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        if probability >= self.threshold:
            self._energy += rms * rms
            self._frames += 1
        elif probability < QUIET_PROBABILITY:
            level = dbfs(rms)
            self._floor_db = level if self._floor_db is None else (
                (1.0 - FLOOR_ALPHA) * self._floor_db + FLOOR_ALPHA * level
            )

    def take(self) -> SpeechLevel | None:
        """Level of the voiced audio since the last call, then reset."""
        if not self._frames:
            return None
        level = SpeechLevel(
            voiced_dbfs=round(dbfs(math.sqrt(self._energy / self._frames)), 1),
            noise_dbfs=None if self._floor_db is None else round(self._floor_db, 1),
        )
        self._energy = 0.0
        self._frames = 0
        return level

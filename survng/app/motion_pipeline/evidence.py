from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class MotionEvidenceSample:
    source: str
    captured_at: float
    values: Mapping[str, Any]


class MotionEvidenceRepository:
    """Thread-safe, per-camera exchange for independently produced evidence."""

    def __init__(self, camera_id: str, max_samples_per_source: int = 64) -> None:
        self.camera_id = camera_id
        self.max_samples_per_source = max(1, int(max_samples_per_source))
        self._samples: dict[str, deque[MotionEvidenceSample]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def configure_source(self, source: str, **metadata: Any) -> None:
        with self._lock:
            self._metadata[source] = dict(metadata)

    def append(self, source: str, captured_at: float, values: Mapping[str, Any]) -> None:
        sample = MotionEvidenceSample(source, captured_at, dict(values))
        with self._lock:
            samples = self._samples.setdefault(
                source,
                deque(maxlen=self.max_samples_per_source),
            )
            samples.append(sample)

    def window(
        self,
        source: str,
        started_at: float,
        ended_at: float,
    ) -> tuple[MotionEvidenceSample, ...]:
        with self._lock:
            return tuple(
                sample
                for sample in self._samples.get(source, ())
                if started_at <= sample.captured_at <= ended_at
            )

    def last(self, source: str) -> MotionEvidenceSample | None:
        with self._lock:
            samples = self._samples.get(source)
            return samples[-1] if samples else None

    def status(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            sources = set(self._metadata) | set(self._samples)
            return {
                source: {
                    **self._metadata.get(source, {}),
                    "sample_count": len(self._samples.get(source, ())),
                    "last": (
                        dict(self._samples[source][-1].values)
                        if self._samples.get(source)
                        else None
                    ),
                    "last_captured_at": (
                        self._samples[source][-1].captured_at
                        if self._samples.get(source)
                        else None
                    ),
                }
                for source in sorted(sources)
            }

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

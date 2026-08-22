from __future__ import annotations

from collections import deque
import math
import threading


class RollingLatencySamples:
    """Thread-safe bounded latency samples for lightweight status telemetry."""

    def __init__(self, maxlen: int = 600) -> None:
        self._values: deque[float] = deque(maxlen=max(1, int(maxlen)))
        self._lock = threading.Lock()

    def add(self, ms: float) -> None:
        value = float(ms)
        if not math.isfinite(value):
            return
        with self._lock:
            self._values.append(max(0.0, value))

    def percentile(self, p: float) -> float | None:
        with self._lock:
            values = sorted(self._values)
        value = self._percentile(values, p)
        return round(value, 3) if value is not None else None

    def snapshot(self) -> dict[str, int | float | None]:
        with self._lock:
            values = sorted(self._values)
        return {
            "samples": len(values),
            "p50_ms": self._rounded_percentile(values, 50),
            "p95_ms": self._rounded_percentile(values, 95),
            "p99_ms": self._rounded_percentile(values, 99),
        }

    @classmethod
    def _rounded_percentile(
        cls,
        sorted_values: list[float],
        p: float,
    ) -> float | None:
        value = cls._percentile(sorted_values, p)
        return round(value, 3) if value is not None else None

    @staticmethod
    def _percentile(sorted_values: list[float], p: float) -> float | None:
        if not sorted_values:
            return None
        percentile = float(p)
        if percentile > 1.0:
            percentile /= 100.0
        percentile = min(1.0, max(0.0, percentile))
        index = min(
            len(sorted_values) - 1,
            max(0, int(math.ceil(len(sorted_values) * percentile) - 1)),
        )
        return float(sorted_values[index])

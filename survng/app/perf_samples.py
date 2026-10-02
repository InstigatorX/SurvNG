from __future__ import annotations

from collections import deque
import math
import threading
import time


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


class TimedLock:
    """Lock wrapper that measures only contended acquisitions.

    An uncontended acquire costs one extra non-blocking attempt; waits are
    sampled so status can distinguish lock contention from slow work.
    """

    def __init__(self, inner: object | None = None, maxlen: int = 2000) -> None:
        self._inner = inner if inner is not None else threading.Lock()
        self._waits = RollingLatencySamples(maxlen)
        self._acquisitions = 0
        self._contended = 0
        self._wait_total_ms = 0.0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if self._inner.acquire(False):
            self._acquisitions += 1
            return True
        if not blocking:
            return False
        started = time.monotonic()
        acquired = self._inner.acquire(True, timeout)
        if acquired:
            waited = (time.monotonic() - started) * 1000.0
            self._acquisitions += 1
            self._contended += 1
            self._wait_total_ms += waited
            self._waits.add(waited)
        return acquired

    def release(self) -> None:
        self._inner.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def snapshot(self) -> dict[str, int | float | None]:
        return {
            "acquisitions": self._acquisitions,
            "contended": self._contended,
            "wait_total_ms": round(self._wait_total_ms, 3),
            **{f"wait_{key}": value for key, value in self._waits.snapshot().items()
               if key != "samples"},
        }

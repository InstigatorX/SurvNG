"""Fixed-memory latency histograms and the continuous event-loop lag monitor.

Both are cheap enough to run for the whole process lifetime: a record is one
``bisect`` plus a few integer updates under a small lock, and memory is fixed
at construction (bucket count x window slot count), independent of how many
values are recorded.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from bisect import bisect_left
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

# Upper bucket bounds in milliseconds; one extra overflow bucket follows.
DEFAULT_BUCKET_BOUNDS_MS: tuple[float, ...] = (
    1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000,
)
DEFAULT_WINDOW_SECONDS = 300.0
DEFAULT_SLOT_SECONDS = 30.0


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


class LatencyHistogram:
    """Thread-safe fixed-bucket histogram with a rolling recent window.

    Cumulative bucket counts, count, total and max cover the whole lifetime.
    The recent window is a ring of time slots; a slot is reset in place when
    the clock moves into it again, so the window covers between
    ``window_seconds - slot_seconds`` and ``window_seconds`` of history.
    Percentiles are approximate: the upper bound of the bucket holding the
    requested rank, capped at the observed window maximum.
    """

    def __init__(
        self,
        bounds_ms: tuple[float, ...] = DEFAULT_BUCKET_BOUNDS_MS,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        slot_seconds: float = DEFAULT_SLOT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        bounds = tuple(float(bound) for bound in bounds_ms)
        if not bounds or any(b <= a for a, b in zip(bounds, bounds[1:])):
            raise ValueError("histogram bounds must be strictly increasing")
        if slot_seconds <= 0 or window_seconds < slot_seconds:
            raise ValueError("window must contain at least one positive slot")
        self.bounds_ms = bounds
        self.slot_seconds = float(slot_seconds)
        self._slots = int(math.ceil(window_seconds / slot_seconds))
        self.window_seconds = self._slots * self.slot_seconds
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        buckets = len(bounds) + 1
        self._counts = [0] * buckets
        self._count = 0
        self._total = 0.0
        self._max = 0.0
        self._slot_epoch = [-1] * self._slots
        self._slot_counts = [[0] * buckets for _ in range(self._slots)]
        self._slot_count = [0] * self._slots
        self._slot_max = [0.0] * self._slots
        self._slot_max_at = [0.0] * self._slots

    def record(self, value_ms: float) -> None:
        value = float(value_ms)
        if not math.isfinite(value):
            return
        value = max(0.0, value)
        bucket = bisect_left(self.bounds_ms, value)
        with self._lock:
            # Read the clock under the lock so a slot is never reset by a
            # writer holding an older epoch.
            epoch = int(self._clock() // self.slot_seconds)
            slot = epoch % self._slots
            self._counts[bucket] += 1
            self._count += 1
            self._total += value
            if value > self._max:
                self._max = value
            counts = self._slot_counts[slot]
            if self._slot_epoch[slot] != epoch:
                self._slot_epoch[slot] = epoch
                for index in range(len(counts)):
                    counts[index] = 0
                self._slot_count[slot] = 0
                self._slot_max[slot] = 0.0
                self._slot_max_at[slot] = 0.0
            counts[bucket] += 1
            self._slot_count[slot] += 1
            if value >= self._slot_max[slot]:
                self._slot_max[slot] = value
                self._slot_max_at[slot] = self._wall_clock()

    def _percentile(self, counts: list[int], total: int, p: float, maximum: float) -> float | None:
        if total <= 0:
            return None
        rank = max(1, int(math.ceil(total * p)))
        seen = 0
        for index, count in enumerate(counts):
            seen += count
            if seen >= rank:
                if index >= len(self.bounds_ms):
                    return maximum
                return min(self.bounds_ms[index], maximum)
        return maximum

    def snapshot(self) -> dict[str, Any]:
        current = int(self._clock() // self.slot_seconds)
        oldest = current - self._slots + 1
        with self._lock:
            cumulative = list(self._counts)
            count, total, maximum = self._count, self._total, self._max
            recent = [0] * len(cumulative)
            recent_count = 0
            recent_max = 0.0
            recent_max_at: float | None = None
            for slot in range(self._slots):
                epoch = self._slot_epoch[slot]
                if epoch < oldest or epoch > current or not self._slot_count[slot]:
                    continue
                for index, value in enumerate(self._slot_counts[slot]):
                    recent[index] += value
                recent_count += self._slot_count[slot]
                if recent_max_at is None or self._slot_max[slot] >= recent_max:
                    recent_max = self._slot_max[slot]
                    recent_max_at = self._slot_max_at[slot]
        return {
            "count": count,
            "total_ms": round(total, 3),
            "max_ms": _round(maximum) if count else None,
            "mean_ms": _round(total / count) if count else None,
            "bucket_upper_bounds_ms": list(self.bounds_ms),
            # One more entry than the bounds: the last counts values above them.
            "bucket_counts": cumulative,
            "recent": {
                "window_seconds": self.window_seconds,
                "count": recent_count,
                "max_ms": _round(recent_max) if recent_count else None,
                "max_at_epoch_seconds": (
                    round(recent_max_at, 3) if recent_count and recent_max_at else None
                ),
                **{
                    f"p{label}_ms": _round(
                        self._percentile(recent, recent_count, p, recent_max)
                    )
                    for label, p in (("50", 0.50), ("95", 0.95), ("99", 0.99))
                },
            },
        }


class EventLoopLagMonitor:
    """Continuously measure how late a fixed-interval asyncio sleep wakes up.

    The 100 ms interval keeps the cost to ten timer wakeups per second while
    guaranteeing that any loop stall longer than the interval is observed
    (as lag of at least ``stall - interval``); shorter stalls are sampled in
    proportion to their duration. The default histogram bounds put stalls
    that users notice (>= 100 ms) in their own buckets.
    """

    DEFAULT_INTERVAL_SECONDS = 0.1
    STALL_LOG_THRESHOLD_MS = 1000.0
    STALL_LOG_INTERVAL_SECONDS = 300.0

    def __init__(
        self,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        *,
        histogram: LatencyHistogram | None = None,
        stall_log_threshold_ms: float = STALL_LOG_THRESHOLD_MS,
        stall_log_interval_seconds: float = STALL_LOG_INTERVAL_SECONDS,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval must be positive")
        self.interval_seconds = float(interval_seconds)
        self.histogram = histogram if histogram is not None else LatencyHistogram()
        self._stall_log_threshold_ms = float(stall_log_threshold_ms)
        self._stall_log_interval = float(stall_log_interval_seconds)
        self._next_stall_log = 0.0
        self._suppressed_stalls = 0
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start on the running loop; a second call while running is a no-op."""
        if self.running:
            return
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name="survng-event-loop-lag-monitor"
        )

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        interval = self.interval_seconds
        while True:
            expected = loop.time() + interval
            await asyncio.sleep(interval)
            lag_ms = max(0.0, (loop.time() - expected) * 1000.0)
            self.histogram.record(lag_ms)
            if lag_ms >= self._stall_log_threshold_ms:
                self._log_stall(lag_ms)

    def _log_stall(self, lag_ms: float) -> None:
        now = time.monotonic()
        if now < self._next_stall_log:
            self._suppressed_stalls += 1
            return
        LOGGER.warning(
            "event loop stalled for %.0f ms (%d similar stalls not logged since last report)",
            lag_ms,
            self._suppressed_stalls,
        )
        self._suppressed_stalls = 0
        self._next_stall_log = now + self._stall_log_interval

    def snapshot(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "interval_ms": round(self.interval_seconds * 1000.0, 3),
            **self.histogram.snapshot(),
        }

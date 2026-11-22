from __future__ import annotations

from collections import deque
import logging
import math
import sys
import threading
import time

from .runtime_metrics import LatencyHistogram

LOGGER = logging.getLogger(__name__)


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


# Frames inside these files are lock/connection plumbing, not the caller.
_PLUMBING_FILES = ("perf_samples.py", "main_database.py", "contextlib.py", "threading.py")
SLOW_SITE_LIMIT = 32
SLOW_SITE_LOG_INTERVAL_SECONDS = 300.0
SLOW_SITE_MAX_CHARS = 160


def caller_site(skip: int = 1) -> str:
    """Return ``path:line function`` of the first frame outside lock plumbing.

    Only code locations are reported, never arguments or locals. The path is
    trimmed to start at ``survng/`` when it is inside the package.
    """
    try:
        frame = sys._getframe(skip + 1)
    except ValueError:
        return "unknown"
    while frame is not None and (
        frame.f_code.co_filename.endswith(_PLUMBING_FILES)
        or "sqlite3" in frame.f_code.co_filename
    ):
        frame = frame.f_back
    if frame is None:
        return "unknown"
    filename = frame.f_code.co_filename.replace("\\", "/")
    marker = filename.rfind("survng/")
    if marker >= 0:
        filename = filename[marker:]
    else:
        filename = filename.rsplit("/", 1)[-1]
    return f"{filename}:{frame.f_lineno} {frame.f_code.co_name}"[:SLOW_SITE_MAX_CHARS]


class SlowSiteTable:
    """Bounded per-call-site record of operations slower than a threshold.

    Diagnostic for lock and transaction stalls. Fast operations cost one
    comparison. At most ``SLOW_SITE_LIMIT`` sites are kept; when full, a new
    site replaces the one with the smallest maximum only if it is slower.
    Each site logs at most one warning per ``SLOW_SITE_LOG_INTERVAL_SECONDS``.
    """

    def __init__(
        self,
        name: str,
        threshold_ms: float,
        *,
        limit: int = SLOW_SITE_LIMIT,
        log_interval_seconds: float = SLOW_SITE_LOG_INTERVAL_SECONDS,
    ) -> None:
        self.name = name
        self.threshold_ms = float(threshold_ms)
        self._limit = max(1, int(limit))
        self._log_interval = float(log_interval_seconds)
        self._lock = threading.Lock()
        self._sites: dict[str, dict[str, float]] = {}
        self._dropped = 0

    def record(self, site: str, duration_ms: float, **extra_ms: float) -> None:
        if duration_ms < self.threshold_ms:
            return
        now = time.time()
        log_entry: dict[str, float] | None = None
        with self._lock:
            entry = self._sites.get(site)
            if entry is None:
                if len(self._sites) >= self._limit:
                    smallest = min(self._sites, key=lambda key: self._sites[key]["max_ms"])
                    if self._sites[smallest]["max_ms"] >= duration_ms:
                        self._dropped += 1
                        return
                    del self._sites[smallest]
                    self._dropped += 1
                entry = self._sites[site] = {
                    "count": 0, "total_ms": 0.0, "max_ms": 0.0, "last_ms": 0.0,
                    "last_at_epoch_seconds": 0.0, "last_logged_at": float("-inf"),
                }
            entry["count"] += 1
            entry["total_ms"] += duration_ms
            entry["max_ms"] = max(entry["max_ms"], duration_ms)
            entry["last_ms"] = duration_ms
            entry["last_at_epoch_seconds"] = now
            for key, value in extra_ms.items():
                entry[f"last_{key}"] = float(value)
                entry[f"max_{key}"] = max(entry.get(f"max_{key}", 0.0), float(value))
            if now - entry["last_logged_at"] >= self._log_interval:
                entry["last_logged_at"] = now
                log_entry = dict(entry)
        if log_entry is not None:
            details = "".join(
                f", {key}={value:.0f} ms" for key, value in sorted(extra_ms.items())
            )
            LOGGER.warning(
                "slow %s: %s took %.0f ms%s (count=%d, max=%.0f ms; "
                "further reports for this site suppressed for %.0f s)",
                self.name, site, duration_ms, details, int(log_entry["count"]),
                log_entry["max_ms"], self._log_interval,
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            sites = [
                {"site": site, **{k: v for k, v in entry.items() if k != "last_logged_at"}}
                for site, entry in self._sites.items()
            ]
            dropped = self._dropped
        sites.sort(key=lambda item: float(item["max_ms"]), reverse=True)
        for item in sites:
            for key, value in list(item.items()):
                if isinstance(value, float):
                    item[key] = round(value, 3)
        return {"threshold_ms": self.threshold_ms, "dropped": dropped, "sites": sites}


class TimedLock:
    """Lock wrapper that measures only contended acquisitions.

    An uncontended acquire costs one extra non-blocking attempt; waits are
    sampled so status can distinguish lock contention from slow work.
    Contended waits and every hold are also recorded in fixed-memory
    histograms (cumulative plus a recent window). Metric failures are
    counted and never prevent the lock from being acquired or released.
    """

    def __init__(
        self,
        inner: object | None = None,
        maxlen: int = 2000,
        *,
        slow_holds: SlowSiteTable | None = None,
    ) -> None:
        self._inner = inner if inner is not None else threading.Lock()
        # Optional per-call-site attribution of long holds (diagnostic).
        self._slow_holds = slow_holds
        self._waits = RollingLatencySamples(maxlen)
        self._holds = RollingLatencySamples(maxlen)
        self._wait_histogram = LatencyHistogram()
        self._hold_histogram = LatencyHistogram()
        self._held_at = threading.local()
        self._acquisitions = 0
        self._contended = 0
        self._wait_total_ms = 0.0
        self._metric_failures = 0

    def _metric_failed(self, error: Exception) -> None:
        self._metric_failures += 1
        if self._metric_failures == 1:
            LOGGER.warning("lock metric recording failed (%s)", type(error).__name__)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if self._inner.acquire(False):
            self._acquisitions += 1
            try:
                self._held_at.stack = [*getattr(self._held_at, "stack", ()), self._hold_start()]
            except Exception as error:
                self._metric_failed(error)
            return True
        if not blocking:
            return False
        started = time.monotonic()
        acquired = self._inner.acquire(True, timeout)
        if acquired:
            self._acquisitions += 1
            self._contended += 1
            try:
                waited = (time.monotonic() - started) * 1000.0
                self._wait_total_ms += waited
                self._waits.add(waited)
                self._wait_histogram.record(waited)
                self._held_at.stack = [*getattr(self._held_at, "stack", ()), self._hold_start()]
            except Exception as error:
                self._metric_failed(error)
        return acquired

    def release(self) -> None:
        try:
            stack = list(getattr(self._held_at, "stack", ()))
            started = stack.pop() if stack else None
            self._held_at.stack = stack
            if started is not None:
                started_at, site = started
                held = (time.monotonic() - started_at) * 1000.0
                self._holds.add(held)
                self._hold_histogram.record(held)
                if self._slow_holds is not None and site is not None:
                    self._slow_holds.record(site, held)
        except Exception as error:
            self._metric_failed(error)
        finally:
            self._inner.release()

    def _hold_start(self) -> tuple[float, str | None]:
        site = caller_site(2) if self._slow_holds is not None else None
        return time.monotonic(), site

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
            **{f"hold_{key}": value for key, value in self._holds.snapshot().items()
               if key != "samples"},
        }

    def histograms(self) -> dict[str, object]:
        """Fixed-bucket wait (contended only) and hold histograms, in ms."""
        return {
            "acquisitions": self._acquisitions,
            "contended": self._contended,
            "metric_failures": self._metric_failures,
            "wait": self._wait_histogram.snapshot(),
            "hold": self._hold_histogram.snapshot(),
            **(
                {"slow_holds": self._slow_holds.snapshot()}
                if self._slow_holds is not None
                else {}
            ),
        }

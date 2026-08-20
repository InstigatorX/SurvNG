from __future__ import annotations

import logging
import threading
from typing import Any, Callable

LOGGER = logging.getLogger("survng.app.object_tracking")


class AdaptiveTrackingLimiter:
    """Semaphore-compatible limiter with a guarded, observable burst slot."""

    def __init__(
        self,
        baseline: int,
        burst_limit: int,
        *,
        burst_enabled: bool = True,
        burst_guard: Callable[[], bool] | None = None,
    ) -> None:
        self.baseline = max(1, int(baseline))
        self.burst_limit = max(self.baseline, int(burst_limit))
        self.burst_enabled = bool(burst_enabled)
        self.burst_guard = burst_guard
        self._condition = threading.Condition()
        self._active = 0
        self._burst_admissions = 0
        self._burst_denials = 0

    def _limit(self) -> int:
        if not self.burst_enabled or self.burst_limit <= self.baseline:
            return self.baseline
        healthy = False
        try:
            healthy = self.burst_guard is None or bool(self.burst_guard())
        except Exception:
            LOGGER.exception("tracking burst capacity guard failed")
        if not healthy:
            self._burst_denials += 1
            return self.baseline
        return self.burst_limit

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                limit = self._limit()
                if self._active < limit:
                    self._active += 1
                    if self._active > self.baseline:
                        self._burst_admissions += 1
                    return True
                if not blocking:
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise ValueError("tracking limiter released too many times")
            self._active -= 1
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "active": self._active,
                "baseline": self.baseline,
                "burst_limit": self.burst_limit,
                "burst_enabled": self.burst_enabled,
                "burst_admissions": self._burst_admissions,
                "burst_denials": self._burst_denials,
            }

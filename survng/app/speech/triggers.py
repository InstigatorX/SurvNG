"""Speech-triggered incident checks: throttling and off-thread dispatch.

Every check costs a recorded-decode detection job, and the episode
controller reserves a new job for each speech notice after a finished one,
so speech needs its own limits: a per-camera cooldown and a rolling
5-minute budget. Dispatch runs on one worker thread so the audio reader
never blocks on the motion pipeline.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger("uvicorn.error")
BUDGET_WINDOW_SECONDS = 300.0
QUEUE_LIMIT = 32


@dataclass(slots=True)
class TriggerThrottle:
    cooldown_seconds: float
    max_per_window: int
    clock: Callable[[], float] = time.monotonic
    _last: float | None = None
    _recent: deque = field(default_factory=deque)
    requested: int = 0
    suppressed: int = 0

    def allow(self) -> bool:
        now = self.clock()
        while self._recent and now - self._recent[0] >= BUDGET_WINDOW_SECONDS:
            self._recent.popleft()
        if self._last is not None and now - self._last < self.cooldown_seconds:
            self.suppressed += 1
            return False
        if len(self._recent) >= self.max_per_window:
            self.suppressed += 1
            return False
        self._last = now
        self._recent.append(now)
        self.requested += 1
        return True

    def status(self) -> dict[str, Any]:
        return {"incident_checks": self.requested, "incident_checks_suppressed": self.suppressed}


class TriggerDispatcher:
    def __init__(self, request: Callable[[str, float], Any]) -> None:
        self.request = request
        self._queue: queue.Queue[tuple[str, float] | None] = queue.Queue(maxsize=QUEUE_LIMIT)
        self._thread: threading.Thread | None = None
        self.dropped = 0
        self.failed = 0

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="speech-incident-trigger", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        thread = self._thread
        if thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread.join(timeout)
        self._thread = None

    def submit(self, camera_id: str, started_at: float) -> bool:
        try:
            self._queue.put_nowait((camera_id, started_at))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            camera_id, started_at = item
            try:
                self.request(camera_id, started_at)
            except Exception:
                self.failed += 1
                LOGGER.warning("Speech incident check failed for camera %s", camera_id, exc_info=True)

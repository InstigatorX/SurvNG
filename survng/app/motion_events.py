from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any, Callable


MotionTrigger = dict[str, Any]
StatCallback = Callable[[str], None]


class MotionEventCoordinator:
    """Owns motion trigger admission, coalescing state, and retry scheduling.

    Camera-specific qualification and incident policy are deliberately supplied by
    the caller. This object owns only the concurrent runtime used to deliver a
    stable batch of triggers to that policy.
    """

    def __init__(self, *, queue_size: int, retry_limit: int) -> None:
        self.queue: queue.Queue[MotionTrigger | None] = queue.Queue(maxsize=queue_size)
        self.retry_batches: deque[MotionTrigger] = deque()
        self.thread: threading.Thread | None = None
        self.active_triggers: list[MotionTrigger] | None = None
        self.adaptive_trigger_pending = False
        self.adaptive_last_completed_at = 0.0
        self.priority_motion_times: deque[float] = deque(maxlen=16)
        self.camera_motion_times: deque[float] = deque(maxlen=32)
        self._retry_limit = retry_limit
        self._lock = threading.RLock()

    def enqueue(
        self,
        trigger: MotionTrigger,
        *,
        evict_oldest: bool = True,
        on_trigger: StatCallback | None = None,
        on_drop: StatCallback | None = None,
    ) -> bool:
        if on_trigger is not None:
            on_trigger("triggers")
        try:
            self.queue.put_nowait(trigger)
            return True
        except queue.Full:
            if not evict_oldest:
                if on_drop is not None:
                    on_drop("dropped_triggers")
                return False
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            if on_drop is not None:
                on_drop("dropped_triggers")
            try:
                self.queue.put_nowait(trigger)
                return True
            except queue.Full:
                return False

    def clear(self) -> None:
        with self._lock:
            self.retry_batches.clear()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

    def signal_stop(self) -> None:
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            # The shared stop event remains authoritative. A full queue cannot
            # strand the consumer because its timed get observes that event.
            pass

    def next_trigger(self, timeout: float) -> MotionTrigger | None:
        with self._lock:
            if self.retry_batches:
                return self.retry_batches.popleft()
        return self.queue.get(timeout=timeout)

    def coalesce(
        self,
        first: MotionTrigger,
        *,
        quiet_seconds: float,
        stop_event: threading.Event,
    ) -> list[MotionTrigger] | None:
        retry_batch = first.get("_retry_batch")
        if isinstance(retry_batch, list):
            return [dict(item) for item in retry_batch]
        triggers = [first]
        quiet_deadline = time.monotonic() + quiet_seconds
        hard_deadline = time.monotonic() + max(2.0, quiet_seconds * 4)
        while not stop_event.is_set():
            remaining = min(quiet_deadline, hard_deadline) - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self.queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                return None
            triggers.append(item)
            quiet_deadline = min(hard_deadline, time.monotonic() + quiet_seconds)
        return triggers

    def schedule_retry(
        self,
        triggers: list[MotionTrigger],
        *,
        stop_event: threading.Event,
        on_retry: StatCallback | None = None,
        on_drop: StatCallback | None = None,
    ) -> None:
        retry_count = max(
            (int(item.get("_event_retry_count") or 0) for item in triggers),
            default=0,
        ) + 1
        if retry_count > self._retry_limit:
            if on_drop is not None:
                on_drop("event_retry_drops")
            return
        retry_triggers = [
            {**item, "_event_retry_count": retry_count}
            for item in triggers
        ]
        if stop_event.wait(0.25 * retry_count):
            return
        wrapper: MotionTrigger = {
            "topic": "internal/retry_batch",
            "message": f"motion event retry {retry_count}",
            "event_at": min(item["event_at"] for item in retry_triggers),
            "received_at": min(
                float(item.get("received_at") or time.time())
                for item in retry_triggers
            ),
            "_retry_batch": retry_triggers,
        }
        with self._lock:
            self.retry_batches.append(wrapper)
        if on_retry is not None:
            on_retry("event_retries")

    def reserve_adaptive(
        self,
        captured_at: float,
        *,
        rearm_seconds: float,
        priority_tolerance_seconds: float,
    ) -> bool:
        with self._lock:
            if (
                self.adaptive_trigger_pending
                or self.matches_recent_priority(
                    captured_at,
                    rearm_seconds=priority_tolerance_seconds,
                )
                or captured_at - self.adaptive_last_completed_at < rearm_seconds
            ):
                return False
            self.adaptive_trigger_pending = True
            return True

    def remember_priority(self, observed_at: float) -> None:
        with self._lock:
            self.priority_motion_times.append(observed_at)

    def matches_recent_priority(self, event_at: float, *, rearm_seconds: float) -> bool:
        with self._lock:
            return any(
                abs(event_at - priority_at) <= rearm_seconds
                for priority_at in self.priority_motion_times
            )

    def remember_camera_motion(self, observed_at: float) -> None:
        with self._lock:
            self.camera_motion_times.append(observed_at)

    def camera_motion_snapshot(self) -> tuple[float, ...]:
        with self._lock:
            return tuple(self.camera_motion_times)

    def latest_camera_motion(self) -> float:
        with self._lock:
            return max(self.camera_motion_times, default=0.0)

    def reserve_with(
        self,
        predicate: Callable[[bool, float], bool],
    ) -> bool:
        """Atomically reserve the adaptive slot when ``predicate`` permits it."""
        with self._lock:
            if not predicate(
                self.adaptive_trigger_pending,
                self.adaptive_last_completed_at,
            ):
                return False
            self.adaptive_trigger_pending = True
            return True

    def defer_adaptive(self, captured_at: float) -> None:
        with self._lock:
            self.adaptive_trigger_pending = False
            self.adaptive_last_completed_at = captured_at

    def complete_adaptive(self, triggers: list[MotionTrigger], completed_at: float) -> None:
        if not any(str(item.get("topic") or "").startswith("adaptive/") for item in triggers):
            return
        with self._lock:
            self.adaptive_trigger_pending = False
            self.adaptive_last_completed_at = completed_at

    def fail_active(self, completed_at: float) -> list[MotionTrigger] | None:
        with self._lock:
            failed = self.active_triggers
            self.active_triggers = None
            if self.adaptive_trigger_pending:
                self.adaptive_trigger_pending = False
                self.adaptive_last_completed_at = completed_at
            return failed

    def reset(self) -> None:
        self.clear()
        with self._lock:
            self.active_triggers = None
            self.adaptive_trigger_pending = False
            self.adaptive_last_completed_at = 0.0
            self.priority_motion_times.clear()
            self.camera_motion_times.clear()

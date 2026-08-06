from __future__ import annotations

import queue
import threading
import time
import math
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any, Callable, Iterable, Iterator, Mapping

from .motion import MotionQualificationResult

StatCallback = Callable[[str], None]


class RetryDisposition(StrEnum):
    SCHEDULED = "scheduled"
    DROPPED = "dropped"
    STOPPED = "stopped"


@dataclass(slots=True)
class MotionTrigger:
    """Typed observation submitted to the motion-decision workflow."""

    topic: str
    message: str
    event_at: datetime
    received_at: float
    prequalified: MotionQualificationResult | None = None
    decision_id: str = ""
    retry_count: int = 0
    retry_batch: tuple[MotionTrigger, ...] | None = None
    retry_qualification_result: MotionQualificationResult | None = None
    retry_diagnostics: dict[str, Any] | None = None
    audit_snapshot_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.topic, str) or not self.topic.strip():
            raise ValueError("motion trigger topic must be a non-empty string")
        if not isinstance(self.message, str):
            raise TypeError("motion trigger message must be a string")
        if not isinstance(self.event_at, datetime):
            raise TypeError("motion trigger event_at must be a datetime")
        if self.event_at.utcoffset() is None:
            raise ValueError("motion trigger event_at must include a timezone")
        if isinstance(self.received_at, bool) or not isinstance(
            self.received_at, (int, float)
        ):
            raise TypeError("motion trigger received_at must be numeric")
        self.received_at = float(self.received_at)
        if not math.isfinite(self.received_at):
            raise ValueError("motion trigger received_at must be finite")
        if isinstance(self.retry_count, bool) or not isinstance(self.retry_count, int):
            raise TypeError("motion trigger retry_count must be an integer")
        if self.retry_count < 0:
            raise ValueError("motion trigger retry_count cannot be negative")
        if self.prequalified is not None and not isinstance(
            self.prequalified, MotionQualificationResult
        ):
            raise TypeError("motion trigger prequalified result has an invalid type")
        if not isinstance(self.decision_id, str):
            raise TypeError("motion trigger decision ID must be a string")
        if self.retry_batch is not None and (
            not isinstance(self.retry_batch, tuple)
            or not all(isinstance(item, MotionTrigger) for item in self.retry_batch)
        ):
            raise TypeError("motion trigger retry batch must contain typed triggers")
        if self.retry_qualification_result is not None and not isinstance(
            self.retry_qualification_result, MotionQualificationResult
        ):
            raise TypeError("motion trigger retry result has an invalid type")
        if self.retry_diagnostics is not None and not isinstance(
            self.retry_diagnostics, dict
        ):
            raise TypeError("motion trigger retry diagnostics must be a dictionary")
        if self.audit_snapshot_path is not None and not isinstance(
            self.audit_snapshot_path, str
        ):
            raise TypeError("motion trigger audit snapshot path must be a string")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MotionTrigger:
        known = {
            "topic", "message", "event_at", "received_at", "prequalified",
            "_motion_decision_id", "_event_retry_count", "_retry_batch",
            "_retry_qualification_result", "_retry_diagnostics",
            "_motion_audit_snapshot_path",
        }
        unknown = set(value) - known
        if unknown:
            raise ValueError(
                f"unsupported motion trigger fields: {', '.join(sorted(unknown))}"
            )
        missing = {"topic", "event_at", "received_at"} - set(value)
        if missing:
            raise ValueError(
                f"missing motion trigger fields: {', '.join(sorted(missing))}"
            )
        raw_retry_batch = value.get("_retry_batch")
        retry_batch = None
        if raw_retry_batch is not None:
            if not isinstance(raw_retry_batch, (list, tuple)):
                raise TypeError("motion trigger retry batch must be a sequence")
            retry_batch = tuple(coerce_motion_trigger(item) for item in raw_retry_batch)
        prequalified = value.get("prequalified")
        if prequalified is not None and not isinstance(
            prequalified, MotionQualificationResult
        ):
            raise TypeError("motion trigger prequalified result has an invalid type")
        retry_result = value.get("_retry_qualification_result")
        if retry_result is not None and not isinstance(
            retry_result, MotionQualificationResult
        ):
            raise TypeError("motion trigger retry result has an invalid type")
        retry_diagnostics = value.get("_retry_diagnostics")
        if retry_diagnostics is not None and not isinstance(
            retry_diagnostics, Mapping
        ):
            raise TypeError("motion trigger retry diagnostics must be a mapping")
        decision_id = value.get("_motion_decision_id", "")
        if not isinstance(decision_id, str):
            raise TypeError("motion trigger decision ID must be a string")
        retry_count = value.get("_event_retry_count", 0)
        if isinstance(retry_count, bool) or not isinstance(retry_count, int):
            raise TypeError("motion trigger retry count must be an integer")
        audit_snapshot_path = value.get("_motion_audit_snapshot_path")
        if audit_snapshot_path is not None and not isinstance(
            audit_snapshot_path, str
        ):
            raise TypeError("motion trigger audit snapshot path must be a string")
        return cls(
            topic=value["topic"],
            message=value.get("message", ""),
            event_at=value["event_at"],
            received_at=value["received_at"],
            prequalified=prequalified,
            decision_id=decision_id,
            retry_count=retry_count,
            retry_batch=retry_batch,
            retry_qualification_result=retry_result,
            retry_diagnostics=(
                dict(retry_diagnostics)
                if retry_diagnostics is not None
                else None
            ),
            audit_snapshot_path=audit_snapshot_path,
        )

    # Temporary read-only compatibility for callers migrating from trigger
    # dictionaries. New production code uses the typed attributes above.
    def __getitem__(self, key: str) -> Any:
        legacy = {
            "topic": self.topic,
            "message": self.message,
            "event_at": self.event_at,
            "received_at": self.received_at,
            "prequalified": self.prequalified,
            "_motion_decision_id": self.decision_id,
            "_event_retry_count": self.retry_count,
            "_retry_batch": self.retry_batch,
            "_retry_qualification_result": self.retry_qualification_result,
            "_retry_diagnostics": self.retry_diagnostics,
            "_motion_audit_snapshot_path": self.audit_snapshot_path,
        }
        if key not in legacy:
            raise KeyError(key)
        return legacy[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def coerce_motion_trigger(value: MotionTrigger | Mapping[str, Any]) -> MotionTrigger:
    return value if isinstance(value, MotionTrigger) else MotionTrigger.from_mapping(value)


@dataclass(frozen=True, slots=True)
class MotionTriggerBatch:
    """An immutable, ordered motion burst delivered as one decision unit."""

    triggers: tuple[MotionTrigger, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.triggers:
            raise ValueError("motion trigger batch cannot be empty")
        if not all(isinstance(item, MotionTrigger) for item in self.triggers):
            raise TypeError("motion trigger batch must contain typed triggers")

    @classmethod
    def coerce(
        cls,
        values: MotionTriggerBatch | Iterable[MotionTrigger | Mapping[str, Any]],
    ) -> MotionTriggerBatch:
        if isinstance(values, cls):
            return values
        return cls(tuple(coerce_motion_trigger(item) for item in values))

    def __iter__(self) -> Iterator[MotionTrigger]:
        return iter(self.triggers)

    def __len__(self) -> int:
        return len(self.triggers)

    def __getitem__(self, index: int) -> MotionTrigger:
        return self.triggers[index]


class MotionEventCoordinator:
    """Owns motion trigger admission, coalescing state, and retry scheduling.

    Camera-specific qualification and incident policy are deliberately supplied by
    the caller. This object owns only the concurrent runtime used to deliver a
    stable batch of triggers to that policy.
    """

    def __init__(self, *, queue_size: int, retry_limit: int) -> None:
        self.queue: queue.Queue[MotionTrigger | Mapping[str, Any] | None] = queue.Queue(
            maxsize=queue_size
        )
        self.retry_batches: deque[MotionTrigger] = deque()
        self.thread: threading.Thread | None = None
        self.active_triggers: MotionTriggerBatch | None = None
        self.adaptive_trigger_pending = False
        self.adaptive_last_completed_at = 0.0
        self.priority_motion_times: deque[float] = deque(maxlen=16)
        self.camera_motion_times: deque[float] = deque(maxlen=32)
        self._retry_limit = retry_limit
        self._lock = threading.RLock()

    def enqueue(
        self,
        trigger: MotionTrigger | Mapping[str, Any],
        *,
        evict_oldest: bool = True,
        on_trigger: StatCallback | None = None,
        on_drop: StatCallback | None = None,
    ) -> bool:
        trigger = coerce_motion_trigger(trigger)
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
            dropped = 0
            try:
                self.queue.get_nowait()
                dropped += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(trigger)
                queued = True
            except queue.Full:
                dropped += 1
                queued = False
            if on_drop is not None:
                for _ in range(dropped):
                    on_drop("dropped_triggers")
            return queued

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
                return coerce_motion_trigger(self.retry_batches.popleft())
        item = self.queue.get(timeout=timeout)
        return None if item is None else coerce_motion_trigger(item)

    def retry_queue_depth(self) -> int:
        with self._lock:
            return len(self.retry_batches)

    def coalesce(
        self,
        first: MotionTrigger | Mapping[str, Any],
        *,
        quiet_seconds: float,
        stop_event: threading.Event,
    ) -> MotionTriggerBatch | None:
        first = coerce_motion_trigger(first)
        if first.retry_batch is not None:
            return MotionTriggerBatch(first.retry_batch)
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
            triggers.append(coerce_motion_trigger(item))
            quiet_deadline = min(hard_deadline, time.monotonic() + quiet_seconds)
        return MotionTriggerBatch(tuple(triggers))

    def schedule_retry(
        self,
        triggers: MotionTriggerBatch | Iterable[MotionTrigger | Mapping[str, Any]],
        *,
        stop_event: threading.Event,
        on_retry: StatCallback | None = None,
        on_drop: StatCallback | None = None,
    ) -> RetryDisposition:
        batch = MotionTriggerBatch.coerce(triggers)
        retry_count = max(
            (item.retry_count for item in batch),
            default=0,
        ) + 1
        if retry_count > self._retry_limit:
            if on_drop is not None:
                on_drop("event_retry_drops")
            return RetryDisposition.DROPPED
        retry_triggers = [
            replace(item, retry_count=retry_count)
            for item in batch
        ]
        if stop_event.wait(0.25 * retry_count):
            return RetryDisposition.STOPPED
        wrapper = MotionTrigger(
            topic="internal/retry_batch",
            message=f"motion event retry {retry_count}",
            event_at=min(item.event_at for item in retry_triggers),
            received_at=min(item.received_at for item in retry_triggers),
            retry_batch=tuple(retry_triggers),
        )
        with self._lock:
            self.retry_batches.append(wrapper)
        if on_retry is not None:
            on_retry("event_retries")
        return RetryDisposition.SCHEDULED

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

    def complete_adaptive(
        self,
        triggers: MotionTriggerBatch | Iterable[MotionTrigger | Mapping[str, Any]],
        completed_at: float,
    ) -> None:
        batch = MotionTriggerBatch.coerce(triggers)
        if not any(item.topic.startswith("adaptive/") for item in batch):
            return
        with self._lock:
            self.adaptive_trigger_pending = False
            self.adaptive_last_completed_at = completed_at

    def set_active(self, batch: MotionTriggerBatch | None) -> None:
        with self._lock:
            self.active_triggers = batch

    def take_failed_active(self) -> MotionTriggerBatch | None:
        with self._lock:
            failed = self.active_triggers
            self.active_triggers = None
            return failed

    def reset(self) -> None:
        self.clear()
        with self._lock:
            self.active_triggers = None
            self.adaptive_trigger_pending = False
            self.adaptive_last_completed_at = 0.0
            self.priority_motion_times.clear()
            self.camera_motion_times.clear()

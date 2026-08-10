from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any, Callable, Iterable, Iterator

from .motion import MotionQualificationResult

StatCallback = Callable[[str], None]


class RetryDisposition(StrEnum):
    SCHEDULED = "scheduled"
    DROPPED = "dropped"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class MotionEventTiming:
    """Explicit source, receipt, and media-sampling clocks for one trigger."""

    sampling_at: datetime
    received_at: datetime
    camera_event_at: datetime | None = None
    camera_to_receive_delta_seconds: float | None = None
    estimated_clock_offset_seconds: float | None = None
    estimated_delivery_delay_seconds: float | None = None
    selection_reason: str = "receipt_time"

    def to_payload(self) -> dict[str, Any]:
        return {
            "sampling_at": self.sampling_at.isoformat(),
            "received_at": self.received_at.isoformat(),
            "camera_event_at": (
                self.camera_event_at.isoformat()
                if self.camera_event_at is not None
                else None
            ),
            "camera_to_receive_delta_seconds": self.camera_to_receive_delta_seconds,
            "estimated_clock_offset_seconds": self.estimated_clock_offset_seconds,
            "estimated_delivery_delay_seconds": self.estimated_delivery_delay_seconds,
            "selection_reason": self.selection_reason,
        }


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
    event_timing: MotionEventTiming | None = None

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
        if self.event_timing is not None and not isinstance(
            self.event_timing, MotionEventTiming
        ):
            raise TypeError("motion trigger event timing has an invalid type")
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
        values: MotionTriggerBatch | Iterable[MotionTrigger],
    ) -> MotionTriggerBatch:
        if isinstance(values, cls):
            return values
        return cls(tuple(values))

    def __iter__(self) -> Iterator[MotionTrigger]:
        return iter(self.triggers)

    def __len__(self) -> int:
        return len(self.triggers)

    def __getitem__(self, index: int) -> MotionTrigger:
        return self.triggers[index]


@dataclass(frozen=True, slots=True)
class MotionSourceObservation:
    """One source observation belonging to the camera's current motion episode."""

    source: str
    observed_at: float


@dataclass(slots=True)
class MotionEpisodeState:
    """Single coordinator-owned record of cross-source episode state."""

    sequence: int = 0
    observations: deque[MotionSourceObservation] = field(
        default_factory=lambda: deque(maxlen=48)
    )
    active_triggers: MotionTriggerBatch | None = None
    adaptive_pending: bool = False
    adaptive_last_completed_at: float = 0.0
    incident_event_id: int | None = None
    latest_observed_at: float | None = None

    def observe(self, source: str, observed_at: float) -> None:
        if not math.isfinite(observed_at):
            raise ValueError("motion observation timestamp must be finite")
        if (
            self.latest_observed_at is None
            or observed_at - self.latest_observed_at > 30.0
        ):
            self.sequence += 1
            self.incident_event_id = None
        self.observations.append(MotionSourceObservation(source, observed_at))
        self.latest_observed_at = max(
            observed_at,
            self.latest_observed_at if self.latest_observed_at is not None else observed_at,
        )

    def times(self, source: str) -> tuple[float, ...]:
        return tuple(
            item.observed_at for item in self.observations if item.source == source
        )


class MotionEventCoordinator:
    """Owns motion trigger admission, coalescing state, and retry scheduling.

    Camera-specific qualification and incident policy are deliberately supplied by
    the caller. This object owns only the concurrent runtime used to deliver a
    stable batch of triggers to that policy.
    """

    def __init__(self, *, queue_size: int, retry_limit: int) -> None:
        self.queue: queue.Queue[MotionTrigger | None] = queue.Queue(
            maxsize=queue_size
        )
        self.retry_batches: deque[MotionTrigger] = deque()
        self._episode = MotionEpisodeState()
        self._retry_limit = retry_limit
        self._lock = threading.RLock()
        self._runtime_metrics = {
            "enqueued": 0,
            "evicted": 0,
            "rejected": 0,
            "queue_high_water": 0,
            "retries_scheduled": 0,
            "retries_dropped": 0,
            "retry_high_water": 0,
        }

    def enqueue(
        self,
        trigger: MotionTrigger,
        *,
        evict_oldest: bool = True,
        on_trigger: StatCallback | None = None,
        on_drop: StatCallback | None = None,
    ) -> bool:
        if not isinstance(trigger, MotionTrigger):
            raise TypeError("motion coordinator accepts only MotionTrigger values")
        if on_trigger is not None:
            on_trigger("triggers")
        try:
            self.queue.put_nowait(trigger)
            self._record_enqueue()
            return True
        except queue.Full:
            if not evict_oldest:
                self._record_rejected()
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
                self._record_enqueue(evicted=dropped)
            except queue.Full:
                dropped += 1
                queued = False
                self._record_rejected(evicted=max(0, dropped - 1))
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
                return self.retry_batches.popleft()
        item = self.queue.get(timeout=timeout)
        return item

    def retry_queue_depth(self) -> int:
        with self._lock:
            return len(self.retry_batches)

    def coalesce(
        self,
        first: MotionTrigger,
        *,
        quiet_seconds: float,
        stop_event: threading.Event,
    ) -> MotionTriggerBatch | None:
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
            triggers.append(item)
            quiet_deadline = min(hard_deadline, time.monotonic() + quiet_seconds)
        return MotionTriggerBatch(tuple(triggers))

    def schedule_retry(
        self,
        triggers: MotionTriggerBatch | Iterable[MotionTrigger],
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
            with self._lock:
                self._runtime_metrics["retries_dropped"] += 1
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
            self._runtime_metrics["retries_scheduled"] += 1
            self._runtime_metrics["retry_high_water"] = max(
                self._runtime_metrics["retry_high_water"],
                len(self.retry_batches),
            )
        if on_retry is not None:
            on_retry("event_retries")
        return RetryDisposition.SCHEDULED

    def runtime_status(self) -> dict[str, int]:
        with self._lock:
            return {
                **self._runtime_metrics,
                "queue_depth": self.queue.qsize(),
                "queue_capacity": self.queue.maxsize,
                "retry_queue_depth": len(self.retry_batches),
            }

    def _record_enqueue(self, *, evicted: int = 0) -> None:
        # qsize() is an observational value because Queue owns its own lock and
        # the consumer may drain immediately after admission. A successful
        # enqueue nevertheless establishes a minimum sampled depth of one.
        observed_depth = max(1, self.queue.qsize())
        with self._lock:
            self._runtime_metrics["enqueued"] += 1
            self._runtime_metrics["evicted"] += max(0, evicted)
            self._runtime_metrics["queue_high_water"] = max(
                self._runtime_metrics["queue_high_water"],
                observed_depth,
            )

    def _record_rejected(self, *, evicted: int = 0) -> None:
        with self._lock:
            self._runtime_metrics["rejected"] += 1
            self._runtime_metrics["evicted"] += max(0, evicted)

    def reserve_adaptive(
        self,
        captured_at: float,
        *,
        rearm_seconds: float,
        priority_tolerance_seconds: float,
    ) -> bool:
        with self._lock:
            if (
                self._episode.adaptive_pending
                or self.matches_recent_priority(
                    captured_at,
                    rearm_seconds=priority_tolerance_seconds,
                )
                or captured_at - self._episode.adaptive_last_completed_at < rearm_seconds
            ):
                return False
            self._episode.adaptive_pending = True
            self._episode.observe("adaptive", captured_at)
            return True

    def remember_priority(self, observed_at: float) -> None:
        with self._lock:
            self._episode.observe("priority", observed_at)

    def matches_recent_priority(self, event_at: float, *, rearm_seconds: float) -> bool:
        with self._lock:
            return any(
                abs(event_at - priority_at) <= rearm_seconds
                for priority_at in self._episode.times("priority")
            )

    def remember_camera_motion(self, observed_at: float) -> None:
        with self._lock:
            self._episode.observe("camera", observed_at)

    def camera_motion_snapshot(self) -> tuple[float, ...]:
        with self._lock:
            return self._episode.times("camera")

    def latest_camera_motion(self) -> float:
        with self._lock:
            return max(self._episode.times("camera"), default=0.0)

    def reserve_with(
        self,
        predicate: Callable[[bool, float], bool],
        *,
        observed_at: float | None = None,
    ) -> bool:
        """Atomically reserve the adaptive slot when ``predicate`` permits it."""
        with self._lock:
            if not predicate(
                self._episode.adaptive_pending,
                self._episode.adaptive_last_completed_at,
            ):
                return False
            self._episode.adaptive_pending = True
            if observed_at is not None:
                self._episode.observe("adaptive", observed_at)
            return True

    def defer_adaptive(self, captured_at: float) -> None:
        with self._lock:
            self._episode.adaptive_pending = False
            self._episode.adaptive_last_completed_at = captured_at

    def complete_adaptive(
        self,
        triggers: MotionTriggerBatch | Iterable[MotionTrigger],
        completed_at: float,
    ) -> None:
        batch = MotionTriggerBatch.coerce(triggers)
        if not any(item.topic.startswith("adaptive/") for item in batch):
            return
        with self._lock:
            self._episode.adaptive_pending = False
            self._episode.adaptive_last_completed_at = completed_at

    def set_active(self, batch: MotionTriggerBatch | None) -> None:
        with self._lock:
            self._episode.active_triggers = batch

    def take_failed_active(self) -> MotionTriggerBatch | None:
        with self._lock:
            failed = self._episode.active_triggers
            self._episode.active_triggers = None
            return failed

    def current_episode_sequence(self) -> int:
        with self._lock:
            return self._episode.sequence

    def link_incident(
        self,
        event_id: int | None,
        *,
        expected_sequence: int | None = None,
    ) -> bool:
        with self._lock:
            if (
                expected_sequence is not None
                and expected_sequence != self._episode.sequence
            ):
                return False
            self._episode.incident_event_id = event_id
            return True

    def active_incident_event_id(self) -> int | None:
        with self._lock:
            return self._episode.incident_event_id

    def episode_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sequence": self._episode.sequence,
                "observations": tuple(self._episode.observations),
                "adaptive_pending": self._episode.adaptive_pending,
                "adaptive_last_completed_at": self._episode.adaptive_last_completed_at,
                "incident_event_id": self._episode.incident_event_id,
                "latest_observed_at": self._episode.latest_observed_at,
            }

    def reset(self) -> None:
        self.clear()
        with self._lock:
            self._episode = MotionEpisodeState(sequence=self._episode.sequence)

    def reset_timebase(self) -> None:
        """Discard wall-clock histories without disturbing queued work."""
        with self._lock:
            # Invalidate callbacks and incident linkage created against the old
            # clock while leaving admitted queue work and reservations intact.
            self._episode.sequence += 1
            self._episode.incident_event_id = None
            self._episode.adaptive_last_completed_at = 0.0
            self._episode.observations.clear()
            self._episode.latest_observed_at = None

    # Transitional properties keep callers source-compatible while all episode
    # state now has one coordinator-owned source of truth.
    @property
    def active_triggers(self) -> MotionTriggerBatch | None:
        with self._lock:
            return self._episode.active_triggers

    @active_triggers.setter
    def active_triggers(self, value: MotionTriggerBatch | None) -> None:
        with self._lock:
            self._episode.active_triggers = value

    @property
    def adaptive_trigger_pending(self) -> bool:
        with self._lock:
            return self._episode.adaptive_pending

    @adaptive_trigger_pending.setter
    def adaptive_trigger_pending(self, value: bool) -> None:
        with self._lock:
            self._episode.adaptive_pending = bool(value)

    @property
    def adaptive_last_completed_at(self) -> float:
        with self._lock:
            return self._episode.adaptive_last_completed_at

    @adaptive_last_completed_at.setter
    def adaptive_last_completed_at(self, value: float) -> None:
        with self._lock:
            self._episode.adaptive_last_completed_at = float(value)

    @property
    def priority_motion_times(self) -> tuple[float, ...]:
        with self._lock:
            return self._episode.times("priority")

    @property
    def camera_motion_times(self) -> tuple[float, ...]:
        with self._lock:
            return self._episode.times("camera")

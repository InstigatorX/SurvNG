from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..perf_samples import RollingLatencySamples


def refinement_frame_count(stages: Any, *, fallback: int = 16) -> int:
    """Return the worst-case distinct decoded samples for a refinement plan.

    Offsets are event-relative here.  Negative offsets remain distinct: once
    mapped onto a recording segment they refer to different frame timestamps.
    Rounding matches the frame-reader cache precision.
    """
    try:
        count = len({
            round(float(offset), 3)
            for stage in stages
            for offset in stage
        })
    except (TypeError, ValueError):
        return fallback
    return max(1, count)


class RecordedDecodeLease:
    """Exactly-once release wrapper for a recorded-decode budget reservation."""

    def __init__(self, release_callback: Callable[[], None]) -> None:
        self._release_callback = release_callback
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            callback = self._release_callback
        callback()

    def __enter__(self) -> RecordedDecodeLease:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


@dataclass(order=True)
class _Waiter:
    incident_epoch: float
    sequence: int
    kind: str
    amount: int = 0
    frames: int = 0


class RecordedDecodeBudget:
    """Global process + estimated-memory budget for recorded OD frame sampling."""

    def __init__(
        self,
        *,
        max_processes: int = 2,
        memory_budget_bytes: int = 256 * 1024 * 1024,
        memory_per_process_bytes: int | None = None,
        estimated_frame_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self._condition = threading.Condition()
        self._max_processes = max(1, int(max_processes))
        self._memory_budget_bytes = max(1, int(memory_budget_bytes))
        self._memory_per_process_bytes = (
            max(1, int(memory_per_process_bytes))
            if memory_per_process_bytes is not None
            else None
        )
        self._estimated_frame_bytes = max(1, int(estimated_frame_bytes))
        self._active_processes = 0
        self._reserved_bytes = 0
        self._sequence = 0
        # Process slots and workflow-memory reservations are independent
        # resources.  Keep independent ordered queues so an unfittable memory
        # request cannot idle an available FFmpeg process slot (or vice versa).
        self._waiters: dict[str, list[_Waiter]] = {"process": [], "memory": []}
        self._admitted_processes = 0
        self._admitted_workflows = 0
        self._process_wait_ms = 0.0
        self._memory_wait_ms = 0.0
        self._process_wait_samples = RollingLatencySamples()
        self._memory_wait_samples = RollingLatencySamples()
        self._process_timeouts = 0
        self._memory_timeouts = 0
        self._cancellations = 0
        self._ffmpeg_attempts = {"hardware": 0, "cpu": 0}
        self._ffmpeg_successes = {"hardware": 0, "cpu": 0}

    @classmethod
    def from_detector_config(cls, config: Any) -> RecordedDecodeBudget:
        max_processes, memory_budget_bytes, memory_per_process_bytes = cls._config_limits(config)
        return cls(
            max_processes=max_processes,
            memory_budget_bytes=memory_budget_bytes,
            memory_per_process_bytes=memory_per_process_bytes,
            estimated_frame_bytes=int(
                getattr(config, "recorded_decode_estimated_frame_mb", 8) or 8
            )
            * 1024
            * 1024,
        )

    def reconfigure(
        self,
        *,
        max_processes: int,
        memory_budget_bytes: int,
        estimated_frame_bytes: int,
        memory_per_process_bytes: int | None = None,
    ) -> None:
        with self._condition:
            self._max_processes = max(1, int(max_processes))
            self._memory_budget_bytes = max(1, int(memory_budget_bytes))
            self._memory_per_process_bytes = (
                max(1, int(memory_per_process_bytes))
                if memory_per_process_bytes is not None
                else None
            )
            self._estimated_frame_bytes = max(1, int(estimated_frame_bytes))
            self._condition.notify_all()

    def reconfigure_from_detector_config(self, config: Any) -> None:
        max_processes, memory_budget_bytes, memory_per_process_bytes = self._config_limits(config)
        self.reconfigure(
            max_processes=max_processes,
            memory_budget_bytes=memory_budget_bytes,
            memory_per_process_bytes=memory_per_process_bytes,
            estimated_frame_bytes=int(
                getattr(config, "recorded_decode_estimated_frame_mb", 8) or 8
            )
            * 1024
            * 1024,
        )

    @staticmethod
    def _config_limits(config: Any) -> tuple[int, int, int]:
        max_processes = int(
            getattr(config, "recorded_decode_max_processes", 2) or 2
        )
        stages = getattr(config, "event_refinement_stages", ()) or ()
        frame_count = refinement_frame_count(stages)
        memory_per_process_bytes = frame_count * int(
            getattr(config, "recorded_decode_estimated_frame_mb", 8) or 8
        ) * 1024 * 1024
        return (
            max_processes,
            max_processes * memory_per_process_bytes,
            memory_per_process_bytes,
        )

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "max_processes": self._max_processes,
                "active_processes": self._active_processes,
                "memory_budget_bytes": self._memory_budget_bytes,
                "memory_per_process_bytes": self._memory_per_process_bytes,
                "reserved_bytes": self._reserved_bytes,
                "estimated_frame_bytes": self._estimated_frame_bytes,
                "waiting": sum(len(waiters) for waiters in self._waiters.values()),
                "admitted_processes": self._admitted_processes,
                "admitted_workflows": self._admitted_workflows,
                "process_wait_ms": round(self._process_wait_ms, 3),
                "memory_wait_ms": round(self._memory_wait_ms, 3),
                "decode_process_wait_ms_p95": self._process_wait_samples.percentile(95),
                "decode_process_wait_ms_p99": self._process_wait_samples.percentile(99),
                "decode_memory_wait_ms_p95": self._memory_wait_samples.percentile(95),
                "decode_memory_wait_ms_p99": self._memory_wait_samples.percentile(99),
                "process_timeouts": self._process_timeouts,
                "memory_timeouts": self._memory_timeouts,
                "cancellations": self._cancellations,
                "ffmpeg_attempts": dict(self._ffmpeg_attempts),
                "ffmpeg_successes": dict(self._ffmpeg_successes),
            }

    def record_ffmpeg(self, backend: str, *, success: bool) -> None:
        normalized = str(backend).strip().lower()
        if normalized not in self._ffmpeg_attempts:
            return
        with self._condition:
            self._ffmpeg_attempts[normalized] += 1
            if success:
                self._ffmpeg_successes[normalized] += 1

    def reserve_workflow(
        self,
        *,
        maximum_frames: int,
        incident_epoch: float,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> RecordedDecodeLease | None:
        frames = max(1, int(maximum_frames))
        requested = frames * self._estimated_frame_bytes
        return self._acquire(
            kind="memory",
            amount=requested,
            frames=frames,
            incident_epoch=incident_epoch,
            deadline=deadline,
            cancelled=cancelled,
        )

    def acquire_process(
        self,
        *,
        incident_epoch: float,
        deadline: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> RecordedDecodeLease | None:
        return self._acquire(
            kind="process",
            amount=1,
            frames=0,
            incident_epoch=incident_epoch,
            deadline=deadline,
            cancelled=cancelled,
        )

    def _acquire(
        self,
        *,
        kind: str,
        amount: int,
        frames: int,
        incident_epoch: float,
        deadline: float | None,
        cancelled: Callable[[], bool] | None,
    ) -> RecordedDecodeLease | None:
        started = time.monotonic()
        with self._condition:
            self._sequence += 1
            waiter = _Waiter(
                incident_epoch=float(incident_epoch),
                sequence=self._sequence,
                kind=kind,
                amount=int(amount),
                frames=int(frames),
            )
            waiters = self._waiters[kind]
            waiters.append(waiter)
            waiters.sort()
            try:
                while True:
                    if cancelled is not None and cancelled():
                        self._cancellations += 1
                        return None
                    if waiters[0] is waiter and self._fits(waiter):
                        self._account(waiter)
                        waiters.remove(waiter)
                        wait_ms = max(0.0, (time.monotonic() - started) * 1000.0)
                        if kind == "process":
                            self._process_wait_ms += wait_ms
                            self._process_wait_samples.add(wait_ms)
                            self._admitted_processes += 1
                        else:
                            self._memory_wait_ms += wait_ms
                            self._memory_wait_samples.add(wait_ms)
                            self._admitted_workflows += 1
                        self._condition.notify_all()
                        return RecordedDecodeLease(
                            lambda current=waiter: self._release(current)
                        )
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        if kind == "process":
                            self._process_timeouts += 1
                        else:
                            self._memory_timeouts += 1
                        return None
                    self._condition.wait(
                        timeout=0.1 if remaining is None else min(0.1, remaining)
                    )
            finally:
                if waiter in waiters:
                    waiters.remove(waiter)
                    self._condition.notify_all()

    def _fits(self, waiter: _Waiter) -> bool:
        if waiter.kind == "process":
            return self._active_processes < self._max_processes
        charge = self._memory_charge(waiter)
        # A workflow queued before a plan/estimate downscale may no longer fit
        # the new ceiling.  Let it run only after the budget drains;
        # retaining its full charge keeps status and release accounting honest.
        return (
            self._reserved_bytes == 0
            if charge > self._memory_budget_bytes
            else self._reserved_bytes + charge <= self._memory_budget_bytes
        )

    def _memory_charge(self, waiter: _Waiter) -> int:
        return max(1, waiter.frames) * self._estimated_frame_bytes

    def _account(self, waiter: _Waiter) -> None:
        if waiter.kind == "process":
            self._active_processes += 1
        else:
            # A queued workflow is charged using the current frame estimate,
            # then that charge is frozen in the lease for correct release.
            waiter.amount = self._memory_charge(waiter)
            self._reserved_bytes += waiter.amount

    def _release(self, waiter: _Waiter) -> None:
        with self._condition:
            if waiter.kind == "process":
                if self._active_processes <= 0:
                    raise ValueError("recorded decode process lease released too many times")
                self._active_processes -= 1
            else:
                if self._reserved_bytes < waiter.amount:
                    raise ValueError("recorded decode memory lease released too many times")
                self._reserved_bytes -= waiter.amount
            self._condition.notify_all()

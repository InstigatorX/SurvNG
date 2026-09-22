"""Process-wide admission, cancellation, and accounting for media work."""

from __future__ import annotations

import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Callable, Mapping


class MediaSessionKind(StrEnum):
    GO2RTC_WEBRTC = "go2rtc_webrtc"
    GO2RTC_MSE = "go2rtc_mse"
    MJPEG = "mjpeg"
    SNAPSHOT = "snapshot"
    CAPTURE_OPEN = "capture_open"
    CAPTURE_LIVE = "capture_live"
    CAPTURE_MAIN = "capture_main"
    RECORDING_REMUX = "recording_remux"
    RECORDING_PREWARM = "recording_prewarm"
    RECORDING_PREVIEW = "recording_preview"
    EVENT_CLIP = "event_clip"
    MEDIA_EXPORT = "media_export"


class MediaResourceClass(StrEnum):
    RELAY = "relay"
    JPEG_ENCODER = "jpeg_encoder"
    CAPTURE_OPEN = "capture_open"
    CAPTURE_PROCESS = "capture_process"
    REMUX_PROCESS = "remux_process"
    TRANSCODE_PROCESS = "transcode_process"
    EXPORT_WORKER = "export_worker"


@dataclass(frozen=True, slots=True)
class MediaSessionRequest:
    kind: MediaSessionKind
    camera_id: str | None = None
    source: str | None = None
    resources: Mapping[MediaResourceClass, int] = field(default_factory=dict)
    owner_generation: str | int | None = None
    correlation_id: str | None = None
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[MediaResourceClass, int] = {}
        for resource, amount in self.resources.items():
            value = int(amount)
            if value <= 0:
                raise ValueError("media resource amounts must be positive")
            normalized[MediaResourceClass(resource)] = value
        object.__setattr__(self, "resources", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class MediaAdmissionPolicy:
    global_limits: Mapping[MediaResourceClass, int] = field(default_factory=dict)
    per_camera_limits: Mapping[MediaResourceClass, int] = field(default_factory=dict)
    kind_limits: Mapping[MediaSessionKind, int] = field(default_factory=dict)

    @classmethod
    def enforced_defaults(cls) -> MediaAdmissionPolicy:
        return cls(
            global_limits={
                MediaResourceClass.RELAY: 12,
                MediaResourceClass.JPEG_ENCODER: 8,
                MediaResourceClass.CAPTURE_OPEN: 2,
                MediaResourceClass.REMUX_PROCESS: 2,
                MediaResourceClass.TRANSCODE_PROCESS: 2,
                MediaResourceClass.EXPORT_WORKER: 1,
            },
            per_camera_limits={
                MediaResourceClass.RELAY: 6,
                MediaResourceClass.JPEG_ENCODER: 2,
                MediaResourceClass.TRANSCODE_PROCESS: 2,
            },
        )


class MediaSessionAdmissionError(RuntimeError):
    pass


class MediaCancellation:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[str], None]] = []
        self._reason = ""
        self._cancelled_at = 0.0

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    @property
    def cancelled_at(self) -> float:
        with self._lock:
            return self._cancelled_at

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def add_callback(self, callback: Callable[[str], None]) -> None:
        reason = ""
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(callback)
                return
            reason = self._reason
        callback(reason)

    def cancel(self, reason: str) -> bool:
        callbacks: list[Callable[[str], None]]
        normalized = str(reason or "cancelled")
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = normalized
            self._cancelled_at = time.monotonic()
            callbacks = self._callbacks
            self._callbacks = []
            self._event.set()
        for callback in callbacks:
            try:
                callback(normalized)
            except Exception:
                # Cancellation is best effort; the owner still observes the
                # token and retains responsibility for resource teardown.
                pass
        return True


@dataclass(slots=True)
class _SessionRecord:
    request: MediaSessionRequest
    cancellation: MediaCancellation
    acquired_at: float
    phase: str = "admitted"
    bytes_in: int = 0
    bytes_out: int = 0
    frames: int = 0
    pid: int | None = None
    cancel_requested_at: float = 0.0


class MediaSessionLease:
    def __init__(
        self,
        manager: MediaSessionManager,
        session_id: str,
        request: MediaSessionRequest,
        cancellation: MediaCancellation,
    ) -> None:
        self._manager = manager
        self.id = session_id
        self.request = request
        self.cancellation = cancellation
        self._closed = False
        self._close_lock = threading.Lock()

    def set_phase(self, phase: str) -> None:
        self._manager._update(self.id, phase=str(phase)[:80])

    def attach_process(self, pid: int) -> None:
        self._manager._update(self.id, pid=max(1, int(pid)))

    def add_usage(
        self,
        *,
        bytes_in: int = 0,
        bytes_out: int = 0,
        frames: int = 0,
    ) -> None:
        self._manager._add_usage(
            self.id,
            bytes_in=max(0, int(bytes_in)),
            bytes_out=max(0, int(bytes_out)),
            frames=max(0, int(frames)),
        )

    def cancelled(self) -> bool:
        return self.cancellation.is_set()

    def close(self, outcome: str = "completed", error: str = "") -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._manager._release(
            self.id,
            outcome=str(outcome or "completed"),
            error=str(error or "")[:160],
        )

    def __enter__(self) -> MediaSessionLease:
        return self

    def __exit__(self, exc_type, exc, _traceback) -> None:
        self.close(
            "failed" if exc is not None else "completed",
            str(exc or ""),
        )


class MediaSessionManager:
    def __init__(
        self,
        policy: MediaAdmissionPolicy | None = None,
    ) -> None:
        self.policy = policy or MediaAdmissionPolicy.enforced_defaults()
        self._condition = threading.Condition(threading.Lock())
        self._sessions: dict[str, _SessionRecord] = {}
        self._resource_usage: Counter[MediaResourceClass] = Counter()
        self._camera_resource_usage: dict[
            str, Counter[MediaResourceClass]
        ] = {}
        self._kind_usage: Counter[MediaSessionKind] = Counter()
        self._peaks: Counter[str] = Counter()
        self._counters: Counter[str] = Counter()
        self._recent: deque[dict[str, object]] = deque(maxlen=64)
        self._accepting = True

    def acquire(
        self,
        request: MediaSessionRequest,
        *,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> MediaSessionLease:
        started = time.monotonic()
        deadline = None if timeout is None else started + max(0.0, timeout)
        with self._condition:
            while True:
                if not self._accepting:
                    self._counters["rejected"] += 1
                    raise MediaSessionAdmissionError(
                        "media session admission is closed"
                    )
                reason = self._capacity_error(request)
                if not reason:
                    return self._admit_locked(request, started)
                if not blocking:
                    self._record_rejection_locked(request, reason)
                    raise MediaSessionAdmissionError(reason)
                remaining = (
                    None
                    if deadline is None
                    else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    self._record_rejection_locked(request, reason)
                    raise MediaSessionAdmissionError(reason)
                self._counters["waits"] += 1
                self._condition.wait(remaining)

    def cancel_camera(
        self,
        camera_id: str,
        reason: str,
        *,
        kinds: set[MediaSessionKind] | None = None,
    ) -> int:
        return self._cancel_matching(
            lambda record: (
                record.request.camera_id == camera_id
                and (kinds is None or record.request.kind in kinds)
            ),
            reason,
        )

    def cancel_generation(self, generation: str | int, reason: str) -> int:
        return self._cancel_matching(
            lambda record: record.request.owner_generation == generation,
            reason,
        )

    def cancel_all(self, reason: str, *, close_admission: bool = False) -> int:
        if close_admission:
            with self._condition:
                self._accepting = False
                self._condition.notify_all()
        return self._cancel_matching(lambda _record: True, reason)

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._sessions:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def snapshot(self, *, include_sessions: bool = False) -> dict[str, object]:
        now = time.monotonic()
        with self._condition:
            by_kind = {
                kind.value: {
                    "active": int(count),
                    "limit": self.policy.kind_limits.get(kind),
                    "peak": int(self._peaks[f"kind:{kind.value}"]),
                }
                for kind, count in self._kind_usage.items()
            }
            by_resource = {
                resource.value: {
                    "used": int(self._resource_usage.get(resource, 0)),
                    "limit": self.policy.global_limits.get(resource),
                    "peak": int(self._peaks[f"resource:{resource.value}"]),
                }
                for resource in set(self.policy.global_limits) | set(self._resource_usage)
            }
            cameras: dict[str, dict[str, object]] = {}
            for camera_id, resources in self._camera_resource_usage.items():
                sessions = [
                    record
                    for record in self._sessions.values()
                    if record.request.camera_id == camera_id
                ]
                cameras[camera_id] = {
                    "active": len(sessions),
                    "resources": {
                        key.value: int(value)
                        for key, value in resources.items()
                        if value
                    },
                }
            ages = [
                max(0.0, now - record.acquired_at)
                for record in self._sessions.values()
            ]
            result: dict[str, object] = {
                "mode": "enforce",
                "accepting": self._accepting,
                "active": len(self._sessions),
                "oldest_active_seconds": round(max(ages, default=0.0), 3),
                "by_kind": by_kind,
                "by_resource": by_resource,
                "cameras": cameras,
                "counters": dict(self._counters),
                "recent": list(self._recent),
            }
            if include_sessions:
                result["sessions"] = [
                    {
                        "id": session_id,
                        "kind": record.request.kind.value,
                        "camera_id": record.request.camera_id,
                        "source": record.request.source,
                        "generation": record.request.owner_generation,
                        "phase": record.phase,
                        "age_seconds": round(
                            max(0.0, now - record.acquired_at),
                            3,
                        ),
                        "cancelled": record.cancellation.is_set(),
                        "bytes_in": record.bytes_in,
                        "bytes_out": record.bytes_out,
                        "frames": record.frames,
                        "pid": record.pid,
                    }
                    for session_id, record in self._sessions.items()
                ]
            return result

    def _admit_locked(
        self,
        request: MediaSessionRequest,
        started: float,
    ) -> MediaSessionLease:
        session_id = uuid.uuid4().hex
        cancellation = MediaCancellation()
        self._sessions[session_id] = _SessionRecord(
            request=request,
            cancellation=cancellation,
            acquired_at=time.monotonic(),
        )
        self._kind_usage[request.kind] += 1
        self._peaks[f"kind:{request.kind.value}"] = max(
            self._peaks[f"kind:{request.kind.value}"],
            self._kind_usage[request.kind],
        )
        camera_resources = (
            self._camera_resource_usage.setdefault(
                request.camera_id,
                Counter(),
            )
            if request.camera_id
            else None
        )
        for resource, amount in request.resources.items():
            self._resource_usage[resource] += amount
            self._peaks[f"resource:{resource.value}"] = max(
                self._peaks[f"resource:{resource.value}"],
                self._resource_usage[resource],
            )
            if camera_resources is not None:
                camera_resources[resource] += amount
        self._counters["admitted"] += 1
        waited_ms = max(0.0, (time.monotonic() - started) * 1000.0)
        self._counters["admission_wait_ms_total"] += round(waited_ms)
        return MediaSessionLease(self, session_id, request, cancellation)

    def _capacity_error(self, request: MediaSessionRequest) -> str:
        kind_limit = self.policy.kind_limits.get(request.kind)
        if kind_limit is not None and self._kind_usage[request.kind] >= kind_limit:
            return f"{request.kind.value} session capacity is exhausted"
        camera_usage = (
            self._camera_resource_usage.get(request.camera_id, Counter())
            if request.camera_id
            else Counter()
        )
        for resource, amount in request.resources.items():
            global_limit = self.policy.global_limits.get(resource)
            if (
                global_limit is not None
                and self._resource_usage[resource] + amount > global_limit
            ):
                return f"{resource.value} capacity is exhausted"
            camera_limit = self.policy.per_camera_limits.get(resource)
            if (
                request.camera_id
                and camera_limit is not None
                and camera_usage[resource] + amount > camera_limit
            ):
                return (
                    f"{resource.value} capacity is exhausted for camera "
                    f"{request.camera_id}"
                )
        return ""

    def _record_rejection_locked(
        self,
        request: MediaSessionRequest,
        reason: str,
    ) -> None:
        self._counters["rejected"] += 1
        self._recent.append(
            {
                "kind": request.kind.value,
                "camera_id": request.camera_id,
                "outcome": "rejected",
                "reason": reason[:160],
            }
        )

    def _cancel_matching(
        self,
        predicate: Callable[[_SessionRecord], bool],
        reason: str,
    ) -> int:
        records: list[_SessionRecord] = []
        with self._condition:
            for record in self._sessions.values():
                if predicate(record) and not record.cancellation.is_set():
                    record.cancel_requested_at = time.monotonic()
                    records.append(record)
        count = 0
        for record in records:
            if record.cancellation.cancel(reason):
                count += 1
        with self._condition:
            self._counters["cancel_requested"] += count
        return count

    def _update(self, session_id: str, **values: object) -> None:
        with self._condition:
            record = self._sessions.get(session_id)
            if record is None:
                return
            for key, value in values.items():
                setattr(record, key, value)

    def _add_usage(
        self,
        session_id: str,
        *,
        bytes_in: int,
        bytes_out: int,
        frames: int,
    ) -> None:
        with self._condition:
            record = self._sessions.get(session_id)
            if record is None:
                return
            record.bytes_in += bytes_in
            record.bytes_out += bytes_out
            record.frames += frames

    def _release(self, session_id: str, *, outcome: str, error: str) -> None:
        with self._condition:
            record = self._sessions.pop(session_id, None)
            if record is None:
                return
            self._kind_usage[record.request.kind] -= 1
            if not self._kind_usage[record.request.kind]:
                self._kind_usage.pop(record.request.kind, None)
            camera_resources = (
                self._camera_resource_usage.get(record.request.camera_id)
                if record.request.camera_id
                else None
            )
            for resource, amount in record.request.resources.items():
                self._resource_usage[resource] -= amount
                if not self._resource_usage[resource]:
                    self._resource_usage.pop(resource, None)
                if camera_resources is not None:
                    camera_resources[resource] -= amount
                    if not camera_resources[resource]:
                        camera_resources.pop(resource, None)
            if (
                record.request.camera_id
                and camera_resources is not None
                and not camera_resources
            ):
                self._camera_resource_usage.pop(record.request.camera_id, None)
            normalized_outcome = (
                "cancelled"
                if record.cancellation.is_set() and outcome == "completed"
                else outcome
            )
            self._counters[normalized_outcome] += 1
            self._recent.append(
                {
                    "kind": record.request.kind.value,
                    "camera_id": record.request.camera_id,
                    "outcome": normalized_outcome[:40],
                    "duration_seconds": round(
                        max(0.0, time.monotonic() - record.acquired_at),
                        3,
                    ),
                    "error": error,
                }
            )
            self._condition.notify_all()

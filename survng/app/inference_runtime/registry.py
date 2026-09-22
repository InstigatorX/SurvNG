from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
import hashlib
import json
import threading
import time
from typing import Any, Protocol
import uuid

import numpy as np

from ..config import DetectorConfig
from .protocol import (
    INFERENCE_PROTOCOL_VERSION,
    WorkerHeartbeat,
    WorkerReady,
    WorkerRegistration,
    WorkerRole,
    decode_packet,
    encode_packet,
)
from .types import InferenceUnavailable, InferenceWorkload


class RegistryTransport(Protocol):
    def request(self, packet: bytes, timeout: float) -> bytes: ...

    def close(self, reason: str) -> None: ...


def detector_config_generation(config: DetectorConfig) -> str:
    payload = config.model_dump(mode="json")
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class _WorkerLease:
    registration: WorkerRegistration
    transport: RegistryTransport
    connection_generation: int
    lease_expires_at: float
    config_generation: str = ""
    statuses: dict[str, Any] = field(default_factory=dict)
    ready: bool = False
    pending_requests: int = 0
    reported_pending_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    last_seen_at: float = field(default_factory=time.monotonic)


class RemoteInferenceRegistry:
    """Process-scoped registry for replaceable remote inference workers."""

    def __init__(
        self,
        config_provider: Callable[[], DetectorConfig],
        *,
        lease_seconds: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config_provider = config_provider
        self._lease_seconds = max(5.0, float(lease_seconds))
        self._clock = clock
        self._lock = threading.RLock()
        self._workers: dict[str, _WorkerLease] = {}
        self._generations: dict[str, int] = {}
        self._route_cursor: dict[str, int] = {}
        self._accepting = True

    def register(
        self,
        registration: WorkerRegistration,
        transport: RegistryTransport,
    ) -> dict[str, Any]:
        if registration.protocol_version != INFERENCE_PROTOCOL_VERSION:
            raise InferenceUnavailable(
                "remote worker protocol version is incompatible"
            )
        previous: _WorkerLease | None = None
        with self._lock:
            if not self._accepting:
                raise InferenceUnavailable("remote worker registry is stopping")
            generation = self._generations.get(registration.worker_id, 0) + 1
            self._generations[registration.worker_id] = generation
            previous = self._workers.get(registration.worker_id)
            now = self._clock()
            self._workers[registration.worker_id] = _WorkerLease(
                registration=registration,
                transport=transport,
                connection_generation=generation,
                lease_expires_at=now + self._lease_seconds,
                last_seen_at=now,
            )
            config = self._config_provider().model_copy(deep=True)
        if previous is not None:
            previous.transport.close("worker connection was replaced")
        return {
            "type": "welcome",
            "protocol_version": INFERENCE_PROTOCOL_VERSION,
            "connection_generation": generation,
            "heartbeat_seconds": max(1.0, self._lease_seconds / 3.0),
            "lease_seconds": self._lease_seconds,
            "config_generation": detector_config_generation(config),
            "detector_config": config.model_dump(mode="json"),
        }

    def mark_ready(self, ready: WorkerReady) -> bool:
        with self._lock:
            worker = self._current_worker(
                ready.worker_id,
                ready.connection_generation,
            )
            if worker is None:
                return False
            expected = detector_config_generation(self._config_provider())
            worker.config_generation = ready.config_generation
            worker.statuses = dict(ready.statuses)
            worker.ready = ready.config_generation == expected
            self._renew(worker)
            return worker.ready

    def heartbeat(self, heartbeat: WorkerHeartbeat) -> bool:
        with self._lock:
            worker = self._current_worker(
                heartbeat.worker_id,
                heartbeat.connection_generation,
            )
            if worker is None:
                return False
            worker.reported_pending_requests = heartbeat.pending_requests
            self._renew(worker)
            return True

    def unregister(self, worker_id: str, connection_generation: int) -> None:
        with self._lock:
            worker = self._current_worker(worker_id, connection_generation)
            if worker is not None:
                self._workers.pop(worker_id, None)

    def request(
        self,
        role: WorkerRole,
        operation: str,
        *,
        frame: np.ndarray | None,
        workload: InferenceWorkload,
        timeout: float,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        if timeout <= 0:
            raise InferenceUnavailable("remote inference deadline expired")
        worker = self._select_worker(role)
        if worker is None:
            raise InferenceUnavailable(
                f"no compatible remote {role} inference worker is ready"
            )
        request_id = uuid.uuid4().hex
        message = {
            "type": "request",
            "protocol_version": INFERENCE_PROTOCOL_VERSION,
            "request_id": request_id,
            "connection_generation": worker.connection_generation,
            "config_generation": worker.config_generation,
            "role": role,
            "operation": operation,
            "workload": int(workload),
            "deadline_unix_ms": int((time.time() + timeout) * 1000),
            "payload": dict(payload or {}),
        }
        packet = encode_packet(message, frame=frame)
        started = self._clock()
        try:
            response_packet = worker.transport.request(packet, timeout)
            response, trailing = decode_packet(response_packet)
            if trailing:
                raise InferenceUnavailable(
                    "remote inference response contained unexpected frame bytes"
                )
            if (
                response.get("type") != "response"
                or response.get("request_id") != request_id
                or int(response.get("connection_generation") or 0)
                != worker.connection_generation
            ):
                raise InferenceUnavailable(
                    "remote inference response did not match the active request"
                )
            if not response.get("ok"):
                raise InferenceUnavailable(
                    str(response.get("error") or "remote inference failed")
                )
            with self._lock:
                current = self._current_worker(
                    worker.registration.worker_id,
                    worker.connection_generation,
                )
                if current is not None:
                    current.completed_requests += 1
                    self._renew(current)
            return response.get("result")
        except (FutureTimeoutError, TimeoutError) as error:
            self._record_failure(worker)
            raise InferenceUnavailable(
                f"remote {role} {operation} timed out"
            ) from error
        except InferenceUnavailable:
            self._record_failure(worker)
            raise
        except Exception as error:
            self._record_failure(worker)
            raise InferenceUnavailable(
                f"remote {role} {operation} transport failed"
            ) from error
        finally:
            with self._lock:
                current = self._current_worker(
                    worker.registration.worker_id,
                    worker.connection_generation,
                )
                if current is not None:
                    current.pending_requests = max(
                        0,
                        current.pending_requests - 1,
                    )
                    current.last_seen_at = max(
                        current.last_seen_at,
                        started,
                    )

    def has_ready_worker(self, role: WorkerRole) -> bool:
        return self._select_worker(role, reserve=False) is not None

    def status(self) -> dict[str, Any]:
        expired = self._prune_expired()
        for worker in expired:
            worker.transport.close("worker lease expired")
        now = self._clock()
        with self._lock:
            workers = [
                {
                    "worker_id": worker.registration.worker_id,
                    "name": worker.registration.name,
                    "roles": list(worker.registration.roles),
                    "slots": worker.registration.slots,
                    "devices": list(worker.registration.devices),
                    "connection_generation": worker.connection_generation,
                    "config_generation": worker.config_generation,
                    "ready": worker.ready,
                    "pending_requests": worker.pending_requests,
                    "reported_pending_requests": (
                        worker.reported_pending_requests
                    ),
                    "completed_requests": worker.completed_requests,
                    "failed_requests": worker.failed_requests,
                    "lease_remaining_seconds": round(
                        max(0.0, worker.lease_expires_at - now),
                        2,
                    ),
                    "statuses": dict(worker.statuses),
                }
                for worker in sorted(
                    self._workers.values(),
                    key=lambda item: item.registration.worker_id,
                )
            ]
            return {
                "accepting": self._accepting,
                "connected": len(workers),
                "ready": sum(bool(item["ready"]) for item in workers),
                "workers": workers,
            }

    def close(self) -> None:
        with self._lock:
            self._accepting = False
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.transport.close("remote worker registry stopped")

    def _select_worker(
        self,
        role: WorkerRole,
        *,
        reserve: bool = True,
    ) -> _WorkerLease | None:
        expired = self._prune_expired()
        for worker in expired:
            worker.transport.close("worker lease expired")
        expected_generation = detector_config_generation(self._config_provider())
        with self._lock:
            candidates = [
                worker
                for worker in self._workers.values()
                if (
                    worker.ready
                    and role in worker.registration.roles
                    and worker.config_generation == expected_generation
                )
            ]
            if not candidates:
                return None
            candidates.sort(
                key=lambda item: (
                    item.pending_requests + item.reported_pending_requests,
                    item.registration.worker_id,
                )
            )
            minimum = (
                candidates[0].pending_requests
                + candidates[0].reported_pending_requests
            )
            equal = [
                worker
                for worker in candidates
                if (
                    worker.pending_requests
                    + worker.reported_pending_requests
                    == minimum
                )
            ]
            cursor = self._route_cursor.get(role, 0) % len(equal)
            worker = equal[cursor]
            self._route_cursor[role] = cursor + 1
            if reserve:
                worker.pending_requests += 1
            return worker

    def _prune_expired(self) -> list[_WorkerLease]:
        now = self._clock()
        with self._lock:
            expired = [
                worker
                for worker in self._workers.values()
                if worker.lease_expires_at <= now
            ]
            for worker in expired:
                current = self._workers.get(worker.registration.worker_id)
                if current is worker:
                    self._workers.pop(worker.registration.worker_id, None)
            return expired

    def _record_failure(self, worker: _WorkerLease) -> None:
        with self._lock:
            current = self._current_worker(
                worker.registration.worker_id,
                worker.connection_generation,
            )
            if current is not None:
                current.failed_requests += 1

    def _current_worker(
        self,
        worker_id: str,
        connection_generation: int,
    ) -> _WorkerLease | None:
        worker = self._workers.get(worker_id)
        if (
            worker is None
            or worker.connection_generation != connection_generation
        ):
            return None
        return worker

    def _renew(self, worker: _WorkerLease) -> None:
        now = self._clock()
        worker.last_seen_at = now
        worker.lease_expires_at = now + self._lease_seconds

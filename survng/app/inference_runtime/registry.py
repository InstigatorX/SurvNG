from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
import hashlib
import json
import threading
import time
from typing import Any, Protocol
import uuid

import numpy as np

from ..config import DetectorConfig, detector_routing_payload
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
    encoded = json.dumps(
        detector_routing_payload(config),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _worker_weight(
    worker_id: str,
    weights: Mapping[str, int] | None,
    default_weight: int,
) -> int:
    if weights is not None and worker_id in weights:
        return int(weights[worker_id])
    return int(default_weight)


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
        generation_provider: Callable[[DetectorConfig], str] = (
            detector_config_generation
        ),
    ) -> None:
        self._config_provider = config_provider
        self._generation_provider = generation_provider
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
        *,
        config: DetectorConfig | None = None,
        config_generation: str = "",
        initial_lease_seconds: float | None = None,
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
            initial_lease = max(
                self._lease_seconds,
                float(initial_lease_seconds or self._lease_seconds),
            )
            self._workers[registration.worker_id] = _WorkerLease(
                registration=registration,
                transport=transport,
                connection_generation=generation,
                lease_expires_at=now + initial_lease,
                last_seen_at=now,
            )
            active_config = (
                config.model_copy(deep=True)
                if config is not None
                else self._config_provider().model_copy(deep=True)
            )
        if previous is not None:
            previous.transport.close("worker connection was replaced")
        return {
            "type": "welcome",
            "protocol_version": INFERENCE_PROTOCOL_VERSION,
            "connection_generation": generation,
            "heartbeat_seconds": max(1.0, self._lease_seconds / 3.0),
            "lease_seconds": self._lease_seconds,
            "config_generation": (
                config_generation
                or self._generation_provider(active_config)
            ),
            "detector_config": active_config.model_dump(mode="json"),
        }

    def config_snapshot(self) -> DetectorConfig:
        with self._lock:
            return self._config_provider().model_copy(deep=True)

    def mark_ready(self, ready: WorkerReady) -> bool:
        with self._lock:
            worker = self._current_worker(
                ready.worker_id,
                ready.connection_generation,
            )
            if worker is None:
                return False
            expected = self._generation_provider(self._config_provider())
            worker.config_generation = ready.config_generation
            worker.statuses = dict(ready.statuses)
            accepted = ready.config_generation == expected
            worker.ready = (
                accepted
                and any(
                    self._role_ready(worker, role)
                    for role in worker.registration.roles
                )
            )
            self._renew(worker)
            return accepted

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
        worker_weights: Mapping[str, int] | None = None,
        default_worker_weight: int = 1,
    ) -> Any:
        if timeout <= 0:
            raise InferenceUnavailable("remote inference deadline expired")
        worker = self._select_worker(
            role,
            weights=worker_weights,
            default_weight=default_worker_weight,
        )
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

    def config_generation(self) -> str:
        return self._generation_provider(self._config_provider())

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

    def start(self, *, lease_seconds: float | None = None) -> None:
        with self._lock:
            if lease_seconds is not None:
                self._lease_seconds = max(5.0, float(lease_seconds))
            self._accepting = True

    def ready_role_loads(
        self,
        role: WorkerRole,
        *,
        weights: Mapping[str, int] | None = None,
        default_weight: int = 1,
    ) -> list[dict[str, Any]]:
        """Return ready workers for a role without reserving a request."""
        expired = self._prune_expired()
        for worker in expired:
            worker.transport.close("worker lease expired")
        expected_generation = self._generation_provider(self._config_provider())
        with self._lock:
            loads: list[dict[str, Any]] = []
            for worker in self._workers.values():
                if not self._worker_matches(worker, role, expected_generation):
                    continue
                weight = _worker_weight(
                    worker.registration.worker_id,
                    weights,
                    default_weight,
                )
                if weights is not None and weight <= 0:
                    continue
                loads.append({
                    "worker_id": worker.registration.worker_id,
                    "name": worker.registration.name,
                    "pending": (
                        worker.pending_requests
                        + worker.reported_pending_requests
                    ),
                    "weight": weight,
                })
            return loads

    def _select_worker(
        self,
        role: WorkerRole,
        *,
        reserve: bool = True,
        weights: Mapping[str, int] | None = None,
        default_weight: int = 1,
    ) -> _WorkerLease | None:
        expired = self._prune_expired()
        for worker in expired:
            worker.transport.close("worker lease expired")
        expected_generation = self._generation_provider(self._config_provider())
        with self._lock:
            scored: list[tuple[float, int, str, _WorkerLease]] = []
            for worker in self._workers.values():
                if not self._worker_matches(worker, role, expected_generation):
                    continue
                pending = (
                    worker.pending_requests
                    + worker.reported_pending_requests
                )
                if weights is None:
                    score = float(pending)
                else:
                    weight = _worker_weight(
                        worker.registration.worker_id,
                        weights,
                        default_weight,
                    )
                    if weight <= 0:
                        continue
                    score = (pending + 1) / weight
                scored.append((
                    score,
                    pending,
                    worker.registration.worker_id,
                    worker,
                ))
            if not scored:
                return None
            scored.sort(key=lambda item: (item[0], item[1], item[2]))
            best_score = scored[0][0]
            equal = [item for item in scored if item[0] == best_score]
            cursor = self._route_cursor.get(role, 0) % len(equal)
            worker = equal[cursor][3]
            self._route_cursor[role] = cursor + 1
            if reserve:
                worker.pending_requests += 1
            return worker

    @staticmethod
    def _worker_matches(
        worker: _WorkerLease,
        role: WorkerRole,
        expected_generation: str,
    ) -> bool:
        return bool(
            worker.ready
            and role in worker.registration.roles
            and RemoteInferenceRegistry._role_ready(worker, role)
            and worker.config_generation == expected_generation
        )

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

    @staticmethod
    def _role_ready(worker: _WorkerLease, role: WorkerRole) -> bool:
        status = worker.statuses.get(role)
        if not isinstance(status, dict) or status.get("enabled") is False:
            return False
        if "ready" in status:
            return bool(status.get("ready"))
        # Object detector status reports the loaded backend instead of a
        # ready flag. Face, ReID, and depth always include ready.
        if role != "object":
            return False
        loaded = bool(
            status.get("openvino_loaded")
            or status.get("opencv_loaded")
            or status.get("coreml_loaded")
            or status.get("loaded_backend")
        )
        isolation = status.get("isolation")
        if not isinstance(isolation, dict):
            return loaded
        if isolation.get("all_workers_alive") is False:
            return False
        if isolation.get("worker_alive") is False:
            return False
        return loaded

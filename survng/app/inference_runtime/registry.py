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
    WorkerUpgradeStatus,
    WorkerRegistration,
    WorkerRole,
    ProtocolError,
    decode_packet,
    encode_packet,
)
from .types import InferenceNotAdmitted, InferenceUnavailable, InferenceWorkload


class RegistryTransport(Protocol):
    def request(
        self,
        packet: bytes,
        timeout: float,
        *,
        admit_timeout: float | None = None,
    ) -> bytes: ...

    def close(self, reason: str) -> None: ...

    def send_control(self, message: dict[str, Any]) -> None: ...


def detector_config_generation(config: DetectorConfig) -> str:
    encoded = json.dumps(
        detector_routing_payload(config),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_ms(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number < 0 or number != number or number in {float("inf"), float("-inf")}:
        return None
    return round(number, 1)


def _bump_role_attempt(worker: _WorkerLease, role: str, outcome: str, delta: int = 1) -> None:
    if not role:
        return
    bucket = worker.role_attempts.setdefault(
        role,
        {"completed": 0, "rerouted": 0, "failed": 0},
    )
    bucket[outcome] = max(0, int(bucket.get(outcome) or 0) + delta)


def _worker_backlog(worker: _WorkerLease) -> int:
    """Busy work is the larger of the primary reservation and the worker queue.

    Adding them double-counts a request the primary is still waiting for.
    The worker queue also includes requests whose primary wait already ended.
    """
    return max(worker.pending_requests, worker.reported_pending_requests)


def _worker_weight(
    worker_id: str,
    weights: Mapping[str, int] | None,
    default_weight: int,
) -> int:
    if weights is not None and worker_id in weights:
        return int(weights[worker_id])
    return int(default_weight)


def _worker_inference_ms(worker: _WorkerLease) -> float | None:
    if worker.inference_samples <= 0:
        return None
    return round(worker.inference_ms_total / worker.inference_samples, 1)


def _worker_placement_ms(worker: _WorkerLease) -> float | None:
    """Round trip once measured; model time until then.

    A non-positive request sample is ignored. Frozen test clocks report 0
    and must not override a real model-time measurement.
    """
    if worker.request_samples > 0:
        return round(worker.request_ms_total / worker.request_samples, 1)
    return _worker_inference_ms(worker)


def _latency_fallback(samples: list[float | None]) -> float:
    """Missing samples use the slowest known time so they are not treated as instant."""
    known = [sample for sample in samples if sample is not None and sample > 0]
    if not known:
        return 1.0
    return max(known)


def _placement_score(
    pending: int,
    weight: int,
    latency_ms: float | None,
    fallback_ms: float,
) -> float:
    latency = fallback_ms if latency_ms is None or latency_ms <= 0 else latency_ms
    return (max(0, pending) + 1) * latency / max(1, weight)


def _deadline_miss(message: str) -> bool:
    text = message.lower()
    return "timed out" in text or "deadline expired" in text


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
    rerouted_requests: int = 0
    failed_requests: int = 0
    last_outcome: str = ""
    last_error: str = ""
    last_role: str = ""
    last_operation: str = ""
    last_inference_ms: float | None = None
    last_request_ms: float | None = None
    inference_ms_total: float = 0.0
    inference_samples: int = 0
    request_ms_total: float = 0.0
    request_samples: int = 0
    role_attempts: dict[str, dict[str, int]] = field(default_factory=dict)
    upgrade_phase: str = ""
    upgrade_detail: str = ""
    upgrade_target_sha: str = ""
    routing_hold: bool = False
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

    def request_upgrade(self, worker_id: str, target_sha: str) -> dict[str, Any]:
        """Ask one connected worker to check out the primary commit and restart."""
        sha = str(target_sha or "").strip().lower()
        if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
            raise InferenceUnavailable("upgrade target must be a full git commit")
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                raise InferenceUnavailable("inference worker is not connected")
            if not worker.ready:
                raise InferenceUnavailable("inference worker is not ready")
            generation = worker.connection_generation
            transport = worker.transport
            worker.upgrade_phase = "requested"
            worker.upgrade_target_sha = sha
            worker.upgrade_detail = ""
        try:
            transport.send_control({
                "type": "upgrade",
                "worker_id": worker_id,
                "connection_generation": generation,
                "target_sha": sha,
            })
        except Exception as error:
            with self._lock:
                current = self._current_worker(worker_id, generation)
                if current is not None:
                    current.upgrade_phase = "failed"
                    current.upgrade_detail = "upgrade request was not sent"
            raise InferenceUnavailable(
                "upgrade request was not sent"
            ) from error
        return self._upgrade_status(worker_id)

    def note_upgrade(self, status: WorkerUpgradeStatus) -> None:
        with self._lock:
            worker = self._current_worker(
                status.worker_id,
                status.connection_generation,
            )
            if worker is None:
                return
            worker.upgrade_phase = status.phase
            worker.upgrade_detail = status.detail
            if status.target_sha:
                worker.upgrade_target_sha = status.target_sha

    def _upgrade_status(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return {}
            return {
                "worker_id": worker_id,
                "software_version": worker.registration.software_version,
                "upgrade_phase": worker.upgrade_phase,
                "upgrade_detail": worker.upgrade_detail,
                "upgrade_target_sha": worker.upgrade_target_sha,
            }

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
        admit_timeout: float | None = None,
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
        now = time.time()
        message = {
            "type": "request",
            "protocol_version": INFERENCE_PROTOCOL_VERSION,
            "request_id": request_id,
            "connection_generation": worker.connection_generation,
            "config_generation": worker.config_generation,
            "role": role,
            "operation": operation,
            "workload": int(workload),
            "deadline_unix_ms": int((now + timeout) * 1000),
            "payload": dict(payload or {}),
        }
        separate_admission = (
            admit_timeout is not None and 0 <= admit_timeout < timeout
        )
        if separate_admission:
            message["admit_unix_ms"] = int((now + float(admit_timeout)) * 1000)
        started = self._clock()
        sent = False
        try:
            packet = encode_packet(message, frame=frame)
            sent = True
            if separate_admission:
                response_packet = worker.transport.request(
                    packet,
                    timeout,
                    admit_timeout=admit_timeout,
                )
            else:
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
            self._record_success(
                worker,
                role=role,
                operation=operation,
                inference_ms=_finite_ms(response.get("inference_ms")),
                request_ms=_finite_ms((self._clock() - started) * 1000.0),
            )
            return response.get("result")
        except ProtocolError as error:
            if not sent:
                raise
            failure = self._fail_attempt(
                worker,
                role,
                operation,
                f"remote {role} {operation} transport failed",
            )
            raise failure from error
        except InferenceNotAdmitted as error:
            failure = self._fail_attempt(
                worker,
                role,
                operation,
                f"remote {role} {operation} was not admitted",
            )
            raise failure from error
        except (FutureTimeoutError, TimeoutError) as error:
            failure = self._fail_attempt(
                worker,
                role,
                operation,
                f"remote {role} {operation} timed out",
            )
            raise failure from error
        except InferenceUnavailable as error:
            self._fail_attempt(
                worker,
                role,
                operation,
                str(error),
                error=error,
            )
            raise
        except Exception as error:
            failure = self._fail_attempt(
                worker,
                role,
                operation,
                f"remote {role} {operation} transport failed",
            )
            raise failure from error
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
                    "software_version": worker.registration.software_version,
                    "upgrade_phase": worker.upgrade_phase,
                    "upgrade_detail": worker.upgrade_detail,
                    "upgrade_target_sha": worker.upgrade_target_sha,
                    "routing_hold": self._routing_held(worker),
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
                    "rerouted_requests": worker.rerouted_requests,
                    "failed_requests": worker.failed_requests,
                    "last_outcome": worker.last_outcome,
                    "last_error": worker.last_error,
                    "last_role": worker.last_role,
                    "last_operation": worker.last_operation,
                    "last_inference_ms": worker.last_inference_ms,
                    "last_request_ms": worker.last_request_ms,
                    "average_inference_ms": (
                        round(
                            worker.inference_ms_total / worker.inference_samples,
                            1,
                        )
                        if worker.inference_samples
                        else None
                    ),
                    "role_attempts": {
                        role_name: dict(counts)
                        for role_name, counts in worker.role_attempts.items()
                    },
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
                    "pending": _worker_backlog(worker),
                    "weight": weight,
                    "inference_ms": _worker_placement_ms(worker),
                    "held": self._routing_held(worker),
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
            matched: list[tuple[_WorkerLease, int, int, float | None, bool]] = []
            for worker in self._workers.values():
                if not self._worker_matches(worker, role, expected_generation):
                    continue
                pending = _worker_backlog(worker)
                weight = _worker_weight(
                    worker.registration.worker_id,
                    weights,
                    default_weight,
                )
                if weights is not None and weight <= 0:
                    continue
                matched.append((
                    worker,
                    pending,
                    weight,
                    _worker_placement_ms(worker),
                    self._routing_held(worker),
                ))
            if not matched:
                return None
            available = [item for item in matched if not item[4]]
            pool = available or matched
            fallback_ms = _latency_fallback([item[3] for item in pool])
            scored: list[tuple[float, int, str, _WorkerLease]] = []
            for worker, pending, weight, latency_ms, _held in pool:
                if weights is None:
                    score = float(pending)
                else:
                    score = _placement_score(
                        pending,
                        weight,
                        latency_ms,
                        fallback_ms,
                    )
                scored.append((
                    score,
                    pending,
                    worker.registration.worker_id,
                    worker,
                ))
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

    def note_rerouted(self, error: BaseException) -> None:
        """Move a worker attempt from failed to rerouted after another target finishes it."""
        worker_id = getattr(error, "worker_id", None)
        generation = getattr(error, "connection_generation", None)
        role = str(getattr(error, "inference_role", "") or "")
        if not isinstance(worker_id, str) or not isinstance(generation, int):
            return
        with self._lock:
            current = self._current_worker(worker_id, generation)
            if current is None or current.failed_requests <= 0:
                return
            current.failed_requests -= 1
            current.rerouted_requests += 1
            current.last_outcome = "rerouted"
            _bump_role_attempt(current, role, "failed", -1)
            _bump_role_attempt(current, role, "rerouted")

    def _record_success(
        self,
        worker: _WorkerLease,
        *,
        role: str,
        operation: str,
        inference_ms: float | None,
        request_ms: float | None,
    ) -> None:
        with self._lock:
            current = self._current_worker(
                worker.registration.worker_id,
                worker.connection_generation,
            )
            if current is None:
                return
            current.completed_requests += 1
            current.last_outcome = "completed"
            current.last_error = ""
            current.last_role = role
            current.last_operation = operation
            if request_ms is not None:
                current.last_request_ms = request_ms
                if request_ms > 0:
                    current.request_ms_total += request_ms
                    current.request_samples += 1
            if inference_ms is not None:
                current.last_inference_ms = inference_ms
                current.inference_ms_total += inference_ms
                current.inference_samples += 1
            current.routing_hold = False
            _bump_role_attempt(current, role, "completed")
            self._renew(current)

    def _fail_attempt(
        self,
        worker: _WorkerLease,
        role: str,
        operation: str,
        message: str,
        error: InferenceUnavailable | None = None,
    ) -> InferenceUnavailable:
        text = message[:240]
        self._record_failure(
            worker,
            role=role,
            operation=operation,
            message=text,
        )
        failure = error if error is not None else InferenceUnavailable(text)
        failure.worker_id = worker.registration.worker_id  # type: ignore[attr-defined]
        failure.connection_generation = worker.connection_generation  # type: ignore[attr-defined]
        failure.inference_role = role  # type: ignore[attr-defined]
        failure.inference_operation = operation  # type: ignore[attr-defined]
        return failure

    def _record_failure(
        self,
        worker: _WorkerLease,
        *,
        role: str,
        operation: str,
        message: str,
    ) -> None:
        with self._lock:
            current = self._current_worker(
                worker.registration.worker_id,
                worker.connection_generation,
            )
            if current is None:
                return
            current.failed_requests += 1
            current.last_outcome = "failed"
            current.last_error = message
            current.last_role = role
            current.last_operation = operation
            if _deadline_miss(message):
                current.routing_hold = True
            _bump_role_attempt(current, role, "failed")

    def _routing_held(self, worker: _WorkerLease) -> bool:
        """Skip a worker that missed a deadline while earlier work is still queued."""
        if worker.routing_hold and _worker_backlog(worker) <= 0:
            worker.routing_hold = False
        return worker.routing_hold

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

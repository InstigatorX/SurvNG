from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ..config import DetectorConfig
from .registry import (
    RemoteInferenceRegistry,
    _latency_fallback,
    _placement_score,
)
from .types import (
    INFERENCE_FAILOVER_SECONDS,
    INFERENCE_REQUEST_TIMEOUT_SECONDS,
    InferenceUnavailable,
    InferenceWorkload,
)
from .worker import _InferenceWorker


class RemoteInferenceWorkerBackend:
    """One logical role slot routed through the process worker registry."""

    def __init__(
        self,
        config: DetectorConfig,
        role: str,
        initial_status: dict[str, Any],
        *,
        registry: RemoteInferenceRegistry,
        start_enabled: bool = True,
    ) -> None:
        self.config = config
        self.role = role
        self.start_enabled = start_enabled
        self._registry = registry
        self._status = dict(initial_status)
        self._started = False
        self._pending = 0
        self._lock = threading.RLock()

    def update_config_reference(self, config: DetectorConfig) -> None:
        with self._lock:
            self.config = config

    def reconfigure(
        self,
        config: DetectorConfig,
        initial_status: dict[str, Any],
        *,
        start_enabled: bool,
    ) -> None:
        with self._lock:
            self.config = config
            self.start_enabled = start_enabled
            self._status = dict(initial_status)

    def start(self) -> bool:
        with self._lock:
            self._started = True
        return True

    def stop(self) -> None:
        with self._lock:
            self._started = False

    def request(
        self,
        operation: str,
        *,
        frame: np.ndarray | None = None,
        timeout: float = INFERENCE_REQUEST_TIMEOUT_SECONDS,
        admission_timeout: float | None = None,
        workload: InferenceWorkload = InferenceWorkload.INTERACTIVE,
        worker_weights: dict[str, int] | None = None,
        default_worker_weight: int = 1,
        **payload: Any,
    ) -> Any:
        del admission_timeout
        with self._lock:
            if not self._started or not self.start_enabled:
                raise InferenceUnavailable(
                    f"remote {self.role} inference is not started"
                )
            self._pending += 1
        try:
            return self._registry.request(
                self.role,
                operation,
                frame=frame,
                workload=InferenceWorkload(workload),
                timeout=timeout,
                payload=payload,
                worker_weights=worker_weights,
                default_worker_weight=default_worker_weight,
            )
        finally:
            with self._lock:
                self._pending = max(0, self._pending - 1)

    def status(
        self,
        workload: InferenceWorkload = InferenceWorkload.OFFLINE,
    ) -> dict[str, Any]:
        del workload
        status = self.cached_status()
        status["registry"] = self._registry.status()
        return status

    def cached_status(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self._status)
            started = self._started
        status["ready"] = bool(
            started
            and self.start_enabled
            and self._registry.has_ready_worker(self.role)
        )
        status["remote"] = True
        status["isolation"] = self.isolation_status()
        return status

    def pending_requests(self) -> int:
        with self._lock:
            return self._pending

    def isolation_status(self) -> dict[str, Any]:
        ready = bool(
            self._started
            and self.start_enabled
            and self._registry.has_ready_worker(self.role)
        )
        return {
            "role": self.role,
            "remote": True,
            "worker_alive": ready,
            "worker_pid": None,
            "pending_requests": self.pending_requests(),
            "last_error": (
                ""
                if ready
                else f"No compatible remote {self.role} worker is ready."
            ),
        }


class RoutedInferenceWorkerBackend:
    """Select local, remote, or incident-fallback execution per config."""

    def __init__(
        self,
        config: DetectorConfig,
        role: str,
        initial_status: dict[str, Any],
        *,
        registry: RemoteInferenceRegistry,
        start_enabled: bool = True,
    ) -> None:
        self.config = config
        self.role = role
        self.start_enabled = start_enabled
        self._registry = registry
        self._initial_status = dict(initial_status)
        self._remote = RemoteInferenceWorkerBackend(
            config,
            role,
            initial_status,
            registry=registry,
            start_enabled=start_enabled,
        )
        self._local = (
            _InferenceWorker(
                config,
                role,
                initial_status,
                start_enabled=start_enabled,
            )
            if config.inference_mode in {"local", "hybrid"}
            else None
        )
        self._started = False
        self._balance_cursor = 0
        self._lock = threading.RLock()

    def update_config_reference(self, config: DetectorConfig) -> None:
        with self._lock:
            self.config = config
            self._remote.update_config_reference(config)
            if self._local is not None:
                self._local.update_config_reference(config)

    def reconfigure(
        self,
        config: DetectorConfig,
        initial_status: dict[str, Any],
        *,
        start_enabled: bool,
    ) -> None:
        with self._lock:
            previous_local = self._local
            need_local = config.inference_mode in {"local", "hybrid"}
            next_local = previous_local
            created_local = False
            if need_local and next_local is None:
                next_local = _InferenceWorker(
                    config,
                    self.role,
                    initial_status,
                    start_enabled=start_enabled,
                )
                created_local = True
            try:
                if next_local is not None:
                    if created_local:
                        if self._started and not next_local.start():
                            raise InferenceUnavailable(
                                f"{self.role} local fallback failed to start"
                            )
                    else:
                        next_local.reconfigure(
                            config,
                            initial_status,
                            start_enabled=start_enabled,
                        )
                self._remote.reconfigure(
                    config,
                    initial_status,
                    start_enabled=start_enabled,
                )
            except BaseException:
                if created_local and next_local is not None:
                    next_local.stop()
                raise
            if not need_local and previous_local is not None:
                previous_local.stop()
                next_local = None
            self.config = config
            self.start_enabled = start_enabled
            self._initial_status = dict(initial_status)
            self._local = next_local

    def start(self) -> bool:
        with self._lock:
            self._started = True
            remote_ready = self._remote.start()
            local_ready = (
                self._local.start()
                if self._local is not None
                else True
            )
            if self.config.inference_mode == "remote":
                return remote_ready
            return local_ready

    def stop(self) -> None:
        with self._lock:
            self._started = False
            failures: list[BaseException] = []
            for backend in (self._remote, self._local):
                if backend is None:
                    continue
                try:
                    backend.stop()
                except BaseException as error:
                    failures.append(error)
            if failures:
                first = failures[0]
                if not isinstance(first, Exception):
                    raise first
                raise RuntimeError(
                    f"{self.role} routed inference shutdown failed"
                ) from first

    def request(
        self,
        operation: str,
        *,
        frame: np.ndarray | None = None,
        timeout: float = INFERENCE_REQUEST_TIMEOUT_SECONDS,
        admission_timeout: float | None = None,
        workload: InferenceWorkload = InferenceWorkload.INTERACTIVE,
        **payload: Any,
    ) -> Any:
        mode = self.config.inference_mode
        if mode == "local" or self.config.inference_balance != "weighted":
            return self._request_remote_first(
                operation,
                frame=frame,
                timeout=timeout,
                admission_timeout=admission_timeout,
                workload=workload,
                **payload,
            )
        return self._request_weighted(
            operation,
            frame=frame,
            timeout=timeout,
            admission_timeout=admission_timeout,
            workload=workload,
            **payload,
        )

    def _request_remote_first(
        self,
        operation: str,
        *,
        frame: np.ndarray | None,
        timeout: float,
        admission_timeout: float | None,
        workload: InferenceWorkload,
        **payload: Any,
    ) -> Any:
        mode = self.config.inference_mode
        if mode == "local":
            if self._local is None:
                raise InferenceUnavailable(
                    f"local {self.role} inference is unavailable"
                )
            return self._local.request(
                operation,
                frame=frame,
                timeout=timeout,
                admission_timeout=admission_timeout,
                workload=workload,
                **payload,
            )
        failover = self._incident_fallback(workload)
        try:
            return self._run_remote(
                operation,
                frame=frame,
                timeout=timeout,
                admission_timeout=admission_timeout,
                workload=workload,
                payload=payload,
                alternate=failover,
            )
        except InferenceUnavailable as error:
            if not failover or self._local is None:
                raise
            result = self._run_local(
                operation,
                frame=frame,
                timeout=timeout,
                admission_timeout=admission_timeout,
                workload=workload,
                payload=payload,
                alternate=False,
            )
            self._registry.note_rerouted(error)
            return result

    def _request_weighted(
        self,
        operation: str,
        *,
        frame: np.ndarray | None,
        timeout: float,
        admission_timeout: float | None,
        workload: InferenceWorkload,
        **payload: Any,
    ) -> Any:
        config = self.config
        weights = dict(config.inference_worker_weights)
        default_weight = int(config.inference_default_worker_weight)
        local_weight = (
            int(config.inference_primary_weight)
            if config.inference_mode != "remote" and self._local is not None
            else 0
        )
        loads = self._usable_remote_loads(
            self._registry.ready_role_loads(
                self.role,
                weights=weights,
                default_weight=default_weight,
            ),
            local_weight,
        )
        local_pending = (
            int(self._local.pending_requests())
            if local_weight > 0 and self._local is not None
            else 0
        )
        local_latency = self._local_inference_ms()
        kind = self._choose_weighted_target(
            local_weight=local_weight,
            local_pending=local_pending,
            local_latency=local_latency,
            loads=loads,
        )
        if kind is None:
            if self._incident_fallback(workload):
                return self._local.request(
                    operation,
                    frame=frame,
                    timeout=timeout,
                    admission_timeout=admission_timeout,
                    workload=workload,
                    **payload,
                )
            raise InferenceUnavailable(
                f"no weighted {self.role} inference target is available"
            )
        remote_has_alternate = bool(
            (local_weight > 0 and self._local is not None)
            or self._incident_fallback(workload)
        )
        try:
            if kind == "local":
                return self._run_local(
                    operation,
                    frame=frame,
                    timeout=timeout,
                    admission_timeout=admission_timeout,
                    workload=workload,
                    payload=payload,
                    alternate=bool(loads),
                )
            return self._run_remote(
                operation,
                frame=frame,
                timeout=timeout,
                admission_timeout=admission_timeout,
                workload=workload,
                payload=payload,
                alternate=remote_has_alternate,
                worker_weights=weights,
                default_worker_weight=default_weight,
            )
        except InferenceUnavailable as error:
            if kind == "remote" and local_weight > 0 and self._local is not None:
                result = self._run_local(
                    operation,
                    frame=frame,
                    timeout=timeout,
                    admission_timeout=admission_timeout,
                    workload=workload,
                    payload=payload,
                    alternate=False,
                )
                self._registry.note_rerouted(error)
                return result
            if kind == "local" and loads:
                return self._run_remote(
                    operation,
                    frame=frame,
                    timeout=timeout,
                    admission_timeout=admission_timeout,
                    workload=workload,
                    payload=payload,
                    alternate=False,
                    worker_weights=weights,
                    default_worker_weight=default_weight,
                )
            if kind == "remote" and self._incident_fallback(workload) and self._local is not None:
                result = self._run_local(
                    operation,
                    frame=frame,
                    timeout=timeout,
                    admission_timeout=admission_timeout,
                    workload=workload,
                    payload=payload,
                    alternate=False,
                )
                self._registry.note_rerouted(error)
                return result
            raise

    def _incident_fallback(self, workload: InferenceWorkload) -> bool:
        return bool(
            self.config.inference_mode == "hybrid"
            and self.config.remote_incident_fallback
            and InferenceWorkload(workload) is InferenceWorkload.INCIDENT_INITIAL
            and self._local is not None
        )

    def _usable_remote_loads(
        self,
        loads: list[dict[str, Any]],
        local_weight: int,
    ) -> list[dict[str, Any]]:
        active = [item for item in loads if not item.get("held")]
        if local_weight > 0:
            return active
        return active or loads

    def _local_inference_ms(self) -> float | None:
        local = self._local
        if local is None:
            return None
        status = local.cached_status()
        runtime = status.get("runtime") if isinstance(status, dict) else None
        if not isinstance(runtime, dict):
            return None
        value = runtime.get("average_inference_ms")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if number <= 0 or number != number or number in {float("inf"), float("-inf")}:
            return None
        return number

    def _attempt_timeout(self, timeout: float, *, alternate: bool) -> float:
        if not alternate:
            return timeout
        return min(timeout, INFERENCE_FAILOVER_SECONDS)

    def _run_local(
        self,
        operation: str,
        *,
        frame: np.ndarray | None,
        timeout: float,
        admission_timeout: float | None,
        workload: InferenceWorkload,
        payload: dict[str, Any],
        alternate: bool,
    ) -> Any:
        admission = admission_timeout
        if alternate:
            admission = (
                INFERENCE_FAILOVER_SECONDS
                if admission is None
                else min(admission, INFERENCE_FAILOVER_SECONDS)
            )
        return self._local.request(
            operation,
            frame=frame,
            timeout=timeout,
            admission_timeout=admission,
            workload=workload,
            **payload,
        )

    def _run_remote(
        self,
        operation: str,
        *,
        frame: np.ndarray | None,
        timeout: float,
        admission_timeout: float | None,
        workload: InferenceWorkload,
        payload: dict[str, Any],
        alternate: bool,
        worker_weights: dict[str, int] | None = None,
        default_worker_weight: int = 1,
    ) -> Any:
        del admission_timeout
        return self._remote.request(
            operation,
            frame=frame,
            timeout=self._attempt_timeout(timeout, alternate=alternate),
            workload=workload,
            worker_weights=worker_weights,
            default_worker_weight=default_worker_weight,
            **payload,
        )

    def _choose_weighted_target(
        self,
        *,
        local_weight: int,
        local_pending: int,
        local_latency: float | None,
        loads: list[dict[str, Any]],
    ) -> str | None:
        fallback_ms = _latency_fallback(
            [local_latency, *[item.get("inference_ms") for item in loads]]
        )
        options: list[tuple[str, float]] = []
        if local_weight > 0:
            options.append((
                "local",
                _placement_score(
                    local_pending,
                    local_weight,
                    local_latency,
                    fallback_ms,
                ),
            ))
        if loads:
            best_remote = min(
                _placement_score(
                    int(item["pending"]),
                    int(item["weight"]),
                    item.get("inference_ms"),
                    fallback_ms,
                )
                for item in loads
            )
            options.append(("remote", best_remote))
        if not options:
            return None
        best_score = min(score for _kind, score in options)
        tied = [kind for kind, score in options if score == best_score]
        if len(tied) == 1:
            return tied[0]
        with self._lock:
            choice = tied[self._balance_cursor % len(tied)]
            self._balance_cursor += 1
            return choice

    def status(
        self,
        workload: InferenceWorkload = InferenceWorkload.OFFLINE,
    ) -> dict[str, Any]:
        if self.config.inference_mode == "remote" or self._local is None:
            return self._remote.status(workload)
        status = self._local.status(workload)
        status["remote"] = self._remote.cached_status()
        status["inference_mode"] = self.config.inference_mode
        return status

    def cached_status(self) -> dict[str, Any]:
        if self.config.inference_mode == "remote" or self._local is None:
            return self._remote.cached_status()
        status = self._local.cached_status()
        status["remote"] = self._remote.cached_status()
        status["inference_mode"] = self.config.inference_mode
        return status

    def pending_requests(self) -> int:
        local = self._local.pending_requests() if self._local else 0
        return local + self._remote.pending_requests()

    def isolation_status(self) -> dict[str, Any]:
        if self.config.inference_mode == "remote" or self._local is None:
            return self._remote.isolation_status()
        status = self._local.isolation_status()
        status["remote"] = self._remote.isolation_status()
        status["inference_mode"] = self.config.inference_mode
        return status

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ..config import DetectorConfig
from .registry import RemoteInferenceRegistry
from .types import (
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
        try:
            return self._remote.request(
                operation,
                frame=frame,
                timeout=timeout,
                admission_timeout=admission_timeout,
                workload=workload,
                **payload,
            )
        except InferenceUnavailable:
            fallback = (
                mode == "hybrid"
                and self.config.remote_incident_fallback
                and InferenceWorkload(workload)
                is InferenceWorkload.INCIDENT_INITIAL
                and self._local is not None
            )
            if not fallback:
                raise
            return self._local.request(
                operation,
                frame=frame,
                timeout=timeout,
                admission_timeout=admission_timeout,
                workload=workload,
                **payload,
            )

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

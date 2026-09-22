from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..config import DetectorConfig
from .types import INFERENCE_REQUEST_TIMEOUT_SECONDS, InferenceWorkload


@runtime_checkable
class InferenceWorkerBackend(Protocol):
    """Transport-neutral boundary for one inference execution slot."""

    config: DetectorConfig
    role: str

    def update_config_reference(self, config: DetectorConfig) -> None: ...

    def reconfigure(
        self,
        config: DetectorConfig,
        initial_status: dict[str, Any],
        *,
        start_enabled: bool,
    ) -> None: ...

    def start(self) -> bool: ...

    def stop(self) -> None: ...

    def request(
        self,
        operation: str,
        *,
        frame: np.ndarray | None = None,
        timeout: float = INFERENCE_REQUEST_TIMEOUT_SECONDS,
        admission_timeout: float | None = None,
        workload: InferenceWorkload = InferenceWorkload.INTERACTIVE,
        **payload: Any,
    ) -> Any: ...

    def status(
        self,
        workload: InferenceWorkload = InferenceWorkload.OFFLINE,
    ) -> dict[str, Any]: ...

    def cached_status(self) -> dict[str, Any]: ...

    def pending_requests(self) -> int: ...

    def isolation_status(self) -> dict[str, Any]: ...


class InferenceWorkerFactory(Protocol):
    """Build one local or remote worker backend for a supervisor role."""

    def __call__(
        self,
        config: DetectorConfig,
        role: str,
        initial_status: dict[str, Any],
        *,
        start_enabled: bool = True,
    ) -> InferenceWorkerBackend: ...

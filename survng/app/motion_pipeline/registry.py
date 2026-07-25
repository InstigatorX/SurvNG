from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .contracts import MotionStage


MotionStageBuilder = Callable[[str, Mapping[str, Any], "MotionStageDependencies"], MotionStage]


@dataclass(frozen=True, slots=True)
class MotionStageDependencies:
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("survng.motion"))
    services: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MotionStageRegistration:
    implementation: str
    builder: MotionStageBuilder
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()


class MotionStageRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, MotionStageRegistration] = {}

    def register(self, registration: MotionStageRegistration) -> None:
        name = registration.implementation.strip().lower()
        if not name:
            raise ValueError("motion stage implementation name cannot be empty")
        if name in self._registrations:
            raise ValueError(f"motion stage implementation already registered: {name}")
        self._registrations[name] = registration

    def registration(self, implementation: str) -> MotionStageRegistration:
        name = implementation.strip().lower()
        try:
            return self._registrations[name]
        except KeyError as error:
            available = ", ".join(sorted(self._registrations)) or "none"
            raise ValueError(
                f"unknown motion stage implementation {implementation!r}; available: {available}"
            ) from error

    def implementations(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))


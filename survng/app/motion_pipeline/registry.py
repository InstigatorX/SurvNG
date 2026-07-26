from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

from .contracts import MotionStage


MotionStageBuilder = Callable[[str, Mapping[str, Any], "MotionStageDependencies"], MotionStage]
MotionOptionType = Literal["boolean", "integer", "number", "string", "string_list", "object"]


@dataclass(frozen=True, slots=True)
class MotionStageOption:
    key: str
    label: str
    value_type: MotionOptionType
    default: Any = None
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    advanced: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "type": self.value_type,
            "default": self.default,
            "description": self.description,
            "advanced": self.advanced,
        }
        if self.minimum is not None:
            payload["minimum"] = self.minimum
        if self.maximum is not None:
            payload["maximum"] = self.maximum
        if self.choices:
            payload["choices"] = list(self.choices)
        return payload


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
    graph: str = "custom"
    category: str = "custom"
    display_name: str = ""
    description: str = ""
    options: tuple[MotionStageOption, ...] = ()
    continuous_analysis: bool = False
    motion_source: str = ""
    observation_kinds: frozenset[str] | None = None


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

    def catalog(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "implementation": name,
                "graph": registration.graph,
                "category": registration.category,
                "name": registration.display_name or name.replace("_", " ").title(),
                "description": registration.description,
                "requires": sorted(registration.requires),
                "provides": sorted(registration.provides),
                "continuous_analysis": registration.continuous_analysis,
                "motion_source": registration.motion_source,
                "observation_kinds": (
                    sorted(registration.observation_kinds)
                    if registration.observation_kinds is not None
                    else []
                ),
                "options": [option.as_dict() for option in registration.options],
            }
            for name, registration in sorted(self._registrations.items())
        )

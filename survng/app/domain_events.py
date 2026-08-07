"""Typed internal event payloads at subsystem boundaries.

The MQTT and SSE adapters intentionally still receive dictionaries. Producers use
these immutable contracts so missing or misspelled fields fail close to the source,
while ``to_payload`` provides the explicit serialization boundary.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


def _required_text(value: object, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


@dataclass(frozen=True, slots=True)
class MotionObserved:
    camera_id: str
    timestamp: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_id", _required_text(self.camera_id, "camera_id"))
        object.__setattr__(self, "timestamp", _required_text(self.timestamp, "timestamp"))
        object.__setattr__(self, "source", _required_text(self.source, "source"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class IncidentCreated:
    event_id: int
    camera_id: str
    timestamp: str
    kind: str = "motion"

    def __post_init__(self) -> None:
        if int(self.event_id) <= 0:
            raise ValueError("event_id must be positive")
        object.__setattr__(self, "event_id", int(self.event_id))
        object.__setattr__(self, "camera_id", _required_text(self.camera_id, "camera_id"))
        object.__setattr__(self, "timestamp", _required_text(self.timestamp, "timestamp"))
        object.__setattr__(self, "kind", _required_text(self.kind, "kind"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class ObjectDetected:
    event_id: int
    camera_id: str
    timestamp: str
    objects: tuple[Mapping[str, Any], ...]
    snapshot_path: str = ""
    recording_path: str = ""
    source: str = ""
    incident_objects: tuple[Mapping[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        if int(self.event_id) <= 0:
            raise ValueError("event_id must be positive")
        object.__setattr__(self, "event_id", int(self.event_id))
        object.__setattr__(self, "camera_id", _required_text(self.camera_id, "camera_id"))
        object.__setattr__(self, "timestamp", _required_text(self.timestamp, "timestamp"))
        object.__setattr__(
            self,
            "objects",
            tuple(deepcopy(dict(item)) for item in self.objects),
        )
        if self.incident_objects is not None:
            object.__setattr__(
                self,
                "incident_objects",
                tuple(deepcopy(dict(item)) for item in self.incident_objects),
            )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "snapshot_path": self.snapshot_path,
            "recording_path": self.recording_path,
            "objects": [deepcopy(dict(item)) for item in self.objects],
        }
        if self.incident_objects is not None:
            payload["incident_objects"] = [
                deepcopy(dict(item)) for item in self.incident_objects
            ]
        if self.source:
            payload["source"] = self.source
        return payload


@dataclass(frozen=True, slots=True)
class TrackingCompleted:
    event_id: int
    camera_id: str
    state: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.event_id) <= 0:
            raise ValueError("event_id must be positive")
        object.__setattr__(self, "event_id", int(self.event_id))
        object.__setattr__(self, "camera_id", _required_text(self.camera_id, "camera_id"))
        object.__setattr__(self, "state", _required_text(self.state, "state"))
        object.__setattr__(self, "details", deepcopy(dict(self.details)))

    def to_payload(self) -> dict[str, Any]:
        return {
            **deepcopy(dict(self.details)),
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "state": self.state,
        }

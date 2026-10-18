"""Live zone occupancy for the native Home Assistant sensor.

A confirmed track supplies the fast updates. A completed scene look supplies a
parked vehicle, including one that is stationary scene context. A failed look
does not change occupancy. Ending a tracking session does not clear a zone
that this merged view still shows.
"""

from __future__ import annotations

import threading
from typing import Any

from .config import CameraConfig, DetectionZone


def labeled_zone_observations(objects: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return label and zones from detector objects and nested scene observations."""
    found: list[dict[str, Any]] = []
    for item in objects or []:
        if not isinstance(item, dict):
            continue
        if item.get("label"):
            found.append({
                "label": item.get("label"),
                "zones": list(item.get("zones") or []),
            })
        for observation in item.get("observations") or []:
            if not isinstance(observation, dict):
                continue
            payload = observation.get("payload", observation)
            if isinstance(payload, dict) and payload.get("label"):
                found.append({
                    "label": payload.get("label"),
                    "zones": list(payload.get("zones") or []),
                })
    return found


def _zone_config(zone: DetectionZone | dict[str, Any]) -> tuple[str, bool, str, set[str]]:
    if isinstance(zone, DetectionZone):
        name = zone.name
        enabled = bool(zone.enabled)
        behavior = zone.behavior
        classes = zone.object_classes
    else:
        name = str(zone.get("name") or "")
        enabled = zone.get("enabled") is not False
        behavior = str(zone.get("behavior") or "incident")
        classes = zone.get("object_classes") or []
    allowed = {str(label).strip().lower() for label in classes if str(label).strip()}
    return name.strip(), enabled, behavior, allowed


def zone_entries(
    zones: list[DetectionZone] | list[dict[str, Any]],
    objects: list[dict[str, Any]],
) -> dict[str, frozenset[str]]:
    """Occupied incident zones. Class filters apply; the notification switch does not."""
    occupied: dict[str, set[str]] = {}
    for zone in zones:
        name, enabled, behavior, allowed = _zone_config(zone)
        if not name or not enabled or behavior != "incident":
            continue
        labels: set[str] = set()
        for item in objects:
            if not isinstance(item, dict) or name not in (item.get("zones") or []):
                continue
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            if allowed and label.lower() not in allowed:
                continue
            labels.add(label)
        if labels:
            occupied[name] = labels
    return {name: frozenset(labels) for name, labels in occupied.items()}


def _signature(entries: dict[str, frozenset[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(sorted(
        (zone, tuple(sorted(labels)))
        for zone, labels in entries.items()
    ))


# A discovery frame is often still "pending": the scene was sampled, and an
# incident has not been decided. "unsupported" means the look finished and
# found no activity, so it must not replace occupancy. "incomplete" never saw
# enough of the scene to publish or clear.
OCCUPANCY_LOOK_VERDICTS = frozenset({"", "pending", "supported"})


def scene_look_publishes(verdict: str | None, *, detection_completed: bool) -> bool:
    """True when a scene sample may replace the camera's occupied zones."""
    if not detection_completed:
        return False
    return str(verdict or "") in OCCUPANCY_LOOK_VERDICTS


class _CameraOccupancy:
    def __init__(self) -> None:
        self.session_open = False
        self.look: dict[str, frozenset[str]] | None = None
        self.track_current: dict[str, frozenset[str]] = {}
        self.lifetime: dict[str, frozenset[str]] = {}
        self.sticky: dict[str, frozenset[str]] = {}
        self.departed: set[str] = set()
        self.published: tuple[tuple[str, tuple[str, ...]], ...] | None = None


class ZoneOccupancyBook:
    """Merge track and scene-look occupancy for one process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cameras: dict[str, _CameraOccupancy] = {}

    def observe(
        self,
        camera: CameraConfig,
        payload: dict[str, Any],
        *,
        fail_safe_seconds: float,
    ) -> dict[str, Any] | None:
        camera_id = str(payload.get("camera_id") or "")
        if not camera_id:
            return None
        source = str(payload.get("source") or "")
        with self._lock:
            state = self._cameras.setdefault(camera_id, _CameraOccupancy())
            if source == "scene":
                self._apply_look(state, camera, payload)
                force = True
            elif source == "track":
                self._apply_tracks(state, camera, payload)
                force = False
            else:
                return None
            occupied = self._compose(state)
            signature = _signature(occupied)
            if not force and signature == state.published:
                return None
            state.published = signature
        return {
            "camera_id": camera_id,
            "timestamp": payload.get("timestamp"),
            "fail_safe_seconds": fail_safe_seconds,
            "zones": [
                {"zone": zone, "classes": list(labels)}
                for zone, labels in signature
            ],
        }

    def _apply_tracks(
        self,
        state: _CameraOccupancy,
        camera: CameraConfig,
        payload: dict[str, Any],
    ) -> None:
        tracks = [item for item in payload.get("tracks") or [] if isinstance(item, dict)]
        current = zone_entries(camera.zones, [
            {"label": item.get("label"), "zones": item.get("current_zones") or []}
            for item in tracks
        ])
        lifetime = zone_entries(camera.zones, [
            {"label": item.get("label"), "zones": item.get("zones") or []}
            for item in tracks
        ])
        moved = set(lifetime) - set(current)
        state.lifetime = lifetime
        state.departed = (state.departed | moved) - set(current)
        disappeared = set(state.track_current) - set(current) - state.departed
        for zone in disappeared:
            state.sticky.setdefault(zone, state.track_current[zone])
        for zone in list(state.sticky):
            if zone in state.departed or zone in current:
                state.sticky.pop(zone, None)
        state.track_current = current
        state.session_open = bool(payload.get("active"))

    def _apply_look(
        self,
        state: _CameraOccupancy,
        camera: CameraConfig,
        payload: dict[str, Any],
    ) -> None:
        seen = zone_entries(camera.zones, list(payload.get("objects") or []))
        state.look = seen
        if state.session_open:
            # This session's live departure still wins. A departure remembered
            # from an earlier session does not hide a vehicle this look sees.
            state.departed = set(state.lifetime) - set(state.track_current)
            state.sticky = {
                zone: labels
                for zone, labels in state.sticky.items()
                if zone in seen or zone in state.track_current
            }
            return
        # A completed look replaces the held track. The vehicle is still
        # occupied only when this frame still sees it.
        state.track_current = {}
        state.sticky = {}
        state.departed = set()
        state.lifetime = {}

    def _compose(self, state: _CameraOccupancy) -> dict[str, frozenset[str]]:
        occupied: dict[str, frozenset[str]] = {}
        if state.look:
            occupied.update(state.look)
        occupied.update(state.sticky)
        occupied.update(state.track_current)
        for zone in state.departed:
            if zone not in state.track_current:
                occupied.pop(zone, None)
        return occupied

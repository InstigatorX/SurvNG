from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping

from .config import CameraTransitionRoute


@dataclass(frozen=True, slots=True)
class DetectionWatch:
    """A bounded opportunity to verify motion on the next camera in a route."""

    source_camera_id: str
    target_camera_id: str
    source_event_id: int
    source_event_at: float
    eligible_at: float
    expires_at: float
    route_name: str
    labels: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_camera_id": self.source_camera_id,
            "target_camera_id": self.target_camera_id,
            "source_event_id": self.source_event_id,
            "source_event_at": self.source_event_at,
            "eligible_at": self.eligible_at,
            "expires_at": self.expires_at,
            "route_name": self.route_name,
            "labels": list(self.labels),
        }


class RouteDetectionWatch:
    """Turns confirmed object incidents into short downstream verification windows.

    Watches are deliberately advisory.  A match may cause object analysis to run,
    but it never makes motion or an object eligible by itself.  A downstream
    incident opens the next set of windows, so configured routes naturally form a
    bounded chain without recursively predicting an entire path.
    """

    def __init__(
        self,
        routes: Iterable[CameraTransitionRoute] = (),
        *,
        maximum_watches: int = 4096,
    ) -> None:
        self._lock = threading.Lock()
        self._routes = tuple(route for route in routes if route.enabled)
        self._maximum_watches = max(16, int(maximum_watches))
        self._watches: list[DetectionWatch] = []
        self._observed_events: set[tuple[str, int]] = set()
        self._observed_order: deque[tuple[str, int]] = deque()
        self._counters = {
            "opened": 0,
            "matched": 0,
            "consumed": 0,
            "expired": 0,
            "overflowed": 0,
        }

    def reconfigure(self, routes: Iterable[CameraTransitionRoute]) -> None:
        with self._lock:
            self._routes = tuple(route for route in routes if route.enabled)
            self._watches.clear()
            self._observed_events.clear()
            self._observed_order.clear()

    def observe_incident(
        self,
        *,
        camera_id: str,
        event_id: int,
        event_at: float,
        objects: Iterable[Mapping[str, object]],
    ) -> tuple[DetectionWatch, ...]:
        labels = tuple(sorted({
            str(item.get("label") or "").strip().lower()
            for item in objects
            if str(item.get("label") or "").strip()
        }))
        if not labels or not camera_id or event_id <= 0:
            return ()
        key = (camera_id, int(event_id))
        with self._lock:
            if key in self._observed_events:
                return ()
            self._observed_events.add(key)
            self._observed_order.append(key)
            while len(self._observed_order) > self._maximum_watches * 8:
                expired_key = self._observed_order.popleft()
                self._observed_events.discard(expired_key)
            created: list[DetectionWatch] = []
            for route in self._routes:
                target = ""
                if route.from_camera == camera_id:
                    target = route.to_camera
                elif route.bidirectional and route.to_camera == camera_id:
                    target = route.from_camera
                if not target:
                    continue
                created.append(DetectionWatch(
                    source_camera_id=camera_id,
                    target_camera_id=target,
                    source_event_id=int(event_id),
                    source_event_at=float(event_at),
                    eligible_at=float(event_at) + float(route.min_seconds),
                    expires_at=float(event_at) + float(route.max_seconds),
                    route_name=route.name or f"{camera_id} → {target}",
                    labels=labels,
                ))
            self._watches.extend(created)
            self._counters["opened"] += len(created)
            self._trim(float(event_at))
            return tuple(created)

    def match(self, camera_id: str, captured_at: float) -> DetectionWatch | None:
        when = float(captured_at)
        with self._lock:
            self._trim(when)
            matches = [
                watch
                for watch in self._watches
                if watch.target_camera_id == camera_id
                and watch.eligible_at <= when <= watch.expires_at
            ]
            if not matches:
                return None
            # Prefer the most recent confirmed upstream evidence.  Matching is
            # intentionally non-consuming because EMA persistence spans samples.
            self._counters["matched"] += 1
            return max(matches, key=lambda watch: watch.source_event_at)

    def consume(self, camera_id: str, source_event_id: int) -> bool:
        """Consume one target/event watch after its verification is reserved."""
        with self._lock:
            before = len(self._watches)
            self._watches = [
                watch
                for watch in self._watches
                if not (
                    watch.target_camera_id == camera_id
                    and watch.source_event_id == int(source_event_id)
                )
            ]
            consumed = len(self._watches) < before
            if consumed:
                self._counters["consumed"] += 1
            return consumed

    def snapshot(self, now: float) -> tuple[dict[str, object], ...]:
        with self._lock:
            self._trim(float(now))
            return tuple(watch.as_dict() for watch in self._watches)

    def status(self, now: float) -> dict[str, object]:
        with self._lock:
            self._trim(float(now))
            return {
                **self._counters,
                "active": len(self._watches),
                "watches": tuple(watch.as_dict() for watch in self._watches),
            }

    def _trim(self, now: float) -> None:
        before = len(self._watches)
        self._watches = [watch for watch in self._watches if watch.expires_at >= now]
        self._counters["expired"] += before - len(self._watches)
        if len(self._watches) > self._maximum_watches:
            self._counters["overflowed"] += len(self._watches) - self._maximum_watches
            self._watches = self._watches[-self._maximum_watches :]

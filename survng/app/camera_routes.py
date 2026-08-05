from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import CameraTransitionRoute


@dataclass(frozen=True, slots=True)
class CameraRouteMatch:
    name: str
    from_camera: str
    to_camera: str
    elapsed_seconds: float
    min_seconds: float
    max_seconds: float
    timing_score: float
    bidirectional: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "route_name": self.name,
            "route_from_camera": self.from_camera,
            "route_to_camera": self.to_camera,
            "route_min_seconds": self.min_seconds,
            "route_max_seconds": self.max_seconds,
            "route_timing_score": self.timing_score,
            "route_bidirectional": self.bidirectional,
        }


def match_camera_route(
    routes: Iterable[CameraTransitionRoute],
    from_camera: str,
    to_camera: str,
    elapsed_seconds: float,
) -> CameraRouteMatch | None:
    """Return the strongest enabled directional route whose timing contains a transition."""
    elapsed = max(0.0, float(elapsed_seconds))
    matches: list[CameraRouteMatch] = []
    for route in routes:
        if not route.enabled:
            continue
        direct = route.from_camera == from_camera and route.to_camera == to_camera
        reverse = (
            route.bidirectional
            and route.from_camera == to_camera
            and route.to_camera == from_camera
        )
        if not (direct or reverse) or not route.min_seconds <= elapsed <= route.max_seconds:
            continue
        matches.append(CameraRouteMatch(
            name=route.name or f"{from_camera} → {to_camera}",
            from_camera=from_camera,
            to_camera=to_camera,
            elapsed_seconds=elapsed,
            min_seconds=route.min_seconds,
            max_seconds=route.max_seconds,
            # A route describes an allowed interval, not an ideal midpoint.
            # Every value inside it is therefore equally valid.
            timing_score=1.0,
            bidirectional=route.bidirectional,
        ))
    return max(matches, key=lambda item: item.timing_score, default=None)

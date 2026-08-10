"""Shared stationary-object policy across motion and semantic attribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StationaryObjectPolicy:
    name: str
    background_learning_seconds: float
    background_learning_rate: float
    displacement_ratio: float
    path_ratio: float
    containment_ratio: float
    progress_ratio: float
    maximum_displacement_ratio: float
    scene_stable_displacement_ratio: float
    scene_stable_path_ratio: float
    scene_memory_ttl_seconds: float = 2 * 60 * 60
    scene_memory_min_iou: float = 0.72
    scene_memory_min_prior_sightings: int = 2

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_POLICIES = {
    "low": StationaryObjectPolicy(
        "low", 12.0, 0.03, 0.006, 0.015, 0.012, 0.18, 0.035, 0.0015, 0.004,
    ),
    "balanced": StationaryObjectPolicy(
        "balanced", 8.0, 0.05, 0.01, 0.025, 0.025, 0.32, 0.065, 0.0025, 0.006,
    ),
    "high": StationaryObjectPolicy(
        "high", 5.0, 0.08, 0.02, 0.05, 0.045, 0.45, 0.12, 0.004, 0.01,
    ),
}


def stationary_object_policy(value: str | None) -> StationaryObjectPolicy:
    return _POLICIES.get(str(value or "balanced").strip().lower(), _POLICIES["balanced"])

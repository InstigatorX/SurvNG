"""Shared stationary-object policy across motion and semantic attribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


class _MovementThresholds(Protocol):
    @property
    def movement_threshold(self) -> float: ...

    @property
    def path_threshold(self) -> float: ...


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
    # Stillness for scene context follows the credible-movement thresholds of
    # the observed box, which grow with its size (see scene_stable_limits).
    # These two ratios are the fixed limits that applied to every box before;
    # they remain the floor, so small and mid-sized boxes keep their tested
    # behaviour.
    scene_stable_displacement_ratio: float
    scene_stable_path_ratio: float
    # A larger box may hold this fraction of its credible-movement thresholds
    # and still be stable. For a 100 px box in a 1000 px frame, balanced 0.45
    # is about the former fixed limits.
    scene_stable_movement_fraction: float
    # Stable never exceeds this fraction of the credible-movement thresholds,
    # so no sighting is both stable and credible movement, and the band
    # between them stays indeterminate and admitted.
    scene_stable_margin: float
    scene_memory_ttl_seconds: float = 2 * 60 * 60
    scene_memory_min_iou: float = 0.72
    scene_memory_min_prior_sightings: int = 2

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def scene_stable_limits(self, motion: _MovementThresholds) -> tuple[float, float]:
        """Largest displacement and extent that count as stillness for this box."""
        return (
            min(
                motion.movement_threshold * self.scene_stable_margin,
                max(
                    self.scene_stable_displacement_ratio,
                    motion.movement_threshold * self.scene_stable_movement_fraction,
                ),
            ),
            min(
                motion.path_threshold * self.scene_stable_margin,
                max(
                    self.scene_stable_path_ratio,
                    motion.path_threshold * self.scene_stable_movement_fraction,
                ),
            ),
        )


_POLICIES = {
    "low": StationaryObjectPolicy(
        "low", 12.0, 0.03, 0.006, 0.015, 0.012, 0.18, 0.035, 0.0015, 0.004,
        0.25, 0.5,
    ),
    "balanced": StationaryObjectPolicy(
        "balanced", 8.0, 0.05, 0.01, 0.025, 0.025, 0.32, 0.065, 0.0025, 0.006,
        0.45, 0.7,
    ),
    "high": StationaryObjectPolicy(
        "high", 5.0, 0.08, 0.02, 0.05, 0.045, 0.45, 0.12, 0.004, 0.01,
        0.6, 0.8,
    ),
}


def stationary_object_policy(value: str | None) -> StationaryObjectPolicy:
    return _POLICIES.get(str(value or "balanced").strip().lower(), _POLICIES["balanced"])

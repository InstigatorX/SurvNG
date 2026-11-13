from __future__ import annotations


from ..config import ObjectTrackingConfig
from .hybrid import HybridObjectTracker
from .sparse_identity import SparseIdentityObjectTracker
from .types import ObjectTrackerBackend, ObjectTrackerBuilder


class ObjectTrackerRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, ObjectTrackerBuilder] = {}

    def register(self, implementation: str, builder: ObjectTrackerBuilder) -> None:
        name = str(implementation or "").strip().lower()
        if not name:
            raise ValueError("object tracker implementation cannot be empty")
        if name in self._builders:
            raise ValueError(f"object tracker implementation already registered: {name}")
        self._builders[name] = builder

    def create(
        self,
        implementation: str,
        config: ObjectTrackingConfig,
        high_confidence_threshold: float,
    ) -> ObjectTrackerBackend:
        name = str(implementation or "").strip().lower()
        builder = self._builders.get(name)
        if builder is None:
            available = ", ".join(sorted(self._builders)) or "none"
            raise ValueError(
                f"unknown object tracker implementation {name!r}; available: {available}"
            )
        return builder(config, high_confidence_threshold)

    def require(self, implementation: str) -> None:
        name = str(implementation or "").strip().lower()
        if name not in self._builders:
            available = ", ".join(sorted(self._builders)) or "none"
            raise ValueError(
                f"unknown object tracker implementation {name!r}; available: {available}"
            )

def build_builtin_object_tracker_registry() -> ObjectTrackerRegistry:
    registry = ObjectTrackerRegistry()
    registry.register("survng_hybrid", HybridObjectTracker)
    registry.register("survng_sparse_identity", SparseIdentityObjectTracker)
    return registry

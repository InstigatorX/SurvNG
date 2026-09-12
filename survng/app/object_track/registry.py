from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from pathlib import Path

from ..config import ObjectTrackingConfig
from .bytetrack import ByteTrackObjectTracker
from .hybrid import HybridObjectTracker
from .types import ObjectTrackerBackend, ObjectTrackerBuilder


TESTED_ULTRALYTICS_TRACKING_VERSION = "8.4.148"
MINIMUM_ULTRALYTICS_TRACKING_VERSION = (8, 4, 108)
MAXIMUM_ULTRALYTICS_TRACKING_VERSION = (8, 5, 0)


def _ultralytics_tracking_version_supported(installed_version: str) -> bool:
    """Accept compatible patches without importing the heavyweight runtime."""
    try:
        release = tuple(
            int(part)
            for part in installed_version.partition("+")[0].partition("-")[0].split(".")[:3]
        )
        release = (*release, *(0 for _ in range(3 - len(release))))
    except (TypeError, ValueError):
        return False
    return MINIMUM_ULTRALYTICS_TRACKING_VERSION <= release < MAXIMUM_ULTRALYTICS_TRACKING_VERSION

def _ultralytics_tracking_dependency_status(
    *,
    tracker_name: str,
    module_name: str,
) -> dict[str, Any]:
    try:
        installed_version = version("ultralytics")
    except PackageNotFoundError:
        installed_version = ""
    package_present = importlib.util.find_spec("ultralytics") is not None
    lap_present = importlib.util.find_spec("lap") is not None
    # find_spec on a dotted module imports its parents (and can initialize
    # PyTorch or Ultralytics settings). Inspect files under the top-level spec
    # instead so capability/status requests stay lazy and side-effect free.
    package_spec = importlib.util.find_spec("ultralytics")
    relative_module = Path(*module_name.split(".")[1:]).with_suffix(".py")
    tracker_present = bool(package_spec and any(
        (Path(location) / relative_module).is_file()
        for location in (package_spec.submodule_search_locations or [])
    ))
    version_supported = _ultralytics_tracking_version_supported(installed_version)
    if tracker_name in {"TrackTrack", "BoT-SORT"}:
        version_supported = installed_version == TESTED_ULTRALYTICS_TRACKING_VERSION
    available = bool(
        installed_version
        and package_present
        and lap_present
        and tracker_present
        and version_supported
    )
    if not installed_version:
        reason = "Ultralytics is not installed."
    elif not lap_present:
        reason = "The LAP assignment dependency is not installed."
    elif not tracker_present:
        reason = f"The installed Ultralytics build does not include {tracker_name}."
    elif not version_supported:
        reason = f"Ultralytics {installed_version} is unsupported for {tracker_name}; install the tested version {TESTED_ULTRALYTICS_TRACKING_VERSION}."
    else:
        reason = ""
    return {
        "available": available,
        "installed_version": installed_version,
        # Keep required_version for API compatibility with older frontends.
        # Availability is capability-based; this is the reproducible version
        # pinned by requirements-ultralytics-tracking.txt and exercised by CI.
        "required_version": TESTED_ULTRALYTICS_TRACKING_VERSION,
        "tested_version": TESTED_ULTRALYTICS_TRACKING_VERSION,
        "is_tested_version": installed_version == TESTED_ULTRALYTICS_TRACKING_VERSION,
        "supported_version_range": "==8.4.148" if tracker_name in {"TrackTrack", "BoT-SORT"} else ">=8.4.108,<8.5",
        "reason": reason,
    }

def ultralytics_deepocsort_dependency_status() -> dict[str, Any]:
    return _ultralytics_tracking_dependency_status(
        tracker_name="Deep OC-SORT",
        module_name="ultralytics.trackers.deep_oc_sort",
    )

def ultralytics_fasttrack_dependency_status() -> dict[str, Any]:
    return _ultralytics_tracking_dependency_status(
        tracker_name="FastTrack",
        module_name="ultralytics.trackers.fast_tracker",
    )

def _build_ultralytics_deepocsort(
    config: ObjectTrackingConfig,
    high_confidence_threshold: float,
) -> ObjectTrackerBackend:
    from ..ultralytics_tracking import UltralyticsDeepOCSortObjectTracker

    return UltralyticsDeepOCSortObjectTracker(config, high_confidence_threshold)

def _build_ultralytics_fasttrack(
    config: ObjectTrackingConfig,
    high_confidence_threshold: float,
) -> ObjectTrackerBackend:
    from ..ultralytics_tracking import UltralyticsFastTrackObjectTracker

    return UltralyticsFastTrackObjectTracker(config, high_confidence_threshold)

def ultralytics_tracktrack_dependency_status() -> dict[str, Any]:
    return _ultralytics_tracking_dependency_status(
        tracker_name="TrackTrack", module_name="ultralytics.trackers.track_tracker",
    )


def ultralytics_botsort_dependency_status() -> dict[str, Any]:
    return _ultralytics_tracking_dependency_status(
        tracker_name="BoT-SORT", module_name="ultralytics.trackers.bot_sort",
    )


def _build_ultralytics_tracktrack(config: ObjectTrackingConfig, high_confidence_threshold: float) -> ObjectTrackerBackend:
    status = ultralytics_tracktrack_dependency_status()
    if not status["available"]:
        raise RuntimeError(status["reason"])
    from ..ultralytics_tracking import UltralyticsTrackTrackObjectTracker
    return UltralyticsTrackTrackObjectTracker(config, high_confidence_threshold)


def _build_ultralytics_botsort(config: ObjectTrackingConfig, high_confidence_threshold: float) -> ObjectTrackerBackend:
    status = ultralytics_botsort_dependency_status()
    if not status["available"]:
        raise RuntimeError(status["reason"])
    from ..ultralytics_tracking import UltralyticsBotSortObjectTracker
    return UltralyticsBotSortObjectTracker(config, high_confidence_threshold)


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
    # Keep the evaluated candidate name as an alias for saved comparison data
    # and transition tests; it now points at the promoted production behavior.
    registry.register("survng_hybrid_candidate", HybridObjectTracker)
    # Compatibility alias for configurations created before the tracker gained
    # SurvNG-specific geometry and appearance association.
    registry.register("bytetrack", ByteTrackObjectTracker)
    # Ultralytics alternatives are registered for bounded offline comparisons.
    # User configuration normalizes optional upstream trackers back to Hybrid,
    # so production sessions cannot select this implementation.
    registry.register("ultralytics_deepocsort", _build_ultralytics_deepocsort)
    registry.register("ultralytics_fasttrack", _build_ultralytics_fasttrack)
    registry.register("ultralytics_tracktrack", _build_ultralytics_tracktrack)
    registry.register("ultralytics_botsort", _build_ultralytics_botsort)
    return registry

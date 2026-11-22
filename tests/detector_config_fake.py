"""Partial detector configuration for recorded-detection test doubles.

CameraWorker and RecordedMotionObjectDetector read these DetectorConfig
fields directly, so doubles must carry them with production defaults.

The recorded-refinement budget fields (``event_refinement_retry_seconds``
and friends) are deliberately left out:
``resolve_recorded_refinement_plan`` falls back to the module constants when
a config lacks them, and many tests shorten those waits by patching the
constants. Pass a budget field as an override to pin it explicitly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from survng.app.config import DetectorConfig

DIRECT_FIELDS = (
    "confidence_threshold",
    "event_candidate_confidence_threshold",
    "event_confirmation_frames",
    "event_class_confirmation_frames",
    "event_class_confidence_thresholds",
    "event_refinement_stages",
    "recorded_adaptive_sampling",
    "require_incident_zone",
    "object_activity_attribution",
    "face_recognition_enabled",
    "face_evidence_enabled",
    "face_evidence_max_extra_frames",
    "face_evidence_timeout_seconds",
    "face_enrich_max_people",
    "scene_discovery_enabled",
    "scene_discovery_interval_seconds",
    "scene_discovery_heartbeat_seconds",
    "tracking",
    "depth",
)


def detector_config(**overrides: Any) -> SimpleNamespace:
    defaults = DetectorConfig()
    values = {name: getattr(defaults, name) for name in DIRECT_FIELDS}
    values.update(overrides)
    return SimpleNamespace(**values)

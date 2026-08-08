"""Transactional targeted configuration application."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, Protocol

from .config import AppConfig, normalize_config

LOGGER = logging.getLogger(__name__)

HOT_CONFIG_FIELDS = frozenset({"base_path", "event_clip_before_seconds", "event_clip_after_seconds", "incident_thumbnail_annotations", "image_storage", "recording_cache_max_gb", "recording_cache_max_days", "recording_cache_prewarm", "audit_ai", "mqtt", "retention", "semantic_search"})
RECORDER_CONFIG_FIELDS = frozenset({"ffmpeg_path", "hardware_acceleration", "recording_segment_seconds"})
DETECTOR_HOT_POLICY_FIELDS = frozenset({"confidence_threshold", "event_confirmation_frames", "event_class_confirmation_frames", "event_class_confidence_thresholds", "object_activity_attribution", "require_incident_zone", "face_max_observations", "face_detection_threshold", "face_match_threshold", "face_auto_identify_enabled", "face_auto_identify_threshold", "face_auto_identify_margin", "face_min_size", "face_max_references"})
TRACKING_SESSION_FIELDS = frozenset({"enabled", "implementation", "excluded_labels", "sample_fps", "max_session_seconds", "lost_timeout_seconds", "min_confirmations", "low_confidence_threshold", "match_iou_threshold", "match_center_distance_ratio", "max_active_cameras", "adaptive_burst_enabled", "burst_max_active_cameras", "capacity_wait_seconds", "deferred_reid_enabled", "deferred_reid_delay_seconds", "deferred_reid_min_crop_pixels", "deferred_reid_rate_per_minute", "related_sequence_window_seconds", "camera_transition_routes", "max_tracks_per_session", "reid_max_age_seconds", "reid_max_embeddings_per_frame", "reid_refresh_interval_frames", "reid_match_threshold", "vehicle_reid_match_threshold", "vehicle_reid_labels"})
DETECTOR_OBJECT_ENGINE_FIELDS = frozenset({"enabled", "backend", "object_worker_count", "model_path", "model_xml", "coreml_model_path", "labels_path", "device", "nms_threshold", "warmup_enabled", "labels"})
DETECTOR_OBJECT_TRACKING_RESET_FIELDS = frozenset({"enabled", "backend", "model_path", "model_xml", "coreml_model_path", "labels_path", "nms_threshold", "labels"})
DETECTOR_FACE_ENGINE_FIELDS = frozenset({"face_recognition_enabled", "face_embedding_model_path", "face_landmark_model_path", "face_detection_model_path", "face_recognition_device"})
DETECTOR_SHARED_ENGINE_FIELDS = frozenset({"cache_enabled", "cache_dir"})
TRACKING_REID_ENGINE_FIELDS = frozenset({"reid_enabled", "reid_model_path", "reid_device", "vehicle_reid_enabled", "vehicle_reid_model_path", "vehicle_reid_device"})


class ConfigurableRuntime(Protocol):
    config: AppConfig
    def reconfigure_recorders(self, config: AppConfig) -> None: ...
    def reconfigure_mqtt(self, config: Any) -> None: ...
    def reconfigure_recording_retention(self, config: AppConfig) -> None: ...
    def reconfigure_image_storage(self, config: Any) -> None: ...
    def reconfigure_semantic_search(self, config: Any) -> None: ...
    def reconfigure_inference(self, config: Any, roles: set[str], *, refresh_tracking: bool) -> None: ...
    def reconfigure_object_tracking(self, config: Any) -> None: ...
    def reconfigure_detector_policy(self, config: Any) -> None: ...


class StorageTasksActiveError(RuntimeError):
    def __init__(self, tasks: list[str]) -> None:
        self.tasks = tasks
        super().__init__("configuration change was not applied because storage work is active: " + f"{', '.join(tasks)}. Wait for it to finish or cancel it from Maintenance.")


def _without_fields(value: dict, fields: frozenset[str]) -> dict:
    return {key: item for key, item in value.items() if key not in fields}


def manager_owned_config(config: AppConfig) -> dict:
    payload = config.model_dump(mode="json")
    for field in HOT_CONFIG_FIELDS | RECORDER_CONFIG_FIELDS:
        payload.pop(field, None)
    for camera in payload.get("cameras", []):
        camera.pop("retention", None)
    payload["detector"] = _without_fields(payload.get("detector", {}), DETECTOR_HOT_POLICY_FIELDS | DETECTOR_OBJECT_ENGINE_FIELDS | DETECTOR_FACE_ENGINE_FIELDS | DETECTOR_SHARED_ENGINE_FIELDS)
    tracking = payload["detector"].get("tracking")
    if isinstance(tracking, dict):
        payload["detector"]["tracking"] = _without_fields(tracking, TRACKING_SESSION_FIELDS | TRACKING_REID_ENGINE_FIELDS)
    return payload


def hot_config_changes(current: AppConfig, incoming: AppConfig) -> list[str]:
    changed = [field for field in sorted(HOT_CONFIG_FIELDS) if getattr(current, field) != getattr(incoming, field)]
    if {c.id: c.retention for c in current.cameras} != {c.id: c.retention for c in incoming.cameras} and "retention" not in changed:
        changed.append("retention")
    return changed


class TargetedConfigApplication:
    """Classify and atomically apply changes that do not rebuild the manager."""

    def __init__(self, *, lock: threading.RLock, save: Callable[..., None], active_exports: Callable[[], list[dict]], storage_error: Callable[[list[str]], Exception] = StorageTasksActiveError) -> None:
        self._lock = lock
        self._save = save
        self._active_exports = active_exports
        self._storage_error = storage_error

    def normalize(self, config: AppConfig, *, assign_ids: bool) -> AppConfig:
        return normalize_config(config.model_copy(deep=True), assign_ids=assign_ids)

    def apply(self, current: AppConfig, incoming: AppConfig, runtime: ConfigurableRuntime, *, persist: bool) -> tuple[AppConfig, dict[str, object]]:
        with self._lock:
            changes = hot_config_changes(current, incoming)
            mqtt_changed = current.mqtt != incoming.mqtt
            recorder_changes = [field for field in sorted(RECORDER_CONFIG_FIELDS) if getattr(current, field) != getattr(incoming, field)]
            retention_changed = "retention" in changes
            image_changed = "image_storage" in changes
            semantic_changed = "semantic_search" in changes
            policy_changed = any(getattr(current.detector, f) != getattr(incoming.detector, f) for f in DETECTOR_HOT_POLICY_FIELDS)
            tracking_changed = any(getattr(current.detector.tracking, f) != getattr(incoming.detector.tracking, f) for f in TRACKING_SESSION_FIELDS)
            roles: set[str] = set()
            if any(getattr(current.detector, f) != getattr(incoming.detector, f) for f in DETECTOR_OBJECT_ENGINE_FIELDS): roles.add("object")
            if any(getattr(current.detector, f) != getattr(incoming.detector, f) for f in DETECTOR_FACE_ENGINE_FIELDS): roles.add("face")
            if any(getattr(current.detector, f) != getattr(incoming.detector, f) for f in DETECTOR_SHARED_ENGINE_FIELDS): roles.update({"object", "face", "reid"})
            if any(getattr(current.detector.tracking, f) != getattr(incoming.detector.tracking, f) for f in TRACKING_REID_ENGINE_FIELDS): roles.add("reid")
            refresh = (("reid" in roles and current.detector.tracking != incoming.detector.tracking) or any(getattr(current.detector, f) != getattr(incoming.detector, f) for f in DETECTOR_OBJECT_TRACKING_RESET_FIELDS) or (tracking_changed and bool(roles)))
            if recorder_changes and (exports := self._active_exports()):
                kinds = sorted({str(job.get("kind") or "media") for job in exports})
                raise self._storage_error([f"media {'/'.join(kinds)} export"])
            if persist: self._save(incoming, assign_ids=False)
            runtime.config = incoming
            applied: list[str] = []
            try:
                steps = [
                    (bool(recorder_changes), "recorders", lambda c: runtime.reconfigure_recorders(c), lambda c: runtime.reconfigure_recorders(c)),
                    (mqtt_changed, "mqtt", lambda c: runtime.reconfigure_mqtt(c.mqtt), lambda c: runtime.reconfigure_mqtt(c.mqtt)),
                    (retention_changed, "retention", runtime.reconfigure_recording_retention, runtime.reconfigure_recording_retention),
                    (image_changed, "image_storage", lambda c: runtime.reconfigure_image_storage(c.image_storage), lambda c: runtime.reconfigure_image_storage(c.image_storage)),
                    (semantic_changed, "semantic_search", lambda c: runtime.reconfigure_semantic_search(c.semantic_search), lambda c: runtime.reconfigure_semantic_search(c.semantic_search)),
                    (bool(roles), "inference", lambda c: runtime.reconfigure_inference(c.detector, roles, refresh_tracking=refresh), lambda c: runtime.reconfigure_inference(c.detector, roles, refresh_tracking=refresh)),
                    (tracking_changed and not refresh, "tracking", lambda c: runtime.reconfigure_object_tracking(c.detector), lambda c: runtime.reconfigure_object_tracking(c.detector)),
                    (policy_changed, "policy", lambda c: runtime.reconfigure_detector_policy(c.detector), lambda c: runtime.reconfigure_detector_policy(c.detector)),
                ]
                for changed, name, forward, _rollback in steps:
                    if changed:
                        # Mark before invocation: a partially applied subsystem
                        # must receive its compensating configuration too.
                        applied.append(name)
                        forward(incoming)
            except BaseException:
                runtime.config = current
                for changed, name, _forward, rollback in reversed(steps):
                    if changed and name in applied:
                        try: rollback(current)
                        except Exception: LOGGER.exception("failed to roll back %s configuration", name)
                if persist:
                    try: self._save(current, assign_ids=False)
                    except Exception: LOGGER.exception("failed to restore persisted configuration after targeted apply failure")
                raise
            restarted = [name for name, changed in (("recorders", bool(recorder_changes)), ("mqtt", mqtt_changed), ("semantic_search", semantic_changed), ("tracking_sessions", tracking_changed or refresh)) if changed]
            restarted.extend(f"{role}_inference" for role in ("object", "face", "reid") if role in roles)
            hot = changes + recorder_changes + (["detector_policy"] if policy_changed else [])
            return incoming, {"apply_mode": "targeted" if restarted else "hot" if hot else "unchanged", "camera_workers_restarted": False, "subsystems_restarted": restarted, "hot_updated": hot}

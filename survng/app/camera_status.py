"""Read-model assembly for camera runtime and motion telemetry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .camera_capture import CameraCaptureService, FRAME_STALE_SECONDS
from .config import CameraConfig, MotionQualificationConfig
from .motion_analysis_service import MotionAnalysisService
from .motion_pipeline import (
    MotionDebugSnapshotStore,
    MotionEvidenceRepository,
    MotionPipeline,
)
from .onvif_events import OnvifEventListener


@dataclass(frozen=True, slots=True)
class CameraStatusHooks:
    runtime_state: Callable[[], tuple[bool, bool, str]]
    motion_settings: Callable[[], tuple[str, str, int]]
    stationary_object_tolerance: Callable[[], str]
    rescue_settings: Callable[[], tuple[bool, float]]
    visual_backup_settings: Callable[[], dict[str, float | int]]
    illumination_filter_enabled: Callable[[], bool]
    suppression_verification_rate: Callable[[], float]
    motion_stats: Callable[[], dict[str, Any]]
    object_tracking_status: Callable[[], dict[str, Any]]
    incident_status: Callable[[], dict[str, Any]]
    event_worker_running: Callable[[], bool]
    event_queue_depth: Callable[[], int]
    retry_queue_depth: Callable[[], int]
    event_runtime: Callable[[], dict[str, int]]
    lifecycle_runtime: Callable[[], dict[str, Any]]
    monotonic: Callable[[], float] = time.monotonic


class CameraStatusService:
    """Build the stable API status payload from injected runtime providers."""

    def __init__(
        self,
        *,
        camera: CameraConfig,
        motion_config: MotionQualificationConfig,
        capture: CameraCaptureService,
        motion_analysis: MotionAnalysisService,
        motion_evidence: MotionEvidenceRepository,
        onvif: OnvifEventListener,
        qualification_pipeline: MotionPipeline,
        observation_pipeline: MotionPipeline,
        fusion_pipeline: MotionPipeline,
        debug_store: MotionDebugSnapshotStore,
        pipeline_origins: dict[str, str],
        hooks: CameraStatusHooks,
    ) -> None:
        self.camera = camera
        self.motion_config = motion_config
        self.capture = capture
        self.motion_analysis = motion_analysis
        self.motion_evidence = motion_evidence
        self.onvif = onvif
        self.qualification_pipeline = qualification_pipeline
        self.observation_pipeline = observation_pipeline
        self.fusion_pipeline = fusion_pipeline
        self.debug_store = debug_store
        self.pipeline_origins = dict(pipeline_origins)
        self.hooks = hooks

    def snapshot(self) -> dict[str, Any]:
        enabled, detection_enabled, last_motion_at = self.hooks.runtime_state()
        capture = self.capture.status()
        analysis = self.motion_analysis.status()
        evidence = self.motion_evidence.status()
        mog2 = evidence.get("mog2", {})
        now = self.hooks.monotonic()
        live_age = self._age(capture["live_frame_monotonic"], now)
        main_age = self._age(capture["main_frame_monotonic"], now)
        connected = bool(
            enabled and live_age is not None and live_age <= FRAME_STALE_SECONDS
        )
        mode, sensitivity, frame_width = self.hooks.motion_settings()
        rescue_enabled, rescue_margin = self.hooks.rescue_settings()
        visual = self.hooks.visual_backup_settings()
        motion = {
            **self.hooks.motion_stats(),
            "mode": mode,
            "sensitivity": sensitivity,
            "stationary_object_tolerance": (
                self.hooks.stationary_object_tolerance()
            ),
            "illumination_filter_enabled": (
                self.hooks.illumination_filter_enabled()
            ),
            "frame_width": frame_width,
            "camera_mode_background_fps": (
                self.motion_config.camera_mode_background_fps
            ),
            "visual_backup": {
                "enabled": mode == "camera_rescue",
                "warmup_seconds": self.motion_config.visual_backup_warmup_seconds,
                "grace_seconds": visual["grace_seconds"],
                "minimum_score": visual["minimum_score"],
                "score_margin": self.motion_config.visual_backup_score_margin,
                "minimum_consecutive": visual["minimum_consecutive"],
                "cooldown_seconds": visual["cooldown_seconds"],
                "maximum_triggers_5m": visual["maximum_triggers_5m"],
                **self.motion_analysis.visual_backup_snapshot(),
            },
            "suppression_verification_rate": (
                self.hooks.suppression_verification_rate()
            ),
            "borderline_rescue_enabled": rescue_enabled,
            "borderline_margin": rescue_margin,
            "mog2_audit_enabled": bool(mog2.get("enabled", False)),
            "mog2_history_seconds": self.motion_config.mog2_history_seconds,
            "mog2_last": mog2.get("last"),
            "evidence_sources": evidence,
            "pipeline_origins": dict(self.pipeline_origins),
            "queue_depth": self.hooks.event_queue_depth(),
            "retry_queue_depth": self.hooks.retry_queue_depth(),
            "analysis_queue_depth": analysis["queue_depth"],
            "analysis_runtime": dict(analysis.get("telemetry") or {}),
            "analysis_worker_running": analysis["worker_running"],
            "event_worker_running": self.hooks.event_worker_running(),
            "event_runtime": self.hooks.event_runtime(),
            "continuous_last_result": analysis["continuous_last_result"],
            "buffered_frames": analysis["buffered_frames"],
            "frame_shape": analysis["frame_shape"],
            "color_buffered_frames": analysis["color_buffered_frames"],
            "color_frame_shape": analysis["color_frame_shape"],
            "pipeline": self.qualification_pipeline.status(),
            "observation_pipeline": self.observation_pipeline.status(),
            "fusion_pipeline": self.fusion_pipeline.status(),
            "debug": self.debug_store.status(),
        }
        return {
            "id": self.camera.id,
            "name": self.camera.name,
            "running": enabled,
            "connected": connected,
            "capture_running": bool(capture["live_running"]),
            "frame_fresh": connected,
            "last_frame_age_seconds": self._rounded(live_age),
            "main_running": bool(capture["main_running"]),
            "main_frame_fresh": bool(
                main_age is not None and main_age <= FRAME_STALE_SECONDS
            ),
            "main_last_frame_age_seconds": self._rounded(main_age),
            "last_frame_at": str(capture["live_frame_at"]),
            "main_last_frame_at": str(capture["main_frame_at"]),
            "last_error": capture["last_error"],
            "main_last_error": capture["main_error"],
            "capture_stats": capture["capture_stats"],
            "stream_dimensions": capture["stream_dimensions"],
            "onvif_enabled": self.camera.onvif.enabled,
            "onvif_connected": self.onvif.connected,
            "onvif_last_event_at": self.onvif.last_event_at,
            "onvif_last_camera_event_at": self.onvif.last_camera_event_at,
            "onvif_last_motion_event_at": self.onvif.last_motion_event_at,
            "last_motion_at": last_motion_at,
            "detection_enabled": detection_enabled,
            "object_tracking": {
                **self.hooks.object_tracking_status(),
                **self.hooks.incident_status(),
            },
            **self._onvif_diagnostics(),
            "motion_qualification": motion,
            "lifecycle": self.hooks.lifecycle_runtime(),
        }

    @staticmethod
    def _age(frame_clock: float | None, now: float) -> float | None:
        return max(0.0, now - frame_clock) if frame_clock is not None else None

    @staticmethod
    def _rounded(age: float | None) -> float | None:
        return round(age, 3) if age is not None else None

    def _onvif_diagnostics(self) -> dict[str, Any]:
        fields = (
            "last_error", "last_connected_at", "last_poll_success_at",
            "last_poll_error", "last_poll_error_at", "retry_attempts",
            "poll_timeouts", "poll_errors", "resubscriptions",
            "notifications_received", "motion_events_received",
            "inactive_motion_events", "unrecognized_notifications",
            "callback_errors", "renewal_attempts", "renewals",
            "renewal_errors", "last_renewed_at", "subscription_current_time",
            "subscription_termination_time", "subscription_lifetime_seconds",
        )
        return {
            f"onvif_{field}": getattr(self.onvif, field)
            for field in fields
        }

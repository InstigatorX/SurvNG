from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from survng.app.camera_status import CameraStatusService
from survng.app.config import CameraConfig, MotionQualificationConfig


def _service(*, enabled: bool = True, live_clock: float | None = 99.0):
    capture = Mock()
    capture.status.return_value = {
        "live_frame_at": "live-at",
        "main_frame_at": "main-at",
        "live_frame_monotonic": live_clock,
        "main_frame_monotonic": 98.0,
        "live_running": True,
        "main_running": True,
        "last_error": "",
        "main_error": "",
        "capture_stats": {"live": {"fps": 20.0}},
        "stream_dimensions": {"live": {"width": 640, "height": 480}},
    }
    analysis = Mock()
    analysis.status.return_value = {
        "queue_depth": 1,
        "worker_running": True,
        "continuous_last_result": {"accepted": True},
        "buffered_frames": 8,
        "frame_shape": [480, 640],
        "color_buffered_frames": 3,
        "color_frame_shape": [480, 640, 3],
    }
    analysis.visual_backup_snapshot.return_value = {"scene_ready": True}
    evidence = Mock()
    evidence.status.return_value = {
        "onvif": {"enabled": True, "last": {"active": True}}
    }
    onvif_values = {
        "connected": True,
        "last_event_at": "event-at",
        "last_camera_event_at": "camera-at",
        "last_motion_event_at": "motion-at",
        "last_error": "",
        "last_connected_at": "connected-at",
        "last_poll_success_at": "poll-at",
        "last_poll_error": "",
        "last_poll_error_at": "",
        "retry_attempts": 1,
        "poll_timeouts": 2,
        "poll_errors": 3,
        "resubscriptions": 4,
        "notifications_received": 5,
        "motion_events_received": 6,
        "inactive_motion_events": 7,
        "unrecognized_notifications": 8,
        "callback_errors": 9,
        "renewal_attempts": 10,
        "renewals": 11,
        "renewal_errors": 12,
        "last_renewed_at": "renewed-at",
        "subscription_current_time": "current-at",
        "subscription_termination_time": "termination-at",
        "subscription_lifetime_seconds": 3600.0,
    }
    onvif = SimpleNamespace(**onvif_values)
    pipelines = [Mock(), Mock(), Mock()]
    for index, pipeline in enumerate(pipelines):
        pipeline.status.return_value = {"pipeline": index}
    debug = Mock()
    debug.status.return_value = {"enabled": False}
    runtime_state = SimpleNamespace(
        lock=threading.Lock(),
        enabled=enabled,
        detection_enabled=True,
    )
    motion_state = Mock()
    motion_state.last_motion_at.return_value = "last-motion"
    motion_state.stats_snapshot.return_value = {"triggers": 4}
    qualification = Mock()
    qualification.settings.return_value = ("camera_rescue", "balanced", 640)
    qualification.stationary_object_tolerance.return_value = "balanced"
    qualification.rescue_settings.return_value = (True, 0.03)
    qualification.visual_backup_settings.return_value = {
            "grace_seconds": 2.0,
            "minimum_score": 0.75,
            "minimum_consecutive": 4,
            "cooldown_seconds": 30.0,
            "maximum_triggers_5m": 2,
        }
    qualification.illumination_filter_enabled.return_value = False
    qualification.suppression_verification_rate.return_value = 0.05
    motion_runtime = Mock()
    motion_runtime.runtime_status.return_value = {
        "event_worker_running": True,
        "event_queue_depth": 2,
        "retry_queue_depth": 1,
        "generation_clean": True,
        "events": {"queue_high_water": 3},
    }
    tracking = Mock()
    tracking.status.return_value = {"active": True}
    incidents = Mock()
    incidents.status.return_value = {"handoff_failures": 0}
    lifecycle = Mock()
    lifecycle.runtime_status.return_value = {"active_worker_count": 4}
    service = CameraStatusService(
        camera=CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://example.invalid/main",
        ),
        motion_config=MotionQualificationConfig(),
        capture=capture,
        motion_analysis=analysis,
        motion_evidence=evidence,
        onvif=onvif,
        qualification_pipeline=pipelines[0],
        observation_pipeline=pipelines[1],
        fusion_pipeline=pipelines[2],
        debug_store=debug,
        pipeline_origins={"qualification": "default"},
        runtime_state=runtime_state,
        motion_state=motion_state,
        qualification=qualification,
        motion_runtime=motion_runtime,
        object_tracking=tracking,
        incidents=incidents,
        lifecycle=lifecycle,
        monotonic=lambda: 100.0,
    )
    return service


def test_status_snapshot_preserves_api_shape_and_dynamic_subsystem_state() -> None:
    status = _service().snapshot()

    assert status["connected"] is True
    assert status["last_frame_age_seconds"] == 1.0
    assert status["object_tracking"] == {
        "active": True,
        "handoff_failures": 0,
    }
    assert status["lifecycle"]["active_worker_count"] == 4
    assert status["onvif_poll_timeouts"] == 2
    assert status["onvif_subscription_lifetime_seconds"] == 3600.0
    motion = status["motion_qualification"]
    assert motion["triggers"] == 4
    assert motion["analysis_worker_running"] is True
    assert motion["event_worker_running"] is True
    assert motion["generation_clean"] is True
    assert motion["event_runtime"]["queue_high_water"] == 3
    assert motion["visual_backup"]["scene_ready"] is True
    assert motion["evidence_sources"]["onvif"]["enabled"] is True


def test_disabled_or_stale_camera_is_not_reported_connected() -> None:
    assert _service(enabled=False).snapshot()["connected"] is False
    assert _service(live_clock=0.0).snapshot()["connected"] is False

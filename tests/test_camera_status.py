from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from survng.app.camera_status import CameraStatusHooks, CameraStatusService
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
        "mog2": {"enabled": True, "last": {"score": 0.7}}
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
    hooks = CameraStatusHooks(
        runtime_state=lambda: (enabled, True, "last-motion"),
        motion_settings=lambda: ("camera_rescue", "balanced", 640),
        stationary_object_tolerance=lambda: "balanced",
        rescue_settings=lambda: (True, 0.03),
        visual_backup_settings=lambda: {
            "grace_seconds": 2.0,
            "minimum_score": 0.75,
            "minimum_consecutive": 4,
            "cooldown_seconds": 30.0,
            "maximum_triggers_5m": 2,
        },
        illumination_filter_enabled=lambda: False,
        suppression_verification_rate=lambda: 0.05,
        motion_stats=lambda: {"triggers": 4},
        object_tracking_status=lambda: {"active": True},
        incident_status=lambda: {"handoff_failures": 0},
        event_worker_running=lambda: True,
        event_queue_depth=lambda: 2,
        retry_queue_depth=lambda: 1,
        event_runtime=lambda: {"queue_high_water": 3},
        lifecycle_runtime=lambda: {"active_worker_count": 4},
        monotonic=lambda: 100.0,
    )
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
        hooks=hooks,
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
    assert motion["event_runtime"]["queue_high_water"] == 3
    assert motion["visual_backup"]["scene_ready"] is True
    assert motion["mog2_audit_enabled"] is True


def test_disabled_or_stale_camera_is_not_reported_connected() -> None:
    assert _service(enabled=False).snapshot()["connected"] is False
    assert _service(live_clock=0.0).snapshot()["connected"] is False

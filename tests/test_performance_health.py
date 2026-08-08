from __future__ import annotations

from survng.app.performance_health import camera_performance_health


def _camera(**overrides):
    camera = {
        "expected_enabled": True,
        "capture": {"live": {"observer_calls": 100, "observer_p95_ms": 1.0}},
        "motion": {
            "sample_fps": 5.0,
            "analysis_wait_ms_p95": 20.0,
            "analysis_runtime": {
                "raw_frames_submitted": 100,
                "mailbox_replacements": 5,
                "capture_to_analysis_count": 100,
                "capture_to_analysis_p95_ms": 40.0,
                "copy_mb_per_second": 0.0,
            },
            "event_runtime": {"queue_high_water": 2, "queue_capacity": 32},
        },
    }
    camera.update(overrides)
    return camera


def test_healthy_camera_stays_below_every_gate() -> None:
    health = camera_performance_health(_camera())

    assert health["status"] == "healthy"
    assert health["summary"] == "Capture and motion processing are keeping up"
    assert all(check["status"] == "healthy" for check in health["checks"])


def test_latency_thresholds_scale_with_configured_sample_rate() -> None:
    slow_rate = _camera()
    slow_rate["motion"]["sample_fps"] = 2.0
    slow_rate["motion"]["analysis_runtime"]["capture_to_analysis_p95_ms"] = 700.0
    fast_rate = _camera()
    fast_rate["motion"]["sample_fps"] = 10.0
    fast_rate["motion"]["analysis_runtime"]["capture_to_analysis_p95_ms"] = 700.0

    slow_check = next(
        check
        for check in camera_performance_health(slow_rate)["checks"]
        if check["key"] == "capture_to_analysis_p95_ms"
    )
    fast_check = next(
        check
        for check in camera_performance_health(fast_rate)["checks"]
        if check["key"] == "capture_to_analysis_p95_ms"
    )

    assert slow_check["status"] == "healthy"
    assert fast_check["status"] == "attention"


def test_multiple_pressure_signals_produce_critical_summary() -> None:
    camera = _camera()
    camera["capture"]["live"]["observer_p95_ms"] = 20.0
    camera["motion"]["analysis_runtime"].update({
        "mailbox_replacements": 80,
        "copy_mb_per_second": 90.0,
    })
    camera["motion"]["event_runtime"]["queue_high_water"] = 30

    health = camera_performance_health(camera)

    assert health["status"] == "critical"
    assert health["summary"] == "Processing is falling materially behind"
    critical = {check["key"] for check in health["checks"] if check["status"] == "critical"}
    assert critical == {
        "capture_observer_p95_ms",
        "mailbox_replacement_percent",
        "event_queue_high_water_percent",
        "motion_copy_mb_per_second",
    }


def test_new_and_intentionally_paused_cameras_are_not_false_alarms() -> None:
    warming = _camera()
    warming["capture"]["live"]["observer_calls"] = 2
    warming["motion"]["analysis_runtime"]["raw_frames_submitted"] = 2
    warming["motion"]["analysis_runtime"]["capture_to_analysis_count"] = 2

    assert camera_performance_health(warming)["status"] == "warming_up"
    assert camera_performance_health({"expected_enabled": False}) == {
        "status": "paused",
        "summary": "Camera is intentionally paused",
        "checks": [],
    }

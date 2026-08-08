"""Actionable health gates derived from existing camera runtime telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PerformanceCheck:
    key: str
    label: str
    status: str
    value: float
    unit: str
    warning: float
    critical: float


def _status(value: float, warning: float, critical: float) -> str:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "attention"
    return "healthy"


def camera_performance_health(camera: dict[str, Any]) -> dict[str, Any]:
    """Turn low-level counters into stable gates scaled to EMA sample rate."""
    if not bool(camera.get("expected_enabled", True)):
        return {
            "status": "paused",
            "summary": "Camera is intentionally paused",
            "checks": [],
        }
    if not bool(camera.get("connected")) or not bool(camera.get("frame_fresh")):
        return {
            "status": "offline",
            "summary": "No fresh camera frames are available",
            "checks": [],
        }
    if not bool(camera.get("detection_enabled", True)):
        return {
            "status": "paused",
            "summary": "Object and motion processing are intentionally paused",
            "checks": [],
        }

    motion = dict(camera.get("motion") or {})
    analysis = dict(motion.get("analysis_runtime") or {})
    event_runtime = dict(motion.get("event_runtime") or {})
    live = dict((camera.get("capture") or {}).get("live") or {})
    sample_fps = max(0.1, float(motion.get("sample_fps") or 5.0))
    sample_interval_ms = 1000.0 / sample_fps
    submitted = int(analysis.get("raw_frames_submitted") or 0)
    replacements = int(analysis.get("mailbox_replacements") or 0)
    replacement_percent = (
        min(100.0, replacements / submitted * 100.0) if submitted else 0.0
    )
    queue_capacity = max(1, int(event_runtime.get("queue_capacity") or 32))
    queue_percent = min(
        100.0,
        float(event_runtime.get("queue_high_water") or 0) / queue_capacity * 100.0,
    )
    definitions = (
        ("capture_observer_p95_ms", "Capture handoff latency", float(live.get("observer_p95_ms") or 0.0), "ms", 5.0, 15.0),
        ("capture_to_analysis_p95_ms", "Capture-to-analysis latency", float(analysis.get("capture_to_analysis_p95_ms") or 0.0), "ms", max(250.0, sample_interval_ms * 1.5), max(750.0, sample_interval_ms * 3.0)),
        ("analysis_wait_p95_ms", "EMA capacity wait", float(motion.get("analysis_wait_ms_p95") or 0.0), "ms", max(250.0, sample_interval_ms * 2.0), max(1000.0, sample_interval_ms * 5.0)),
        ("mailbox_replacement_percent", "Stale-frame replacement rate", replacement_percent, "%", 35.0, 70.0),
        ("event_queue_high_water_percent", "Motion event queue peak", queue_percent, "%", 50.0, 80.0),
        ("motion_copy_mb_per_second", "Motion frame copy rate", float(analysis.get("copy_mb_per_second") or 0.0), "MB/s", 25.0, 75.0),
    )
    checks = [
        PerformanceCheck(
            key=key,
            label=label,
            status=_status(value, warning, critical),
            value=round(value, 3),
            unit=unit,
            warning=round(warning, 3),
            critical=round(critical, 3),
        )
        for key, label, value, unit, warning, critical in definitions
    ]
    sample_count = max(
        int(live.get("observer_calls") or 0),
        int(analysis.get("capture_to_analysis_count") or 0),
        submitted,
    )
    if sample_count < 20:
        overall = "warming_up"
        summary = "Collecting a representative processing sample"
    elif any(check.status == "critical" for check in checks):
        overall = "critical"
        summary = "Processing is falling materially behind"
    elif any(check.status == "attention" for check in checks):
        overall = "attention"
        summary = "Processing is current but under pressure"
    else:
        overall = "healthy"
        summary = "Capture and motion processing are keeping up"
    return {
        "status": overall,
        "summary": summary,
        "sample_count": sample_count,
        "checks": [asdict(check) for check in checks],
    }

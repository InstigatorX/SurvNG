"""Deterministic cross-boundary replay for the motion-to-incident path."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from survng.app.config import CameraConfig, DetectionZone
from survng.app.motion import MotionQualificationResult
from survng.app.motion_coordinator import (
    VisualBackupAction,
    VisualBackupPolicy,
    VisualBackupReplaySample,
    replay_visual_backup,
)
from survng.app.motion_pipeline import MotionDecisionHandler
from survng.app.zones import apply_detection_zones


class _ReplayEvents:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def add_event(self, **payload: Any) -> dict[str, Any]:
        event = {"id": len(self.events) + 1, **payload}
        self.events.append(event)
        return event

    def add_motion_audit(self, **payload: Any) -> dict[str, Any]:
        return {"id": 1, **payload}


def _qualification(accepted: bool, score: float, reason: str) -> MotionQualificationResult:
    return MotionQualificationResult(
        accepted=accepted,
        score=score,
        threshold=0.5,
        reason=reason,
        frame_count=4,
        features={},
        telemetry={},
    )


def _replay_scenario() -> dict[str, Any]:
    policy = VisualBackupPolicy(
        warmup_seconds=0.0,
        grace_seconds=1.0,
        minimum_score=0.7,
        score_margin=0.1,
        minimum_consecutive=3,
        cooldown_seconds=15.0,
        maximum_triggers_5m=2,
        sample_fps=2.0,
        background_fps=2.0,
    )
    stable = _qualification(False, 0.0, "no_motion_blobs")
    credible = _qualification(True, 0.82, "credible_motion")
    samples = tuple(
        VisualBackupReplaySample(at, stable)
        for at in (90.0, 90.75, 91.5)
    ) + tuple(
        VisualBackupReplaySample(at, credible)
        for at in (100.0, 100.5, 101.0)
    )
    decisions = replay_visual_backup(policy, samples)
    assert decisions[-1].action == VisualBackupAction.READY

    camera = CameraConfig(
        id="gate",
        name="Gate",
        stream_url="rtsp://camera.invalid/main",
        zones=[
            DetectionZone(
                name="Driveway",
                behavior="incident",
                object_classes=["car"],
                points=[
                    {"x": 0.0, "y": 0.5},
                    {"x": 1.0, "y": 0.5},
                    {"x": 1.0, "y": 1.0},
                    {"x": 0.0, "y": 1.0},
                ],
            )
        ],
    )
    objects = [
        {
            "label": "car",
            "confidence": 0.91,
            "box": {"x1": 200, "y1": 150, "x2": 440, "y2": 350},
        }
    ]
    apply_detection_zones(camera, objects, 640, 360, 0.35)
    events = _ReplayEvents()
    published: list[tuple[str, dict[str, Any]]] = []
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    handler = MotionDecisionHandler(
        camera_id="gate",
        events=events,
        detection_provider=lambda _at: (frame, objects, "recording.mp4"),
        snapshot_writer=lambda _frame, _at: "snapshot.webp",
        object_serializer=lambda value: json.dumps(value, separators=(",", ":")),
        event_callback=lambda kind, payload: published.append((kind, payload)),
    )
    outcome = handler.handle(
        "ema/visual-backup",
        "credible motion",
        datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
        {"trigger_origin": "ema"},
    )
    return {
        "actions": [decision.action.value for decision in decisions],
        "event_id": outcome.event_id,
        "object_detected": outcome.object_detected,
        "zones": objects[0]["zones"],
        "incident_eligible": objects[0]["incident_eligible"],
        "published": published,
        "stored_camera": events.events[0]["camera_id"],
    }


def test_motion_to_object_incident_replay_is_deterministic() -> None:
    first = _replay_scenario()
    second = _replay_scenario()

    assert first == second
    assert first["event_id"] == 1
    assert first["object_detected"] is True
    assert first["zones"] == ["Driveway"]
    assert first["incident_eligible"] is True
    assert [kind for kind, _payload in first["published"]] == ["incident", "object"]

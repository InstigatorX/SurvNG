from __future__ import annotations

import pytest

from survng.app.domain_events import (
    IncidentCreated,
    MotionObserved,
    ObjectDetected,
    TrackingCompleted,
)


def test_motion_observed_serializes_canonical_boundary_payload() -> None:
    event = MotionObserved("gate", "2026-08-07T12:00:00+00:00", "onvif")
    assert event.to_payload() == {
        "camera_id": "gate",
        "timestamp": "2026-08-07T12:00:00+00:00",
        "source": "onvif",
    }


def test_incident_created_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError, match="event_id"):
        IncidentCreated(0, "gate", "2026-08-07T12:00:00+00:00")
    with pytest.raises(ValueError, match="camera_id"):
        IncidentCreated(1, "", "2026-08-07T12:00:00+00:00")


def test_object_detected_copies_mutable_objects_at_boundary() -> None:
    detected = {"label": "person", "score": 0.9, "box": {"x1": 1}}
    event = ObjectDetected(
        event_id=4,
        camera_id="front-door",
        timestamp="2026-08-07T12:00:00+00:00",
        objects=(detected,),
    )
    detected["label"] = "car"
    detected["box"]["x1"] = 99
    assert event.to_payload()["objects"] == [
        {"label": "person", "score": 0.9, "box": {"x1": 1}}
    ]
    assert "incident_objects" not in event.to_payload()


def test_tracking_completed_reserves_identity_fields_from_details() -> None:
    event = TrackingCompleted(
        event_id=7,
        camera_id="gate",
        state="complete",
        details={"event_id": 99, "camera_id": "wrong", "state": "failed", "tracks": []},
    )
    assert event.to_payload() == {
        "event_id": 7,
        "camera_id": "gate",
        "state": "complete",
        "tracks": [],
    }

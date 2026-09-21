from unittest.mock import Mock

import pytest

from survng.app.config import CameraConfig, DetectorConfig, NativeStationaryConfig
from survng.app.live_detections import DetectionSnapshot
from survng.app.native_activity import NativeActivity
from survng.app.native_motion import NativeMotion


def box(x=10, scale=1):
    return {"x1": x * scale, "y1": 10 * scale, "x2": (x + 20) * scale, "y2": 30 * scale}


def obj(label="car", x=10, provenance="native_fresh_detection"):
    return {
        "label": label,
        "box": box(x),
        "confidence": 0.95,
        "detection_provenance": provenance,
    }


def activity(**native):
    """Build activity with production stationary labels (includes person)."""
    events = Mock()
    events.add_event.side_effect = [{"id": n} for n in range(1, 20)]
    events.open_incident = Mock(
        side_effect=[{"id": n, "observation_count": 1} for n in range(100, 120)]
    )
    # Production default: person is gated like vehicles. Do not empty labels.
    stationary = {"enabled": True, **native.pop("stationary", {})}
    detector = DetectorConfig(
        enabled=True,
        event_confirmation_frames=1,
        native={
            "stationary": stationary,
            **native,
        },
    )
    return NativeActivity(
        CameraConfig(id="test", name="Test", stream_url="rtsp://example.test/live"),
        detector,
        events,
        Mock(),
        Mock(return_value=""),
    )


def feed(activity, sequence, pts, objects, session="one"):
    snap = DetectionSnapshot(
        pts,
        sequence,
        100,
        100,
        tuple(objects),
        session,
        "native_fresh_detection",
        100 + pts,
    )
    activity.consume(snap, now=100 + pts, epoch=1000 + pts)


@pytest.mark.parametrize("fps", [2, 5])
def test_parked_vehicle_is_context_without_incident(fps):
    a = activity()
    for seq in range(1, 30 * fps):
        feed(a, seq, seq / fps, [obj(x=10 + (seq % 3 - 1) * 0.1)])
    a.events.add_event.assert_not_called()
    track = next(t for t in a.tracks.values() if t["label"] == "car")
    assert track["motion_state"] == "stationary"
    assert track["activity_eligible"] is False
    assert a.health == "healthy"


def test_moving_vehicle_opens_incident():
    a = activity()
    for seq in range(1, 21):
        feed(a, seq, seq / 5, [obj(x=10 + seq)])
    assert a.event_id == 1


def test_moving_vehicle_parks_and_completes():
    a = activity()
    for seq in range(1, 21):
        feed(a, seq, seq / 5, [obj(x=10 + seq)])
    assert a.event_id == 1
    for seq in range(21, 151):
        feed(a, seq, seq / 5, [obj(x=30)])
    assert a.event_id is None
    assert a.events.add_event.call_count == 1
    track = next(t for t in a.tracks.values() if t["label"] == "car")
    assert track["motion_state"] == "stationary"


def test_standing_person_is_context_without_incident():
    """Production gates person: standing/jittery boxes must not open incidents."""
    a = activity()
    assert "person" in a.config.native.stationary.labels
    for seq in range(1, 61):
        feed(a, seq, seq / 5, [obj("person", x=10 + (seq % 3 - 1) * 0.1)])
    a.events.add_event.assert_not_called()
    track = next(t for t in a.tracks.values() if t["label"] == "person")
    assert track["motion_state"] == "stationary"
    assert track["activity_eligible"] is False


def test_moving_person_opens_incident():
    a = activity()
    for seq in range(1, 21):
        feed(a, seq, seq / 5, [obj("person", x=10 + seq)])
    assert a.event_id == 1
    track = next(t for t in a.tracks.values() if t["label"] == "person")
    assert track["motion_state"] == "moving"
    assert track["activity_eligible"] is True


def test_moving_person_not_suppressed_by_parked_car():
    a = activity()
    for seq in range(1, 61):
        feed(a, seq, seq / 5, [obj()])
    for seq in range(61, 81):
        feed(a, seq, seq / 5, [obj(), obj("person", x=40 + (seq - 61))])
    assert a.event_id == 1
    participants = a.events.update_native_incident_state.call_args.args[2]
    assert "person" in {item["label"] for item in participants}
    assert any(
        item["label"] == "person" and item.get("activity_eligible")
        for item in participants
    )


def test_predicted_motion_cannot_wake_stationary_vehicle():
    a = activity()
    for seq in range(1, 61):
        feed(a, seq, seq / 5, [obj()])
    for seq in range(61, 71):
        feed(a, seq, seq / 5, [obj(x=seq, provenance="native_tracked_prediction")])
    a.events.add_event.assert_not_called()


def test_suppression_can_be_disabled():
    a = activity()
    a.config.native.stationary.enabled = False
    feed(a, 1, 0.2, [obj()])
    assert a.event_id == 1


def test_soft_assoc_rematch_inherits_moving_state():
    """Driveway chase: identity flicker must not reset motion to uncertain."""
    a = activity()
    for seq in range(1, 21):
        feed(a, seq, seq / 5, [obj(x=10 + seq)])
    assert a.event_id == 1
    keys_before = set(a.registry.tracks)
    # Jump farther than soft-assoc nearness so a new assoc id is issued.
    feed(a, 21, 21 / 5, [obj(x=80)])
    new_keys = set(a.registry.tracks) - keys_before
    assert new_keys
    new_key = next(iter(new_keys))
    state = a._activity_states[new_key]
    assert state["motion_state"] == "moving"
    assert state["activity_eligible"] is True
    assert a.counts["motion_rematch_adoptions"] >= 1
    assert a.event_id == 1


def test_soft_assoc_rematch_keeps_parked_suppressed():
    a = activity()
    for seq in range(1, 61):
        feed(a, seq, seq / 5, [obj(x=10 + (seq % 3 - 1) * 0.1)])
    a.events.add_event.assert_not_called()
    keys_before = set(a.registry.tracks)
    feed(a, 61, 61 / 5, [obj(x=80)])
    new_keys = set(a.registry.tracks) - keys_before
    assert new_keys
    new_key = next(iter(new_keys))
    state = a._activity_states[new_key]
    assert state["activity_eligible"] is False
    a.events.add_event.assert_not_called()


def test_whole_window_detects_out_and_back_and_scales_with_box():
    for scale in (1, 10):
        motion = NativeMotion()
        states = [
            motion.update(box(x, scale), i * 0.5, NativeStationaryConfig(), 2)
            for i, x in enumerate((10, 14, 18, 14, 10))
        ]
        assert "moving" in states


def test_single_box_outlier_does_not_wake_stationary_track():
    motion = NativeMotion()
    policy = NativeStationaryConfig()
    for i in range(100):
        motion.update(box(), i * 0.2, policy, 2)
    assert motion.state == "stationary"
    assert motion.update(box(50), 20, policy, 2) == "stationary"

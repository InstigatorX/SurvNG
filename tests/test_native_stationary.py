from unittest.mock import Mock

from survng.app.config import CameraConfig, DetectorConfig
from survng.app.live_detections import DetectionSnapshot
from survng.app.native_activity import NativeActivity


def box(x=10, scale=1):
    return {"x1": x * scale, "y1": 10 * scale, "x2": (x + 20) * scale, "y2": 30 * scale}


def obj(label="car", x=10, provenance="native_fresh_detection"):
    return {
        "label": label,
        "box": box(x),
        "confidence": 0.95,
        "detection_provenance": provenance,
    }


def activity():
    events = Mock()
    events.add_event.side_effect = [{"id": n} for n in range(1, 20)]
    events.open_incident = Mock(
        side_effect=[{"id": n, "observation_count": 1} for n in range(100, 120)]
    )
    return NativeActivity(
        CameraConfig(id="test", name="Test", stream_url="rtsp://example.test/live"),
        DetectorConfig(enabled=True),
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


def test_parked_vehicle_is_scene_activity_under_minimal_policy():
    """Stationary motion no longer gates admission; presence opens an incident."""
    a = activity()
    feed(a, 1, 0.2, [obj(x=10)])
    assert a.event_id == 1
    track = next(t for t in a.tracks.values() if t["label"] == "car")
    assert track["activity_eligible"] is True
    assert track["motion_state"] == "presence"


def test_person_and_car_share_one_scene_incident():
    a = activity()
    feed(a, 1, 0.2, [obj(), obj("person", x=40)])
    assert a.event_id == 1
    participants = a.events.update_native_incident_state.call_args.args[2]
    assert {item["label"] for item in participants} == {"person", "car"}


def test_predicted_motion_cannot_open_scene_incident():
    a = activity()
    feed(a, 1, 0.2, [obj(provenance="native_tracked_prediction")])
    a.events.add_event.assert_not_called()

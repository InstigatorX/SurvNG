from unittest.mock import Mock

import pytest
from pydantic import ValidationError

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


@pytest.fixture
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


def test_parked_vehicle_is_scene_activity_under_minimal_policy(activity):
    """Stationary motion no longer gates admission; presence opens an incident."""
    feed(activity, 1, 0.2, [obj(x=10)])
    assert activity.event_id == 1
    track = next(t for t in activity.tracks.values() if t["label"] == "car")
    assert track["activity_eligible"] is True
    assert track["motion_state"] == "presence"


def test_person_and_car_share_one_scene_incident(activity):
    feed(activity, 1, 0.2, [obj(), obj("person", x=40)])
    assert activity.event_id == 1
    participants = activity.events.update_native_incident_state.call_args.args[2]
    assert {item["label"] for item in participants} == {"person", "car"}


def test_predicted_motion_cannot_open_scene_incident(activity):
    feed(activity, 1, 0.2, [obj(provenance="native_tracked_prediction")])
    activity.events.add_event.assert_not_called()


def test_small_person_translation_still_becomes_moving():
    motion = NativeMotion()
    policy = NativeStationaryConfig()
    states = []
    for i in range(12):
        detected = {"x1": 490 + i * 2, "y1": 263, "x2": 508 + i * 2, "y2": 307}
        states.append(motion.update(detected, i * 0.2, policy, 2))
    assert "moving" in states


def test_stationary_policy_includes_people_by_default():
    assert "person" in NativeStationaryConfig().labels


def test_scale_change_without_translation_still_becomes_moving():
    motion = NativeMotion()
    policy = NativeStationaryConfig()
    states = []
    for i in range(12):
        width = 18 + i * 1.5
        height = 44 + i * 3
        detected = {
            "x1": 500 - width / 2,
            "y1": 300 - height / 2,
            "x2": 500 + width / 2,
            "y2": 300 + height / 2,
        }
        states.append(motion.update(detected, i * 0.2, policy, 2))
    assert "moving" in states


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


def test_gap_resets_moving_evidence():
    motion = NativeMotion()
    policy = NativeStationaryConfig()
    for i in range(10):
        motion.update(box(10 + i), i * 0.2, policy, 2)
    assert motion.state == "moving"
    assert motion.update(box(60), 10, policy, 2) == "uncertain"


def test_stationary_thresholds_have_hysteresis():
    with pytest.raises(ValidationError, match="below"):
        NativeStationaryConfig(stationary_threshold=0.2, moving_threshold=0.1)

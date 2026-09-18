from dataclasses import replace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from survng.app.config import CameraConfig, DetectorConfig, NativeStationaryConfig
from survng.app.live_detections import DetectionSnapshot
from survng.app.native_activity import NativeActivity
from survng.app.native_motion import NativeMotion


def box(x=10, scale=1):
    return {"x1": x * scale, "y1": 10 * scale, "x2": (x + 20) * scale, "y2": 30 * scale}


def obj(label="car", x=10, native_id=7, provenance="native_fresh_detection"):
    return {"label": label, "box": box(x), "confidence": .95, "native_track_id": native_id,
            "detection_provenance": provenance}


@pytest.fixture
def activity():
    events = Mock()
    events.add_event.side_effect = [{"id": n} for n in range(1, 20)]
    return NativeActivity(CameraConfig(id="test", name="Test", stream_url="rtsp://example.test/live"),
                          DetectorConfig(enabled=True), events, Mock(), Mock(return_value=""))


def feed(activity, sequence, pts, objects, session="one"):
    snap = DetectionSnapshot(pts, sequence, 100, 100, tuple(objects), session,
                             "native_fresh_detection", 100 + pts)
    activity.consume(snap, now=100 + pts, epoch=1000 + pts)


@pytest.mark.parametrize("fps", [2, 5])
def test_parked_vehicle_is_context_without_incident_at_different_rates(activity, fps):
    for seq in range(1, 30 * fps):
        feed(activity, seq, seq / fps, [obj(x=10 + (seq % 3 - 1) * .1)])
    activity.events.add_event.assert_not_called()
    assert activity.tracks[(7, "car")]["motion_state"] == "stationary"
    assert activity.tracks[(7, "car")]["confirmed"]
    assert not activity.status()["tracks"][0]["activity_eligible"]
    assert activity.health == "healthy"


def test_moving_vehicle_parks_completes_and_departure_starts_new_episode(activity):
    for seq in range(1, 21):
        feed(activity, seq, seq / 5, [obj(x=10 + seq)])
    assert activity.event_id == 1
    for seq in range(21, 151):
        feed(activity, seq, seq / 5, [obj(x=30)])
    assert activity.event_id is None
    assert activity.events.add_event.call_count == 1
    assert activity.tracks[(7, "car")]["motion_state"] == "stationary"
    assert activity.events.update_native_incident_state.call_args.args[1]["state"] == "complete"
    for seq in range(151, 161):
        feed(activity, seq, seq / 5, [obj(x=30 + seq - 150)])
    assert activity.event_id == 2
    stored = activity.events.update_native_incident_state.call_args.args[1]["tracks"][0]
    assert stored["first_seen"] >= "1970-01-01T00:17:10"
    assert stored["box_history"][0][0] >= 1030


def test_moving_person_is_not_suppressed_or_polluted_by_parked_car(activity):
    for seq in range(1, 61):
        feed(activity, seq, seq / 5, [obj()])
    for seq in range(61, 81):
        feed(activity, seq, seq / 5, [obj(), obj("person", x=10 + (seq - 60) * 2, native_id=8)])
    assert activity.event_id == 1
    tracks = activity.events.update_native_incident_state.call_args.args[1]["tracks"]
    assert {t["label"] for t in tracks} == {"person", "car"}
    assert next(t for t in tracks if t["label"] == "car")["activity_eligible"] is False
    for seq in range(81, 121):
        feed(activity, seq, seq / 5, [obj()])
    assert activity.event_id is None
    assert activity.tracks[(7, "car")]["motion_state"] == "stationary"
    assert activity.events.add_event.call_count == 1


def test_stationary_id_change_and_reconnect_do_not_invent_movement(activity):
    for seq in range(1, 61):
        feed(activity, seq, seq / 5, [obj()])
    for seq in range(61, 121):
        feed(activity, seq, seq / 5, [obj(native_id=8)])
    for seq in range(1, 61):
        feed(activity, seq, seq / 5, [obj(native_id=8)], session="two")
    activity.events.add_event.assert_not_called()


def test_predicted_motion_cannot_wake_stationary_vehicle(activity):
    for seq in range(1, 61):
        feed(activity, seq, seq / 5, [obj()])
    for seq in range(61, 71):
        feed(activity, seq, seq / 5, [obj(x=seq, provenance="native_tracked_prediction")])
    activity.events.add_event.assert_not_called()


def test_suppression_can_be_disabled(activity):
    activity.config.native.stationary.enabled = False
    feed(activity, 1, .2, [obj()])
    feed(activity, 2, .4, [obj()])
    assert activity.event_id == 1


def test_stationary_person_tree_jitter_is_suppressed(activity):
    tree_boxes = [
        {"x1": 40, "y1": 20, "x2": 58, "y2": 64},
        {"x1": 39, "y1": 21, "x2": 59, "y2": 64},
        {"x1": 40, "y1": 20, "x2": 59, "y2": 63},
        {"x1": 40, "y1": 21, "x2": 58, "y2": 64},
        {"x1": 39, "y1": 20, "x2": 59, "y2": 64},
    ]
    for seq in range(1, 76):
        detected = obj("person", native_id=8)
        detected["box"] = tree_boxes[(seq - 1) % len(tree_boxes)]
        feed(activity, seq, seq / 5, [detected])
    activity.events.add_event.assert_not_called()
    track = activity.tracks[(8, "person")]
    assert track["motion_state"] == "stationary"
    assert track["motion_extent"] <= activity.config.native.stationary.stationary_threshold


def test_small_person_translation_still_becomes_moving():
    motion = NativeMotion()
    policy = NativeStationaryConfig()
    states = []
    for i in range(12):
        detected = {"x1": 490 + i * 2, "y1": 263, "x2": 508 + i * 2, "y2": 307}
        states.append(motion.update(detected, i * .2, policy, 2))
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
        states.append(motion.update(detected, i * .2, policy, 2))
    assert "moving" in states


def test_whole_window_detects_out_and_back_and_scales_with_box():
    for scale in (1, 10):
        motion = NativeMotion()
        states = [motion.update(box(x, scale), i * .5, NativeStationaryConfig(), 2)
                  for i, x in enumerate((10, 14, 18, 14, 10))]
        assert "moving" in states


def test_single_box_outlier_does_not_wake_stationary_track():
    motion = NativeMotion()
    policy = NativeStationaryConfig()
    for i in range(100):
        motion.update(box(), i * .2, policy, 2)
    assert motion.state == "stationary"
    assert motion.update(box(50), 20, policy, 2) == "stationary"


def test_gap_resets_moving_evidence():
    motion = NativeMotion()
    policy = NativeStationaryConfig()
    for i in range(10):
        motion.update(box(10 + i), i * .2, policy, 2)
    assert motion.state == "moving"
    assert motion.update(box(60), 10, policy, 2) == "uncertain"


def test_stationary_thresholds_have_hysteresis():
    with pytest.raises(ValidationError, match="below"):
        NativeStationaryConfig(stationary_threshold=.2, moving_threshold=.1)


@pytest.mark.parametrize("fps,window", [(.5, 2), (2, .5)])
def test_moving_vehicle_at_low_rate_has_enough_evidence(activity, fps, window):
    activity.config.live_sample_fps = fps
    activity.config.native.stationary.window_seconds = window
    for seq in range(1, 15):
        feed(activity, seq, seq / fps + (seq % 2) * .02, [obj(x=10 + seq * 5)])
    assert activity.events.add_event.called


@pytest.mark.parametrize("with_gap", [False, True])
def test_startup_or_post_gap_single_outlier_does_not_admit(activity, with_gap):
    if with_gap:
        for seq in range(1, 41):
            feed(activity, seq, seq / 2, [obj()])
    offset = 30 if with_gap else 0
    for seq, x in enumerate([10, 10, 50, 10, 10, 10], start=41):
        feed(activity, seq, offset + (seq - 40) / 2, [obj(x=x)])
    activity.events.add_event.assert_not_called()


def test_confirmation_tolerates_consistently_slow_half_fps(activity):
    activity.config.live_sample_fps = .5
    for seq in range(1, 20):
        feed(activity, seq, seq * 2.01, [obj(x=10 + seq * 5)])
        activity.tick(now=100 + seq * 2.01 + 2.005)
        assert activity.health == "healthy"
    assert activity.events.add_event.called

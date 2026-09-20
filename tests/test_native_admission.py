from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app.config import AppConfig, CameraConfig, DetectorConfig
from survng.app.native_activity import NativeActivity
from survng.app.native_evidence import Candidate
from survng.app.native_main_frame import NativeMainFrameVerifier, context_crop
from survng.app.live_detections import DetectionSnapshot


def main_frames_from_evidence(evidence):
    config = getattr(
        evidence,
        "config",
        AppConfig(
            cameras=[
                {
                    "id": "front",
                    "name": "Front",
                    "stream_url": "rtsp://unused.invalid",
                }
            ]
        ),
    )
    main_frames = NativeMainFrameVerifier(
        lambda: config,
        Mock(),
        Counter(),
    )
    main_frames.read_frame = evidence.read_frame
    main_frames.project_main = evidence.project_main
    main_frames.detector = evidence.verifier
    return main_frames


def setup_activity(**detector_kwargs):
    events = Mock()
    events.add_event.side_effect = [{"id": 1}, {"id": 2}]
    events.open_incident = Mock(
        side_effect=[{"id": 10, "observation_count": 1}, {"id": 11, "observation_count": 1}]
    )
    detector_kwargs.setdefault("event_confirmation_frames", 1)
    return NativeActivity(
        CameraConfig(id="front", name="Front", stream_url="rtsp://unused.invalid"),
        DetectorConfig(**detector_kwargs),
        events,
        Mock(),
        Mock(return_value="preview"),
    )


def feed(a, seq, objects=None):
    obj = {
        "label": "dog",
        "confidence": 0.8,
        "box": {"x1": 20, "y1": 20, "x2": 40, "y2": 60},
        "detection_provenance": "native_fresh_detection",
    }
    now = 100 + seq / 5
    a.consume(
        DetectionSnapshot(
            seq / 5,
            seq,
            100,
            100,
            tuple([obj] if objects is None else objects),
            "session",
            "native_fresh_detection",
            now,
        ),
        now=now,
        epoch=1000 + seq / 5,
    )


def test_scene_activity_opens_incident_on_first_fresh_detection():
    a = setup_activity()
    feed(a, 1)
    assert a.events.add_event.call_count == 1
    assert a.event_id == 1
    assert a.incident_id == 10
    assert a.status()["active"] is True


def test_tracking_classes_filter_scene_activity():
    a = setup_activity(native={"tracking_classes": ["person"]})
    feed(a, 1)  # dog
    a.events.add_event.assert_not_called()
    person = {
        "label": "person",
        "confidence": 0.9,
        "box": {"x1": 10, "y1": 10, "x2": 30, "y2": 50},
        "detection_provenance": "native_fresh_detection",
    }
    feed(a, 2, [person])
    assert a.events.add_event.call_count == 1


def test_multi_object_scene_joins_one_incident():
    a = setup_activity()
    feed(
        a,
        1,
        [
            {
                "label": "dog",
                "confidence": 0.9,
                "box": {"x1": 20, "y1": 20, "x2": 40, "y2": 60},
                "detection_provenance": "native_fresh_detection",
            },
            {
                "label": "person",
                "confidence": 0.88,
                "box": {"x1": 70, "y1": 10, "x2": 90, "y2": 80},
                "detection_provenance": "native_fresh_detection",
            },
        ],
    )
    assert a.events.add_event.call_count == 1
    participants = a.events.update_native_incident_state.call_args.args[2]
    assert {item["label"] for item in participants} == {"dog", "person"}


def test_inactivity_closes_scene_incident():
    a = setup_activity()
    feed(a, 1)
    assert a.event_id == 1
    for seq in range(2, 40):
        feed(a, seq, [])
    assert a.event_id is None
    final = a.events.update_native_incident_state.call_args.args[1]
    assert final["state"] == "complete"


def test_policy_reset_clears_open_scene_incident():
    a = setup_activity()
    feed(a, 1)
    assert a.event_id == 1
    a.finish("policy_changed", now=101)
    assert a.event_id is None


def test_below_confidence_spike_does_not_open_scene_activity():
    a = setup_activity(confidence_threshold=0.65, event_confirmation_frames=1)
    feed(
        a,
        1,
        [
            {
                "label": "person",
                "confidence": 0.48,
                "box": {"x1": 20, "y1": 20, "x2": 40, "y2": 60},
                "detection_provenance": "native_fresh_detection",
            }
        ],
    )
    a.events.add_event.assert_not_called()
    assert a.event_id is None


def test_confirmation_frames_required_before_scene_activity():
    a = setup_activity(confidence_threshold=0.65, event_confirmation_frames=2)
    person = {
        "label": "person",
        "confidence": 0.71,
        "box": {"x1": 20, "y1": 20, "x2": 40, "y2": 60},
        "detection_provenance": "native_fresh_detection",
    }
    feed(a, 1, [person])
    a.events.add_event.assert_not_called()
    feed(a, 2, [person])
    assert a.events.add_event.call_count == 1


def test_incident_zone_required_blocks_outside_zone_activity():
    camera = CameraConfig(
        id="front",
        name="Front",
        stream_url="rtsp://unused.invalid",
        require_incident_zone=True,
        zones=[
            {
                "name": "Entry",
                "behavior": "incident",
                "enabled": True,
                "points": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 0.2, "y": 0.0},
                    {"x": 0.2, "y": 0.2},
                    {"x": 0.0, "y": 0.2},
                ],
            }
        ],
    )
    events = Mock()
    events.add_event.side_effect = [{"id": 1}]
    events.open_incident = Mock(return_value={"id": 10, "observation_count": 1})
    a = NativeActivity(
        camera,
        DetectorConfig(
            confidence_threshold=0.65,
            require_incident_zone=True,
            event_confirmation_frames=1,
        ),
        events,
        Mock(),
        Mock(return_value="preview"),
    )
    # Bottom-right box is outside Entry.
    feed(
        a,
        1,
        [
            {
                "label": "person",
                "confidence": 0.9,
                "box": {"x1": 70, "y1": 70, "x2": 90, "y2": 90},
                "detection_provenance": "native_fresh_detection",
            }
        ],
    )
    a.events.add_event.assert_not_called()


def test_cover_verifier_requires_spatial_match_and_clear_view():
    rng = np.random.default_rng(7)
    main = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
    box = {"x1": 300, "y1": 200, "x2": 340, "y2": 250}
    obj = {"label": "dog", "box": box, "confidence": 0.8}
    config = AppConfig(
        cameras=[{"id": "front", "name": "Front", "stream_url": "rtsp://unused.invalid"}]
    )
    evidence = SimpleNamespace(
        config=config,
        read_frame=Mock(return_value=main),
        project_main=Mock(return_value=[obj]),
        verifier=Mock(),
    )
    service = main_frames_from_evidence(evidence)
    candidate = Candidate(0, main[::2, ::2], [obj], 0)
    crop, left, top = context_crop(main, box)
    detection = dict(
        obj,
        box={k: v - (left if k.startswith("x") else top) for k, v in box.items()},
    )
    evidence.verifier.detect.return_value = [detection]
    confirmed = service.verify_candidate("front", candidate)
    assert confirmed["status"] == "confirmed"
    evidence.verifier.detect.assert_called()

    evidence.verifier.detect.return_value = []
    assert service.verify_candidate("front", candidate)["status"] != "confirmed"

    fragment = dict(
        detection,
        confidence=0.95,
        box={
            **detection["box"],
            "x2": detection["box"]["x1"] + 12,
            "y2": detection["box"]["y1"] + 15,
        },
    )
    evidence.verifier.detect.return_value = [fragment]
    assert service.verify_candidate("front", candidate)["status"] != "confirmed"

    evidence.read_frame.return_value = None
    assert service.verify_candidate("front", candidate)["vote"] in {
        "unavailable",
        "unaligned",
        "negative",
    }
    evidence.read_frame.return_value = main
    evidence.project_main.return_value = []
    assert service.verify_candidate("front", candidate)["vote"] == "unaligned"
    assert crop.shape[0] >= 192 and crop.shape[1] >= 192


def test_cancelled_cover_verify_stops_before_inference():
    cancelled = __import__("threading").Event()
    main = np.zeros((20, 20, 3), np.uint8)

    def read(*args):
        cancelled.set()
        return main

    evidence = SimpleNamespace(
        read_frame=Mock(side_effect=read),
        project_main=Mock(),
        verifier=Mock(),
    )
    service = main_frames_from_evidence(evidence)
    result = service.verify_candidate(
        "front",
        Candidate(0, main[::2, ::2], [], 0),
        cancelled=cancelled,
    )
    assert result.get("reason") == "stopped" or result.get("vote") in {
        "unavailable",
        "unaligned",
        "negative",
    }
    evidence.read_frame.assert_called()
    evidence.verifier.detect.assert_not_called()


@pytest.mark.parametrize("offset", [0.5, -0.5, 1.0, -1.0])
def test_verification_finds_time_skewed_main_pose_and_preserves_frame_time(offset):
    main = np.random.default_rng(9).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    obj = {
        "label": "person",
        "confidence": 0.8,
        "box": {"x1": 300, "y1": 200, "x2": 340, "y2": 280},
    }
    config = AppConfig(
        cameras=[{"id": "front", "name": "Front", "stream_url": "rtsp://unused.invalid"}]
    )
    selected = main.copy()
    evidence = SimpleNamespace(
        config=config,
        read_frame=Mock(
            side_effect=lambda camera, epoch, source: selected if epoch == 100 + offset else main
        ),
        project_main=Mock(
            side_effect=lambda candidate, frame: [obj] if frame is selected else []
        ),
        verifier=Mock(),
    )
    _, left, top = context_crop(main, obj["box"])
    evidence.verifier.detect.return_value = [
        dict(
            obj,
            box={
                k: v - (left if k.startswith("x") else top)
                for k, v in obj["box"].items()
            },
        )
    ]
    service = main_frames_from_evidence(evidence)
    result = service.verify_candidate(
        "front", Candidate(100, main[::2, ::2], [obj], 0)
    )
    assert result["status"] == "confirmed"
    assert result["cover"][0] is selected
    assert result["cover"][1]["frame_captured_at_epoch"] == 100 + offset
    assert result["cover"][2] == 100 + offset
    evidence.verifier.detect.assert_called_once()
    assert evidence.read_frame.call_count <= 5


def test_time_window_does_not_bypass_geometry_alignment():
    main = np.random.default_rng(9).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    evidence = SimpleNamespace(
        read_frame=Mock(return_value=main),
        project_main=Mock(return_value=[]),
        verifier=Mock(),
    )
    result = main_frames_from_evidence(evidence).verify_candidate(
        "front", Candidate(100, main[::2, ::2], [], 0)
    )
    assert result["status"] != "confirmed"
    assert result["vote"] == "unaligned"
    assert [call.args[1] for call in evidence.read_frame.call_args_list] == [
        100,
        100.5,
        99.5,
        101,
        99,
    ]
    evidence.verifier.detect.assert_not_called()


def test_cancel_during_time_window_decode_stops_before_matching_or_inference():
    cancelled = __import__("threading").Event()
    main = np.ones((40, 40, 3), np.uint8)

    def read(camera, epoch, source):
        if epoch != 100:
            cancelled.set()
        return main

    evidence = SimpleNamespace(
        read_frame=Mock(side_effect=read),
        project_main=Mock(return_value=[]),
        verifier=Mock(),
    )
    result = main_frames_from_evidence(evidence).verify_candidate(
        "front",
        Candidate(100, main[::2, ::2], [], 0),
        cancelled=cancelled,
    )
    assert result.get("reason") == "stopped"
    assert evidence.read_frame.call_count == 2
    evidence.project_main.assert_called_once()
    evidence.verifier.detect.assert_not_called()


def test_ambiguous_pose_checks_nearby_time_before_deciding():
    main = np.random.default_rng(8).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    obj = {
        "label": "person",
        "confidence": 0.9,
        "box": {"x1": 300, "y1": 200, "x2": 340, "y2": 280},
    }
    config = AppConfig(
        cameras=[{"id": "front", "name": "Front", "stream_url": "rtsp://unused.invalid"}]
    )
    evidence = SimpleNamespace(
        config=config,
        read_frame=Mock(return_value=main),
        project_main=Mock(return_value=[obj]),
        verifier=Mock(),
    )
    _, left, top = context_crop(main, obj["box"])
    actual = dict(
        obj,
        box={
            k: v - (left if k.startswith("x") else top) for k, v in obj["box"].items()
        },
    )
    fragment = dict(actual, box={**actual["box"], "x2": actual["box"]["x1"] + 5})
    evidence.verifier.detect.side_effect = [[fragment], [], [actual]]
    result = main_frames_from_evidence(evidence).verify_candidate(
        "front", Candidate(100, main[::2, ::2], [obj], 0)
    )
    assert result["status"] == "confirmed"
    assert result["vote"] == "confirmed"
    assert result["checks"] == [
        {"epoch": 100, "votes": ["ambiguous", "negative", "confirmed"]}
    ]
    assert result["cover"][2] == 99.5

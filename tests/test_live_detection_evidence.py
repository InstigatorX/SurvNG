from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app.camera import _AutoStreamAlignment
from survng.app.camera_capture import CameraCaptureService
from survng.app.config import AppConfig, CameraConfig, DetectorConfig, ObjectTrackingConfig
from survng.app.config_application import live_detection_threshold, manager_owned_config
from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector, RecordedDetectionResult, TimestampedLiveFrame
from survng.app.motion_pipeline.decision_handler import MotionDecisionHandler
from survng.app.object_tracking_lifecycle import ObjectTrackingLifecycle
from survng.app.dlstreamer_capture import DlStreamerCaptureHandle, _StreamInbox
from survng.app.dlstreamer_protocol import TYPE_FRAME, MessageReader, encode_frame
from survng.app.live_detections import DetectionHistory, DetectionSnapshot
from survng.dlstreamer_live import _detection_metadata


def payload(pts=10.0, sequence=1, objects=None):
    return {"schema_version": 1, "source_pts": pts, "inference_sequence": sequence,
            "width": 640, "height": 360, "objects": [] if objects is None else objects}


def car():
    return {"label": "car", "confidence": .9, "box": {"x1": 100, "y1": 80, "x2": 300, "y2": 240}}


def test_empty_result_is_not_missing_and_clears_positive():
    history = DetectionHistory()
    history.reset("live-1")
    history.add(DetectionSnapshot.parse(payload(objects=[car()]), session="live-1"))
    assert history.match(pts=10.1, session="live-1", detect_fps=5).objects
    history.add(DetectionSnapshot.parse(payload(10.2, 2), session="live-1"))
    assert history.match(pts=10.25, session="live-1", detect_fps=5).objects == ()
    assert history.match(pts=11, session="live-1", detect_fps=5) is None


def test_matching_uses_cadence_but_caps_age_and_never_uses_future():
    history = DetectionHistory()
    history.reset("s")
    history.add(DetectionSnapshot.parse(payload(objects=[car()]), session="s"))
    assert history.match(pts=9.99, session="s", detect_fps=5) is None
    assert history.match(pts=10.3, session="s", detect_fps=5) is None
    assert history.match(pts=10.3, session="s", detect_fps=2.5) is not None
    assert history.match(pts=10.51, session="s", detect_fps=.5) is None
    assert history.match(pts=float("nan"), session="s", detect_fps=5) is None


def test_reconnect_with_identical_pts_cannot_match_old_evidence():
    history = DetectionHistory()
    history.reset("old")
    old = DetectionSnapshot.parse(payload(objects=[car()]), session="old")
    history.add(old)
    history.reset("new")
    history.add(old)  # delayed result from retired capture
    assert history.match(pts=10, session="old", detect_fps=5) is None
    assert history.match(pts=10, session="new", detect_fps=5) is None
    history.add(DetectionSnapshot.parse(payload(), session="new"))
    assert history.match(pts=10, session="new", detect_fps=5).objects == ()


def test_snapshot_histories_are_bounded_and_reject_reordered_results():
    history = DetectionHistory()
    history.reset("s")
    for i in range(100):
        history.add(DetectionSnapshot.parse(payload(float(i), i + 1), session="s"))
    assert len(history.snapshots) == 32
    history.add(DetectionSnapshot.parse(payload(98, 101, [car()]), session="s"))
    assert history.match(pts=99, session="s", detect_fps=5).objects == ()
    assert history.status()["out_of_order"] == 1


@pytest.mark.parametrize("field,value", [
    ("source_pts", float("nan")), ("source_pts", float("inf")), ("source_pts", -1),
    ("width", 0), ("height", "360"), ("inference_sequence", 0),
    ("schema_version", 0), ("objects", None),
])
def test_invalid_metadata_is_rejected_not_interpreted_as_empty(field, value):
    data = payload()
    data[field] = value
    with pytest.raises(ValueError):
        DetectionSnapshot.parse(data)


def test_scale_boxes_without_mutating_cache_and_clip_to_evidence():
    snapshot = DetectionSnapshot.parse(payload(objects=[car()]), session="s")
    objects = snapshot.scaled_objects(320, 180)
    assert objects[0]["box"] == {"x1": 50, "y1": 40, "x2": 150, "y2": 120}
    objects[0]["box"]["x1"] = 999
    assert snapshot.objects[0]["box"]["x1"] == 100


def test_inbox_resets_identity_and_flushes_old_frames_on_backward_pts():
    inbox = _StreamInbox()
    old_session = inbox.session
    frame = np.zeros((2, 2), np.uint8)
    inbox.put_frame(frame, 1, 10)
    inbox.add_detection_snapshot(payload())
    inbox.put_frame(frame, 2, 0)
    assert inbox.session != old_session
    assert inbox.pop_detection_snapshots() == []
    received = inbox.get_frame(.01)
    assert received[2] == 0 and received[3] == inbox.session
    assert inbox.get_frame(.01) is None


def test_metadata_queue_is_bounded_before_capture_reader_drains():
    inbox = _StreamInbox()
    for i in range(100):
        inbox.add_detection_snapshot(payload(float(i), i + 1))
    snapshots = inbox.pop_detection_snapshots()
    assert len(snapshots) == 32 and snapshots[-1].source_pts == 99


def test_standalone_harvest_does_not_relabel_returned_frame(monkeypatch):
    handle = DlStreamerCaptureHandle(read_timeout_ms=100)
    first = np.zeros((2, 2), np.uint8)
    monkeypatch.setattr(handle, "_next_frame", lambda _timeout: (first, 4, 1.0, "s"))
    reader = MessageReader()
    reader.feed(encode_frame(width=2, height=2, sequence=5, pts=2.0, pixels=bytes(4)))
    kind, body = reader.pop()
    assert kind == TYPE_FRAME
    monkeypatch.setattr(handle, "_harvest_available_messages", lambda: handle._apply_message(kind, body))
    ok, returned = handle.read()
    assert ok and returned is first
    assert handle.pop_frame_identity() == (4, 1.0, "s")


@pytest.mark.parametrize("objects", [[], [{"x": 100, "y": 80, "w": 200, "h": 160,
                                        "detection": {"label": "car", "confidence": .9}}]])
def test_native_metadata_adapter_uses_messages_not_pixel_mapping(objects):
    class Buffer:
        pts = 10_000_000_000

        def map(self, *_args):
            raise AssertionError("do not copy/map video pixels to read inference metadata")

    class VideoFrame:
        def __init__(self, buffer, *, caps):
            assert isinstance(buffer, Buffer)

        def messages(self):
            return [json.dumps({"objects": objects})]

    sample = SimpleNamespace(
        get_buffer=Buffer,
        get_caps=lambda: SimpleNamespace(get_structure=lambda _i: SimpleNamespace(
            get_value=lambda key: {"width": 640, "height": 360}[key])),
    )
    result = _detection_metadata(sample, VideoFrame, inference_sequence=1,
                                 gst_second=1_000_000_000, clock_time_none=2**64-1)
    assert result["source_pts"] == 10.0
    assert bool(result["objects"]) == bool(objects)


def test_auto_alignment_accepts_native_gray_input():
    rng = np.random.default_rng(13)
    gray = rng.integers(0, 256, (320, 480), dtype=np.uint8)
    color = np.repeat(gray[:, :, None], 3, axis=2)
    estimate = _AutoStreamAlignment._estimate(gray, color)
    assert estimate is not None
    assert np.allclose(estimate, [1, 1, 0, 0], atol=.01)


def test_live_zone_projection_does_not_change_crop_or_motion_coordinates():
    camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://fixture.invalid/main", zones=[{
        "name": "main-right", "points": [{"x": .65, "y": .1}, {"x": .85, "y": .1},
                                         {"x": .85, "y": .9}, {"x": .65, "y": .9}],
    }])
    detector = SimpleNamespace(config=DetectorConfig(confidence_threshold=.5), detect=Mock())
    backend = RecordedMotionObjectDetector(camera, detector, SimpleNamespace(), lambda: None)
    obj = {"label": "car", "confidence": .9, "box": {"x1": 10, "y1": 20, "x2": 30, "y2": 60}}
    frame = np.zeros((100, 100), np.uint8)
    plain = backend._detect_objects(frame, precomputed=[obj], enrich_faces=False)
    shifted = backend._detect_objects(frame, precomputed=[obj], enrich_faces=False,
                                      spatial_alignment={"reliable": True, "offset_x": .5})
    assert not plain[0]["incident_eligible"]
    assert shifted[0]["incident_eligible"]
    assert shifted[0]["box"] == obj["box"]
    detector.detect.assert_not_called()


def test_live_motion_correlation_does_not_apply_main_transform_twice():
    events = Mock()
    events.add_event.return_value = {"id": 42}
    frame = np.zeros((100, 100), np.uint8)
    obj = {"label": "car", "confidence": .9, "incident_eligible": True,
           "box": {"x1": 10, "y1": 20, "x2": 30, "y2": 60}}
    handler = MotionDecisionHandler(
        camera_id="gate", events=events,
        detection_provider=lambda _at: RecordedDetectionResult(frame, [obj], "", {}, frame_source="live_fast_path"),
        snapshot_writer=lambda _frame, _at: "snapshot.jpg", object_serializer=json.dumps,
        spatial_alignment={"reliable": True, "scale_x": 1, "scale_y": 1, "offset_x": .5, "offset_y": 0},
    )
    outcome = handler.handle("adaptive/visual_backup", "backup", datetime.now(timezone.utc),
                             {"features": {"motion_regions": [[.1, .2, .3, .6]]}},
                             require_eligible_object=True, require_motion_correlation=True)
    assert outcome.event_id == 42
    assert outcome.detected_objects[0]["motion_correlation"] == "spatial"


def test_capture_status_clears_stale_boxes_and_old_session_admission():
    now = [10.0]
    capture = CameraCaptureService(camera_id="gate", source_url=lambda _source: "", backend=None,
                                   monotonic_clock=lambda: now[0])
    capture._generation = 1
    capture._stop.clear()
    capture._detection_history["live"].reset("s")
    capture._detection_history["live"].add(DetectionSnapshot.parse(payload(objects=[car()]), session="s"))
    capture._publish_frame("live", np.zeros((180, 320), np.uint8), source_pts=10.0, source_session="s")
    assert capture.status()["live_detections"][0]["box"]["x1"] == 50
    assert capture.matched_snapshot("live", source_pts=10, generation=1, source_session="old") is None
    now[0] += capture.stale_seconds + 1
    assert capture.status()["live_detections"] == []


def test_capture_graph_rebuilds_when_tracking_rate_or_low_candidate_floor_changes():
    config = AppConfig()
    changed = config.model_copy(deep=True)
    changed.detector.tracking.sample_fps += 1
    assert manager_owned_config(config) != manager_owned_config(changed)
    changed = config.model_copy(deep=True)
    changed.detector.tracking.low_confidence_threshold = .01
    assert live_detection_threshold(changed) == .01
    assert manager_owned_config(config) != manager_owned_config(changed)


def test_cropped_live_event_waits_for_main_tracking_seed():
    camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://fixture.invalid/main")
    frame = np.zeros((360, 640), np.uint8)
    sample = TimestampedLiveFrame(frame, time.time(), time.monotonic(), 1, 1, 1,
                                  spatial_alignment={"reliable": True, "offset_x": .2})
    detector = SimpleNamespace(config=DetectorConfig(require_incident_zone=False), detect=Mock())
    backend = RecordedMotionObjectDetector(
        camera, detector, SimpleNamespace(), lambda: None,
        timestamped_live_frame_provider=lambda: sample,
        live_detections_provider=lambda _sample: DetectionSnapshot.parse(payload(objects=[car()]), session="s"),
    )
    result = backend.detect_initial(datetime.now(timezone.utc))
    assert result.objects[0]["live_detection_session"] == "s"
    assert result.objects[0]["live_inference_sequence"] == 1
    assert result.objects[0]["live_detection_source_pts"] == 10.0
    assert result.objects[0]["incident_eligible"]
    assert result.objects[0]["tracking_geometry_trusted"] is False
    session = SimpleNamespace(config=ObjectTrackingConfig())
    assert ObjectTrackingLifecycle._trackable_objects(session, result.objects) == []
    # Refined main results have no provisional geometry restriction.
    assert ObjectTrackingLifecycle._trackable_objects(session, [car()])

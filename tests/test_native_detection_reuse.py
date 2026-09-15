"""Native capture → frame resolver → session detector boundary → real Hybrid."""
from dataclasses import replace
from copy import deepcopy
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app.camera_capture import CameraCaptureService, CapturedFrame
from survng.app.config import CameraConfig, ObjectTrackingConfig
from survng.app.live_detections import DetectionSnapshot
from survng.app.object_track.hybrid import HybridObjectTracker
from survng.app.object_track.session import ObjectTrackingSession
from survng.app.object_track.types import TrackingFrame
from survng.app.tracking_frames import CameraFrameTimeline
from survng.dlstreamer_live import _NativeInferenceEvidence, _detection_metadata


def person():
    return {"label": "person", "confidence": .9,
            "box": {"x1": 100, "y1": 80, "x2": 300, "y2": 240}}


def setup_case(objects=()):
    camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://fixture.invalid/main")
    capture = CameraCaptureService(camera_id="gate", source_url=lambda _: "", backend=None)
    capture._generation = 1
    capture._detection_history["live"].reset("s")
    frame = CapturedFrame("live", np.zeros((180, 320), np.uint8), 100, 100, "", 320, 180, 1,
                          generation=1, source_pts=10, source_session="s")
    snapshot = DetectionSnapshot.parse({"schema_version": 1, "source_pts": 10,
        "inference_sequence": 1, "width": 640, "height": 360, "objects": list(objects),
        "provenance": "native_fresh_detection"}, session="s")
    timeline = CameraFrameTimeline(camera=camera, capture=capture, recorder=Mock(),
        stop_event=threading.Event(), sample_fps=lambda: 5, requires_inference=True)
    detector = SimpleNamespace(config=SimpleNamespace(confidence_threshold=.7),
                               detect=Mock(return_value=[person()]), detect_tracking=Mock(return_value=[person()]))
    session = ObjectTrackingSession(camera, ObjectTrackingConfig(), detector, lambda: None,
                                   lambda *args: {}, None, threading.BoundedSemaphore(1))
    return capture, frame, snapshot, timeline, session


@pytest.mark.parametrize("objects", [[], [person()]])
@pytest.mark.parametrize("route", ["direct", "timeline"])
def test_native_positive_and_empty_bypass_detector(objects, route):
    capture, frame, snapshot, timeline, session = setup_case(objects)
    # Metadata may arrive after pixels. Rehydrate the already-buffered frame.
    timeline.remember_capture(frame)
    capture._detection_history["live"].add(snapshot)
    if route == "timeline":
        timeline._hydrate_live_results()
        evidence = timeline.live_frames[0]
    else:
        evidence = timeline.tracking_frame(frame)
    assert evidence.native_fresh and not evidence.requires_inference
    result = session._tracking_detections_for_frame(frame.image, catchup=route == "timeline", evidence=evidence)
    session.detector.detect_tracking.assert_not_called()
    session.detector.detect.assert_not_called()
    tracker = HybridObjectTracker(ObjectTrackingConfig(), .7)
    tracked = tracker.update(result, 100)
    assert len(tracked) == len(objects)
    if objects:
        assert tracked[0]["label"] == "person"
        assert tracked[0]["box"] == {"x1": 50, "y1": 40, "x2": 150, "y2": 120}
        assert tracked[0]["detection_provenance"] == "native_fresh_detection"
    assert session.status()["native_detection_hits"] == 1
    assert session.status()["native_detection_empty_hits"] == (not objects)


@pytest.mark.parametrize("failure", ["missing", "stale", "nearby", "future", "session", "generation", "unknown", "prediction", "malformed", "disabled"])
def test_untrusted_native_results_fall_back(failure):
    capture, frame, snapshot, timeline, session = setup_case([person()])
    if failure == "stale": snapshot = replace(snapshot, source_pts=9)
    if failure == "nearby": snapshot = replace(snapshot, source_pts=9.99)
    if failure == "future": snapshot = replace(snapshot, source_pts=10.01)
    if failure == "session": frame = replace(frame, source_session="retired")
    if failure == "generation": capture._generation = 2
    if failure == "unknown": snapshot = replace(snapshot, provenance="unknown")
    if failure == "prediction": snapshot = replace(snapshot, provenance="native_tracked_prediction")
    if failure == "malformed":
        with pytest.raises(ValueError):
            DetectionSnapshot.parse({"schema_version": 1, "source_pts": float("nan")})
    if failure not in ("missing", "disabled", "malformed"):
        capture._detection_history["live"].add(snapshot)
    evidence = timeline.tracking_frame(frame)
    assert evidence.requires_inference and not evidence.native_fresh
    result = session._tracking_detections_for_frame(frame.image, catchup=False, evidence=evidence)
    session.detector.detect_tracking.assert_called_once()
    assert result[0]["detection_provenance"] == "fallback_live_inference"
    assert session.status()["fallback_detector_calls"] == 1
    if failure in ("nearby", "stale"):
        assert capture._detection_history["live"].status()["native_detection_stale"] == 1


def test_main_frame_keeps_detector_even_with_native_snapshot():
    _, frame, snapshot, _, session = setup_case([person()])
    main = replace(frame, source="main")
    result = session._tracking_detections_for_frame(main.image, catchup=True, evidence=TrackingFrame(main, snapshot))
    session.detector.detect_tracking.assert_called_once()
    assert result[0]["detection_provenance"] == "recorded_refinement"
    assert session.status()["fallback_detector_calls"] == 0


def test_prediction_assistance_cannot_observe_or_create_tracks():
    tracker = HybridObjectTracker(ObjectTrackingConfig(), .7)
    tracker.update([person()], 100)
    prediction = {**person(), "native_track_id": 999, "detection_provenance": "native_tracked_prediction"}
    before = deepcopy(tracker._tracks)
    tracker.assist_predictions([prediction], 100.2)
    assert tracker._native_positions
    assert tracker._tracks == before
    tracker.update([], 100.2)
    track = tracker._tracks[1]
    assert track.hits == before[1].hits
    assert track.confidence == before[1].confidence
    assert track.box_history == before[1].box_history
    assert track.last_seen == before[1].last_seen
    assert not tracker._native_positions
    assert list(tracker._tracks) == [1]
    tracker.assist_predictions([prediction], 100.4)
    tracker.update([person()], 100.4)  # Only actual fallback inference adds a hit.
    assert tracker._tracks[1].hits == before[1].hits + 1
    empty = HybridObjectTracker(ObjectTrackingConfig(), .7)
    empty.assist_predictions([prediction], 100)
    empty.update([], 100)
    assert not empty._tracks


def test_provenance_is_captured_before_drops_and_tracker_predictions():
    region = SimpleNamespace(rect=lambda: SimpleNamespace(x=100, y=80, w=200, h=160),
                             label=lambda: "person", confidence=lambda: .9)
    frame_type = lambda *args, **kw: SimpleNamespace(regions=lambda: [region])
    ledger = _NativeInferenceEvidence(3, tracking=True)
    for pts in range(7):
        ledger.observe(SimpleNamespace(pts=pts), None, frame_type)
    # Appsink lost the first four frames. Cadence must not restart there.
    assert ledger.pop(4)[0] == "native_tracked_prediction"
    assert ledger.pop(6) == ("native_fresh_detection", [person()])
    assert ledger.pop(99) == ("unknown", [])
    # Empty inference is explicit; a failed adapter is unknown, never empty AI evidence.
    empty = _NativeInferenceEvidence(1)
    empty.observe(SimpleNamespace(pts=0), None, lambda *a, **k: SimpleNamespace(regions=lambda: []))
    assert empty.pop(0) == ("native_fresh_detection", [])
    empty.observe(SimpleNamespace(pts=1), None, Mock(side_effect=ValueError("bad metadata")))
    assert empty.pop(1) == ("unknown", []) and empty.invalid == 1


def test_fresh_snapshot_excludes_rois_appended_by_native_tracker():
    import json
    sample = SimpleNamespace(get_buffer=lambda: SimpleNamespace(pts=10_000_000_000),
        get_caps=lambda: SimpleNamespace(get_structure=lambda _: SimpleNamespace(
            get_value=lambda k: {"width": 640, "height": 360}[k])))
    frame_type = lambda *a, **k: SimpleNamespace(messages=lambda: [json.dumps({"objects": [
        {"x": 50, "y": 50, "w": 20, "h": 20, "object_id": 999,
         "detection": {"label": "car", "confidence": 0}},
    ]})])
    result = _detection_metadata(sample, frame_type, inference_sequence=1,
        gst_second=10**9, clock_time_none=2**64-1, native_result=("native_fresh_detection", [person()]))
    assert result["objects"] == [person()]
    assert result["provenance"] == "native_fresh_detection"


def test_pts_reuse_cannot_join_old_metadata_to_new_pixels():
    ledger = _NativeInferenceEvidence(3, tracking=True)
    frame_type = lambda *a, **k: SimpleNamespace(regions=lambda: [])
    for pts in (10, 11, 12, 10, 11, 12, 13):
        ledger.observe(SimpleNamespace(pts=pts), None, frame_type)
    assert ledger.pop(10) == ("unknown", [])
    assert ledger.pop(13) == ("unknown", [])
    assert ledger.invalid == 1


@pytest.mark.parametrize("kind", ["positive", "empty", "missing", "prediction", "wrong_pts", "unknown"])
def test_initial_admission_uses_fresh_only_and_falls_back(kind):
    import time
    from datetime import datetime, timezone
    from survng.app.config import DetectorConfig
    from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector, TimestampedLiveFrame
    capture, frame, snapshot, timeline, _ = setup_case([] if kind == "empty" else [person()])
    if kind == "missing": snapshot = None
    if kind == "prediction": snapshot = replace(snapshot, provenance="native_tracked_prediction")
    if kind == "unknown": snapshot = replace(snapshot, provenance="unknown")
    if kind == "wrong_pts": snapshot = replace(snapshot, source_pts=9.99)
    sample = TimestampedLiveFrame(frame.image, time.time(), time.monotonic(), 1, 1, 1,
                                  source_pts=10, source_session="s")
    detector = SimpleNamespace(config=DetectorConfig(require_incident_zone=False),
                               detect=Mock(return_value=[person()]), detect_initial=Mock(return_value=[person()]))
    backend = RecordedMotionObjectDetector(timeline.camera, detector, SimpleNamespace(), lambda: None,
        timestamped_live_frame_provider=lambda: sample, live_detections_provider=lambda _: snapshot)
    result = backend.detect_initial(datetime.now(timezone.utc))
    assert detector.detect_initial.call_count == (kind not in ("positive", "empty"))
    if kind == "positive":
        assert result.objects[0]["detection_provenance"] == "native_fresh_detection"


def test_direct_camera_provider_uses_the_same_live_resolver():
    from survng.app.camera import CameraWorker
    capture, frame, snapshot, timeline, _ = setup_case([person()])
    capture._detection_history["live"].add(snapshot)
    timeline.captured = lambda source: frame if source == "live" else None
    worker = SimpleNamespace(tracking_frames=timeline, _live_tracking_geometry_trusted=lambda: True)
    assert CameraWorker._get_tracking_capture(worker).native_fresh


def test_skipped_frame_without_gvatrack_is_not_a_tracker_prediction():
    ledger = _NativeInferenceEvidence(3)
    frame_type = lambda *a, **k: SimpleNamespace(regions=lambda: [])
    ledger.observe(SimpleNamespace(pts=1), None, frame_type)
    ledger.observe(SimpleNamespace(pts=2), None, frame_type)
    assert ledger.pop(1) == ("native_fresh_detection", [])
    assert ledger.pop(2) == ("unknown", [])

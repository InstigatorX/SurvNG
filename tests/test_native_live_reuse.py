"""Production live inference reuse, exercised across capture and consumers."""

from dataclasses import replace
from datetime import datetime, timezone
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app.camera import CameraWorker
from survng.app.camera_capture import CameraCaptureService, CapturedFrame
from survng.app.config import AppConfig, CameraConfig, DetectorConfig, ObjectTrackingConfig
from survng.app.config_application import manager_owned_config
from survng.app.dlstreamer_capture import _SharedLiveProcess, _StreamInbox
from survng.app.dlstreamer_protocol import TYPE_DETECTIONS, MessageReader, encode_json
from survng.app.live_detections import DetectionSnapshot
from survng.app.manager import AppManager
from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector, TimestampedLiveFrame
from survng.app.object_tracking import ObjectTrackingSession
from survng.app.tracking_frames import CameraFrameTimeline


OBJECT = {"label": "person", "confidence": .9,
          "box": {"x1": 2, "y1": 4, "x2": 18, "y2": 36}}


def fixture(case):
    camera = CameraConfig(id="fixture", name="Fixture", stream_url="rtsp://example.invalid/main",
                          live_stream_url="rtsp://example.invalid/sub")
    capture = CameraCaptureService(camera_id=camera.id, source_url=camera.source_url, backend=Mock())
    capture._generation = 3
    history = capture._detection_history["live"]
    history.reset("current")
    frame = CapturedFrame("live", np.zeros((20, 20, 3), np.uint8), time.time(), time.monotonic(),
                          "", 20, 20, 7, generation=3, source_pts=10, source_session="current")
    payload = {"schema_version": 1, "source_pts": 10, "inference_sequence": 1,
               "width": 40, "height": 40, "objects": [] if case == "empty" else [OBJECT]}
    if case == "stale":
        payload["source_pts"] = 9
    elif case == "nearby":
        payload["source_pts"] = 9.9
    elif case == "future":
        payload["source_pts"] = 10.1
    elif case == "wrong_generation":
        frame = replace(frame, generation=2)
    elif case == "invalid":
        payload["objects"] = [{**OBJECT, "confidence": float("nan")}]
    if case not in {"missing", "native_off"}:
        inbox = _StreamInbox()
        inbox.session = "old" if case == "wrong_session" else "current"
        shared = _SharedLiveProcess([], read_timeout_ms=1000)
        shared._inboxes["live"] = inbox
        reader = MessageReader()
        reader.feed(encode_json(TYPE_DETECTIONS, payload, stream_id="live"))
        shared._dispatch(*reader.pop())
        if case == "invalid":
            assert inbox.status["invalid_detection_snapshots"] == 1
        capture._active_handles["live"] = SimpleNamespace(pop_detection_snapshots=inbox.pop_detection_snapshots)
    recorder = Mock()
    recorder.recording_rows_between.return_value = []
    timeline = CameraFrameTimeline(camera=camera, capture=capture, recorder=recorder,
        stop_event=threading.Event(), sample_fps=lambda: 5, requires_inference=True)
    detector = SimpleNamespace(config=DetectorConfig(require_incident_zone=False),
        detect=Mock(side_effect=AssertionError("wrong inference workload")),
        detect_tracking=Mock(return_value=[]), detect_initial=Mock(return_value=[]))
    return camera, capture, frame, timeline, detector


CASES = ["positive", "empty", "missing", "nearby", "stale", "future", "wrong_session",
         "wrong_generation", "invalid", "native_off"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("consumer", ["catchup", "live"])
def test_tracking_reuses_only_completed_exact_native_inference(case, consumer):
    camera, capture, frame, timeline, detector = fixture(case)
    if consumer == "catchup":
        timeline.remember_capture(frame)
        batch = timeline.read_recorded_frames(frame.captured_at_epoch - .1, frame.captured_at_epoch, 5, 20)
        assert len(batch.frames) == 1
        evidence = batch.frames[0]
    else:
        worker = CameraWorker.__new__(CameraWorker)
        worker.capture = capture
        worker.tracking_frames = SimpleNamespace(captured=lambda source: frame if source == "live" else None)
        worker._effective_spatial_alignment = {"reliable": True}
        evidence = worker._get_tracking_capture()
    session = ObjectTrackingSession(camera=camera, config=ObjectTrackingConfig(), detector=detector,
        frame_provider=lambda: None, update_event=lambda *_: {}, publisher=None,
        limiter=threading.BoundedSemaphore(1))
    objects = session._tracking_detections_for_frame(frame.image, catchup=consumer == "catchup", evidence=evidence)
    hit = case in {"positive", "empty"}
    assert evidence.requires_inference is not hit
    assert (evidence.detection is not None) is hit
    if hit:
        detector.detect_tracking.assert_not_called()
        assert objects == ([{**OBJECT, "box": {"x1": 1, "y1": 2, "x2": 9, "y2": 18}}]
                           if case == "positive" else [])
    else:
        detector.detect_tracking.assert_called_once()
        assert detector.detect_tracking.call_args.args[0] is frame.image


@pytest.mark.parametrize("case", CASES)
def test_initial_live_reuses_exact_results_and_falls_back_on_misses(case):
    camera, capture, frame, _timeline, detector = fixture(case)
    sample = TimestampedLiveFrame(frame.image, frame.captured_at_epoch, frame.captured_at_monotonic,
        frame.sequence, 1, frame.generation, source_pts=frame.source_pts,
        source_session=frame.source_session, spatial_alignment={"reliable": True})
    backend = RecordedMotionObjectDetector(camera, detector, Mock(), lambda: None,
        timestamped_live_frame_provider=lambda: sample,
        live_detections_provider=lambda value: capture.matched_snapshot("live",
            source_pts=value.source_pts, generation=value.capture_generation,
            source_session=value.source_session, exact=True))
    result = backend.detect_initial(datetime.now(timezone.utc))
    assert result.refinement_pending
    if case in {"positive", "empty"}:
        detector.detect_initial.assert_not_called()
        if case == "positive":
            assert result.objects[0]["label"] == "person"
            assert result.objects[0]["live_inference_sequence"] == 1
    else:
        detector.detect_initial.assert_called_once()
        assert detector.detect_initial.call_args.args[0] is frame.image


def test_late_empty_result_is_retained_without_reusing_a_nearby_detection():
    _camera, capture, frame, timeline, detector = fixture("missing")
    timeline.remember_capture(frame)
    assert timeline.live_frames[0].requires_inference
    history = capture._detection_history["live"]
    history.add(DetectionSnapshot(10, 1, 20, 20, (), "current"))
    timeline.remember_capture(replace(frame, sequence=8, source_pts=10.05,
                                      captured_at_epoch=frame.captured_at_epoch + .05))
    assert timeline.live_frames[0].detection.objects == ()
    assert not timeline.live_frames[0].requires_inference
    history.reset("replacement")
    timeline._hydrate_live_results()
    assert timeline.live_frames[0].detection.objects == ()
    detector.detect_tracking.assert_not_called()


@pytest.mark.parametrize("enabled,native,backend,expected", [
    (True, True, "openvino", True), (False, True, "openvino", False),
    (True, False, "openvino", False), (True, True, "coreml", False),
])
def test_manager_enables_native_live_detection_and_supports_rollback(tmp_path, enabled, native, backend, expected):
    config = AppConfig(storage_dir=str(tmp_path), detector=DetectorConfig(
        enabled=enabled, native_live_detection=native, backend=backend, model_path="fixture.xml"))
    manager = AppManager(config)
    try:
        assert manager.capture_backend.options.detect_enabled is expected
        command = manager.capture_backend.command()
        assert ("--model" in command) is expected
        assert command[command.index("--detect-fps") + 1] == "5.000000"
    finally:
        manager.stop_all()


def test_native_rollback_rebuilds_capture_generation():
    before = AppConfig()
    assert before.detector.native_live_detection
    after = before.model_copy(deep=True)
    after.detector.native_live_detection = False
    assert manager_owned_config(before) != manager_owned_config(after)

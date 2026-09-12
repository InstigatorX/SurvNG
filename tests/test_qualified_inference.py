"""Demand-driven inference must preserve evidence through scheduling delays."""
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import numpy as np
import pytest

from survng.app.camera_capture import CapturedFrame
from survng.app.config import CameraConfig, ObjectTrackingConfig
from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector, TimestampedLiveFrame
from survng.app.object_tracking import ObjectTrackingSession
from survng.app.object_track.types import TrackingFrame
from survng.app.tracking_frames import CameraFrameTimeline


def camera():
    return CameraConfig(id="fixture", name="Fixture", stream_url="rtsp://example.invalid/main")


@pytest.mark.parametrize("change,expected", [
    ("new_frame", "person"), ("delay", "fast_frame_stale"),
    ("capture", "fast_frame_invalidated"), ("lifecycle", "fast_frame_invalidated"),
    ("session", "fast_frame_invalidated"), ("disconnect", "fast_frame_invalidated"),
])
def test_initial_response_retains_pixels_and_rejects_expired_identity(monkeypatch, change, expected):
    clock = [100.0]
    pixels = np.full((20, 20, 3), (10, 40, 90), dtype=np.uint8)
    evidence = TimestampedLiveFrame(pixels, 100.0, 50.0, 7, 4, 8, source_session="original")
    current = [evidence]
    calls = []

    def detect(frame, confidence_threshold=None):
        calls.append(frame)
        if change == "delay":
            clock[0] += 1.1
        elif change == "capture":
            current[0] = replace(evidence, capture_generation=9)
        elif change == "lifecycle":
            current[0] = replace(evidence, camera_generation=5)
        elif change == "session":
            current[0] = replace(evidence, source_session="replacement")
        elif change == "disconnect":
            current[0] = None
        else:
            current[0] = replace(evidence, sequence=8, frame=np.zeros_like(pixels))
        return [{"label": "person", "confidence": .9, "box": {"x1": 2, "y1": 2, "x2": 18, "y2": 18}}]

    detector = SimpleNamespace(config=SimpleNamespace(confidence_threshold=.5, require_incident_zone=False),
                               detect=Mock(side_effect=AssertionError("wrong workload")), detect_initial=detect)
    backend = RecordedMotionObjectDetector(camera(), detector, Mock(), lambda: None,
        timestamped_live_frame_provider=lambda: current[0], timestamped_evidence_frame_provider=lambda token: evidence)
    monkeypatch.setattr("survng.app.motion_pipeline.object_detection.time.time", lambda: clock[0])
    result = backend.detect_initial(datetime.fromtimestamp(100, timezone.utc), {
        "evidence_frame_at_epoch": 100, "evidence_frame_sequence": 7,
        "evidence_capture_generation": 8, "evidence_lifecycle_generation": 4,
    })
    assert len(calls) == 1 and calls[0] is pixels
    assert result.refinement_pending
    if expected == "person":
        assert result.frame is pixels
        assert result.objects[0]["label"] == "person"
        assert result.objects[0]["frame_sequence"] == 7
    else:
        assert result.frame is None
        assert result.objects[0]["status"] == expected


def test_demand_live_tracking_uses_tracking_priority_with_exact_pixels():
    pixels = np.zeros((20, 20, 3), dtype=np.uint8)
    detector = SimpleNamespace(config=SimpleNamespace(confidence_threshold=.5),
                              detect=Mock(side_effect=AssertionError("wrong workload")),
                              detect_tracking=Mock(return_value=[{"status": "inference_deferred"}]))
    session = ObjectTrackingSession(camera=camera(), config=ObjectTrackingConfig(), detector=detector,
        frame_provider=lambda: None, update_event=lambda *_: {}, publisher=None,
        limiter=threading.BoundedSemaphore(1))
    sample = TrackingFrame(CapturedFrame("live", pixels, 100, 50, "", 20, 20, 7), requires_inference=True)
    objects = session._tracking_detections_for_frame(pixels, catchup=True, evidence=sample)
    assert objects == [{"status": "inference_deferred"}]
    assert detector.detect_tracking.call_args.args[0] is pixels


def test_demand_timeline_retains_uninferred_frames_without_polling_metadata():
    capture, recorder = Mock(), Mock()
    recorder.recording_rows_between.return_value = []
    timeline = CameraFrameTimeline(camera=camera(), capture=capture, recorder=recorder,
        stop_event=threading.Event(), sample_fps=lambda: 5, requires_inference=True)
    frame = CapturedFrame("live", np.zeros((20, 20, 3), dtype=np.uint8), 100, 50, "", 20, 20, 7,
                          generation=1, source_pts=3, source_session="test")
    timeline.remember_capture(frame)
    assert timeline.live_frames[0].captured is frame
    assert timeline.live_frames[0].requires_inference
    capture.matched_snapshot.assert_not_called()
    batch = timeline.read_recorded_frames(99.8, 100, 5, 640)
    assert len(batch.frames) == 1
    assert batch.frames[0].captured is frame
    assert batch.frames[0].requires_inference

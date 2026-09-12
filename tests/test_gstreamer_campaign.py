"""Integration regressions for GStreamer evidence and lifecycle boundaries."""

from datetime import datetime, timezone
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import cv2

from survng.app.config import ObjectTrackingConfig
from survng.app.dlstreamer_capture import _SharedLiveProcess, _StreamInbox
from survng.app.camera_capture import CameraCaptureService
from survng.app.object_track.session import ObjectTrackingSession
from survng.app.object_track.types import TrackingFrame
from survng.openvino_config import latency_compile_config
from tests.test_gstreamer_review_contracts import captured, snapshot
from tests.test_tracking_frames import _service


@pytest.mark.parametrize("device", ["GPU", "GPU.0", "gpu.1"])
def test_gpu_compilation_is_bounded_without_reducing_inference(device):
    assert latency_compile_config(device) == {
        "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "COMPILATION_NUM_THREADS": 1,
    }


@pytest.mark.parametrize("device", ["CPU", "NPU"])
def test_non_gpu_compilation_keeps_existing_policy(device):
    assert latency_compile_config(device) == {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}


@pytest.mark.parametrize("device", ["AUTO", "AUTO:GPU,CPU", "MULTI:GPU,CPU", "HETERO:GPU,CPU"])
def test_virtual_device_compilation_scopes_thread_limit_to_gpu(device):
    settings = latency_compile_config(device)
    assert "COMPILATION_NUM_THREADS" not in settings
    assert settings["DEVICE_PROPERTIES"] == {"GPU": {"COMPILATION_NUM_THREADS": 1}}
    assert ("NUM_STREAMS" in settings) == (device != "AUTO")


@pytest.mark.parametrize("source,promoted", [
    ("live_fast_path", False), ("live_fallback", False),
    ("recorded_main", False), ("live_fast_path", True),
])
def test_tracking_seed_respects_live_provenance_and_consumes_inference_once(monkeypatch, source, promoted):
    epoch = 1000.0
    obj = {"label": "person", "confidence": .9,
           "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 80},
           "incident_eligible": True}
    seed = dict(obj)
    seed["frame_source"] = source
    if source == "live_fast_path":
        seed.update(frame_source="live_fast_path", live_detection_session="s",
                    live_inference_sequence=1, live_detection_source_pts=epoch)
    if promoted:
        seed["snapshot_source"] = "recorded_main"
    timeline = _service(capture=SimpleNamespace(matched_snapshot=lambda *_a, **_k: None))
    for index in range(1, 7):
        result = snapshot(epoch if index == 1 else epoch + index / 2, index, [obj])
        timeline.live_frames.append(TrackingFrame(captured(epoch + index / 2, index), result))
    updates = []
    detector = SimpleNamespace(config=SimpleNamespace(confidence_threshold=.7), detect=Mock())
    session = ObjectTrackingSession(
        camera=timeline.camera, config=ObjectTrackingConfig(max_session_seconds=3),
        detector=detector, frame_provider=lambda: None,
        catchup_frame_provider=timeline.read_recorded_frames,
        update_event=lambda _id, tracking, _objects: updates.append(tracking) or {},
        publisher=None, limiter=threading.BoundedSemaphore(1),
    )
    appearances, covers = Mock(), Mock()
    monkeypatch.setattr(session, "_annotate_appearances", appearances)
    monkeypatch.setattr(session, "_consider_cover_candidate", covers)
    session.set_accepting(True)
    try:
        # EMA expands luma to three channels; shape alone cannot establish color.
        assert session.start(42, datetime.fromtimestamp(epoch, timezone.utc), [seed],
                             np.zeros((180, 320, 3), np.uint8))
        assert session.wait_stopped(2)
    finally:
        session.stop()
    assert updates[-1]["completion_reason"] == "tracking_window_complete"
    assert updates[-1]["frames_processed"] == (5 if source == "live_fast_path" else 6)
    luma_seed = source in {"live_fast_path", "live_fallback"} and not promoted
    assert appearances.call_count == covers.call_count == (0 if luma_seed else 1)
    detector.detect.assert_not_called()


def test_color_main_history_wins_near_tie_with_live_sidecar():
    timeline = _service(capture=SimpleNamespace(matched_snapshot=lambda *_a, **_k: None))
    timeline.live_frames.append(TrackingFrame(captured(100), snapshot(100)))
    timeline.remember(np.zeros((180, 320, 3), np.uint8), 100.1)
    batch = timeline.read_recorded_frames(99.9, 101, 2, 640)
    assert len(batch.frames) == 1
    assert batch.frames[0][0] == 100.1
    assert batch.frames[0][1].ndim == 3


def test_finalized_recording_wins_near_tie_over_both_capture_histories(tmp_path, monkeypatch):
    timeline = _service(capture=SimpleNamespace(matched_snapshot=lambda *_a, **_k: None))
    timeline.live_frames.append(TrackingFrame(captured(100), snapshot(100)))
    timeline.remember(np.zeros((180, 320, 3), np.uint8), 100.1)
    path = tmp_path / "recording.mp4"
    path.touch()
    timeline.recorder.recording_rows_between.return_value = [
        {"path": str(path), "start_epoch": 100, "end_epoch": 101},
    ]
    recorded = np.ones((180, 320, 3), np.uint8)
    monkeypatch.setattr("survng.app.tracking_frames.sampled_video_frames",
                        lambda *_a, **_kw: iter([(100.2, recorded)]))
    batch = timeline.read_recorded_frames(99.9, 101, 2, 640)
    assert len(batch.frames) == 1
    assert batch.frames[0][0] == 100.2
    assert batch.frames[0][1] is recorded


def test_camera_video_resume_does_not_masquerade_as_shared_inference_stall(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("survng.app.dlstreamer_capture.time.monotonic", lambda: now[0])
    shared = _SharedLiveProcess([], read_timeout_ms=5000)
    inbox = _StreamInbox()
    shared._inboxes["live"] = inbox
    inbox.inference_started_at = inbox.last_inference_at = 9.6
    inbox.put_frame(np.zeros((2, 2), np.uint8), 1, 10)
    # A 4.8-second RTSP pause is within the camera's five-second read budget.
    # Video resumes before the asynchronous detector can complete its result.
    now[0] = 14.8
    inbox.put_frame(np.zeros((2, 2), np.uint8), 2, 14.8)
    shared._check_inference_progress(14.9)
    # Recovery grace must not defeat the watchdog if video then continues
    # without any inference progress (including empty results).
    for index in range(1, 27):
        now[0] = 14.8 + index / 5
        inbox.put_frame(np.zeros((2, 2), np.uint8), index + 2, now[0])
    with pytest.raises(RuntimeError, match="inference stalled"):
        shared._check_inference_progress(now[0])


def test_live_video_does_not_keep_an_old_jpeg_preview_fresh():
    now = [10.0]
    capture = CameraCaptureService(camera_id="gate", source_url=lambda _s: "", backend=Mock(),
                                   monotonic_clock=lambda: now[0])
    capture._stop.clear()
    ok, encoded = cv2.imencode(".jpg", np.zeros((10, 10, 3), np.uint8))
    assert ok
    capture._store_preview("live", SimpleNamespace(pop_jpeg=lambda: encoded.tobytes()))
    assert capture._publish_frame("live", np.zeros((10, 10), np.uint8))
    assert capture.latest_jpeg() is not None
    now[0] += capture.stale_seconds + .1
    assert capture._publish_frame("live", np.zeros((10, 10), np.uint8))
    assert capture.latest_jpeg() is None
    assert capture.latest_preview_image() is None

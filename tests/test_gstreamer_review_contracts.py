"""Cross-subsystem regressions from the v1.3 GStreamer integration review."""

from datetime import datetime, timezone
from fractions import Fraction
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from survng import dlstreamer_live as native
from survng.dlstreamer_model import configured_model
from survng.app.camera import CameraWorker
from survng.app.camera_capture import CapturedFrame, CaptureOpenLimiter
from survng.app.config import AppConfig, CameraConfig, ObjectTrackingConfig
from survng.app.config_application import motion_config_changes
from survng.app.dlstreamer_capture import DlStreamerCaptureBackend, DlStreamerCaptureOptions, _SharedLiveProcess
from survng.app.live_detections import DetectionHistory, DetectionSnapshot
from survng.app.object_track.session import ObjectTrackingSession
from survng.app.object_track.types import TrackingFrame
from tests.test_camera_worker import make_worker
from tests.test_tracking_frames import _service


def captured(epoch, sequence=1, session="s"):
    image = np.zeros((180, 320), np.uint8)
    image.setflags(write=False)
    return CapturedFrame("live", image, epoch, epoch, "", 320, 180, sequence, 1, epoch, session)


def snapshot(epoch, sequence=1, objects=(), session="s"):
    return DetectionSnapshot.parse({
        "schema_version": 1,
        "source_pts": epoch, "inference_sequence": sequence,
        "width": 320, "height": 180, "objects": list(objects),
    }, session=session)


def worker_for(timeline, alignment=None):
    worker = CameraWorker.__new__(CameraWorker)
    worker._stream_alignment = SimpleNamespace(enabled=False, observe=lambda _f: None)
    worker._effective_spatial_alignment = alignment or {"reliable": True}
    worker.runtime_state = SimpleNamespace(lock=threading.Lock(), generation=1)
    worker.motion_runtime = Mock()
    worker.tracking_frames = timeline
    return worker


def test_live_callback_restores_bridge_without_exposing_luma_to_color_refinement():
    history = DetectionHistory()
    history.reset("s")
    capture = SimpleNamespace(matched_snapshot=lambda _source, **kw: history.match(
        pts=kw["source_pts"], session=kw["source_session"], detect_fps=5,
    ))
    timeline = _service(capture=capture)
    worker = worker_for(timeline)
    for index in range(1, 13):
        epoch = 100 + index / 2
        history.add(snapshot(epoch, index))
        worker._capture_frame(captured(epoch, index))
    batch = timeline.read_recorded_frames(100, 106, 2, 640)
    assert worker.motion_runtime.submit_frame.call_count == 12
    assert len(batch.frames) == 12
    assert all(isinstance(item, TrackingFrame) and item.detection is not None for item in batch)
    assert batch.frames[-1].captured.source_pts == 106
    assert batch.frames[-1].captured.source_session == "s"
    assert batch.frames[-1].captured.image.ndim == 2
    assert list(timeline.recorded_frames(100, 106, 2, 640)) == []


def test_late_empty_result_survives_capture_history_eviction_and_reconnect():
    current = [None]
    capture = SimpleNamespace(matched_snapshot=lambda _source, **_kw: current[0])
    timeline = _service(capture=capture)
    timeline.remember_capture(captured(100))
    assert timeline.live_frames[0].detection is None
    current[0] = snapshot(100)
    # Below tracking sample interval: still harvest asynchronous metadata.
    timeline.remember_capture(captured(100.1, 2))
    assert timeline.live_frames[0].detection.objects == ()
    current[0] = None
    timeline.remember_capture(captured(100.5, 3, "new-session"))
    batch = timeline.read_recorded_frames(99.9, 101, 2, 640)
    assert batch.interruption == "capture_generation_changed"
    assert batch.frames[0].detection is not None
    assert batch.frames[0].captured.source_session == "s"


def test_late_exact_empty_replaces_earlier_positive_match():
    obj = {"label": "person", "confidence": .9,
           "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 80}}
    current = [snapshot(99.9, objects=[obj])]
    timeline = _service(capture=SimpleNamespace(matched_snapshot=lambda *_a, **_k: current[0]))
    timeline.remember_capture(captured(100))
    assert timeline.live_frames[0].detection.objects
    current[0] = snapshot(100, 2)
    timeline.remember_capture(captured(100.1, 2))
    assert timeline.live_frames[0].detection.objects == ()


def test_missing_sidecar_does_not_mask_available_main_history():
    timeline = _service(capture=SimpleNamespace(matched_snapshot=lambda *_a, **_k: None))
    timeline.remember_capture(captured(100))
    timeline.remember(np.zeros((180, 320, 3), np.uint8), 100.1)
    batch = timeline.read_recorded_frames(99.9, 101, 2, 640)
    assert len(batch.frames) == 1
    assert batch.frames[0][0] == 100.1
    assert batch.frames[0][1].ndim == 3


@pytest.mark.parametrize("alignment", [{"reliable": False}, {"reliable": True, "offset_x": .2}])
def test_untrusted_or_cropped_live_views_do_not_enter_main_coordinate_history(alignment):
    timeline = _service()
    worker_for(timeline, alignment)._capture_frame(captured(100))
    assert not timeline.live_frames


def test_buffered_sidecars_keep_tracking_continuity_without_rgb_inference(monkeypatch):
    # Exact media timestamps, independent of datetime microsecond rounding.
    seed = 1000.0
    obj = {"label": "person", "confidence": .9,
           "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 80}, "incident_eligible": True}
    timeline = _service(capture=SimpleNamespace(matched_snapshot=lambda *_a, **_k: None))
    positive = snapshot(seed + .5, 1, [obj])
    for index, result in enumerate((positive, positive, None,
                                    snapshot(seed + 2, 2), snapshot(seed + 2.5, 3), snapshot(seed + 3, 4)), 1):
        timeline.live_frames.append(TrackingFrame(captured(seed + index / 2, index), result))
    detector = SimpleNamespace(config=SimpleNamespace(confidence_threshold=.7), detect=Mock())
    updates = []
    session = ObjectTrackingSession(
        camera=timeline.camera, config=ObjectTrackingConfig(max_session_seconds=3),
        detector=detector, frame_provider=lambda: None,
        catchup_frame_provider=timeline.read_recorded_frames,
        update_event=lambda _id, tracking, _objects: updates.append(tracking) or {},
        publisher=None, limiter=threading.BoundedSemaphore(1),
    )
    enrich = Mock()
    appearances = Mock()
    covers = Mock()
    monkeypatch.setattr(session, "_enrich_tracking_depth", enrich)
    monkeypatch.setattr(session, "_annotate_appearances", appearances)
    monkeypatch.setattr(session, "_consider_cover_candidate", covers)
    session.set_accepting(True)
    try:
        assert session.start(42, datetime.fromtimestamp(seed, timezone.utc), [obj], np.zeros((180, 320, 3), np.uint8))
        assert session.wait_stopped(2)
    finally:
        session.stop()
    assert updates[-1]["completion_reason"] == "tracking_window_complete"
    assert updates[-1]["frames_processed"] == 4  # duplicate/missing did not count
    detector.detect.assert_not_called()
    enrich.assert_not_called()
    assert appearances.call_count == covers.call_count == 1  # color seed only


def run_supervisor():
    return native._run_supervisor(
        None, SimpleNamespace(test_source=False, main_fps=2), detect=False,
        model_path=None, instance_id="", rate=Fraction(5), detect_rate=Fraction(5),
        qualifier_width=320, jpeg_rate=None, open_timeout=3, stdout=io.BytesIO(),
    )


def test_timed_out_native_teardown_fences_replacement(monkeypatch):
    started, exited, release = (threading.Event() for _ in range(3))
    calls = []
    def pump(*_args, **kwargs):
        calls.append(kwargs["stream_id"])
        started.set()
        kwargs["stop_event"].wait(2)
        release.wait(2)
        exited.set()
    class Input:
        def __init__(self):
            self.buffer = self
            self.index = 0
        def readline(self):
            self.index += 1
            if self.index == 2:
                assert started.wait(1)
                return b'{"op":"remove","stream_id":"old"}\n'
            stream_id = "old" if self.index == 1 else "new"
            return (json.dumps({"op": "add", "stream_id": stream_id, "url": "rtsp://fixture.invalid/live"}) + "\n").encode()
    monkeypatch.setattr(native, "_pump_pipeline", pump)
    monkeypatch.setattr(native.sys, "stdin", Input())
    monkeypatch.setattr(native, "STREAM_STOP_TIMEOUT_SECONDS", .02)
    try:
        with pytest.raises(native.StreamShutdownError, match="supervisor replacement required"):
            run_supervisor()
        assert calls == ["old"]
        assert not exited.is_set()
    finally:
        release.set()
        assert exited.wait(1)


def test_native_stream_width_is_per_camera_and_main_remains_color_width(monkeypatch):
    widths = {}
    def pump(*_args, **kwargs):
        widths[kwargs["stream_id"]] = kwargs["qualifier_width"]
        kwargs["stop_event"].wait(1)
    commands = [json.dumps({"op": "add", "stream_id": role + str(width),
                           "source_role": role, "frame_width": width,
                           "url": "rtsp://fixture.invalid/live"}) for role, width in (("live", 320), ("live", 960), ("main", 960))]
    monkeypatch.setattr(native, "_pump_pipeline", pump)
    monkeypatch.setattr(native.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(("\n".join(commands) + "\n").encode())))
    assert run_supervisor() == 0
    assert widths == {"live320": 320, "live960": 960, "main960": 640}


def test_camera_resolution_override_reaches_capture_and_restarts_only_that_camera(tmp_path):
    current = AppConfig(cameras=[CameraConfig(id="gate", name="Gate", stream_url="rtsp://fixture.invalid/main")])
    incoming = current.model_copy(deep=True)
    incoming.cameras[0].motion_qualification.frame_width = 960
    worker = make_worker(incoming.cameras[0], tmp_path, motion_config=incoming.motion_qualification)
    assert worker.capture._frame_width() == 960
    assert motion_config_changes(current, incoming)[0] == {"gate"}
    shared = _SharedLiveProcess([], read_timeout_ms=100)
    shared._send = Mock()
    shared.add_stream("gate", "rtsp://fixture.invalid/main", frame_width=worker.capture._frame_width())
    assert shared._send.call_args.args[0]["frame_width"] == 960


def test_nms_command_propagates_configured_threshold():
    commands = []
    for value in (.1, .9):
        backend = DlStreamerCaptureBackend(CaptureOpenLimiter(), DlStreamerCaptureOptions(nms_threshold=value))
        commands.append(backend.command())
        assert native._parser().parse_args(commands[-1][3:]).nms_threshold == value
        backend.close()
    assert commands[0] != commands[1]


def test_nms_ir_override_preserves_weights_exporter_metadata_and_original(tmp_path):
    model = tmp_path / "model.xml"
    original = '<net><rt_info><model_info><model_type value="YOLO"/><end2end value="True"/><iou_threshold value="0.7"/></model_info></rt_info></net>'
    model.write_text(original)
    weights = model.with_suffix(".bin")
    weights.write_bytes(b"weights")
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text("description: YOLO26\ntask: detect\n")
    with configured_model(model, "", .3) as (configured, proc):
        assert proc == ""
        assert ET.parse(configured).find("rt_info/model_info/iou_threshold").get("value") == "0.3"
        assert ET.parse(configured).find("rt_info/model_info/end2end").get("value") == "True"
        assert configured.with_suffix(".bin").resolve() == weights
        assert (configured.parent / "metadata.yaml").resolve() == metadata
    assert not configured.exists()
    assert model.read_text() == original and weights.read_bytes() == b"weights"


def test_nms_model_proc_preserves_converter_preprocessing_and_final_outputs(tmp_path):
    proc = tmp_path / "proc.json"
    original = {"json_schema_version": "2.2.0", "input_preproc": [{"format": "image", "color_space": "RGB"}],
                "output_postproc": [{"converter": "yolo_v26", "labels": ["person"]}]}
    proc.write_text(json.dumps(original))
    with configured_model(tmp_path / "model.xml", str(proc), .2) as (model, configured):
        result = json.loads(Path(configured).read_text())
        assert result["input_preproc"] == original["input_preproc"]
        assert result["output_postproc"] == [{"converter": "yolo_v26", "labels": ["person"], "iou_threshold": 1.0}]
        assert model == tmp_path / "model.xml"
    assert json.loads(proc.read_text()) == original


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.0])
def test_invalid_nms_policy_fails_visibly(tmp_path, value):
    with pytest.raises(ValueError, match="NMS threshold"):
        with configured_model(tmp_path / "model.xml", "", value):
            pass

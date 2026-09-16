from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
import time

import pytest

from survng.app.config import AppConfig, CameraConfig, DetectorConfig
from survng.app.live_detections import DetectionSnapshot
from survng.app.native_activity import NativeActivity


def observation(sequence, *, objects=None, session="stream-a", provenance="native_fresh_detection", received=100):
    if objects is None:
        objects = [{"label": "person", "confidence": .9, "box": {"x1": 10, "y1": 10, "x2": 30, "y2": 80}, "native_track_id": 7,
                    "detection_provenance": "native_fresh_detection"}]
    return DetectionSnapshot(sequence / 5, sequence, 100, 100, tuple(objects), session, provenance, received)


@pytest.fixture
def activity():
    events = Mock()
    events.add_event.side_effect = [{"id": 1}, {"id": 2}, {"id": 3}]
    return NativeActivity(CameraConfig(id="front", name="Front", stream_url="rtsp://example.test/live"),
                          DetectorConfig(enabled=True), events, Mock(), Mock(return_value=""))


def feed(activity, sequence, **kwargs):
    activity.consume(observation(sequence, **kwargs), now=100 + sequence / 5, epoch=1000 + sequence / 5)


def test_admission_requires_distinct_fresh_observations(activity):
    feed(activity, 1)
    feed(activity, 1)
    assert activity.event_id is None
    feed(activity, 2)
    assert activity.event_id == 1
    assert activity.events.add_event.call_count == 1
    assert activity.tracks[(7, "person")]["observations"] == 2


def test_predictions_cannot_create_or_extend_activity(activity):
    feed(activity, 1, provenance="native_tracked_prediction")
    assert activity.tracks == {}
    feed(activity, 2)
    feed(activity, 3)
    assert activity.event_id == 1
    for sequence in range(4, 32):
        feed(activity, sequence, provenance="native_tracked_prediction", received=100 + sequence / 5)
        activity.tick(now=100 + sequence / 5)
    assert activity.event_id is None
    assert activity.events.add_event.call_count == 1


def test_prediction_roi_on_fresh_frame_is_not_evidence(activity):
    obj = dict(observation(1).objects[0], detection_provenance="native_tracked_prediction", confidence=1.0)
    feed(activity, 1, objects=[obj])
    feed(activity, 2, objects=[obj])
    assert activity.tracks == {}
    activity.events.add_event.assert_not_called()


def test_empty_fresh_result_ends_presence(activity):
    feed(activity, 1)
    feed(activity, 2)
    for sequence in range(3, 30):
        feed(activity, sequence, objects=[], received=100 + sequence / 5)
    assert activity.event_id is None
    assert activity.health == "healthy"
    assert activity.events.update_object_tracking.call_args.args[1]["state"] == "complete"



def test_completion_timer_between_fresh_frames_waits_for_evidence(activity):
    feed(activity, 1)
    feed(activity, 2)
    # Last presence at 100.4; empty frames are healthy up to 105.2.
    for sequence in range(3, 27):
        feed(activity, sequence, objects=[], received=100 + sequence / 5)
    activity.tick(now=105.45)
    assert activity.health == "healthy"
    assert activity.event_id == 1
    # The next fresh empty frame covers the inactivity deadline.
    activity.consume(observation(27, objects=[], received=105.6), now=105.6, epoch=1005.6)
    assert activity.event_id is None
    assert activity.events.update_object_tracking.call_args.args[1]["state"] == "complete"


def test_completion_wait_still_detects_metadata_loss(activity):
    feed(activity, 1)
    feed(activity, 2)
    for sequence in range(3, 27):
        feed(activity, sequence, objects=[], received=100 + sequence / 5)
    activity.tick(now=105.45)
    assert activity.event_id == 1
    activity.tick(now=107.3)
    assert activity.event_id is None
    assert activity.health == "metadata_stale"
    assert activity.events.update_object_tracking.call_args.args[1]["state"] == "metadata_lost"

def test_reconnect_restarts_confirmation_and_qualifies_identity(activity):
    feed(activity, 1)
    feed(activity, 2)
    first = activity.tracks[(7, "person")]["native_identity"]
    feed(activity, 1, session="stream-b")
    assert activity.event_id is None
    feed(activity, 2, session="stream-b")
    assert activity.event_id == 2
    assert activity.tracks[(7, "person")]["native_identity"] != first


def test_missing_metadata_reports_health_and_settles(activity):
    feed(activity, 1)
    feed(activity, 2)
    activity.tick(now=107)
    assert activity.health == "metadata_stale"
    assert activity.event_id is None
    assert activity.events.update_object_tracking.call_args.args[1]["state"] == "metadata_lost"


@pytest.mark.parametrize("change", [{"native_track_id": None}, {"native_track_id": True}, {"confidence": .01}])
def test_unusable_observations_do_not_admit(activity, change):
    obj = dict(observation(1).objects[0], **change)
    feed(activity, 1, objects=[obj])
    feed(activity, 2, objects=[obj])
    activity.events.add_event.assert_not_called()


def test_stale_and_unknown_results_cannot_admit(activity):
    feed(activity, 1, received=90)
    feed(activity, 2, provenance="unknown")
    activity.events.add_event.assert_not_called()


def test_missing_fresh_confirmation_is_not_consecutive(activity):
    feed(activity, 1)
    feed(activity, 2, objects=[])
    feed(activity, 3)
    activity.events.add_event.assert_not_called()
    feed(activity, 4)
    assert activity.event_id == 1


def test_native_manager_has_no_python_model_workers(tmp_path):
    from survng.app.manager import AppManager
    config = AppConfig(storage_dir=str(tmp_path / "storage"), database_dir=str(tmp_path / "db"),
                       recording_index_dir=str(tmp_path / "index"), cameras=[], retention={"enabled": False})
    manager = AppManager(config)
    try:
        manager.inference.start_core()
        manager.inference.start_auxiliary()
        assert not hasattr(manager.detector, "detect")
        assert not hasattr(manager.inference, "tracking_factory")
        assert manager.detector_status()["python_inference_workers"] == 0
        assert not manager.face_recognizer.enabled
        assert not manager.person_reidentifier.enabled
    finally:
        manager.stop_all()


def test_native_event_persists_replay_and_notification_duration(tmp_path):
    from survng.app.event_store import EventStore
    from survng.app.incident_lifecycle import IncidentLifecycle
    from survng.app.incident_presenter import _event_row, _incident_row
    from survng.app.incident_utils import event_end_epoch
    events = EventStore(tmp_path)
    activity = NativeActivity(CameraConfig(id="front", name="Front", stream_url="rtsp://example.test/live"),
                              DetectorConfig(enabled=True), events, Mock(), Mock(return_value=""))
    feed(activity, 1)
    feed(activity, 2)
    for sequence in range(3, 18):
        feed(activity, sequence, received=100 + sequence / 5)
    row = events.get(activity.event_id)
    public = _event_row(row)
    track = public["object_tracking"]["tracks"][0]
    assert track["box_history"][-1][0] > track["box_history"][0][0]
    assert public["object_tracking"]["frame_width"] == 100
    assert event_end_epoch(row) > 1002
    lifecycle = IncidentLifecycle(Mock())
    lifecycle.start()
    try:
        lifecycle.track_incident(row, "Front")
        payload = lifecycle.snapshot()[0]
        assert payload["duration_seconds"] >= 2
        lifecycle.complete_event(row["id"])
        assert lifecycle.snapshot()[0]["state"] == "complete"
    finally:
        lifecycle.close()


def test_native_only_routes_do_not_register_comparison_or_python_inference():
    from survng.app.main import app
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    # Routers are also inspected directly on FastAPI versions with deferred inclusion.
    from survng.app.native_routes import create_native_router
    native = create_native_router(lambda: None)
    assert any(route.path.endswith("/native") for route in native.routes)
    assert not any("tracking-comparison" in path or path.endswith("/detect") or "motion-debug" in path for path in paths)


def test_missing_model_cannot_silently_use_frames_only():
    from survng.app.manager import validate_manager_configuration
    with pytest.raises(ValueError, match="model_path"):
        validate_manager_configuration(AppConfig(detector={"enabled": True}))


@pytest.mark.parametrize("interval", [1, 3])
def test_real_native_camera_creates_incident_without_python_inference(tmp_path, interval):
    """Real synthetic GStreamer source -> IPC -> capture -> native activity -> DB."""
    import json
    import subprocess
    from survng.app.dlstreamer_capture import DlStreamerCaptureBackend, DlStreamerCaptureOptions
    from survng.app.camera_capture import CaptureOpenLimiter
    from survng.app.event_store import EventStore
    from survng.app.image_storage import DurableImageWriter
    from survng.app.config import ImageStorageConfig
    from survng.app.native_camera import NativeCameraWorker
    probe = subprocess.run(["/usr/bin/python3", "-c", "from survng.dlstreamer_live import _apply_dlstreamer_env; _apply_dlstreamer_env(); import subprocess,sys; sys.exit(subprocess.call([sys.executable, '-c', \"import openvino; from survng.dlstreamer_live import _load_gstreamer; assert _load_gstreamer().ElementFactory.find('gvadetect')\"]))"], capture_output=True, timeout=20)
    if probe.returncode:
        pytest.skip("system DL Streamer runtime unavailable")
    model = tmp_path / "fixture.xml"
    generator = '''import sys,numpy as np,openvino as ov
from openvino import opset13 as ops
image=ops.parameter([1,3,64,64],np.float32,name="image")
zero=ops.multiply(ops.reduce_mean(image,ops.constant([0,1,2,3]),False),ops.constant(0,np.float32))
boxes=ops.constant(np.array([[[[0,1,.9,.1,.1,.6,.8],[-1,0,0,0,0,0,0]]]],dtype=np.float32))
output=ops.add(boxes,zero,name="detection_out")
output.output(0).get_tensor().set_names({"detection_out"})
ov.save_model(ov.Model([output],[image]),sys.argv[1],compress_to_fp16=False)
'''
    subprocess.run(["/usr/bin/python3", "-c", generator, str(model)], capture_output=True, check=True, timeout=20)
    proc = model.with_suffix(".json")
    proc.write_text(json.dumps({"json_schema_version": "2.2.0", "input_preproc": [{"layer_name": "image", "format": "image"}], "output_postproc": [{"layer_name": "detection_out", "converter": "detection_output", "labels": ["background", "person"]}]}))
    from survng.app.manager import AppManager
    manager = AppManager(AppConfig(
        storage_dir=str(tmp_path), database_dir=str(tmp_path / "db"),
        recording_index_dir=str(tmp_path / "recording-index"),
        detector=DetectorConfig(enabled=True, model_path=str(model), native={"inference_interval": interval}),
        cameras=[CameraConfig(id="front", name="Front", stream_url="rtsp://unused.invalid/live", record=False)],
        retention={"enabled": False},
    ))
    backend = manager.capture_backend
    original_command = backend.command
    backend.command = lambda *args, **kwargs: [*original_command(*args, **kwargs), "--test-source"]
    events = manager.events
    worker = manager.workers["front"]
    try:
        manager.start_all()
        deadline = time.monotonic() + 20
        while worker.activity.event_id is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert worker.activity.event_id is not None, worker.status()
        event_id = worker.activity.event_id
        row = events.get(event_id)
        assert row["topic"] == "native/object-presence"
        assert worker.status()["native_activity"]["counters"]["fresh_frames"] >= 2
        assert worker.status()["live_pipeline"]["native_tracking"] == "short-term-imageless"
        assert worker.status()["live_pipeline"]["inference_interval"] == interval
        if interval > 1:
            assert worker.status()["native_activity"]["counters"].get("prediction_frames", 0) > 0
        old_session = worker.activity.session
        # Kill only this test's private synthetic-source process. Capture must
        # recover natively and cannot carry an ID into the new stream session.
        backend._shared._process.kill()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if worker.activity.event_id not in {None, event_id} and worker.activity.session != old_session:
                break
            time.sleep(.05)
        assert worker.activity.event_id not in {None, event_id}, worker.status()
        assert worker.activity.session != old_session
        from survng.app.incident_utils import incident_event_groups
        assert len(incident_event_groups([events.get(event_id), events.get(worker.activity.event_id)])) == 2
        manager.set_detection("front", False)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = worker.status()
            if status.get("live_pipeline", {}).get("detect") is False and status["connected"]:
                break
            time.sleep(.05)
        assert status["live_pipeline"]["detect"] is False
        assert status["connected"]
        assert worker.activity.event_id is None
        assert manager.detector_status()["python_inference_workers"] == 0
        assert manager.incidents.snapshot()[0]["state"] == "complete"
        assert manager._mqtt_server_status()["state"]["health"] in {"ok", "degraded"}
    finally:
        manager.stop_all()


def test_post_tracker_metadata_keeps_fresh_identity_separate_from_prediction():
    import json
    from types import SimpleNamespace
    from survng.dlstreamer_live import _detection_metadata
    fresh = dict(observation(1).objects[0])
    post_tracker = {"objects": [
        {"id": 7, "x": 10, "y": 10, "w": 20, "h": 70, "detection": {"label": "person", "confidence": .9}},
        {"id": 9, "x": 40, "y": 10, "w": 20, "h": 70, "detection": {"label": "person", "confidence": 1.0}},
    ]}
    structure = SimpleNamespace(get_value=lambda name: 100)
    caps = SimpleNamespace(get_structure=lambda index: structure)
    sample = SimpleNamespace(get_buffer=lambda: SimpleNamespace(pts=200000000), get_caps=lambda: caps)
    video_frame = lambda *args, **kwargs: SimpleNamespace(messages=lambda: [json.dumps(post_tracker)])
    result = _detection_metadata(sample, video_frame, inference_sequence=1, gst_second=1000000000,
                                 clock_time_none=-1, native_result=("native_fresh_detection", [fresh]))
    assert result["objects"][0]["native_track_id"] == 7
    assert result["objects"][0]["detection_provenance"] == "native_fresh_detection"
    assert result["objects"][1]["detection_provenance"] == "native_tracked_prediction"


def test_native_zones_filter_admission_and_main_overlay_requires_known_geometry(activity):
    from survng.app.config import DetectionZone
    activity.camera.live_stream_url = "rtsp://example.test/cropped"
    activity.camera.zones = [DetectionZone(name="ignore-person", behavior="ignore",
        points=[{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}])]
    feed(activity, 1)
    feed(activity, 2)
    assert activity.event_id is None
    activity.camera.zones = []
    feed(activity, 3)
    feed(activity, 4)
    payload = activity.events.update_object_tracking.call_args.args[1]
    assert payload["recording_overlay_compatible"] is False
    activity.camera.native_same_field_of_view = True
    activity.persist("active", now=102)
    assert activity.events.update_object_tracking.call_args.args[1]["recording_overlay_compatible"] is True


def test_zone_threshold_lowering_rebuilds_native_graph():
    import threading
    from survng.app.config import DetectionZone
    from survng.app.config_routes import ConfigRouteDependencies, create_config_router
    camera = CameraConfig(id="front", name="Front", stream_url="rtsp://unused.invalid/live")
    current = AppConfig(cameras=[camera])
    runtime = Mock(workers={"front": Mock()})
    apply = Mock(return_value=(current, {"camera_workers_restarted": True}))
    deps = ConfigRouteDependencies(get_config=lambda: current, get_manager=lambda: runtime,
        publish_config=Mock(), apply_config=apply, reload_manager=Mock(), save_config=Mock(),
        validate_config=Mock(), lock=threading.RLock(), probe_limiter=threading.BoundedSemaphore(1))
    bundle = create_config_router(deps)
    endpoint = next(route.endpoint for route in bundle.routes if route.path == "/api/config/cameras/{camera_id}/zones")
    zone = DetectionZone(name="person", confidence_threshold=.01,
        points=[{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}])
    result = endpoint("front", [zone])
    assert result["workers_restarted"]
    apply.assert_called_once()
    runtime.update_camera_zones.assert_not_called()


def test_resolution_change_restarts_identity_and_confirmation(activity):
    feed(activity, 1)
    feed(activity, 2)
    original = activity.tracks[(7, "person")]["native_identity"]
    activity.consume(replace(observation(3), width=200), now=100.6, epoch=1000.6)
    assert activity.event_id is None
    activity.consume(replace(observation(4), width=200), now=100.8, epoch=1000.8)
    assert activity.event_id == 2
    assert activity.tracks[(7, "person")]["native_identity"] != original


@pytest.mark.parametrize("new_session", [True, False])
def test_reconnected_video_gets_metadata_grace_after_long_outage(monkeypatch, new_session):
    import threading
    from types import SimpleNamespace
    from survng.app.native_camera import NativeCameraWorker
    monkeypatch.setattr("survng.app.native_camera.time.monotonic", lambda: 100.0)
    worker = NativeCameraWorker.__new__(NativeCameraWorker)
    worker._stop = Mock()
    worker._stop.wait.side_effect = [False, True]
    worker._stop.is_set.return_value = False
    worker._lock = threading.RLock()
    worker._enabled_at = 1.0
    worker._last_native_healthy_at = 1.0
    worker._native_frame_session = "old-session"
    worker.runtime_state = SimpleNamespace(detection_enabled=True)
    worker.config = DetectorConfig(enabled=True)
    worker.capture = Mock()
    worker.capture.native_observations.return_value = []
    worker.capture.latest.return_value = SimpleNamespace(
        source_session="reconnected-session" if new_session else "old-session", captured_at_monotonic=100.0)
    worker.activity = Mock(health="metadata_stale")
    worker._restart_native_stream = Mock()
    worker._run()
    if new_session:
        worker._restart_native_stream.assert_not_called()
        assert worker._last_native_healthy_at == 100.0
    else:
        worker._restart_native_stream.assert_called_once_with(100.0)
    assert worker.activity.health != "healthy"


def test_native_observation_clock_is_stable_across_arrival_bursts():
    from types import SimpleNamespace
    from survng.app.native_camera import NativeCameraWorker
    worker = NativeCameraWorker.__new__(NativeCameraWorker)
    worker._observation_clock = None
    worker._last_observation_epoch = 0
    first = replace(observation(1), source_pts=9)
    frame = SimpleNamespace(source_session=first.session, source_pts=10, captured_at_epoch=100)
    assert worker._observation_epoch(first, frame, 100, 100) == 99
    # Decode catches up in a burst; the frame receipt offset changes by .9s.
    frame.source_pts, frame.captured_at_epoch = 11, 100.1
    second = replace(first, source_pts=9.2)
    assert worker._observation_epoch(second, frame, 100.1, 100.1) == pytest.approx(99.2)
    reconnected = replace(first, source_pts=0, session="new", received_monotonic=101)
    assert worker._observation_epoch(reconnected, None, 101, 101) == 101


@pytest.mark.parametrize("interval", [1, 3, 5])
def test_native_interval_reaches_capture_and_status(tmp_path, interval):
    from survng.app.manager import AppManager
    config = AppConfig(storage_dir=str(tmp_path), cameras=[], retention={"enabled": False})
    config.detector.native.inference_interval = interval
    manager = AppManager(config)
    try:
        assert manager.capture_backend.options.inference_interval == interval
        assert manager.detector_status()["inference_interval"] == interval
        assert manager.detector_status()["effective_inference_fps"] == 5 / interval
    finally:
        manager.stop_all()


def test_sparse_fresh_frames_preserve_confirmation_and_health(activity):
    activity.config.live_sample_fps = 0.5
    activity.config.native.inference_interval = 5
    for sequence, now in [(1, 100), (2, 110)]:
        activity.consume(observation(sequence, received=now), now=now, epoch=1000+now)
    assert activity.event_id == 1
    activity.tick(now=120)
    assert activity.health == "healthy"
    assert activity.event_id == 1
    tracking = activity.events.update_object_tracking.call_args.args[1]
    assert tracking["sample_fps"] == 0.1
    activity.tick(now=126)
    assert activity.health == "metadata_stale"
    assert activity.event_id is None


@pytest.mark.parametrize("interval", [0, 6, 1.5])
def test_native_interval_rejects_invalid_values(interval):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        DetectorConfig(native={"inference_interval": interval})


def test_native_interval_change_requires_shared_capture_reload():
    from survng.app.config_application import manager_owned_config
    current = AppConfig()
    incoming = current.model_copy(deep=True)
    incoming.detector.native.inference_interval = 2
    assert manager_owned_config(current) != manager_owned_config(incoming)

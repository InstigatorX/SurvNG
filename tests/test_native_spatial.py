from dataclasses import replace
from unittest.mock import Mock

import pytest

from survng.app.config import CameraConfig, DetectionZone, DetectorConfig, NativeRoiConfig
from survng.app.native_activity import NativeActivity
from survng.app.live_detections import DetectionSnapshot
from survng.app.zones import apply_detection_zones
from survng.native_spatial import analytics_zones, inference_rectangle, spatial_plan, RoiInput, ROI_LABEL


def camera():
    return CameraConfig(id="gate", name="Gate", stream_url="rtsp://unused.invalid/live", zones=[
        DetectionZone(name="drive", points=[{"x": .2, "y": .4}, {"x": .8, "y": .4}, {"x": .8, "y": .9}, {"x": .2, "y": .9}]),
        DetectionZone(name="ignore", behavior="ignore", object_classes=["car"], points=[{"x": .5, "y": .5}, {"x": 1, "y": .5}, {"x": 1, "y": 1}, {"x": .5, "y": 1}]),
    ])


def test_roi_defaults_padding_selection_and_safe_fallback():
    c = camera()
    assert inference_rectangle(spatial_plan(c), 1000, 500) == (0, 0, 1000, 500)
    c.native_roi = NativeRoiConfig(enabled=True, padding=.1)
    assert inference_rectangle(spatial_plan(c), 1000, 500) == (100, 150, 800, 350)
    c.native_roi.zone_names = ["ignore"]
    assert inference_rectangle(spatial_plan(c), 1000, 500) == (0, 0, 1000, 500)
    c.native_roi.zone_names = ["drive"]
    c.zones[0].enabled = False
    assert inference_rectangle(spatial_plan(c), 1000, 500) == (0, 0, 1000, 500)


def test_plan_revision_and_geometry():
    c = camera()
    p = spatial_plan(c)
    assert analytics_zones(p, 100, 200)[0]["points"][0] == {"x": 20, "y": 80}
    c.zones[0].points[0].x = .3
    assert spatial_plan(c)["revision"] != p["revision"]
    assert p["zones"][0]["points"][0]["x"] == .2
    c.zones[0].enabled = False
    assert analytics_zones(spatial_plan(c), 100, 200)[0]["id"] == "1"


@pytest.mark.parametrize("interval", [1, 3, 5])
def test_full_frame_sweeps_follow_inference_count(interval):
    c = camera()
    c.native_roi = NativeRoiConfig(enabled=True, padding=0, full_frame_interval=3)
    roi = RoiInput(spatial_plan(c), interval)
    frame = Mock()
    frame.video_info.return_value = Mock(width=100, height=100)
    rectangles = []
    for _ in range(interval * 7):
        assert roi.process_frame(frame)
        rectangles.append(frame.add_region.call_args.args)
    assert all(rectangles[i][:4] == (0, 0, 100, 100) for i in (0, interval*3, interval*6))
    assert rectangles[interval][:4] == (20, 40, 60, 50)
    assert all(rect[-2:] == (ROI_LABEL, 1.0) for rect in rectangles)


@pytest.mark.parametrize("label,confidence,x,y,native_ids", [
    ("person", .9, 30, 60, ["0"]), ("car", .9, 60, 70, ["0", "1"]),
    ("person", .9, 60, 70, ["0", "1"]), ("car", .2, 30, 60, ["0"]),
    ("person", .9, 10, 30, []),
    # Native ray casting excludes right/bottom edges; SurvNG includes them.
    ("person", .9, 80, 70, []), ("person", .9, 30, 90, []),
])
def test_native_membership_preserves_incident_policy(label, confidence, x, y, native_ids):
    c = camera()
    obj = {"label": label, "confidence": confidence, "box": {"x1": x-5, "x2": x+5, "y1": y-10, "y2": y}, "native_zone_ids": native_ids}
    from copy import deepcopy
    expected = apply_detection_zones(c, [deepcopy(obj)], 100, 100, .5, True, {})
    actual = apply_detection_zones(c, [deepcopy(obj)], 100, 100, .5, True, {}, native_membership=True)
    assert actual == expected


def test_untracked_context_falls_back_to_python_zone_geometry():
    c = camera()
    obj = {
        "label": "person",
        "confidence": .9,
        "box": {"x1": 25, "x2": 35, "y1": 50, "y2": 60},
    }
    from copy import deepcopy
    expected = apply_detection_zones(
        c, [deepcopy(obj)], 100, 100, .5, True, {}
    )
    actual = apply_detection_zones(
        c, [deepcopy(obj)], 100, 100, .5, True, {},
        native_membership=True,
    )
    assert actual == expected


def test_native_membership_replaces_polygon_work_away_from_edges(monkeypatch):
    c = camera()
    obj = {"label": "person", "confidence": .9, "box": {"x1": 25, "x2": 35, "y1": 50, "y2": 60}, "native_zone_ids": ["0"]}
    def forbidden(*args):
        raise AssertionError("full polygon evaluation called")
    monkeypatch.setattr("survng.app.zones._point_in_polygon", forbidden)
    assert apply_detection_zones(c, [obj], 100, 100, .5, native_membership=True)[0]["incident_eligible"]


def test_revision_mismatch_cannot_admit_or_supply_empty_coverage():
    c = camera()
    activity = NativeActivity(c, DetectorConfig(enabled=True, event_confirmation_frames=1), Mock(), Mock(), Mock(), native_zones=True)
    observation = DetectionSnapshot(1, 1, 100, 100, (), "session", "native_fresh_detection", 100, "old")
    activity.consume(observation, now=100, epoch=1000)
    assert activity.last_fresh == 0
    assert activity.counts["invalid_zone_metadata"] == 1
    activity.consume(replace(observation, source_pts=2, inference_sequence=2, zone_revision=spatial_plan(c)["revision"]), now=100, epoch=1000)
    assert activity.last_fresh == 100


def test_per_object_invalid_zone_ids_rejected_without_track_id():
    """Fail closed on bad native zone fields even when gvatrack IDs are absent."""
    c = camera()
    events = Mock()
    events.add_event.side_effect = [{"id": 1}]
    events.open_incident = Mock(return_value={"id": 10, "observation_count": 1})
    activity = NativeActivity(
        c, DetectorConfig(enabled=True, event_confirmation_frames=1), events, Mock(), Mock(), native_zones=True
    )
    revision = spatial_plan(c)["revision"]
    bad = {
        "label": "person",
        "confidence": 0.9,
        "box": {"x1": 25, "x2": 35, "y1": 50, "y2": 60},
        "native_zone_revision": revision,
        "native_zone_ids": ["999"],
    }
    observation = DetectionSnapshot(
        1, 1, 100, 100, (bad,), "session", "native_fresh_detection", 100, revision
    )
    activity.consume(observation, now=100, epoch=1000)
    assert activity.last_fresh == 0
    assert activity.health == "zone_metadata_invalid"
    assert activity.counts["invalid_zone_metadata"] == 1
    assert "native_track_id" not in bad

    good = {**bad, "native_zone_ids": ["0"]}
    activity.consume(
        replace(observation, source_pts=2, inference_sequence=2, objects=(good,)),
        now=100,
        epoch=1000,
    )
    assert activity.last_fresh == 100
    assert activity.health == "healthy"


@pytest.mark.parametrize("kwargs", [{"padding": -.1}, {"padding": .6}, {"full_frame_interval": 0}, {"full_frame_interval": 31}])
def test_invalid_roi_configuration(kwargs):
    with pytest.raises(ValueError):
        NativeRoiConfig(**kwargs)


def test_real_native_polygon_parity():
    import json
    import subprocess
    from pathlib import Path
    from copy import deepcopy
    if not Path('/opt/intel/dlstreamer').is_dir():
        pytest.skip('installed native runtime unavailable; native CI runs the spatial smoke')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['/usr/bin/python3', str(root/'scripts/gstreamer-spatial-check.py'), '--geometry'],
                            cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-3000:]
    data = json.loads(result.stdout)
    c = camera()
    c.zones = [DetectionZone.model_validate(z) for z in data['zones']]
    for record in data['records']:
        obj = record['object']
        expected = apply_detection_zones(c, [deepcopy(obj)], record['width'], record['height'], .5)
        actual = apply_detection_zones(c, [deepcopy(obj)], record['width'], record['height'], .5, native_membership=True)
        assert actual == expected, record


def test_binding_snapshots_plan_and_zone_updates_restart_capture():
    from survng.app.native_camera import NativeCaptureBinding, NativeCameraWorker
    import threading
    from types import SimpleNamespace
    c = camera()
    backend = Mock(startup_timeout_ms=10)
    handle = SimpleNamespace()
    backend.create_handle.return_value = handle
    binding = NativeCaptureBinding(backend, lambda: True, spatial=lambda: spatial_plan(c))
    binding.create_handle()
    old_revision = handle.spatial_plan['revision']
    c.zones[0].name = 'renamed'
    assert handle.spatial_plan['zones'][0]['name'] == 'drive'
    worker = SimpleNamespace(camera=c, runtime_state=SimpleNamespace(enabled=True), _lock=threading.RLock(),
                             activity=Mock(), stop=Mock(), start=Mock())
    updated_zones = [z.model_copy(deep=True) for z in c.zones]
    updated_zones[0].color = "#000000"
    NativeCameraWorker.update_zones(worker, updated_zones)
    worker.stop.assert_called_once()
    worker.start.assert_called_once()
    assert worker.activity.zone_revision != old_revision


def test_invalid_zone_metadata_is_rejected_by_native_adapter():
    from survng.dlstreamer_live import _normalize_gva_objects
    obj = {'label':'car', 'confidence':.9, 'x':0, 'y':0, 'w':10, 'h':10, 'zone_violations':'0'}
    assert _normalize_gva_objects({'objects':[obj]}) == []


def test_roi_change_requires_structural_reload():
    from survng.app.config import AppConfig
    from survng.app.config_application import manager_owned_config
    initial = AppConfig(cameras=[camera()])
    updated = initial.model_copy(deep=True)
    updated.cameras[0].native_roi.enabled = True
    assert manager_owned_config(initial) != manager_owned_config(updated)


def test_zone_start_failure_restores_previous_plan():
    from survng.app.native_camera import NativeCameraWorker
    import threading
    from types import SimpleNamespace
    c = camera()
    revision = spatial_plan(c)['revision']
    old = [z.model_copy(deep=True) for z in c.zones]
    worker = SimpleNamespace(camera=c, runtime_state=SimpleNamespace(enabled=True), _lock=threading.RLock(),
                             activity=Mock(), stop=Mock(), start=Mock(side_effect=[RuntimeError('startup failed'), None]))
    updated = [z.model_copy(deep=True) for z in old]
    updated[0].name = 'new'
    with pytest.raises(RuntimeError, match='startup failed'):
        NativeCameraWorker.update_zones(worker, updated)
    assert c.zones == old
    assert worker.activity.zone_revision == revision
    assert worker.start.call_count == 2

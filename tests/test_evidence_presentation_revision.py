"""Public annotations and current-event raster routes share one revision."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from survng.app.appearance_routes import AppearanceRouteDependencies, create_appearance_router
from survng.app.image_cache import LocalImageCache
from survng.app.incident_payload import canonical_incident_payload
from survng.app.incident_presenter import (
    _event_row, _incident_row, _incident_list_payload, _recording_grid_incident_payload,
)


def test_public_compact_and_notification_evidence_keeps_revision():
    raw = {
        "id": 7, "camera_id": "gate", "created_at": "2026-09-13T20:00:00+00:00",
        "snapshot_path": "snapshots/private-name.webp", "evidence_revision": 12,
        "objects_json": json.dumps([{
            "label": "person", "confidence": .9, "incident_eligible": True,
            "provisional_detection": True, "snapshot_visible": False,
        }]),
    }
    public = _event_row(raw)
    assert public["snapshot_path"] == "available"
    summary = _incident_list_payload(_incident_row("gate", [public]))
    assert summary["evidence_revision"] == summary["events"][0]["evidence_revision"] == 12
    assert summary["objects"][0]["snapshot_visible"] is False
    assert _recording_grid_incident_payload(summary)["evidence_revision"] == 12
    notification = canonical_incident_payload({
        **summary, "scene_objects": [{"label": "person", "certainty": "possible"}],
        "alert_decisions": [{"eligible": False}],
    })
    assert notification["snapshot_url"].endswith("/7/snapshot.jpg?v=12")
    assert notification["evidence_revision"] == 12
    # Observation membership is independent of whether evidence can alert yet.
    assert notification["objects"][0]["label"] == "person"
    assert notification["has_objects"] is True
    assert notification["alert_decisions"][0]["eligible"] is False


def _routes(tmp_path):
    path = tmp_path / "snapshots" / "frame.png"
    path.parent.mkdir()
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    frame[:, 120:] = 255
    assert cv2.imwrite(str(path), frame)
    event = {"id": 7, "snapshot_path": str(path), "evidence_revision": 1}
    manager = SimpleNamespace(
        storage_dir=tmp_path, events=SimpleNamespace(get=lambda _: dict(event)),
        image_cache=LocalImageCache(tmp_path / "cache"),
    )
    routes = create_appearance_router(AppearanceRouteDependencies(
        get_manager=lambda: manager, manager_lock=threading.RLock(),
    )).handlers
    return routes, event


@pytest.mark.parametrize("endpoint", ["event_snapshot", "event_thumbnail"])
def test_current_image_alias_revalidates_and_rejects_stale_revision(tmp_path, endpoint):
    routes, event = _routes(tmp_path)
    response = routes[endpoint](7, v="1")
    assert response.headers["cache-control"] == "private, no-cache"
    event["evidence_revision"] = 2
    with pytest.raises(HTTPException) as stale:
        routes[endpoint](7, v="1")
    assert stale.value.status_code == 409
    assert "no-store" in stale.value.headers["Cache-Control"]
    for legacy in ("", "available", "snapshots/old.webp"):
        assert routes[endpoint](7, v=legacy).headers["cache-control"] == "private, no-cache"
    assert routes[endpoint](7, v="2").status_code == 200


@pytest.mark.parametrize("endpoint", ["event_snapshot", "event_thumbnail"])
def test_unchanged_image_revalidates_without_resending_it(tmp_path, endpoint):
    routes, event = _routes(tmp_path)
    first = routes[endpoint](7, v="1")
    etag = first.headers["etag"]
    def asking(value):
        return SimpleNamespace(headers={"if-none-match": value})
    again = routes[endpoint](7, v="1", request=asking(etag))
    assert again.status_code == 304 and again.body == b""
    assert again.headers["cache-control"] == "private, no-cache" and again.headers["etag"] == etag
    assert routes[endpoint](7, v="1", request=asking(f'W/{etag}, "other"')).status_code == 304
    assert routes[endpoint](7, v="1", request=asking('"other"')).status_code == 200
    assert routes[endpoint](7, v="1", request=asking("")).status_code == 200
    # A changed revision is still refused, whatever the browser holds.
    event["evidence_revision"] = 2
    with pytest.raises(HTTPException) as stale:
        routes[endpoint](7, v="1", request=asking(etag))
    assert stale.value.status_code == 409


def test_a_different_thumbnail_size_is_a_different_image(tmp_path):
    routes, _event = _routes(tmp_path)
    small = routes["event_thumbnail"](7, v="1", width=160)
    large = routes["event_thumbnail"](7, v="1", width=240)
    assert small.headers["etag"] != large.headers["etag"]
    asked = SimpleNamespace(headers={"if-none-match": small.headers["etag"]})
    assert routes["event_thumbnail"](7, v="1", width=240, request=asked).status_code == 200


def test_a_download_is_always_sent(tmp_path):
    routes, _event = _routes(tmp_path)
    etag = routes["event_snapshot"](7, v="1").headers["etag"]
    asked = SimpleNamespace(headers={"if-none-match": etag})
    assert routes["event_snapshot"](7, download=True, v="1", request=asked).status_code == 200


def _camera_like(width, height):
    rows, columns = np.mgrid[0:height, 0:width]
    frame = np.stack([columns * 255 // width, rows * 255 // height, (rows + columns) * 255 // (width + height)], -1).astype(np.uint8)
    cv2.rectangle(frame, (width // 5, height // 4), (width // 2, height // 2), (20, 200, 240), -1)
    cv2.circle(frame, (3 * width // 4, height // 3), height // 6, (250, 250, 250), 6)
    return frame


def test_plain_thumbnails_decode_large_jpegs_at_reduced_size(tmp_path):
    from survng.app import appearance_routes as routes_module
    path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(path), _camera_like(3840, 2160), [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert routes_module._jpeg_dimensions(path) == (3840, 2160)
    assert routes_module._jpeg_dimensions(tmp_path / "missing.jpg") is None
    flags = []
    real = cv2.imread
    def reading(name, flag=cv2.IMREAD_COLOR):
        flags.append(flag)
        return real(name, flag)
    with patch.object(routes_module.cv2, "imread", side_effect=reading):
        small = routes_module._read_frame_for_width(path, 320)
        medium = routes_module._read_frame_for_width(path, 640)
        wide = routes_module._read_frame_for_width(path, 2560)
    # 2160 // 4 = 540 covers 320; 2160 // 2 = 1080 covers 640; nothing reduced covers 2560.
    assert flags == [cv2.IMREAD_REDUCED_COLOR_4, cv2.IMREAD_REDUCED_COLOR_2, cv2.IMREAD_COLOR]
    assert small.shape[1] >= 320 and medium.shape[1] >= 640 and wide.shape[:2] == (2160, 3840)
    # The reduced decode shows the same picture as a full decode scaled down.
    full = cv2.imdecode(np.frombuffer(routes_module._jpeg_thumbnail(real(str(path)), 320, 90), np.uint8), cv2.IMREAD_COLOR)
    fast = cv2.imdecode(np.frombuffer(routes_module._jpeg_thumbnail(small, 320, 90), np.uint8), cv2.IMREAD_COLOR)
    assert fast.shape == full.shape == (180, 320, 3)
    assert float(np.abs(fast.astype(int) - full.astype(int)).mean()) < 3.0


def test_small_and_non_jpeg_snapshots_decode_normally(tmp_path):
    from survng.app import appearance_routes as routes_module
    small = tmp_path / "small.jpg"
    assert cv2.imwrite(str(small), _camera_like(640, 360))
    assert routes_module._read_frame_for_width(small, 320).shape[:2] == (360, 640)
    png = tmp_path / "frame.png"
    assert cv2.imwrite(str(png), _camera_like(1920, 1080))
    assert routes_module._jpeg_dimensions(png) is None
    assert routes_module._read_frame_for_width(png, 320).shape[:2] == (1080, 1920)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"\xff\xd8\xff")
    assert routes_module._jpeg_dimensions(broken) is None
    assert routes_module._read_frame_for_width(broken, 320) is None


def test_thumbnail_of_a_large_snapshot_has_the_asked_width(tmp_path):
    routes, event = _routes(tmp_path)
    path = tmp_path / "snapshots" / "large.jpg"
    assert cv2.imwrite(str(path), _camera_like(2560, 1440))
    event["snapshot_path"] = str(path)
    response = routes["event_thumbnail"](7, v="1", width=320)
    assert cv2.imread(str(response.path)).shape[:2] == (180, 320)


def test_object_crop_cache_changes_when_annotations_change_on_same_image(tmp_path):
    routes, event = _routes(tmp_path)
    event["objects_json"] = json.dumps([{
        "label": "person", "box": {"x1": 10, "y1": 20, "x2": 70, "y2": 100},
    }])
    first = routes["event_thumbnail"](7, object_focus=True, v="1")
    event["objects_json"] = json.dumps([{
        "label": "person", "box": {"x1": 170, "y1": 20, "x2": 230, "y2": 100},
    }])
    event["evidence_revision"] = 2
    second = routes["event_thumbnail"](7, object_focus=True, v="2")
    assert first.path != second.path
    assert cv2.imread(str(first.path)).mean() < cv2.imread(str(second.path)).mean()

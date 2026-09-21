"""Public annotations and current-event raster routes share one revision."""
import json
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from survng.app.appearance_routes import AppearanceRouteDependencies, create_appearance_router
from survng.app.image_cache import LocalImageCache
from survng.app.incident_payload import IncidentPayloadBuilder
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
    notification = IncidentPayloadBuilder._incident_payload({
        "events": {7: raw}, "camera_id": "gate", "camera_name": "Gate",
    }, "complete")
    assert notification["snapshot_url"].endswith("/7/snapshot.jpg?v=12")
    assert notification["evidence_revision"] == 12
    # Provisional detections remain visible in the incident UI, but notification
    # payloads wait for refined evidence before claiming an object.
    assert notification["objects"] == []
    assert notification["has_objects"] is False


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

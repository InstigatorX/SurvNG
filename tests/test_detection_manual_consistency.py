from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from survng.app.config import AppConfig
from survng.app.detection_routes import DetectionRouteDependencies, create_detection_router
from survng.app.event_store import EventStore
from survng.app.manager import AppManager
from survng.app.semantic_search import (
    SemanticEvidence,
    SemanticIndex,
    SemanticModelIdentity,
    SemanticSearchService,
)


@pytest.fixture
def manual_detection(tmp_path: Path):
    store = EventStore(tmp_path)
    snapshot = tmp_path / "original.jpg"
    assert cv2.imwrite(str(snapshot), np.zeros((48, 64, 3), dtype=np.uint8))
    objects = [{
        "label": "person", "confidence": 0.9, "track_id": "person-1",
        "box": {"x1": 2, "y1": 4, "x2": 20, "y2": 40},
    }]
    event = store.add_event(
        camera_id="gate", kind="motion", snapshot_path=str(snapshot),
        objects_json=json.dumps(objects),
    )
    manager = SimpleNamespace(
        config=AppConfig(), storage_dir=tmp_path, events=store,
        detector=SimpleNamespace(detect=Mock(return_value=objects)),
        detector_status=lambda: {}, publish_event=Mock(),
    )
    deps = DetectionRouteDependencies(
        get_manager=lambda: manager, get_config=lambda: manager.config,
        manager_lock=threading.RLock(), get_comparison_limiter=Mock(),
        ensure_event_clip=Mock(), dependency_status=Mock(),
        comparison_runner=Mock(), sample_video_frames=Mock(),
    )
    endpoint = create_detection_router(deps).handlers["detect_event_snapshot"]
    return manager, event, endpoint


@pytest.mark.parametrize("replacement", ["refinement", "tracking_cover"])
def test_manual_detection_rejects_a_replaced_snapshot(manual_detection, tmp_path, replacement):
    manager, event, endpoint = manual_detection
    promoted_snapshot = tmp_path / "promoted.jpg"
    assert cv2.imwrite(str(promoted_snapshot), np.full((96, 128, 3), 255, dtype=np.uint8))
    current_box = {"x1": 80, "y1": 20, "x2": 120, "y2": 90}
    expected_event = {}

    def detect(frame, confidence_threshold=None):
        assert frame.shape[:2] == (48, 64)
        assert frame.mean() == 0
        if replacement == "refinement":
            promoted = manager.events.refine_event_evidence(
                event["id"], snapshot_path=str(promoted_snapshot), recording_path="",
                objects_json=json.dumps([{"label": "car", "confidence": 0.9, "box": current_box}]),
            )
        else:
            promoted = manager.events.promote_tracking_cover(
                event["id"], snapshot_path=str(promoted_snapshot), captured_at=1_700_000_000,
                frame_width=128, frame_height=96,
                tracked_objects=[{"track_id": "person-1", "box": current_box, "confidence": 0.9}],
                cover_metrics={},
            )
        assert promoted is not None
        expected_event.update(promoted)
        return [{"label": "person", "confidence": 0.9, "box": {"x1": 1, "y1": 2, "x2": 10, "y2": 40}}]

    manager.detector.detect = detect
    with pytest.raises(HTTPException) as error:
        endpoint(event["id"])

    assert error.value.status_code == 409
    assert "refresh and try again" in error.value.detail
    assert manager.events.get(event["id"]) == expected_event
    assert expected_event["snapshot_path"] == "promoted.jpg"
    assert json.loads(expected_event["objects_json"])[0]["box"] == current_box
    manager.publish_event.assert_not_called()


@pytest.mark.parametrize("objects", [[], [{
    "label": "car", "confidence": 0.9, "incident_eligible": False,
    "box": {"x1": 30, "y1": 10, "x2": 60, "y2": 40},
}]])
def test_manual_correction_invalidates_search_and_updates_clients(manual_detection, tmp_path, objects):
    manager, event, endpoint = manual_detection
    index = SemanticIndex(manager.events.db_path)
    identity = SemanticModelIdentity("test", "model", "prep", 3)
    index.upsert([
        SemanticEvidence(event["id"], "gate", event["created_at"], "object_crop", "person:0", event["snapshot_path"], "person"),
    ], [[1, 0, 0]], identity)
    search = SemanticSearchService(manager.config.semantic_search, index, tmp_path, {})
    search.encoder = SimpleNamespace(identity=identity)
    manager.semantic_search = search
    manager.state_events = Mock()
    manager.mqtt = Mock()
    manager.publish_event = lambda kind, payload: AppManager.publish_event(manager, kind, payload)
    manager.detector.detect.return_value = objects

    result = endpoint(event["id"])

    assert result["persisted"] is True
    assert result["object_count"] == 0
    persisted = manager.events.get(event["id"])
    assert json.loads(persisted["objects_json"]) == result["objects"]
    assert index.search([1, 0, 0], identity) == []
    manager.state_events.publish.assert_called_once_with(
        "incident", {"event_id": event["id"], "camera_id": "gate", "updated": True},
    )
    manager.mqtt.publish.assert_not_called()
    manager.mqtt.track_incident.assert_not_called()
    if objects:
        assert search._queue.get_nowait()[2] == persisted
    else:
        assert search._queue.empty()


def test_manual_positive_detection_keeps_object_notification(manual_detection):
    manager, event, endpoint = manual_detection
    result = endpoint(event["id"])
    assert result["persisted"] is True
    assert result["object_count"] == 1
    manager.publish_event.assert_called_once()
    kind, payload = manager.publish_event.call_args.args
    assert kind == "object"
    assert payload["source"] == "manual_openvino"
    assert payload["event_id"] == event["id"]

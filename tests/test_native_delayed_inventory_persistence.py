"""Incident inventory must survive cover promotion and late inventory writes.

Scene activity opens immediately. Registry, activity, incident inventory,
EventStore transactions, and cover promotion are real.
"""
from copy import deepcopy
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from survng.app.config import AppConfig, CameraConfig, DetectorConfig
from survng.app.event_store import EventStore
from survng.app.image_storage import DurableImageWriter
from survng.app.live_detections import DetectionSnapshot
from survng.app.media_storage import MediaStorageRegistry
from survng.app.native_activity import NativeActivity
from survng.app.native_evidence import Candidate, NativeEvidenceService, event_tracking


def detection(label, native_id, left, width=15):
    return {
        "label": label,
        "confidence": 0.9,
        "box": {"x1": left, "y1": 20, "x2": left + width, "y2": 70},
        "native_track_id": native_id,
        "detection_provenance": "native_fresh_detection",
    }


def observe(activity, sequence, objects):
    now = 100 + sequence / 5
    activity.consume(
        DetectionSnapshot(
            sequence / 5,
            sequence,
            200,
            100,
            tuple(objects),
            "session",
            "native_fresh_detection",
            now,
        ),
        now=now,
        epoch=1000 + sequence / 5,
    )


def subject_objects(event):
    objects, _ = event_tracking(event)
    return [item for item in objects if item.get("label")]


def test_scene_inventory_survives_real_store_cover_and_completion(tmp_path):
    camera = CameraConfig(
        id="front",
        name="Front",
        stream_url="rtsp://unused.invalid",
        native_same_field_of_view=True,
    )
    detector = DetectorConfig(
        enabled=True,
        confidence_threshold=0.5,
        event_confirmation_frames=1,
        native={
            "verification_enabled": True,
            "tracking_classes": ["dog", "person", "car", "package"],
            "activity_timeout_seconds": 5,
            "stationary": {"enabled": False},
        },
    )
    config = AppConfig(storage_dir=str(tmp_path), cameras=[camera], detector=detector)
    media = MediaStorageRegistry(tmp_path, config.media_storage)
    events = EventStore(tmp_path, media_storage=media)
    pixels = np.random.default_rng(17).integers(0, 256, (100, 200, 3), dtype=np.uint8)
    preview = tmp_path / "snapshots" / "preview.jpg"
    preview.parent.mkdir(exist_ok=True)
    assert cv2.imwrite(str(preview), pixels)
    published = []
    activity = NativeActivity(
        camera,
        detector,
        events,
        lambda kind, payload: published.append((kind, deepcopy(payload))),
        Mock(return_value=str(preview)),
    )
    activity.offer_evidence = Mock()

    # Multiple distinct subjects share one scene incident.
    for sequence in (1, 2, 3):
        objects = [
            detection("dog", 9, 20 + sequence),
            detection("person", 10, 60),
            detection("person", 12, 95),
            detection("car", 11, 130, width=55),
        ]
        if sequence == 2:
            objects.append(detection("package", 13, 2, width=12))
        observe(activity, sequence, objects)

    assert activity.event_id is not None
    event_id = activity.event_id
    nomination_epoch = 1000 + 1 / 5
    nominated = next(
        item
        for item in subject_objects(events.get(event_id))
        if item.get("native_track_id") == 9
    )

    for sequence in range(4, 70):
        observe(activity, sequence, [])
    assert activity.event_id is None
    assert not activity.registry.tracks

    saved = events.get(event_id)
    subjects = subject_objects(saved)
    expected_ids = {9, 10, 11, 12, 13}
    assert {item["native_track_id"] for item in subjects} == expected_ids
    assert len(subjects) == 5
    assert sum(item["label"] == "person" for item in subjects) == 2
    _, tracking = event_tracking(saved)
    assert tracking["state"] == "complete"
    assert tracking.get("implementation") == "native_observations"
    incident = events.incident_for_event(event_id)
    assert incident is not None
    assert incident["state"] == "complete"
    assert incident["observation_count"] >= 1
    assert {
        item["native_track_id"]
        for item in incident["participants"]
        if item.get("native_track_id") is not None
    } == expected_ids

    # Later high-res cover promotion annotates the verified primary without
    # shrinking the durable multi-object inventory.
    evidence = NativeEvidenceService(
        config,
        events,
        Mock(),
        DurableImageWriter(config.image_storage),
        media,
    )
    larger = cv2.resize(pixels, (800, 400))
    promoted_box = {"x1": 98, "y1": 84, "x2": 168, "y2": 280}
    promoted = deepcopy(nominated)
    promoted.update(
        box=promoted_box,
        detection_frame_width=800,
        detection_frame_height=400,
        frame_source="recorded_main",
        frame_captured_at_epoch=1000.8,
        native_cover_verified=True,
        box_provenance="detected_in_main",
    )
    evidence.main_frames.verify_candidate = Mock(
        return_value={
            "status": "confirmed",
            "cover": (larger, promoted, 1000.8),
        }
    )
    result = evidence.process(
        event_id,
        [Candidate(nomination_epoch, pixels, [nominated], 5)],
    )
    assert result["status"] == "promoted"
    saved = events.get(event_id)
    selected_path = saved["snapshot_path"]
    assert selected_path != preview.relative_to(tmp_path).as_posix()
    assert (tmp_path / selected_path).exists()
    assert cv2.imread(str(tmp_path / selected_path)).shape[:2] == (400, 800)

    # A late inventory write with live-raster geometry must retain the selected
    # main-raster annotation on the primary subject.
    post_cover = subject_objects(saved)
    live_inventory = [
        {
            **item,
            "snapshot_visible": False,
            "box": {"x1": 10, "y1": 10, "x2": 20, "y2": 40},
            "detection_frame_width": 200,
            "detection_frame_height": 100,
        }
        for item in post_cover
    ]
    events.update_native_incident_state(
        event_id,
        {
            **tracking,
            "state": "complete",
            "implementation": "native_observations",
            "incident_id": incident["id"],
        },
        live_inventory,
    )
    saved = events.get(event_id)
    subjects = subject_objects(saved)
    assert saved["snapshot_path"] == selected_path
    assert {item["native_track_id"] for item in subjects} == expected_ids
    assert len(subjects) == 5
    primary = next(item for item in subjects if item["native_track_id"] == 9)
    assert primary["box"] == promoted_box
    assert primary["detection_frame_width"] == 800
    assert primary["detection_frame_height"] == 400
    assert primary["frame_captured_at_epoch"] == 1000.8
    assert primary["snapshot_visible"] is True
    assert all(
        item["snapshot_visible"] is False
        for item in subjects
        if item["native_track_id"] != 9
    )

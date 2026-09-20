"""Incident inventory must survive delayed admission and selected-cover writes.

The observation stream and verifier result are deterministic. Registry, activity,
incident inventory, EventStore transactions, and cover promotion are real.
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
        "confidence": .9,
        "box": {"x1": left, "y1": 20, "x2": left + width, "y2": 70},
        "native_track_id": native_id,
        "detection_provenance": "native_fresh_detection",
    }


def observe(activity, sequence, objects):
    now = 100 + sequence / 5
    activity.consume(
        DetectionSnapshot(
            sequence / 5, sequence, 200, 100, tuple(objects),
            "session", "native_fresh_detection", now,
        ),
        now=now,
        epoch=1000 + sequence / 5,
    )


def subject_objects(event):
    objects, _ = event_tracking(event)
    return [item for item in objects if item.get("label")]


@pytest.mark.parametrize("late_context", [False, True])
def test_delayed_inventory_survives_real_store_cover_and_completion(tmp_path, late_context):
    camera = CameraConfig(
        id="front", name="Front", stream_url="rtsp://unused.invalid",
        native_same_field_of_view=True,
    )
    detector = DetectorConfig(
        enabled=True,
        confidence_threshold=.5,
        event_confirmation_frames=2,
        native={
            "verification_enabled": True,
            "tracking_classes": ["dog"],
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
    main = cv2.resize(pixels, (400, 200))
    first_cover = preview.with_name("admission.jpg")
    assert cv2.imwrite(str(first_cover), main)
    published = []
    activity = NativeActivity(
        camera, detector, events,
        lambda kind, payload: published.append((kind, deepcopy(payload))),
        Mock(return_value=str(preview)),
    )
    ready = {}
    activity.admission = Mock()
    activity.admission.poll.side_effect = lambda token: ready.pop(token, None)
    activity.nominate = Mock()
    activity.verified_snapshot = Mock(return_value=str(first_cover))
    activity.offer_evidence = Mock()

    # Two people of the same class must remain two objects, not a set of labels.
    for sequence in (1, 2, 3):
        objects = [
            detection("dog", 9, 20 + sequence),
            detection("person", 10, 60),
            detection("person", 12, 95),
            detection("car", 11, 130, width=55),
        ]
        if sequence == 2:
            objects.append(detection("bird", 99, 1, width=10))
        observe(activity, sequence, objects)
    assert len(activity._verification_pending) == 1
    token, pending = next(iter(activity._verification_pending.items()))
    nominated = deepcopy(pending["track"])
    nomination_epoch = pending["epoch"]

    # A credible secondary object appears after nomination but within the
    # activity window. It must survive even if all live identities later expire.
    for sequence in (4, 5):
        observe(activity, sequence, [detection("package", 13, 2, width=12)])
    for sequence in range(6, 70):
        observe(activity, sequence, [])
    assert not activity.registry.tracks
    assert not published

    # This cat belongs to a different scene, well beyond the old activity
    # window. Verification completion time must not define incident contents.
    for sequence in range(70, 81):
        observe(activity, sequence, [detection("cat", 20, 160)] if late_context else [])
    assert not published
    verified = deepcopy(nominated)
    main_box = {"x1": 48, "y1": 43, "x2": 81, "y2": 143}
    verified.update(
        box=main_box,
        detection_frame_width=400,
        detection_frame_height=200,
        frame_source="recorded_main",
        frame_captured_at_epoch=1000.5,
        native_cover_verified=True,
        box_provenance="detected_in_main",
    )
    ready[token] = {"status": "confirmed", "votes": ["confirmed"],
                    "cover": (main, verified, 1000.5)}
    activity.tick(now=116.2)

    created = [payload for kind, payload in published
               if kind == "incident" and not payload.get("updated")]
    assert len(created) == 1
    event_id = created[0]["event_id"]
    saved = events.get(event_id)
    objects, tracking = event_tracking(saved)
    subjects = subject_objects(saved)
    expected_ids = {9, 10, 11, 12, 13}
    assert {item["native_track_id"] for item in subjects} == expected_ids
    assert len(subjects) == 5
    assert sum(item["label"] == "person" for item in subjects) == 2
    assert tracking["state"] == "complete"
    assert tracking.get("implementation") == "native_observations"
    assert "tracks" not in tracking or tracking.get("tracks") in (None, [])
    incident = events.incident_for_event(event_id)
    assert incident is not None
    assert incident["state"] == "complete"
    assert incident["observation_count"] >= 1
    assert {item["native_track_id"] for item in incident["participants"] if item.get("native_track_id") is not None} == expected_ids
    assert saved["snapshot_path"] == first_cover.relative_to(tmp_path).as_posix()
    primary = next(item for item in subjects if item["native_track_id"] == 9)
    assert primary["box"] == main_box
    assert primary["detection_frame_width"] == 400
    assert primary["frame_captured_at_epoch"] == 1000.5
    assert primary["snapshot_visible"] is True
    assert all(item["snapshot_visible"] is False
               for item in subjects if item["native_track_id"] != 9)
    # The current registry must never supply coordinates for the old nominee's
    # DetectionSnapshot when a delayed result is activated.
    activity.offer_evidence.assert_not_called()

    # A later high-res replacement can annotate only the verified primary,
    # but it must not reduce the durable five-object incident inventory.
    evidence = NativeEvidenceService(
        config, events, Mock(), DurableImageWriter(config.image_storage), media,
    )
    larger = cv2.resize(pixels, (800, 400))
    promoted_box = {"x1": 98, "y1": 84, "x2": 168, "y2": 280}
    promoted = deepcopy(verified)
    promoted["box"] = promoted_box
    evidence.main_frames.verify_candidate = Mock(return_value={
        "status": "confirmed", "cover": (larger, promoted, 1000.8),
    })
    result = evidence.process(
        event_id, [Candidate(nomination_epoch, pixels, [nominated], 5)],
    )
    assert result["status"] == "promoted"
    saved = events.get(event_id)
    selected_path = saved["snapshot_path"]
    assert selected_path != first_cover.relative_to(tmp_path).as_posix()
    assert (tmp_path / selected_path).exists()
    assert cv2.imread(str(tmp_path / selected_path)).shape[:2] == (400, 800)

    # Simulate a late inventory write with live-raster geometry. The common
    # presentation merge must retain the selected main-raster annotation.
    post_cover = subject_objects(saved)
    live_inventory = [
        {
            **item,
            "snapshot_visible": False,
            "box": {
                "x1": 10,
                "y1": 10,
                "x2": 20,
                "y2": 40,
            },
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
    assert all(item["snapshot_visible"] is False
               for item in subjects if item["native_track_id"] != 9)

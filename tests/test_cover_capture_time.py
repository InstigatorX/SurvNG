"""An event's cover frame time is reported separately from its trigger time."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from survng.app.config import ImageStorageConfig
from survng.app.events import EventStore
from survng.app.image_storage import DurableImageWriter
from survng.app.incident_presenter import (
    _event_row,
    _incident_list_payload,
    _incident_row,
    snapshot_captured_epoch,
)

TRIGGER = datetime(2026, 11, 11, 1, 21, 50, 364664, tzinfo=timezone.utc)
CAPTURED = datetime(2026, 11, 11, 1, 21, 58, 380000, tzinfo=timezone.utc)


def car(**extra) -> dict:
    return {"label": "car", "confidence": 0.98, "box": {"x1": 10, "y1": 10, "x2": 60, "y2": 50}, **extra}


def row(objects, snapshot_path="snapshots/gate/cover.webp") -> dict:
    return {"id": 7, "camera_id": "gate", "kind": "motion", "created_at": TRIGGER.isoformat(),
            "snapshot_path": snapshot_path, "recording_path": "", "objects_json": json.dumps(objects)}


def captured(event: dict) -> datetime:
    return datetime.fromisoformat(event["snapshot_captured_at"])


def test_cover_time_is_the_trigger_plus_the_selected_sample_offset():
    event = _event_row(row([car(temporal_sample_offset_seconds=8.015336)]))

    assert abs((captured(event) - CAPTURED).total_seconds()) < 1e-3
    # The trigger time is still reported, unchanged.
    assert event["created_at"] == TRIGGER.isoformat()


def test_explicit_frame_times_win_over_the_offset():
    promoted = car(temporal_sample_offset_seconds=0.5,
                   frame_captured_at_epoch=TRIGGER.timestamp() + 2,
                   snapshot_captured_at=CAPTURED.isoformat())
    assert captured(_event_row(row([promoted]))) == CAPTURED

    recorded = car(temporal_sample_offset_seconds=0.5, frame_captured_at_epoch=CAPTURED.timestamp())
    assert captured(_event_row(row([recorded]))) == CAPTURED


def test_objects_not_shown_on_the_cover_do_not_set_its_time():
    elsewhere = car(snapshot_visible=False, temporal_sample_offset_seconds=-6.0,
                    frame_captured_at_epoch=TRIGGER.timestamp() - 6)
    on_cover = car(temporal_sample_offset_seconds=8.015336)

    event = _event_row(row([elsewhere, on_cover]))

    assert abs((captured(event) - CAPTURED).total_seconds()) < 1e-3


def test_cover_without_objects_uses_the_snapshot_writer_stamp(tmp_path):
    # Every snapshot writer names the file after the frame's capture time.
    from survng.app.camera_media import CameraMediaService
    from survng.app.config import CameraConfig

    media = CameraMediaService(
        camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        storage_dir=tmp_path, image_writer=DurableImageWriter(ImageStorageConfig()),
        motion_detector=None, frame_provider=lambda _source: None,
        rejected_sample_rate=lambda: 0.0, stop_requested=lambda: False,
    )
    path = media.write_snapshot(np.zeros((20, 30, 3), np.uint8), CAPTURED)
    assert path

    status_only = [{"status": "motion_qualification", "motion_qualification": {}}]
    event = _event_row(row(status_only, snapshot_path=path))

    assert captured(event) == CAPTURED


def test_unknown_cover_time_is_omitted_so_clients_fall_back_to_the_trigger():
    assert "snapshot_captured_at" not in _event_row(row([car()], snapshot_path="snapshots/gate/first.webp"))
    assert "snapshot_captured_at" not in _event_row(row([car(temporal_sample_offset_seconds=8.0)], snapshot_path=""))
    assert snapshot_captured_epoch("not a time", [car(temporal_sample_offset_seconds=8.0)], "") is None
    assert snapshot_captured_epoch(TRIGGER.isoformat(), [car(temporal_sample_offset_seconds=float("nan"))],
                                   "snapshots/gate/20261340-999999-000000-1.webp") is None


def test_incident_and_list_payloads_carry_the_cover_time():
    event = _event_row(row([car(temporal_sample_offset_seconds=8.015336)]))

    incident = _incident_row("gate", [event])
    compact = _incident_list_payload(incident)

    assert incident["snapshot_captured_at"] == event["snapshot_captured_at"]
    assert compact["snapshot_captured_at"] == event["snapshot_captured_at"]
    assert compact["events"][0]["snapshot_captured_at"] == event["snapshot_captured_at"]
    assert compact["events"][0]["created_at"] == TRIGGER.isoformat()


def test_incident_cards_carry_the_cover_time(tmp_path):
    store = EventStore(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    cover = snapshots / "20261111-012158-380000-000000001.webp"
    cover.write_bytes(b"cover")
    store.add_event("gate", "motion", created_at=TRIGGER.isoformat(), snapshot_path=str(cover),
                    objects_json=json.dumps([car(temporal_sample_offset_seconds=8.015336)]))

    card = store.list_scene_incident_cards()[0]

    assert abs((datetime.fromisoformat(card["events"][0]["snapshot_captured_at"]) - CAPTURED).total_seconds()) < 1e-3
    assert card["events"][0]["created_at"] == TRIGGER.isoformat()
    assert "snapshot_captured_at" in card

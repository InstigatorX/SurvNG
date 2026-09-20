"""Cover promotion may enrich multi-object annotations without a higher beauty score."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from survng.app.event_store import EventStore
from survng.app.native_event_projection import (
    cover_visible_object_count,
    should_adopt_native_cover,
)


def test_should_adopt_native_cover_enriches_when_more_objects_visible():
    existing = [
        {
            "label": "person",
            "track_id": 1,
            "snapshot_visible": True,
            "native_cover_score": 10.0,
            "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
        },
        {
            "label": "car",
            "track_id": 2,
            "snapshot_visible": False,
            "box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
        },
    ]
    enriched = [
        {
            "label": "person",
            "track_id": 1,
            "snapshot_visible": True,
            "native_cover_score": 9.0,
            "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
        },
        {
            "label": "car",
            "track_id": 2,
            "snapshot_visible": True,
            "native_cover_score": 9.0,
            "box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
        },
    ]
    assert cover_visible_object_count(existing) == 1
    assert cover_visible_object_count(enriched) == 2
    assert should_adopt_native_cover(existing, enriched, 9.0, 10.0) == "cover_enriched"
    assert should_adopt_native_cover(existing, existing, 9.0, 10.0) is None
    assert should_adopt_native_cover(existing, existing, 10.1, 10.0) == "promoted"


def test_event_store_enriches_multi_object_cover_without_higher_score(tmp_path):
    store = EventStore(Path(tmp_path))
    first = tmp_path / "first.webp"
    second = tmp_path / "second.webp"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    event = store.add_event(
        camera_id="test",
        kind="motion",
        created_at=datetime.now(timezone.utc).isoformat(),
        snapshot_path=str(first),
        objects_json=json.dumps(
            [
                {
                    "label": "person",
                    "track_id": 1,
                    "snapshot_visible": True,
                    "native_cover_score": 10.0,
                    "box": {"x1": 10, "y1": 20, "x2": 40, "y2": 80},
                    "detection_frame_width": 1280,
                    "detection_frame_height": 720,
                },
                {
                    "label": "car",
                    "track_id": 2,
                    "snapshot_visible": False,
                    "box": {"x1": 100, "y1": 100, "x2": 200, "y2": 180},
                    "detection_frame_width": 640,
                    "detection_frame_height": 360,
                },
                {"status": "object_tracking", "object_tracking": {"state": "complete"}},
            ]
        ),
    )
    diagnostics = {}
    enriched = [
        {
            "label": "person",
            "track_id": 1,
            "snapshot_visible": True,
            "native_cover_score": 8.5,
            "native_cover_verified": True,
            "box_provenance": "detected_in_main",
            "box": {"x1": 12, "y1": 22, "x2": 42, "y2": 82},
            "detection_frame_width": 1280,
            "detection_frame_height": 720,
        },
        {
            "label": "car",
            "track_id": 2,
            "snapshot_visible": True,
            "native_cover_score": 8.5,
            "native_cover_verified": True,
            "box_provenance": "detected_in_main",
            "box": {"x1": 400, "y1": 300, "x2": 900, "y2": 600},
            "detection_frame_width": 1280,
            "detection_frame_height": 720,
        },
    ]
    updated = store.promote_native_evidence(
        int(event["id"]),
        str(second),
        enriched,
        [(str(second), enriched)],
        8.5,
        diagnostics=diagnostics,
    )
    assert updated is not None
    assert diagnostics["reason"] == "cover_enriched"
    objects = json.loads(updated["objects_json"])
    visible = [item for item in objects if item.get("label") and item.get("snapshot_visible") is True]
    assert {item["label"] for item in visible} == {"person", "car"}
    car = next(item for item in visible if item["label"] == "car")
    assert car["detection_frame_width"] == 1280
    assert car["box"]["x1"] == 400
    # Same object count at a lower score still retains the better cover.
    diagnostics = {}
    retained = store.promote_native_evidence(
        int(event["id"]),
        str(first),
        enriched,
        [(str(first), enriched)],
        7.0,
        diagnostics=diagnostics,
    )
    assert retained is None
    assert diagnostics["reason"] == "better_cover_retained"

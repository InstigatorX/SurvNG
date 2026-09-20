"""Cover witnesses annotate concurrent peers without softening zone admission."""
from __future__ import annotations

from unittest.mock import Mock

import numpy as np

from survng.app.config import AppConfig, CameraConfig
from survng.app.native_evidence import (
    candidate_score,
    concurrent_scene_objects,
    enrich_candidate_objects,
)
from survng.app.native_event_projection import (
    merge_cover_objects,
    merge_inventory_objects,
    should_adopt_native_cover,
)
from survng.app.native_main_frame import NativeMainFrameVerifier


def _verifier():
    config = AppConfig()
    config.cameras = [
        CameraConfig(id="test", name="Test", stream_url="rtsp://unused.invalid"),
    ]
    return NativeMainFrameVerifier(lambda: config, Mock(), {})


def test_match_scene_promotes_outside_zone_sibling_as_witness():
    verifier = _verifier()
    main = np.zeros((720, 1280, 3), dtype=np.uint8)
    primary = {
        "label": "car",
        "track_id": 2140,
        "episode_identity": "car:2140",
        "confidence": 0.92,
        "box": {"x1": 900.0, "y1": 100.0, "x2": 1100.0, "y2": 220.0},
        "incident_eligible": True,
        "zones": ["Road"],
    }
    witness = {
        "label": "car",
        "track_id": 2138,
        "episode_identity": "car:2138",
        "confidence": 0.88,
        "box": {"x1": 500.0, "y1": 100.0, "x2": 700.0, "y2": 220.0},
        "incident_eligible": False,
        "zones": [],
        "zone_admission_reason": "outside_incident_zone",
    }
    detections = [
        {"label": "car", "confidence": 0.94, "box": {"x1": 905, "y1": 105, "x2": 1095, "y2": 215}},
        {"label": "car", "confidence": 0.9, "box": {"x1": 505, "y1": 105, "x2": 695, "y2": 215}},
    ]
    scene = verifier.match_scene_detections(
        "test",
        [primary, witness],
        main,
        10.0,
        primary=dict(primary),
        detections=detections,
    )
    by_id = {item["episode_identity"]: item for item in scene}
    assert by_id["car:2140"]["snapshot_visible"] is True
    assert by_id["car:2140"]["incident_eligible"] is True
    assert by_id["car:2140"]["cover_role"] == "primary"
    assert by_id["car:2138"]["snapshot_visible"] is True
    assert by_id["car:2138"]["incident_eligible"] is False
    assert by_id["car:2138"]["cover_role"] == "witness"
    assert by_id["car:2138"]["zone_admission_reason"] == "outside_incident_zone"
    assert by_id["car:2138"]["box_provenance"] == "detected_in_main"


def test_match_scene_exclusive_extent_keeps_one_od_per_identity():
    verifier = _verifier()
    main = np.zeros((720, 1280, 3), dtype=np.uint8)
    primary = {
        "label": "car",
        "track_id": 1,
        "episode_identity": "car:1",
        "confidence": 0.9,
        "box": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 200.0},
        "incident_eligible": True,
    }
    witness = {
        "label": "car",
        "track_id": 2,
        "episode_identity": "car:2",
        "confidence": 0.85,
        "box": {"x1": 105.0, "y1": 105.0, "x2": 195.0, "y2": 195.0},
        "incident_eligible": False,
    }
    detections = [
        {"label": "car", "confidence": 0.93, "box": {"x1": 102, "y1": 102, "x2": 198, "y2": 198}},
    ]
    scene = verifier.match_scene_detections(
        "test",
        [primary, witness],
        main,
        10.0,
        primary=dict(primary),
        detections=detections,
    )
    # Primary extent-claims the only OD; witness must not also claim it.
    assert scene[0]["episode_identity"] == "car:1"
    assert scene[0]["box"]["x1"] == 102
    assert len(scene) == 1


def test_enrich_candidate_objects_adds_concurrent_witness_skips_late_identity():
    tracking = {
        "frame_width": 640,
        "frame_height": 360,
        "tracks": [
            {
                "label": "car",
                "track_id": 2140,
                "episode_identity": "car:2140",
                "confidence": 0.9,
                "incident_eligible": True,
                "zones": ["Road"],
                "box_history": [[100.0, 400, 80, 520, 160]],
            },
            {
                "label": "car",
                "track_id": 2138,
                "episode_identity": "car:2138",
                "confidence": 0.85,
                "incident_eligible": False,
                "zones": [],
                "zone_admission_reason": "outside_incident_zone",
                "box_history": [[100.0, 200, 80, 320, 160]],
            },
            {
                "label": "car",
                "track_id": 2143,
                "episode_identity": "car:2143",
                "confidence": 0.8,
                "incident_eligible": True,
                "zones": ["Road"],
                # First seen well after cover epoch.
                "box_history": [[109.0, 50, 100, 120, 150]],
            },
        ],
    }
    nominated = [
        {
            "label": "car",
            "track_id": 2140,
            "episode_identity": "car:2140",
            "confidence": 0.91,
            "incident_eligible": True,
            "box": {"x1": 401, "y1": 81, "x2": 519, "y2": 159},
        }
    ]
    merged = enrich_candidate_objects(nominated, tracking, 100.0, live_size=(640, 360))
    by_id = {item["episode_identity"]: item for item in merged}
    assert set(by_id) == {"car:2140", "car:2138"}
    assert by_id["car:2138"]["incident_eligible"] is False
    assert by_id["car:2140"]["box"]["x1"] == 401
    assert "car:2143" not in by_id

    at_cover = concurrent_scene_objects(tracking, 100.0)
    assert {item["episode_identity"] for item in at_cover} == {"car:2140", "car:2138"}


def test_merge_preserves_witness_visibility_through_inventory_update():
    existing = [
        {
            "label": "car",
            "track_id": 2140,
            "incident_eligible": True,
            "zones": ["Road"],
            "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 40},
            "snapshot_visible": True,
        },
        {
            "label": "car",
            "track_id": 2138,
            "incident_eligible": False,
            "zones": [],
            "zone_admission_reason": "outside_incident_zone",
            "box": {"x1": 50, "y1": 10, "x2": 80, "y2": 40},
            "snapshot_visible": False,
        },
    ]
    cover = [
        {
            "label": "car",
            "track_id": 2140,
            "incident_eligible": True,
            "box": {"x1": 100, "y1": 100, "x2": 400, "y2": 400},
            "detection_frame_width": 1280,
            "detection_frame_height": 720,
            "snapshot_visible": True,
            "cover_role": "primary",
            "box_provenance": "detected_in_main",
            "native_cover_verified": True,
        },
        {
            "label": "car",
            "track_id": 2138,
            "incident_eligible": False,
            "box": {"x1": 500, "y1": 100, "x2": 800, "y2": 400},
            "detection_frame_width": 1280,
            "detection_frame_height": 720,
            "snapshot_visible": True,
            "cover_role": "witness",
            "box_provenance": "detected_in_main",
            "native_cover_verified": True,
            "zone_admission_reason": "outside_incident_zone",
        },
    ]
    projected = merge_cover_objects(existing, cover)
    witness = next(item for item in projected if item["track_id"] == 2138)
    assert witness["snapshot_visible"] is True
    assert witness["cover_role"] == "witness"
    assert witness["incident_eligible"] is False
    assert witness["box"]["x1"] == 500

    inventory = [
        {
            "label": "car",
            "track_id": 2140,
            "incident_eligible": True,
            "zones": ["Road"],
            "box": {"x1": 11, "y1": 11, "x2": 41, "y2": 41},
        },
        {
            "label": "car",
            "track_id": 2138,
            "incident_eligible": False,
            "zones": [],
            "zone_admission_reason": "outside_incident_zone",
            "box": {"x1": 51, "y1": 11, "x2": 81, "y2": 41},
        },
    ]
    merged = merge_inventory_objects(projected, inventory)
    witness = next(item for item in merged if item["track_id"] == 2138)
    assert witness["snapshot_visible"] is True
    assert witness["cover_role"] == "witness"
    assert witness["box"]["x1"] == 500
    assert witness["incident_eligible"] is False


def test_candidate_score_bonuses_concurrent_peers_without_requiring_eligibility():
    rng = np.random.default_rng(0)
    image = rng.integers(30, 220, size=(360, 640, 3), dtype=np.uint8)
    single = [
        {
            "label": "car",
            "confidence": 0.9,
            "incident_eligible": True,
            "box": {"x1": 200, "y1": 80, "x2": 320, "y2": 160},
        }
    ]
    multi = [
        *single,
        {
            "label": "car",
            "confidence": 0.85,
            "incident_eligible": False,
            "box": {"x1": 400, "y1": 80, "x2": 520, "y2": 160},
        },
    ]
    alone = candidate_score(image, single)
    together = candidate_score(image, multi)
    assert alone is not None and together is not None
    assert together == alone + 0.15


def test_cover_enriched_still_prefers_more_visible_objects_including_witness():
    existing = [
        {
            "label": "car",
            "track_id": 1,
            "snapshot_visible": True,
            "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
        },
        {
            "label": "car",
            "track_id": 2,
            "snapshot_visible": False,
            "incident_eligible": False,
            "box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
        },
    ]
    enriched = [
        {
            "label": "car",
            "track_id": 1,
            "snapshot_visible": True,
            "cover_role": "primary",
            "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
        },
        {
            "label": "car",
            "track_id": 2,
            "snapshot_visible": True,
            "cover_role": "witness",
            "incident_eligible": False,
            "box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
        },
    ]
    assert should_adopt_native_cover(existing, enriched, 8.0, 10.0) == "cover_enriched"

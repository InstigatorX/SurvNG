"""Cover promotion must annotate co-present inventory on the verified main raster."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from survng.app.config import AppConfig, CameraConfig
from survng.app.native_main_frame import NativeMainFrameVerifier


def _verifier():
    config = AppConfig()
    config.cameras = [
        CameraConfig(id="test", name="Test", stream_url="rtsp://unused.invalid"),
    ]
    counts = {}
    verifier = NativeMainFrameVerifier(lambda: config, Mock(), counts)
    return verifier, config


def test_match_scene_detections_promotes_siblings_onto_main():
    verifier, _ = _verifier()
    main = np.zeros((720, 1280, 3), dtype=np.uint8)
    primary = {
        "label": "person",
        "track_id": 1,
        "native_track_id": 11,
        "confidence": 0.91,
        "box": {"x1": 200.0, "y1": 100.0, "x2": 320.0, "y2": 500.0},
        "incident_eligible": False,
        "zones": [],
    }
    car = {
        "label": "car",
        "track_id": 2,
        "native_track_id": 12,
        "confidence": 0.88,
        "box": {"x1": 600.0, "y1": 300.0, "x2": 1000.0, "y2": 600.0},
        "incident_eligible": False,
        "zones": ["Parked Car"],
    }
    person_b = {
        "label": "person",
        "track_id": 3,
        "native_track_id": 13,
        "confidence": 0.84,
        "box": {"x1": 400.0, "y1": 120.0, "x2": 480.0, "y2": 420.0},
        "incident_eligible": True,
        "zones": ["Entry"],
    }
    detections = [
        {"label": "person", "confidence": 0.93, "box": {"x1": 205, "y1": 105, "x2": 315, "y2": 495}},
        {"label": "car", "confidence": 0.9, "box": {"x1": 610, "y1": 310, "x2": 990, "y2": 590}},
        {"label": "person", "confidence": 0.86, "box": {"x1": 405, "y1": 125, "x2": 475, "y2": 415}},
    ]
    verifier.detect = Mock(return_value=detections)

    scene = verifier.match_scene_detections(
        "test",
        [primary, car, person_b],
        main,
        1234.5,
        primary=dict(primary, native_cover_verified=True, box_provenance="detected_in_main"),
        detections=detections,
    )
    assert len(scene) == 3
    assert scene[0]["track_id"] == 1
    assert scene[0]["snapshot_visible"] is True
    assert scene[0]["detection_frame_width"] == 1280
    assert {item["track_id"] for item in scene} == {1, 2, 3}
    assert all(item["snapshot_visible"] is True for item in scene)
    assert all(item["box_provenance"] == "detected_in_main" for item in scene)
    assert all(item["detection_frame_height"] == 720 for item in scene)
    car_item = next(item for item in scene if item["track_id"] == 2)
    assert car_item["box"]["x1"] == 610


def test_project_main_keeps_edge_clipped_siblings(monkeypatch):
    verifier, _ = _verifier()
    live = np.zeros((512, 896, 3), dtype=np.uint8)
    main = np.zeros((2160, 3840, 3), dtype=np.uint8)
    person = {
        "label": "person",
        "track_id": 62,
        "box": {"x1": 368.0, "y1": 261.0, "x2": 429.0, "y2": 442.0},
    }
    # Touches the live bottom edge; modest sy>1 previously dropped this car.
    car = {
        "label": "car",
        "track_id": 3,
        "box": {"x1": 413.0, "y1": 307.0, "x2": 882.0, "y2": 512.0},
    }
    monkeypatch.setattr(
        "survng.app.native_main_frame.estimate_stream_alignment",
        lambda *_args, **_kwargs: (0.998439, 1.01508, 0.000318, -0.006778),
    )
    from survng.app.native_evidence_common import Candidate

    projected = verifier.project_main(Candidate(1.0, live, [person, car], 5.0), main)
    assert {item["track_id"] for item in projected} == {62, 3}
    car_box = next(item["box"] for item in projected if item["track_id"] == 3)
    assert car_box["y2"] == 2160.0
    assert car_box["y1"] < car_box["y2"]


def test_verify_candidate_exposes_cover_objects_for_scene(monkeypatch):
    verifier, _ = _verifier()
    live = np.random.default_rng(3).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    main = cv2.resize(live, (1280, 720))
    person = {
        "label": "person",
        "track_id": 1,
        "confidence": 0.9,
        "box": {"x1": 100.0, "y1": 80.0, "x2": 200.0, "y2": 300.0},
        "zones": [],
    }
    car = {
        "label": "car",
        "track_id": 2,
        "confidence": 0.88,
        "box": {"x1": 300.0, "y1": 150.0, "x2": 500.0, "y2": 300.0},
        "zones": [],
    }
    projected = [
        dict(person, box={k: v * 2 for k, v in person["box"].items()}),
        dict(car, box={k: v * 2 for k, v in car["box"].items()}),
    ]
    verifier.project_main = Mock(return_value=deepcopy(projected))
    verifier.read_frame = Mock(return_value=main)

    def fake_detect(image, *, priority="cover"):
        # Crop path and full-frame path both need plausible detections.
        if image.shape[0] < main.shape[0]:
            return [{
                "label": "person",
                "confidence": 0.92,
                "box": {
                    "x1": 20.0,
                    "y1": 20.0,
                    "x2": 120.0,
                    "y2": 240.0,
                },
            }]
        return [
            {"label": "person", "confidence": 0.92, "box": projected[0]["box"]},
            {"label": "car", "confidence": 0.9, "box": projected[1]["box"]},
        ]

    verifier.detect = Mock(side_effect=fake_detect)
    monkeypatch.setattr(
        "survng.app.native_main_frame.image_quality",
        lambda image: 1.0,
    )
    monkeypatch.setattr(
        "survng.app.native_main_frame.matches_object_extent",
        lambda expected, actual: True,
    )
    from survng.app.native_evidence_common import Candidate

    result = verifier.verify_candidate(
        "test",
        Candidate(100.0, live, [person, car], 5.0),
        priority="cover",
    )
    assert result["status"] == "confirmed"
    assert len(result["cover"]) == 3
    assert len(result["cover_objects"]) == 2
    assert {item["label"] for item in result["cover_objects"]} == {"person", "car"}
    assert all(item["snapshot_visible"] is True for item in result["cover_objects"])

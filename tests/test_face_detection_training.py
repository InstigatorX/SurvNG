from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from survng.app.face_detection import parse_ssd_face_detections
from survng.app.face_detection_dataset import (
    FaceDetectionDataset,
    FaceDetectionDatasetError,
    assert_export_has_no_identity,
    file_sha256,
    load_person_observations,
    propose_day_splits,
)
from survng.app.face_detection_evaluation import (
    evaluate_predictions,
    latency_gate,
    parse_ssd_face_detections as evaluation_parser,
)
from survng.app.face_detection_geometry import upper_body_window
from survng.app.face_detection_routes import (
    FaceDetectionRouteDependencies,
    create_face_detection_router,
    embedding_fingerprint_unchanged,
)
from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector


def test_upper_body_window_matches_the_live_padding() -> None:
    window = upper_body_window((100.0, 200.0, 300.0, 600.0), 1000, 800)
    assert window == (84, 180, 316, 472)
    assert upper_body_window((0.0, 0.0, 10.0, 10.0), 100, 100) is None
    assert upper_body_window((10.0, 2.0, 80.0, 40.0), 100, 50)[1] == 0


def test_live_face_scan_uses_the_shared_window() -> None:
    frame = np.zeros((800, 1000, 3), dtype=np.uint8)
    seen: list[tuple[int, int]] = []

    def detect(crop: np.ndarray) -> list[dict]:
        seen.append((crop.shape[0], crop.shape[1]))
        return [{
            "label": "face",
            "confidence": 0.91,
            "box": {"x1": 1.0, "y1": 2.0, "x2": 12.0, "y2": 14.0},
        }]

    faces = RecordedMotionObjectDetector._detect_faces_in_people(
        frame,
        [{"label": "person", "confidence": 0.8, "box": {"x1": 100, "y1": 200, "x2": 300, "y2": 600}}],
        detect,
        max_people=4,
    )
    assert seen == [(292, 232)]
    assert faces[0]["box"] == {"x1": 85.0, "y1": 182.0, "x2": 96.0, "y2": 194.0}
    assert faces[0]["parent_person_box"]["x1"] == 100


def test_ssd_parser_reads_seven_float_rows() -> None:
    raw = np.array([
        [0, 1, 0.91, 0.10, 0.20, 0.50, 0.60],
        [0, 1, 0.20, 0.00, 0.00, 0.20, 0.20],
        [0, 1, 0.80, 0.90, 0.90, 0.10, 0.20],
    ], dtype=np.float32)
    detections = parse_ssd_face_detections(raw, 100, 50, 0.60)
    assert len(detections) == 1
    assert detections[0]["confidence"] == 0.91
    assert detections[0]["box"]["x1"] == pytest.approx(10.0)
    assert detections[0]["box"]["y1"] == pytest.approx(10.0)
    assert detections[0]["box"]["x2"] == pytest.approx(50.0)
    assert detections[0]["box"]["y2"] == pytest.approx(30.0)
    assert evaluation_parser(raw, 100, 50, 0.60) == detections
    with pytest.raises(ValueError):
        parse_ssd_face_detections(np.array([0.5, 0.2], dtype=np.float32), 10, 10, 0.60)


def test_labels_stay_out_of_the_face_gallery_package() -> None:
    root = Path(__file__).resolve().parents[1] / "survng" / "app" / "face_store"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "face_detection_samples" not in source
    assert "face_detection_annotations" not in source


def test_materialize_labels_splits_and_export_hash(tmp_path: Path) -> None:
    store = FaceDetectionDataset(tmp_path / "face.sqlite3", tmp_path / "crops")
    frames = {}
    for day in (1, 2, 3):
        image = np.zeros((400, 220, 3), dtype=np.uint8)
        image[20:80, 30:90] = (20, 40, day)
        frames[day] = image
    stats = store.materialize(
        [_observation(day, frames[day], source=f"person-{day}") for day in (1, 2, 3)],
        frame_reader=lambda _path, seek: frames[int(seek)],
        timezone_name="UTC",
        limit=10,
    )
    assert stats["stored"] == 3
    queued = store.queue()
    assert len(queued) == 3
    assert all(sample["baseline_miss"] for sample in queued)
    columns = {
        row[1]
        for row in sqlite3.connect(store.database_path).execute(
            "pragma table_info(face_detection_samples)"
        )
    }
    assert "person_id" not in columns
    assert "name" not in columns
    for sample in queued:
        labeled = store.label(sample["id"], [{
            "kind": "face",
            "box": {"x1": 4, "y1": 4, "x2": 40, "y2": 40},
        }])
        assert labeled["disposition"] == "reviewed"
        assert "person_id" not in labeled
    with pytest.raises(FaceDetectionDatasetError):
        store.label(queued[0]["id"], [
            {"kind": "no_face"},
            {"kind": "face", "box": {"x1": 4, "y1": 4, "x2": 20, "y2": 20}},
        ])
    with pytest.raises(FaceDetectionDatasetError, match="Freeze splits"):
        store.export(tmp_path / "too-soon")
    assignment = store.assign_splits()
    assert set(assignment.values()) == {"train", "calibration", "held_out"}
    manifest = store.export(tmp_path / "export")
    assert_export_has_no_identity(manifest)
    for sample in manifest["samples"]:
        exported = tmp_path / "export" / sample["path"]
        assert file_sha256(exported) == sample["sha256"]


def test_identical_pixels_cannot_cross_splits(tmp_path: Path) -> None:
    days = propose_day_splits(["2026-01-01", "2026-01-02", "2026-01-03"])
    first, second = _days_in_different_splits(days)
    store = FaceDetectionDataset(tmp_path / "face.sqlite3", tmp_path / "crops")
    blank = np.zeros((80, 80, 3), dtype=np.uint8)
    distinct = blank.copy()
    distinct[0, 0] = (1, 2, 3)
    third = next(day for day in days if day not in {first, second})
    frames = {first: blank, second: blank, third: distinct}
    store.materialize(
        [
            _observation(_day_number(day), frames[day], source=day, seek=_day_number(day))
            for day in days
        ],
        frame_reader=lambda _path, seek, frames=frames: frames[_day_key(seek)],
        timezone_name="UTC",
        limit=10,
    )
    for sample in store.queue(limit=10):
        store.label(sample["id"], [{"kind": "no_face"}])
    with pytest.raises(FaceDetectionDatasetError, match="cross splits"):
        store.assign_splits()


def test_promotion_gate_and_latency() -> None:
    face = {"x1": 10.0, "y1": 10.0, "x2": 40.0, "y2": 40.0}
    samples = [
        _scored("held_out", face, candidate=[_prediction(face, 0.95)], baseline=[]),
        _scored("held_out", None, candidate=[], baseline=[]),
        _scored("calibration", face, candidate=[_prediction(face, 0.95)], baseline=[]),
        _scored("calibration", None, candidate=[], baseline=[]),
    ]
    # Two held-out faces are required for a stable recall comparison. The miss
    # set is the face the baseline did not find.
    samples.append(_scored("held_out", face, candidate=[_prediction(face, 0.96)], baseline=[_prediction(face, 0.97)], sample_id="known"))
    report = evaluate_predictions(samples, call_seconds=[0.001] * 4)
    assert report["gate"]["passed"] is True
    assert report["miss_set"]["recall"] == 1.0
    slow = latency_gate([0.2])
    assert slow["passed"] is False
    failed = evaluate_predictions(
        [_scored("held_out", face, candidate=[], baseline=[]), _scored("held_out", None, candidate=[_prediction(face, 0.9)], baseline=[])],
        call_seconds=[0.001],
    )
    assert failed["gate"]["passed"] is False


def test_embedding_fingerprint_must_match_after_a_detector_swap() -> None:
    assert embedding_fingerprint_unchanged("model-a", "model-a")
    assert not embedding_fingerprint_unchanged("model-a", "model-b")
    assert not embedding_fingerprint_unchanged("", "")


def test_scene_observations_require_an_exact_person_frame(tmp_path: Path) -> None:
    database = tmp_path / "events.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        create table scene_observations (
            id text, event_id integer, camera_id text, captured_epoch real,
            recording_path text, payload_json text
        )
        """
    )
    exact = {
        "label": "person",
        "frame_timestamp_exact": True,
        "box": {"x1": 1, "y1": 2, "x2": 30, "y2": 80},
        "detection_frame_width": 100,
        "detection_frame_height": 80,
    }
    missed_time = dict(exact, frame_timestamp_exact=False)
    connection.execute(
        "insert into scene_observations values ('exact', 7, 'gate', 10, 'clip.mp4', ?)",
        (json.dumps(exact),),
    )
    connection.execute(
        "insert into scene_observations values ('inexact', 8, 'gate', 11, 'clip.mp4', ?)",
        (json.dumps(missed_time),),
    )
    connection.commit()
    connection.close()
    rows = load_person_observations(database)
    assert [row["source_observation_id"] for row in rows] == ["exact"]


def test_face_finding_route_rejects_a_mixed_no_face_label(tmp_path: Path) -> None:
    frame = np.full((160, 120, 3), 8, dtype=np.uint8)
    manager = SimpleNamespace(
        database_dir=tmp_path,
        events=SimpleNamespace(db_path=tmp_path / "missing.sqlite3"),
        recorder=SimpleNamespace(recording_at=lambda camera_id, epoch: {
            "path": "clip.mp4",
            "start_epoch": epoch - 1,
        }),
        faces=SimpleNamespace(recognition_status=lambda: {"model_fingerprint": "embed-1"}),
    )
    app = FastAPI()
    app.include_router(create_face_detection_router(FaceDetectionRouteDependencies(
        get_manager=lambda: manager,
        get_config=lambda: SimpleNamespace(
            ffmpeg_path="ffmpeg",
            detector=SimpleNamespace(face_detection_model_path=""),
        ),
        frame_reader=lambda _path, _seek: frame,
        fingerprint_reader=lambda: "embed-1",
    )))
    client = TestClient(app)
    database = tmp_path / "events.sqlite3"
    _write_exact_observation(database)
    manager.events.db_path = database
    stored = client.post("/api/face-detection/samples/materialize", json={"timezone": "UTC", "limit": 5})
    assert stored.status_code == 200
    assert stored.json()["stored"] == 1
    queued = client.get("/api/face-detection/samples")
    sample = queued.json()["samples"][0]
    rejected = client.put(f"/api/face-detection/samples/{sample['id']}", json={
        "annotations": [
            {"kind": "no_face"},
            {"kind": "face", "box": {"x1": 2, "y1": 2, "x2": 20, "y2": 20}},
        ],
    })
    assert rejected.status_code == 422
    crop = client.get(sample["crop_url"])
    assert crop.status_code == 200
    assert crop.headers["content-type"].startswith("image/png")


def _observation(day: int, frame: np.ndarray, *, source: str, seek: float | None = None) -> dict:
    del frame
    height, width = 400, 220
    return {
        "source_observation_id": source,
        "event_id": day,
        "camera_id": "gate",
        "captured_epoch": datetime(2026, 1, day, 12, tzinfo=timezone.utc).timestamp(),
        "recording_path": "clip.mp4",
        "seek_seconds": day if seek is None else seek,
        "person_box": {"x1": 10.0, "y1": 20.0, "x2": float(width - 10), "y2": float(height - 20)},
        "detection_frame_width": width,
        "detection_frame_height": height,
        "frame_timestamp_exact": True,
    }


def _day_number(day: str) -> int:
    return int(day.rsplit("-", 1)[1])


def _day_key(seek: float) -> str:
    return f"2026-01-{int(seek):02d}"


def _days_in_different_splits(assignment: dict[str, str]) -> tuple[str, str]:
    grouped: dict[str, list[str]] = {}
    for day, split in assignment.items():
        grouped.setdefault(split, []).append(day)
    splits = [days[0] for days in grouped.values() if days]
    return splits[0], splits[1]


def _prediction(box: dict, confidence: float) -> dict:
    return {"confidence": confidence, "box": box}


def _scored(split: str, face: dict | None, *, candidate: list[dict], baseline: list[dict], sample_id: str = "") -> dict:
    annotations = [{"kind": "face", "box": face}] if face else [{"kind": "no_face", "box": None}]
    return {
        "id": sample_id or f"{split}-{len(candidate)}-{face is not None}",
        "split": split,
        "disposition": "reviewed",
        "annotations": annotations,
        "candidate": candidate,
        "baseline": baseline,
    }


def _write_exact_observation(database: Path) -> None:
    connection = sqlite3.connect(database)
    connection.execute(
        """
        create table scene_observations (
            id text, event_id integer, camera_id text, captured_epoch real,
            recording_path text, payload_json text
        )
        """
    )
    payload = {
        "label": "person",
        "frame_timestamp_exact": True,
        "box": {"x1": 10, "y1": 20, "x2": 100, "y2": 150},
        "detection_frame_width": 120,
        "detection_frame_height": 160,
    }
    connection.execute(
        "insert into scene_observations values ('obs-1', 4, 'gate', 1000, 'clip.mp4', ?)",
        (json.dumps(payload),),
    )
    connection.commit()
    connection.close()

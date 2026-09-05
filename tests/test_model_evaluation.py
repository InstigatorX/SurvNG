from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from survng.app.config import AppConfig
from survng.app.model_evaluation import ModelEvaluationRunner, _reference_labels


def _model(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    xml = directory / "best.xml"
    xml.write_text("<xml/>", encoding="utf-8")
    xml.with_suffix(".bin").write_bytes(b"weights")
    return xml


def test_reference_labels_include_only_objects_visible_in_snapshot() -> None:
    event = {
        "objects_json": json.dumps([
            {"label": "car"},
            {"label": "person", "snapshot_visible": False},
            {"status": "motion_qualification"},
        ])
    }

    assert _reference_labels(event) == {"car"}


def test_model_path_must_stay_inside_models_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    outside = _model(tmp_path, "outside")
    (tmp_path / "models").mkdir()
    runner = ModelEvaluationRunner(lambda: None, AppConfig)

    with pytest.raises(ValueError, match="models directory"):
        runner._validated_model_path(str(outside))


def test_comparison_uses_identical_corpus_and_reports_disagreements(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    baseline = _model(tmp_path / "models", "old")
    candidate = _model(tmp_path / "models", "new")
    storage = tmp_path / "storage"
    storage.mkdir()
    events = []
    for event_id, camera_id in ((1, "gate"), (2, "yard")):
        snapshot = storage / f"{event_id}.jpg"
        cv2.imwrite(str(snapshot), np.zeros((32, 48, 3), dtype=np.uint8))
        events.append({
            "id": event_id,
            "camera_id": camera_id,
            "created_at": f"2026-08-13T12:00:0{event_id}+00:00",
            "snapshot_path": snapshot.name,
            "objects_json": json.dumps([{"label": "person"}]),
        })
    # The minimum production corpus is ten. Repeat camera-balanced rows with
    # distinct IDs while sharing the same two harmless test images.
    events = [dict(events[index % 2], id=index + 1) for index in range(10)]
    lease_entries: list[str] = []

    class Lease:
        def __enter__(self):
            lease_entries.append("enter")

        def __exit__(self, *_args):
            lease_entries.append("exit")

    manager = SimpleNamespace(
        storage_dir=storage,
        detector=SimpleNamespace(offline_device_lease=lambda: Lease()),
        events=SimpleNamespace(
            recent=lambda _limit: events,
            motion_audits=lambda **_kwargs: ([], 0),
        ),
    )

    class FakeDetector:
        def __init__(self, config) -> None:
            self.path = config.model_path

        def status(self):
            return {"openvino_loaded": True, "loaded_device": "GPU", "loaded_backend": "openvino", "input_shape": [640, 640], "model_load_ms": 1.0}

        def detect(self, _frame, confidence_threshold=None):
            label = "car" if "/new/" in self.path else "person"
            return [{"label": label, "confidence": confidence_threshold, "box": [0, 0, 10, 10]}]

    runner = ModelEvaluationRunner(lambda: manager, lambda: AppConfig())
    with patch("survng.app.model_evaluation.OpenVinoDetector", FakeDetector):
        runner._run(str(baseline.resolve()), str(candidate.resolve()), 10, 0.25)

    status = runner.status()
    assert status["status"] == "completed"
    assert status["result"]["sample_count"] == 10
    assert status["result"]["compared_sample_count"] == 10
    assert status["result"]["failed_sample_count"] == 0
    assert status["result"]["camera_count"] == 2
    assert status["result"]["disagreement_frames"] == 10
    assert status["result"]["baseline"]["label_counts"] == {"person": 10}
    assert status["result"]["candidate"]["label_counts"] == {"car": 10}
    assert status["result"]["stored_evidence_recall"] == {
        "baseline": 1.0,
        "candidate": 0.0,
        "reference_labels": 10,
    }
    assert lease_entries == ["enter", "exit"] * 22


@pytest.mark.parametrize(
    ("baseline_errors", "candidate_errors"),
    [(set(range(10)), set(range(10))), (set(), set(range(10))), ({0}, {1, 3})],
)
def test_failed_inference_pairs_do_not_count_as_agreement_or_recall(
    tmp_path, baseline_errors, candidate_errors,
):
    events = []
    for index in range(10):
        snapshot = tmp_path / f"{index}.png"
        assert cv2.imwrite(str(snapshot), np.full((24, 32, 3), index, dtype=np.uint8))
        events.append({
            "id": index + 1, "camera_id": "gate", "snapshot_path": snapshot.name,
            "objects_json": json.dumps([{"label": "person"}]),
        })
    manager = SimpleNamespace(
        storage_dir=tmp_path,
        events=SimpleNamespace(recent=lambda _limit: events, motion_audits=lambda **_kwargs: ([], 0)),
    )

    class Detector:
        def __init__(self, config):
            self.candidate = config.model_path == "candidate.xml"

        def status(self):
            return {"openvino_loaded": True}

        def detect(self, frame, confidence_threshold=None):
            index = int(frame[0, 0, 0])
            if index in (candidate_errors if self.candidate else baseline_errors):
                return [{"status": "inference_error"}]
            return [{"label": "car" if self.candidate and index == 2 else "person"}]

    runner = ModelEvaluationRunner(lambda: manager, AppConfig)
    with patch("survng.app.model_evaluation.OpenVinoDetector", Detector):
        runner._run("baseline.xml", "candidate.xml", 10, 0.25)

    status = runner.status()
    result = status["result"]
    compared = 10 - len(baseline_errors | candidate_errors)
    assert status["status"] == ("completed" if compared else "failed")
    assert bool(status["error"]) is (not compared)
    assert result["sample_count"] == 10
    assert result["compared_sample_count"] == compared
    assert result["failed_sample_count"] == 10 - compared
    assert result["baseline"]["errors"] == len(baseline_errors)
    assert result["candidate"]["errors"] == len(candidate_errors)
    for model, errors in (("baseline", baseline_errors), ("candidate", candidate_errors)):
        if len(errors) == 10:
            assert result[model]["average_ms"] == result[model]["p95_ms"] == 0
    assert result["disagreement_frames"] == (1 if compared else 0)
    assert result["agreement_frames"] == (compared - 1 if compared else 0)
    assert result["stored_evidence_recall"] == {
        "baseline": 1.0 if compared else None,
        "candidate": round((compared - 1) / compared, 3) if compared else None,
        "reference_labels": compared,
    }
    if compared:
        assert [row["source_id"] for row in result["disagreements"]] == [3]
    else:
        assert result["disagreements"] == []


def test_unreadable_corpus_fails_without_claiming_model_agreement(tmp_path):
    image = tmp_path / "corrupt.jpg"
    image.write_bytes(b"not an image")
    events = [{
        "id": index + 1, "camera_id": "gate", "snapshot_path": image.name,
        "objects_json": json.dumps([{"label": "person"}]),
    } for index in range(10)]
    manager = SimpleNamespace(
        storage_dir=tmp_path,
        events=SimpleNamespace(recent=lambda _limit: events, motion_audits=lambda **_kwargs: ([], 0)),
    )
    runner = ModelEvaluationRunner(lambda: manager, AppConfig)
    with patch("survng.app.model_evaluation.OpenVinoDetector") as detector:
        detector.return_value.status.return_value = {"openvino_loaded": True}
        runner._run("baseline.xml", "candidate.xml", 10, 0.25)
        detector.return_value.detect.assert_not_called()
    status = runner.status()
    assert status["status"] == "failed"
    assert status["error"] == "No images could be compared successfully."
    result = status["result"]
    assert result["compared_sample_count"] == 0
    assert result["failed_sample_count"] == 10
    assert result["agreement_frames"] == result["disagreement_frames"] == 0
    for model in ("baseline", "candidate"):
        assert result[model]["errors"] == 10
        assert result[model]["average_ms"] == result[model]["p95_ms"] == 0

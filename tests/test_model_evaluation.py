from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from survng.app.config import AppConfig
from survng.app.model_evaluation import ModelEvaluationRunner


def _model(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    xml = directory / "best.xml"
    xml.write_text("<xml/>", encoding="utf-8")
    xml.with_suffix(".bin").write_bytes(b"weights")
    return xml


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
    manager = SimpleNamespace(
        storage_dir=storage,
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
    assert status["result"]["camera_count"] == 2
    assert status["result"]["disagreement_frames"] == 10
    assert status["result"]["baseline"]["label_counts"] == {"person": 10}
    assert status["result"]["candidate"]["label_counts"] == {"car": 10}
    assert status["result"]["stored_evidence_recall"] == {
        "baseline": 1.0,
        "candidate": 0.0,
        "reference_labels": 10,
    }

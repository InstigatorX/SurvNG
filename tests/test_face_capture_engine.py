from pathlib import Path

import numpy as np
import pytest

from survng.app.config import DetectorConfig
from survng.app.face_capture.engine import ALIGNMENT_TEMPLATE, FaceCaptureEngine, align_face, resolve_model_path

CONFIG = DetectorConfig(face_capture_enabled=True)
MODELS_PRESENT = all(
    Path(path).is_file()
    for path in (CONFIG.face_capture_detector_model_path, CONFIG.face_capture_embedding_model_path)
)


def test_alignment_maps_the_landmarks_onto_the_template():
    image = np.zeros((300, 300, 3), np.uint8)
    image[140:160, 140:160] = 255
    landmarks = ALIGNMENT_TEMPLATE * 2.0 + 50.0  # a face twice the template size
    aligned = align_face(image, landmarks)
    assert aligned is not None and aligned.shape == (112, 112, 3)
    assert align_face(image, np.full((5, 2), np.nan)) is None


def test_a_models_path_falls_back_to_the_local_models_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "x").mkdir(parents=True)
    (tmp_path / "models" / "x" / "m.onnx").write_bytes(b"")
    assert resolve_model_path("/models/x/m.onnx") == Path("models/x/m.onnx")
    with pytest.raises(FileNotFoundError):
        resolve_model_path("/models/x/missing.onnx")


def test_missing_models_leave_the_engine_unavailable_not_crashed():
    engine = FaceCaptureEngine(CONFIG.model_copy(update={"face_capture_detector_model_path": "/nowhere.onnx"}))
    assert not engine.ready and "nowhere" in engine.error
    with pytest.raises(RuntimeError):
        engine.capture(np.zeros((100, 100, 3), np.uint8))


@pytest.mark.skipif(not MODELS_PRESENT, reason="face capture models are not installed")
def test_the_installed_models_load_and_find_nothing_in_a_blank_crop():
    engine = FaceCaptureEngine(CONFIG)
    assert engine.ready, engine.error
    assert engine.capture(np.zeros((400, 300, 3), np.uint8)) == []
    assert engine.capture(np.zeros((4, 4, 3), np.uint8)) == []

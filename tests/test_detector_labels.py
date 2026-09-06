from __future__ import annotations

import numpy as np
import pytest

from survng.app.config import CameraConfig, DetectionZone, DetectorConfig
from survng.app.detector import OpenVinoDetector
from survng.app.inference_runtime.process import load_detector_labels
from survng.app.system_routes import openvino_package_classes
from survng.app.zones import apply_detection_zones


def test_selected_classes_only_package_keeps_catalog_and_inference_labels(tmp_path):
    model = tmp_path / "model.xml"
    model.touch()
    (tmp_path / "classes.txt").write_text("person\ncar\n", encoding="utf-8")
    classes, _task, error = openvino_package_classes(model)
    assert error == ""
    assert classes == ["person", "car"]
    # Selecting XML in Admin clears the old explicit labels_path.
    config = DetectorConfig(enabled=False, model_path=str(model), labels_path="")
    detector = OpenVinoDetector(config)
    assert detector.labels == load_detector_labels(config) == classes
    detector.input_shape = (640, 640)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _, metadata = detector._preprocess(frame)
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, 0] = [160, 230, 480, 410, .9, 0]
    objects = detector._parse_yolo_e2e_output(output, metadata)
    camera = CameraConfig(
        id="test", name="Test", stream_url="rtsp://example.invalid",
        zones=[DetectionZone(name="entry", object_classes=["person"], points=[
            {"x": .1, "y": .7}, {"x": .9, "y": .7},
            {"x": .9, "y": .85}, {"x": .1, "y": .85},
        ])],
    )
    apply_detection_zones(camera, objects, 1920, 1080, .45)
    assert objects[0]["label"] == "person"
    assert objects[0]["incident_eligible"] is True


@pytest.mark.parametrize("override", ["none", "inline", "path"])
def test_package_metadata_and_explicit_override_precedence(tmp_path, override):
    model = tmp_path / "model.xml"
    model.touch()
    (tmp_path / "metadata.yaml").write_text("names:\n  1: truck\n  0: person\n", encoding="utf-8")
    (tmp_path / "classes.txt").write_text("fallback\n", encoding="utf-8")
    explicit_path = tmp_path / "custom.txt"
    explicit_path.write_text("explicit\n", encoding="utf-8")
    config = DetectorConfig(
        enabled=False, model_path=str(model),
        labels=["inline"] if override != "none" else [],
        labels_path=str(explicit_path) if override == "path" else "",
    )
    expected = {"none": ["person", "truck"], "inline": ["inline"], "path": ["explicit"]}[override]
    assert load_detector_labels(config) == expected
    assert OpenVinoDetector(config).labels == expected
    assert openvino_package_classes(model)[0] == ["person", "truck"]


@pytest.mark.parametrize("metadata", ["names: {}\n", "names: [\n"])
def test_empty_or_invalid_metadata_falls_back_to_adjacent_classes(tmp_path, metadata):
    model = tmp_path / "model.xml"
    model.touch()
    (tmp_path / "metadata.yaml").write_text(metadata, encoding="utf-8")
    (tmp_path / "classes.txt").write_text("\n person \ncar\n", encoding="utf-8")
    config = DetectorConfig(enabled=False, model_path=str(model))
    assert load_detector_labels(config) == OpenVinoDetector(config).labels == ["person", "car"]
    assert openvino_package_classes(model)[0] == ["person", "car"]

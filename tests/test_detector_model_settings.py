from pathlib import Path

import numpy as np
import pytest

from survng.app.config import DetectorConfig
from survng.app.detector import OpenVinoDetector
from survng.app.detector_model_settings import ModelSettingsError, read_model_metadata, resolve_output_format


@pytest.mark.parametrize("metadata,graph,shape,expected", [
    ({"args": {"nms": True}}, False, [1, 2000, 6], "yolo-e2e"),
    ({"args": {"end2end": True, "nms": False}}, False, [1, 1, 6], "yolo-e2e"),
    ({"args": "{'nms': False, 'end2end': False}"}, False, [1, 6, 300], "yolo"),
    ({}, True, [1, -1, 6], "yolo-e2e"),
    ({}, False, [1, 84, 8400], "yolo"),
    ({}, False, [1, 6, 8400], "yolo"),
    ({}, False, [1, 8400, 7], "yolo"),
    ({"args": {"nms": False}}, False, [1, 8400, 7], "yolo"),
    ({}, False, [1, 300, 6], "yolo-e2e"),
    ({}, False, [100, 7], "ssd"),
    ({}, False, [1, 1, 100, 7], "ssd"),
])
def test_output_contract(metadata, graph, shape, expected):
    assert resolve_output_format([shape], "auto", metadata, graph)[0] == expected


def test_manual_override_resolves_two_class_ambiguity():
    shape = [[1, 6, 300]]
    assert resolve_output_format(shape, "auto", {})[2]
    assert resolve_output_format(shape, "yolo", {}) == ("yolo", "manual override", [])


def test_rank_three_ssd_can_be_selected_explicitly():
    assert resolve_output_format([[1, 100, 7]], "ssd", {}) == ("ssd", "manual override", [])


@pytest.mark.parametrize("shapes,metadata", [
    ([[1, 10]], {}),
    ([[1, 84, 8400]], {"nms": True}),
    ([[1, 300, 40], [1, 32, 160, 160]], {"nms": True}),
    ([[1, 56, 8400]], {"task": "pose"}),
])
def test_unsupported_contract_is_actionable(shapes, metadata):
    with pytest.raises(ModelSettingsError):
        resolve_output_format(shapes, "auto", metadata)


def test_malformed_metadata_is_reported(tmp_path):
    (tmp_path / "metadata.yaml").write_text("[not, a, mapping]")
    metadata, warnings = read_model_metadata(tmp_path / "model.xml")
    assert metadata == {}
    assert warnings


def export_test_model(tmp_path: Path, precision="f32", layout="NCHW", compressed=False):
    ov = pytest.importorskip("openvino")
    ops = ov.opset13
    dtype = np.float16 if precision == "f16" else np.float32
    shape = [1, 3, 32, 32] if layout == "NCHW" else [1, 32, 32, 3]
    parameter = ops.parameter(shape, getattr(ov.Type, precision))
    # Confidence follows normalized image intensity, so this tests preprocessing
    # through actual inference, as well as preserving duplicate final detections.
    boxes = np.array([[[4, 4, 24, 24, 0, 0], [4, 4, 24, 24, 0, 0]]], dtype=dtype)
    score_mask = np.zeros_like(boxes)
    score_mask[:, :, 4] = 1
    intensity = ops.reduce_mean(parameter, ops.constant([0, 1, 2, 3]), False)
    result = ops.add(ops.constant(boxes), ops.multiply(intensity, ops.constant(score_mask)))
    model = ov.Model([result], [parameter])
    model.set_rt_info(True, ["model_info", "nms"])
    model.set_rt_info(["person"], ["model_info", "labels"])
    path = tmp_path / "model.xml"
    ov.save_model(model, path, compress_to_fp16=compressed)
    return path


@pytest.mark.parametrize("precision,layout,compressed", [
    ("f16", "NCHW", False), ("f32", "NCHW", False),
    ("f32", "NCHW", True), ("f16", "NHWC", False),
])
def test_real_openvino_precision_and_final_detections(tmp_path, precision, layout, compressed):
    path = export_test_model(tmp_path, precision, layout, compressed)
    detector = OpenVinoDetector(DetectorConfig(
        enabled=True, model_path=str(path), cache_enabled=False,
    ))
    assert detector.enabled, detector.status()
    status = detector.status()
    assert status["model_settings"]["input_precision"] == precision
    assert status["model_settings"]["input_layout"] == layout
    assert status["model_settings"]["output_source"] == "embedded NMS metadata"
    assert detector.labels == ["person"]
    if compressed:
        assert "f16" in status["model_settings"]["constant_precisions"]
    assert status["warmup_error"] == ""
    detections = detector.detect(np.full((32, 32, 3), 255, dtype=np.uint8))
    assert len(detections) == 2  # Applying NMS twice would remove one.
    assert detections[0]["confidence"] == 1.0
    assert detections[0]["box"] == {"x1": 4, "y1": 4, "x2": 24, "y2": 24}


def test_invalid_layout_disables_detector_with_status_error(tmp_path):
    path = export_test_model(tmp_path)
    detector = OpenVinoDetector(DetectorConfig(
        enabled=True, model_path=str(path), model_input_layout="NHWC", cache_enabled=False,
    ))
    assert not detector.enabled
    assert "three-channel" in detector.status()["model_settings"]["error"]


def test_single_final_detection_is_preserved():
    detector = OpenVinoDetector(DetectorConfig(enabled=False, labels=["person"]))
    _, metadata = detector._preprocess(np.zeros((300, 300, 3), np.uint8))
    assert len(detector._parse_yolo_e2e_output(np.array([[[10, 10, 20, 20, .9, 0]]]), metadata)) == 1


def test_final_segmentation_does_not_apply_nms_twice():
    detector = OpenVinoDetector(DetectorConfig(enabled=False, labels=["person"]))
    detector.model_metadata = {"nms": True}
    detector.output_format = detector._resolve_output_format([[1, 2, 38], [1, 32, 8, 8]])
    assert detector.output_format == "yolo-seg-e2e"
    _, metadata = detector._preprocess(np.zeros((300, 300, 3), np.uint8))
    detections = np.zeros((1, 2, 38), dtype=np.float32)
    detections[0, :, :6] = [10, 10, 20, 20, .9, 0]
    objects = detector._parse_yolo_seg_outputs([detections, np.zeros((1, 32, 8, 8))], metadata)
    assert len(objects) == 2
    assert detector.status()["model_settings"]["nms"] == "model final detections"

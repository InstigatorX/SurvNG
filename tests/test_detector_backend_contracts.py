from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from survng.app.config import CameraConfig, DetectionZone, DetectorConfig
from survng.app.detector import OpenVinoDetector
from survng.app.zones import apply_detection_zones


class CaptureNet:
    def __init__(self, output):
        self.output = output
        self.tensor = None

    def setInput(self, tensor):
        self.tensor = tensor.copy()

    def forward(self):
        return self.output


@pytest.mark.parametrize("failure_stage", ["compile", "infer_request"])
def test_opencv_fallback_discards_partial_openvino_state(tmp_path, failure_stage):
    class Preprocessor:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: self

    class Core:
        def set_property(self, *_args):
            pass

        def read_model(self, **_kwargs):
            return SimpleNamespace(input=lambda _index: SimpleNamespace(shape=[1, 3, 320, 320]))

        def compile_model(self, **_kwargs):
            def fail():
                raise RuntimeError("simulated load failure")

            if failure_stage == "compile":
                fail()
            return SimpleNamespace(create_infer_request=fail)

    fake_openvino = SimpleNamespace(
        Core=Core, Layout=lambda value: value,
        Type=SimpleNamespace(u8="u8", f32="f32"),
    )
    fake_preprocess = SimpleNamespace(
        PrePostProcessor=lambda _model: Preprocessor(),
        ColorFormat=SimpleNamespace(BGR="BGR", RGB="RGB"),
    )
    model_path = tmp_path / "detector.onnx"
    model_path.touch()
    net = CaptureNet(np.zeros((1, 5, 2100), dtype=np.float32))
    with (
        patch.dict(sys.modules, {"openvino": fake_openvino, "openvino.preprocess": fake_preprocess}),
        patch("cv2.dnn.readNetFromONNX", return_value=net),
    ):
        detector = OpenVinoDetector(DetectorConfig(
            enabled=True, model_path=str(model_path), labels=["person"],
            cache_enabled=False, warmup_enabled=False,
        ))
    frame = np.full((320, 320, 3), [255, 128, 0], dtype=np.uint8)
    assert detector.detect(frame) == []
    assert detector.status()["loaded_backend"] == "opencv-dnn"
    assert detector.compiled_model is None
    assert detector.infer_request is None
    assert net.tensor.shape == (1, 3, 320, 320)
    assert net.tensor.dtype == np.float32
    np.testing.assert_allclose(net.tensor[0, :, 0, 0], [0, 128 / 255, 1])


@pytest.mark.parametrize("backend", ["ssd", "coreml"])
@pytest.mark.parametrize("frame_size", [(1920, 1080), (1080, 1920)])
def test_letterboxed_normalized_boxes_reach_source_zone(backend, frame_size, monkeypatch):
    width, height = frame_size
    detector = OpenVinoDetector(DetectorConfig(enabled=False, labels=["person", "car"]))
    detector.input_shape = (640, 640)
    detector.enabled = True
    source_box = np.array([width * .05, height * .25, width * .25, height * .75])
    scale = 640 / max(width, height)
    padding = np.array([(640 - round(width * scale)) // 2, (640 - round(height * scale)) // 2] * 2)
    model_box = (source_box * scale + padding) / 640
    if backend == "ssd":
        detector.cv_net = CaptureNet(np.array([[[[0, 1, .9, *model_box]]]], dtype=np.float32))
    else:
        x1, y1, x2, y2 = model_box
        detector.coreml_image_input = True
        detector.coreml_input_name = "image"
        # Pillow/Core ML are macOS-only dependencies. Keep real resize/padding
        # and mock only the final image wrapper at the native predict boundary.
        monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(
            Image=SimpleNamespace(fromarray=lambda rgb: rgb),
        ))

        def predict(inputs):
            assert inputs["image"].shape == (640, 640, 3)
            return {
                "coordinates": np.array([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]]),
                "confidence": np.array([[.9, .1]]),
            }

        detector.coreml_model = SimpleNamespace(predict=predict)
    objects = detector.detect(np.zeros((height, width, 3), dtype=np.uint8))
    assert len(objects) == 1
    np.testing.assert_allclose(list(objects[0]["box"].values()), source_box, atol=1)
    camera = CameraConfig(
        id="test", name="Test", stream_url="rtsp://example.invalid",
        zones=[DetectionZone(name="entry", object_classes=["person"], points=[
            {"x": 0, "y": .7}, {"x": .3, "y": .7},
            {"x": .3, "y": .8}, {"x": 0, "y": .8},
        ])],
    )
    apply_detection_zones(camera, objects, width, height, .45)
    assert objects[0]["incident_eligible"] is True
    assert objects[0]["zones"] == ["entry"]


@pytest.mark.parametrize("backend", ["opencv", "coreml"])
@pytest.mark.parametrize("layout", ["e2e", "raw_two_class"])
@pytest.mark.parametrize("batched", [True, False])
def test_alternate_backends_distinguish_e2e_from_two_class_yolo(backend, layout, batched):
    detector = OpenVinoDetector(DetectorConfig(enabled=False, labels=["person", "car"]))
    detector.enabled = True
    detector.input_shape = (640, 640)
    if layout == "e2e":
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, :2] = [[160, 230, 480, 410, .9, 0], [160, 230, 480, 410, .7, 1]]
    else:
        output = np.zeros((1, 6, 8400), dtype=np.float32)
        output[0, :, :2] = np.array([[320, 320, 320, 180, .9, .1], [320, 320, 320, 180, .1, .7]]).T
    if not batched:
        output = output[0]
    if backend == "opencv":
        detector.cv_net = CaptureNet(output)
    else:
        detector.coreml_model = SimpleNamespace(predict=lambda _inputs: {"output": output})
    objects = detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert [(item["label"], item["confidence"]) for item in objects] == [("person", .9), ("car", .7)]
    assert all(item["box"] == {"x1": 480, "y1": 270, "x2": 1440, "y2": 810} for item in objects)


def test_coreml_coordinates_without_metadata_keep_original_frame_contract():
    detector = OpenVinoDetector(DetectorConfig(enabled=False, labels=["person", "car"]))
    objects = detector._parse_coreml_coordinates(
        np.array([[.5, .5, .5, .5]]), np.array([[.9, .1]]), 1920, 1080,
    )
    assert objects[0]["box"] == {"x1": 480, "y1": 270, "x2": 1440, "y2": 810}


def test_openvino_declared_rank_two_output_keeps_ssd_contract():
    detector = OpenVinoDetector(DetectorConfig(enabled=False))
    detector.output_layer = SimpleNamespace(shape=(100, 7))
    detector.output_layers = [detector.output_layer]
    assert detector._detect_output_format() == "ssd"

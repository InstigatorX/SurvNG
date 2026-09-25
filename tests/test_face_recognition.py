from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from survng.app.config import DetectorConfig
from survng.app.face_recognition import OpenVinoFaceRecognizer


@pytest.mark.parametrize("profile,color,expected", [
    ("legacy_openvino", "BGR", [0., 127.5, 255.]),
    ("adaface", "BGR", [-1., 0., 1.]),
    ("arcface", "RGB", [1., 0., -1.]),
])
def test_embedding_profiles_use_explicit_pixel_contract(profile, color, expected):
    recognizer = OpenVinoFaceRecognizer(DetectorConfig(
        face_recognition_enabled=False, face_embedding_profile=profile,
    ))
    recognizer.input_shape = (112, 112)
    recognizer.input_color_order = color
    image = np.full((112, 112, 3), [0., 127.5, 255.], dtype=np.float32)
    tensor = recognizer._embedding_tensor(image)
    assert tensor.shape == (1, 3, 112, 112)
    np.testing.assert_allclose(tensor[0, :, 0, 0], expected)


def test_profiles_load_and_produce_distinct_fingerprints(tmp_path):
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops

    source = ops.parameter([1, 3, 112, 112], np.float32)
    model = ov.Model([ops.reduce_mean(source, [2, 3], False)], [source])
    model_path = tmp_path / "embedding.xml"
    ov.serialize(model, model_path)
    landmark_input = ops.parameter([1, 3, 48, 48], np.float32)
    coordinates = OpenVinoFaceRecognizer._ARCFACE_TEMPLATE.reshape(1, 10) / 112
    landmarks = ov.Model([ops.constant(coordinates)], [landmark_input])
    landmark_path = tmp_path / "landmarks.xml"
    ov.serialize(landmarks, landmark_path)
    fingerprints = []
    for profile in ("legacy_openvino", "adaface", "arcface"):
        recognizer = OpenVinoFaceRecognizer(DetectorConfig(
            face_recognition_enabled=True, face_embedding_profile=profile,
            face_embedding_model_path=str(model_path),
            face_landmark_model_path=str(landmark_path),
            face_recognition_device="CPU", cache_enabled=False,
        ))
        assert recognizer.ready, recognizer.error
        assert recognizer.input_color_order == ("RGB" if profile == "arcface" else "BGR")
        assert recognizer.status()["embedding_profile"] == profile
        fingerprints.append(recognizer.model_fingerprint)
        embedding = recognizer.embed(np.full((112, 112, 3), [0, 128, 255], np.uint8))
        assert np.isclose(np.linalg.norm(embedding), 1)
    assert len(set(fingerprints)) == 3
    assert fingerprints[0] == OpenVinoFaceRecognizer._fingerprint(model_path, landmark_path)


@pytest.mark.parametrize("coordinates", [np.zeros(10), np.ones(10) * 2, np.full(10, np.nan)])
def test_unusable_landmarks_reject_embedding(coordinates):
    recognizer = OpenVinoFaceRecognizer(DetectorConfig(face_recognition_enabled=False))
    from unittest.mock import Mock
    recognizer._landmark_request = Mock()
    recognizer._landmark_output = "output"
    recognizer._landmark_request.infer.return_value = {"output": coordinates}
    with pytest.raises(ValueError, match="landmark"):
        recognizer._align(np.zeros((112, 112, 3), np.uint8))


class FaceRecognitionTest(unittest.TestCase):
    def test_small_face_uses_upscaling_interpolation(self) -> None:
        face = np.zeros((48, 48, 3), dtype=np.uint8)
        with patch("survng.app.face_recognition.cv2.resize", wraps=cv2.resize) as resize:
            tensor = OpenVinoFaceRecognizer._image_tensor(face, (128, 128), "NCHW")

        self.assertEqual(tensor.shape, (1, 3, 128, 128))
        self.assertEqual(resize.call_args.kwargs["interpolation"], cv2.INTER_LINEAR)

    def test_image_input_rejects_unsupported_channel_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "three-channel"):
            OpenVinoFaceRecognizer._image_input([1, 1, 128, 128])
        with self.assertRaisesRegex(ValueError, "three-channel"):
            OpenVinoFaceRecognizer._image_input([1, 128, 128, 4])

    def test_landmark_and_embedding_inference_are_serialized_together(self) -> None:
        recognizer = OpenVinoFaceRecognizer(DetectorConfig(face_recognition_enabled=False))
        recognizer._infer_request = object()
        recognizer._input = "input"
        recognizer._output = "output"
        recognizer.input_shape = (4, 4)
        recognizer.input_layout = "NCHW"
        active = 0
        maximum_active = 0
        state_lock = threading.Lock()

        def align(face: np.ndarray) -> np.ndarray:
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return face

        class InferRequest:
            @staticmethod
            def infer(_inputs):
                return {"output": np.asarray([[1.0, 0.0]], dtype=np.float32)}

        recognizer._align = align  # type: ignore[method-assign]
        recognizer._infer_request = InferRequest()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        results: list[np.ndarray] = []
        workers = [threading.Thread(target=lambda: results.append(recognizer.embed(frame))) for _ in range(4)]

        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        self.assertEqual(maximum_active, 1)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(np.allclose(result, [1.0, 0.0]) for result in results))


if __name__ == "__main__":
    unittest.main()

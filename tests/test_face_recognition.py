from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from survng.app.config import DetectorConfig
from survng.app.face_recognition import OpenVinoFaceRecognizer


class FaceRecognitionTest(unittest.TestCase):
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

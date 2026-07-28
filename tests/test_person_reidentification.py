from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from survng.app.config import DetectorConfig
from survng.app.person_reidentification import (
    OpenVinoAppearanceReidentifier,
    OpenVinoPersonReidentifier,
)


class PersonReidentificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reidentifier = OpenVinoPersonReidentifier(DetectorConfig())

    def test_image_layout_detection_supports_nchw_and_nhwc(self) -> None:
        self.assertEqual(
            self.reidentifier._image_input([1, 3, 256, 128]),
            ("NCHW", (128, 256)),
        )
        self.assertEqual(
            self.reidentifier._image_input([1, 256, 128, 3]),
            ("NHWC", (128, 256)),
        )
        with self.assertRaisesRegex(ValueError, "four-dimensional"):
            self.reidentifier._image_input([3, 256, 128])

    def test_embed_normalizes_output_and_serializes_inference(self) -> None:
        active = 0
        maximum_active = 0
        state_lock = threading.Lock()

        class InferRequest:
            @staticmethod
            def infer(_inputs):
                nonlocal active, maximum_active
                with state_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.02)
                with state_lock:
                    active -= 1
                return {"output": np.asarray([[3.0, 4.0]], dtype=np.float32)}

        self.reidentifier._infer_request = InferRequest()
        self.reidentifier._input = "input"
        self.reidentifier._output = "output"
        workers = []
        results: list[np.ndarray] = []
        crop = np.zeros((32, 16, 3), dtype=np.uint8)
        for _ in range(3):
            worker = threading.Thread(
                target=lambda: results.append(self.reidentifier.embed(crop))
            )
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()

        self.assertEqual(maximum_active, 1)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(np.allclose(result, [0.6, 0.8]) for result in results))

    def test_embed_rejects_invalid_input_and_output(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            self.reidentifier.embed(np.zeros((32, 16, 3), dtype=np.uint8))

        class InvalidInferRequest:
            @staticmethod
            def infer(_inputs):
                return {"output": np.asarray([[np.nan, 0.0]], dtype=np.float32)}

        self.reidentifier._infer_request = InvalidInferRequest()
        self.reidentifier._input = "input"
        self.reidentifier._output = "output"
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.reidentifier.embed(np.zeros((32, 16, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "too small"):
            self.reidentifier.embed(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_appearance_router_uses_label_specific_models(self) -> None:
        config = DetectorConfig.model_validate({
            "tracking": {
                "vehicle_reid_enabled": True,
                "vehicle_reid_model_path": "missing-vehicle.xml",
            },
        })
        router = OpenVinoAppearanceReidentifier(config)
        router.vehicle._infer_request = type("Infer", (), {
            "infer": staticmethod(
                lambda _inputs: {"vehicle": np.asarray([[0.0, 2.0]])}
            ),
        })()
        router.vehicle._input = "input"
        router.vehicle._output = "vehicle"

        vehicle = router.embed_for_label(
            "car",
            np.zeros((32, 32, 3), dtype=np.uint8),
        )

        self.assertTrue(router.supports_label("car"))
        self.assertFalse(router.supports_label("dog"))
        self.assertTrue(np.allclose(vehicle, [0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export-mobileclip2-openvino.py"
SPEC = importlib.util.spec_from_file_location("survng_semantic_exporter", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)

SIGLIP_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export-siglip2-openvino.py"
SIGLIP_SPEC = importlib.util.spec_from_file_location(
    "survng_siglip2_exporter", SIGLIP_SCRIPT_PATH
)
assert SIGLIP_SPEC is not None and SIGLIP_SPEC.loader is not None
SIGLIP_EXPORTER = importlib.util.module_from_spec(SIGLIP_SPEC)
SIGLIP_SPEC.loader.exec_module(SIGLIP_EXPORTER)


class SemanticExporterTest(unittest.TestCase):
    def test_parity_validation_accepts_equivalent_normalized_vectors(self) -> None:
        reference = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        candidate = reference * 4.0

        result = EXPORTER._validate_pair(reference, candidate, "test", 0.999)

        self.assertGreaterEqual(result["minimum_cosine"], 0.999999)

    def test_parity_validation_rejects_incompatible_vectors(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "parity failed"):
            EXPORTER._validate_pair(
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                np.asarray([[0.0, 1.0]], dtype=np.float32),
                "test",
                0.99,
            )

    def test_siglip2_cross_modal_validation_detects_mismatched_towers(self) -> None:
        images = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        correct_text = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        wrong_text = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        with self.assertRaisesRegex(RuntimeError, "cross-modal OpenVINO parity failed"):
            SIGLIP_EXPORTER._validate_cross_modal(
                images, correct_text, images, wrong_text
            )

        result = SIGLIP_EXPORTER._validate_cross_modal(
            images, correct_text, images, correct_text
        )
        self.assertEqual(result, {"maximum_cosine_error": 0.0})


if __name__ == "__main__":
    unittest.main()

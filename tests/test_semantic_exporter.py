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


if __name__ == "__main__":
    unittest.main()

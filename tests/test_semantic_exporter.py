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

    def test_make_tree_world_readable_relaxes_mkdtemp_mode(self) -> None:
        import stat
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pkg"
            root.mkdir(mode=0o700)
            nested = root / "tokenizer"
            nested.mkdir(mode=0o700)
            file_path = nested / "semantic_model.json"
            file_path.write_text("{}\n", encoding="utf-8")
            file_path.chmod(0o600)

            EXPORTER._make_tree_world_readable(root)

            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o755)
            self.assertEqual(stat.S_IMODE(nested.stat().st_mode), 0o755)
            self.assertEqual(stat.S_IMODE(file_path.stat().st_mode), 0o644)

if __name__ == "__main__":
    unittest.main()

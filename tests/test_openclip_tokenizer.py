from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

from survng.app.openclip_tokenizer import OpenClipBpeTokenizer


class OpenClipTokenizerTest(unittest.TestCase):
    def test_official_vocabulary_tokenizes_known_clip_text(self) -> None:
        try:
            import open_clip
            from open_clip.tokenizer import default_bpe
        except ImportError:
            self.skipTest("OpenCLIP export dependency is not installed")
        tokenizer = OpenClipBpeTokenizer(Path(default_bpe()))
        expected = open_clip.get_tokenizer("MobileCLIP2-B")(["hello world", "red truck"])

        actual = tokenizer(["hello world", "red truck"])

        self.assertEqual(actual.tolist(), expected.numpy().tolist())

    def test_invalid_vocabulary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("#version: 0.2\na b\n")
            with self.assertRaisesRegex(RuntimeError, "unexpected size"):
                OpenClipBpeTokenizer(path)

    def test_long_text_is_truncated_with_end_token(self) -> None:
        try:
            from open_clip.tokenizer import default_bpe
        except ImportError:
            self.skipTest("OpenCLIP export dependency is not installed")
        tokenizer = OpenClipBpeTokenizer(Path(default_bpe()), context_length=8)

        tokens = tokenizer(["word " * 100])

        self.assertEqual(tokens.shape, (1, 8))
        self.assertEqual(tokens[0, -1], tokenizer.end_token)

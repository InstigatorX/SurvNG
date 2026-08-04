from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from survng.app.config import ImageStorageConfig
from survng.app.image_storage import DurableImageWriter


class DurableImageWriterTest(unittest.TestCase):
    def test_default_writer_atomically_stores_decodable_webp(self) -> None:
        frame = np.full((48, 64, 3), 127, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "snapshots"
            writer = DurableImageWriter(ImageStorageConfig())

            path = writer.write(directory, "event", frame)

            self.assertIsNotNone(path)
            assert path is not None
            self.assertEqual(path.suffix, ".webp")
            decoded = cv2.imread(str(path))
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape, frame.shape)
            self.assertEqual(list(directory.glob("*.tmp")), [])

    def test_reconfigure_switches_new_images_to_requested_jpeg_quality(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DurableImageWriter(ImageStorageConfig())
            writer.reconfigure(ImageStorageConfig(format="jpeg", quality=73))
            with patch("survng.app.image_storage.cv2.imencode", wraps=cv2.imencode) as encode:
                path = writer.write(Path(tmpdir), "event", frame)

        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path.suffix, ".jpg")
        self.assertEqual(encode.call_args.args[0], ".jpg")
        self.assertEqual(encode.call_args.args[2], [cv2.IMWRITE_JPEG_QUALITY, 73])

    def test_failed_encode_leaves_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            writer = DurableImageWriter(ImageStorageConfig())
            with patch("survng.app.image_storage.cv2.imencode", return_value=(False, None)):
                path = writer.write(directory, "event", np.zeros((8, 8, 3), dtype=np.uint8))

            self.assertIsNone(path)
            self.assertEqual(list(directory.iterdir()), [])

    def test_stored_images_includes_mixed_supported_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            for name in ("old.jpg", "new.webp", "portable.png", "ignore.mp4"):
                (directory / name).write_bytes(b"test")

            names = sorted(path.name for path in DurableImageWriter.stored_images(directory))

        self.assertEqual(names, ["new.webp", "old.jpg", "portable.png"])


if __name__ == "__main__":
    unittest.main()

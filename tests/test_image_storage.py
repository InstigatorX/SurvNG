from __future__ import annotations

import concurrent.futures
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
            expected_mode = 0o600 | (directory.stat().st_mode & 0o044)
            self.assertEqual(path.stat().st_mode & 0o777, expected_mode)

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

    def test_webp_encode_failure_falls_back_to_decodable_jpeg(self) -> None:
        frame = np.full((24, 32, 3), 91, dtype=np.uint8)
        original_encode = cv2.imencode

        def encode(suffix, *args, **kwargs):
            if suffix == ".webp":
                return False, None
            return original_encode(suffix, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DurableImageWriter(ImageStorageConfig())
            with patch("survng.app.image_storage.cv2.imencode", side_effect=encode):
                path = writer.write(Path(tmpdir), "event", frame)

            self.assertIsNotNone(path)
            assert path is not None
            self.assertEqual(path.suffix, ".jpg")
            self.assertIsNotNone(cv2.imread(str(path)))

    def test_filename_stem_cannot_escape_target_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "snapshots"
            writer = DurableImageWriter(ImageStorageConfig())

            path = writer.write(
                directory,
                "../../outside / event",
                np.zeros((8, 8, 3), dtype=np.uint8),
            )

            self.assertIsNotNone(path)
            assert path is not None
            self.assertEqual(path.parent, directory)
            self.assertNotIn("..", path.name)
            self.assertFalse((Path(tmpdir) / "outside").exists())

    def test_file_permissions_follow_target_directory_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "shared"
            directory.mkdir()
            directory.chmod(0o750)
            writer = DurableImageWriter(ImageStorageConfig())

            path = writer.write(
                directory,
                "event",
                np.zeros((8, 8, 3), dtype=np.uint8),
            )

            self.assertIsNotNone(path)
            assert path is not None
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)

    def test_chmod_failure_keeps_safely_created_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DurableImageWriter(ImageStorageConfig())
            with patch("survng.app.image_storage.os.fchmod", side_effect=OSError("unsupported")):
                path = writer.write(
                    Path(tmpdir),
                    "event",
                    np.zeros((8, 8, 3), dtype=np.uint8),
                )

            self.assertIsNotNone(path)
            assert path is not None
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_failed_atomic_replace_preserves_existing_image_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            existing = directory / "event.webp"
            existing.write_bytes(b"existing")
            writer = DurableImageWriter(ImageStorageConfig())

            with patch("survng.app.image_storage.os.replace", side_effect=OSError("busy")):
                path = writer.write(
                    directory,
                    "event",
                    np.zeros((8, 8, 3), dtype=np.uint8),
                )

            self.assertIsNone(path)
            self.assertEqual(existing.read_bytes(), b"existing")
            self.assertEqual(list(directory.glob(".*.tmp")), [])

    def test_parallel_writes_remain_decodable_during_reconfiguration(self) -> None:
        frame = np.full((16, 24, 3), 63, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            writer = DurableImageWriter(ImageStorageConfig())

            def write(index: int) -> Path | None:
                writer.reconfigure(ImageStorageConfig(
                    format="webp" if index % 2 else "jpeg",
                    quality=80 + index % 10,
                ))
                return writer.write(directory, f"event-{index}", frame)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                paths = list(executor.map(write, range(40)))

            self.assertTrue(all(path is not None for path in paths))
            self.assertEqual(len({path.name for path in paths if path is not None}), 40)
            self.assertTrue(all(cv2.imread(str(path)) is not None for path in paths if path))

    def test_stored_images_includes_mixed_supported_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            for name in ("old.jpg", "new.webp", "portable.png", "ignore.mp4"):
                (directory / name).write_bytes(b"test")

            names = sorted(path.name for path in DurableImageWriter.stored_images(directory))

        self.assertEqual(names, ["new.webp", "old.jpg", "portable.png"])


if __name__ == "__main__":
    unittest.main()

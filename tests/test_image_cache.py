from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from survng.app.image_cache import LocalImageCache


class LocalImageCacheTest(unittest.TestCase):
    def test_reuses_cached_result_without_rebuilding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LocalImageCache(Path(tmpdir))
            calls = 0

            def build() -> bytes:
                nonlocal calls
                calls += 1
                return b"jpeg"

            first = cache.get_or_create("events", "same-image", build)
            second = cache.get_or_create("events", "same-image", build)

            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), b"jpeg")
            self.assertEqual(calls, 1)

    def test_concurrent_requests_build_a_key_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LocalImageCache(Path(tmpdir))
            calls = 0
            calls_lock = threading.Lock()
            barrier = threading.Barrier(4)
            results: list[Path] = []

            def build() -> bytes:
                nonlocal calls
                with calls_lock:
                    calls += 1
                time.sleep(0.02)
                return b"jpeg"

            def request() -> None:
                barrier.wait()
                results.append(cache.get_or_create("faces", "same-face", build))

            threads = [threading.Thread(target=request) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(calls, 1)
            self.assertEqual(len(set(results)), 1)

    def test_rejects_empty_namespace_and_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LocalImageCache(Path(tmpdir))
            with self.assertRaises(ValueError):
                cache.get_or_create("../", "image", lambda: b"jpeg")
            with self.assertRaises(ValueError):
                cache.get_or_create("events", "image", lambda: b"")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from survng.app.recorder import Recorder


class RecorderTest(unittest.TestCase):
    def test_recording_rows_use_configured_segment_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            hour_dir = Path(tmpdir) / "recordings" / "back-middle" / "2026-07-11" / "11"
            hour_dir.mkdir(parents=True)
            first = hour_dir / "20260711-113000.mp4"
            second = hour_dir / "20260711-113010.mp4"
            first.write_bytes(b"a")
            second.write_bytes(b"b")

            rows = recorder.recording_rows("back-middle", limit=10)

        self.assertEqual([row["name"] for row in rows], ["20260711-113000.mp4", "20260711-113010.mp4"])
        self.assertEqual(rows[0]["duration_seconds"], 10.0)
        self.assertEqual(rows[1]["duration_seconds"], 10.0)


if __name__ == "__main__":
    unittest.main()

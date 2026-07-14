from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from survng.app.recorder import Recorder


class RecorderTest(unittest.TestCase):
    def test_disabled_camera_is_excluded_from_watchdog_wanted_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            camera = Mock(id="front-door", record=True, record_sub=True, live_stream_url="rtsp://camera/sub")
            recorder.set_camera_enabled(camera.id, False)

            self.assertEqual(recorder._wanted_keys({camera.id: camera}), {})

            recorder.set_camera_enabled(camera.id, True)
            self.assertEqual(
                set(recorder._wanted_keys({camera.id: camera})),
                {(camera.id, "main"), (camera.id, "live")},
            )

    @patch.object(Recorder, "_owned_ffmpeg_recorders", return_value={})
    def test_stopping_one_source_does_not_target_the_other_source(self, owned_recorders) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            recorder.stop("front-door", "main")

        owned_recorders.assert_called_once_with({("front-door", "main")})

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

    def test_recording_rows_can_read_sub_stream_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            hour_dir = Path(tmpdir) / "recordings" / "back-middle" / "live" / "2026-07-11" / "11"
            hour_dir.mkdir(parents=True)
            clip = hour_dir / "20260711-113000.mp4"
            clip.write_bytes(b"sub")

            rows = recorder.recording_rows("back-middle", limit=10, source="live")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "20260711-113000.mp4")
        self.assertEqual(rows[0]["source"], "live")


if __name__ == "__main__":
    unittest.main()

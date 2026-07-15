from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from survng.app.recorder import Recorder


class RecorderTest(unittest.TestCase):
    @staticmethod
    def _row(path: Path, start_epoch: float = 1_784_000_000.0) -> dict:
        return {
            "path": str(path),
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "modified_at": path.stat().st_mtime,
            "start_epoch": start_epoch,
            "duration_seconds": 10.0,
            "end_epoch": start_epoch + 10.0,
            "source": "main",
        }

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

    def test_recording_range_removes_missing_index_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "missing.mp4"
            clip.write_bytes(b"recording")
            row = self._row(clip)
            recorder._store_recording_rows("front-door", "main", [row])
            clip.unlink()

            rows = recorder.recording_rows_between(
                "front-door",
                row["start_epoch"] - 1,
                row["end_epoch"] + 1,
            )
            with recorder._index_connection() as connection:
                indexed = connection.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]

        self.assertEqual(rows, [])
        self.assertEqual(indexed, 0)

    def test_incremental_pruner_removes_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "retained.mp4"
            clip.write_bytes(b"recording")
            recorder._store_recording_rows("front-door", "main", [self._row(clip)])
            clip.unlink()

            removed = recorder._prune_missing_index_rows()

        self.assertEqual(removed, 1)

    def test_validation_batch_checks_unvalidated_startup_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "startup.mp4"
            clip.write_bytes(b"recording")
            recorder._store_recording_rows("front-door", "main", [self._row(clip)])

            with patch.object(recorder, "_probe_recording", return_value=(9.5, "")) as probe:
                validated = recorder._validate_index_batch()
            with recorder._index_connection() as connection:
                indexed = dict(connection.execute(
                    "SELECT duration_seconds, end_epoch, playable, validated FROM recordings WHERE path = ?",
                    (str(clip),),
                ).fetchone())

        self.assertEqual(validated, 1)
        probe.assert_called_once_with(clip)
        self.assertEqual(indexed["duration_seconds"], 9.5)
        self.assertEqual(indexed["playable"], 1)
        self.assertEqual(indexed["validated"], 1)


if __name__ == "__main__":
    unittest.main()

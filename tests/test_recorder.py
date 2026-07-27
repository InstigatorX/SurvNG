from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from survng.app.baichuan_native import ffmpeg_input_args, ffmpeg_timestamp_repair_args
from survng.app.config import CameraConfig
from survng.app.recorder import Recorder


class RecorderTest(unittest.TestCase):
    @staticmethod
    def _age_file(path: Path, seconds: float = 30.0) -> None:
        old_epoch = time.time() - seconds
        os.utime(path, (old_epoch, old_epoch))

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

    def test_rtsp_recording_generates_missing_pts_and_discards_large_regressions(self) -> None:
        camera = CameraConfig(
            id="upper-garage",
            name="Upper Garage",
            stream_url="rtsp://camera/main",
            live_stream_url="rtsp://camera/sub",
        )

        self.assertEqual(
            ffmpeg_input_args(camera, "live"),
            [
                "-fflags",
                "+genpts",
                "-dts_error_threshold",
                "10",
                "-rtsp_transport",
                "tcp",
                "-i",
                "rtsp://camera/sub",
            ],
        )
        repair_args = ffmpeg_timestamp_repair_args(camera)
        self.assertEqual(repair_args[0], "-bsf:v")
        self.assertIn("setts=pts=", repair_args[1])
        self.assertIn("PREV_OUTPTS", repair_args[1])
        self.assertIn("PREV_OUTDTS", repair_args[1])

    def test_native_recording_allows_ffmpeg_to_probe_h264_or_h265(self) -> None:
        camera = CameraConfig.model_validate(
            {
                "id": "gate",
                "name": "Gate",
                "stream_url": "reolink://admin:password@camera.local?channel=0",
            }
        )

        args = ffmpeg_input_args(camera, "main")

        self.assertEqual(args[-2:], ["-i", "pipe:0"])
        self.assertNotIn("h264", args)

    def test_persistent_non_monotonic_dts_restarts_recorder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            process = Mock(
                pid=1234,
                stderr=iter(["Non-monotonic DTS\n"] * 12),
            )
            with patch.object(recorder, "_kill_pid") as kill_pid:
                recorder._monitor_ffmpeg_stderr(("upper-garage", "live"), process)

        kill_pid.assert_called_once_with(1234)

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

    def test_start_failure_is_nonfatal_and_backed_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            camera = CameraConfig(
                id="gate",
                name="Gate",
                stream_url="rtsp://camera/main",
            )
            with patch.object(recorder, "_ensure_recording_dirs", side_effect=OSError(28, "No space left")) as ensure_dirs:
                with patch("survng.app.recorder.subprocess.Popen") as popen:
                    recorder.start(camera, "main")
                    recorder.start(camera, "main")

            self.assertEqual(ensure_dirs.call_count, 1)
            popen.assert_not_called()
            self.assertNotIn((camera.id, "main"), recorder.processes)

    def test_recorder_maps_only_primary_media_and_normalizes_audio_for_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
            with patch(
                "survng.app.recorder.subprocess.Popen",
                side_effect=RuntimeError("capture command"),
            ) as popen:
                recorder.start(camera, "main")

        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-map") + 1], "0:v:0")
        second_map = command.index("-map", command.index("-map") + 1)
        self.assertEqual(command[second_map + 1], "0:a:0?")
        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertEqual(command[command.index("-c:a") + 1], "aac")

    @patch.object(Recorder, "_owned_ffmpeg_recorders", return_value={})
    def test_stop_during_start_cancels_new_recorder_before_registration(self, _owned_recorders) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
            process = Mock(pid=4321, stderr=iter(()))
            process.poll.return_value = None

            def cancel_start(*_args):
                recorder.stop(camera.id, "main")
                return None

            with (
                patch("survng.app.recorder.subprocess.Popen", return_value=process),
                patch("survng.app.recorder.start_ffmpeg_pipe", side_effect=cancel_start),
                patch.object(recorder, "_kill_pid") as kill_pid,
            ):
                recorder.start(camera, "main")

        self.assertNotIn((camera.id, "main"), recorder.processes)
        kill_pid.assert_called_once_with(4321)

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

    def test_timezone_qualified_segment_names_are_unambiguous_across_dst(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            daylight = Path(tmpdir) / "20261101-013000-0400.mp4"
            standard = Path(tmpdir) / "20261101-013000-0500.mp4"

            daylight_epoch = recorder.recording_start_epoch(daylight)
            standard_epoch = recorder.recording_start_epoch(standard)

        self.assertIsNotNone(daylight_epoch)
        self.assertIsNotNone(standard_epoch)
        self.assertEqual(float(standard_epoch) - float(daylight_epoch), 3600.0)

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

    def test_recording_range_can_discover_legacy_main_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            hour_dir = Path(tmpdir) / "recordings" / "back-middle" / "2026-07-11" / "11"
            hour_dir.mkdir(parents=True)
            clip = hour_dir / "20260711-113000.mp4"
            clip.write_bytes(b"x" * 2048)
            start = recorder.recording_start_epoch(clip)
            self.assertIsNotNone(start)

            rows = recorder.recording_rows_between("back-middle", float(start) - 1, float(start) + 11)

        self.assertEqual([row["name"] for row in rows], [clip.name])

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

    def test_recording_range_discovers_files_missing_from_partial_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            hour_dir = Path(tmpdir) / "recordings" / "front-door" / "main" / "2026-07-11" / "11"
            hour_dir.mkdir(parents=True)
            first = hour_dir / "20260711-113000.mp4"
            second = hour_dir / "20260711-113010.mp4"
            first.write_bytes(b"x" * 2048)
            second.write_bytes(b"y" * 2048)
            start = recorder.recording_start_epoch(first)
            self.assertIsNotNone(start)
            recorder._store_recording_rows("front-door", "main", [self._row(first, float(start))])

            rows = recorder.recording_rows_between(
                "front-door",
                float(start) - 1,
                float(start) + 21,
            )

        self.assertEqual([row["name"] for row in rows], [first.name, second.name])

    def test_active_recorder_does_not_hide_last_historical_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            hour_dir = Path(tmpdir) / "recordings" / "front-door" / "main" / "2020-01-02" / "03"
            hour_dir.mkdir(parents=True)
            clips = [hour_dir / "20200102-030000.mp4", hour_dir / "20200102-030010.mp4"]
            for clip in clips:
                clip.write_bytes(b"x")
            process = Mock()
            process.poll.return_value = None
            recorder.processes[("front-door", "main")] = (process, None, Mock(), Mock())

            rows = recorder._recording_rows_for_files("front-door", "main", clips)

        self.assertEqual([row["name"] for row in rows], [clip.name for clip in clips])

    def test_active_recorder_hides_segment_started_before_hour_rollover_until_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            now_epoch = time.time()
            clip_start = now_epoch - 1
            previous_hour = datetime.fromtimestamp(now_epoch - 3600)
            hour_dir = (
                Path(tmpdir) / "recordings" / "front-door" / "live"
                / previous_hour.strftime("%Y-%m-%d") / previous_hour.strftime("%H")
            )
            hour_dir.mkdir(parents=True)
            clip = hour_dir / datetime.fromtimestamp(clip_start).strftime("%Y%m%d-%H%M%S.mp4")
            clip.write_bytes(b"still growing")
            process = Mock()
            process.poll.return_value = None
            recorder.processes[("front-door", "live")] = (process, None, Mock(), Mock())

            with patch("survng.app.recorder.time.time", return_value=now_epoch):
                rows = recorder._recording_rows_for_files("front-door", "live", [clip])

        self.assertEqual(rows, [])

    def test_recording_availability_merges_segments_but_preserves_gaps(self) -> None:
        rows = [
            {"start_epoch": 100.0, "end_epoch": 110.0},
            {"start_epoch": 110.1, "end_epoch": 120.0},
            {"start_epoch": 140.0, "end_epoch": 150.0},
        ]

        ranges = Recorder._merge_availability_rows(rows, 105.0, 145.0)

        self.assertEqual(ranges, [
            {
                "start_epoch": 105.0,
                "end_epoch": 120.0,
                "duration_seconds": 15.0,
                "segment_count": 2,
            },
            {
                "start_epoch": 140.0,
                "end_epoch": 145.0,
                "duration_seconds": 5.0,
                "segment_count": 1,
            },
        ])

    def test_recording_availability_returns_compact_index_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            rows = []
            for index, start_epoch in enumerate((1_784_000_000.0, 1_784_000_012.0, 1_784_000_040.0)):
                clip = Path(tmpdir) / f"clip-{index}.mp4"
                clip.write_bytes(b"x" * 2048)
                rows.append(self._row(clip, start_epoch=start_epoch))
            recorder._store_recording_rows("front-door", "main", rows)

            availability = recorder.recording_availability_between(
                "front-door",
                1_784_000_000.0,
                1_784_000_060.0,
            )

        self.assertEqual(availability["segment_count"], 3)
        self.assertEqual(len(availability["ranges"]), 2)
        self.assertEqual(availability["ranges"][0]["segment_count"], 2)
        self.assertEqual(availability["ranges"][1]["segment_count"], 1)
        self.assertNotIn("path", availability["ranges"][0])

    def test_refresh_recording_edge_indexes_only_completed_recent_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            now = datetime.now()
            hour_dir = (
                Path(tmpdir) / "recordings" / "front-door" / "main"
                / now.strftime("%Y-%m-%d") / now.strftime("%H")
            )
            hour_dir.mkdir(parents=True)
            starts = [now.timestamp() - 40, now.timestamp() - 20, now.timestamp() - 10]
            for start_epoch in starts:
                clip = hour_dir / datetime.fromtimestamp(start_epoch).strftime("%Y%m%d-%H%M%S.mp4")
                clip.write_bytes(b"x" * 2048)
            process = Mock()
            process.poll.return_value = None
            recorder.processes[("front-door", "main")] = (process, None, Mock(), Mock())

            indexed = recorder.refresh_recording_edge(
                "front-door",
                "main",
                now.timestamp() - 20,
            )
            availability = recorder.recording_availability_between(
                "front-door",
                now.timestamp() - 30,
                now.timestamp(),
            )

        self.assertEqual(indexed, 1)
        self.assertEqual(availability["segment_count"], 1)
        self.assertLessEqual(availability["ranges"][0]["end_epoch"], starts[-1])

    def test_index_loop_starts_with_recent_discovery_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            recorder._index_stop.set()
            with patch.object(recorder, "refresh_recording_index") as refresh:
                recorder._recording_index_loop({})

        refresh.assert_called_once_with({}, full=False, run_maintenance=False)

    def test_recent_index_discovery_defers_media_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            now = datetime.now()
            hour_dir = (
                Path(tmpdir) / "recordings" / "front-door" / "main"
                / now.strftime("%Y-%m-%d") / now.strftime("%H")
            )
            hour_dir.mkdir(parents=True)
            clip = hour_dir / now.strftime("%Y%m%d-%H%M%S.mp4")
            clip.write_bytes(b"x" * 2048)
            camera = Mock(id="front-door", record=True, record_sub=False, live_stream_url="")

            with patch.object(recorder, "_probe_recording") as probe:
                recorder.refresh_recording_index(
                    {camera.id: camera},
                    full=False,
                    run_maintenance=False,
                )
            with recorder._index_connection() as connection:
                indexed = dict(connection.execute(
                    "SELECT path, validated FROM recordings WHERE path = ?",
                    (str(clip),),
                ).fetchone())

        probe.assert_not_called()
        self.assertEqual(indexed["path"], str(clip))
        self.assertEqual(indexed["validated"], 0)

    def test_source_reconciliation_discovers_history_and_prunes_stale_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            hour_dir = Path(tmpdir) / "recordings" / "front-door" / "main" / "2026-01-02" / "03"
            hour_dir.mkdir(parents=True)
            clips = [
                hour_dir / "20260102-030000.mp4",
                hour_dir / "20260102-030010.mp4",
            ]
            for clip in clips:
                clip.write_bytes(b"x" * 2048)
            stale = Path(tmpdir) / "stale.mp4"
            stale.write_bytes(b"stale")
            recorder._store_recording_rows("front-door", "main", [self._row(stale)])
            stale.unlink()

            recorder._reconcile_recording_source("front-door", "main")
            with recorder._index_connection() as connection:
                paths = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT path FROM recordings WHERE camera_id = ? AND source = ?",
                        ("front-door", "main"),
                    )
                }

        self.assertEqual(paths, {str(clip) for clip in clips})

    def test_incremental_pruner_removes_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "retained.mp4"
            clip.write_bytes(b"recording")
            recorder._store_recording_rows("front-door", "main", [self._row(clip)])
            clip.unlink()

            removed = recorder._prune_missing_index_rows()

        self.assertEqual(removed, 1)

    def test_index_can_be_stored_outside_recording_storage(self) -> None:
        with tempfile.TemporaryDirectory() as storage, tempfile.TemporaryDirectory() as index:
            recorder = Recorder(
                "ffmpeg",
                Path(storage),
                segment_seconds=10,
                index_dir=Path(index),
            )

            self.assertEqual(recorder.recordings_dir, Path(storage) / "recordings")
            self.assertEqual(recorder.index_path, Path(index) / "recordings.sqlite3")
            self.assertTrue(recorder.index_path.is_file())

    def test_periodic_maintenance_does_not_run_full_reconcile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            with (
                patch.object(recorder, "_prune_missing_index_rows", side_effect=lambda limit: recorder._index_stop.set()),
                patch.object(recorder, "_validate_index_batch"),
                patch.object(recorder, "_backfill_stream_fingerprints"),
                patch.object(recorder, "_reconcile_recording_source") as reconcile,
                patch.object(recorder._index_stop, "wait", return_value=False),
            ):
                recorder._recording_index_maintenance_loop({})

        reconcile.assert_not_called()

    def test_validation_batch_checks_unvalidated_startup_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "startup.mp4"
            clip.write_bytes(b"recording")
            self._age_file(clip)
            row = self._row(clip)
            recorder._store_recording_rows("front-door", "main", [row])
            recorder.queue_recording_validation([row])

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

    def test_validation_defers_a_segment_until_ffmpeg_can_finalize_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            now_epoch = time.time()
            clip = Path(tmpdir) / datetime.fromtimestamp(now_epoch - 1).strftime("%Y%m%d-%H%M%S.mp4")
            clip.write_bytes(b"unfinished mp4")
            row = self._row(clip, start_epoch=now_epoch - 1)
            recorder._store_recording_rows("front-door", "main", [row])
            recorder.queue_recording_validation([row])

            with (
                patch("survng.app.recorder.time.time", return_value=now_epoch),
                patch.object(recorder, "_probe_recording") as probe,
            ):
                validated = recorder._validate_index_batch(limit=1)
            with recorder._index_connection() as connection:
                deferred = dict(connection.execute(
                    "SELECT playable, validated, health_error FROM recordings WHERE path = ?",
                    (str(clip),),
                ).fetchone())
            deferred_queued = str(clip) in recorder._validation_pending_set

            old_epoch = now_epoch - 30
            os.utime(clip, (old_epoch, old_epoch))
            with (
                patch("survng.app.recorder.time.time", return_value=now_epoch + 15),
                patch.object(recorder, "_probe_recording", return_value=(10.0, "")) as retry_probe,
            ):
                retried = recorder._validate_index_batch(limit=1)
            with recorder._index_connection() as connection:
                restored = dict(connection.execute(
                    "SELECT playable, validated, health_error FROM recordings WHERE path = ?",
                    (str(clip),),
                ).fetchone())

        self.assertEqual(validated, 0)
        probe.assert_not_called()
        self.assertEqual(deferred["playable"], 1)
        self.assertEqual(deferred["validated"], 0)
        self.assertEqual(deferred["health_error"], "")
        self.assertTrue(deferred_queued)
        self.assertEqual(retried, 1)
        retry_probe.assert_called_once_with(clip)
        self.assertEqual(restored["playable"], 1)
        self.assertEqual(restored["validated"], 1)
        self.assertEqual(restored["health_error"], "")

    def test_transient_playback_failure_is_revalidated_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "recoverable.mp4"
            clip.write_bytes(b"recording")
            self._age_file(clip)
            row = self._row(clip)
            row["validated"] = True
            recorder._store_recording_rows("front-door", "main", [row])

            recorder.schedule_revalidation(clip, "temporary remux failure")
            with patch.object(recorder, "_probe_recording", return_value=(9.75, "")):
                recorder._validate_index_batch(limit=1)
            with recorder._index_connection() as connection:
                indexed = dict(connection.execute(
                    "SELECT playable, validated, health_error FROM recordings WHERE path = ?",
                    (str(clip),),
                ).fetchone())

        self.assertEqual(indexed["playable"], 1)
        self.assertEqual(indexed["validated"], 1)
        self.assertEqual(indexed["health_error"], "")

    def test_fingerprint_backfill_is_independent_of_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "already-validated.mp4"
            clip.write_bytes(b"recording")
            row = self._row(clip)
            row["validated"] = True
            recorder._store_recording_rows("front-door", "main", [row])
            recorder.queue_stream_fingerprints(recorder.recording_rows_between(
                "front-door",
                row["start_epoch"] - 1,
                row["end_epoch"] + 1,
            ))

            with patch("survng.app.recorder.mp4_stream_fingerprint", return_value="stream-v1"):
                updated = recorder._backfill_stream_fingerprints(limit=1)
            with recorder._index_connection() as connection:
                indexed = dict(connection.execute(
                    "SELECT stream_fingerprint, fingerprint_checked, validated FROM recordings WHERE path = ?",
                    (str(clip),),
                ).fetchone())

        self.assertEqual(updated, 1)
        self.assertEqual(indexed["stream_fingerprint"], "stream-v1")
        self.assertEqual(indexed["fingerprint_checked"], 1)
        self.assertEqual(indexed["validated"], 1)

    def test_maintenance_discovers_unqueued_validation_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "unqueued.mp4"
            clip.write_bytes(b"recording")
            self._age_file(clip)
            row = self._row(clip)
            recorder._store_recording_rows("front-door", "main", [row])

            with (
                patch.object(recorder, "_probe_recording", return_value=(9.0, "")),
                patch("survng.app.recorder.mp4_stream_fingerprint", return_value="stream-v2"),
            ):
                self.assertEqual(recorder._validate_index_batch(limit=1), 1)
            with recorder._index_connection() as connection:
                indexed = dict(connection.execute(
                    "SELECT validated, fingerprint_checked, stream_fingerprint FROM recordings WHERE path = ?",
                    (str(clip),),
                ).fetchone())

        self.assertEqual(indexed["validated"], 1)
        self.assertEqual(indexed["fingerprint_checked"], 1)
        self.assertEqual(indexed["stream_fingerprint"], "stream-v2")

    def test_maintenance_discovers_unqueued_fingerprint_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "fingerprint.mp4"
            clip.write_bytes(b"recording")
            row = self._row(clip)
            row["validated"] = True
            recorder._store_recording_rows("front-door", "main", [row])

            with patch("survng.app.recorder.mp4_stream_fingerprint", return_value="stream-v3"):
                self.assertEqual(recorder._backfill_stream_fingerprints(limit=1), 1)
            with recorder._index_connection() as connection:
                indexed = dict(connection.execute(
                    "SELECT validated, fingerprint_checked, stream_fingerprint FROM recordings WHERE path = ?",
                    (str(clip),),
                ).fetchone())

        self.assertEqual(indexed["validated"], 1)
        self.assertEqual(indexed["fingerprint_checked"], 1)
        self.assertEqual(indexed["stream_fingerprint"], "stream-v3")


if __name__ == "__main__":
    unittest.main()

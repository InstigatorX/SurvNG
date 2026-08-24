from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from collections import deque
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from survng.app.config import CameraConfig, MediaStorageConfig, MediaStorageLocationConfig
from survng.app.ffmpeg_input import ffmpeg_input_args, ffmpeg_timestamp_repair_args
from survng.app.go2rtc import Go2RtcError
from survng.app.media_storage import MediaStorageRegistry
from survng.app.recorder import AudioStreamInfo, Recorder


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

    def test_multiple_recording_locations_are_indexed_and_searched_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            registry = MediaStorageRegistry(root / "metadata", MediaStorageConfig(locations=[
                MediaStorageLocationConfig(id="one", path=str(first), roles=["recordings"], reserve_percent=0),
                MediaStorageLocationConfig(id="two", path=str(second), roles=["recordings"], reserve_percent=0),
            ]))
            recorder = Recorder("ffmpeg", root / "metadata", media_storage=registry)
            paths = []
            for location, hour in ((first, "00"), (second, "01")):
                path = location / "recordings" / "gate" / "main" / "2026-08-13" / hour / f"20260813-{hour}0000+0000.mp4"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"recording")
                paths.append(path)

            recorder.refresh_recording_index({"gate": CameraConfig(id="gate", name="Gate", stream_url="rtsp://gate")}, full=True)
            rows = recorder.recording_rows("gate", limit=10)

            self.assertEqual({row["path"] for row in rows}, {str(path) for path in paths})
            self.assertEqual({row["location_id"] for row in rows}, {"one", "two"})

    def test_recording_location_backfill_is_bounded_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            registry = MediaStorageRegistry(root / "metadata", MediaStorageConfig(locations=[
                MediaStorageLocationConfig(id="one", path=str(first), roles=["recordings"], reserve_percent=0),
                MediaStorageLocationConfig(id="two", path=str(second), roles=["recordings"], reserve_percent=0),
            ]))
            recorder = Recorder("ffmpeg", root / "metadata", media_storage=registry)
            for index in range(5):
                location = first if index % 2 == 0 else second
                path = location / "recordings" / "gate" / "main" / f"clip-{index}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"recording")
                recorder._store_recording_rows(
                    "gate",
                    "main",
                    [self._row(path, start_epoch=1_784_000_000.0 + index * 10)],
                )
            with recorder._index_connection() as connection:
                connection.execute("UPDATE recordings SET location_id = 'default'")

            with patch(
                "survng.app.media_storage.shutil.disk_usage",
                side_effect=AssertionError("location migration must not probe storage health"),
            ):
                self.assertEqual(recorder._backfill_recording_location_id_batch(limit=2), 2)
            with recorder._index_connection() as connection:
                first_pass = connection.execute(
                    "SELECT location_id FROM recordings ORDER BY rowid"
                ).fetchall()
            self.assertEqual(
                [str(row["location_id"]) for row in first_pass],
                ["one", "two", "default", "default", "default"],
            )

            self.assertEqual(recorder._backfill_recording_location_id_batch(limit=2), 2)
            self.assertEqual(recorder._backfill_recording_location_id_batch(limit=2), 1)
            self.assertEqual(recorder._backfill_recording_location_id_batch(limit=2), 0)
            with recorder._index_connection() as connection:
                final_rows = connection.execute(
                    "SELECT path, location_id FROM recordings ORDER BY rowid"
                ).fetchall()
                cursor = connection.execute(
                    "SELECT value FROM recording_index_metadata "
                    "WHERE key = 'recording_location_backfill_cursor'"
                ).fetchone()["value"]
            self.assertEqual(
                [str(row["location_id"]) for row in final_rows],
                ["one", "two", "one", "two", "one"],
            )
            self.assertEqual(cursor, "-1")

    def test_recording_location_backfill_restarts_when_topology_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            later = root / "later"
            first.mkdir()
            later.mkdir()
            initial_registry = MediaStorageRegistry(root / "metadata", MediaStorageConfig(locations=[
                MediaStorageLocationConfig(id="one", path=str(first), roles=["recordings"]),
            ]))
            recorder = Recorder("ffmpeg", root / "metadata", media_storage=initial_registry)
            unmatched = later / "recordings" / "gate" / "main" / "clip.mp4"
            unmatched.parent.mkdir(parents=True)
            unmatched.write_bytes(b"recording")
            recorder._store_recording_rows("gate", "main", [self._row(unmatched)])
            with recorder._index_connection() as connection:
                connection.execute("UPDATE recordings SET location_id = 'default'")
            self.assertEqual(recorder._backfill_recording_location_id_batch(limit=10), 1)
            self.assertEqual(recorder._backfill_recording_location_id_batch(limit=10), 0)

            expanded_registry = MediaStorageRegistry(root / "metadata", MediaStorageConfig(locations=[
                MediaStorageLocationConfig(id="one", path=str(first), roles=["recordings"]),
                MediaStorageLocationConfig(id="later", path=str(later), roles=["recordings"]),
            ]))
            recorder.media_storage = expanded_registry
            self.assertEqual(recorder._backfill_recording_location_id_batch(limit=10), 1)
            with recorder._index_connection() as connection:
                location_id = connection.execute(
                    "SELECT location_id FROM recordings WHERE path = ?",
                    (str(unmatched),),
                ).fetchone()["location_id"]
            self.assertEqual(location_id, "later")

    def test_multi_location_indexer_start_does_not_run_location_backfill_inline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            registry = MediaStorageRegistry(root / "metadata", MediaStorageConfig(locations=[
                MediaStorageLocationConfig(id="one", path=str(first), roles=["recordings"]),
                MediaStorageLocationConfig(id="two", path=str(second), roles=["recordings"]),
            ]))
            recorder = Recorder("ffmpeg", root / "metadata", media_storage=registry)
            with (
                patch.object(recorder, "_backfill_recording_location_id_batch") as backfill,
                patch.object(recorder.retention, "start"),
                patch("survng.app.recording_process.recorder.threading.Thread.start"),
            ):
                recorder.start_indexer([])

            backfill.assert_not_called()

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

    def test_persistent_non_monotonic_dts_schedules_epoch_rollover(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            process = Mock(
                pid=1234,
                stderr=iter(["Non-monotonic DTS\n"] * 12),
            )
            recorder._monitor_ffmpeg_stderr(("upper-garage", "live"), process)

            health = recorder.timestamp_health()[("upper-garage", "live")]
            self.assertTrue(health["rollover_pending"])
            self.assertEqual(health["discontinuities"], 1)
            self.assertEqual(health["last_reason"], "non_monotonic_dts")
            self.assertTrue(recorder._watchdog_wake.is_set())

    def test_ffmpeg_8_invalid_timestamp_pairs_are_coalesced_into_one_rollover(self) -> None:
        lines = []
        for index in range(20):
            lines.extend([
                f"[vist#0:0/h264 @ 0x1] DTS {index}, next:999 st:0 invalid dropping\n",
                f"[vist#0:0/h264 @ 0x1] PTS {index}, next:999 invalid dropping st:0\n",
            ])
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            process = Mock(pid=4321, stderr=iter(lines))

            with self.assertLogs("survng.app.recorder", level="WARNING") as captured:
                recorder._monitor_ffmpeg_stderr(("gate", "main"), process)

            health = recorder.timestamp_health()[("gate", "main")]

        self.assertEqual(health["discontinuities"], 1)
        self.assertEqual(health["last_reason"], "invalid_dts")
        self.assertEqual(
            sum("scheduling an epoch rollover" in line for line in captured.output),
            1,
        )
        self.assertFalse(any("invalid dropping" in line for line in captured.output))

    def test_ffmpeg_8_pts_only_discontinuity_still_schedules_recovery(self) -> None:
        lines = [
            f"[vist#0:0/h264 @ 0x1] PTS {index}, next:999 invalid dropping st:0\n"
            for index in range(12)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            recorder._monitor_ffmpeg_stderr(
                ("gate", "main"),
                Mock(pid=4321, stderr=iter(lines)),
            )
            health = recorder.timestamp_health()[("gate", "main")]

        self.assertTrue(health["rollover_pending"])
        self.assertEqual(health["last_reason"], "invalid_pts")

    def test_reconcile_applies_timestamp_rollover_to_only_the_expected_process(self) -> None:
        camera = CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://camera/main",
        )
        key = ("gate", "main")
        old_process = Mock(pid=1234)
        old_process.poll.return_value = None
        replacement = Mock(pid=5678)
        replacement.poll.return_value = None
        item = (old_process, Mock(), Mock())
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            recorder.processes[key] = item
            recorder._request_timestamp_rollover(key, old_process.pid, "invalid_dts")

            def stop(_camera_id, _source):
                recorder.processes.pop(key, None)

            def start(_camera, _source):
                recorder.processes[key] = (replacement, Mock(), Mock())

            with patch.object(recorder, "stop", side_effect=stop) as stop_recorder, patch.object(
                recorder,
                "start",
                side_effect=start,
            ) as start_recorder, patch.object(recorder, "cleanup_duplicate_recorders"):
                recorder.reconcile({camera.id: camera})

            health = recorder.timestamp_health()[key]

        stop_recorder.assert_called_once_with("gate", "main")
        start_recorder.assert_called_once_with(camera, "main")
        self.assertEqual(health["epoch_rollovers"], 1)
        self.assertFalse(health["rollover_pending"])

    def test_timestamp_rollovers_are_rate_limited_without_repeating_log_flood(self) -> None:
        key = ("gate", "main")
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            recorder._timestamp_rollover_history[key] = deque([time.monotonic()] * 3)

            with self.assertLogs("survng.app.recorder", level="ERROR") as captured:
                requested = recorder._request_timestamp_rollover(key, 1234, "invalid_dts")
            health = recorder.timestamp_health()[key]

        self.assertTrue(requested)
        self.assertEqual(health["rate_limited"], 1)
        self.assertFalse(health["rollover_pending"])
        self.assertEqual(sum("rate-limited" in line for line in captured.output), 1)

    def test_stale_timestamp_rollover_cannot_restart_a_replacement_process(self) -> None:
        camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
        key = ("gate", "main")
        replacement = Mock(pid=5678)
        replacement.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            recorder.processes[key] = (replacement, Mock(), Mock())
            recorder._timestamp_rollover_pending[key] = (1234, "invalid_dts")

            with patch.object(recorder, "stop") as stop_recorder, patch.object(
                recorder,
                "start",
            ) as start_recorder, patch.object(recorder, "cleanup_duplicate_recorders"):
                recorder.reconcile({camera.id: camera})

        stop_recorder.assert_not_called()
        start_recorder.assert_not_called()

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
                with patch("survng.app.recording_process.recorder.subprocess.Popen") as popen:
                    recorder.start(camera, "main")
                    recorder.start(camera, "main")

            self.assertEqual(ensure_dirs.call_count, 1)
            popen.assert_not_called()
            self.assertNotIn((camera.id, "main"), recorder.processes)

    def test_recorder_copies_mp4_compatible_aac_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
            with patch(
                "survng.app.recording_process.recorder.subprocess.Popen",
                side_effect=RuntimeError("capture command"),
            ) as popen, patch.object(
                recorder,
                "_probe_audio_stream",
                return_value=AudioStreamInfo(known=True, codec="aac", sample_rate=8000),
            ):
                recorder.start(camera, "main")

        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-map") + 1], "0:v:0")
        second_map = command.index("-map", command.index("-map") + 1)
        self.assertEqual(command[second_map + 1], "0:a:0?")
        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertEqual(command[command.index("-c:a") + 1], "copy")
        self.assertNotIn("-b:a", command)

    def test_recorder_transcodes_incompatible_low_rate_audio_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
            with patch(
                "survng.app.recording_process.recorder.subprocess.Popen",
                side_effect=RuntimeError("capture command"),
            ) as popen, patch.object(
                recorder,
                "_probe_audio_stream",
                return_value=AudioStreamInfo(known=True, codec="pcm_mulaw", sample_rate=8000),
            ):
                recorder.start(camera, "main")

        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertEqual(command[command.index("-b:a") + 1], "48k")

    def test_recorder_omits_audio_options_when_source_has_no_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
            with patch(
                "survng.app.recording_process.recorder.subprocess.Popen",
                side_effect=RuntimeError("capture command"),
            ) as popen, patch.object(
                recorder,
                "_probe_audio_stream",
                return_value=AudioStreamInfo(known=True),
            ):
                recorder.start(camera, "main")

        command = popen.call_args.args[0]
        self.assertNotIn("0:a:0?", command)
        self.assertNotIn("-c:a", command)

    def test_audio_probe_recovers_from_transient_go2rtc_metadata_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
            with (
                patch.object(recorder.go2rtc, "stream", return_value=Mock(host="camera")),
                patch.object(
                    recorder.go2rtc,
                    "audio_stream_info",
                    side_effect=[
                        Go2RtcError("producer is initializing"),
                        {"available": True, "codec": "aac", "sample_rate": 8000},
                    ],
                ) as audio_info,
                patch("survng.app.recording_process.recorder.time.sleep"),
            ):
                info = recorder._probe_audio_stream(camera, "main")

        self.assertEqual(info, AudioStreamInfo(known=True, codec="aac", sample_rate=8000))
        self.assertEqual(audio_info.call_count, 2)

    def test_unknown_url_audio_uses_recoverable_optional_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
            with patch.object(
                recorder,
                "_probe_audio_stream",
                return_value=AudioStreamInfo(known=False),
            ) as probe:
                first = recorder._audio_output_args(camera, "main")
                second = recorder._audio_output_args(camera, "main")

        self.assertEqual(first, ["-map", "0:a:0?", "-c:a", "copy"])
        self.assertEqual(second, first)
        self.assertEqual(probe.call_count, 2)

    @patch.object(Recorder, "_owned_ffmpeg_recorders", return_value={})
    def test_stop_during_start_cancels_new_recorder_before_registration(self, _owned_recorders) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
            process = Mock(pid=4321, stderr=iter(()))
            process.poll.return_value = None

            def popen_and_cancel(*_args, **_kwargs):
                recorder.stop(camera.id, "main")
                return process

            with (
                patch(
                    "survng.app.recording_process.recorder.subprocess.Popen",
                    side_effect=popen_and_cancel,
                ),
                patch.object(
                    recorder,
                    "_probe_audio_stream",
                    return_value=AudioStreamInfo(known=True, codec="aac", sample_rate=8000),
                ),
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

    def test_recording_rows_use_configured_segment_duration_from_local_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            hour_dir = Path(tmpdir) / "recordings" / "back-middle" / "2026-07-11" / "11"
            hour_dir.mkdir(parents=True)
            first = hour_dir / "20260711-113000.mp4"
            second = hour_dir / "20260711-113010.mp4"
            first.write_bytes(b"a")
            second.write_bytes(b"b")

            indexed = recorder._recording_rows_for_files("back-middle", "main", [first, second])
            recorder._store_recording_rows("back-middle", "main", indexed)
            with patch.object(
                recorder,
                "recent_files",
                side_effect=AssertionError("media storage scan"),
            ):
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

            indexed = recorder._recording_rows_for_files("back-middle", "live", [clip])
            recorder._store_recording_rows("back-middle", "live", indexed)
            rows = recorder.recording_rows("back-middle", limit=10, source="live")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "20260711-113000.mp4")
        self.assertEqual(rows[0]["source"], "live")

    def test_watchdog_survives_transient_reconciliation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            calls = 0

            def reconcile(_camera_map) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError("temporary process listing failure")
                recorder._watchdog_stop.set()

            with patch.object(recorder, "reconcile", side_effect=reconcile):
                recorder._watchdog({}, interval=0.001)

        self.assertEqual(calls, 2)

    def test_stop_item_tolerates_process_that_does_not_reap_after_sigkill(self) -> None:
        process = Mock(pid=4321)
        process.poll.return_value = None
        process.wait.side_effect = [
            __import__("subprocess").TimeoutExpired("ffmpeg", 5),
            __import__("subprocess").TimeoutExpired("ffmpeg", 5),
        ]
        stop_event = Mock()
        keeper = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            with patch("survng.app.recording_process.recorder.os.killpg"):
                recorder._stop_item((process, stop_event, keeper))

        stop_event.set.assert_called_once_with()
        keeper.join.assert_called_once_with(timeout=1)

    def test_stop_item_reaps_ffmpeg_after_sigterm(self) -> None:
        process = Mock(pid=4321)
        process.poll.return_value = None
        process.wait.return_value = 0
        stop_event = Mock()
        keeper = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            with patch("survng.app.recording_process.recorder.os.killpg") as killpg:
                recorder._stop_item((process, stop_event, keeper))

        killpg.assert_called_once_with(4321, __import__("signal").SIGTERM)
        process.wait.assert_called_once_with(timeout=5)
        stop_event.set.assert_called_once_with()
        keeper.join.assert_called_once_with(timeout=1)

    def test_status_cleans_up_stopped_process_without_holding_state_lock(self) -> None:
        process = Mock()
        process.poll.return_value = 1
        stop_event = Mock()
        keeper = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            recorder.processes[("gate", "main")] = (process, stop_event, keeper)

            status = recorder.status()

        self.assertEqual(status, {})
        stop_event.set.assert_called_once_with()
        keeper.join.assert_called_once_with(timeout=1)

    def test_recording_at_uses_local_index_without_scanning_media_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "recording.mp4"
            clip.write_bytes(b"recording")
            row = self._row(clip)
            recorder._store_recording_rows("gate", "main", [row])

            with patch.object(
                recorder,
                "recent_files",
                side_effect=AssertionError("media storage scan"),
            ):
                result = recorder.recording_at("gate", row["start_epoch"] + 5)

        self.assertIsNotNone(result)
        self.assertEqual(result["path"], str(clip))

    def test_near_live_recording_miss_requests_background_edge_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            event_epoch = time.time() - 2

            with patch.object(
                recorder,
                "recent_files",
                side_effect=AssertionError("media storage scan"),
            ):
                result = recorder.recording_at("gate", event_epoch)

            requests = recorder._take_recording_edge_refreshes()

        self.assertIsNone(result)
        self.assertEqual(requests, {("gate", "main"): event_epoch})
        self.assertTrue(recorder._index_wake.is_set())

    def test_recording_at_removes_stale_index_row_and_requests_edge_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "missing.mp4"
            clip.write_bytes(b"recording")
            event_epoch = time.time() - 2
            row = self._row(clip, start_epoch=event_epoch - 5)
            recorder._store_recording_rows("gate", "main", [row])
            clip.unlink()

            result = recorder.recording_at("gate", event_epoch)
            requests = recorder._take_recording_edge_refreshes()
            with recorder._index_connection() as connection:
                indexed = connection.execute(
                    "SELECT COUNT(*) FROM recordings WHERE path = ?",
                    (str(clip),),
                ).fetchone()[0]

        self.assertIsNone(result)
        self.assertEqual(indexed, 0)
        self.assertEqual(requests, {("gate", "main"): event_epoch})

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

    def test_index_only_recording_range_does_not_scan_media_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "indexed.mp4"
            clip.write_bytes(b"x" * 2048)
            row = self._row(clip)
            recorder._store_recording_rows("gate", "main", [row])

            with patch.object(recorder, "_recording_search_dirs", side_effect=AssertionError("media scan")):
                rows = recorder.recording_rows_between(
                    "gate",
                    row["start_epoch"] - 1,
                    row["end_epoch"] + 1,
                    discover_missing=False,
                )

        self.assertEqual([item["path"] for item in rows], [str(clip)])

    def test_playback_lease_temporarily_protects_manifest_recordings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            external = Path(tmpdir) / "recordings" / "incident.mp4"
            external.parent.mkdir(parents=True)
            external.write_bytes(b"incident")
            recorder = Recorder(
                "ffmpeg",
                Path(tmpdir),
                segment_seconds=10,
                protected_recording_paths=lambda: {str(external)},
            )
            clip = Path(tmpdir) / "recordings" / "gate" / "main" / "clip.mp4"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"recording")

            with patch("survng.app.recording_process.recorder.time.monotonic", return_value=100.0):
                recorder.lease_recordings_for_playback([{"path": str(clip)}], ttl_seconds=20)
                protected = recorder.retention.protected_paths_provider()
            with patch("survng.app.recording_process.recorder.time.monotonic", return_value=121.0):
                expired = recorder.retention.protected_paths_provider()

        self.assertEqual(protected, {str(external.resolve()), str(clip.resolve())})
        self.assertEqual(expired, {str(external.resolve())})

    def test_playback_lease_rejects_paths_outside_recording_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            outside = Path(tmpdir) / "outside.mp4"

            recorder.lease_recordings_for_playback([{"path": str(outside)}])

            self.assertEqual(recorder.retention.protected_paths_provider(), set())

    def test_retention_delete_guard_is_atomic_with_playback_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "recordings" / "gate" / "main" / "clip.mp4"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"recording")
            recorder.lease_recordings_for_playback(
                [{"path": str(clip)}], ttl_seconds=20
            )

            deleted = recorder._delete_recording_for_retention(clip)

            self.assertFalse(deleted)
            self.assertTrue(clip.exists())

    def test_manifest_validation_removes_missing_rows_from_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            present = Path(tmpdir) / "recordings" / "gate" / "main" / "present.mp4"
            missing = present.with_name("missing.mp4")
            present.parent.mkdir(parents=True)
            present.write_bytes(b"recording")
            rows = [self._row(present), {**self._row(present), "path": str(missing), "name": missing.name}]
            recorder._store_recording_rows("gate", "main", rows)

            filtered = recorder.discard_missing_recording_rows(rows)
            with recorder._index_connection() as connection:
                indexed = [
                    str(row["path"])
                    for row in connection.execute("SELECT path FROM recordings ORDER BY path")
                ]

        self.assertEqual(filtered, [rows[0]])
        self.assertEqual(indexed, [str(present)])

    def test_recording_index_paths_rebase_when_storage_mount_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_storage = root / "systemd-media"
            new_storage = root / "docker-media"
            index_dir = root / "index"
            old_recorder = Recorder(
                "ffmpeg",
                old_storage,
                segment_seconds=10,
                index_dir=index_dir,
            )
            relative = Path("gate") / "main" / "2026-07-30" / "00" / "clip.mp4"
            old_clip = old_storage / "recordings" / relative
            old_clip.parent.mkdir(parents=True)
            old_clip.write_bytes(b"recording")
            old_recorder._store_recording_rows("gate", "main", [self._row(old_clip)])
            new_clip = new_storage / "recordings" / relative
            new_clip.parent.mkdir(parents=True)
            new_clip.write_bytes(b"recording")

            new_recorder = Recorder(
                "ffmpeg",
                new_storage,
                segment_seconds=10,
                index_dir=index_dir,
            )
            new_recorder._rebase_recording_index_paths()
            rows = new_recorder.recording_rows_between(
                "gate",
                1_783_999_999.0,
                1_784_000_011.0,
                discover_missing=False,
            )
            with new_recorder._index_connection() as connection:
                recorded_root = connection.execute(
                    "SELECT value FROM recording_index_metadata WHERE key = 'recordings_root'"
                ).fetchone()["value"]

        self.assertEqual([row["path"] for row in rows], [str(new_clip.resolve())])
        self.assertEqual(recorded_root, str(new_recorder.recordings_dir.resolve()))

    def test_recording_index_rebase_is_batched_and_handles_recordings_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "recordings-archive"
            old_storage = root / "recordings" / "systemd-media"
            new_storage = root / "docker-media"
            index_dir = root / "index"
            old_recorder = Recorder(
                "ffmpeg",
                old_storage,
                segment_seconds=10,
                index_dir=index_dir,
            )
            relatives = [
                Path("gate") / "main" / "2026-08-06" / "17" / f"clip-{index}.mp4"
                for index in range(3)
            ]
            old_paths = [old_recorder.recordings_dir / relative for relative in relatives]
            for path in old_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"recording")
            old_recorder._store_recording_rows(
                "gate",
                "main",
                [self._row(path) for path in old_paths],
            )
            new_recorder = Recorder(
                "ffmpeg",
                new_storage,
                segment_seconds=10,
                index_dir=index_dir,
            )

            with patch("survng.app.recording_process.index.RECORDING_PATH_REBASE_BATCH_SIZE", 1):
                new_recorder._rebase_recording_index_paths()

            with new_recorder._index_connection() as connection:
                stored = {
                    str(row["path"])
                    for row in connection.execute("SELECT path FROM recordings")
                }

        self.assertEqual(
            stored,
            {str(new_recorder.recordings_dir / relative) for relative in relatives},
        )

    def test_legacy_index_at_current_root_gets_fast_rebase_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = (
                recorder.recordings_dir
                / "gate"
                / "main"
                / "2026-08-06"
                / "17"
                / "clip.mp4"
            )
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"recording")
            recorder._store_recording_rows("gate", "main", [self._row(clip)])

            recorder._rebase_recording_index_paths()

            with recorder._index_connection() as connection:
                marker = connection.execute(
                    "SELECT value FROM recording_index_metadata WHERE key = 'recordings_root'"
                ).fetchone()["value"]
                stored_path = connection.execute(
                    "SELECT path FROM recordings"
                ).fetchone()["path"]

        self.assertEqual(marker, str(recorder.recordings_dir.resolve()))
        self.assertEqual(stored_path, str(clip))

    def test_stale_recorder_cleanup_terminates_processes_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            active = 0
            maximum = 0
            lock = threading.Lock()

            def kill(_pid: int) -> None:
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.02)
                with lock:
                    active -= 1

            with (
                patch.object(
                    recorder,
                    "_owned_ffmpeg_recorders",
                    return_value={
                        ("gate", "main"): [101],
                        ("gate", "live"): [102],
                    },
                ),
                patch.object(recorder, "_kill_pid", side_effect=kill),
            ):
                recorder.cleanup_stale_recorders({("gate", "main"), ("gate", "live")})

        self.assertEqual(maximum, 2)

    def test_stale_recorder_cleanup_propagates_termination_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            with (
                patch.object(
                    recorder,
                    "_owned_ffmpeg_recorders",
                    return_value={("gate", "main"): [101]},
                ),
                patch.object(
                    recorder,
                    "_kill_pid",
                    side_effect=PermissionError("not owner"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "failed to terminate recorder processes: 101",
                ):
                    recorder.cleanup_stale_recorders({("gate", "main")})

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
            recorder.processes[("front-door", "main")] = (process, Mock(), Mock())

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
            recorder.processes[("front-door", "live")] = (process, Mock(), Mock())

            with patch("survng.app.recording_process.index.time.time", return_value=now_epoch):
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

    def test_index_only_availability_does_not_touch_recording_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "clip.mp4"
            clip.write_bytes(b"x" * 2048)
            recorder._store_recording_rows(
                "front-door",
                "main",
                [self._row(clip, start_epoch=1_784_000_000.0)],
            )

            with patch.object(Path, "is_file", side_effect=AssertionError("filesystem accessed")):
                availability = recorder.recording_availability_between(
                    "front-door",
                    1_784_000_000.0,
                    1_784_000_020.0,
                    discover_missing=False,
                )

        self.assertEqual(availability["segment_count"], 1)

    def test_grid_availability_reads_all_cameras_and_sources_in_one_index_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            front_main = Path(tmpdir) / "front-main.mp4"
            front_live = Path(tmpdir) / "front-live.mp4"
            gate_live = Path(tmpdir) / "gate-live.mp4"
            for path in (front_main, front_live, gate_live):
                path.write_bytes(b"x" * 2048)
            recorder._store_recording_rows(
                "front-door", "main", [self._row(front_main, start_epoch=1_784_000_000.0)],
            )
            recorder._store_recording_rows(
                "front-door", "live", [self._row(front_live, start_epoch=1_784_000_010.0)],
            )
            recorder._store_recording_rows(
                "gate", "live", [self._row(gate_live, start_epoch=1_784_000_020.0)],
            )

            availability = recorder.recording_grid_availability_between(
                ["front-door", "gate", "missing"],
                1_784_000_000.0,
                1_784_000_040.0,
            )

        self.assertEqual(availability["front-door"]["main"]["segment_count"], 1)
        self.assertEqual(availability["front-door"]["live"]["segment_count"], 1)
        self.assertEqual(availability["gate"]["live"]["segment_count"], 1)
        self.assertEqual(availability["gate"]["main"]["segment_count"], 0)
        self.assertEqual(availability["missing"]["live"]["ranges"], [])

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
            recorder.processes[("front-door", "main")] = (process, Mock(), Mock())

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

    def test_start_indexer_restores_missing_maintenance_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            index_thread = Mock(name="existing-index-thread")
            index_thread.is_alive.return_value = True
            recorder._index_thread = index_thread
            maintenance_thread = Mock(name="replacement-maintenance-thread")
            with patch(
                "survng.app.recording_process.index.threading.Thread",
                return_value=maintenance_thread,
            ) as thread_factory:
                recorder.start_indexer([])

        thread_factory.assert_called_once()
        self.assertEqual(
            thread_factory.call_args.kwargs["name"],
            "recording-index-maintenance",
        )
        maintenance_thread.start.assert_called_once_with()

    def test_index_loop_services_requested_near_live_refresh_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            event_epoch = time.time()
            recorder.request_recording_edge_refresh("gate", "main", event_epoch)

            def stop_after_refresh(*_args) -> int:
                recorder._index_stop.set()
                return 1

            with (
                patch.object(recorder, "refresh_recording_index") as refresh_index,
                patch.object(
                    recorder,
                    "refresh_recording_edge",
                    side_effect=stop_after_refresh,
                ) as refresh_edge,
            ):
                recorder._recording_index_loop({})

        refresh_index.assert_called_once_with({}, full=False, run_maintenance=False)
        refresh_edge.assert_called_once_with("gate", "main", event_epoch)

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

    def test_recent_index_discovery_does_not_probe_healthy_recorder_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            now = datetime.now()
            hour_dir = (
                Path(tmpdir) / "recordings" / "front-door" / "main"
                / now.strftime("%Y-%m-%d") / now.strftime("%H")
            )
            hour_dir.mkdir(parents=True)
            clips = []
            for offset in (30, 20, 10):
                clip = hour_dir / datetime.fromtimestamp(
                    now.timestamp() - offset
                ).strftime("%Y%m%d-%H%M%S.mp4")
                clip.write_bytes(b"x" * 2048)
                clips.append(clip)
            camera = Mock(id="front-door", record=True, record_sub=False, live_stream_url="")

            with (
                patch.object(recorder, "queue_recording_validation") as queue_validation,
                patch.object(recorder, "_probe_recording") as probe,
            ):
                recorder.refresh_recording_index(
                    {camera.id: camera},
                    full=False,
                    run_maintenance=False,
                )

        queue_validation.assert_not_called()
        probe.assert_not_called()
        self.assertEqual(recorder._validation_pending_set, set())

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

    def test_full_health_does_not_mark_rows_added_during_storage_walk_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            new_segment = Path(tmpdir) / "new-segment.mp4"
            new_segment.write_bytes(b"recording")

            def discover(**_kwargs):
                recorder._store_recording_rows(
                    "front-door",
                    "main",
                    [self._row(new_segment, 200.0)],
                )
                # Simulate a directory that the long walk visited before this
                # segment was finalized and indexed.
                return {}

            with patch.object(recorder, "_recording_files_by_source", side_effect=discover):
                health = recorder.storage_index_health(full=True)

        self.assertEqual(health["missing_index_files"], [])

    def test_full_health_revalidates_before_treating_undiscovered_rows_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            hour = Path(tmpdir) / "recordings" / "gate" / "main" / "2026-01-01" / "00"
            hour.mkdir(parents=True)
            present = hour / "20260101-000000.mp4"
            present.write_bytes(b"segment")
            self._age_file(present, 120)
            recorder._store_recording_rows(
                "gate",
                "main",
                recorder._recording_rows_for_files("gate", "main", [present]),
            )

            with patch.object(recorder, "_recording_files_by_source", return_value={}):
                health = recorder.storage_index_health(full=True)
                repairs = recorder.reconcile_storage_index(full=True, health=health)

            with recorder._index_connection() as connection:
                remaining = {
                    str(row[0]) for row in connection.execute("SELECT path FROM recordings")
                }

        self.assertEqual(health["missing_index_files"], [])
        self.assertEqual(health["index_rows_retained_present"], 1)
        self.assertEqual(repairs["stale_index_rows_removed"], 0)
        self.assertEqual(remaining, {str(present)})

    def test_quick_health_does_not_mark_discovered_paths_stale_on_probe_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            now = datetime.now()
            hour = (
                Path(tmpdir)
                / "recordings"
                / "gate"
                / "main"
                / now.strftime("%Y-%m-%d")
                / now.strftime("%H")
            )
            hour.mkdir(parents=True)
            clip = hour / f"{now.strftime('%Y%m%d-%H%M%S')}.mp4"
            clip.write_bytes(b"segment")
            self._age_file(clip, 120)
            recorder._store_recording_rows(
                "gate",
                "main",
                recorder._recording_rows_for_files("gate", "main", [clip]),
            )

            def fail_presence(path):
                if Path(path) == clip:
                    return "unknown"
                from survng.app.media_storage import path_presence as real_presence
                return real_presence(path)

            with patch("survng.app.recording_process.index.path_presence", side_effect=fail_presence):
                health = recorder.storage_index_health(full=False)
                repairs = recorder.reconcile_storage_index(full=False, health=health)

            with recorder._index_connection() as connection:
                remaining = {
                    str(row[0]) for row in connection.execute("SELECT path FROM recordings")
                }

        self.assertEqual(health["missing_index_files"], [])
        self.assertEqual(repairs["stale_index_rows_removed"], 0)
        self.assertEqual(remaining, {str(clip)})

    def test_reconcile_never_deletes_paths_from_same_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            hour = Path(tmpdir) / "recordings" / "gate" / "main" / "2026-01-01" / "00"
            hour.mkdir(parents=True)
            clip = hour / "20260101-000000.mp4"
            clip.write_bytes(b"segment")
            self._age_file(clip, 120)
            rows = recorder._recording_rows_for_files("gate", "main", [clip])
            recorder._store_recording_rows("gate", "main", rows)
            health = {
                "files_by_source": {("gate", "main"): [clip]},
                "unindexed_files": [],
                "missing_index_files": [str(clip)],
                "indexed_recordings": 1,
            }

            repairs = recorder.reconcile_storage_index(full=True, health=health)

            with recorder._index_connection() as connection:
                remaining = {
                    str(row[0]) for row in connection.execute("SELECT path FROM recordings")
                }

        self.assertEqual(repairs["stale_index_rows_removed"], 0)
        self.assertEqual(remaining, {str(clip)})

    def test_prune_ignores_transient_stat_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "retained.mp4"
            clip.write_bytes(b"recording")
            recorder._store_recording_rows("front-door", "main", [self._row(clip)])

            with patch(
                "survng.app.recording_process.index.path_presence",
                return_value="unknown",
            ):
                removed = recorder._prune_missing_index_rows()

            with recorder._index_connection() as connection:
                remaining = connection.execute("SELECT count(*) FROM recordings").fetchone()[0]

        self.assertEqual(removed, 0)
        self.assertEqual(remaining, 1)

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
                patch.object(recorder, "_prune_missing_index_rows") as prune,
                patch.object(
                    recorder,
                    "_validate_index_batch",
                    side_effect=lambda limit: recorder._index_stop.set(),
                ) as validate,
                patch.object(recorder, "_backfill_stream_fingerprints"),
                patch.object(recorder, "_backfill_recording_location_id_batch") as location_backfill,
                patch.object(recorder, "_reconcile_recording_source") as reconcile,
                patch.object(recorder._index_stop, "wait", return_value=False),
            ):
                recorder._recording_index_maintenance_loop({})

        prune.assert_not_called()
        location_backfill.assert_called_once_with()
        validate.assert_called_once_with(limit=20)
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
                patch("survng.app.recording_process.index.time.time", return_value=now_epoch),
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
                patch("survng.app.recording_process.index.time.time", return_value=now_epoch + 15),
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

            with patch("survng.app.recording_process.index.mp4_stream_fingerprint", return_value="stream-v1"):
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

    def test_background_validation_ignores_unqueued_historical_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "unqueued.mp4"
            clip.write_bytes(b"recording")
            self._age_file(clip)
            recorder._store_recording_rows("front-door", "main", [self._row(clip)])

            with patch.object(recorder, "_probe_recording") as probe:
                validated = recorder._validate_index_batch(limit=1)

        self.assertEqual(validated, 0)
        probe.assert_not_called()

    def test_manual_maintenance_discovers_unqueued_validation_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "unqueued.mp4"
            clip.write_bytes(b"recording")
            self._age_file(clip)
            row = self._row(clip)
            recorder._store_recording_rows("front-door", "main", [row])

            with (
                patch.object(recorder, "_probe_recording", return_value=(9.0, "")),
                patch("survng.app.recording_process.index.mp4_stream_fingerprint", return_value="stream-v2"),
            ):
                self.assertEqual(
                    recorder._validate_index_batch(limit=1, discover_unqueued=True),
                    1,
                )
            with recorder._index_connection() as connection:
                indexed = dict(connection.execute(
                    "SELECT validated, fingerprint_checked, stream_fingerprint FROM recordings WHERE path = ?",
                    (str(clip),),
                ).fetchone())

        self.assertEqual(indexed["validated"], 1)
        self.assertEqual(indexed["fingerprint_checked"], 1)
        self.assertEqual(indexed["stream_fingerprint"], "stream-v2")

    def test_background_fingerprinting_ignores_unqueued_historical_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "fingerprint.mp4"
            clip.write_bytes(b"recording")
            row = self._row(clip)
            row["validated"] = True
            recorder._store_recording_rows("front-door", "main", [row])

            with patch("survng.app.recording_process.index.mp4_stream_fingerprint") as fingerprint:
                updated = recorder._backfill_stream_fingerprints(limit=1)

        self.assertEqual(updated, 0)
        fingerprint.assert_not_called()

    def test_manual_maintenance_discovers_unqueued_fingerprint_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder("ffmpeg", Path(tmpdir), segment_seconds=10)
            clip = Path(tmpdir) / "fingerprint.mp4"
            clip.write_bytes(b"recording")
            row = self._row(clip)
            row["validated"] = True
            recorder._store_recording_rows("front-door", "main", [row])

            with patch("survng.app.recording_process.index.mp4_stream_fingerprint", return_value="stream-v3"):
                self.assertEqual(
                    recorder._backfill_stream_fingerprints(
                        limit=1,
                        discover_unqueued=True,
                    ),
                    1,
                )
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

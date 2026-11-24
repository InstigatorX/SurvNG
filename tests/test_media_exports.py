from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from survng.app.media_exports import MediaExportManager, MediaExportStore
from survng.app.config import MediaStorageConfig, MediaStorageLocationConfig
from survng.app.media_storage import MediaStorageRegistry


class FakeRecorder:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.leases: list[tuple[list[dict], float]] = []

    def recording_rows_between(
        self,
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        source: str,
        *,
        discover_missing: bool,
    ) -> list[dict]:
        self.request = (camera_id, start_epoch, end_epoch, source, discover_missing)
        return list(self.rows)

    def lease_recordings_for_playback(self, rows: list[dict], *, ttl_seconds: float) -> None:
        self.leases.append((list(rows), ttl_seconds))


class MediaExportTest(unittest.TestCase):
    def setUp(self) -> None:
        capacity = patch(
            "survng.app.media_storage.shutil.disk_usage",
            return_value=SimpleNamespace(total=1000, used=500, free=500),
        )
        capacity.start()
        self.addCleanup(capacity.stop)

    def test_exports_follow_role_specific_media_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root / "external"
            external.mkdir()
            registry = MediaStorageRegistry(
                root / "storage",
                MediaStorageConfig(locations=[MediaStorageLocationConfig(
                    id="archive",
                    path=str(external),
                    roles=["exports"],
                )]),
            )

            manager = MediaExportManager(
                root / "storage",
                root / "database",
                recorder=lambda: FakeRecorder([]),
                ffmpeg_path=lambda: "ffmpeg",
                hardware_backend=lambda: "cpu",
                media_storage=registry,
            )

            self.assertEqual(manager.exports_dir, external / "exports")
            self.assertFalse(manager.exports_dir.exists())
            self.assertEqual(manager._prepare_output_directory(), external / "exports")
            self.assertTrue(manager.recording_dir.is_dir())
            self.assertTrue(manager.timelapse_dir.is_dir())

    def test_full_storage_allows_startup_and_export_worker_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media_roots = [root / "first", root / "second"]
            for media_root in media_roots:
                media_root.mkdir()
            registry = MediaStorageRegistry(root / "storage", MediaStorageConfig(
                locations=[MediaStorageLocationConfig(
                    id=path.name, path=str(path), roles=["exports"], reserve_percent=15,
                ) for path in media_roots],
            ))
            segment = root / "segment.mp4"
            segment.write_bytes(b"source")
            recorder = FakeRecorder([{
                "path": str(segment), "start_epoch": 100.0, "end_epoch": 110.0,
                "duration_seconds": 10.0,
            }])
            free_bytes = {path: 0 for path in media_roots}

            def usage(path: Path) -> SimpleNamespace:
                free = free_bytes[Path(path)]
                return SimpleNamespace(total=1000, used=1000 - free, free=free)

            def wait_for_job(manager: MediaExportManager, job_id: str) -> dict:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    job = manager.store.get(job_id)
                    if job and job["status"] in {"completed", "failed"}:
                        return job
                    time.sleep(0.01)
                self.fail("export worker did not finish the job")

            payload = {
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0, "end_epoch": 110.0, "options": {},
            }
            with patch("survng.app.media_storage.shutil.disk_usage", side_effect=usage):
                manager = MediaExportManager(
                    root / "storage", root / "database", recorder=lambda: recorder,
                    ffmpeg_path=lambda: "ffmpeg", hardware_backend=lambda: "cpu",
                    media_storage=registry,
                )
                manager._build_recording = Mock(return_value=(segment, []))
                self.assertTrue(all(not (path / "exports").exists() for path in media_roots))
                manager.start()
                try:
                    failed = wait_for_job(manager, str(manager.create(payload)["id"]))
                    self.assertEqual(failed["status"], "failed")
                    self.assertIn("no writable media location supports exports", failed["error"])
                    manager._build_recording.assert_not_called()
                    self.assertTrue(manager.is_running())
                    self.assertFalse((root / "storage" / "exports").exists())
                    self.assertTrue(all(not (path / "exports").exists() for path in media_roots))

                    free_bytes[media_roots[1]] = 130
                    completed = wait_for_job(manager, str(manager.create(payload)["id"]))
                    self.assertEqual(completed["status"], "completed", completed["error"])
                    output, _ = manager.output_path(str(completed["id"]))
                    self.assertEqual(output.parent, media_roots[1] / "exports" / "recording")
                    manifest = media_roots[1] / "exports" / "manifests" / f"{completed['id']}.json"
                    self.assertTrue(manifest.exists())
                    self.assertFalse((media_roots[0] / "exports").exists())
                    manager._delete_job_files(completed)
                    self.assertFalse(output.exists())
                    self.assertFalse(manifest.exists())
                finally:
                    self.assertTrue(manager.stop())

    def test_missing_media_mount_allows_export_worker_startup_without_creating_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "unmounted" / "media"
            registry = MediaStorageRegistry(root / "storage", MediaStorageConfig(
                locations=[MediaStorageLocationConfig(
                    id="archive", path=str(missing), roles=["exports"], require_mount=True,
                )],
            ))
            manager = MediaExportManager(
                root / "storage", root / "database", recorder=lambda: FakeRecorder([]),
                ffmpeg_path=lambda: "ffmpeg", hardware_backend=lambda: "cpu",
                media_storage=registry,
            )
            manager.start()
            try:
                self.assertTrue(manager.is_running())
                self.assertFalse(missing.exists())
                self.assertFalse((root / "storage").exists())
                self.assertEqual(manager.list(), [])
            finally:
                self.assertTrue(manager.stop())

    def test_historical_export_remains_accessible_across_configured_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            registry = MediaStorageRegistry(
                root / "storage",
                MediaStorageConfig(locations=[
                    MediaStorageLocationConfig(
                        id="first", path=str(first), roles=["exports"]
                    ),
                    MediaStorageLocationConfig(
                        id="second", path=str(second), roles=["exports"]
                    ),
                ]),
            )
            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: "ffmpeg",
                hardware_backend=lambda: "cpu", media_storage=registry,
            )
            historical_root = (
                second if manager.exports_dir == first / "exports" else first
            )
            output = historical_root / "exports" / "recording" / "old.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"historic")
            job = manager.store.create({
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 1.0, "end_epoch": 2.0, "options": {},
            })
            manager.store.update(
                str(job["id"]), status="completed", output_path=str(output),
                output_name=output.name,
            )

            resolved, _name = manager.output_path(str(job["id"]))
            manager._delete_job_files(manager.store.get(str(job["id"])) or {})

            self.assertEqual(resolved, output)
            self.assertFalse(output.exists())

    def test_worker_can_restart_without_consuming_stale_shutdown_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: "ffmpeg",
                hardware_backend=lambda: "cpu",
            )

            manager.start()
            self.assertTrue(manager.stop())
            manager.start()
            try:
                self.assertIsNotNone(manager._thread)
                self.assertTrue(manager._thread.is_alive())
            finally:
                self.assertTrue(manager.stop())

    def test_store_persists_jobs_and_marks_interrupted_work_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = MediaExportStore(root)
            job = store.create({
                "kind": "recording",
                "camera_id": "gate",
                "source": "main",
                "start_epoch": 100.0,
                "end_epoch": 120.0,
                "options": {},
            })
            store.update(str(job["id"]), status="running", phase="Encoding")

            restored = MediaExportStore(root).get(str(job["id"]))

            self.assertEqual(restored["status"], "failed")
            self.assertEqual(restored["error"], "export interrupted by server restart")

    def test_manager_assigns_retention_to_job_interrupted_by_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = MediaExportStore(root / "database")
            job = store.create({
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0, "end_epoch": 120.0, "options": {},
            })
            store.update(str(job["id"]), status="running")

            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: "ffmpeg",
                hardware_backend=lambda: "cpu",
            )
            restored = manager.store.get(str(job["id"]))

            self.assertEqual(restored["status"], "failed")
            self.assertTrue(restored["expires_at"])

    def test_continuous_groups_report_real_recording_gaps(self) -> None:
        rows = [
            {"start_epoch": 100.0, "end_epoch": 110.0},
            {"start_epoch": 110.2, "end_epoch": 120.0},
            {"start_epoch": 124.0, "end_epoch": 134.0},
        ]

        groups, gaps = MediaExportManager._continuous_groups(rows, 105.0, 130.0)

        self.assertEqual([len(group) for group in groups], [2, 1])
        self.assertEqual(gaps, [{"start_epoch": 120.0, "end_epoch": 124.0, "duration_seconds": 4.0}])

    def test_continuous_groups_split_incompatible_stream_revisions_without_fake_gap(self) -> None:
        rows = [
            {"start_epoch": 100.0, "end_epoch": 110.0, "stream_fingerprint": "h264-a"},
            {"start_epoch": 110.0, "end_epoch": 120.0, "stream_fingerprint": "h264-b"},
        ]

        groups, gaps = MediaExportManager._continuous_groups(rows, 100.0, 120.0)

        self.assertEqual([len(group) for group in groups], [1, 1])
        self.assertEqual(gaps, [])

    def test_vaapi_command_has_one_combined_filter_and_cpu_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage",
                root / "database",
                recorder=lambda: FakeRecorder([]),
                ffmpeg_path=lambda: "/config/ffmpeg",
                hardware_backend=lambda: "vaapi",
            )
            commands = manager._timelapse_commands(
                "vaapi", root / "input.ffconcat", root / "out.mp4",
                "fps=1/10,scale=1280:-2,setpts=N/(30*TB)", 30,
            )

            self.assertEqual([name for name, _ in commands], ["vaapi", "cpu"])
            vaapi = commands[0][1]
            self.assertEqual(vaapi.count("-vf"), 1)
            self.assertIn("hwupload", vaapi[vaapi.index("-vf") + 1])
            self.assertEqual(commands[1][1].count("-vf"), 1)

    def test_timelapse_command_trims_in_filter_and_uses_cfr_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: "ffmpeg",
                hardware_backend=lambda: "qsv",
                hardware_device=lambda _backend: "/dev/dri/renderD129",
            )
            commands = manager._timelapse_commands(
                "qsv", root / "input.ffconcat", root / "out.mp4",
                "trim=start=3:duration=20,select='gte(t-prev_selected_t,5)'", 24,
            )

            qsv = commands[0][1]
            self.assertNotIn("-ss", qsv)
            self.assertNotIn("-t", qsv)
            self.assertEqual(qsv[qsv.index("-qsv_device") + 1], "/dev/dri/renderD129")
            self.assertEqual(qsv[qsv.index("-fps_mode") + 1], "cfr")

    def _command_manager(self, root: Path, backend: str) -> MediaExportManager:
        return MediaExportManager(
            root / "storage", root / "database",
            recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: "ffmpeg",
            hardware_backend=lambda: backend,
            hardware_device=lambda _backend: "/dev/dri/renderD129",
        )

    def test_gpu_filter_chain_requires_a_source_that_fills_the_target(self) -> None:
        chain = MediaExportManager._gpu_filter_chain
        source = {"width": 3840, "height": 2160}

        vaapi = chain("vaapi", source, (1920, 1080), before=["fps=20"], after=["setsar=1"])
        qsv = chain("qsv", source, (1920, 1080), before=["fps=20"], after=["setsar=1"])

        self.assertEqual(
            vaapi,
            "fps=20,format=nv12|vaapi,hwupload,scale_vaapi=w=1920:h=1080:format=nv12,setsar=1",
        )
        self.assertEqual(qsv, "fps=20,vpp_qsv=w=1920:h=1080:format=nv12,setsar=1")
        # A 4:3 source in a 16:9 frame needs the software pad.
        self.assertIsNone(chain(
            "vaapi", {"width": 2592, "height": 1944}, (1920, 1080), before=[], after=[],
        ))
        self.assertIsNone(chain("cpu", source, (1920, 1080), before=[], after=[]))
        self.assertIsNone(chain("vaapi", {}, (1920, 1080), before=[], after=[]))

    def test_recording_commands_try_full_gpu_then_hardware_encode_then_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = (root / "input.ffconcat", root / "out.mp4", "scale=1920:1080,fps=20", 1.25, 10.0, False)

            vaapi = self._command_manager(root, "vaapi")._recording_commands(
                "vaapi", *arguments, gpu_filters="hwupload,scale_vaapi=w=1920:h=1080", source_codec="hevc",
            )
            qsv = self._command_manager(root, "qsv")._recording_commands(
                "qsv", *arguments, gpu_filters="vpp_qsv=w=1920:h=1080", source_codec="hevc",
            )
            unsupported = self._command_manager(root, "qsv")._recording_commands(
                "qsv", *arguments, gpu_filters="vpp_qsv=w=1920:h=1080", source_codec="mjpeg",
            )

            self.assertEqual([name for name, _ in vaapi], ["vaapi-gpu", "vaapi", "cpu"])
            full = vaapi[0][1]
            # Decoder options are input options.
            self.assertLess(full.index("-hwaccel_output_format"), full.index("-i"))
            self.assertEqual(full[full.index("-vf") + 1], "hwupload,scale_vaapi=w=1920:h=1080")
            self.assertEqual(full[full.index("-ss") + 1], "1.250000")
            self.assertIn("hwupload", vaapi[1][1][vaapi[1][1].index("-vf") + 1])
            self.assertNotIn("-hwaccel", vaapi[1][1])

            self.assertEqual([name for name, _ in qsv], ["qsv-gpu", "qsv", "cpu"])
            full = qsv[0][1]
            self.assertLess(full.index("hevc_qsv"), full.index("-i"))
            self.assertNotIn("-pix_fmt", full)
            self.assertEqual([name for name, _ in unsupported], ["qsv", "cpu"])

    def test_keyframe_timelapse_skips_non_keyframes_in_software_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commands = self._command_manager(root, "vaapi")._timelapse_commands(
                "vaapi", root / "input.ffconcat", root / "out.mp4", "select=1", 30,
                keyframes_only=True, gpu_filters="hwupload", source_codec="h264",
            )
            dense = self._command_manager(root, "vaapi")._timelapse_commands(
                "vaapi", root / "input.ffconcat", root / "out.mp4", "select=1", 30,
                gpu_filters="hwupload", source_codec="h264",
            )

            self.assertEqual([name for name, _ in commands], ["vaapi", "cpu"])
            for _name, command in commands:
                self.assertLess(command.index("-skip_frame"), command.index("-i"))
                self.assertEqual(command[command.index("-skip_frame") + 1], "nokey")
            self.assertEqual([name for name, _ in dense], ["vaapi-gpu", "vaapi", "cpu"])
            for _name, command in dense:
                self.assertNotIn("-skip_frame", command)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
    def test_real_timelapse_samples_keyframes_once_per_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            # 40 s at 10 fps with a keyframe every second.
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10",
                    "-t", "40", "-c:v", "libx264", "-preset", "ultrafast", "-g", "10",
                    "-keyint_min", "10", "-sc_threshold", "0", "-pix_fmt", "yuv420p", "-y", str(source),
                ],
                check=True, timeout=60,
            )
            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: shutil.which("ffmpeg"),
                hardware_backend=lambda: "cpu",
            )
            rows = [{
                "path": str(source), "start_epoch": 1000.0, "end_epoch": 1040.0,
                "duration_seconds": 40.0,
            }]
            work = root / "work"
            work.mkdir()
            ran: list[list[str]] = []
            run_process = manager._run_process

            def recording_run(command, *args, **kwargs):
                ran.append(command)
                return run_process(command, *args, **kwargs)

            manager._run_process = recording_run  # type: ignore[method-assign]

            def frames(interval: float) -> int:
                job = manager.store.create({
                    "kind": "timelapse", "camera_id": "gate", "source": "main",
                    "start_epoch": 1000.0, "end_epoch": 1040.0,
                    "options": {"sample_interval_seconds": interval, "width": 320},
                })
                output, _gaps = manager._build_timelapse(job, rows, work, threading.Event())
                counted = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(output),
                    ],
                    stdout=subprocess.PIPE, text=True, check=True, timeout=30,
                )
                return int(counted.stdout.strip())

            self.assertAlmostEqual(manager._keyframe_interval(source), 1.0, places=3)
            # 5 s is at least twice the keyframe spacing: keyframes only.
            keyframe_count = frames(5.0)
            self.assertIn("-skip_frame", ran[-1])
            self.assertIn("floor(t/5.000000)", ran[-1][ran[-1].index("-vf") + 1])
            # 1 s is not, so every frame is decoded as before.
            frames(1.0)
            self.assertNotIn("-skip_frame", ran[-1])
            # Decoding every frame samples the same eight source frames. The
            # CFR output may repeat the last one, identically in both modes.
            manager._keyframe_interval = lambda _path: None  # type: ignore[method-assign]
            self.assertEqual(frames(5.0), keyframe_count)
            self.assertNotIn("-skip_frame", ran[-1])
            self.assertGreaterEqual(keyframe_count, 8)
            self.assertLessEqual(keyframe_count, 10)

    def _timelapse_attempts(
        self, root: Path, profile: dict, rows: list[dict], options: dict
    ) -> list[tuple[str, list[str]]]:
        """Every attempt ``_build_timelapse`` would run, each made to fail."""
        manager = self._command_manager(root, "vaapi")
        manager._probe_source = lambda _path: profile  # type: ignore[method-assign]
        manager._keyframe_interval = lambda _path: None  # type: ignore[method-assign]
        attempts: list[tuple[str, list[str]]] = []
        build = manager._timelapse_commands

        def capture(*args, **kwargs):
            commands = build(*args, **kwargs)
            attempts.extend(commands)
            return commands

        def fail(*_args, **_kwargs) -> None:
            raise RuntimeError("not run")

        manager._timelapse_commands = capture  # type: ignore[method-assign]
        manager._run_process = fail  # type: ignore[method-assign]
        work = root / "work"
        work.mkdir(exist_ok=True)
        job = manager.store.create({
            "kind": "timelapse", "camera_id": "gate", "source": "main",
            "start_epoch": rows[0]["start_epoch"], "end_epoch": rows[-1]["end_epoch"],
            "options": options,
        })
        with self.assertRaises(RuntimeError):
            manager._build_timelapse(job, rows, work, threading.Event())
        return attempts

    def test_gpu_timelapse_width_matches_ffmpeg_half_up_rounding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = [{"path": "a.mp4", "start_epoch": 0.0, "end_epoch": 60.0, "duration_seconds": 60.0}]
            # scale=480:-2 on 512x1200 gives 1126 in FFmpeg; round() gives 1124.
            attempts = self._timelapse_attempts(
                Path(temporary), {"width": 512, "height": 1200, "codec": "h264"}, rows, {"width": 480},
            )

            self.assertEqual([name for name, _ in attempts], ["vaapi-gpu", "vaapi", "cpu"])
            full = attempts[0][1]
            self.assertIn("scale_vaapi=w=480:h=1126:format=nv12", full[full.index("-vf") + 1])

    def test_gpu_timelapse_is_not_attempted_across_different_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = [
                {"path": "a.mp4", "start_epoch": 0.0, "end_epoch": 60.0, "duration_seconds": 60.0,
                 "stream_fingerprint": "4k"},
                {"path": "b.mp4", "start_epoch": 60.0, "end_epoch": 120.0, "duration_seconds": 60.0,
                 "stream_fingerprint": "1080p"},
            ]
            attempts = self._timelapse_attempts(
                Path(temporary), {"width": 3840, "height": 2160, "codec": "h264"}, rows, {"height": 1080},
            )

            self.assertEqual([name for name, _ in attempts], ["vaapi", "cpu"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
    def test_short_keyframe_timelapse_is_replaced_by_a_full_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            # Intra refresh: one real keyframe, at the start. The spacing
            # probe is overridden to stand for a camera whose keyframes
            # were frequent only while it was measured.
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10",
                    "-t", "40", "-c:v", "libx264", "-preset", "ultrafast", "-g", "10",
                    "-x264-params", "intra-refresh=1", "-pix_fmt", "yuv420p", "-y", str(source),
                ],
                check=True, timeout=60,
            )
            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: shutil.which("ffmpeg"),
                hardware_backend=lambda: "cpu",
            )
            rows = [{
                "path": str(source), "start_epoch": 1000.0, "end_epoch": 1040.0,
                "duration_seconds": 40.0,
            }]
            work = root / "work"
            work.mkdir()
            ran: list[list[str]] = []
            run_process = manager._run_process

            def recording_run(command, *args, **kwargs):
                ran.append(command)
                return run_process(command, *args, **kwargs)

            manager._run_process = recording_run  # type: ignore[method-assign]
            job = manager.store.create({
                "kind": "timelapse", "camera_id": "gate", "source": "main",
                "start_epoch": 1000.0, "end_epoch": 1040.0,
                "options": {"sample_interval_seconds": 5.0, "width": 320},
            })
            # The decoder-based probe is not fooled by the container flags.
            self.assertIsNone(manager._keyframe_interval(source))
            manager._keyframe_interval = lambda _path: 1.0  # type: ignore[method-assign]

            output, _gaps = manager._build_timelapse(job, rows, work, threading.Event())

            self.assertEqual(["-skip_frame" in command for command in ran], [True, False])
            self.assertGreaterEqual(manager._video_frame_count(output), 8)

    def test_header_only_video_is_not_accepted_as_completed_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: "ffmpeg",
                hardware_backend=lambda: "cpu",
            )
            output = root / "empty.mp4"
            output.write_bytes(bytes(262))

            self.assertFalse(manager._valid_video_output(output))
            self.assertFalse(manager._valid_media_container(output))

    def test_video_validation_decodes_one_frame_without_scanning_entire_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: "/config/ffmpeg",
                hardware_backend=lambda: "cpu",
            )
            output = root / "long-export.mp4"
            output.write_bytes(bytes(2048))
            manager._valid_media_container = lambda _path: True  # type: ignore[method-assign]

            with patch(
                "survng.app.media_exports.subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ) as run:
                valid = manager._valid_video_output(output)

            self.assertTrue(valid)
            command = run.call_args.args[0]
            self.assertNotIn("-count_frames", command)
            self.assertEqual(command[command.index("-frames:v") + 1], "1")

    def test_worker_publishes_single_compatible_mp4_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "segment.mp4"
            segment.write_bytes(b"source")
            recorder = FakeRecorder([{
                "path": str(segment),
                "start_epoch": 100.0,
                "end_epoch": 110.0,
                "duration_seconds": 10.0,
            }])
            manager = MediaExportManager(
                root / "storage",
                root / "database",
                recorder=lambda: recorder,
                ffmpeg_path=lambda: "ffmpeg",
                hardware_backend=lambda: "cpu",
            )

            def fake_run(
                command: list[str], cancel: threading.Event, timeout: float, *, process_name: str = ""
            ) -> None:
                self.assertFalse(cancel.is_set())
                self.assertGreater(timeout, 0)
                self.assertEqual(process_name, "survng-export")
                Path(command[-1]).write_bytes(b"export")

            manager._run_process = fake_run  # type: ignore[method-assign]
            manager._valid_video_output = lambda _path: True  # type: ignore[method-assign]
            manager._probe_source = lambda _path: {  # type: ignore[method-assign]
                "width": 1920, "height": 1080, "fps": 20.0, "has_audio": True,
            }
            manager.start()
            try:
                created = manager.create({
                    "kind": "recording",
                    "camera_id": "gate",
                    "source": "main",
                    "start_epoch": 101.0,
                    "end_epoch": 109.0,
                    "options": {},
                })
                deadline = time.monotonic() + 3
                completed = None
                while time.monotonic() < deadline:
                    completed = manager.get(str(created["id"]))
                    if completed and completed["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.02)
            finally:
                manager.stop()

            self.assertEqual(completed["status"], "completed")
            self.assertTrue(completed["download_url"].endswith("/download"))
            output, name = manager.output_path(str(created["id"]))
            self.assertEqual(output.read_bytes(), b"export")
            self.assertEqual(name, completed["output_name"])
            self.assertTrue(name.endswith(".mp4"))
            self.assertTrue((manager.manifest_dir / f"{created['id']}.json").is_file())
            self.assertTrue(recorder.leases)
            self.assertEqual(recorder.request[-1], False)

    def test_recording_commands_use_hardware_then_cpu_and_normalize_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: "/config/ffmpeg",
                hardware_backend=lambda: "qsv",
                hardware_device=lambda _backend: "/dev/dri/renderD129",
            )

            commands = manager._recording_commands(
                "qsv", root / "input.ffconcat", root / "out.mp4",
                "scale=1920:1080,fps=20", 1.25, 10.0, True,
            )

            self.assertEqual([name for name, _command in commands], ["qsv", "cpu"])
            qsv = commands[0][1]
            self.assertEqual(qsv[qsv.index("-qsv_device") + 1], "/dev/dri/renderD129")
            self.assertIn("h264_qsv", qsv)
            self.assertIn("aac", qsv)
            self.assertEqual(qsv[qsv.index("-af") + 1], "asetpts=PTS-STARTPTS")

    def test_height_resolution_preserves_camera_aspect_ratio_with_even_dimensions(self) -> None:
        self.assertEqual(
            MediaExportManager._target_dimensions({"width": 2560, "height": 1920}, 1080),
            (1440, 1080),
        )
        self.assertEqual(
            MediaExportManager._target_dimensions({"width": 1920, "height": 1080}, 720),
            (1280, 720),
        )
        self.assertEqual(
            MediaExportManager._target_dimensions({"width": 896, "height": 672}, 0),
            (896, 672),
        )

    def test_recording_with_multiple_spans_is_joined_as_one_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index, start in enumerate((100.0, 120.0), start=1):
                segment = root / f"segment-{index}.mp4"
                segment.write_bytes(b"source")
                rows.append({
                    "path": str(segment), "start_epoch": start,
                    "end_epoch": start + 10.0, "duration_seconds": 10.0,
                    "stream_fingerprint": f"revision-{index}",
                })
            recorder = FakeRecorder(rows)
            manager = MediaExportManager(
                root / "storage", root / "database", recorder=lambda: recorder,
                ffmpeg_path=lambda: "ffmpeg", hardware_backend=lambda: "cpu",
            )
            manager._probe_source = lambda _path: {  # type: ignore[method-assign]
                "width": 1280, "height": 720, "fps": 20.0, "has_audio": False,
            }
            commands: list[list[str]] = []

            def fake_run(
                command: list[str], _cancel: threading.Event, timeout: float, *, process_name: str = ""
            ) -> None:
                self.assertGreater(timeout, 0)
                self.assertEqual(process_name, "survng-export")
                commands.append(command)
                Path(command[-1]).write_bytes(b"export")

            manager._run_process = fake_run  # type: ignore[method-assign]
            manager._valid_video_output = lambda _path: True  # type: ignore[method-assign]
            work = root / "work"
            work.mkdir()
            job = manager.store.create({
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0, "end_epoch": 130.0, "options": {},
            })

            output, gaps = manager._build_recording(job, rows, work, threading.Event())

            self.assertEqual(output.name, "recording.mp4")
            self.assertEqual(len(commands), 3)
            self.assertEqual(commands[-1][commands[-1].index("-c") + 1], "copy")
            self.assertEqual(gaps[0]["duration_seconds"], 10.0)

    def test_media_encoder_uses_role_specific_executable_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ffmpeg = root / "ffmpeg"
            ffmpeg.write_bytes(b"binary")
            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: str(ffmpeg),
                hardware_backend=lambda: "cpu",
            )

            export_command = manager._named_process_command(
                [str(ffmpeg), "-version"], "survng-export"
            )
            timelapse_command = manager._named_process_command(
                [str(ffmpeg), "-version"], "survng-timelapse"
            )

            self.assertEqual(Path(export_command[0]).name, "survng-export")
            self.assertEqual(Path(timelapse_command[0]).name, "survng-timelapse")
            self.assertEqual(Path(export_command[0]).resolve(), ffmpeg.resolve())
            self.assertEqual(Path(timelapse_command[0]).resolve(), ffmpeg.resolve())

    def test_cancel_or_delete_removes_completed_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage",
                root / "database",
                recorder=lambda: FakeRecorder([]),
                ffmpeg_path=lambda: "ffmpeg",
                hardware_backend=lambda: "cpu",
            )
            job = manager.store.create({
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0, "end_epoch": 110.0, "options": {},
            })
            manager.recording_dir.mkdir(parents=True)
            output = manager.recording_dir / "result.mp4"
            output.write_bytes(b"media")
            manager.store.update(
                str(job["id"]), status="completed", output_path=str(output),
                output_name="result.mp4", size_bytes=5,
            )

            result = manager.cancel_or_delete(str(job["id"]))

            self.assertEqual(result, {"id": job["id"], "deleted": True})
            self.assertFalse(output.exists())
            self.assertIsNone(manager.get(str(job["id"])))

    def test_protected_export_survives_expiry_and_storage_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage",
                root / "database",
                recorder=lambda: FakeRecorder([]),
                ffmpeg_path=lambda: "ffmpeg",
                hardware_backend=lambda: "cpu",
                max_storage_bytes=1,
            )
            job = manager.store.create({
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0, "end_epoch": 110.0, "options": {},
            })
            manager.recording_dir.mkdir(parents=True)
            output = manager.recording_dir / "protected.mp4"
            output.write_bytes(b"protected-media")
            manager.store.update(
                str(job["id"]),
                status="completed",
                output_path=str(output),
                output_name=output.name,
                size_bytes=output.stat().st_size,
                expires_at="2000-01-01T00:00:00+00:00",
            )

            protected = manager.set_protected(str(job["id"]), True)
            manager.cleanup()

            self.assertTrue(protected["protected"])
            self.assertEqual(protected["expires_at"], "")
            self.assertTrue(output.exists())
            self.assertIsNotNone(manager.get(str(job["id"])))

    def test_protected_export_requires_force_for_manual_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: "ffmpeg",
                hardware_backend=lambda: "cpu",
            )
            job = manager.store.create({
                "kind": "timelapse", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0, "end_epoch": 110.0, "options": {},
            })
            manager.timelapse_dir.mkdir(parents=True)
            output = manager.timelapse_dir / "protected.mp4"
            output.write_bytes(b"media")
            manager.store.update(
                str(job["id"]), status="completed", output_path=str(output),
                output_name=output.name, size_bytes=5,
            )
            manager.set_protected(str(job["id"]), True)

            with self.assertRaises(PermissionError):
                manager.cancel_or_delete(str(job["id"]))
            result = manager.cancel_or_delete(str(job["id"]), force=True)

            self.assertTrue(result["deleted"])
            self.assertFalse(output.exists())

    def test_export_store_filters_and_counts_library_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = MediaExportStore(Path(temporary))
            gate = store.create({
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0, "end_epoch": 110.0, "options": {},
                "label": "Driveway arrival", "origin": "assistant",
            })
            foyer = store.create({
                "kind": "timelapse", "camera_id": "foyer", "source": "live",
                "start_epoch": 200.0, "end_epoch": 220.0, "options": {},
            })
            store.update(str(gate["id"]), status="completed", protected=1)
            store.update(str(foyer["id"]), status="running")

            protected = store.list(camera_id="gate", protected=True)
            active = store.list(status="active")

            self.assertEqual([row["id"] for row in protected], [gate["id"]])
            self.assertEqual(protected[0]["label"], "Driveway arrival")
            self.assertEqual(protected[0]["origin"], "assistant")
            self.assertEqual([row["id"] for row in active], [foyer["id"]])
            self.assertEqual(store.count(kind="timelapse"), 1)
            summary = store.summary()
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["active"], 1)
            self.assertEqual(summary["protected"], 1)

    def test_batch_updates_labels_protection_and_deletes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: "ffmpeg",
                hardware_backend=lambda: "cpu",
            )
            jobs = [manager.store.create({
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0 + index, "end_epoch": 110.0 + index,
                "options": {},
            }) for index in range(2)]
            for job in jobs:
                manager.store.update(str(job["id"]), status="completed")

            renamed = manager.set_label(str(jobs[0]["id"]), "Front gate clip")
            protected = manager.batch([str(job["id"]) for job in jobs], "protect")
            deleted = manager.batch([str(job["id"]) for job in jobs], "delete")

            self.assertEqual(renamed["label"], "Front gate clip")
            self.assertEqual(len(protected["results"]), 2)
            self.assertTrue(all(item["protected"] for item in protected["results"]))
            self.assertEqual(len(deleted["results"]), 2)
            self.assertEqual(manager.count(), 0)

    def test_queue_full_failure_receives_retention_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage", root / "database",
                recorder=lambda: FakeRecorder([]), ffmpeg_path=lambda: "ffmpeg",
                hardware_backend=lambda: "cpu",
            )
            for index in range(100):
                manager._queue.put_nowait(f"occupied-{index}")
            job = manager.store.create({
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0, "end_epoch": 110.0, "options": {},
            })

            with self.assertRaises(RuntimeError):
                manager._enqueue(str(job["id"]))

            failed = manager.store.get(str(job["id"]))
            self.assertEqual(failed["status"], "failed")
            self.assertTrue(failed["expires_at"])

    def test_cancel_after_render_prevents_export_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "segment.mp4"
            segment.write_bytes(b"source")
            recorder = FakeRecorder([{
                "path": str(segment), "start_epoch": 100.0,
                "end_epoch": 110.0, "duration_seconds": 10.0,
            }])
            manager = MediaExportManager(
                root / "storage", root / "database", recorder=lambda: recorder,
                ffmpeg_path=lambda: "ffmpeg", hardware_backend=lambda: "cpu",
            )
            job = manager.store.create({
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0, "end_epoch": 110.0, "options": {},
            })
            cancel = threading.Event()

            def finish_then_cancel(*_args: object) -> tuple[Path, list[dict[str, float]]]:
                output = root / "rendered.mp4"
                output.write_bytes(b"rendered")
                cancel.set()
                return output, []

            manager._build_recording = finish_then_cancel  # type: ignore[method-assign]
            with self.assertRaises(InterruptedError):
                manager._execute(job, cancel)

            self.assertFalse(any(manager.recording_dir.iterdir()))

    def test_manifest_failure_removes_published_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "segment.mp4"
            segment.write_bytes(b"source")
            recorder = FakeRecorder([{
                "path": str(segment), "start_epoch": 100.0,
                "end_epoch": 110.0, "duration_seconds": 10.0,
            }])
            manager = MediaExportManager(
                root / "storage", root / "database", recorder=lambda: recorder,
                ffmpeg_path=lambda: "ffmpeg", hardware_backend=lambda: "cpu",
            )
            job = manager.store.create({
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0, "end_epoch": 110.0, "options": {},
            })
            rendered = root / "rendered.mp4"
            rendered.write_bytes(b"rendered")
            manager._build_recording = lambda *_args: (rendered, [])  # type: ignore[method-assign]
            manager._write_manifest = Mock(side_effect=OSError("disk full"))  # type: ignore[method-assign]

            with self.assertRaises(OSError):
                manager._execute(job, threading.Event())

            self.assertFalse(any(manager.recording_dir.iterdir()))
            self.assertFalse(any(manager.manifest_dir.iterdir()))

    def test_restart_removes_export_published_before_completion_was_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage", root / "database", recorder=lambda: FakeRecorder([]),
                ffmpeg_path=lambda: "ffmpeg", hardware_backend=lambda: "cpu",
            )
            job = manager.store.create({
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0, "end_epoch": 110.0, "options": {},
            })
            manager.store.update(str(job["id"]), status="running", phase="Finalizing")
            manager.recording_dir.mkdir(parents=True)
            manager.manifest_dir.mkdir(parents=True)
            orphan = manager.recording_dir / f"{job['id']}-gate-19700101-000140-recording.mp4"
            orphan.write_bytes(b"published before crash")
            manifest = manager.manifest_dir / f"{job['id']}.json"
            manifest.write_text("{}", encoding="utf-8")

            recovered = MediaExportManager(
                root / "storage", root / "database", recorder=lambda: FakeRecorder([]),
                ffmpeg_path=lambda: "ffmpeg", hardware_backend=lambda: "cpu",
            )

            self.assertFalse(orphan.exists())
            self.assertFalse(manifest.exists())
            failed = recovered.store.get(str(job["id"]))
            self.assertEqual(failed["status"], "failed")

    def test_publish_syncs_file_and_directory_before_returning_final_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage", root / "database", recorder=lambda: FakeRecorder([]),
                ffmpeg_path=lambda: "ffmpeg", hardware_backend=lambda: "cpu",
            )
            source = root / "rendered.mp4"
            source.write_bytes(b"rendered")
            job = {"id": "job-sync", "camera_id": "gate", "kind": "recording", "start_epoch": 100.0}

            with patch("survng.app.media_exports.os.fsync") as fsync:
                final, _name = manager._publish(job, source, threading.Event())

            self.assertTrue(final.exists())
            self.assertGreaterEqual(fsync.call_count, 2)

    def test_cleanup_claim_prevents_late_protection_from_reporting_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = MediaExportManager(
                root / "storage", root / "database", recorder=lambda: FakeRecorder([]),
                ffmpeg_path=lambda: "ffmpeg", hardware_backend=lambda: "cpu",
            )
            job = manager.store.create({
                "kind": "recording", "camera_id": "gate", "source": "main",
                "start_epoch": 100.0, "end_epoch": 110.0, "options": {},
            })
            manager.store.update(str(job["id"]), status="completed")

            claimed = manager.store.claim_unprotected_cleanup(str(job["id"]))

            self.assertIsNotNone(claimed)
            with self.assertRaisesRegex(RuntimeError, "currently being deleted"):
                manager.set_protected(str(job["id"]), True)


if __name__ == "__main__":
    unittest.main()

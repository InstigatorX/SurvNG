from __future__ import annotations

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
            self.assertTrue(manager.recording_dir.is_dir())
            self.assertTrue(manager.timelapse_dir.is_dir())

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

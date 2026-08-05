from __future__ import annotations

import tempfile
import struct
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException

from survng.app import main


class RecordingApiTest(unittest.TestCase):
    @staticmethod
    def _box(box_type: bytes, payload: bytes) -> bytes:
        return struct.pack(">I4s", len(payload) + 8, box_type) + payload

    def test_fragment_timestamp_offset_updates_media_in_place(self) -> None:
        tkhd = bytearray(16)
        struct.pack_into(">I", tkhd, 12, 1)
        mdhd = bytearray(16)
        struct.pack_into(">I", mdhd, 12, 1000)
        init_data = self._box(
            b"moov",
            self._box(b"trak", self._box(b"tkhd", bytes(tkhd)) + self._box(b"mdia", self._box(b"mdhd", bytes(mdhd)))),
        )
        tfhd = bytes(4) + struct.pack(">I", 1)
        tfdt = bytes(4) + struct.pack(">I", 100)
        media_data = self._box(b"moof", self._box(b"traf", self._box(b"tfhd", tfhd) + self._box(b"tfdt", tfdt)))
        with tempfile.TemporaryDirectory() as tmpdir:
            init_path = Path(tmpdir) / "init.mp4"
            media_path = Path(tmpdir) / "media.m4s"
            init_path.write_bytes(init_data)
            media_path.write_bytes(media_data)

            main._offset_fmp4_timestamps(init_path, media_path, 2.0)
            updated = media_path.read_bytes()

        tfdt_position = updated.index(b"tfdt") + 8
        self.assertEqual(struct.unpack_from(">I", updated, tfdt_position)[0], 2100)

    def test_recording_range_rejects_non_finite_and_out_of_bounds_epochs(self) -> None:
        with self.assertRaises(HTTPException) as non_finite:
            main._validate_recording_range(float("nan"), 100.0, 90_000, "invalid range")
        self.assertEqual(non_finite.exception.status_code, 400)

        with self.assertRaises(HTTPException) as out_of_bounds:
            main._validate_recording_range(0.0, 100_000.0, 90_000, "invalid range")
        self.assertEqual(out_of_bounds.exception.status_code, 400)

    def test_direct_fragment_lookup_validates_range_before_scanning_recordings(self) -> None:
        manager = SimpleNamespace(camera=lambda _camera_id: object())
        with patch.object(main, "manager", manager), patch.object(main, "_recording_day_rows") as recording_rows:
            with self.assertRaises(HTTPException) as invalid:
                main._recording_day_fmp4_paths(
                    "gate",
                    "segment.mp4",
                    100.0,
                    float("inf"),
                )

        self.assertEqual(invalid.exception.status_code, 400)
        recording_rows.assert_not_called()

    def test_event_clip_cache_key_preserves_explicit_zero_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SimpleNamespace(storage_dir=Path(tmpdir))
            with patch.object(main, "manager", manager):
                path = main._event_clip_path(
                    {"id": 42, "camera_id": "Front Door"},
                    before=0.0,
                    after=2.0,
                )

        self.assertTrue(path.name.startswith("42-0-2000-a3-"))
        self.assertIn("front-door", path.parts)

    def test_recording_day_reads_both_sources_from_index_only(self) -> None:
        recorder = Mock()
        recorder.recording_availability_between.side_effect = (
            lambda _camera_id, _start, _end, source, **_kwargs: {
                "ranges": [{"start_epoch": 100.0, "end_epoch": 110.0}] if source == "main" else [],
                "segment_count": 1 if source == "main" else 0,
            }
        )
        manager = SimpleNamespace(
            camera=lambda _camera_id: object(),
            recorder=recorder,
            events=SimpleNamespace(for_camera_range=lambda *_args, **_kwargs: []),
        )

        with patch.object(main, "manager", manager):
            payload = main.recording_day("gate", 100.0, 200.0, "main")

        self.assertEqual(payload["available_sources"], ["main"])
        self.assertEqual(recorder.recording_availability_between.call_count, 2)
        for call in recorder.recording_availability_between.call_args_list:
            self.assertIs(call.kwargs["discover_missing"], False)

    def test_recording_grid_day_aggregates_cameras_from_local_index_only(self) -> None:
        recorder = Mock()
        available = {"ranges": [{"start_epoch": 100.0, "end_epoch": 110.0}], "segment_count": 1}
        unavailable = {"ranges": [], "segment_count": 0}
        recorder.recording_grid_availability_between.return_value = {
            "gate": {"main": unavailable, "live": available},
            "garage": {"main": available, "live": unavailable},
        }
        events = SimpleNamespace(between_compact=Mock(return_value=[]))
        manager = SimpleNamespace(
            recorder=recorder,
            events=events,
        )
        config = SimpleNamespace(
            cameras=[
                SimpleNamespace(id="gate", name="Gate"),
                SimpleNamespace(id="garage", name="Garage"),
            ],
        )

        with patch.object(main, "manager", manager), patch.object(main, "config", config):
            payload = main.recording_grid_day(100.0, 200.0, "live")

        self.assertEqual(payload["view"], "all_cameras")
        self.assertEqual(payload["available_sources"], ["live", "main"])
        self.assertEqual(len(payload["cameras"]), 2)
        self.assertEqual(payload["cameras"][0]["recording_count"], 1)
        self.assertEqual(payload["cameras"][1]["recording_count"], 0)
        self.assertEqual(payload["recordings"][0]["camera_id"], "gate")
        events.between_compact.assert_called_once()
        recorder.recording_grid_availability_between.assert_called_once_with(
            ["gate", "garage"], 100.0, 200.0,
        )
        recorder.recording_availability_between.assert_not_called()

    def test_recording_day_groups_events_into_thumbnail_incidents(self) -> None:
        recorder = Mock()
        recorder.recording_availability_between.return_value = {
            "ranges": [{"start_epoch": 100.0, "end_epoch": 200.0}],
            "segment_count": 10,
        }
        events = [
            {
                "id": 8,
                "camera_id": "gate",
                "kind": "object",
                "created_at": "1970-01-01T00:02:10+00:00",
                "snapshot_path": "snapshots/gate/8.jpg",
                "objects_json": '[{"label":"car","confidence":0.91}]',
            },
            {
                "id": 7,
                "camera_id": "gate",
                "kind": "motion",
                "created_at": "1970-01-01T00:02:00+00:00",
                "snapshot_path": "snapshots/gate/7.jpg",
                "objects_json": "[]",
            },
        ]
        manager = SimpleNamespace(
            camera=lambda _camera_id: object(),
            recorder=recorder,
            events=SimpleNamespace(for_camera_range=lambda *_args, **_kwargs: events),
        )

        with patch.object(main, "manager", manager):
            payload = main.recording_day("gate", 100.0, 200.0, "main")

        self.assertEqual(len(payload["events"]), 2)
        self.assertEqual(len(payload["incidents"]), 1)
        self.assertTrue(payload["incidents"][0]["has_objects"])
        self.assertEqual(payload["incidents"][0]["labels"], ["car"])
        self.assertEqual(payload["incidents"][0]["representative_event_id"], 8)

    def test_create_media_export_uses_shared_manager_and_configured_source(self) -> None:
        export_manager = Mock()
        export_manager.create.return_value = {
            "id": "job-1",
            "status": "queued",
            "download_url": "",
        }
        manager = SimpleNamespace(camera=lambda camera_id: object() if camera_id == "gate" else None)
        request = main.MediaExportRequest(
            kind="timelapse",
            camera_id="gate",
            source="live",
            start_epoch=100.0,
            end_epoch=400.0,
            sample_interval_seconds=10,
            output_fps=24,
            width=1920,
            height=1080,
        )

        with patch.object(main, "manager", manager), patch.object(main, "_media_export_manager", return_value=export_manager):
            payload = main.create_media_export(request)

        self.assertEqual(payload["id"], "job-1")
        export_manager.create.assert_called_once_with({
            "kind": "timelapse",
            "camera_id": "gate",
            "source": "live",
            "start_epoch": 100.0,
            "end_epoch": 400.0,
            "options": {
                "sample_interval_seconds": 10.0,
                "output_fps": 24,
                "height": 1080,
            },
            "label": "",
            "origin": "manual",
        })

    def test_media_export_rejects_overlong_clip_before_queueing(self) -> None:
        export_manager = Mock()
        manager = SimpleNamespace(camera=lambda _camera_id: object())
        request = main.MediaExportRequest(
            kind="recording",
            camera_id="gate",
            start_epoch=100.0,
            end_epoch=100.0 + 24 * 60 * 60 + 1,
        )

        with patch.object(main, "manager", manager), patch.object(main, "_media_export_manager", return_value=export_manager):
            with self.assertRaises(HTTPException) as invalid:
                main.create_media_export(request)

        self.assertEqual(invalid.exception.status_code, 400)
        export_manager.create.assert_not_called()

    def test_timelapse_api_keeps_legacy_width_when_height_is_not_supplied(self) -> None:
        export_manager = Mock()
        export_manager.create.return_value = {"id": "legacy-width", "status": "queued"}
        manager = SimpleNamespace(camera=lambda _camera_id: object())
        request = main.MediaExportRequest(
            kind="timelapse",
            camera_id="gate",
            start_epoch=100.0,
            end_epoch=400.0,
            width=1920,
        )

        with patch.object(main, "manager", manager), patch.object(
            main, "_media_export_manager", return_value=export_manager
        ):
            main.create_media_export(request)

        options = export_manager.create.call_args.args[0]["options"]
        self.assertEqual(options["width"], 1920)
        self.assertNotIn("height", options)

    def test_public_media_export_applies_proxy_base_path_to_download(self) -> None:
        with patch.object(main.config, "base_path", "/survng"):
            payload = main._public_media_export({
                "id": "job-1",
                "status": "completed",
                "download_url": "/api/exports/job-1/download",
                "media_url": "/api/exports/job-1/media",
            })

        self.assertEqual(payload["download_url"], "/survng/api/exports/job-1/download")
        self.assertEqual(payload["media_url"], "/survng/api/exports/job-1/media")

    def test_export_library_api_applies_filters_and_reports_total(self) -> None:
        export_manager = Mock()
        export_manager.list.return_value = [{"id": "job-1", "download_url": ""}]
        export_manager.count.return_value = 7

        with patch.object(main, "_media_export_manager", return_value=export_manager):
            payload = main.list_media_exports(
                limit=25,
                offset=5,
                camera_id="gate",
                kind="timelapse",
                status="completed",
                protected=True,
            )

        expected_filters = {
            "camera_id": "gate",
            "kind": "timelapse",
            "status": "completed",
            "protected": True,
        }
        export_manager.list.assert_called_once_with(25, offset=5, **expected_filters)
        export_manager.count.assert_called_once_with(**expected_filters)
        self.assertEqual(payload["total"], 7)
        self.assertEqual(payload["offset"], 5)

    def test_export_protection_api_updates_manager(self) -> None:
        export_manager = Mock()
        export_manager.set_protected.return_value = {
            "id": "job-1", "protected": True, "download_url": "",
        }

        with patch.object(main, "_media_export_manager", return_value=export_manager):
            payload = main.protect_media_export(
                "job-1", main.MediaExportProtectionRequest(protected=True)
            )

        export_manager.set_protected.assert_called_once_with("job-1", True)
        self.assertTrue(payload["protected"])

    def test_export_metadata_and_summary_apis_use_shared_manager(self) -> None:
        export_manager = Mock()
        export_manager.set_label.return_value = {
            "id": "job-1", "label": "Gate delivery", "download_url": "",
        }
        export_manager.summary.return_value = {
            "total": 3, "bytes": 1024, "protected": 1,
        }

        with patch.object(main, "_media_export_manager", return_value=export_manager):
            renamed = main.update_media_export_metadata(
                "job-1", main.MediaExportMetadataRequest(label=" Gate delivery ")
            )
            summary = main.media_export_summary()

        export_manager.set_label.assert_called_once_with("job-1", " Gate delivery ")
        self.assertEqual(renamed["label"], "Gate delivery")
        self.assertEqual(summary["protected"], 1)

    def test_export_batch_api_rewrites_public_media_urls(self) -> None:
        export_manager = Mock()
        export_manager.batch.return_value = {
            "action": "protect",
            "results": [{
                "id": "job-1",
                "protected": True,
                "download_url": "/api/exports/job-1/download",
                "media_url": "/api/exports/job-1/media",
            }],
            "errors": [],
        }

        with patch.object(main.config, "base_path", "/survng"), patch.object(
            main, "_media_export_manager", return_value=export_manager
        ):
            payload = main.batch_media_exports(main.MediaExportBatchRequest(
                ids=["job-1"], action="protect"
            ))

        export_manager.batch.assert_called_once_with(["job-1"], "protect")
        self.assertEqual(
            payload["results"][0]["media_url"],
            "/survng/api/exports/job-1/media",
        )

    def test_protected_export_delete_requires_force(self) -> None:
        export_manager = Mock()
        export_manager.cancel_or_delete.side_effect = PermissionError("job-1")

        with patch.object(main, "_media_export_manager", return_value=export_manager):
            with self.assertRaises(HTTPException) as protected:
                main.delete_media_export("job-1")

        self.assertEqual(protected.exception.status_code, 409)
        export_manager.cancel_or_delete.assert_called_once_with("job-1", force=False)

    def test_recording_updates_requests_async_edge_refresh_then_reads_index_only(self) -> None:
        recorder = Mock()
        recorder.recording_availability_between.return_value = {
            "ranges": [{"start_epoch": 180.0, "end_epoch": 200.0}],
            "segment_count": 2,
        }
        events = Mock()
        events.for_camera_range.return_value = []
        manager = SimpleNamespace(
            camera=lambda _camera_id: object(),
            recorder=recorder,
            events=events,
        )

        with patch.object(main, "manager", manager):
            payload = main.recording_updates("gate", 100.0, 200.0, 190.0, "main")

        recorder.request_recording_edge_refresh.assert_called_once_with(
            "gate",
            "main",
            190.0,
        )
        recorder.refresh_recording_edge.assert_not_called()
        self.assertEqual(payload["availability"], [{"start_epoch": 180.0, "end_epoch": 200.0}])
        self.assertIs(
            recorder.recording_availability_between.call_args.kwargs["discover_missing"],
            False,
        )
        event_start = datetime.fromisoformat(events.for_camera_range.call_args.args[1]).timestamp()
        self.assertEqual(event_start, 100.0)

    def test_recording_updates_keep_five_minute_overlap_for_late_incidents(self) -> None:
        recorder = Mock()
        recorder.recording_availability_between.return_value = {
            "ranges": [], "segment_count": 0,
        }
        events = Mock()
        events.for_camera_range.return_value = []
        manager = SimpleNamespace(
            camera=lambda _camera_id: object(), recorder=recorder, events=events,
        )

        with patch.object(main, "manager", manager):
            main.recording_updates("gate", 100.0, 1000.0, 900.0, "main")

        event_start = datetime.fromisoformat(events.for_camera_range.call_args.args[1]).timestamp()
        self.assertEqual(event_start, 600.0)

    def test_recording_preview_uses_quantized_local_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recordings_dir = root / "recordings"
            recordings_dir.mkdir()
            source_path = recordings_dir / "segment.mp4"
            source_path.write_bytes(b"recording")
            manager = SimpleNamespace(
                database_dir=root / "database",
                recorder=SimpleNamespace(recordings_dir=recordings_dir),
            )

            def create_preview(command: list[str], **_kwargs: object) -> SimpleNamespace:
                Path(command[-1]).write_bytes(b"jpeg")
                return SimpleNamespace(returncode=0, stderr=b"")

            row = {
                "path": str(source_path),
                "start_epoch": 100.0,
                "end_epoch": 110.0,
            }
            with (
                patch.object(main, "manager", manager),
                patch.object(main.subprocess, "run", side_effect=create_preview) as run,
                patch.object(main, "_maintain_recording_preview_cache"),
            ):
                first = main._recording_preview_path(row, 107.9)
                second = main._recording_preview_path(row, 109.8)

            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), b"jpeg")
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("-ss") + 1], "5.000")
            self.assertIn(str(manager.database_dir / "recording-preview-cache"), str(first))

    def test_recording_preview_reports_index_gap_without_storage_scan(self) -> None:
        recorder = Mock()
        recorder.recording_rows_between.return_value = []
        manager = SimpleNamespace(
            camera=lambda _camera_id: object(),
            recorder=recorder,
        )
        with patch.object(main, "manager", manager):
            with self.assertRaises(HTTPException) as missing:
                main.recording_preview("gate", 100.0, "main")

        self.assertEqual(missing.exception.status_code, 404)
        recorder.recording_rows_between.assert_called_once_with(
            "gate",
            99.999,
            100.001,
            "main",
            discover_missing=False,
        )

    def test_fresh_recording_rows_bypass_stale_cache_and_lease_result(self) -> None:
        stale = [{"path": "/recordings/stale.mp4", "size_bytes": 2048}]
        current = [{"path": "/recordings/current.mp4", "size_bytes": 4096}]
        recorder = Mock()
        recorder.recording_rows_between.return_value = current
        recorder.discard_missing_recording_rows.return_value = current
        manager = SimpleNamespace(recorder=recorder)
        cache_key = ("gate", "main", 100, 200)

        with (
            patch.object(main, "manager", manager),
            patch.object(main, "RECORDING_DAY_CACHE", {cache_key: (100.0, stale)}),
            patch.object(main.time, "monotonic", return_value=101.0),
        ):
            rows = main._recording_day_rows(
                "gate",
                100.0,
                200.0,
                "main",
                fresh=True,
            )

        self.assertEqual(rows, current)
        recorder.recording_rows_between.assert_called_once_with(
            "gate",
            100.0,
            200.0,
            "main",
            discover_missing=False,
        )
        recorder.discard_missing_recording_rows.assert_called_once_with(current)
        recorder.lease_recordings_for_playback.assert_called_once_with(current)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import struct
import unittest
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

    def test_recording_updates_requests_async_edge_refresh_then_reads_index_only(self) -> None:
        recorder = Mock()
        recorder.recording_availability_between.return_value = {
            "ranges": [{"start_epoch": 180.0, "end_epoch": 200.0}],
            "segment_count": 2,
        }
        manager = SimpleNamespace(
            camera=lambda _camera_id: object(),
            recorder=recorder,
            events=SimpleNamespace(for_camera_range=lambda *_args, **_kwargs: []),
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

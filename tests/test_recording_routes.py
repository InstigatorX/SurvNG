from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest import TestCase
from urllib.parse import parse_qs, urlparse

from fastapi import HTTPException, Response

from survng.app.recording_routes import (
    RecordingRouteDependencies,
    create_recording_router,
)
from survng.app.manager_access import ManagerAccessCoordinator
from survng.app.encoded_fragments import (
    EncodedFragment,
    StreamDescription,
)
from survng.app.recording_media import RECORDING_FMP4_VERSION


class _Recorder:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def recording_availability_between(self, *_args, **_kwargs) -> dict:
        return {
            "ranges": [{"marker": self.marker}],
            "segment_count": 1,
        }


class _Events:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def for_camera_range(self, *_args, **_kwargs) -> list[dict]:
        return []


class _Manager:
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.recorder = _Recorder(marker)
        self.events = _Events(marker)
        self.config = SimpleNamespace(recording_segment_seconds=10, cameras=[])

    def camera(self, camera_id: str) -> object | None:
        return object() if camera_id == "gate" else None


def _dependencies(get_manager) -> RecordingRouteDependencies:
    return RecordingRouteDependencies(
        get_manager=get_manager,
        get_config=lambda: SimpleNamespace(
            recording_segment_seconds=10,
            cameras=[],
        ),
        get_media_exports=lambda: SimpleNamespace(),
        public_url=lambda value: value,
        recording_rows=lambda *_args, **_kwargs: [],
        recording_day_rows=lambda *_args, **_kwargs: [],
        recording_preview_path=lambda *_args, **_kwargs: Path("preview.jpg"),
        recording_preview_timestamp=lambda _path: (None, "requested_offset"),
        recording_segment_path=lambda *_args, **_kwargs: Path("segment.mp4"),
        recording_day_fmp4_paths=lambda *_args, **_kwargs: (
            Path("init.mp4"),
            Path("media.m4s"),
        ),
        recording_file_response=lambda *_args, **_kwargs: None,
        event_clip_window=lambda *_args: (5.0, 5.0),
        ensure_event_clip=lambda *_args, **_kwargs: Path("clip.mp4"),
    )


class RecordingRouteLifecycleTests(TestCase):
    def test_hls_media_response_prewarms_the_next_fragment(self) -> None:
        manager = _Manager("current")
        fragments = tuple(
            EncodedFragment(
                sequence=index,
                segment_name=f"segment-{index}.mp4",
                wall_start_epoch=100.0 + (index - 1) * 5,
                wall_end_epoch=105.0 + (index - 1) * 5,
                media_start_seconds=(index - 1) * 5,
                media_duration_seconds=5.0,
                keyframe_aligned=True,
                stream=StreamDescription(fingerprint="h264"),
                source_identity=f"revision-{index}",
                source_path=Path(f"segment-{index}.mp4"),
            )
            for index in (1, 2)
        )
        materialized = []

        class Source:
            def fragments(self, *_args, **_kwargs):
                return iter(fragments)

            def materialize(self, selected):
                materialized.append(selected)
                return replace(
                    selected,
                    stream=replace(
                        selected.stream,
                        init_path=Path("init.mp4"),
                    ),
                    media_path=Path(f"{selected.segment_name}.m4s"),
                )

        source = Source()
        dependencies = replace(
            _dependencies(lambda: manager),
            encoded_fragment_source=lambda _manager: source,
            recording_file_response=lambda path, media_type: Response(
                content=str(path),
                media_type=media_type,
            ),
        )
        response = create_recording_router(dependencies).handlers[
            "recording_day_hls_segment"
        ](
            "gate",
            "segment-1.mp4",
            100,
            110,
            "main",
            7.25,
            False,
            "revision-1",
        )

        self.assertEqual(
            [(item.segment_name, item.media_start_seconds) for item in materialized],
            [("segment-1.mp4", 7.25)],
        )
        self.assertIsNotNone(response.background)
        asyncio.run(response.background())
        self.assertEqual(
            [(item.segment_name, item.media_start_seconds) for item in materialized],
            [("segment-1.mp4", 7.25), ("segment-2.mp4", 12.25)],
        )

    def test_hls_fragment_urls_pin_identity_and_manifest_media_offset(self) -> None:
        manager = _Manager("current")
        fragment = EncodedFragment(
            sequence=1,
            segment_name="segment.mp4",
            wall_start_epoch=100.0,
            wall_end_epoch=105.0,
            media_start_seconds=0.0,
            media_duration_seconds=5.0,
            keyframe_aligned=True,
            stream=StreamDescription(fingerprint="h264"),
            source_identity="revision-123",
            source_path=Path("segment.mp4"),
        )
        materialized = []

        class Source:
            def fragments(self, *_args, **_kwargs):
                return iter((fragment,))

            def materialize(self, selected):
                materialized.append(selected)
                return replace(
                    selected,
                    stream=replace(
                        selected.stream,
                        init_path=Path("init.mp4"),
                    ),
                    media_path=Path("media.m4s"),
                )

        source = Source()
        dependencies = replace(
            _dependencies(lambda: manager),
            encoded_fragment_source=lambda _manager: source,
            recording_day_rows=lambda *_args, **_kwargs: self.fail(
                "legacy row path should not be used"
            ),
            recording_file_response=lambda path, media_type: (
                path,
                media_type,
            ),
        )
        handlers = create_recording_router(dependencies).handlers

        playlist = handlers["recording_day_hls_playlist"](
            "gate",
            100,
            110,
        )
        urls = [
            line.split('URI="', 1)[1].split('"', 1)[0]
            if line.startswith("#EXT-X-MAP:")
            else line
            for line in playlist.body.decode().splitlines()
            if line.startswith("#EXT-X-MAP:")
            or (line and not line.startswith("#"))
        ]
        self.assertEqual(len(urls), 2)
        for url in urls:
            self.assertEqual(
                parse_qs(urlparse(url).query)["fragment_id"],
                ["revision-123"],
            )

        response = handlers["recording_day_hls_segment"](
            "gate",
            "segment.mp4",
            100,
            110,
            "main",
            7.25,
            False,
            "revision-123",
        )
        self.assertEqual(response, (Path("media.m4s"), "video/iso.segment"))
        self.assertEqual(materialized[-1].media_start_seconds, 7.25)

        with self.assertRaises(HTTPException) as stale:
            handlers["recording_day_hls_segment"](
                "gate",
                "segment.mp4",
                100,
                110,
                "main",
                7.25,
                False,
                "stale-revision",
            )
        self.assertEqual(stale.exception.status_code, 409)

    def test_day_playlist_emits_ext_x_start_for_native_safari(self) -> None:
        manager = _Manager("current")
        rows = [{
            "name": "segment.mp4",
            "start_epoch": 100.0,
            "end_epoch": 110.0,
            "duration_seconds": 10.0,
            "stream_fingerprint": "h264",
        }]
        handlers = create_recording_router(
            replace(
                _dependencies(lambda: manager),
                recording_day_rows=lambda *_args, **_kwargs: rows,
            )
        ).handlers
        body = handlers["recording_day_hls_playlist"](
            "gate",
            100,
            110,
            "main",
            7.25,
        ).body.decode()
        self.assertIn("#EXT-X-START:TIME-OFFSET=7.250,PRECISE=YES", body)
        # Keep the tag ahead of media so Safari can honor it during load.
        self.assertLess(
            body.index("#EXT-X-START:"),
            body.index("#EXTINF:"),
        )
        clipped = handlers["recording_day_hls_playlist"](
            "gate",
            100,
            110,
            "main",
            99.0,
        ).body.decode()
        self.assertIn("#EXT-X-START:TIME-OFFSET=9.990,PRECISE=YES", clipped)

    def test_day_and_event_playlists_version_every_fragment_url(self) -> None:
        manager = _Manager("current")
        manager.events.get = lambda _event_id: {
            "id": 1, "camera_id": "gate", "created_at": "1970-01-01T00:01:45+00:00",
        }
        rows = [{
            "name": f"segment-{index}.mp4", "start_epoch": 100 + index * 5,
            "end_epoch": 105 + index * 5, "duration_seconds": 5,
            "stream_fingerprint": "",  # exercise every independent init too
        } for index in range(2)]
        dependencies = replace(
            _dependencies(lambda: manager),
            recording_day_rows=lambda *_args, **_kwargs: rows,
        )
        handlers = create_recording_router(dependencies).handlers
        for kind, response in (
            ("day", handlers["recording_day_hls_playlist"]("gate", 100, 110)),
            ("event", handlers["event_stream"](1)),
        ):
            with self.subTest(kind=kind):
                urls = []
                for line in response.body.decode().splitlines():
                    if line.startswith("#EXT-X-MAP:"):
                        urls.append(line.split('URI="', 1)[1].split('"', 1)[0])
                    elif line and not line.startswith("#"):
                        urls.append(line)
                self.assertEqual(len(urls), 4)
                for url in urls:
                    query = parse_qs(urlparse(url).query)
                    self.assertEqual(query["v"], [str(RECORDING_FMP4_VERSION)])
                    self.assertIn(query["media_offset"], (["0.000"], ["5.000"]))
                    if kind == "event":
                        self.assertEqual(query["trim_end"], ["true"])

    def test_native_segment_uses_indexed_epoch_lookup(self) -> None:
        manager = _Manager("current")
        requested: list[tuple[object, ...]] = []
        dependencies = replace(
            _dependencies(lambda: manager),
            recording_segment_path=lambda *args: (
                requested.append(args) or Path("segment.mp4")
            ),
        )

        response = create_recording_router(dependencies).handlers[
            "recording_segment"
        ]("gate", 105.0, "live")

        self.assertEqual(requested, [(manager, "gate", 105.0, "live")])
        self.assertEqual(response.media_type, "video/mp4")
        self.assertEqual(response.headers["cache-control"], "private, max-age=3600")

    def test_mobile_segment_transcodes_only_the_selected_indexed_segment(self) -> None:
        manager = _Manager("current")
        row = {"path": "segment.mp4", "start_epoch": 150, "end_epoch": 180, "duration_seconds": 30}
        manager.recorder.recording_rows_between = lambda *args, **kwargs: [row]
        calls = []
        dependencies = replace(_dependencies(lambda: manager),
            ensure_event_clip=lambda *args, **kwargs: (calls.append((args, kwargs)) or Path("mobile.mp4")))
        response = create_recording_router(dependencies).handlers["recording_segment"]("gate", 155, "main", mobile=True)
        event = calls[0][0][1]
        self.assertEqual(event["_recording_rows"], [row])
        self.assertEqual(calls[0][1]["after"], 30)
        self.assertEqual(calls[0][1]["before"], 0)
        self.assertEqual(response.path, Path("mobile.mp4"))

    def test_mobile_window_disables_browser_cache_for_growing_windows(self) -> None:
        dependencies = _dependencies(lambda: _Manager("current"))
        response = create_recording_router(dependencies).handlers["recording_mobile_window"]("gate", 155, "main")
        self.assertEqual(response.headers["cache-control"], "private, no-store")

    def test_native_segment_rejects_invalid_epoch_before_lookup(self) -> None:
        dependencies = _dependencies(lambda: _Manager("current"))
        with self.assertRaises(HTTPException) as invalid:
            create_recording_router(dependencies).handlers["recording_segment"](
                "gate", float("nan"), "main"
            )

        self.assertEqual(invalid.exception.status_code, 400)

    def test_exact_preview_reports_requested_and_decoded_timestamps(self) -> None:
        recorder = SimpleNamespace(
            recording_rows_between=lambda *_args, **_kwargs: [{
                "path": "/recordings/gate.mp4",
                "start_epoch": 100.0,
                "end_epoch": 110.0,
            }]
        )
        manager = SimpleNamespace(
            camera=lambda camera_id: object() if camera_id == "gate" else None,
            recorder=recorder,
        )
        dependencies = replace(
            _dependencies(lambda: manager),
            recording_preview_path=lambda *_args, **_kwargs: Path("preview.jpg"),
            recording_preview_timestamp=lambda _path: (107.909, "source_pts"),
        )

        response = create_recording_router(dependencies).handlers[
            "recording_preview"
        ]("gate", 107.9, "main", 1280, True)

        self.assertEqual(response.headers["x-survng-requested-timestamp"], "107.900000")
        self.assertEqual(response.headers["x-survng-actual-timestamp"], "107.909000")
        self.assertEqual(response.headers["x-survng-timestamp-source"], "source_pts")

    def test_request_holds_manager_lease_while_query_runs(self) -> None:
        active_manager = _Manager("current")
        access = ManagerAccessCoordinator()
        lock = threading.RLock()

        def recording_rows(manager, *_args, **_kwargs):
            self.assertIs(manager, active_manager)
            self.assertEqual(access.active_leases(active_manager), 1)
            return []

        dependencies = replace(
            _dependencies(lambda: active_manager),
            recording_rows=recording_rows,
            manager_lock=lock,
            manager_access=access,
        )

        create_recording_router(dependencies).handlers["recordings"](
            "gate", 10, "main"
        )
        self.assertEqual(access.active_leases(active_manager), 0)

    def test_recording_helper_receives_request_manager_snapshot(self) -> None:
        active_manager = _Manager("current")
        helper_managers: list[_Manager] = []
        dependencies = replace(
            _dependencies(lambda: active_manager),
            recording_rows=lambda manager, *_args, **_kwargs: (
                helper_managers.append(manager) or []
            ),
        )

        bundle = create_recording_router(dependencies)
        bundle.handlers["recordings"]("gate", 10, "main")

        self.assertEqual(helper_managers, [active_manager])

    def test_request_snapshots_one_manager_generation(self) -> None:
        managers = [_Manager("old"), _Manager("new")]
        calls = 0

        def get_manager() -> _Manager:
            nonlocal calls
            manager = managers[min(calls, 1)]
            calls += 1
            return manager

        bundle = create_recording_router(_dependencies(get_manager))
        response = bundle.handlers["recording_day"](
            "gate",
            100.0,
            200.0,
            "main",
        )

        self.assertEqual(calls, 1)
        self.assertEqual(response["recordings"], [{"marker": "old"}])

    def test_new_request_observes_new_manager_generation(self) -> None:
        active = _Manager("old")

        def get_manager() -> _Manager:
            return active

        bundle = create_recording_router(_dependencies(get_manager))
        first = bundle.handlers["recording_day"]("gate", 100.0, 200.0, "main")
        active = _Manager("new")
        second = bundle.handlers["recording_day"]("gate", 100.0, 200.0, "main")

        self.assertEqual(first["recordings"], [{"marker": "old"}])
        self.assertEqual(second["recordings"], [{"marker": "new"}])

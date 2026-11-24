from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
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
        # Recording files start on keyframes (segment muxer, stream copy), so
        # the playlist declares independent segments in its header.
        header = body.split("#EXT-X-MAP:", 1)[0].splitlines()
        self.assertIn("#EXT-X-INDEPENDENT-SEGMENTS", header)
        self.assertEqual(header[-1], "#EXT-X-START:TIME-OFFSET=7.250,PRECISE=YES")
        clipped = handlers["recording_day_hls_playlist"](
            "gate",
            100,
            110,
            "main",
            99.0,
        ).body.decode()
        self.assertIn("#EXT-X-START:TIME-OFFSET=9.990,PRECISE=YES", clipped)

    def test_day_playlist_grows_only_while_its_window_can_receive_recordings(
        self,
    ) -> None:
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

        def playlist(now: float, growing: bool, start: float | None = None) -> str:
            with patch("survng.app.recording_routes.time.time", return_value=now):
                return handlers["recording_day_hls_playlist"](
                    "gate", 0, 900, "main", start, growing,
                ).body.decode()

        # Open window: append-only, no end marker, so the player re-reads it
        # instead of the page loading a new playlist at the live edge.
        open_window = playlist(500.0, True, 9.0)
        self.assertIn("#EXT-X-PLAYLIST-TYPE:EVENT", open_window)
        self.assertNotIn("#EXT-X-ENDLIST", open_window)
        self.assertIn("#EXT-X-START:TIME-OFFSET=9.000,PRECISE=YES", open_window)
        # An unfinished playlist with no start tag opens near its end.
        self.assertIn(
            "#EXT-X-START:TIME-OFFSET=0.000,PRECISE=YES",
            playlist(500.0, True),
        )
        # The target duration may not change between re-reads. With only a
        # 10.0 s segment listed so far, a 10.04 s one can still follow.
        self.assertIn("#EXT-X-TARGETDURATION:11", open_window)
        rows.append({
            "name": "later.mp4", "start_epoch": 110.0, "end_epoch": 120.04,
            "duration_seconds": 10.04, "stream_fingerprint": "h264",
        })
        self.assertIn("#EXT-X-TARGETDURATION:11", playlist(510.0, True))
        rows.pop()
        # The last recording of a window is indexed after the window ends.
        self.assertNotIn("#EXT-X-ENDLIST", playlist(929.0, True))
        # Closed window: the playlist ends so playback moves to the next one.
        closed = playlist(931.0, True)
        self.assertIn("#EXT-X-PLAYLIST-TYPE:VOD", closed)
        self.assertTrue(closed.rstrip().endswith("#EXT-X-ENDLIST"))
        self.assertNotIn("#EXT-X-START", closed)
        # Not requested: unchanged finished playlist, even for an open window.
        default = playlist(500.0, False)
        self.assertIn("#EXT-X-PLAYLIST-TYPE:VOD", default)
        self.assertTrue(default.rstrip().endswith("#EXT-X-ENDLIST"))
        self.assertNotIn("#EXT-X-START", default)
        # A finished playlist is read once and keeps its exact value.
        self.assertIn("#EXT-X-TARGETDURATION:10", default)

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

    def test_preview_strip_prefers_the_sub_stream_and_describes_its_tiles(self) -> None:
        recorded = {
            "live": [{"path": "/recordings/gate-sub.mp4", "start_epoch": 103.0, "end_epoch": 113.0}],
            "main": [{"path": "/recordings/gate.mp4", "start_epoch": 100.0, "end_epoch": 110.0}],
        }
        asked: list[str] = []

        def rows_between(_camera_id, _start, _end, source, **_kwargs):
            asked.append(source)
            return recorded[source]

        manager = SimpleNamespace(
            camera=lambda camera_id: object() if camera_id == "gate" else None,
            recorder=SimpleNamespace(recording_rows_between=rows_between),
        )
        built: list[tuple[str, int]] = []

        def strip(_manager, row, *, width):
            built.append((row["path"], width))
            return Path("strip.jpg"), {"count": 10, "columns": 6, "rows": 2, "step_seconds": 1}

        handler = create_recording_router(
            replace(_dependencies(lambda: manager), recording_preview_strip=strip)
        ).handlers["recording_preview_strip"]

        response = handler("gate", 107.9, "main", 320)

        # The sub stream shows the same scene and decodes several times faster.
        self.assertEqual(built, [("/recordings/gate-sub.mp4", 320)])
        self.assertEqual(response.headers["x-survng-strip-source"], "live")
        # The tiles are placed by this segment's own start, not the main one's.
        self.assertEqual(response.headers["x-survng-strip-start"], "103.000")
        self.assertEqual(response.headers["x-survng-strip-end"], "113.000")
        self.assertEqual(response.headers["x-survng-strip-step"], "1")
        self.assertEqual(response.headers["x-survng-strip-count"], "10")
        self.assertEqual(response.headers["x-survng-strip-columns"], "6")
        self.assertEqual(response.headers["x-survng-strip-rows"], "2")

        # No sub recording at that time: the stream being played is used.
        built.clear()
        response = handler("gate", 101.0, "main", 320)
        self.assertEqual(built, [("/recordings/gate.mp4", 320)])
        self.assertEqual(response.headers["x-survng-strip-source"], "main")
        self.assertEqual(response.headers["x-survng-strip-start"], "100.000")

        # The sub stream is indexed but its file has gone: the stream being
        # played still has this time.
        def strip_without_sub(_manager, row, *, width):
            if "sub" in row["path"]:
                raise HTTPException(status_code=404, detail="recording file not found")
            return strip(_manager, row, width=width)

        fallback = create_recording_router(
            replace(_dependencies(lambda: manager), recording_preview_strip=strip_without_sub)
        ).handlers["recording_preview_strip"]
        built.clear()
        response = fallback("gate", 107.9, "main", 320)
        self.assertEqual(built, [("/recordings/gate.mp4", 320)])
        self.assertEqual(response.headers["x-survng-strip-source"], "main")
        # With no other stream to turn to, the reason is reported as it was.
        def strip_forbidden(_manager, row, *, width):
            raise HTTPException(status_code=403, detail="outside media storage")

        with self.assertRaises(HTTPException) as gone:
            create_recording_router(
                replace(_dependencies(lambda: manager), recording_preview_strip=strip_forbidden)
            ).handlers["recording_preview_strip"]("gate", 107.9, "live", 320)
        self.assertEqual(gone.exception.status_code, 403)
        # A busy generator is not a reason to try the other stream, which
        # would only add to the load.
        def busy_sub(_manager, row, *, width):
            if "sub" in row["path"]:
                raise HTTPException(status_code=429, detail="busy")
            return strip(_manager, row, width=width)

        built.clear()
        with self.assertRaises(HTTPException) as refused:
            create_recording_router(
                replace(_dependencies(lambda: manager), recording_preview_strip=busy_sub)
            ).handlers["recording_preview_strip"]("gate", 107.9, "main", 320)
        self.assertEqual(refused.exception.status_code, 429)
        self.assertEqual(built, [])

        # Nothing recorded there, or a row that ends before it: a gap.
        with self.assertRaises(HTTPException) as gap:
            handler("gate", 150.0, "main", 320)
        self.assertEqual(gap.exception.status_code, 404)
        # The answer says how far the gap reaches: from the end of the last
        # recording on either stream (the sub stream's, at 113) to the next
        # one, or to the edge of the window looked at.
        self.assertEqual(gap.exception.headers["X-SurvNG-Gap-Start"], "113.000")
        self.assertEqual(gap.exception.headers["X-SurvNG-Gap-End"], "450.000")
        # At the live edge the gap is not said to reach into the future:
        # what is being recorded now is only not indexed yet.
        with patch("survng.app.recording_routes.time.time", return_value=152.5):
            with self.assertRaises(HTTPException) as live_edge:
                handler("gate", 150.0, "main", 320)
        self.assertEqual(live_edge.exception.headers["X-SurvNG-Gap-End"], "152.500")
        # A segment indexed between the two lookups: no gap after all, and
        # nothing is claimed about one.
        lookups = {"count": 0}

        def rows_then_indexed(_camera_id, start, end, source, **_kwargs):
            lookups["count"] += 1
            if end - start < 1:
                return []
            return [{"path": "/recordings/new.mp4", "start_epoch": 145.0, "end_epoch": 155.0}]

        racing = SimpleNamespace(
            camera=lambda camera_id: object() if camera_id == "gate" else None,
            recorder=SimpleNamespace(recording_rows_between=rows_then_indexed),
        )
        with self.assertRaises(HTTPException) as raced:
            create_recording_router(
                replace(_dependencies(lambda: racing), recording_preview_strip=strip)
            ).handlers["recording_preview_strip"]("gate", 150.0, "main", 320)
        self.assertEqual(raced.exception.status_code, 404)
        self.assertIsNone(raced.exception.headers)
        recorded["main"].append({"path": "/recordings/later.mp4", "start_epoch": 160.0, "end_epoch": 170.0})
        with self.assertRaises(HTTPException) as bounded:
            handler("gate", 150.0, "main", 320)
        self.assertEqual(bounded.exception.headers["X-SurvNG-Gap-End"], "160.000")
        recorded["main"].pop()
        asked.clear()
        handler("gate", 107.9, "live", 320)
        self.assertEqual(asked, ["live"])
        with self.assertRaises(HTTPException) as bad_width:
            handler("gate", 107.9, "main", 4000)
        self.assertEqual(bad_width.exception.status_code, 400)

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

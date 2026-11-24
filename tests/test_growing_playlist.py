"""Growing day playlists stay append-only while storage misbehaves."""

from __future__ import annotations

import errno
import json
import logging
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from fastapi import HTTPException

from survng.app.encoded_fragments import (
    EncodedFragment,
    FragmentUnavailable,
    FragmentWindow,
    IndexedMp4FragmentSource,
    StreamDescription,
)
from survng.app.growing_playlists import (
    GrowingPlaylistKey,
    GrowingPlaylistRecord,
    GrowingPlaylistRegistry,
    RateLimitedLog,
    merge_rows,
)
from survng.app.manager_access import ManagerAccessCoordinator
from survng.app.recording_routes import (
    RecordingRouteDependencies,
    create_recording_router,
)

FIXTURES = Path(__file__).parent / "fixtures" / "day_playlists_v1_2.json"
WINDOW = (0.0, 900.0)
OPEN = 500.0  # the window can still receive recordings
CLOSED = 931.0  # past end + max(30, 3 * 10 s)
LOGGER_NAME = "survng.app.recording_routes"


class _Manager:
    def __init__(self, marker: str, segment_seconds: float = 10.0) -> None:
        self.marker = marker
        self.config = SimpleNamespace(
            recording_segment_seconds=segment_seconds,
            cameras=[],
        )

    def camera(self, camera_id: str) -> object | None:
        return object() if camera_id == "gate" else None


class _Storage:
    """A recording index over real files, with injectable storage faults."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.rows: list[dict] = []
        self.failing: set[str] = set()
        self.remuxed: list[tuple[str, float, float]] = []

    def add(self, name: str, start: float, duration: float = 10.0) -> None:
        path = self.root / name
        path.write_bytes(b"x" * 2048)
        self.rows.append({
            "name": name,
            "path": str(path),
            "start_epoch": start,
            "end_epoch": start + duration,
            "duration_seconds": duration,
            "stream_fingerprint": "h264",
        })

    def row(self, name: str) -> dict:
        return next(row for row in self.rows if row["name"] == name)

    def load(self, _camera, start, end, _source) -> list[dict]:
        return [
            dict(row) for row in self.rows
            if row["end_epoch"] > start and row["start_epoch"] < end
        ]

    def resolve(self, value: object) -> Path:
        # Mirrors the production resolver: a file that is gone is an HTTP 404
        # from the resolver itself; a flaky stat is an OSError.
        path = Path(str(value))
        if path.name in self.failing:
            raise OSError(errno.EIO, "Input/output error", str(path))
        if not path.exists():
            raise HTTPException(status_code=404, detail="recording file not found")
        return path

    def remux(self, path: Path, duration: float, offset: float):
        self.remuxed.append((path.name, duration, offset))
        return self.root / "init.mp4", self.root / "media.m4s"

    def source(self) -> IndexedMp4FragmentSource:
        return IndexedMp4FragmentSource(
            row_loader=self.load,
            path_resolver=self.resolve,
            remuxer=self.remux,
        )


class _Harness:
    def __init__(self, root: Path, *, guarded: bool = False) -> None:
        self.storage = _Storage(root)
        self.manager = _Manager("first")
        dependencies = RecordingRouteDependencies(
            get_manager=lambda: self.manager,
            get_config=lambda: self.manager.config,
            get_media_exports=lambda: SimpleNamespace(),
            public_url=lambda value: value,
            recording_rows=lambda *_args, **_kwargs: [],
            recording_day_rows=lambda *_args, **_kwargs: self.fail_legacy(),
            recording_preview_path=lambda *_args, **_kwargs: Path("p.jpg"),
            recording_preview_timestamp=lambda _path: (None, ""),
            recording_segment_path=lambda *_args, **_kwargs: Path("s.mp4"),
            recording_day_fmp4_paths=lambda *_args, **_kwargs: (
                Path("init.mp4"),
                Path("media.m4s"),
            ),
            recording_file_response=lambda path, media_type: (path, media_type),
            event_clip_window=lambda *_args: (5.0, 5.0),
            ensure_event_clip=lambda *_args, **_kwargs: Path("clip.mp4"),
            encoded_fragment_source=lambda _manager: self.storage.source(),
        )
        if guarded:
            dependencies = replace(
                dependencies,
                manager_access=ManagerAccessCoordinator(),
                manager_lock=threading.RLock(),
            )
        bundle = create_recording_router(dependencies)
        self.handlers = bundle.handlers
        self.registry = bundle.growing_playlists

    @staticmethod
    def fail_legacy() -> list[dict]:
        raise AssertionError("the legacy row path must not be used")

    def playlist(
        self,
        now: float = OPEN,
        *,
        growing: bool = True,
        reload: str = "1-0",
        start: float | None = None,
    ) -> str:
        with patch("survng.app.recording_routes.time.time", return_value=now):
            return self.handlers["recording_day_hls_playlist"](
                "gate", *WINDOW, "main", start, growing, reload,
            ).body.decode()

    def segment(self, url: str, kind: str = "media.m4s", now: float = OPEN):
        with patch("survng.app.recording_routes.time.time", return_value=now):
            return self._segment(url, kind)

    def _segment(self, url: str, kind: str):
        parsed = urlparse(url)
        name = parsed.path.split("/")[2]
        query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        handler = self.handlers[
            "recording_day_hls_segment" if kind == "media.m4s"
            else "recording_day_hls_init"
        ]
        return handler(
            "gate",
            name,
            float(query["start_epoch"]),
            float(query["end_epoch"]),
            query["source"],
            float(query["media_offset"]),
            query.get("trim_end") == "true",
            query.get("fragment_id", ""),
            query.get("growing") == "true",
        )


def _media_lines(body: str) -> list[str]:
    return [line for line in body.splitlines() if line.startswith("day/")]


def _entry_lines(body: str) -> list[str]:
    lines = body.splitlines()
    start = next(
        index for index, line in enumerate(lines)
        if line.startswith(("#EXT-X-MAP:", "#EXT-X-PROGRAM-DATE-TIME:"))
    )
    return [line for line in lines[start:] if line != "#EXT-X-ENDLIST"]


def _offset(url: str) -> str:
    return parse_qs(urlparse(url).query)["media_offset"][0]


class GrowingPlaylistTests(TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.harness = _Harness(self.root)
        self.storage = self.harness.storage

    def test_stat_error_on_served_segment_keeps_entry_and_logs_once(self) -> None:
        self.storage.add("a.mp4", 100.0)
        self.storage.add("b.mp4", 110.0)
        first = self.harness.playlist()
        self.storage.failing.add("b.mp4")
        with self.assertLogs(LOGGER_NAME, logging.WARNING) as logs:
            again = self.harness.playlist()
            third = self.harness.playlist()
        self.assertEqual(again, first)
        self.assertEqual(third, first)
        kept = [line for line in logs.output if "could not be checked" in line]
        self.assertEqual(len(kept), 1, logs.output)
        self.assertIn("b.mp4", kept[0])
        # A later recording is still appended while b.mp4 stays unreadable,
        # and it can be fetched although another row of its window cannot.
        self.storage.add("c.mp4", 120.0)
        grown = self.harness.playlist()
        self.assertEqual(_entry_lines(grown)[: len(_entry_lines(first))], _entry_lines(first))
        urls = _media_lines(grown)
        self.assertEqual(len(urls), 3)
        self.assertEqual(_offset(urls[2]), "20.000")
        self.assertEqual(
            self.harness.segment(urls[2]),
            (self.root / "media.m4s", "video/iso.segment"),
        )
        # The unreadable entry's own segment is a clean 404, not a hang.
        with self.assertRaises(HTTPException) as unreadable:
            self.harness.segment(urls[1])
        self.assertEqual(unreadable.exception.status_code, 404)

    def test_earlier_row_changes_or_disappears_between_re_reads(self) -> None:
        for index, name in enumerate(("a.mp4", "b.mp4", "c.mp4")):
            self.storage.add(name, 100.0 + 10.0 * index)
        first = self.harness.playlist()
        served_b = _media_lines(first)[1]
        # The index measures b.mp4 (10.4 s, not the nominal 10 s) and drops
        # a.mp4 (validation marked it unplayable; its file is gone too).
        self.storage.row("b.mp4")["duration_seconds"] = 10.4
        self.storage.rows.remove(self.storage.row("a.mp4"))
        (self.root / "a.mp4").unlink()
        self.storage.add("d.mp4", 130.4)
        with self.assertLogs(LOGGER_NAME, logging.WARNING) as logs:
            grown = self.harness.playlist()
        self.assertEqual(
            _entry_lines(grown)[: len(_entry_lines(first))],
            _entry_lines(first),
        )
        urls = _media_lines(grown)
        self.assertEqual([urlparse(url).path.split("/")[2] for url in urls],
                         ["a.mp4", "b.mp4", "c.mp4", "d.mp4"])
        # The new entry continues the served media timeline (3 x 10 s).
        self.assertEqual(_offset(urls[3]), "30.000")
        self.assertTrue(any("no longer in the recording index" in line for line in logs.output))
        self.assertTrue(any("since revised" in line for line in logs.output))
        # The revised entry is served as it was listed, not refused with 409.
        with self.assertLogs(LOGGER_NAME, logging.WARNING):
            self.harness.segment(served_b)
        self.assertEqual(self.storage.remuxed[-1], ("b.mp4", 10.0, 10.0))
        # The entry whose file has gone fails on its own, cleanly.
        with self.assertRaises(HTTPException) as gone:
            self.harness.segment(urls[0])
        self.assertEqual(gone.exception.status_code, 404)

    def test_revision_check_still_applies_to_entries_never_served(self) -> None:
        self.storage.add("a.mp4", 100.0)
        url = _media_lines(self.harness.playlist())[0]
        stale = url.replace("fragment_id=", "fragment_id=stale")
        with self.assertRaises(HTTPException) as refused:
            self.harness.segment(stale)
        self.assertEqual(refused.exception.status_code, 409)
        # Non-growing segment URLs keep the strict revision check.
        plain = _media_lines(self.harness.playlist(growing=False))[0]
        self.storage.row("a.mp4")["duration_seconds"] = 10.4
        with self.assertRaises(HTTPException) as strict:
            self.harness.segment(plain)
        self.assertEqual(strict.exception.status_code, 409)

    def test_first_read_with_an_unreadable_row_fails_and_stores_nothing(self) -> None:
        self.storage.add("a.mp4", 100.0)
        self.storage.add("b.mp4", 110.0)
        self.storage.failing.add("a.mp4")
        with self.assertRaises(HTTPException) as failed:
            self.harness.playlist()
        self.assertEqual(failed.exception.status_code, 404)
        self.assertEqual(list(self.harness.registry.keys()), [])
        # Once the storage recovers the playlist is served from both rows.
        self.storage.failing.clear()
        self.assertEqual(len(_media_lines(self.harness.playlist())), 2)

    def test_new_unreadable_row_holds_back_later_rows_until_readable(self) -> None:
        self.storage.add("a.mp4", 100.0)
        first = self.harness.playlist()
        self.storage.add("b.mp4", 110.0)
        self.storage.add("c.mp4", 120.0)
        self.storage.failing.add("b.mp4")
        self.assertEqual(self.harness.playlist(), first)
        self.storage.failing.clear()
        urls = _media_lines(self.harness.playlist())
        self.assertEqual([_offset(url) for url in urls], ["0.000", "10.000", "20.000"])

    def test_window_closing_finishes_the_served_playlist_and_drops_it(self) -> None:
        self.storage.add("a.mp4", 100.0)
        self.storage.add("b.mp4", 110.0)
        first = self.harness.playlist()
        self.assertIn("#EXT-X-PLAYLIST-TYPE:EVENT", first)
        self.assertNotIn("#EXT-X-ENDLIST", first)
        self.storage.rows.remove(self.storage.row("a.mp4"))
        self.storage.add("c.mp4", 120.0)
        closed = self.harness.playlist(CLOSED)
        self.assertIn("#EXT-X-PLAYLIST-TYPE:VOD", closed)
        self.assertTrue(closed.endswith("#EXT-X-ENDLIST\n"))
        self.assertEqual(
            _entry_lines(closed)[: len(_entry_lines(first))],
            _entry_lines(first),
        )
        self.assertEqual(len(_media_lines(closed)), 3)
        self.assertEqual(list(self.harness.registry.keys()), [])
        # With the record gone the next read is an ordinary finished one.
        after = self.harness.playlist(CLOSED)
        self.assertNotIn("a.mp4", after)
        self.assertNotIn("growing=true", after)

    def test_record_expires_after_its_window_can_no_longer_grow(self) -> None:
        self.storage.add("a.mp4", 100.0)
        self.harness.playlist()
        self.assertEqual(len(list(self.harness.registry.keys())), 1)
        with patch("survng.app.growing_playlists.time.time", return_value=10_000.0):
            self.assertIsNone(self.harness.registry.served_entry(
                ("gate", "main", "0.000", "900.000"), "a.mp4", "x",
            ))
            self.assertEqual(list(self.harness.registry.keys()), [])

    def test_non_growing_requests_are_byte_for_byte_unchanged(self) -> None:
        expected = json.loads(FIXTURES.read_text())
        specs = [
            ("seg 1.mp4", 100.0, 10.0, "h264", "id-a"),
            ("seg2.mp4", 110.0, 10.04, "h264", "id-b"),
            ("seg3.mp4", 120.04, 9.5, "hevc", "id-c"),
        ]
        fragments = []
        offset = 0.0
        for sequence, (name, start, duration, fingerprint, identity) in enumerate(specs, 1):
            fragments.append(EncodedFragment(
                sequence=sequence, segment_name=name, wall_start_epoch=start,
                wall_end_epoch=start + duration, media_start_seconds=offset,
                media_duration_seconds=duration, keyframe_aligned=True,
                stream=StreamDescription(fingerprint=fingerprint),
                source_identity=identity, source_path=Path(name),
            ))
            offset += duration
        legacy_rows = [
            {"name": name, "start_epoch": start, "end_epoch": start + duration,
             "duration_seconds": duration, "stream_fingerprint": fingerprint}
            for name, start, duration, fingerprint, _identity in specs
        ]

        class Source:
            def fragments(self, *_args, **_kwargs):
                return iter(fragments)

            def materialize(self, fragment):
                return fragment

        manager = _Manager("current")
        manager.events = SimpleNamespace(get=lambda _id: {
            "id": 1, "camera_id": "gate",
            "created_at": "1970-01-01T00:01:55+00:00",
        })
        base = RecordingRouteDependencies(
            get_manager=lambda: manager,
            get_config=lambda: manager.config,
            get_media_exports=lambda: None,
            public_url=lambda value: "https://nvr.example" + value,
            recording_rows=lambda *_args, **_kwargs: [],
            recording_day_rows=lambda *_args, **_kwargs: [dict(row) for row in legacy_rows],
            recording_preview_path=None,
            recording_preview_timestamp=None,
            recording_segment_path=None,
            recording_day_fmp4_paths=None,
            recording_file_response=None,
            event_clip_window=lambda *_args: (10.0, 10.0),
            ensure_event_clip=None,
        )
        for tag, dependencies in (
            ("source", replace(base, encoded_fragment_source=lambda _m: Source())),
            ("legacy", base),
        ):
            handlers = create_recording_router(dependencies).handlers
            playlist = handlers["recording_day_hls_playlist"]
            # A growing playlist of the same window must not affect them.
            with patch("survng.app.recording_routes.time.time", return_value=500.0):
                playlist("gate", 100, 900, "main", None, True)
            with self.subTest(tag=tag):
                with patch("survng.app.recording_routes.time.time", return_value=500.0):
                    self.assertEqual(
                        playlist("gate", 100, 900, "main", 7.25, False).body.decode(),
                        expected[f"day_{tag}"],
                    )
                    self.assertEqual(
                        playlist("gate", 100, 900).body.decode(),
                        expected[f"day_default_{tag}"],
                    )
                with patch("survng.app.recording_routes.time.time", return_value=5000.0):
                    self.assertEqual(
                        playlist("gate", 100, 900, "main", None, True).body.decode(),
                        expected[f"day_closed_growing_{tag}"],
                    )
                self.assertEqual(
                    handlers["event_stream"](1).body.decode(),
                    expected[f"event_{tag}"],
                )

    def test_non_growing_segment_still_fails_on_any_unreadable_row(self) -> None:
        self.storage.add("a.mp4", 100.0)
        self.storage.add("b.mp4", 110.0)
        url = _media_lines(self.harness.playlist(growing=False))[1]
        self.storage.failing.add("a.mp4")
        with self.assertRaises(HTTPException) as failed:
            self.harness.segment(url)
        self.assertEqual(failed.exception.status_code, 404)
        with self.assertRaises(HTTPException):
            self.harness.playlist(growing=False)

    def test_record_survives_a_manager_reload(self) -> None:
        harness = _Harness(self.root, guarded=True)
        harness.storage.add("a.mp4", 100.0)
        first = harness.playlist()
        self.assertIn("#EXT-X-TARGETDURATION:11", first)
        # A config reload replaces the manager (here with a shorter segment).
        harness.manager = _Manager("second", segment_seconds=4.0)
        harness.storage.add("b.mp4", 110.0, 4.0)
        grown = harness.playlist()
        self.assertTrue(grown.startswith(first))
        self.assertIn("#EXT-X-TARGETDURATION:11", grown)
        self.assertEqual(_offset(_media_lines(grown)[1]), "10.000")
        self.assertEqual(
            harness.segment(_media_lines(grown)[1]),
            (self.root / "media.m4s", "video/iso.segment"),
        )

    def test_interleaved_viewers_each_keep_their_own_playlist(self) -> None:
        self.storage.add("a.mp4", 100.0)
        self.storage.add("b.mp4", 110.0)
        viewer_one = self.harness.playlist(reload="1-0")
        self.storage.row("a.mp4")["duration_seconds"] = 10.4
        viewer_two = self.harness.playlist(reload="1-1")
        self.assertNotEqual(viewer_one, viewer_two)
        self.assertEqual(_offset(_media_lines(viewer_two)[1]), "10.400")
        self.storage.add("c.mp4", 120.4)
        one_again = self.harness.playlist(reload="1-0")
        two_again = self.harness.playlist(reload="1-1")
        self.assertTrue(one_again.startswith(viewer_one))
        self.assertTrue(two_again.startswith(viewer_two))
        self.assertEqual(_offset(_media_lines(one_again)[2]), "20.000")
        self.assertEqual(_offset(_media_lines(two_again)[2]), "20.400")
        # A viewer re-reading the same URL shares its record.
        self.assertEqual(self.harness.playlist(reload="1-0"), one_again)


class GrowingPlaylistUnitTests(TestCase):
    def _row(self, name: str, start: float, *, available: bool = True) -> dict:
        fragment = EncodedFragment(
            sequence=1, segment_name=name, wall_start_epoch=start,
            wall_end_epoch=start + 10.0, media_start_seconds=0.0,
            media_duration_seconds=10.0, keyframe_aligned=True,
            stream=StreamDescription(fingerprint="h264"),
            source_identity=f"id-{name}" if available else "",
            source_path=Path(name), available=available,
        )
        return {
            "name": name, "start_epoch": start, "end_epoch": start + 10.0,
            "duration_seconds": 10.0, "stream_fingerprint": "h264",
            "_encoded_fragment": fragment,
        }

    def _record(self, name: str = "k") -> GrowingPlaylistRecord:
        return GrowingPlaylistRecord(
            key=GrowingPlaylistKey("gate", "main", "0.000", "900.000", "0.000", name),
            header=("#EXTM3U", "#EXT-X-PLAYLIST-TYPE:EVENT"),
            query="q",
            target_duration=11,
            expires_at=float("inf"),
        )

    def test_unreadable_new_row_is_skipped_after_the_hold(self) -> None:
        record = self._record()
        rows = [self._row("a", 100.0), self._row("b", 110.0, available=False),
                self._row("c", 120.0)]
        fingerprints = ["h264"] * 3
        report = merge_rows(record, rows, fingerprints, now_monotonic=0.0, hold_seconds=30.0)
        self.assertEqual(report.appended, ("a",))
        self.assertEqual(report.held, ("b",))
        report = merge_rows(record, rows, fingerprints, now_monotonic=29.0, hold_seconds=30.0)
        self.assertEqual(report.appended, ())
        report = merge_rows(record, rows, fingerprints, now_monotonic=31.0, hold_seconds=30.0)
        self.assertEqual(report.skipped, ("b",))
        self.assertEqual(report.appended, ("c",))
        # Readable again later, it can no longer be inserted.
        rows[1] = self._row("b", 110.0)
        report = merge_rows(record, rows, fingerprints, now_monotonic=40.0)
        self.assertEqual(report.out_of_order, ("b",))
        self.assertEqual([entry.name for entry in record.entries], ["a", "c"])

    def test_registry_is_bounded_by_records_and_entries(self) -> None:
        registry = GrowingPlaylistRegistry(max_records=2, max_entries=3)
        with registry.lock:
            for name in ("one", "two", "three"):
                record = self._record(name)
                merge_rows(record, [self._row("a", 100.0)], ["h264"], now_monotonic=0.0)
                registry.put(record)
            self.assertEqual([key.reload for key in registry.keys()], ["two", "three"])
            big = self._record("big")
            merge_rows(
                big,
                [self._row(name, 100.0 + 10 * index) for index, name in enumerate("abcd")],
                ["h264"] * 4,
                now_monotonic=0.0,
            )
            evicted = registry.put(big)
        # A record larger than the whole budget is not kept.
        self.assertIn(big.key, evicted)
        self.assertNotIn(big.key, list(registry.keys()))

    def test_rate_limited_log_counts_suppressed_messages(self) -> None:
        now = [0.0]
        log = RateLimitedLog(logging.getLogger("growing-test"), interval_seconds=60.0,
                             clock=lambda: now[0])
        with self.assertLogs("growing-test", logging.WARNING) as logs:
            self.assertTrue(log.log(logging.WARNING, "kind", "gate", "message %s", 1))
            self.assertFalse(log.log(logging.WARNING, "kind", "gate", "message %s", 2))
            self.assertTrue(log.log(logging.WARNING, "kind", "porch", "message %s", 3))
            now[0] = 61.0
            self.assertTrue(log.log(logging.WARNING, "kind", "gate", "message %s", 4))
        self.assertEqual(len(logs.output), 3)
        self.assertIn("1 similar messages suppressed", logs.output[-1])

    def test_tolerant_listing_marks_unreadable_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = _Storage(Path(directory))
            storage.add("a.mp4", 100.0)
            storage.add("b.mp4", 110.0)
            storage.add("c.mp4", 120.0)
            storage.failing.add("b.mp4")
            (Path(directory) / "c.mp4").unlink()
            window = FragmentWindow("gate", "main", 0.0, 900.0)
            with self.assertRaises(FragmentUnavailable):
                list(storage.source().fragments(window))
            fragments = list(storage.source().fragments(window, tolerate_unavailable=True))
            self.assertEqual([fragment.available for fragment in fragments], [True, False, False])
            self.assertEqual(fragments[1].unavailable_reason, "OSError")
            self.assertEqual(fragments[2].unavailable_reason, "HTTPException")
            with self.assertRaises(FragmentUnavailable):
                storage.source().materialize(fragments[1])

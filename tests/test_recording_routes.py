from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest import TestCase

from survng.app.recording_routes import (
    RecordingRouteDependencies,
    create_recording_router,
)
from survng.app.manager_access import ManagerAccessCoordinator


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
        recording_day_fmp4_paths=lambda *_args, **_kwargs: (
            Path("init.mp4"),
            Path("media.m4s"),
        ),
        recording_file_response=lambda *_args, **_kwargs: None,
        event_clip_window=lambda *_args: (5.0, 5.0),
        ensure_event_clip=lambda *_args, **_kwargs: Path("clip.mp4"),
    )


class RecordingRouteLifecycleTests(TestCase):
    def test_day_hls_declares_each_independent_recording_a_discontinuity(self) -> None:
        rows = [
            {
                "name": "first.mp4",
                "start_epoch": 100.0,
                "duration_seconds": 10.0,
                "stream_fingerprint": "same-stream",
            },
            {
                "name": "second.mp4",
                "start_epoch": 110.0,
                "duration_seconds": 10.0,
                "stream_fingerprint": "same-stream",
            },
        ]
        bundle = create_recording_router(
            replace(
                _dependencies(lambda: _Manager("current")),
                recording_day_rows=lambda *_args, **_kwargs: rows,
            )
        )

        response = bundle.handlers["recording_day_hls_playlist"](
            "gate", 100.0, 120.0, "main"
        )
        playlist = response.body.decode("utf-8")

        self.assertEqual(playlist.count("#EXT-X-DISCONTINUITY"), 1)
        self.assertEqual(playlist.count("#EXT-X-MAP:"), 2)
        self.assertNotIn("media_offset=", playlist)
        self.assertLess(
            playlist.index("#EXT-X-DISCONTINUITY"),
            playlist.index('second.mp4/init.mp4'),
        )

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

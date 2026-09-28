"""Camera episode playback keeps source membership, bounds, and cache identity."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException

from survng.app import main
from survng.app.events import EventStore


class ScenePlaybackApiTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = EventStore(self.root)
        self.start = 1780000000.0
        person = {"label": "person", "confidence": .8,
                  "box": {"x1": 10, "y1": 10, "x2": 30, "y2": 60}}
        self.event = self.events.add_event(
            "gate", "motion", created_at=self.iso(self.start), objects_json=json.dumps([person]),
        )
        self.events.record_scene_observations(
            self.event["id"], [{**person, "captured_at_epoch": self.start + 7200}],
        )
        self.episode = self.events.scene_incident(event_id=self.event["id"])["episodes"][0]
        self.other = self.events.add_event(
            "driveway", "motion", created_at=self.iso(self.start), objects_json=json.dumps([person]),
        )
        self.other_episode = self.events.scene_incident(event_id=self.other["id"])["episodes"][0]
        self.manager = SimpleNamespace(
            events=self.events, storage_dir=self.root,
            config=SimpleNamespace(event_clip_before_seconds=5, event_clip_after_seconds=5),
        )
        self.clip = self.root / "clip.mp4"
        self.clip.write_bytes(b"test-media-response")

    @staticmethod
    def iso(epoch):
        return datetime.fromtimestamp(epoch, timezone.utc).isoformat()

    def test_later_mp4_chunk_uses_its_wall_clock_origin(self):
        start, end = self.start + 6300, self.start + 7200
        with patch.dict(main.__dict__, {"manager": self.manager}), patch.object(
            main._recording_media_runtime, "_ensure_event_clip", return_value=self.clip,
        ) as ensure:
            response = main.event_clip(
                self.event["id"], episode_id=self.episode["id"], start_epoch=start, end_epoch=end,
                before=100, after=100,
            )
        enriched = ensure.call_args.args[0]
        self.assertEqual(enriched["id"], self.event["id"])
        self.assertEqual(enriched["camera_id"], "gate")
        self.assertEqual(enriched["created_at"], self.iso(start))
        self.assertEqual(enriched["scene_clip_start_epoch"], start)
        self.assertEqual(ensure.call_args.kwargs["before"], 0)
        self.assertEqual(ensure.call_args.kwargs["after"], 900)
        self.assertEqual(response.path, self.clip)
        self.assertEqual(self.events.get(self.event["id"])["created_at"], self.iso(self.start), "playback must not rewrite evidence time")

    def test_later_hls_chunk_uses_matching_camera_and_manifest_window(self):
        start, end = self.start + 6300, self.start + 7200
        source = Mock()
        source.fragments.return_value = [SimpleNamespace(
            segment_name=f"gate-later-{index}.mp4", wall_start_epoch=start + index * 300, wall_end_epoch=start + (index + 1) * 300,
            media_duration_seconds=300, stream=SimpleNamespace(fingerprint="gate-codec"), source_identity=f"later-segment-{index}",
        ) for index in range(3)]
        with patch.dict(main.__dict__, {"manager": self.manager}), patch(
            "survng.app.recording_media_runtime.IndexedMp4FragmentSource", return_value=source,
        ):
            response = main.event_stream(
                self.event["id"], episode_id=self.episode["id"], start_epoch=start, end_epoch=end,
            )
        window = source.fragments.call_args.args[0]
        self.assertEqual((window.camera_id, window.source, window.start_epoch, window.end_epoch), ("gate", "main", start, end))
        playlist = response.body.decode()
        self.assertIn(f"start_epoch={start:.3f}&end_epoch={end:.3f}", playlist)
        self.assertIn(f"#EXT-X-PROGRAM-DATE-TIME:{self.iso(start)}", playlist)
        self.assertEqual(playlist.count("#EXTINF:300.000"), 3)
        self.assertNotIn("/cameras/driveway/", playlist)

    def test_wrong_episode_or_missing_event_rejects_before_media_work(self):
        for endpoint in (main.event_clip, main.event_stream):
            for event_id, episode_id in ((self.event["id"], self.other_episode["id"]),
                                         (self.other["id"], self.episode["id"]), (999999, self.episode["id"])):
                with self.subTest(endpoint=endpoint.__name__, event=event_id, episode=episode_id), patch.dict(main.__dict__, {"manager": self.manager}), patch.object(main._recording_media_runtime, "_ensure_event_clip") as ensure, patch(
                    "survng.app.recording_media_runtime.IndexedMp4FragmentSource",
                ) as fragments:
                    with self.assertRaises(HTTPException) as error:
                        endpoint(event_id, episode_id=episode_id, start_epoch=self.start, end_epoch=self.start + 1)
                    self.assertEqual(error.exception.status_code, 404)
                    ensure.assert_not_called()
                    fragments.assert_not_called()

    def test_invalid_windows_reject_before_media_work(self):
        cases = [
            (self.start - 1, self.start + 1), (self.start + 7199, self.start + 7201),
            (self.start, self.start + 3601), (self.start, self.start),
            (self.start + 10, self.start), (float("nan"), self.start + 1),
            (self.start, float("inf")), (float("-inf"), self.start),
        ]
        for endpoint in (main.event_clip, main.event_stream):
            for start, end in cases:
                with self.subTest(endpoint=endpoint.__name__, start=start, end=end), patch.dict(main.__dict__, {"manager": self.manager}), patch.object(main._recording_media_runtime, "_ensure_event_clip") as ensure, patch(
                    "survng.app.recording_media_runtime.IndexedMp4FragmentSource",
                ) as fragments:
                    with self.assertRaises(HTTPException) as error:
                        endpoint(self.event["id"], episode_id=self.episode["id"], start_epoch=start, end_epoch=end)
                    self.assertEqual(error.exception.status_code, 400)
                    ensure.assert_not_called()
                    fragments.assert_not_called()

    def test_partial_episode_parameters_reject_and_instantaneous_episode_allows_one_second(self):
        with patch.dict(main.__dict__, {"manager": self.manager}), patch.object(
            main._recording_media_runtime, "_ensure_event_clip", return_value=self.clip,
        ) as ensure:
            for kwargs in ({"episode_id": self.episode["id"]}, {"start_epoch": self.start, "end_epoch": self.start + 1}):
                with self.subTest(kwargs=kwargs), self.assertRaises(HTTPException) as error:
                    main.event_clip(self.event["id"], **kwargs)
                self.assertEqual(error.exception.status_code, 400)
            main.event_clip(self.other["id"], episode_id=self.other_episode["id"], start_epoch=self.start, end_epoch=self.start + 1)
        self.assertEqual(ensure.call_args.kwargs["after"], 1)

    def test_cache_paths_separate_chunk_origins_and_preserve_legacy_key(self):
        runtime = main._recording_media_runtime
        base = {"id": self.event["id"], "camera_id": "gate"}
        with patch.dict(main.__dict__, {"manager": self.manager}), patch.object(runtime, "_hardware_acceleration_mode", return_value="none"):
            legacy = runtime._event_clip_path(base, before=0, after=900)
            first = runtime._event_clip_path({**base, "scene_clip_start_epoch": self.start}, before=0, after=900)
            later = runtime._event_clip_path({**base, "scene_clip_start_epoch": self.start + 6300}, before=0, after=900)
            repeated = runtime._event_clip_path({**base, "scene_clip_start_epoch": self.start + 6300}, before=0, after=900)
        self.assertEqual(len({legacy, first, later}), 3)
        self.assertEqual(later, repeated)
        self.assertNotIn("-scene-", legacy.name)
        self.assertIn(f"-scene-{int((self.start + 6300) * 1000)}", later.name)


if __name__ == "__main__":
    unittest.main()

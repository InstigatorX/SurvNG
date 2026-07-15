from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from survng.app.events import EventStore


class EventStoreTest(unittest.TestCase):
    def test_compact_queries_omit_media_fields_and_support_keyset_paging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            events = [
                store.add_event(
                    camera_id="front-door",
                    kind="motion",
                    message=f"message-{index}",
                    snapshot_path=f"/snapshots/{index}.jpg",
                    recording_path=f"/recordings/{index}.mp4",
                    created_at="2026-07-15T12:00:00+00:00",
                )
                for index in range(7)
            ]

            first_page = store.recent_compact(3)
            cursor = first_page[-1]
            second_page = store.recent_compact(3, cursor["created_at"], int(cursor["id"]))

            self.assertEqual([row["id"] for row in first_page], [events[6]["id"], events[5]["id"], events[4]["id"]])
            self.assertEqual([row["id"] for row in second_page], [events[3]["id"], events[2]["id"], events[1]["id"]])
            self.assertEqual(
                set(first_page[0]),
                {"id", "camera_id", "kind", "objects_json", "created_at"},
            )

    def test_between_compact_returns_complete_range_without_media_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            for index in range(25):
                store.add_event(
                    camera_id="back-middle",
                    kind="motion",
                    message="large payload" * 100,
                    snapshot_path=f"/snapshots/{index}.jpg",
                    created_at=f"2026-07-15T12:00:{index:02d}+00:00",
                )

            rows = store.between_compact(
                "2026-07-15T12:00:00+00:00",
                "2026-07-15T12:01:00+00:00",
            )

            self.assertEqual(len(rows), 25)
            self.assertNotIn("message", rows[0])
            self.assertNotIn("snapshot_path", rows[0])

    def test_get_many_hydrates_only_requested_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            first = store.add_event(camera_id="gate", kind="motion", message="first")
            second = store.add_event(camera_id="gate", kind="motion", message="second")
            store.add_event(camera_id="gate", kind="motion", message="third")

            rows = store.get_many([int(second["id"]), int(first["id"])])

            self.assertEqual({row["message"] for row in rows}, {"first", "second"})
            self.assertEqual(len(rows), 2)

    def test_update_objects_persists_manual_detection_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            event = store.add_event(
                camera_id="back-middle",
                kind="motion",
                snapshot_path="/tmp/snapshot.jpg",
                objects_json=json.dumps([{"status": "no_recorded_frame"}]),
                created_at="2026-07-11T15:36:57+00:00",
            )

            updated = store.update_objects(
                int(event["id"]),
                json.dumps([{"label": "car", "confidence": 0.8, "detection_source": "manual_openvino"}]),
            )

            self.assertIsNotNone(updated)
            loaded = store.get(int(event["id"]))
            self.assertEqual(updated, loaded)
            objects = json.loads(loaded["objects_json"])
            self.assertEqual(objects[0]["label"], "car")
            self.assertEqual(objects[0]["detection_source"], "manual_openvino")


if __name__ == "__main__":
    unittest.main()

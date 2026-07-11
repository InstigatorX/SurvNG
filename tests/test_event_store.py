from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from survng.app.events import EventStore


class EventStoreTest(unittest.TestCase):
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

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from survng.app.incident_utils import event_snapshot_path, stable_incident_id, stable_incident_key


class IncidentIdentityTest(unittest.TestCase):
    def test_incident_id_is_stable_for_the_same_first_event(self) -> None:
        events = [{"id": 41}, {"id": 42}]
        before_append = stable_incident_id("front-door", events[0]["id"])
        events.append({"id": 43})
        after_append = stable_incident_id("front-door", events[0]["id"])

        self.assertEqual(before_append, after_append)
        self.assertEqual(stable_incident_id("front-door", 41), "incident-front-door-41")
        self.assertEqual(stable_incident_key("front-door", 41), "front-door-41")

    def test_incident_id_is_distinct_for_camera_and_first_event(self) -> None:
        self.assertNotEqual(stable_incident_id("front-door", 41), stable_incident_id("front-door", 42))
        self.assertNotEqual(stable_incident_id("front-door", 41), stable_incident_id("back-door", 41))


class EventSnapshotPathTest(unittest.TestCase):
    def test_allows_image_beneath_storage_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            snapshot = storage / "snapshots" / "front-door" / "event.jpg"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"jpeg")

            self.assertEqual(
                event_snapshot_path(storage, {"snapshot_path": str(snapshot)}),
                snapshot.resolve(),
            )

    def test_rejects_file_outside_storage_directory(self) -> None:
        with tempfile.TemporaryDirectory() as storage_dir, tempfile.TemporaryDirectory() as other_dir:
            outside = Path(other_dir) / "event.jpg"
            outside.write_bytes(b"jpeg")

            with self.assertRaises(PermissionError):
                event_snapshot_path(Path(storage_dir), {"snapshot_path": str(outside)})

    def test_rejects_non_image_file_inside_storage_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            database = storage / "survng.sqlite3"
            database.write_bytes(b"database")

            with self.assertRaises(PermissionError):
                event_snapshot_path(storage, {"snapshot_path": str(database)})

    def test_rejects_missing_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                event_snapshot_path(Path(tmpdir), {"snapshot_path": ""})


if __name__ == "__main__":
    unittest.main()

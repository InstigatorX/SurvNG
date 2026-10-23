from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from survng.app.events import EventStore
from survng.app.faces import FaceStore


class OrphanEventReferenceRepairTest(unittest.TestCase):
    def test_tracking_comparison_orphans_are_deleted_on_store_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = EventStore(root)
            event = store.add_event("gate", "object")
            store.save_tracking_comparison(
                event_id=int(event["id"]),
                camera_id="gate",
                event_created_at=str(event["created_at"]),
                result={"tracks": 1},
            )
            with store._connect() as connection:
                connection.execute("delete from events where id = ?", (event["id"],))

            EventStore(root)

            with store._connect() as connection:
                remaining = connection.execute(
                    "select count(*) from tracking_comparisons where event_id = ?",
                    (event["id"],),
                ).fetchone()[0]
            self.assertEqual(remaining, 0)

    def test_motion_audit_orphan_links_are_nulled_on_store_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = EventStore(root)
            linked = store.add_event("gate", "object")
            related = store.add_event("gate", "object")
            with store._connect() as connection:
                connection.execute(
                    """
                    insert into motion_audits(
                        event_id, related_event_id, camera_id, snapshot_path,
                        created_at, mode, sensitivity, score, threshold, reason
                    ) values (?, ?, 'gate', '', ?, 'audit', 'balanced', 0.1, 0.5, 'test')
                    """,
                    (
                        linked["id"],
                        related["id"],
                        "2026-01-01T00:00:00+00:00",
                    ),
                )
                connection.execute(
                    "delete from events where id in (?, ?)",
                    (linked["id"], related["id"]),
                )

            EventStore(root)

            with store._connect() as connection:
                row = connection.execute(
                    "select event_id, related_event_id from motion_audits"
                ).fetchone()
                remaining_events = connection.execute(
                    "select count(*) from events"
                ).fetchone()[0]
            self.assertEqual(row[0], None)
            self.assertEqual(row[1], None)
            self.assertEqual(remaining_events, 0)

    def test_face_observation_orphans_are_deleted_on_store_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            database = root / "database"
            storage.mkdir()
            events = EventStore(storage, database_dir=database)
            FaceStore(storage, start_recognition=False, database_dir=database)
            event = events.add_event("gate", "object")
            with sqlite3.connect(events.db_path) as connection:
                connection.execute(
                    """
                    insert into face_observations(
                        event_id, object_index, camera_id, snapshot_path,
                        box_json, observed_at, created_at
                    ) values (?, 0, 'gate', 'snapshots/gate/gone.jpg', '{}', ?, ?)
                    """,
                    (
                        event["id"],
                        "2026-01-01T00:00:00+00:00",
                        "2026-01-01T00:00:00+00:00",
                    ),
                )
                connection.execute(
                    """
                    insert into face_unknown_members(
                        observation_id, cluster_id, updated_at
                    )
                    select id, 1, created_at from face_observations
                    """
                )
                connection.execute("delete from events where id = ?", (event["id"],))

            EventStore(storage, database_dir=database)

            with sqlite3.connect(events.db_path) as connection:
                observations = connection.execute(
                    "select count(*) from face_observations"
                ).fetchone()[0]
                members = connection.execute(
                    "select count(*) from face_unknown_members"
                ).fetchone()[0]
            self.assertEqual(observations, 0)
            self.assertEqual(members, 0)

    def test_new_tracking_comparisons_declare_event_fk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            with store._connect() as connection:
                sql = connection.execute(
                    "select sql from sqlite_master "
                    "where type = 'table' and name = 'tracking_comparisons'"
                ).fetchone()[0]
            self.assertIn("references events(id) on delete cascade", sql.lower())


if __name__ == "__main__":
    unittest.main()

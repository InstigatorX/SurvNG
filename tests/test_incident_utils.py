from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from survng.app.incident_utils import (
    event_epoch,
    event_snapshot_path,
    incident_event_groups,
    portable_media_path,
    snapshot_media_type,
    stable_incident_id,
    stable_incident_key,
    stored_media_path,
)
from survng.app.config import MediaStorageConfig, MediaStorageLocationConfig
from survng.app.media_storage import MediaStorageRegistry


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

    def test_grouping_keeps_noisy_camera_separate_from_other_recent_incidents(self) -> None:
        rows = [
            {
                "id": index + 1,
                "camera_id": "noisy-camera",
                "created_at": f"2026-07-15T12:00:{index:02d}+00:00",
            }
            for index in range(30)
        ]
        rows.extend([
            {"id": 101, "camera_id": "front-door", "created_at": "2026-07-15T12:01:00+00:00"},
            {"id": 102, "camera_id": "front-door", "created_at": "2026-07-15T12:02:00+00:00"},
            {"id": 103, "camera_id": "gate", "created_at": "2026-07-15T12:03:00+00:00"},
        ])

        groups = incident_event_groups(rows, gap_seconds=45)

        self.assertEqual(len(groups), 4)
        self.assertEqual([camera_id for camera_id, _ in groups], ["gate", "front-door", "front-door", "noisy-camera"])
        noisy_group = next(events for camera_id, events in groups if camera_id == "noisy-camera")
        self.assertEqual(len(noisy_group), 30)

    def test_grouping_orders_incidents_by_start_not_latest_activity(self) -> None:
        rows = [
            {"id": 1, "camera_id": "older", "created_at": "2026-07-15T12:00:00+00:00"},
            {"id": 2, "camera_id": "older", "created_at": "2026-07-15T12:00:30+00:00"},
            {"id": 3, "camera_id": "older", "created_at": "2026-07-15T12:01:00+00:00"},
            {"id": 4, "camera_id": "newer", "created_at": "2026-07-15T12:01:15+00:00"},
            {"id": 5, "camera_id": "older", "created_at": "2026-07-15T12:01:30+00:00"},
        ]

        groups = incident_event_groups(rows, gap_seconds=45)

        self.assertEqual([camera_id for camera_id, _ in groups], ["newer", "older"])


class IncidentTimeTest(unittest.TestCase):
    def test_invalid_legacy_timestamp_sorts_as_oldest(self) -> None:
        self.assertEqual(event_epoch({"created_at": "not-a-date"}), 0.0)
        self.assertEqual(event_epoch({}), 0.0)

    def test_naive_legacy_timestamp_is_interpreted_as_utc(self) -> None:
        self.assertEqual(
            event_epoch({"created_at": "1970-01-01T00:00:01"}),
            1.0,
        )


class EventSnapshotPathTest(unittest.TestCase):
    def test_snapshot_media_types_are_deterministic_for_supported_formats(self) -> None:
        self.assertEqual(snapshot_media_type(Path("event.webp")), "image/webp")
        self.assertEqual(snapshot_media_type(Path("event.JPG")), "image/jpeg")
        self.assertEqual(snapshot_media_type(Path("event.png")), "image/png")

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

    def test_resolves_portable_snapshot_under_active_storage_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            systemd_storage = root / "systemd-media"
            docker_storage = root / "docker-media"
            relative = Path("motion_samples/gate/audit.jpg")
            for storage in (systemd_storage, docker_storage):
                snapshot = storage / relative
                snapshot.parent.mkdir(parents=True)
                snapshot.write_bytes(b"jpeg")

            self.assertEqual(
                stored_media_path(systemd_storage, relative.as_posix()),
                (systemd_storage / relative).resolve(),
            )
            self.assertEqual(
                stored_media_path(docker_storage, relative.as_posix()),
                (docker_storage / relative).resolve(),
            )

    def test_converts_verified_legacy_mount_to_portable_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "media"
            snapshot = storage / "motion_samples" / "gate" / "audit.jpg"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"jpeg")

            self.assertEqual(
                portable_media_path(
                    storage,
                    "/mnt/frigate/SurvNG/motion_samples/gate/audit.jpg",
                ),
                "motion_samples/gate/audit.jpg",
            )

    def test_does_not_convert_unverified_external_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            external = "/untrusted/snapshots/gate/missing.jpg"
            self.assertEqual(portable_media_path(Path(tmpdir), external), external)

    def test_rejects_relative_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as storage_dir, tempfile.TemporaryDirectory() as other_dir:
            outside = Path(other_dir) / "event.jpg"
            outside.write_bytes(b"jpeg")
            traversal = Path("..") / Path(other_dir).name / outside.name

            with self.assertRaises(PermissionError):
                stored_media_path(Path(storage_dir), traversal)

    def test_rejects_file_outside_storage_directory(self) -> None:
        with tempfile.TemporaryDirectory() as storage_dir, tempfile.TemporaryDirectory() as other_dir:
            outside = Path(other_dir) / "event.jpg"
            outside.write_bytes(b"jpeg")

            with self.assertRaises(PermissionError):
                event_snapshot_path(Path(storage_dir), {"snapshot_path": str(outside)})

    def test_allows_snapshot_inside_configured_external_media_location(self) -> None:
        with tempfile.TemporaryDirectory() as storage_dir, tempfile.TemporaryDirectory() as media_dir:
            snapshot = Path(media_dir) / "snapshots" / "gate" / "event.webp"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"webp")
            registry = MediaStorageRegistry(Path(storage_dir), MediaStorageConfig(locations=[
                MediaStorageLocationConfig(id="media", path=media_dir, roles=["snapshots"]),
            ]))

            self.assertEqual(
                event_snapshot_path(
                    Path(storage_dir),
                    {"snapshot_path": str(snapshot)},
                    registry,
                ),
                snapshot.resolve(),
            )

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

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from survng.app.events import EventStore
from survng.app.faces import FaceStore
from survng.app.recorder import Recorder
from survng.app.storage_maintenance import (
    DEFAULT_ORPHAN_MIN_AGE_SECONDS,
    StorageMaintenanceRunner,
    StorageReconciler,
)
from survng.app.config import MediaStorageConfig, MediaStorageLocationConfig
from survng.app.media_storage import MediaStorageRegistry


class StorageReconcilerTest(unittest.TestCase):
    def test_full_scan_includes_external_snapshot_and_motion_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            external = root / "external"
            storage.mkdir()
            external.mkdir()
            registry = MediaStorageRegistry(
                storage,
                MediaStorageConfig(locations=[MediaStorageLocationConfig(
                    id="archive",
                    path=str(external),
                    roles=["snapshots", "motion_audits"],
                )]),
            )
            events = EventStore(storage, database_dir=root / "database", media_storage=registry)
            recorder = Recorder(
                "ffmpeg",
                storage,
                segment_seconds=10,
                index_dir=root / "recording-index",
            )
            referenced = external / "snapshots" / "gate" / "known.webp"
            referenced.parent.mkdir(parents=True)
            referenced.write_bytes(b"image")
            events.add_event("gate", "object", snapshot_path=str(referenced))
            orphan = external / "motion_samples" / "gate" / "orphan.webp"
            orphan.parent.mkdir(parents=True)
            orphan.write_bytes(b"orphan")
            old = time.time() - 120
            os.utime(orphan, (old, old))

            summary = StorageReconciler(
                storage,
                events.db_path,
                recorder,
                media_storage=registry,
            ).run(full=True)["summary"]

            self.assertEqual(summary["missing_event_snapshots"], 0)
            self.assertEqual(summary["orphan_media_files"], 1)
            self.assertEqual(summary["orphan_media_samples"], [
                "archive:motion_samples/gate/orphan.webp"
            ])
            self.assertEqual(summary["media_locations"][0]["id"], "archive")

    def test_scan_and_repair_preserve_history_and_reconcile_recording_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            database = root / "database"
            index = root / "recording-index"
            storage.mkdir()
            events = EventStore(storage, database_dir=database)
            FaceStore(storage, start_recognition=False, database_dir=database)
            recorder = Recorder("ffmpeg", storage, segment_seconds=10, index_dir=index)

            known_snapshot = storage / "snapshots" / "gate" / "known.jpg"
            known_snapshot.parent.mkdir(parents=True)
            known_snapshot.write_bytes(b"image")
            orphan_snapshot = storage / "snapshots" / "gate" / "orphan.jpg"
            orphan_snapshot.write_bytes(b"orphan")
            old = time.time() - 120
            orphan_snapshot.touch()
            orphan_snapshot.chmod(0o600)
            os.utime(orphan_snapshot, (old, old))

            missing_snapshot = storage / "snapshots" / "gate" / "deleted.jpg"
            missing_recording = storage / "recordings" / "gate" / "main" / "deleted.mp4"
            event_id = events.add_event(
                "gate", "object", snapshot_path=str(missing_snapshot),
                recording_path=str(missing_recording),
            )["id"]
            events.add_event("gate", "object", snapshot_path=str(known_snapshot))
            with sqlite3.connect(events.db_path) as connection:
                connection.execute(
                    "INSERT INTO motion_audits(camera_id, snapshot_path, created_at, mode, sensitivity, score, threshold, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("gate", str(storage / "motion_samples" / "missing.jpg"), "2026-01-01T00:00:00+00:00", "audit", "balanced", 0.1, 0.5, "test"),
                )
                connection.execute(
                    "INSERT INTO face_observations(event_id, object_index, camera_id, snapshot_path, box_json, observed_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (event_id, 0, "gate", str(missing_snapshot), "{}", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
                )

            hour = storage / "recordings" / "gate" / "main" / "2026-01-01" / "00"
            hour.mkdir(parents=True)
            unindexed = hour / "20260101-000000.mp4"
            unindexed.write_bytes(b"segment")
            stale = hour / "20260101-000010.mp4"
            stale.write_bytes(b"gone")
            recorder._store_recording_rows("gate", "main", recorder._recording_rows_for_files("gate", "main", [stale]))
            stale.unlink()

            reconciler = StorageReconciler(storage, events.db_path, recorder)
            scan = reconciler.run(full=True)
            summary = scan["summary"]
            self.assertEqual(summary["missing_event_snapshots"], 1)
            self.assertEqual(summary["missing_event_recordings"], 1)
            self.assertEqual(summary["missing_motion_snapshots"], 1)
            self.assertEqual(summary["missing_face_snapshots"], 1)
            self.assertEqual(summary["orphan_media_files"], 1)
            self.assertEqual(summary["missing_index_rows"], 1)
            self.assertEqual(summary["unindexed_recording_files"], 1)
            self.assertEqual(summary["missing_index_samples"], ["recordings/gate/main/2026-01-01/00/20260101-000010.mp4"])
            self.assertEqual(summary["unindexed_samples"], ["recordings/gate/main/2026-01-01/00/20260101-000000.mp4"])

            with patch.object(recorder, "storage_index_health", wraps=recorder.storage_index_health) as health:
                repaired = reconciler.run(apply=True, full=True)
            self.assertEqual(health.call_count, 1)
            self.assertEqual(repaired["repairs"]["stale_index_rows_removed"], 1)
            self.assertEqual(repaired["repairs"]["recordings_reindexed"], 1)
            self.assertEqual(repaired["repairs"]["event_media_references_cleared"], 2)
            self.assertEqual(repaired["repairs"]["motion_sample_references_cleared"], 1)
            self.assertEqual(repaired["repairs"]["face_media_references_cleared"], 1)
            self.assertEqual(repaired["summary"]["missing_index_rows"], 0)
            self.assertEqual(repaired["summary"]["unindexed_recording_files"], 0)
            self.assertTrue(repaired["summary"]["recording_snapshot_reused"])
            with sqlite3.connect(events.db_path) as connection:
                row = connection.execute(
                    "SELECT snapshot_path, recording_path FROM events WHERE id = ?", (event_id,)
                ).fetchone()
                self.assertEqual(row, ("", ""))
                self.assertEqual(connection.execute("SELECT count(*) FROM events").fetchone()[0], 2)
            with recorder._index_connection() as connection:
                paths = {str(row[0]) for row in connection.execute("SELECT path FROM recordings")}
            self.assertEqual(paths, {str(unindexed)})
            self.assertTrue(orphan_snapshot.is_file())

    def test_runner_rejects_overlapping_jobs(self) -> None:
        runner = StorageMaintenanceRunner()
        blocker = __import__("threading").Event()

        class BlockingReconciler:
            def run(self, *, apply: bool = False, full: bool = False, **_kwargs) -> dict[str, object]:
                blocker.wait(1)
                return {"mode": "repair" if apply else "scan"}

        factory = lambda _cancel, _progress: BlockingReconciler()
        runner.start(factory, apply=False)
        with self.assertRaisesRegex(RuntimeError, "already running"):
            runner.start(factory, apply=True)
        blocker.set()
        deadline = time.time() + 1
        while runner.status()["status"] == "running" and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(runner.status()["status"], "complete")

    def test_runner_cancels_active_job(self) -> None:
        runner = StorageMaintenanceRunner()

        class CancellableReconciler:
            def __init__(self, cancelled) -> None:
                self.cancelled = cancelled

            def run(self, *, apply: bool = False, full: bool = False, **_kwargs) -> dict[str, object]:
                while not self.cancelled.wait(0.01):
                    pass
                raise InterruptedError("cancelled")

        runner.start(lambda cancelled, _progress: CancellableReconciler(cancelled), apply=False, full=True)
        state = runner.cancel()
        self.assertEqual(state["status"], "cancelling")
        deadline = time.time() + 1
        while runner.status()["status"] == "cancelling" and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(runner.status()["status"], "cancelled")

    def test_runner_stop_cancels_and_joins_active_job(self) -> None:
        runner = StorageMaintenanceRunner()

        class CancellableReconciler:
            def __init__(self, cancelled) -> None:
                self.cancelled = cancelled

            def run(self, *, apply: bool = False, full: bool = False, **_kwargs) -> dict[str, object]:
                self.cancelled.wait(1)
                raise InterruptedError("cancelled")

        runner.start(
            lambda cancelled, _progress: CancellableReconciler(cancelled),
            apply=False,
        )

        self.assertTrue(runner.stop(timeout=1.0))
        self.assertEqual(runner.status()["status"], "cancelled")

    def test_runner_status_does_not_expose_mutable_internal_progress(self) -> None:
        runner = StorageMaintenanceRunner()
        blocker = threading.Event()

        class BlockingReconciler:
            def run(self, *, apply: bool = False, full: bool = False, **_kwargs) -> dict[str, object]:
                blocker.wait(1)
                return {}

        runner.start(lambda _cancel, _progress: BlockingReconciler(), apply=False)
        state = runner.status()
        state["progress"]["phase"] = "tampered"

        self.assertEqual(runner.status()["progress"]["phase"], "Starting")
        blocker.set()
        self.assertTrue(runner.stop(timeout=1.0))

    def test_runner_start_failure_restores_idle_state(self) -> None:
        runner = StorageMaintenanceRunner()
        with patch.object(threading.Thread, "start", side_effect=RuntimeError("no thread")):
            with self.assertRaisesRegex(RuntimeError, "no thread"):
                runner.start(lambda _cancel, _progress: object(), apply=False)

        self.assertEqual(runner.status(), {"status": "idle"})
        self.assertTrue(runner.stop())

    def test_repair_reports_unindexable_recording_as_still_unindexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            database = root / "database"
            index = root / "recording-index"
            storage.mkdir()
            events = EventStore(storage, database_dir=database)
            recorder = Recorder("ffmpeg", storage, segment_seconds=10, index_dir=index)
            hour = storage / "recordings" / "gate" / "main" / "2026-01-01" / "00"
            hour.mkdir(parents=True)
            malformed = hour / "not-a-segment.mp4"
            malformed.write_bytes(b"segment")
            old = time.time() - 120
            os.utime(malformed, (old, old))

            repaired = StorageReconciler(storage, events.db_path, recorder).run(
                apply=True,
                full=True,
            )

        self.assertEqual(repaired["repairs"]["recordings_reindexed"], 0)
        self.assertEqual(repaired["summary"]["unindexed_recording_files"], 1)

    def test_repair_then_fresh_scan_stays_clean_after_confirmed_fixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            database = root / "database"
            index = root / "recording-index"
            storage.mkdir()
            events = EventStore(storage, database_dir=database)
            recorder = Recorder("ffmpeg", storage, segment_seconds=10, index_dir=index)

            missing_snapshot = storage / "snapshots" / "gate" / "deleted.jpg"
            event_id = events.add_event(
                "gate", "object", snapshot_path=str(missing_snapshot),
            )["id"]
            hour = storage / "recordings" / "gate" / "main" / "2026-01-01" / "00"
            hour.mkdir(parents=True)
            unindexed = hour / "20260101-000000.mp4"
            unindexed.write_bytes(b"segment")
            stale = hour / "20260101-000010.mp4"
            stale.write_bytes(b"gone")
            recorder._store_recording_rows(
                "gate", "main", recorder._recording_rows_for_files("gate", "main", [stale])
            )
            stale.unlink()

            reconciler = StorageReconciler(storage, events.db_path, recorder)
            repaired = reconciler.run(apply=True, full=True)
            fresh = reconciler.run(full=True)

            self.assertEqual(repaired["summary"]["missing_index_rows"], 0)
            self.assertEqual(repaired["summary"]["unindexed_recording_files"], 0)
            self.assertEqual(fresh["summary"]["missing_index_rows"], 0)
            self.assertEqual(fresh["summary"]["unindexed_recording_files"], 0)
            self.assertEqual(fresh["summary"]["missing_event_snapshots"], 0)
            with sqlite3.connect(events.db_path) as connection:
                row = connection.execute(
                    "SELECT snapshot_path FROM events WHERE id = ?", (event_id,)
                ).fetchone()
                self.assertEqual(row[0], "")

    def test_full_scan_does_not_report_present_media_omitted_from_walk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            database = root / "database"
            storage.mkdir()
            events = EventStore(storage, database_dir=database)
            recorder = Recorder("ffmpeg", storage, segment_seconds=10, index_dir=root / "index")
            present = storage / "snapshots" / "gate" / "present.jpg"
            present.parent.mkdir(parents=True)
            present.write_bytes(b"image")
            events.add_event("gate", "object", snapshot_path=str(present))

            reconciler = StorageReconciler(storage, events.db_path, recorder)
            real_rglob = Path.rglob

            def hide_present(self, pattern):
                for path in real_rglob(self, pattern):
                    if path == present:
                        continue
                    yield path

            with patch.object(Path, "rglob", hide_present):
                summary = reconciler.run(full=True)["summary"]
                repaired = reconciler.run(apply=True, full=True)

            self.assertEqual(summary["missing_event_snapshots"], 0)
            self.assertEqual(repaired["repairs"]["event_media_references_cleared"], 0)
            with sqlite3.connect(events.db_path) as connection:
                row = connection.execute("SELECT snapshot_path FROM events").fetchone()
                self.assertTrue(row[0])
                self.assertTrue(str(row[0]).endswith("snapshots/gate/present.jpg"))

    def test_repair_skips_clearing_when_media_location_is_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            external = root / "external"
            storage.mkdir()
            external.mkdir()
            registry = MediaStorageRegistry(
                storage,
                MediaStorageConfig(locations=[MediaStorageLocationConfig(
                    id="archive",
                    path=str(external),
                    roles=["snapshots"],
                )]),
            )
            events = EventStore(storage, database_dir=root / "database", media_storage=registry)
            recorder = Recorder(
                "ffmpeg",
                storage,
                segment_seconds=10,
                index_dir=root / "index",
            )
            missing = external / "snapshots" / "gate" / "gone.jpg"
            events.add_event("gate", "object", snapshot_path=str(missing))
            offline = registry.status("archive")
            offline_status = type(offline)(
                **{
                    **{field: getattr(offline, field) for field in offline.__dataclass_fields__},
                    "state": "not_mounted",
                    "error": "required mount is absent",
                }
            )

            with patch.object(registry, "status", return_value=offline_status):
                repaired = StorageReconciler(
                    storage,
                    events.db_path,
                    recorder,
                    media_storage=registry,
                ).run(apply=True, full=True)

            self.assertEqual(repaired["repairs"]["event_media_references_cleared"], 0)
            self.assertGreaterEqual(repaired["repairs"]["media_references_skipped"], 1)
            with sqlite3.connect(events.db_path) as connection:
                row = connection.execute("SELECT snapshot_path FROM events").fetchone()
                self.assertEqual(row[0], str(missing))

    def test_orphan_reclaim_dry_run_counts_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            storage.mkdir()
            events = EventStore(storage, database_dir=root / "database")
            recorder = Recorder(
                "ffmpeg", storage, segment_seconds=10, index_dir=root / "index"
            )
            known = storage / "snapshots" / "gate" / "known.jpg"
            known.parent.mkdir(parents=True)
            known.write_bytes(b"image")
            events.add_event("gate", "object", snapshot_path=str(known))
            orphan = storage / "snapshots" / "gate" / "orphan.jpg"
            orphan.write_bytes(b"orphan")
            old = time.time() - DEFAULT_ORPHAN_MIN_AGE_SECONDS - 60
            os.utime(orphan, (old, old))

            result = StorageReconciler(storage, events.db_path, recorder).run(
                apply=True,
                full=True,
                reclaim_orphans=True,
                reclaim_dry_run=True,
            )

            self.assertEqual(result["repairs"]["orphan_media_reclaim_candidates"], 1)
            self.assertEqual(result["repairs"]["orphan_media_deleted"], 0)
            self.assertTrue(orphan.is_file())
            self.assertIn("dry-run", result["note"])

    def test_orphan_reclaim_deletes_only_old_unreferenced_incident_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            storage.mkdir()
            events = EventStore(storage, database_dir=root / "database")
            recorder = Recorder(
                "ffmpeg", storage, segment_seconds=10, index_dir=root / "index"
            )
            known = storage / "snapshots" / "gate" / "known.jpg"
            known.parent.mkdir(parents=True)
            known.write_bytes(b"image")
            events.add_event("gate", "object", snapshot_path=str(known))
            old_orphan = storage / "snapshots" / "gate" / "old-orphan.jpg"
            old_orphan.write_bytes(b"old")
            young_orphan = storage / "motion_samples" / "gate" / "young.webp"
            young_orphan.parent.mkdir(parents=True)
            young_orphan.write_bytes(b"young")
            recording_orphan = (
                storage / "recordings" / "gate" / "main" / "2026-01-01" / "00" / "loose.mp4"
            )
            recording_orphan.parent.mkdir(parents=True)
            recording_orphan.write_bytes(b"segment")
            old = time.time() - 3600
            young = time.time() - 120
            os.utime(old_orphan, (old, old))
            os.utime(young_orphan, (young, young))
            os.utime(recording_orphan, (old, old))

            result = StorageReconciler(storage, events.db_path, recorder).run(
                apply=True,
                full=True,
                reclaim_orphans=True,
                reclaim_dry_run=False,
                orphan_min_age_seconds=600,
            )

            self.assertEqual(result["repairs"]["orphan_media_deleted"], 1)
            self.assertFalse(old_orphan.exists())
            self.assertTrue(known.is_file())
            self.assertTrue(young_orphan.is_file())
            self.assertTrue(recording_orphan.is_file())
            self.assertEqual(result["summary"]["orphan_media_files"], 1)

    def test_orphan_reclaim_keeps_scene_gallery_and_acquisition_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            storage.mkdir()
            events = EventStore(storage, database_dir=root / "database")
            recorder = Recorder(
                "ffmpeg", storage, segment_seconds=10, index_dir=root / "index"
            )
            cover = storage / "snapshots" / "gate" / "cover.jpg"
            cover.parent.mkdir(parents=True)
            cover.write_bytes(b"cover")
            event_id = events.add_event("gate", "object", snapshot_path=str(cover))["id"]
            gallery = storage / "snapshots" / "gate" / "gallery.jpg"
            asset = storage / "snapshots" / "gate" / "supporting.jpg"
            acquired = storage / "snapshots" / "gate" / "acquired.jpg"
            pending_cover = storage / "snapshots" / "gate" / "pending-cover.jpg"
            orphan = storage / "snapshots" / "gate" / "orphan.jpg"
            for path, payload in (
                (gallery, b"gallery"),
                (asset, b"asset"),
                (acquired, b"acquired"),
                (pending_cover, b"pending"),
                (orphan, b"orphan"),
            ):
                path.write_bytes(payload)
            old = time.time() - 3600
            for path in (gallery, asset, acquired, pending_cover, orphan):
                os.utime(path, (old, old))
            with sqlite3.connect(events.db_path) as connection:
                connection.execute(
                    "INSERT INTO scene_evidence_images("
                    "id, event_id, frame_key, captured_epoch, snapshot_path, payload_json, state"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("gallery-1", event_id, "frame-1", old, str(gallery), "{}", "ready"),
                )
                connection.execute(
                    "INSERT INTO scene_snapshot_assets("
                    "snapshot_path, camera_id, created_at, snapshot_size_bytes"
                    ") VALUES (?, ?, ?, ?)",
                    (str(asset), "gate", "2026-01-01T00:00:00+00:00", asset.stat().st_size),
                )
                connection.execute(
                    "INSERT INTO acquired_samples("
                    "id, camera_id, captured_epoch, source, status, metadata_json, "
                    "snapshot_path, recording_path, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "sample-1", "gate", old, "test", "complete", "{}",
                        str(acquired), "", old,
                    ),
                )
                connection.execute(
                    "INSERT INTO event_cover_requirements("
                    "event_id, state, deadline_epoch, available_at_epoch, payload_json, "
                    "created_at, updated_at"
                    ") VALUES (?, 'pending', ?, ?, json_object('snapshot_path', ?), ?, ?)",
                    (event_id, old + 86400, old, str(pending_cover), old, old),
                )

            result = StorageReconciler(storage, events.db_path, recorder).run(
                apply=True,
                full=True,
                reclaim_orphans=True,
                reclaim_dry_run=False,
                orphan_min_age_seconds=600,
            )

            self.assertEqual(result["repairs"]["orphan_media_deleted"], 1)
            self.assertFalse(orphan.exists())
            self.assertTrue(cover.is_file())
            self.assertTrue(gallery.is_file())
            self.assertTrue(asset.is_file())
            self.assertTrue(acquired.is_file())
            self.assertTrue(pending_cover.is_file())

    def test_orphan_reclaim_skips_when_media_location_is_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            external = root / "external"
            storage.mkdir()
            external.mkdir()
            registry = MediaStorageRegistry(
                storage,
                MediaStorageConfig(locations=[MediaStorageLocationConfig(
                    id="archive",
                    path=str(external),
                    roles=["snapshots", "motion_audits"],
                )]),
            )
            events = EventStore(
                storage, database_dir=root / "database", media_storage=registry
            )
            recorder = Recorder(
                "ffmpeg", storage, segment_seconds=10, index_dir=root / "index"
            )
            orphan = external / "snapshots" / "gate" / "orphan.jpg"
            orphan.parent.mkdir(parents=True)
            orphan.write_bytes(b"orphan")
            old = time.time() - 3600
            os.utime(orphan, (old, old))
            offline = registry.status("archive")
            offline_status = type(offline)(
                **{
                    **{field: getattr(offline, field) for field in offline.__dataclass_fields__},
                    "state": "not_mounted",
                    "error": "required mount is absent",
                }
            )

            with patch.object(registry, "status", return_value=offline_status):
                result = StorageReconciler(
                    storage,
                    events.db_path,
                    recorder,
                    media_storage=registry,
                ).run(
                    apply=True,
                    full=True,
                    reclaim_orphans=True,
                    reclaim_dry_run=False,
                    orphan_min_age_seconds=600,
                )

            self.assertEqual(result["repairs"]["orphan_media_deleted"], 0)
            self.assertGreaterEqual(result["repairs"]["orphan_media_reclaim_skipped"], 1)
            self.assertTrue(orphan.is_file())

    def test_orphan_reclaim_requires_apply_and_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            storage.mkdir()
            events = EventStore(storage, database_dir=root / "database")
            recorder = Recorder(
                "ffmpeg", storage, segment_seconds=10, index_dir=root / "index"
            )
            orphan = storage / "snapshots" / "gate" / "orphan.jpg"
            orphan.parent.mkdir(parents=True)
            orphan.write_bytes(b"orphan")
            old = time.time() - DEFAULT_ORPHAN_MIN_AGE_SECONDS - 60
            os.utime(orphan, (old, old))

            scan = StorageReconciler(storage, events.db_path, recorder).run(
                reclaim_orphans=True,
                reclaim_dry_run=False,
                full=True,
            )
            self.assertEqual(scan["repairs"]["orphan_media_deleted"], 0)
            self.assertTrue(orphan.is_file())
            self.assertIn("requires apply=true and full=true", scan["note"])


if __name__ == "__main__":
    unittest.main()

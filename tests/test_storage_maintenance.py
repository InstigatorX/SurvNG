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
from survng.app.storage_maintenance import StorageMaintenanceRunner, StorageReconciler


class StorageReconcilerTest(unittest.TestCase):
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
            def run(self, *, apply: bool = False, full: bool = False) -> dict[str, object]:
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

            def run(self, *, apply: bool = False, full: bool = False) -> dict[str, object]:
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

            def run(self, *, apply: bool = False, full: bool = False) -> dict[str, object]:
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
            def run(self, *, apply: bool = False, full: bool = False) -> dict[str, object]:
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


if __name__ == "__main__":
    unittest.main()

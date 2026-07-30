from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

from survng.app.config import CameraConfig, RecordingRetentionConfig
from survng.app.recording_retention import GIB, RecordingRetentionService


DiskUsage = namedtuple("DiskUsage", "total used free")


class RecordingRetentionServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.storage = Path(self.temporary.name)
        self.recordings = self.storage / "recordings"
        self.recordings.mkdir()
        self.database = self.storage / "recordings.sqlite3"
        with self.connection() as connection:
            connection.execute(
                """
                CREATE TABLE recordings (
                    path TEXT PRIMARY KEY, camera_id TEXT NOT NULL, source TEXT NOT NULL,
                    name TEXT NOT NULL, size_bytes INTEGER NOT NULL, modified_at REAL NOT NULL,
                    start_epoch REAL NOT NULL, duration_seconds REAL NOT NULL,
                    end_epoch REAL NOT NULL
                )
                """
            )

    def connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection

    def service(self, **overrides: object) -> RecordingRetentionService:
        config = RecordingRetentionConfig.model_validate({
            "storage_limit_tb": 1,
            "main_days": 7,
            "live_days": 21,
            "cleanup_batch_files": 100,
            **overrides,
        })
        service = RecordingRetentionService(
            self.storage,
            self.recordings,
            self.connection,
            config,
        )
        service._cameras = {
            "gate": CameraConfig(
                id="gate",
                name="Gate",
                stream_url="rtsp://camera/main",
            )
        }
        return service

    def insert_recording(
        self,
        *,
        age_days: int,
        source: str = "main",
        size: int = 10 * GIB,
        path: Path | None = None,
    ) -> Path:
        now = time.time()
        start = now - age_days * 86400
        target = path or (
            self.recordings / "gate" / source / "2026-01-01" / "00" / f"{source}-{age_days}.mp4"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"recording")
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO recordings
                    (path, camera_id, source, name, size_bytes, modified_at,
                     start_epoch, duration_seconds, end_epoch)
                VALUES (?, 'gate', ?, ?, ?, ?, ?, 10, ?)
                """,
                (str(target), source, target.name, size, start, start, start + 10),
            )
        return target

    @patch(
        "survng.app.recording_retention.shutil.disk_usage",
        return_value=DiskUsage(10 * 1024**4, 5 * 1024**4, 5 * 1024**4),
    )
    def test_plan_reports_expired_recordings_without_deleting_them(self, _usage) -> None:
        old = self.insert_recording(age_days=8)
        recent = self.insert_recording(age_days=2)

        plan = self.service().plan()

        self.assertEqual(plan["reclaim"]["expired_files"], 1)
        self.assertEqual(plan["reclaim"]["expired_bytes"], 10 * GIB)
        self.assertIn("age", plan["reclaim"]["reasons"])
        self.assertTrue(old.exists())
        self.assertTrue(recent.exists())

    @patch(
        "survng.app.recording_retention.shutil.disk_usage",
        return_value=DiskUsage(10 * 1024**4, 5 * 1024**4, 5 * 1024**4),
    )
    def test_apply_removes_only_expired_recording_and_its_index_row(self, _usage) -> None:
        old = self.insert_recording(age_days=8)
        recent = self.insert_recording(age_days=2)

        outcome = self.service().run_once(apply=True)

        self.assertEqual(outcome["result"]["deleted_files"], 1)
        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())
        with self.connection() as connection:
            paths = {str(row[0]) for row in connection.execute("SELECT path FROM recordings")}
        self.assertEqual(paths, {str(recent)})

    @patch(
        "survng.app.recording_retention.shutil.disk_usage",
        return_value=DiskUsage(10 * 1024**4, 9 * 1024**4, 1 * 1024**4),
    )
    def test_low_space_reclaims_oldest_even_inside_age_window(self, _usage) -> None:
        oldest = self.insert_recording(age_days=3, size=2 * 1024**4)

        outcome = self.service(storage_limit_tb=9).run_once(apply=True)

        self.assertIn("free_space", outcome["plan"]["reclaim"]["reasons"])
        self.assertEqual(outcome["result"]["deleted_files"], 1)
        self.assertFalse(oldest.exists())

    @patch(
        "survng.app.recording_retention.shutil.disk_usage",
        return_value=DiskUsage(10 * 1024**4, 5 * 1024**4, 5 * 1024**4),
    )
    def test_camera_override_changes_expiration(self, _usage) -> None:
        old = self.insert_recording(age_days=10)
        service = self.service()
        service._cameras["gate"] = CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://camera/main",
            retention={"main_days": 14},
        )

        plan = service.plan()

        self.assertEqual(plan["per_camera"][0]["retention_days"], 14)
        self.assertEqual(plan["reclaim"]["expired_files"], 0)
        self.assertTrue(old.exists())

    @patch(
        "survng.app.recording_retention.shutil.disk_usage",
        return_value=DiskUsage(10 * 1024**4, 5 * 1024**4, 5 * 1024**4),
    )
    def test_path_outside_recordings_is_never_deleted(self, _usage) -> None:
        outside = self.storage / "important.mp4"
        self.insert_recording(age_days=30, path=outside)

        outcome = self.service().run_once(apply=True)

        self.assertEqual(outcome["result"]["failed_files"], 1)
        self.assertTrue(outside.exists())

    @patch(
        "survng.app.recording_retention.shutil.disk_usage",
        return_value=DiskUsage(10 * 1024**4, 5 * 1024**4, 5 * 1024**4),
    )
    def test_incident_referenced_recording_is_protected(self, _usage) -> None:
        old = self.insert_recording(age_days=30)
        service = self.service()
        service.protected_paths_provider = lambda: {str(old.resolve())}

        outcome = service.run_once(apply=True)

        self.assertEqual(outcome["result"]["deleted_files"], 0)
        self.assertTrue(old.exists())


class RecordingRetentionConfigTest(unittest.TestCase):
    def test_defaults_are_dry_run_and_watermarks_are_ordered(self) -> None:
        config = RecordingRetentionConfig()
        self.assertTrue(config.enabled)
        self.assertFalse(config.automatic_cleanup)
        self.assertLess(config.emergency_free_percent, config.minimum_free_percent)
        self.assertLess(config.minimum_free_percent, config.target_free_percent)

    def test_invalid_watermarks_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target free"):
            RecordingRetentionConfig(minimum_free_percent=20, target_free_percent=20)
        with self.assertRaisesRegex(ValueError, "emergency free"):
            RecordingRetentionConfig(emergency_free_percent=15)

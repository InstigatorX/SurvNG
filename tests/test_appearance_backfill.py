from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from survng.app.appearance_backfill import DeferredAppearanceBackfill
from survng.app.appearance_index import AppearanceIndex
from survng.app.config import ObjectTrackingConfig


class _Events:
    def __init__(self, event: dict) -> None:
        self.event = event

    def get(self, event_id: int):
        return self.event if event_id == self.event["id"] else None


class _Encoder:
    def supports_label(self, label: str) -> bool:
        return label == "car"

    def embed_for_label(self, label: str, crop: np.ndarray) -> np.ndarray:
        assert label == "car"
        assert crop.size > 0
        return np.asarray([0.6, 0.8], dtype=np.float32)

    def model_identity_for_label(self, label: str):
        return {
            "model_kind": "vehicle",
            "model_fingerprint": "vehicle-test",
            "embedding_size": 2,
            "match_threshold": 0.8,
        }


class DeferredAppearanceBackfillTest(unittest.TestCase):
    def test_empty_queue_does_not_reserve_or_wait_for_database_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "events.db"
            with sqlite3.connect(database) as connection:
                connection.execute("pragma journal_mode = wal")
                connection.execute(
                    "create table events (id integer primary key, created_at text not null)"
                )
            service = DeferredAppearanceBackfill(
                database,
                Path(tmp),
                ObjectTrackingConfig(
                    vehicle_reid_enabled=True,
                    vehicle_reid_model_path="vehicle.xml",
                ),
                _Events({"id": 7}),
                AppearanceIndex(database),
                _Encoder(),
            )
            blocker = sqlite3.connect(database, timeout=0.1)
            blocker.execute("begin immediate")
            try:
                started = time.monotonic()
                self.assertIsNone(service._claim())
                elapsed = time.monotonic() - started
            finally:
                blocker.rollback()
                blocker.close()

            self.assertLess(elapsed, 1.0)

    def test_future_job_does_not_reserve_or_wait_for_database_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "events.db"
            with sqlite3.connect(database) as connection:
                connection.execute("pragma journal_mode = wal")
                connection.execute(
                    "create table events (id integer primary key, created_at text not null)"
                )
                connection.execute(
                    "insert into events values (7, '2026-08-05T21:52:05+00:00')"
                )
            service = DeferredAppearanceBackfill(
                database,
                Path(tmp),
                ObjectTrackingConfig(
                    vehicle_reid_enabled=True,
                    vehicle_reid_model_path="vehicle.xml",
                ),
                _Events({"id": 7}),
                AppearanceIndex(database),
                _Encoder(),
            )
            self.assertTrue(service.enqueue(7, "gate", delay_seconds=60))
            blocker = sqlite3.connect(database, timeout=0.1)
            blocker.execute("begin immediate")
            try:
                started = time.monotonic()
                self.assertIsNone(service._claim())
                elapsed = time.monotonic() - started
            finally:
                blocker.rollback()
                blocker.close()

            self.assertLess(elapsed, 1.0)

    def test_worker_retries_transient_database_lock_when_claiming(self) -> None:
        service = DeferredAppearanceBackfill.__new__(DeferredAppearanceBackfill)
        service.config = ObjectTrackingConfig(
            reid_enabled=True,
            reid_model_path="person.xml",
            deferred_reid_enabled=True,
        )
        service._stop = threading.Event()
        service._wake = threading.Event()

        def claim():
            if service._claim.call_count == 1:
                raise sqlite3.OperationalError("database is locked")
            service._stop.set()
            return None

        service._claim = Mock(side_effect=claim)

        service._run()

        self.assertEqual(service._claim.call_count, 2)

    def test_jobs_are_removed_when_their_event_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "events.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "create table events (id integer primary key, created_at text not null)"
                )
                connection.execute(
                    "insert into events values (7, '2026-08-05T21:52:05+00:00')"
                )
            service = DeferredAppearanceBackfill(
                database,
                Path(tmp),
                ObjectTrackingConfig(
                    vehicle_reid_enabled=True,
                    vehicle_reid_model_path="vehicle.xml",
                ),
                _Events({"id": 7}),
                AppearanceIndex(database),
                _Encoder(),
            )
            self.assertTrue(service.enqueue(7, "gate", delay_seconds=0))

            with service._connect() as connection:
                connection.execute("delete from events where id = 7")
                remaining = connection.execute(
                    "select count(*) from appearance_backfill_jobs where event_id = 7"
                ).fetchone()[0]

            self.assertEqual(remaining, 0)

            # Upgrade cleanup also removes jobs orphaned by older builds that
            # declared the foreign key without enabling SQLite enforcement.
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """
                    insert into appearance_backfill_jobs (
                        event_id, camera_id, state, available_at, created_at, updated_at
                    ) values (99, 'gate', 'queued', 0, 'now', 'now')
                    """
                )
            service._init_db()
            with sqlite3.connect(database) as connection:
                remaining = connection.execute(
                    "select count(*) from appearance_backfill_jobs where event_id = 99"
                ).fetchone()[0]
            self.assertEqual(remaining, 0)

    def test_snapshot_fallback_indexes_missing_event_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "events.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "create table events (id integer primary key, created_at text not null)"
                )
                connection.execute(
                    "insert into events values (7, '2026-08-05T21:52:05+00:00')"
                )
            snapshot = root / "snapshots" / "gate" / "event.webp"
            snapshot.parent.mkdir(parents=True)
            cv2.imwrite(str(snapshot), np.full((100, 200, 3), 127, dtype=np.uint8))
            event = {
                "id": 7,
                "camera_id": "gate",
                "created_at": "2026-08-05T21:52:05+00:00",
                "snapshot_path": "snapshots/gate/event.webp",
                "objects_json": json.dumps([{
                    "label": "car",
                    "incident_eligible": True,
                    "box": {"x1": 20, "y1": 10, "x2": 180, "y2": 90},
                    "detection_frame_width": 200,
                    "detection_frame_height": 100,
                    "snapshot_quality_score": 0.9,
                }]),
            }
            index = AppearanceIndex(database)
            service = DeferredAppearanceBackfill(
                database,
                root,
                ObjectTrackingConfig(
                    vehicle_reid_enabled=True,
                    vehicle_reid_model_path="vehicle.xml",
                    deferred_reid_min_crop_pixels=256,
                ),
                _Events(event),
                index,
                _Encoder(),
            )

            state, count, reason = service.process_event(7)
            self.assertEqual((state, count), ("completed", 1), reason)
            self.assertTrue(index.has_event(7))
            self.assertEqual(service.process_event(7)[0], "skipped")


if __name__ == "__main__":
    unittest.main()

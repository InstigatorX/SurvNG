from __future__ import annotations

import sqlite3
import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from survng.app.ema_route_cache import (
    EmaCandidateSubmitResult,
    EmaRouteCandidateCache,
    compact_ema_candidate,
)


def _payload(score: float = 0.5545) -> dict:
    return {
        "accepted": True,
        "score": score,
        "threshold": 0.48,
        "reason": "qualified",
        "frame_count": 3,
        "features": {
            "motion_regions": [[0.1, 0.2, 0.3, 0.4]],
            "motion_region_track_id": "track-1",
            "primary_motion_source": "ema",
            "large_diagnostic_graph": list(range(1000)),
        },
        "telemetry": {"large": list(range(1000))},
    }


class EmaRouteCandidateCacheTest(unittest.TestCase):
    def test_compaction_retains_replay_evidence_only(self) -> None:
        compact = compact_ema_candidate(_payload())
        self.assertEqual(compact["schema"], 1)
        self.assertEqual(compact["score"], 0.5545)
        self.assertEqual(
            compact["features"],
            {
                "motion_regions": [[0.1, 0.2, 0.3, 0.4]],
                "motion_region_track_id": "track-1",
                "primary_motion_source": "ema",
            },
        )
        self.assertEqual(compact["telemetry"], {})

    def test_candidate_survives_restart_and_is_window_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ema-route-cache.sqlite3"
            captured_at = time.time()
            cache = EmaRouteCandidateCache(path)
            cache.start()
            self.assertEqual(
                cache.submit("back-right", captured_at, _payload()),
                EmaCandidateSubmitResult.QUEUED,
            )
            cache.close()

            recreated = EmaRouteCandidateCache(path)
            recreated.start()
            rows = recreated.between("back-right", captured_at - 1.0, captured_at + 1.0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], captured_at)
            self.assertEqual(rows[0][1]["score"], 0.5545)
            self.assertEqual(
                recreated.between("back-right", captured_at + 1.1, captured_at + 2.0),
                [],
            )
            recreated.close()

    def test_submit_is_nonblocking_while_database_is_locked_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ema-route-cache.sqlite3"
            cache = EmaRouteCandidateCache(path)
            cache.start()
            blocker = sqlite3.connect(path, timeout=1.0)
            blocker.execute("begin immediate")
            started = time.monotonic()
            base_at = time.time()
            for index in range(40):
                cache.submit(
                    f"camera-{index % 13}", base_at + index * 0.6, _payload()
                )
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.1)
            deadline = time.monotonic() + 1.0
            while cache.status()["busy_retries"] == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertGreater(cache.status()["busy_retries"], 0)
            blocker.rollback()
            blocker.close()
            deadline = time.monotonic() + 4.0
            while cache.status()["persisted"] < 40 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(cache.status()["persisted"], 40)
            self.assertFalse(cache.status()["degraded"])
            cache.close()

    def test_concurrent_cameras_are_fair_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmaRouteCandidateCache(
                Path(tmpdir) / "ema-route-cache.sqlite3",
                capacity=52,
                per_camera_capacity=4,
                batch_size=13,
            )
            cache.start()
            barrier = threading.Barrier(14)
            base_at = time.time()

            def produce(camera: int) -> None:
                barrier.wait()
                for index in range(20):
                    cache.submit(
                        f"camera-{camera}",
                        base_at + index / 10.0,
                        _payload(0.5 + camera / 100.0),
                    )

            threads = [threading.Thread(target=produce, args=(index,)) for index in range(13)]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join()
            self.assertLessEqual(cache.status()["queue_high_water"], 52)
            cache.close()
            for camera in range(13):
                self.assertTrue(
                    cache.between(f"camera-{camera}", base_at - 1.0, base_at + 3.0)
                )

    def test_same_bucket_coalesces_latest_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmaRouteCandidateCache(Path(tmpdir) / "ema-route-cache.sqlite3")
            cache.start()
            blocker = sqlite3.connect(cache.path, timeout=1.0)
            blocker.execute("begin immediate")
            captured_at = math.floor(time.time() * 2.0) / 2.0 + 0.01
            self.assertEqual(cache.submit("gate", captured_at, _payload(0.6)), "queued")
            self.assertEqual(
                cache.submit("gate", captured_at + 0.2, _payload(0.8)), "coalesced"
            )
            blocker.rollback()
            blocker.close()
            cache.close()
            rows = cache.between("gate", captured_at - 1.0, captured_at + 1.0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1]["score"], 0.8)

    def test_close_rejects_new_work_and_stops_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmaRouteCandidateCache(Path(tmpdir) / "ema-route-cache.sqlite3")
            cache.start()
            cache.close_admission()
            self.assertEqual(
                cache.submit("gate", 5000.0, _payload()),
                EmaCandidateSubmitResult.STOPPED,
            )
            cache.close()
            self.assertFalse(cache.status()["alive"])

    def test_start_migrates_recent_legacy_rows_and_removes_legacy_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy = root / "detection-jobs.sqlite3"
            captured_at = time.time()
            with sqlite3.connect(legacy) as connection:
                connection.execute(
                    "create table ema_route_candidates ("
                    "camera_id text not null, captured_at real not null, "
                    "payload_json text not null, created_at text not null, "
                    "primary key(camera_id,captured_at))"
                )
                connection.execute(
                    "insert into ema_route_candidates values (?,?,?,?)",
                    ("gate", captured_at, '{"accepted":true,"score":0.8}', "now"),
                )
            cache = EmaRouteCandidateCache(
                root / "ema-route-cache.sqlite3", legacy_jobs_path=legacy
            )
            cache.start()
            self.assertEqual(len(cache.between("gate", captured_at - 1, captured_at + 1)), 1)
            with sqlite3.connect(legacy) as connection:
                self.assertIsNone(
                    connection.execute(
                        "select 1 from sqlite_master where type='table' "
                        "and name='ema_route_candidates'"
                    ).fetchone()
                )
            cache.close()

    def test_shutdown_deadline_discards_optional_work_under_persistent_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ema-route-cache.sqlite3"
            cache = EmaRouteCandidateCache(path)
            cache.start()
            blocker = sqlite3.connect(path, timeout=1.0)
            blocker.execute("begin immediate")
            cache.submit("gate", time.time(), _payload())
            deadline = time.monotonic() + 2.0
            while cache.status()["busy_retries"] < 5 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertGreaterEqual(cache.status()["busy_retries"], 5)
            started = time.monotonic()
            cache.close(timeout=0.05)
            elapsed = time.monotonic() - started
            blocker.rollback()
            blocker.close()
            self.assertLess(elapsed, 0.3)
            self.assertFalse(cache.status()["alive"])
            self.assertGreaterEqual(cache.status()["shutdown_drops"], 1)

    def test_fatal_write_error_accounts_for_inflight_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmaRouteCandidateCache(Path(tmpdir) / "ema-route-cache.sqlite3")
            cache.start()
            with patch.object(
                cache,
                "_commit_batch",
                side_effect=sqlite3.OperationalError("disk I/O error"),
            ):
                cache.submit("gate", time.time(), _payload())
                deadline = time.monotonic() + 1.0
                while cache.status()["alive"] and time.monotonic() < deadline:
                    time.sleep(0.01)
            status = cache.status()
            self.assertFalse(status["alive"])
            self.assertEqual(status["write_error_drops"], 1)
            self.assertTrue(status["degraded"])
            self.assertEqual(
                cache.submit("gate", time.time(), _payload()),
                EmaCandidateSubmitResult.STOPPED,
            )

    def test_per_camera_overflow_keeps_newest_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ema-route-cache.sqlite3"
            cache = EmaRouteCandidateCache(
                path, capacity=8, per_camera_capacity=2, batch_size=1
            )
            cache.start()
            blocker = sqlite3.connect(path, timeout=1.0)
            blocker.execute("begin immediate")
            base_at = time.time()
            cache.submit("gate", base_at, _payload(0.5))
            deadline = time.monotonic() + 1.0
            while cache.status()["busy_retries"] == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            for offset, score in ((1.0, 0.6), (2.0, 0.7), (3.0, 0.8)):
                cache.submit("gate", base_at + offset, _payload(score))
            self.assertEqual(cache.status()["per_camera_overflow"]["gate"], 1)
            blocker.rollback()
            blocker.close()
            cache.close()
            rows = cache.between("gate", base_at - 1.0, base_at + 4.0)
            self.assertEqual({row[1]["score"] for row in rows}, {0.5, 0.7, 0.8})

    def test_global_overflow_evicts_from_largest_camera_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ema-route-cache.sqlite3"
            cache = EmaRouteCandidateCache(
                path, capacity=3, per_camera_capacity=3, batch_size=1
            )
            cache.start()
            blocker = sqlite3.connect(path, timeout=1.0)
            blocker.execute("begin immediate")
            base_at = time.time()
            cache.submit("gate", base_at, _payload(0.5))
            deadline = time.monotonic() + 1.0
            while cache.status()["busy_retries"] == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            cache.submit("gate", base_at + 1.0, _payload(0.6))
            cache.submit("gate", base_at + 2.0, _payload(0.7))
            cache.submit("front", base_at + 1.0, _payload(0.8))
            cache.submit("back", base_at + 1.0, _payload(0.9))
            self.assertEqual(cache.status()["per_camera_overflow"]["gate"], 1)
            blocker.rollback()
            blocker.close()
            cache.close()
            gate = cache.between("gate", base_at - 1.0, base_at + 3.0)
            self.assertEqual({row[1]["score"] for row in gate}, {0.5, 0.7})
            self.assertTrue(cache.between("front", base_at, base_at + 2.0))
            self.assertTrue(cache.between("back", base_at, base_at + 2.0))


if __name__ == "__main__":
    unittest.main()

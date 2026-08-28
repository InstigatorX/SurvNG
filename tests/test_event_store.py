from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from survng.app.events import EventStore


class EventStoreTest(unittest.TestCase):
    def test_detection_job_survives_store_recreation_and_expired_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = EventStore(root)
            self.assertEqual(
                store.enqueue_detection_job(
                    job_id="job-1",
                    camera_id="gate",
                    dedupe_key="episode:7",
                    payload={"event_at": "2026-08-15T12:00:00+00:00"},
                ),
                "queued",
            )
            self.assertEqual(store.enqueue_detection_job(
                job_id="job-1",
                camera_id="gate",
                dedupe_key="episode:7",
                payload={"event_at": "2026-08-15T12:00:00+00:00"},
            ), "coalesced")
            claimed = store.claim_detection_job("gate", lease_seconds=1.0)
            self.assertEqual(claimed["payload"]["event_at"], "2026-08-15T12:00:00+00:00")

            recreated = EventStore(root)
            with recreated._connect_jobs() as connection:
                connection.execute(
                    "update detection_jobs set lease_expires_at = 0 where id = 'job-1'"
                )
            reclaimed = recreated.claim_detection_job("gate")
            self.assertEqual(reclaimed["id"], "job-1")
            self.assertEqual(reclaimed["attempts"], 2)
            recreated.complete_detection_job("job-1", 42)
            status = recreated.detection_job_status("gate")
            self.assertEqual(status["completed"], 1)
            self.assertEqual(status["oldest_age_ms"], 0.0)

    def test_detection_jobs_prioritize_newest_incident_over_restart_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            now = datetime.now(timezone.utc)
            older_created_at = (now - timedelta(hours=2)).isoformat()
            newer_created_at = (now - timedelta(hours=1)).isoformat()
            event_at = now.isoformat()
            for job_id in ("older", "newer"):
                self.assertEqual(
                    store.enqueue_detection_job(
                        job_id=job_id,
                        camera_id="gate",
                        dedupe_key=f"episode:{job_id}",
                        payload={"event_at": event_at},
                    ),
                    "queued",
                )
            with store._connect_jobs() as connection:
                connection.execute(
                    "update detection_jobs set created_at = ? where id = 'older'",
                    (older_created_at,),
                )
                connection.execute(
                    "update detection_jobs set created_at = ? where id = 'newer'",
                    (newer_created_at,),
                )

            claimed = store.claim_detection_job(
                "gate",
                maximum_age_seconds=86400.0,
            )

            self.assertEqual(claimed["id"], "newer")

    def test_detection_job_claim_reclaims_fresh_expired_lease_before_newer_queued(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            for job_id in ("zombie", "fresh"):
                self.assertEqual(
                    store.enqueue_detection_job(
                        job_id=job_id,
                        camera_id="gate",
                        dedupe_key=f"episode:{job_id}",
                        payload={"event_at": "2026-08-22T20:00:00+00:00"},
                    ),
                    "queued",
                )
            with store._connect_jobs() as connection:
                connection.execute(
                    "update detection_jobs set state = 'running', "
                    "lease_expires_at = 0, lease_owner = 'dead-worker' "
                    "where id = 'zombie'",
                )

            claimed = store.claim_detection_job("gate", lease_owner="refiner-a")

            self.assertEqual(claimed["id"], "zombie")
            self.assertEqual(claimed["attempts"], 1)

    def test_stale_queued_detection_jobs_become_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store.enqueue_detection_job(
                job_id="stale", camera_id="gate", dedupe_key="episode:stale",
                payload={"event_at": "2026-08-22T19:00:00+00:00"},
            )
            with store._connect_jobs() as connection:
                connection.execute(
                    "update detection_jobs set created_at = ? where id = 'stale'",
                    ("2026-08-22T19:00:00+00:00",),
                )

            expired = store.expire_stale_detection_jobs(
                "gate", maximum_age_seconds=1.0,
            )

            self.assertEqual(expired, 1)
            self.assertIsNone(store.claim_detection_job("gate"))
            with store._connect_jobs() as connection:
                row = connection.execute(
                    "select state, last_error from detection_jobs where id = 'stale'"
                ).fetchone()
            self.assertEqual((row["state"], row["last_error"]), ("failed", "stale_refinement"))

    def test_stale_expired_lease_does_not_block_fresh_detection_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            for job_id in ("zombie", "fresh"):
                store.enqueue_detection_job(
                    job_id=job_id,
                    camera_id="gate",
                    dedupe_key=f"episode:{job_id}",
                    payload={"event_at": "2026-08-22T20:00:00+00:00"},
                )
            with store._connect_jobs() as connection:
                connection.execute(
                    "update detection_jobs set state = 'running', "
                    "lease_expires_at = 0, lease_owner = 'dead-worker', "
                    "created_at = ? where id = 'zombie'",
                    ("2026-08-22T19:00:00+00:00",),
                )

            claimed = store.claim_detection_job(
                "gate",
                lease_owner="refiner-a",
                maximum_age_seconds=1.0,
            )

            self.assertEqual(claimed["id"], "fresh")
            with store._connect_jobs() as connection:
                row = connection.execute(
                    "select state, last_error from detection_jobs where id = 'zombie'"
                ).fetchone()
            self.assertEqual((row["state"], row["last_error"]), ("failed", "stale_refinement"))

    def test_security_work_ledger_is_separate_from_general_event_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            blocker = store._connect()
            blocker.execute("begin immediate")
            try:
                self.assertTrue(store.enqueue_motion_trigger(
                    job_id="trigger-1",
                    camera_id="gate",
                    payload={"score": np.float32(0.75)},
                ))
            finally:
                blocker.rollback()
                blocker.close()
            self.assertTrue(store.jobs_db_path.exists())
            with store._connect_jobs() as connection:
                self.assertEqual(connection.execute("pragma synchronous").fetchone()[0], 2)
                self.assertIsNone(connection.execute(
                    "select 1 from sqlite_master where type='table' "
                    "and name='ema_route_candidates'"
                ).fetchone())
            with store._connect() as connection:
                legacy = connection.execute(
                    "select count(*) from sqlite_master where type = 'table' "
                    "and name in ('detection_jobs', 'motion_trigger_jobs')"
                ).fetchone()[0]
            self.assertEqual(legacy, 0)

    def test_legacy_security_jobs_migrate_before_source_tables_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            database = root / "survng.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "create table detection_jobs (id text primary key, camera_id text, "
                    "dedupe_key text, payload_json text, state text, attempts integer, "
                    "available_at real, lease_expires_at real, event_id integer, "
                    "last_error text, created_at text, updated_at text)"
                )
                connection.execute(
                    "insert into detection_jobs values "
                    "('job-1','gate','episode:1','{}','queued',0,0,null,null,'','now','now')"
                )
                connection.execute(
                    "create table motion_trigger_jobs (id text primary key, camera_id text, "
                    "payload_json text, state text, attempts integer, available_at real, "
                    "lease_expires_at real, last_error text, created_at text, updated_at text)"
                )
                connection.execute(
                    "insert into motion_trigger_jobs values "
                    "('trigger-1','gate','{}','queued',0,0,null,'','now','now')"
                )

            store = EventStore(root)

            status = store.detection_job_status("gate")
            self.assertEqual(status["queued"], 1)
            self.assertEqual(status["oldest_age_ms"], 0.0)
            self.assertEqual(store.motion_trigger_status("gate"), {"queued": 1})
            with store._connect() as connection:
                remaining = connection.execute(
                    "select count(*) from sqlite_master where type = 'table' "
                    "and name in ('detection_jobs', 'motion_trigger_jobs')"
                ).fetchone()[0]
            self.assertEqual(remaining, 0)

    def test_nonfinite_durable_job_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            with self.assertRaises(ValueError):
                store.enqueue_motion_trigger(
                    job_id="trigger-1",
                    camera_id="gate",
                    payload={"score": float("nan")},
                )

    def test_motion_trigger_exact_retry_coalesces_but_occurrence_collision_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            payload = {
                "topic": "adaptive/visual_backup",
                "event_at": "2026-08-16T14:44:14+00:00",
                "episode_id": "gate:i1:g1:e3",
                "detection_intent_id": "gate:i1:g1:e3:request:1",
                "lifecycle_generation": 1,
                "retry_count": 0,
            }
            self.assertTrue(store.enqueue_motion_trigger(
                job_id="intent-1", camera_id="gate", payload=payload
            ))
            self.assertFalse(store.enqueue_motion_trigger(
                job_id="intent-1",
                camera_id="gate",
                payload={**payload, "retry_count": 1},
            ))
            with self.assertRaisesRegex(RuntimeError, "different occurrence"):
                store.enqueue_motion_trigger(
                    job_id="intent-1",
                    camera_id="gate",
                    payload={
                        **payload,
                        "event_at": "2026-08-16T14:45:14+00:00",
                    },
                )

    def test_route_motion_trigger_replay_is_idempotent_across_capture_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            route_id = "route:gate:lower-garage:53952"
            payload = {
                "topic": "adaptive/visual_backup",
                "event_at": "2026-08-22T18:10:01+00:00",
                "episode_id": route_id,
                "detection_intent_id": route_id,
                "lifecycle_generation": 4,
            }
            self.assertTrue(
                store.enqueue_motion_trigger(
                    job_id=route_id,
                    camera_id="gate",
                    payload=payload,
                )
            )
            # Same route identity with a later EMA capture time / worker
            # generation must coalesce instead of raising.
            self.assertFalse(
                store.enqueue_motion_trigger(
                    job_id=route_id,
                    camera_id="gate",
                    payload={
                        **payload,
                        "event_at": "2026-08-22T18:10:07+00:00",
                        "lifecycle_generation": 5,
                    },
                )
            )
            self.assertEqual(store.motion_trigger_status("gate"), {"queued": 1})
            with self.assertRaisesRegex(RuntimeError, "different occurrence"):
                store.enqueue_motion_trigger(
                    job_id=route_id,
                    camera_id="porch",
                    payload=payload,
                )
            other_route = "route:gate:lower-garage:53953"
            with self.assertRaisesRegex(RuntimeError, "different occurrence"):
                store.enqueue_motion_trigger(
                    job_id=route_id,
                    camera_id="gate",
                    payload={
                        **payload,
                        "episode_id": other_route,
                        "detection_intent_id": other_route,
                    },
                )
            # route:v2: identities remain distinct from legacy route: keys for
            # the same target/source/event tuple until an explicit migration
            # rewrites persisted jobs.
            migrated = "route:v2:gate:lower-garage:53952"
            self.assertTrue(
                store.enqueue_motion_trigger(
                    job_id=migrated,
                    camera_id="gate",
                    payload={
                        **payload,
                        "episode_id": migrated,
                        "detection_intent_id": migrated,
                    },
                )
            )
            self.assertEqual(store.motion_trigger_status("gate")["queued"], 2)

    def test_motion_trigger_lease_owner_prevents_stale_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store.enqueue_motion_trigger(
                job_id="trigger-1",
                camera_id="gate",
                payload={"topic": "onvif/motion"},
            )
            self.assertIsNotNone(store.claim_motion_trigger(
                "gate", "trigger-1", lease_owner="generation-a"
            ))
            self.assertIsNone(store.claim_motion_trigger(
                "gate", "trigger-1", lease_owner="generation-b"
            ))
            store.complete_motion_trigger("trigger-1", lease_owner="generation-b")
            self.assertEqual(store.motion_trigger_status("gate"), {"running": 1})
            store.complete_motion_trigger("trigger-1", lease_owner="generation-a")
            self.assertEqual(store.motion_trigger_status("gate"), {})

    def test_route_watch_consumption_survives_store_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = EventStore(root)

            self.assertFalse(store.route_watch_consumed("back-left", 44720))
            store.mark_route_watch_consumed("back-left", 44720)
            store.mark_route_watch_consumed("back-left", 44720)
            with store._connect_jobs() as connection:
                connection.execute(
                    "update route_watch_consumptions set consumed_at = ? "
                    "where target_camera_id = ? and source_event_id = ?",
                    ("2020-01-01T00:00:00+00:00", "back-left", 44720),
                )
            store.mark_route_watch_consumed("back-right", 44721)

            recreated = EventStore(root)
            self.assertFalse(recreated.route_watch_consumed("back-left", 44720))
            self.assertTrue(recreated.route_watch_consumed("back-right", 44721))

    def test_route_target_admission_is_durable_and_returns_canonical_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = EventStore(root)
            first = store.add_event(
                camera_id="upper-garage",
                kind="motion",
                detection_intent_id="route-direct",
                route_origin_camera_id="gate",
                route_origin_event_id=44720,
            )
            duplicate = store.add_event(
                camera_id="upper-garage",
                kind="motion",
                detection_intent_id="route-via-lower",
                route_origin_camera_id="gate",
                route_origin_event_id=44720,
            )

            self.assertTrue(first["created"])
            self.assertFalse(duplicate["created"])
            self.assertEqual(duplicate["id"], first["id"])
            with store._connect() as connection:
                count = connection.execute(
                    "select count(*) from events where camera_id = ?",
                    ("upper-garage",),
                ).fetchone()[0]
            self.assertEqual(count, 1)

            recreated = EventStore(root)
            self.assertTrue(recreated.route_target_admitted(
                "gate", 44720, "upper-garage"
            ))

    def test_route_target_admission_is_atomic_across_store_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stores = (EventStore(root), EventStore(root))
            barrier = threading.Barrier(2)
            outcomes: list[dict] = []

            def admit(index: int) -> None:
                barrier.wait()
                outcomes.append(stores[index].add_event(
                    camera_id="upper-garage",
                    kind="motion",
                    detection_intent_id=f"alternate-{index}",
                    route_origin_camera_id="gate",
                    route_origin_event_id=44720,
                ))

            threads = [threading.Thread(target=admit, args=(index,)) for index in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5.0)

            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(len(outcomes), 2)
            self.assertEqual(sum(bool(item["created"]) for item in outcomes), 1)
            self.assertEqual(len({int(item["id"]) for item in outcomes}), 1)
            with stores[0]._connect() as connection:
                count = connection.execute(
                    "select count(*) from events where camera_id = ?",
                    ("upper-garage",),
                ).fetchone()[0]
            self.assertEqual(count, 1)

    def test_duplicate_route_admission_removes_unreferenced_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = EventStore(root)
            store.add_event(
                camera_id="upper-garage",
                kind="motion",
                detection_intent_id="route-direct",
                route_origin_camera_id="gate",
                route_origin_event_id=44720,
            )
            snapshot = root / "snapshots" / "upper-garage" / "alternate.webp"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"alternate")

            duplicate = store.add_event(
                camera_id="upper-garage",
                kind="motion",
                snapshot_path=str(snapshot),
                detection_intent_id="route-via-lower",
                route_origin_camera_id="gate",
                route_origin_event_id=44720,
            )

            self.assertFalse(duplicate["created"])
            self.assertFalse(snapshot.exists())

    def test_detection_job_lease_owner_prevents_stale_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store.enqueue_detection_job(
                job_id="job-1",
                camera_id="gate",
                dedupe_key="episode:1",
                payload={},
            )
            claimed = store.claim_detection_job(
                "gate", lease_owner="generation-a"
            )
            self.assertIsNotNone(claimed)
            store.complete_detection_job(
                "job-1", 10, lease_owner="generation-b"
            )
            status = store.detection_job_status("gate")
            self.assertEqual(status["running"], 1)
            self.assertGreaterEqual(status["oldest_age_ms"], 0.0)
            store.complete_detection_job(
                "job-1", 10, lease_owner="generation-a"
            )
            status = store.detection_job_status("gate")
            self.assertEqual(status["completed"], 1)
            self.assertEqual(status["oldest_age_ms"], 0.0)

    def test_detection_intent_idempotently_links_one_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            created_at = "2026-08-15T12:00:00+00:00"
            first = store.add_event(
                "gate",
                "motion",
                created_at=created_at,
                detection_intent_id="intent-1",
            )
            replay = store.add_event(
                "gate",
                "motion",
                created_at=created_at,
                detection_intent_id="intent-1",
            )
            self.assertTrue(first["created"])
            self.assertFalse(replay["created"])
            self.assertEqual(first["id"], replay["id"])
            with store._connect() as connection:
                count = connection.execute("select count(*) from events").fetchone()[0]
            self.assertEqual(count, 1)

    def test_detection_job_collision_rejects_different_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store.enqueue_detection_job(
                job_id="job-1",
                camera_id="gate",
                dedupe_key="intent:first",
                payload={"event_at": "2026-08-15T12:00:00+00:00"},
            )

            with self.assertRaisesRegex(RuntimeError, "different occurrence"):
                store.enqueue_detection_job(
                    job_id="job-1",
                    camera_id="gate",
                    dedupe_key="intent:first",
                    payload={"event_at": "2026-08-16T12:00:00+00:00"},
                )
            with self.assertRaisesRegex(RuntimeError, "different occurrence"):
                store.enqueue_detection_job(
                    job_id="job-2",
                    camera_id="gate",
                    dedupe_key="intent:first",
                    payload={"event_at": "2026-08-15T12:00:00+00:00"},
                )

    def test_detection_job_retry_metadata_does_not_change_occurrence_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            payload = {
                "topic": "adaptive/visual_backup",
                "event_at": "2026-08-16T14:44:14+00:00",
                "qualification": {"detection_intent_id": "intent-1", "retry_count": 0},
                "initial_outcome": {"processing_timing": {"queue_wait_ms": 20}},
                "require_eligible_object": True,
                "require_motion_correlation": True,
            }
            self.assertEqual(store.enqueue_detection_job(
                job_id="job-1",
                camera_id="gate",
                dedupe_key="intent:intent-1",
                payload=payload,
            ), "queued")
            self.assertEqual(store.enqueue_detection_job(
                job_id="job-1",
                camera_id="gate",
                dedupe_key="intent:intent-1",
                payload={
                    **payload,
                    "qualification": {
                        "detection_intent_id": "intent-1",
                        "retry_count": 1,
                    },
                    "initial_outcome": {
                        "processing_timing": {"queue_wait_ms": 45},
                    },
                },
            ), "coalesced")

    def test_route_detection_job_replay_ignores_capture_time_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            route_id = "route:lower-garage:upper-garage:57986"
            payload = {
                "topic": "adaptive/visual_backup",
                "event_at": "2026-08-27T23:40:23.987714+00:00",
                "qualification": {"detection_intent_id": route_id},
                "existing_event_id": None,
                "require_eligible_object": True,
                "require_motion_correlation": True,
            }
            self.assertEqual(store.enqueue_detection_job(
                job_id="job-1",
                camera_id="lower-garage",
                dedupe_key=f"intent:{route_id}",
                payload=payload,
            ), "queued")
            self.assertEqual(store.enqueue_detection_job(
                job_id="job-1",
                camera_id="lower-garage",
                dedupe_key=f"intent:{route_id}",
                payload={
                    **payload,
                    "event_at": "2026-08-27T23:40:26.359163+00:00",
                },
            ), "coalesced")

            with self.assertRaisesRegex(RuntimeError, "different occurrence"):
                store.enqueue_detection_job(
                    job_id="job-1",
                    camera_id="lower-garage",
                    dedupe_key=f"intent:{route_id}",
                    payload={
                        **payload,
                        "event_at": "2026-08-27T23:40:26.359163+00:00",
                        "require_motion_correlation": False,
                    },
                )

    def test_detection_job_pruning_is_bounded_and_preserves_active_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            for index in range(4):
                store.enqueue_detection_job(
                    job_id=f"job-{index}",
                    camera_id="gate",
                    dedupe_key=f"intent:{index}",
                    payload={"event_at": f"2026-08-0{index + 1}T12:00:00+00:00"},
                )
            with store._connect_jobs() as connection:
                connection.execute(
                    "update detection_jobs set state='completed', "
                    "updated_at='2026-01-01T00:00:00+00:00' where id in ('job-0','job-1')"
                )
                connection.execute(
                    "update detection_jobs set state='failed', "
                    "updated_at='2026-01-01T00:00:00+00:00' where id='job-2'"
                )

            self.assertEqual(store.prune_detection_jobs(limit=2, force=True), 2)
            self.assertEqual(store.prune_detection_jobs(limit=2, force=True), 1)
            status = store.detection_job_status("gate")
            self.assertEqual(status["queued"], 1)
            self.assertGreaterEqual(status["oldest_age_ms"], 0.0)

    def test_detection_intent_collision_rejects_different_event_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store.add_event(
                "gate",
                "motion",
                created_at="2026-08-15T12:00:00+00:00",
                detection_intent_id="intent-1",
            )

            with self.assertRaisesRegex(RuntimeError, "different occurrence"):
                store.add_event(
                    "gate",
                    "motion",
                    created_at="2026-08-16T12:00:00+00:00",
                    detection_intent_id="intent-1",
                )

    def test_motion_trigger_lease_is_not_stolen_by_store_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = EventStore(root)
            store.enqueue_motion_trigger(
                job_id="trigger-1",
                camera_id="gate",
                payload={"topic": "onvif/motion"},
            )
            self.assertIsNotNone(store.claim_motion_trigger("gate"))
            concurrent = EventStore(root)
            self.assertIsNone(concurrent.claim_motion_trigger("gate"))
            store.release_motion_trigger("trigger-1")
            self.assertIsNotNone(concurrent.claim_motion_trigger("gate"))

    def test_motion_trigger_poison_job_reaches_explicit_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store.enqueue_motion_trigger(
                job_id="trigger-1",
                camera_id="gate",
                payload={"topic": "onvif/motion"},
            )
            for attempt in range(1, 6):
                if attempt > 1:
                    with store._connect_jobs() as connection:
                        connection.execute(
                            "update motion_trigger_jobs set available_at = 0 "
                            "where id = 'trigger-1'"
                        )
                self.assertIsNotNone(
                    store.claim_motion_trigger("gate", "trigger-1")
                )
                retrying = store.fail_motion_trigger("trigger-1", "detector failed")
                self.assertEqual(retrying, attempt < 5)
            self.assertEqual(store.motion_trigger_status("gate"), {"failed": 1})
            self.assertIsNone(store.claim_motion_trigger("gate"))

    def test_snapshot_size_backfill_yields_writer_between_bounded_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = EventStore(root)
            for index in range(5):
                snapshot = root / "snapshots" / "gate" / f"legacy-{index}.webp"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(f"snapshot-{index}".encode())
                store.add_event(
                    camera_id="gate",
                    kind="object",
                    snapshot_path=str(snapshot),
                )
            with store._connect() as connection:
                connection.execute("update events set snapshot_size_bytes = 0")

            original_connect = store._connect
            connection_count = 0

            def counted_connect():
                nonlocal connection_count
                connection_count += 1
                return original_connect()

            store._connect = counted_connect  # type: ignore[method-assign]
            updated = store.migrate_snapshot_sizes(limit=5, write_batch_size=2)

            self.assertEqual(updated, 5)
            # One read, three independently committed size batches, and one
            # checked-path transaction keep ordinary writers from waiting on
            # the filesystem migration.
            self.assertEqual(connection_count, 5)
            with original_connect() as connection:
                sizes = connection.execute(
                    "select snapshot_size_bytes from events order by id"
                ).fetchall()
            self.assertTrue(all(int(row[0]) > 0 for row in sizes))

    def test_missing_snapshot_cohort_does_not_starve_later_size_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = EventStore(root)
            snapshots = root / "snapshots" / "gate"
            snapshots.mkdir(parents=True)
            for index in range(3):
                store.add_event(
                    camera_id="gate",
                    kind="object",
                    snapshot_path=str(snapshots / f"missing-{index}.webp"),
                )
            for index in range(2):
                snapshot = snapshots / f"present-{index}.webp"
                snapshot.write_bytes(f"present-{index}".encode())
                store.add_event(
                    camera_id="gate",
                    kind="object",
                    snapshot_path=str(snapshot),
                )
            with store._connect() as connection:
                connection.execute("update events set snapshot_size_bytes = 0")

            self.assertEqual(store.migrate_snapshot_sizes(limit=3), 0)
            self.assertEqual(store.migrate_snapshot_sizes(limit=3), 2)
            with store._connect() as connection:
                rows = connection.execute(
                    "select snapshot_path, snapshot_size_bytes from events order by id"
                ).fetchall()
            self.assertTrue(all(int(row["snapshot_size_bytes"]) == 0 for row in rows[:3]))
            self.assertTrue(all(int(row["snapshot_size_bytes"]) > 0 for row in rows[3:]))

    def test_snapshot_retention_plan_is_read_only_and_does_not_stat_legacy_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot = root / "snapshots" / "gate" / "legacy.webp"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"legacy")
            store = EventStore(root)
            store.add_event(
                camera_id="gate",
                kind="object",
                snapshot_path=str(snapshot),
                created_at="2020-01-01T00:00:00+00:00",
            )
            with store._connect() as connection:
                connection.execute("update events set snapshot_size_bytes = 0")

            with patch.object(
                store,
                "_snapshot_file_size",
                side_effect=AssertionError("retention planning must not stat media"),
            ):
                plan = store.snapshot_retention_plan(
                    datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
                )

            self.assertEqual(plan["file_count"], 1)
            self.assertEqual(plan["bytes"], 0)
            self.assertEqual(plan["unindexed_files"], 1)
            self.assertEqual(plan["expired_files"], 1)

    def test_snapshot_retention_plan_does_not_acquire_writer_mutex(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store._lock.acquire()
            try:
                completed = threading.Event()
                failures: list[BaseException] = []

                def plan() -> None:
                    try:
                        store.snapshot_retention_plan(
                            datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
                        )
                    except BaseException as error:
                        failures.append(error)
                    finally:
                        completed.set()

                worker = threading.Thread(target=plan)
                worker.start()
                self.assertTrue(completed.wait(1.0))
                worker.join(timeout=1.0)
                self.assertFalse(failures)
            finally:
                store._lock.release()

    def test_snapshot_cleanup_yields_writer_between_reference_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = EventStore(root)
            for index in range(5):
                snapshot = root / "snapshots" / "gate" / f"old-{index}.webp"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(f"old-{index}".encode())
                store.add_event(
                    camera_id="gate",
                    kind="object",
                    snapshot_path=str(snapshot),
                    created_at="2020-01-01T00:00:00+00:00",
                )

            original_connect = store._connect
            connection_count = 0

            def counted_connect():
                nonlocal connection_count
                connection_count += 1
                return original_connect()

            store.SNAPSHOT_REFERENCE_WRITE_BATCH = 2
            store._connect = counted_connect  # type: ignore[method-assign]
            result = store.apply_snapshot_retention(
                datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp(),
                100,
            )

            self.assertEqual(result["deleted_files"], 5)
            # One transactional claim, three independently committed reference
            # cleanup batches, and one claim release transaction.
            self.assertEqual(connection_count, 5)
            with original_connect() as connection:
                self.assertEqual(
                    connection.execute(
                        "select count(*) from events where snapshot_path != ''"
                    ).fetchone()[0],
                    0,
                )

    def test_snapshot_retention_reports_usage_and_clears_expired_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot = root / "snapshots" / "gate" / "old.webp"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"snapshot-bytes")
            store = EventStore(root)
            event = store.add_event(
                camera_id="gate",
                kind="object",
                snapshot_path=str(snapshot),
                created_at="2020-01-01T00:00:00+00:00",
            )

            plan = store.snapshot_retention_plan(
                datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
            )
            result = store.apply_snapshot_retention(
                datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp(),
                100,
            )

            self.assertEqual(plan["bytes"], len(b"snapshot-bytes"))
            self.assertEqual(plan["expired_files"], 1)
            self.assertEqual(result["deleted_files"], 1)
            self.assertFalse(snapshot.exists())
            with store._connect() as connection:
                row = connection.execute(
                    "select snapshot_path from events where id = ?",
                    (int(event["id"]),),
                ).fetchone()
            self.assertEqual(row["snapshot_path"], "")

    def test_snapshot_retention_protects_pinned_face_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot = root / "snapshots" / "gate" / "reference.webp"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"reference")
            store = EventStore(root)
            event = store.add_event(
                camera_id="gate",
                kind="object",
                snapshot_path=str(snapshot),
                created_at="2020-01-01T00:00:00+00:00",
            )
            with store._connect() as connection:
                connection.execute(
                    """
                    create table face_observations (
                        id integer primary key, snapshot_path text not null,
                        reference_pinned integer not null default 0
                    )
                    """
                )
                connection.execute(
                    "insert into face_observations(snapshot_path, reference_pinned) values (?, 1)",
                    (event["snapshot_path"],),
                )

            plan = store.snapshot_retention_plan(
                datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
            )
            result = store.apply_snapshot_retention(
                datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp(),
                100,
            )

            self.assertEqual(plan["expired_files"], 0)
            self.assertEqual(result["selected_files"], 0)
            self.assertTrue(snapshot.exists())

    def test_reopening_store_recovers_snapshot_claim_after_unlink_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot = root / "snapshots" / "gate" / "old.webp"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"snapshot")
            store = EventStore(root)
            event = store.add_event(
                camera_id="gate",
                kind="object",
                snapshot_path=str(snapshot),
                created_at="2020-01-01T00:00:00+00:00",
            )
            with store._connect() as connection:
                connection.execute(
                    "insert into media_deletion_claims(path, role, claimed_at) values (?, 'snapshot', ?)",
                    (event["snapshot_path"], "2026-01-01T00:00:00+00:00"),
                )
            snapshot.unlink()

            recovered = EventStore(root)

            with recovered._connect() as connection:
                row = connection.execute(
                    "select snapshot_path from events where id = ?", (event["id"],)
                ).fetchone()
                claims = connection.execute(
                    "select count(*) from media_deletion_claims where role = 'snapshot'"
                ).fetchone()[0]
            self.assertEqual(row["snapshot_path"], "")
            self.assertEqual(claims, 0)
    def test_protected_recording_paths_use_lexical_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = EventStore(root)
            recording = root / "secondary" / "recordings" / "gate" / ".." / "gate" / "segment.mp4"
            store.add_event(
                camera_id="gate",
                kind="object",
                snapshot_path="",
                recording_path=str(recording),
                objects_json="[]",
            )

            with patch(
                "survng.app.event_store.store.Path.resolve",
                side_effect=AssertionError("protection lookup must not resolve media paths"),
            ):
                protected = store.protected_recording_paths()

            self.assertEqual(protected, {str(root / "secondary" / "recordings" / "gate" / "segment.mp4")})

    def test_refinement_removes_replaced_snapshot_when_unreferenced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshots = root / "snapshots" / "gate"
            snapshots.mkdir(parents=True)
            old = snapshots / "old.webp"
            new = snapshots / "new.webp"
            old.write_bytes(b"old")
            new.write_bytes(b"new")
            store = EventStore(root)
            event = store.add_event(
                camera_id="gate",
                kind="motion",
                snapshot_path=str(old),
                objects_json="[]",
            )

            store.refine_event_evidence(
                int(event["id"]),
                snapshot_path=str(new),
                recording_path="",
                objects_json="[]",
            )

            self.assertFalse(old.exists())
            self.assertTrue(new.exists())

    def test_refinement_keeps_old_snapshot_until_audit_reference_moves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshots = root / "snapshots" / "gate"
            snapshots.mkdir(parents=True)
            old = snapshots / "old.webp"
            new = snapshots / "new.webp"
            old.write_bytes(b"old")
            new.write_bytes(b"new")
            store = EventStore(root)
            event = store.add_event(
                camera_id="gate",
                kind="motion",
                snapshot_path=str(old),
                objects_json="[]",
            )
            audit_payload = {
                "decision_id": "gate-decision",
                "camera_id": "gate",
                "created_at": "2026-08-08T12:00:00+00:00",
                "mode": "camera_rescue",
                "sensitivity": "balanced",
                "score": 0.8,
                "threshold": 0.5,
                "reason": "visual_backup_trigger",
                "object_detected": True,
                "trigger_count": 1,
                "features": {},
                "category": "visual_backup",
                "event_id": int(event["id"]),
            }
            store.add_motion_audit(snapshot_path=str(old), **audit_payload)

            store.refine_event_evidence(
                int(event["id"]),
                snapshot_path=str(new),
                recording_path="",
                objects_json="[]",
            )
            self.assertTrue(old.exists())

            store.add_motion_audit(snapshot_path=str(new), **audit_payload)

            self.assertFalse(old.exists())
            self.assertTrue(new.exists())

    def test_refinement_cover_promotes_better_unique_subject_without_changing_admission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshots = root / "snapshots" / "back-right"
            snapshots.mkdir(parents=True)
            old = snapshots / "live.webp"
            new = snapshots / "main.webp"
            old.write_bytes(b"live")
            new.write_bytes(b"main-frame")
            store = EventStore(root)
            event = store.add_event(
                camera_id="back-right",
                kind="motion",
                snapshot_path=str(old),
                objects_json=json.dumps([{
                    "label": "person",
                    "confidence": 0.7759,
                    "incident_eligible": True,
                    "provisional_detection": True,
                    "frame_captured_at_epoch": 1000.0,
                    "detection_frame_width": 896,
                    "detection_frame_height": 512,
                    "box": {"x1": 524, "y1": 257, "x2": 591, "y2": 347},
                }]),
            )

            promoted = store.promote_refinement_cover(
                int(event["id"]),
                snapshot_path=str(new),
                recording_path="recordings/back-right/main.mp4",
                captured_at=1004.0,
                frame_width=4512,
                frame_height=2512,
                cover_objects=[
                    {
                        "label": "person",
                        "confidence": 0.91,
                        "confidence_eligible": True,
                        "zone_eligible": True,
                        "temporal_consensus": True,
                        "incident_eligible": False,
                        "box": {"x1": 3900, "y1": 1500, "x2": 4300, "y2": 2200},
                    },
                    {
                        "label": "car",
                        "confidence": 0.94,
                        "temporal_consensus": True,
                        "box": {"x1": 400, "y1": 700, "x2": 1100, "y2": 1300},
                    },
                ],
                source="recorded_main",
                timestamp_exact=True,
            )

            self.assertIsNotNone(promoted)
            updated = store.get(int(event["id"]))
            self.assertEqual(updated["snapshot_path"], "snapshots/back-right/main.webp")
            objects = json.loads(updated["objects_json"])
            person = next(item for item in objects if item.get("label") == "person")
            self.assertTrue(person["incident_eligible"])
            self.assertEqual(person["confidence"], 0.7759)
            self.assertEqual(person["box"], {"x1": 3900, "y1": 1500, "x2": 4300, "y2": 2200})
            self.assertEqual(person["detection_frame_width"], 4512)
            self.assertEqual(person["snapshot_source"], "recorded_main")
            self.assertTrue(person["snapshot_presentation_only"])
            promotion = next(
                item["cover_promotion"]
                for item in objects
                if item.get("status") == "cover_promotion"
            )
            self.assertTrue(promotion["admission_preserved"])
            self.assertFalse(old.exists())

    def test_refinement_cover_declines_ambiguous_same_label_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshots = root / "snapshots" / "gate"
            snapshots.mkdir(parents=True)
            old = snapshots / "live.webp"
            new = snapshots / "main.webp"
            old.write_bytes(b"live")
            new.write_bytes(b"main")
            store = EventStore(root)
            event = store.add_event(
                camera_id="gate",
                kind="motion",
                snapshot_path=str(old),
                objects_json=json.dumps([{
                    "label": "person",
                    "confidence": 0.9,
                    "incident_eligible": True,
                    "provisional_detection": True,
                    "frame_captured_at_epoch": 1000.0,
                    "detection_frame_width": 640,
                    "detection_frame_height": 360,
                    "box": {"x1": 20, "y1": 20, "x2": 80, "y2": 180},
                }]),
            )
            candidates = [
                {
                    "label": "person",
                    "confidence": 0.9,
                    "temporal_consensus": True,
                    "box": {"x1": x, "y1": 100, "x2": x + 200, "y2": 700},
                }
                for x in (200, 900)
            ]

            self.assertIsNone(store.promote_refinement_cover(
                int(event["id"]),
                snapshot_path=str(new),
                recording_path="",
                captured_at=1002.0,
                frame_width=1920,
                frame_height=1080,
                cover_objects=candidates,
                source="recorded_main",
                timestamp_exact=True,
            ))
            self.assertEqual(store.get(int(event["id"]))["snapshot_path"], "snapshots/gate/live.webp")
            self.assertTrue(old.exists())

    def test_camera_intelligence_effectiveness_lifecycle_is_durable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            evaluation = store.create_camera_intelligence_evaluation(
                camera_id="gate",
                baseline_review_id=4,
                evaluation_hours=24,
                applied_changes=[{"setting": "frame_width", "proposed": 480}],
                baseline_result={"analyzed": 8},
            )
            old_applied_at = "2026-01-01T00:00:00+00:00"
            with store._connect() as conn:
                conn.execute(
                    "update camera_intelligence_evaluations set applied_at = ? where id = ?",
                    (old_applied_at, evaluation["id"]),
                )

            ready = store.get_camera_intelligence_evaluation(evaluation["id"])
            reviewing = store.start_camera_intelligence_followup(
                evaluation["id"],
                12,
            )
            completed = store.complete_camera_intelligence_evaluation(
                evaluation["id"],
                followup_result={"analyzed": 7},
                comparison={"outcome": "improved"},
            )
            reloaded = EventStore(Path(tmpdir)).latest_camera_intelligence_evaluation(
                "gate"
            )

        self.assertEqual(ready["status"], "ready")
        self.assertEqual(reviewing["status"], "reviewing")
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(reloaded["comparison"]["outcome"], "improved")
        self.assertEqual(reloaded["baseline_result"]["analyzed"], 8)
        self.assertEqual(reloaded["followup_result"]["analyzed"], 7)

    def test_recent_camera_range_is_filtered_and_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store.add_event("gate", "motion", created_at="2026-07-28T10:00:00+00:00")
            newest = store.add_event("gate", "motion", created_at="2026-07-28T11:00:00+00:00")
            store.add_event("foyer", "motion", created_at="2026-07-28T11:30:00+00:00")

            rows = store.recent_for_camera_range(
                "gate",
                "2026-07-28T09:00:00+00:00",
                "2026-07-28T12:00:00+00:00",
                limit=1,
            )

        self.assertEqual([row["id"] for row in rows], [newest["id"]])

    def test_telemetry_activity_groups_events_objects_and_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            now = datetime.fromisoformat("2026-07-28T12:30:00+00:00")
            store.add_event(
                "gate",
                "motion",
                created_at="2026-07-28T12:10:00+00:00",
                objects_json=json.dumps([
                    {"label": "person", "incident_eligible": True},
                    {"label": "car", "incident_eligible": False},
                    {"status": "motion_qualification"},
                ]),
            )
            store.add_event(
                "foyer",
                "motion",
                created_at="2026-07-28T10:10:00+00:00",
                objects_json=json.dumps([
                    {"label": "person"},
                    {"label": "dog", "incident_eligible": True},
                ]),
            )
            store.add_event(
                "gate",
                "motion",
                created_at="2026-07-27T01:00:00+00:00",
                objects_json=json.dumps([{"label": "person", "incident_eligible": True}]),
            )

            summary = store.telemetry_activity(hours=24, now=now)

            self.assertEqual(len(summary["hourly"]), 24)
            self.assertEqual(summary["last_hour"]["events"], 1)
            self.assertEqual(summary["last_hour"]["object_incidents"], 1)
            self.assertEqual(summary["last_24h"]["events"], 2)
            self.assertEqual(summary["last_24h"]["objects"], 3)
            self.assertEqual(summary["last_24h"]["labels"], {"person": 2, "dog": 1})
            self.assertEqual(summary["by_camera"]["gate"]["last_24h"]["objects"], 1)
            self.assertEqual(summary["by_camera"]["foyer"]["last_hour"]["events"], 0)
            self.assertEqual(sum(item["events"] for item in summary["by_camera"]["foyer"]["hourly"]), 1)
            self.assertEqual(sum(item["objects"] for item in summary["by_camera"]["foyer"]["hourly"]), 2)

    def test_telemetry_activity_bounds_requested_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            now = datetime.fromisoformat("2026-07-28T12:30:00+00:00")

            self.assertEqual(store.telemetry_activity(hours=0, now=now)["hours"], 1)
            self.assertEqual(store.telemetry_activity(hours=999, now=now)["hours"], 168)

    def test_tracking_capacity_history_includes_waits_and_legacy_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            now = datetime.fromisoformat("2026-08-02T12:30:00+00:00")
            store.add_event(
                "gate",
                "motion",
                created_at="2026-08-02T12:01:00+00:00",
                objects_json=json.dumps([{
                    "status": "object_tracking",
                    "object_tracking": {
                        "state": "complete",
                        "capacity_wait_seconds": 2.5,
                    },
                }]),
            )
            store.add_event(
                "gate",
                "motion",
                created_at="2026-08-02T12:02:00+00:00",
                objects_json=json.dumps([{
                    "status": "object_tracking",
                    "object_tracking": {"state": "skipped_capacity"},
                }]),
            )

            history = store.tracking_capacity_activity(
                hours=1,
                bucket_minutes=15,
                camera_id="gate",
                now=now,
            )
            active = [point for point in history if point["attempts"]]

            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["attempts"], 2)
            self.assertEqual(active[0]["waited"], 1)
            self.assertEqual(active[0]["skipped"], 1)
            self.assertEqual(active[0]["wait_seconds_average"], 2.5)

    def test_motion_effectiveness_separates_visual_filters_from_state_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            created_at = "2099-07-27T12:00:00+00:00"

            def qualification_objects(
                *,
                object_label: str = "",
                borderline: bool = False,
                verification_rescue: bool = False,
            ) -> str:
                objects: list[dict] = [{
                    "status": "motion_qualification",
                    "motion_qualification": {
                        "mode": "camera",
                        "accepted": not borderline,
                        "borderline_candidate": borderline,
                        "suppression_verification_candidate": verification_rescue,
                        "suppression_verification_rescued": verification_rescue,
                    },
                }]
                if object_label:
                    objects.append({"label": object_label, "incident_eligible": True})
                return json.dumps(objects)

            object_event = store.add_event(
                "gate", "motion", objects_json=qualification_objects(object_label="person"),
                created_at=created_at,
            )
            rescued_event = store.add_event(
                "gate", "motion", objects_json=qualification_objects(borderline=True),
                created_at=created_at,
            )
            store.add_event(
                "gate", "motion",
                objects_json=qualification_objects(
                    object_label="person",
                    verification_rescue=True,
                ),
                created_at=created_at,
            )
            base_audit = {
                "camera_id": "gate",
                "snapshot_path": "",
                "created_at": created_at,
                "mode": "camera",
                "sensitivity": "balanced",
                "score": 0.4,
                "threshold": 0.48,
                "object_detected": None,
                "trigger_count": 1,
                "features": {},
            }
            store.add_motion_audit(**base_audit, reason="micro_jitter")
            store.add_motion_audit(**base_audit, reason="event_state_active")
            store.add_motion_audit(
                **{**base_audit, "object_detected": False, "features": {"suppression_verification": True}},
                reason="stationary_foreground",
            )
            store.add_motion_audit(
                **{**base_audit, "object_detected": False},
                reason="micro_jitter",
                event_id=int(rescued_event["id"]),
            )

            summary = store.motion_effectiveness(days=7)["by_camera"]["gate"]["camera"]

            self.assertEqual(summary["allowed_events"], 3)
            self.assertEqual(summary["object_events"], 2)
            self.assertEqual(summary["no_object_events"], 1)
            self.assertEqual(summary["borderline_rescued"], 1)
            self.assertEqual(summary["suppression_verification_checks"], 2)
            self.assertEqual(summary["suppression_verification_rescues"], 1)
            self.assertEqual(summary["visual_filtered"], 2)
            self.assertEqual(summary["state_deduplicated"], 1)
            self.assertEqual(summary["unreviewed_visual_filters"], 1)
            self.assertEqual(summary["total_decisions"], 6)
            self.assertEqual(summary["visual_rejection_rate"], 0.4)
            self.assertEqual(summary["object_yield_rate"], 0.6667)

    def test_database_can_be_local_while_media_paths_remain_in_storage(self) -> None:
        with tempfile.TemporaryDirectory() as storage, tempfile.TemporaryDirectory() as database:
            snapshot = Path(storage) / "snapshots" / "gate" / "event.jpg"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"jpeg")
            store = EventStore(Path(storage), database_dir=Path(database))

            event = store.add_event("gate", "motion", snapshot_path=str(snapshot))
            reloaded = EventStore(Path(storage), database_dir=Path(database))
            persisted_snapshot = reloaded.get(int(event["id"]))["snapshot_path"]

            self.assertEqual(store.db_path, Path(database) / "survng.sqlite3")
            self.assertEqual(persisted_snapshot, "snapshots/gate/event.jpg")

    def test_migrates_events_and_motion_audits_to_portable_media_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "media"
            database = root / "database"
            snapshot = storage / "snapshots" / "gate" / "event.jpg"
            audit = storage / "motion_samples" / "gate" / "audit.jpg"
            recording = storage / "recordings" / "gate" / "segment.mp4"
            for path in (snapshot, audit, recording):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"media")
            store = EventStore(storage, database_dir=database)
            event = store.add_event("gate", "motion")
            motion = store.add_motion_audit(
                camera_id="gate",
                snapshot_path="",
                created_at="2026-07-30T12:00:00+00:00",
                mode="adaptive",
                sensitivity="balanced",
                score=0.7,
                threshold=0.5,
                reason="accepted",
                object_detected=True,
                trigger_count=1,
                features={},
            )
            with store._connect() as connection:
                connection.execute(
                    "update events set snapshot_path = ?, recording_path = ? where id = ?",
                    (
                        "/mnt/frigate/SurvNG/snapshots/gate/event.jpg",
                        "/mnt/frigate/SurvNG/recordings/gate/segment.mp4",
                        int(event["id"]),
                    ),
                )
                connection.execute(
                    "update motion_audits set snapshot_path = ? where id = ?",
                    (
                        "/mnt/frigate/SurvNG/motion_samples/gate/audit.jpg",
                        int(motion["id"]),
                    ),
                )
                connection.execute(
                    "delete from survng_metadata where key = 'portable_media_paths'"
                )

            migrated = EventStore(storage, database_dir=database)
            migrated_event = migrated.get(int(event["id"]))
            migrated_audit = migrated.get_motion_audit(int(motion["id"]))

            self.assertEqual(migrated_event["snapshot_path"], "snapshots/gate/event.jpg")
            self.assertEqual(migrated_event["recording_path"], "recordings/gate/segment.mp4")
            self.assertEqual(migrated_audit["snapshot_path"], "motion_samples/gate/audit.jpg")

    def test_recent_and_camera_range_queries_reject_unbounded_negative_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            for index in range(3):
                store.add_event(
                    camera_id="gate",
                    kind="motion",
                    created_at=f"2026-07-15T12:00:0{index}+00:00",
                )

            self.assertEqual(len(store.recent(-1)), 1)
            self.assertEqual(
                len(store.for_camera_range("gate", "2026", "2027", limit=-1)),
                1,
            )

    def test_camera_range_uses_half_open_time_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            included = store.add_event(
                camera_id="gate",
                kind="motion",
                created_at="2026-07-15T12:00:00+00:00",
            )
            store.add_event(
                camera_id="gate",
                kind="motion",
                created_at="2026-07-15T12:00:10+00:00",
            )

            rows = store.for_camera_range(
                "gate",
                "2026-07-15T12:00:00+00:00",
                "2026-07-15T12:00:10+00:00",
            )

            self.assertEqual([row["id"] for row in rows], [included["id"]])

    def test_terminal_motion_ai_review_cannot_be_reopened_by_stale_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            review = store.create_motion_ai_review("gate", 3)
            completed = store.update_motion_ai_review(
                int(review["id"]),
                status="completed",
                analyzed=3,
                result={"summary": "done"},
            )

            stale = store.update_motion_ai_review(
                int(review["id"]),
                status="completed",
                analyzed=1,
                result={"summary": "stale"},
            )

            self.assertEqual(stale["status"], "completed")
            self.assertEqual(stale["analyzed"], completed["analyzed"])
            self.assertEqual(stale["result"], {"summary": "done"})

    def test_parallel_store_instances_wait_for_writers_instead_of_losing_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = EventStore(Path(tmpdir))
            second = EventStore(Path(tmpdir))
            barrier = threading.Barrier(3)
            failures: list[Exception] = []

            def write(store: EventStore, prefix: str) -> None:
                try:
                    barrier.wait()
                    for index in range(50):
                        store.add_event(camera_id=prefix, kind="motion", message=str(index))
                except Exception as exc:
                    failures.append(exc)

            threads = [
                threading.Thread(target=write, args=(first, "one")),
                threading.Thread(target=write, args=(second, "two")),
            ]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=10)

            self.assertFalse(failures)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(len(first.recent(200)), 100)
            with sqlite3.connect(first.db_path) as connection:
                self.assertEqual(connection.execute("pragma journal_mode").fetchone()[0], "wal")

    def test_legacy_malformed_objects_do_not_break_audit_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            with sqlite3.connect(store.db_path) as connection:
                connection.execute(
                    """
                    insert into events (camera_id, kind, objects_json, created_at)
                    values (?, ?, ?, ?)
                    """,
                    (
                        "gate",
                        "motion",
                        json.dumps(["legacy", {"motion_qualification": "invalid"}]),
                        "2026-07-15T12:00:00+00:00",
                    ),
                )

            reloaded = EventStore(Path(tmpdir))
            rows, total = reloaded.motion_audits()

            self.assertEqual(rows, [])
            self.assertEqual(total, 0)

    def test_motion_audit_rejects_non_finite_numeric_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            base = {
                "camera_id": "gate",
                "snapshot_path": "",
                "created_at": "2026-07-15T12:00:00+00:00",
                "mode": "audit",
                "sensitivity": "balanced",
                "score": 0.5,
                "threshold": 0.4,
                "reason": "accepted",
                "object_detected": False,
                "trigger_count": 1,
                "features": {},
            }
            for field in ("score", "threshold"):
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "must be finite"):
                        store.add_motion_audit(**{**base, field: math.nan})
            with self.assertRaises(ValueError):
                store.add_motion_audit(**{**base, "features": {"noise": math.inf}})

    def test_motion_audit_rejects_decision_ids_that_would_collide_if_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "at most 128"):
                store.add_motion_audit(
                    camera_id="gate",
                    snapshot_path="",
                    created_at="2026-07-15T12:00:00+00:00",
                    mode="audit",
                    sensitivity="balanced",
                    score=0.5,
                    threshold=0.4,
                    reason="accepted",
                    object_detected=False,
                    trigger_count=1,
                    features={},
                    decision_id="x" * 129,
                )

    def test_completed_migrations_are_not_rescanned_on_every_store_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            EventStore(Path(tmpdir))
            with (
                patch.object(EventStore, "_backfill_motion_audits") as backfill,
                patch.object(EventStore, "_rebase_media_paths") as rebase,
            ):
                EventStore(Path(tmpdir))

            backfill.assert_not_called()
            rebase.assert_not_called()

    def test_compact_queries_keep_media_availability_and_support_keyset_paging(self) -> None:
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
                {"id", "camera_id", "kind", "snapshot_path", "recording_path", "objects_json", "created_at"},
            )
            self.assertTrue(first_page[0]["snapshot_path"])
            self.assertTrue(first_page[0]["recording_path"])

    def test_between_compact_returns_complete_range_without_large_payloads(self) -> None:
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
            self.assertTrue(rows[0]["snapshot_path"])

    def test_page_between_uses_stable_cursor_and_camera_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            first = store.add_event(
                camera_id="gate",
                kind="motion",
                snapshot_path="snapshots/gate/first.webp",
                created_at="2026-08-10T12:00:00+00:00",
            )
            second = store.add_event(
                camera_id="gate",
                kind="motion",
                snapshot_path="snapshots/gate/second.webp",
                created_at="2026-08-10T12:00:00+00:00",
            )
            store.add_event(
                camera_id="foyer",
                kind="motion",
                snapshot_path="snapshots/foyer/third.webp",
                created_at="2026-08-10T12:00:01+00:00",
            )

            page_one = store.page_between(
                "2026-08-10T11:59:00+00:00",
                "2026-08-10T12:01:00+00:00",
                limit=1,
                camera_ids=("gate",),
                require_snapshot=True,
            )
            page_two = store.page_between(
                "2026-08-10T11:59:00+00:00",
                "2026-08-10T12:01:00+00:00",
                limit=1,
                before_created_at=page_one[-1]["created_at"],
                before_id=int(page_one[-1]["id"]),
                camera_ids=("gate",),
                require_snapshot=True,
            )

            self.assertEqual([row["id"] for row in page_one], [second["id"]])
            self.assertEqual([row["id"] for row in page_two], [first["id"]])

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

    def test_replace_detected_objects_preserves_concurrent_tracking_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            event = store.add_event(
                camera_id="gate",
                kind="motion",
                objects_json=json.dumps([
                    {"label": "old", "confidence": 0.8},
                    {"status": "motion_qualification", "motion_qualification": {"score": 0.7}},
                    {"status": "object_tracking", "object_tracking": {"state": "active"}},
                ]),
            )

            updated = store.replace_detected_objects(
                int(event["id"]),
                json.dumps([{"label": "person", "confidence": 0.9}]),
            )

        self.assertIsNotNone(updated)
        objects = json.loads(str(updated["objects_json"]))
        self.assertEqual(objects[0]["label"], "person")
        self.assertEqual(
            [item["status"] for item in objects[1:]],
            ["motion_qualification", "object_tracking"],
        )

    def test_motion_audits_filter_and_backfill_rejected_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            event = store.add_event(
                camera_id="gate",
                kind="motion",
                snapshot_path=f"{tmpdir}/snapshots/gate/rejected.jpg",
                objects_json=json.dumps([
                    {"label": "car", "confidence": 0.82, "incident_eligible": True},
                    {
                        "status": "motion_qualification",
                        "motion_qualification": {
                            "mode": "audit",
                            "sensitivity": "balanced",
                            "score": 0.41,
                            "threshold": 0.48,
                            "reason": "edge_motion",
                            "trigger_count": 3,
                            "would_suppress": True,
                            "features": {"persistence": 0.75, "interior": 0.0},
                        },
                    },
                ], separators=(",", ":")),
                created_at="2026-07-16T12:00:00+00:00",
            )

            reloaded = EventStore(Path(tmpdir))
            object_rows, object_total = reloaded.motion_audits(outcome="object")
            clear_rows, clear_total = reloaded.motion_audits(outcome="clear")

            self.assertEqual(object_total, 1)
            self.assertEqual(clear_total, 0)
            self.assertEqual(object_rows[0]["event_id"], event["id"])
            self.assertEqual(object_rows[0]["reason"], "edge_motion")
            self.assertEqual(json.loads(object_rows[0]["features_json"])["persistence"], 0.75)
            self.assertEqual(clear_rows, [])

    def test_motion_audits_preserve_skipped_detector_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            audit = store.add_motion_audit(
                camera_id="front-door",
                snapshot_path="",
                created_at="2026-07-16T12:00:00+00:00",
                mode="enforce",
                sensitivity="balanced",
                score=0.22,
                threshold=0.48,
                reason="low_persistence",
                object_detected=None,
                trigger_count=2,
                features={
                    "persistence": 0.2,
                    "motion_tracks": [{
                        "id": 3,
                        "score": 0.71,
                        "persistence": 0.8,
                        "box": [0.1, 0.2, 0.3, 0.5],
                        "path": [[0.15, 0.25], [0.2, 0.3]],
                    }],
                },
            )

            rows, total = store.motion_audits(camera_id="front-door", outcome="not_run")

            self.assertEqual(total, 1)
            self.assertEqual(rows[0]["id"], audit["id"])
            self.assertIsNone(rows[0]["object_detected"])
            tracks = json.loads(rows[0]["features_json"])["motion_tracks"]
            self.assertEqual(tracks[0]["id"], 3)
            self.assertEqual(tracks[0]["path"][-1], [0.2, 0.3])

    def test_motion_audit_pipeline_configuration_is_deduplicated_and_hydrated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            configuration = [{"id": "threshold", "implementation": "adaptive"}]
            base = {
                "camera_id": "gate",
                "snapshot_path": "",
                "mode": "camera",
                "sensitivity": "balanced",
                "score": 0.2,
                "threshold": 0.5,
                "object_detected": None,
                "trigger_count": 1,
                "features": {
                    "pipeline_telemetry": {
                        "graphs": {
                            "qualification": {
                                "configuration": configuration,
                                "invocation_timings": {},
                            }
                        }
                    }
                },
            }
            first = store.add_motion_audit(
                **base,
                created_at="2026-08-10T12:00:00+00:00",
                reason="low_score",
                decision_id="one",
            )
            store.add_motion_audit(
                **base,
                created_at="2026-08-10T12:00:01+00:00",
                reason="low_persistence",
                decision_id="two",
            )

            with sqlite3.connect(store.db_path) as conn:
                config_count = conn.execute(
                    "select count(*) from motion_audit_pipeline_configs"
                ).fetchone()[0]
                raw_features = json.loads(conn.execute(
                    "select features_json from motion_audits where id = ?",
                    (first["id"],),
                ).fetchone()[0])
            self.assertEqual(config_count, 1)
            raw_graph = raw_features["pipeline_telemetry"]["graphs"]["qualification"]
            self.assertNotIn("configuration", raw_graph)
            self.assertIn("configuration_fingerprint", raw_graph)
            hydrated = store.get_motion_audit(int(first["id"]))
            hydrated_features = json.loads(hydrated["features_json"])
            self.assertEqual(
                hydrated_features["pipeline_telemetry"]["graphs"]["qualification"]["configuration"],
                configuration,
            )

    def test_incident_activity_audits_are_summarized_per_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            event = store.add_event(
                camera_id="foyer",
                kind="motion",
                created_at="2026-08-10T12:00:00+00:00",
            )
            base = {
                "camera_id": "foyer",
                "snapshot_path": "",
                "mode": "camera",
                "sensitivity": "balanced",
                "score": 0.1,
                "threshold": 0.5,
                "reason": "event_state_active",
                "object_detected": None,
                "trigger_count": 1,
                "features": {},
                "related_event_id": int(event["id"]),
            }
            first = store.add_motion_audit(
                **base,
                created_at="2026-08-10T12:00:01+00:00",
                decision_id="active-one",
            )
            second = store.add_motion_audit(
                **base,
                created_at="2026-08-10T12:00:02+00:00",
                decision_id="active-two",
            )

            rows, total = store.motion_audits(
                camera_id="foyer",
                include_incident_activity=True,
            )
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(total, 1)
            self.assertEqual(rows[0]["trigger_count"], 2)
            features = json.loads(rows[0]["features_json"])
            self.assertEqual(features["episode_observation_count"], 2)
            self.assertEqual(
                features["episode_last_observed_at"],
                "2026-08-10T12:00:02+00:00",
            )

    def test_incident_activity_coalescing_replaces_and_cleans_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshots = root / "snapshots" / "foyer"
            snapshots.mkdir(parents=True)
            old = snapshots / "old.webp"
            new = snapshots / "new.webp"
            old.write_bytes(b"old")
            new.write_bytes(b"new")
            store = EventStore(root)
            event = store.add_event(camera_id="foyer", kind="motion")
            base = {
                "camera_id": "foyer",
                "mode": "camera",
                "sensitivity": "balanced",
                "score": 0.1,
                "threshold": 0.5,
                "reason": "event_state_active",
                "object_detected": None,
                "trigger_count": 1,
                "features": {},
                "related_event_id": int(event["id"]),
            }
            store.add_motion_audit(
                **base,
                snapshot_path=str(old),
                created_at="2026-08-10T12:00:01+00:00",
                decision_id="active-one",
            )
            updated = store.add_motion_audit(
                **base,
                snapshot_path=str(new),
                created_at="2026-08-10T12:00:02+00:00",
                decision_id="active-two",
            )
            retained = store.add_motion_audit(
                **base,
                snapshot_path="",
                created_at="2026-08-10T12:00:03+00:00",
                decision_id="active-three",
            )

            self.assertFalse(old.exists())
            self.assertTrue(new.exists())
            self.assertEqual(updated["snapshot_path"], "snapshots/foyer/new.webp")
            self.assertEqual(retained["snapshot_path"], "snapshots/foyer/new.webp")

    def test_active_followup_audits_are_persisted_and_reported_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store.add_motion_audit(
                camera_id="gate",
                snapshot_path="",
                created_at=datetime.now(timezone.utc).isoformat(),
                mode="adaptive",
                sensitivity="balanced",
                score=0.8,
                threshold=0.48,
                reason="active_event_followup",
                object_detected=True,
                trigger_count=1,
                features={},
                category="active_followup",
            )

            rows, total = store.motion_audits(category="active_followup")
            summary = store.motion_effectiveness(days=1)["by_camera"]["gate"]["adaptive"]

            self.assertEqual(total, 1)
            self.assertEqual(rows[0]["category"], "active_followup")
            self.assertEqual(summary["active_followup_attempts"], 1)
            self.assertEqual(summary["active_followup_objects"], 1)
            self.assertEqual(summary["visual_filtered"], 0)
            self.assertEqual(summary["total_decisions"], 0)

    def test_motion_audits_filter_visual_backup_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            base = {
                "camera_id": "gate",
                "snapshot_path": "",
                "created_at": "2026-07-31T16:02:07+00:00",
                "mode": "camera_rescue",
                "sensitivity": "balanced",
                "score": 0.82,
                "threshold": 0.48,
                "object_detected": False,
                "trigger_count": 1,
                "features": {},
            }
            store.add_motion_audit(**base, reason="visual_backup_trigger", category="visual_backup")
            store.add_motion_audit(
                **{**base, "created_at": "2026-07-31T16:03:00+00:00"},
                reason="low_persistence",
            )

            backups, backup_total = store.motion_audits(category="visual_backup")
            qualification, qualification_total = store.motion_audits(category="qualification")

            self.assertEqual(backup_total, 1)
            self.assertEqual(backups[0]["reason"], "visual_backup_trigger")
            self.assertEqual(qualification_total, 1)
            self.assertEqual(qualification[0]["reason"], "low_persistence")

    def test_visual_backup_event_is_recovered_by_incremental_audit_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            event = store.add_event(
                camera_id="gate",
                kind="motion",
                created_at="2026-07-31T16:02:07+00:00",
                objects_json=json.dumps([
                    {"label": "car", "confidence": 0.88, "incident_eligible": True},
                    {
                        "status": "motion_qualification",
                        "motion_qualification": {
                            "mode": "camera_rescue",
                            "sensitivity": "balanced",
                            "score": 0.81,
                            "threshold": 0.48,
                            "reason": "qualified",
                            "trigger_count": 1,
                            "trigger_source": "visual_backup",
                            "would_suppress": False,
                            "features": {"persistence": 1.0},
                        },
                    },
                ]),
            )

            reloaded = EventStore(Path(tmpdir))
            rows, total = reloaded.motion_audits(category="visual_backup")

            self.assertEqual(total, 1)
            self.assertEqual(rows[0]["event_id"], event["id"])
            self.assertEqual(rows[0]["reason"], "visual_backup_trigger")
            self.assertTrue(rows[0]["object_detected"])
            features = json.loads(rows[0]["features_json"])
            self.assertEqual(features["visual_backup_original_reason"], "qualified")

            summary = reloaded.motion_effectiveness(days=90)["by_camera"]["gate"]["camera_rescue"]
            self.assertEqual(summary["visual_backup_attempts"], 1)
            self.assertEqual(summary["visual_backup_objects"], 1)
            self.assertEqual(summary["visual_filtered"], 0)

    def test_visual_backup_without_object_is_not_counted_as_visual_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store.add_motion_audit(
                camera_id="gate",
                snapshot_path="",
                created_at=datetime.now(timezone.utc).isoformat(),
                mode="camera_rescue",
                sensitivity="balanced",
                score=0.8,
                threshold=0.48,
                reason="visual_backup_trigger",
                object_detected=False,
                trigger_count=1,
                features={},
                category="visual_backup",
            )

            summary = store.motion_effectiveness(days=1)["by_camera"]["gate"]["camera_rescue"]

            self.assertEqual(summary["visual_backup_attempts"], 1)
            self.assertEqual(summary["visual_backup_no_object"], 1)
            self.assertEqual(summary["visual_filtered"], 0)
            self.assertEqual(summary["total_decisions"], 0)

    def test_visual_backup_scene_learning_hold_is_not_counted_as_detector_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store.add_motion_audit(
                camera_id="front-door",
                snapshot_path="",
                created_at=datetime.now(timezone.utc).isoformat(),
                mode="camera_rescue",
                sensitivity="balanced",
                score=0.8,
                threshold=0.48,
                reason="startup_not_ready",
                object_detected=None,
                trigger_count=0,
                features={"visual_backup_scene_ready": False},
                category="visual_backup",
            )

            summary = store.motion_effectiveness(days=1)["by_camera"]["front-door"]["camera_rescue"]

            self.assertEqual(summary["visual_backup_not_ready"], 1)
            self.assertEqual(summary["visual_backup_attempts"], 0)
            self.assertEqual(summary["visual_backup_incomplete"], 0)

    def test_visual_backup_below_threshold_is_not_counted_as_detector_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store.add_motion_audit(
                camera_id="back-left",
                snapshot_path="motion.webp",
                created_at=datetime.now(timezone.utc).isoformat(),
                mode="camera_rescue",
                sensitivity="balanced",
                score=0.697,
                threshold=0.48,
                reason="visual_backup_below_threshold",
                object_detected=None,
                trigger_count=0,
                features={"visual_backup_required_score": 0.73},
                category="visual_backup",
            )

            summary = store.motion_effectiveness(days=1)["by_camera"]["back-left"]["camera_rescue"]

            self.assertEqual(summary["visual_backup_below_threshold"], 1)
            self.assertEqual(summary["visual_backup_attempts"], 0)
            self.assertEqual(summary["visual_backup_incomplete"], 0)

    def test_suppressed_motion_audit_retry_is_idempotent_after_read_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            payload = {
                "decision_id": "decision-gate-1800",
                "camera_id": "gate",
                "snapshot_path": "",
                "created_at": "2026-07-26T18:00:00+00:00",
                "mode": "enforce",
                "sensitivity": "balanced",
                "score": 0.2,
                "threshold": 0.48,
                "reason": "low_score",
                "object_detected": None,
                "trigger_count": 1,
                "features": {"persistence": 0.2},
            }
            with patch.object(
                store,
                "get_motion_audit",
                side_effect=RuntimeError("post-commit read failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "post-commit read failed"):
                    store.add_motion_audit(**payload)

            retry = store.add_motion_audit(
                **{**payload, "snapshot_path": "snapshot-now-available.jpg"}
            )
            rows, total = store.motion_audits(camera_id="gate")

            self.assertEqual(total, 1)
            self.assertEqual(rows[0]["id"], retry["id"])
            self.assertEqual(rows[0]["decision_id"], "decision-gate-1800")
            self.assertEqual(rows[0]["snapshot_path"], "snapshot-now-available.jpg")

    def test_active_observation_links_to_incident_and_legacy_row_is_backfilled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            event = store.add_event(
                camera_id="foyer",
                kind="motion",
                created_at="2026-07-27T12:17:07+00:00",
                objects_json=json.dumps([{"label": "person", "confidence": 0.9}]),
            )
            audit = store.add_motion_audit(
                camera_id="foyer",
                snapshot_path="audit.jpg",
                created_at="2026-07-27T12:17:16+00:00",
                mode="camera",
                sensitivity="balanced",
                score=0.728,
                threshold=0.4817,
                reason="event_state_active",
                object_detected=None,
                trigger_count=1,
                features={},
            )
            self.assertIsNone(audit["related_event_id"])

            reloaded = EventStore(Path(tmpdir))
            linked = reloaded.get_motion_audit(int(audit["id"]))
            observations = reloaded.motion_audits_for_related_events([int(event["id"])])
            visible_audits, visible_total = reloaded.motion_audits(camera_id="foyer")
            all_audits, all_total = reloaded.motion_audits(
                camera_id="foyer",
                include_incident_activity=True,
            )

            self.assertEqual(linked["related_event_id"], event["id"])
            self.assertEqual([row["id"] for row in observations], [audit["id"]])
            self.assertEqual(visible_audits, [])
            self.assertEqual(visible_total, 0)
            self.assertEqual([row["id"] for row in all_audits], [audit["id"]])
            self.assertEqual(all_total, 1)

    def test_tracking_comparison_history_persists_verdict_and_compact_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            comparison = store.save_tracking_comparison(
                event_id=17,
                camera_id="foyer",
                event_created_at="2026-07-27T19:19:50+00:00",
                result={"frames_processed": 12, "engines": {"survng_hybrid": {"track_count": 2}}},
            )

            reviewed = store.set_tracking_comparison_verdict(
                comparison["id"],
                "survng_hybrid",
            )
            history = EventStore(Path(tmpdir)).tracking_comparison_history(camera_id="foyer")
            summary = store.tracking_comparison_summary(camera_id="foyer")

            self.assertEqual(reviewed["verdict"], "survng_hybrid")
            self.assertIsNotNone(reviewed["reviewed_at"])
            self.assertEqual(history[0]["result"]["frames_processed"], 12)
            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["reviewed"], 1)
            self.assertEqual(summary["verdicts"]["survng_hybrid"], 1)
            self.assertEqual(summary["verdicts"]["ultralytics_deepocsort"], 0)

    def test_tracking_comparison_rerun_resets_verdict_and_prunes_per_camera(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            store.TRACKING_COMPARISON_HISTORY_PER_CAMERA = 2
            first = store.save_tracking_comparison(
                event_id=1,
                camera_id="gate",
                event_created_at="one",
                result={"frames_processed": 1},
            )
            store.set_tracking_comparison_verdict(first["id"], "inconclusive")
            rerun = store.save_tracking_comparison(
                event_id=1,
                camera_id="gate",
                event_created_at="one",
                result={"frames_processed": 2},
            )
            store.save_tracking_comparison(event_id=2, camera_id="gate", event_created_at="two", result={})
            store.save_tracking_comparison(event_id=3, camera_id="gate", event_created_at="three", result={})

            history = store.tracking_comparison_history(camera_id="gate")

            self.assertEqual(rerun["verdict"], "")
            self.assertIsNone(rerun["reviewed_at"])
            self.assertEqual([row["event_id"] for row in history], [3, 2])

            with self.assertRaisesRegex(ValueError, "invalid tracking comparison verdict"):
                store.set_tracking_comparison_verdict(history[0]["id"], "automatic")

    def test_tracking_comparison_accepts_deep_ocsort_and_historic_botsort_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            current = store.save_tracking_comparison(
                event_id=1,
                camera_id="gate",
                event_created_at="current",
                result={"engines": {"ultralytics_deepocsort": {}}},
            )
            historic = store.save_tracking_comparison(
                event_id=2,
                camera_id="gate",
                event_created_at="historic",
                result={"engines": {"ultralytics_botsort": {}}},
            )

            store.set_tracking_comparison_verdict(current["id"], "ultralytics_deepocsort")
            store.set_tracking_comparison_verdict(historic["id"], "ultralytics_botsort")
            summary = store.tracking_comparison_summary(camera_id="gate")

            self.assertEqual(summary["verdicts"]["ultralytics_deepocsort"], 1)
            self.assertEqual(summary["verdicts"]["ultralytics_botsort"], 1)


if __name__ == "__main__":
    unittest.main()

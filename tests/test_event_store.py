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

from survng.app.events import EventStore


class EventStoreTest(unittest.TestCase):
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

    def test_process_lifecycle_events_are_durable_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = EventStore(root)
            now = datetime(2026, 8, 7, 12, tzinfo=timezone.utc)
            store.record_lifecycle_event(
                "stale-instance",
                "startup_started",
                occurred_at=now - timedelta(days=9),
            )
            store.record_lifecycle_event(
                "instance-a",
                "startup_started",
                occurred_at=now - timedelta(minutes=2),
            )
            store.record_lifecycle_event(
                "instance-a",
                "startup_ready",
                occurred_at=now - timedelta(minutes=1),
                details={"cameras": 13},
            )

            events = EventStore(root).lifecycle_events(hours=1, now=now)
            with sqlite3.connect(store.db_path) as conn:
                stored_count = conn.execute(
                    "select count(*) from system_lifecycle_events"
                ).fetchone()[0]

        self.assertEqual([event["kind"] for event in events], [
            "startup_started",
            "startup_ready",
        ])
        self.assertEqual(events[-1]["details"], {"cameras": 13})
        self.assertEqual(stored_count, 2)

    def test_process_lifecycle_rejects_unknown_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "unsupported lifecycle event"):
                store.record_lifecycle_event("instance-a", "restarting")

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

    def test_runtime_telemetry_persists_fps_and_counter_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            first_at = datetime.fromisoformat("2026-08-02T12:00:00+00:00")

            def status(
                read_failures: int,
                analysis_drops: int,
                fps: float,
                main_starts: int,
                frames_sampled: int,
            ) -> list[dict]:
                return [{
                    "id": "gate",
                    "connected": True,
                    "last_frame_age_seconds": 0.1,
                    "main_last_frame_age_seconds": 0.2,
                    "lifecycle": {"enabled": True},
                    "capture_stats": {
                        "live": {"fps": fps, "read_failures": read_failures, "open_failures": 0, "observer_p99_ms": float(read_failures)},
                        "main": {"fps": fps / 2, "read_failures": 0, "open_failures": 0, "starts": main_starts},
                    },
                    "motion_qualification": {
                        "analysis_frames_dropped": analysis_drops,
                        "analysis_runtime": {
                            "capture_to_analysis_p95_ms": float(analysis_drops),
                            "preprocess_p99_ms": float(analysis_drops) / 2,
                            "frames_sampled": frames_sampled,
                            "copy_bytes": analysis_drops * 100,
                        },
                        "event_runtime": {
                            "evicted": analysis_drops,
                            "rejected": analysis_drops // 2,
                            "retries_dropped": analysis_drops // 3,
                        },
                        "active_followup_candidates": frames_sampled // 10,
                        "active_followup_triggers": frames_sampled // 20,
                        "active_followup_objects": frames_sampled // 50,
                        "active_followup_no_object": frames_sampled // 25,
                    },
                    "object_tracking": {"active": False},
                }]

            store.record_runtime_telemetry(
                status(2, 3, 10.0, 4, 100),
                sampled_at=first_at,
                system_runtime={
                    "cpu_load_percent": 20.0,
                    "memory_used_percent": 60.0,
                    "inference_ms": 24.0,
                },
            )
            store.record_runtime_telemetry(
                status(4, 8, 12.0, 7, 200),
                sampled_at=first_at + timedelta(minutes=1),
                system_runtime={
                    "cpu_load_percent": 40.0,
                    "memory_used_percent": 70.0,
                    "inference_ms": 28.0,
                },
            )
            history = store.runtime_telemetry_history(
                hours=2,
                bucket_minutes=1,
                camera_id="gate",
                now=first_at + timedelta(minutes=2),
            )

            self.assertEqual(len(history), 2)
            self.assertEqual(history[-1]["live_fps"], 12.0)
            self.assertEqual(history[-1]["main_fps"], 6.0)
            self.assertEqual(history[-1]["capture_read_failures"], 2)
            self.assertEqual(history[-1]["capture_interruptions"], 2)
            self.assertEqual(history[-1]["main_capture_starts"], 3)
            self.assertEqual(history[-1]["analysis_frames_dropped"], 5)
            self.assertEqual(history[-1]["analysis_frames_sampled"], 100)
            self.assertEqual(history[-1]["active_followup_candidates"], 10)
            self.assertEqual(history[-1]["active_followup_triggers"], 5)
            self.assertEqual(history[-1]["active_followup_objects"], 2)
            self.assertEqual(history[-1]["active_followup_no_object"], 4)
            self.assertEqual(history[-1]["analysis_coverage_percent"], 95.238)
            self.assertEqual(history[-1]["camera_availability_percent"], 100.0)
            self.assertEqual(history[-1]["unavailable_cameras"], 0)
            self.assertEqual(history[-1]["capture_observer_p99_ms"], 4.0)
            self.assertEqual(history[-1]["capture_to_analysis_p95_ms"], 8.0)
            self.assertEqual(history[-1]["preprocess_p99_ms"], 4.0)
            self.assertEqual(history[-1]["motion_copy_bytes"], 500)
            self.assertEqual(history[-1]["event_evictions"], 5)
            self.assertEqual(history[-1]["event_rejections"], 3)
            self.assertEqual(history[-1]["event_retry_drops"], 1)
            self.assertEqual(history[-1]["event_delivery_failures"], 9)
            self.assertEqual(history[-1]["cpu_load_percent"], 40.0)
            self.assertEqual(history[-1]["memory_used_percent"], 70.0)
            self.assertEqual(history[-1]["inference_ms"], 28.0)

    def test_runtime_telemetry_availability_excludes_intentionally_disabled_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            sampled_at = datetime.fromisoformat("2026-08-02T12:00:00+00:00")
            store.record_runtime_telemetry(
                [
                    {
                        "id": "gate",
                        "connected": True,
                        "last_frame_age_seconds": 0.2,
                        "lifecycle": {"enabled": True},
                    },
                    {
                        "id": "garage",
                        "connected": False,
                        "last_frame_age_seconds": 30.0,
                        "expected_enabled": True,
                        "lifecycle": {"enabled": False},
                    },
                    {
                        "id": "foyer",
                        "connected": False,
                        "last_frame_age_seconds": None,
                        "expected_enabled": False,
                        "lifecycle": {"enabled": True},
                    },
                ],
                sampled_at=sampled_at,
            )

            history = store.runtime_telemetry_history(
                hours=1,
                bucket_minutes=1,
                now=sampled_at + timedelta(minutes=1),
            )

            self.assertEqual(history[0]["camera_availability_percent"], 50.0)
            self.assertEqual(history[0]["expected_cameras"], 2)
            self.assertEqual(history[0]["unavailable_cameras"], 1)
            paused_history = store.runtime_telemetry_history(
                hours=1,
                bucket_minutes=1,
                camera_id="foyer",
                now=sampled_at + timedelta(minutes=1),
            )
            self.assertIsNone(paused_history[0]["camera_availability_percent"])
            self.assertIsNone(paused_history[0]["analysis_coverage_percent"])

    def test_runtime_telemetry_persists_process_memory_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            sampled_at = datetime.fromisoformat("2026-08-02T12:00:00+00:00")
            store.record_runtime_telemetry(
                [],
                sampled_at=sampled_at,
                process_memory={
                    "rss_bytes": 100,
                    "anonymous_rss_bytes": 80,
                    "pss_bytes": 90,
                    "private_dirty_bytes": 70,
                    "anonymous_huge_pages_bytes": 20,
                    "threads": 12,
                    "file_descriptors": 34,
                    "malloc": {
                        "allocated_bytes": 50,
                        "free_bytes": 10,
                        "mmap_bytes": 30,
                    },
                },
                worker_memory={"total_rss_bytes": 60, "total_pss_bytes": 55},
                memory_maintenance={
                    "successful_trims": 2,
                    "reclaimed_total_bytes": 400,
                },
            )

            history = store.process_memory_history(
                hours=2,
                bucket_minutes=1,
                now=sampled_at + timedelta(minutes=1),
            )

            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["rss_bytes"], 100)
            self.assertEqual(history[0]["malloc_allocated_bytes"], 50)
            self.assertEqual(history[0]["malloc_free_bytes"], 10)
            self.assertEqual(history[0]["worker_rss_bytes"], 60)
            self.assertEqual(history[0]["allocator_trim_count"], 2)
            self.assertEqual(history[0]["allocator_trim_reclaimed_bytes"], 400)
            self.assertEqual(history[0]["threads"], 12)
            self.assertEqual(history[0]["file_descriptors"], 34)

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

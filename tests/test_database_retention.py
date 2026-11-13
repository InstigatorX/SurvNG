"""Database retention removes only unreferenced working data."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from survng.app.config import DatabaseRetentionConfig
from survng.app.database_retention import DatabaseRetention
from survng.app.events import EventStore

NOW = 1_800_000_000.0
HOUR = 3600.0
DAY = 86400.0


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


TABLES = (
    "acquired_samples", "acquired_observations", "scene_activity_decisions",
    "scene_activity_measurements", "scene_candidate_jobs", "motion_audits",
)


def _ids(store: EventStore, table: str) -> set:
    with sqlite3.connect(store.db_path) as db:
        ids = {row[0] for row in db.execute(f"select id from {table}")}
    baseline = getattr(store, "test_baseline", {}).get(table, set())
    pruned = baseline - ids
    assert not pruned, f"store-created {table} rows were pruned: {pruned}"
    return ids - baseline


@pytest.fixture()
def store(tmp_path: Path) -> EventStore:
    store = EventStore(tmp_path)
    event_id = store.add_event("gate", "motion", snapshot_path="snapshots/gate/event.jpg")["id"]
    established_id = store.add_event("gate", "motion")["id"]
    # Rows the store itself created for those events are event-linked and
    # must survive; comparisons below cover only the fixture rows.
    store.test_baseline = {table: _ids(store, table) for table in TABLES}
    (tmp_path / "motion_samples" / "gate").mkdir(parents=True)
    for name in ("old.jpg", "shared.jpg", "linked.jpg", "recent.jpg"):
        (tmp_path / "motion_samples" / "gate" / name).write_bytes(b"jpg")

    with sqlite3.connect(store.db_path) as db:  # fixture rows; FK checks off
        def sample(sample_id, age, *, source="live_discovery", observations=()):
            db.execute(
                "insert into acquired_samples values (?,?,?,?,?,?,?,?,?)",
                (sample_id, "gate", NOW - age, source, "complete", "{}", "", "", NOW - age),
            )
            for observation_id in observations:
                db.execute(
                    "insert or ignore into acquired_observations values (?,?,?,?,?,?)",
                    (observation_id, "gate", NOW - age, "{}", "", ""),
                )
                db.execute(
                    "insert into acquired_sample_observations values (?,?)",
                    (sample_id, observation_id),
                )

        def decision(decision_id, verdict, age, samples):
            db.execute(
                "insert into scene_activity_decisions values (?,?,?,?,?,?,?,?,?)",
                (decision_id, "gate", json.dumps(samples), verdict,
                 NOW - age if verdict == "supported" else None, "r", "1", "{}", NOW - age),
            )

        # Empty live-discovery frames.
        sample("empty-old", 2 * DAY)
        decision("pending-old", "pending", 2 * DAY, ["empty-old"])
        db.execute(
            "insert into scene_activity_measurements values (?,?,?,?)",
            ("measure-old", "empty-old", "{}", NOW - 2 * DAY),
        )
        sample("empty-recent", 2 * HOUR)
        decision("pending-recent", "pending", 2 * HOUR, ["empty-recent"])
        sample("empty-event", 2 * DAY)
        db.execute("insert into acquired_sample_events values (?,?)", ("empty-event", event_id))
        decision("pending-established", "pending", 2 * DAY, ["empty-event"])
        db.execute(
            "insert or replace into scene_event_establishment values (?,?)",
            (established_id, "pending-established"),
        )
        # Frames with detections follow the activity schedule.
        sample("seen-10d", 10 * DAY, observations=["obs-10d"])
        decision("unsupported-10d", "unsupported", 10 * DAY, ["seen-10d"])
        sample("seen-40d", 40 * DAY, observations=["obs-40d"])
        decision("unsupported-40d", "unsupported", 40 * DAY, ["seen-40d"])
        sample("supported-40d", 40 * DAY, observations=["obs-supported"])
        decision("supported-40d", "supported", 40 * DAY, ["supported-40d"])
        # A finished, unadmitted candidate job and its seed sample.
        sample("seed-40d", 40 * DAY, source="recorded", observations=["obs-seed"])
        db.execute(
            "insert into scene_candidate_jobs (id,camera_id,start_epoch,end_epoch,"
            "seed_sample_ids_json,deadline_epoch,available_at_epoch,state,created_at,updated_at) "
            "values (?,?,?,?,?,?,?,?,?,?)",
            ("job-40d", "gate", NOW - 40 * DAY, NOW - 40 * DAY, '["seed-40d"]',
             NOW - 40 * DAY, NOW - 40 * DAY, "complete", NOW - 40 * DAY, NOW - 40 * DAY),
        )
        db.execute("insert into scene_candidate_seeds values (?,?)", ("seed-40d", "job-40d"))
        db.execute(
            "insert into scene_candidate_jobs (id,camera_id,start_epoch,end_epoch,"
            "seed_sample_ids_json,deadline_epoch,available_at_epoch,state,created_at,updated_at) "
            "values (?,?,?,?,?,?,?,?,?,?)",
            ("job-open", "gate", NOW - 40 * DAY, NOW - 40 * DAY, "[]",
             NOW + HOUR, NOW, "pending", NOW - 40 * DAY, NOW - 40 * DAY),
        )
        # Observations: one backs an event's scene observation, one is orphaned.
        for observation_id in ("obs-retained", "obs-orphan"):
            db.execute(
                "insert into acquired_observations values (?,?,?,?,?,?)",
                (observation_id, "gate", NOW - 40 * DAY, "{}", "", ""),
            )
        db.execute(
            "insert into scene_observations (id,event_id,episode_id,object_id,camera_id,"
            "captured_epoch,track_key,object_index,payload_json,snapshot_path,recording_path,"
            "source_observation_id) values (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("scene-1", event_id, "episode", "object", "gate", NOW - 40 * DAY, "t", 0, "{}",
             "", "", "obs-retained"),
        )

        def audit(audit_id, age, path, event=None):
            db.execute(
                "insert into motion_audits (id,event_id,camera_id,snapshot_path,created_at,mode,"
                "sensitivity,score,threshold,reason,object_detected,trigger_count,features_json,"
                "decision_id,related_event_id,category) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (audit_id, event, "gate", path, _iso(NOW - age), "m", 1.0, 0.1, 0.2, "clear",
                 0, 1, "{}", "", None, "visual_backup"),
            )

        audit(1, 40 * DAY, "motion_samples/gate/old.jpg")
        audit(2, 40 * DAY, "motion_samples/gate/shared.jpg")
        audit(3, 5 * DAY, "motion_samples/gate/shared.jpg")
        audit(4, 40 * DAY, "motion_samples/gate/linked.jpg", event=event_id)
        audit(5, 5 * DAY, "motion_samples/gate/recent.jpg")
    return store




def _retention(store: EventStore, **overrides) -> DatabaseRetention:
    config = DatabaseRetentionConfig(**{"dry_run": False, **overrides})
    return DatabaseRetention(store, lambda: config, clock=lambda: NOW, pause_seconds=0)


def test_dry_run_counts_without_deleting(store: EventStore) -> None:
    before = {table: _ids(store, table) for table in (
        "acquired_samples", "scene_activity_decisions", "motion_audits", "scene_candidate_jobs",
    )}

    result = _retention(store, dry_run=True).run_once()

    assert result["dry_run"] is True
    assert result["steps"]["activity_decisions"]["eligible"] == 2
    assert result["steps"]["motion_audits"]["eligible"] == 2
    assert all(step["deleted"] == 0 for step in result["steps"].values())
    assert {table: _ids(store, table) for table in before} == before
    assert (store.storage_dir / "motion_samples" / "gate" / "old.jpg").exists()


def test_prunes_only_unreferenced_rows_in_dependency_order(store: EventStore) -> None:
    result = _retention(store).run_once()

    assert _ids(store, "scene_activity_decisions") == {
        "pending-recent", "pending-established", "unsupported-10d", "supported-40d",
    }
    assert _ids(store, "acquired_samples") == {
        "empty-recent", "empty-event", "seen-10d", "supported-40d",
    }
    assert _ids(store, "scene_candidate_jobs") == {"job-open"}
    assert _ids(store, "acquired_observations") == {"obs-10d", "obs-supported", "obs-retained"}
    assert _ids(store, "scene_activity_measurements") == set()
    assert _ids(store, "motion_audits") == {3, 4, 5}
    assert result["steps"]["acquired_samples"]["deleted"] == 3
    assert not any(step.get("skipped_batches") for step in result["steps"].values())

    samples = store.storage_dir / "motion_samples" / "gate"
    assert not (samples / "old.jpg").exists()
    assert (samples / "shared.jpg").exists()  # still named by a retained audit
    assert (samples / "linked.jpg").exists()
    assert result["motion_sample_files_deleted"] == 1

    # Nothing left to do: a second run is a no-op.
    again = _retention(store).run_once()
    assert all(step["deleted"] == 0 for step in again["steps"].values())


def test_status_reports_policy_and_last_run(store: EventStore) -> None:
    retention = _retention(store, dry_run=True)
    assert retention.status()["last_run"] is None

    retention.run_once()
    status = retention.status()

    assert status["dry_run"] is True
    assert status["runs"] == 1
    assert status["policy"]["motion_audit_days"] == 30
    assert status["last_run"]["steps"]["motion_audits"]["eligible"] == 2


def test_stop_interrupts_between_batches(store: EventStore) -> None:
    retention = _retention(store)
    retention._stop.set()

    result = retention.run_once()

    assert result["interrupted"] is True
    assert "pending-old" in _ids(store, "scene_activity_decisions")


@pytest.mark.parametrize(("dry_run", "expected"), ((True, 90.0), (False, 30.0)))
def test_motion_effectiveness_range_follows_audit_retention(dry_run: bool, expected: float) -> None:
    import threading
    from types import SimpleNamespace
    from unittest.mock import Mock

    from survng.app.config import AppConfig
    from survng.app.intelligence_routes import IntelligenceService

    config = AppConfig(database_retention=DatabaseRetentionConfig(dry_run=dry_run))
    events = SimpleNamespace(motion_effectiveness=Mock(return_value={}))
    manager = SimpleNamespace(events=events, config=config)
    service = IntelligenceService(SimpleNamespace(
        manager_lock=threading.RLock(), get_manager=lambda: manager,
    ))

    service.motion_effectiveness(days=90.0)

    events.motion_effectiveness.assert_called_once_with(days=expected)


def test_reschedule_wakes_the_loop_to_run_after_the_startup_delay(store: EventStore) -> None:
    import threading

    config = DatabaseRetentionConfig(dry_run=True, interval_hours=24.0)
    retention = DatabaseRetention(
        store, lambda: config, pause_seconds=0, startup_delay_seconds=0.2,
    )
    ran = threading.Event()
    retention.run_once = lambda **_: ran.set()  # type: ignore[method-assign]
    retention.start()
    try:
        assert ran.wait(5.0)  # startup run
        ran.clear()
        assert not ran.wait(0.5)  # next run is a day away
        retention.reschedule()
        assert ran.wait(5.0)  # policy change: runs after the short delay
    finally:
        retention.stop()
    assert retention._thread is None


@pytest.mark.parametrize("parent", (
    "scene_candidate_jobs", "scene_activity_decisions", "acquired_samples", "acquired_observations",
))
def test_foreign_keys_into_pruned_tables_are_indexed(tmp_path: Path, parent: str) -> None:
    # Each pruned parent row makes SQLite check every child foreign key; an
    # unindexed child column turns each delete into a full scan of that table.
    store = EventStore(tmp_path)
    unindexed = []
    with sqlite3.connect(store.db_path) as db:
        tables = [row[0] for row in db.execute("select name from sqlite_master where type='table'")]
        for table in tables:
            for fk in db.execute(f"pragma foreign_key_list('{table}')").fetchall():
                if fk[2] != parent:
                    continue
                leading = {
                    db.execute(f"pragma index_info('{index[1]}')").fetchone()[2]
                    for index in db.execute(f"pragma index_list('{table}')").fetchall()
                }
                if fk[3] not in leading:
                    unindexed.append(f"{table}.{fk[3]}")
    assert not unindexed, f"unindexed foreign keys into {parent}: {unindexed}"


def test_audit_snapshot_images_are_removed_only_when_nothing_names_them(tmp_path: Path) -> None:
    # Most audits keep their image in the snapshot directory, often shared
    # with an event or incident under another spelling of the same path.
    store = EventStore(tmp_path)
    snapshots = tmp_path / "snapshots" / "gate"
    snapshots.mkdir(parents=True)
    for name in ("orphan.jpg", "event.jpg", "incident.jpg"):
        (snapshots / name).write_bytes(b"jpg")
    store.add_event("gate", "motion", snapshot_path=str(snapshots / "event.jpg"))
    with sqlite3.connect(store.db_path) as db:
        db.execute("create table if not exists incident_observations (snapshot_path text)")
        db.execute(
            "insert into incident_observations values (?)",
            (str(snapshots / "incident.jpg"),),
        )
        for audit_id, name in enumerate(("orphan.jpg", "event.jpg", "incident.jpg"), start=101):
            db.execute(
                "insert into motion_audits (id,event_id,camera_id,snapshot_path,created_at,mode,"
                "sensitivity,score,threshold,reason,object_detected,trigger_count,features_json,"
                "decision_id,related_event_id,category) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (audit_id, None, "gate", f"snapshots/gate/{name}", _iso(NOW - 40 * DAY), "m",
                 1.0, 0.1, 0.2, "clear", 0, 1, "{}", "", None, "visual_backup"),
            )

    result = _retention(store).run_once()

    assert result["steps"]["motion_audits"]["deleted"] == 3
    assert not (snapshots / "orphan.jpg").exists()
    assert (snapshots / "event.jpg").exists()  # event names it by absolute path
    assert (snapshots / "incident.jpg").exists()  # incident observation names it
    assert result["motion_sample_files_deleted"] == 1
    assert result["motion_sample_files_failed"] == 0
    with sqlite3.connect(store.db_path) as db:
        assert db.execute(
            "select count(*) from media_deletion_work where role = 'motion_audit'"
        ).fetchone()[0] == 0
        assert db.execute(
            "select count(*) from media_deletion_claims where role = 'motion_audit'"
        ).fetchone()[0] == 0


def test_restart_finishes_motion_deletion_work_recorded_before_unlink(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    image = tmp_path / "snapshots" / "gate" / "left.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpg")
    raw = "snapshots/gate/left.jpg"
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            "create table if not exists media_deletion_work ("
            "path text primary key, role text not null, created_at text not null)"
        )
        db.execute(
            "insert into media_deletion_work values (?, 'motion_audit', ?)",
            (raw, _iso(NOW)),
        )
        db.execute(
            "insert into media_deletion_claims (path, role, claimed_at) values (?, 'motion_audit', ?)",
            (raw, _iso(NOW)),
        )
    result = _retention(store).run_once()
    assert not image.exists()
    assert result["motion_sample_files_deleted"] == 1
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("select count(*) from media_deletion_claims where path = ?", (raw,)).fetchone()[0] == 0


def test_sample_kept_when_a_decision_survives_the_reference_snapshot(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event_id = store.add_event("gate", "motion")["id"]
    with sqlite3.connect(store.db_path) as db:
        db.execute("pragma foreign_keys=off")
        db.execute(
            "insert into acquired_samples values (?,?,?,?,?,?,?,?,?)",
            ("late-sample", "gate", NOW - 40 * DAY, "recorded", "complete", "{}", "", "", NOW - 40 * DAY),
        )
        db.execute(
            "insert into scene_activity_decisions values (?,?,?,?,?,?,?,?,?)",
            ("late-decision", "gate", '["late-sample"]', "unsupported",
             None, "r", "1", "{}", NOW - 40 * DAY),
        )
        db.execute(
            "insert or replace into scene_event_establishment values (?,?)",
            (event_id, "late-decision"),
        )
    retention = _retention(store)
    retention._referenced_decisions = lambda: set()
    retention.run_once()
    assert "late-sample" in _ids(store, "acquired_samples")
    assert "late-decision" in _ids(store, "scene_activity_decisions")


def test_retention_indexes_legacy_references_before_pruning(store: EventStore) -> None:
    with store._connect() as conn:
        conn.execute("delete from scene_decision_samples")
        conn.execute("update scene_acquisition_migrations set cursor=0 where name='decision_sample_refs_v1'")
    _retention(store).run_once()
    assert "supported-40d" in _ids(store, "acquired_samples")
    assert "empty-old" not in _ids(store, "acquired_samples")
    with store._connect() as conn:
        assert conn.execute("select cursor from scene_acquisition_migrations where name='decision_sample_refs_v1'").fetchone()[0] == -1
        plan = [row[3] for row in conn.execute(
            "explain query plan select s.id from acquired_samples s "
            "where not exists(select 1 from scene_decision_samples d where d.sample_id=s.id)",
        )]
    assert any("SEARCH d USING COVERING INDEX" in step for step in plan)
    assert not any(step.startswith("SCAN d") or "json_each" in step for step in plan)


def test_interrupted_reference_backfill_does_not_prune_samples(store: EventStore, monkeypatch) -> None:
    retention = _retention(store)
    before = _ids(store, "acquired_samples")
    with store._connect() as conn:
        conn.execute("delete from scene_decision_samples")
        conn.execute("update scene_acquisition_migrations set cursor=0 where name='decision_sample_refs_v1'")
    backfill = store.backfill_scene_decision_samples

    def stop_after_one_page():
        result = backfill(batch_size=1)
        retention._stop.set()
        return result

    monkeypatch.setattr(store, "backfill_scene_decision_samples", stop_after_one_page)
    result = retention.run_once()
    assert result["interrupted"] is True
    assert _ids(store, "acquired_samples") == before

import sqlite3
import threading
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from survng.app.database_polling import database_polling_session, polling_connection
from survng.app.event_store import EventStore


def test_poll_reads_reuse_connections_see_commits_and_close_on_error(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    with sqlite3.connect(path) as writer:
        writer.execute("pragma journal_mode=wal")
        writer.execute("create table jobs(value integer)")
        with pytest.raises(RuntimeError, match="worker failed"):
            with database_polling_session():
                with polling_connection(path) as first:
                    assert first.execute("select count(*) from jobs").fetchone()[0] == 0
                    assert not first.in_transaction
                    with pytest.raises(sqlite3.OperationalError, match="readonly"):
                        first.execute("insert into jobs values(1)")
                writer.execute("insert into jobs values(2)")
                writer.commit()
                with polling_connection(path) as second:
                    assert first is second
                    assert second.execute("select value from jobs").fetchone()[0] == 2
                assert writer.execute("pragma wal_checkpoint(truncate)").fetchone()[0] == 0
                raise RuntimeError("worker failed")
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            first.execute("select 1")
        with database_polling_session(), polling_connection(path) as recovered:
            assert recovered is not first
            assert recovered.execute("select value from jobs").fetchone()[0] == 2


def test_poll_connections_are_thread_scoped_and_standalone_reads_close(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    sqlite3.connect(path).close()
    other = []
    with database_polling_session(), polling_connection(path) as first:
        def worker():
            with database_polling_session(), polling_connection(path) as connection:
                other.append(connection)
                connection.execute("select 1").fetchone()
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(2)
        assert not thread.is_alive()
        assert len(other) == 1 and other[0] is not first
        assert first.execute("select 1").fetchone()[0] == 1
    with polling_connection(path) as standalone:
        assert standalone.execute("select 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        standalone.execute("select 1")


def test_missing_poll_ledger_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        with database_polling_session(), polling_connection(path):
            pass
    assert not path.exists()


def test_idle_claims_never_open_writer_connections(tmp_path):
    store = EventStore(tmp_path)
    with database_polling_session(), \
            patch.object(store, "_connect", side_effect=AssertionError("idle writer")), \
            patch.object(store, "_connect_jobs", side_effect=AssertionError("idle jobs writer")):
        for _ in range(3):
            assert store.claim_detection_job("gate", lease_owner="worker") is None
            assert store.claim_scene_candidate("gate", lease_owner="worker") is None
            assert store.claim_cover_requirement("gate", lease_owner="worker") is None
            assert store.claim_scene_tracking("gate", "worker") is None


def test_reused_scene_probe_observes_arrivals_retry_time_and_expired_leases(tmp_path):
    store = EventStore(tmp_path)
    event = store.add_event(
        camera_id="gate", kind="motion",
        created_at=datetime.fromtimestamp(1000, timezone.utc).isoformat(),
        objects_json='[{"label":"person","confidence":0.9,"box":{"x1":10,"y1":10,"x2":30,"y2":70},"detection_frame_width":100,"detection_frame_height":100}]',
    )
    with database_polling_session(), patch("survng.app.event_store.scene_tracking.time.time", return_value=2000):
        assert store.claim_scene_tracking("gate", "worker") is None
        queued = store.enqueue_scene_tracking(event["id"], 1000, 1010)
        assert queued is not None
        with store._connect() as conn:
            conn.execute("update scene_analysis_jobs set retry_at=2001")
        assert store.claim_scene_tracking("gate", "worker") is None
        with patch("survng.app.event_store.scene_tracking.time.time", return_value=2002):
            claimed = store.claim_scene_tracking("gate", "worker")
            assert claimed["episode_id"] == queued["episode_id"]
            assert store.claim_scene_tracking("gate", "competitor") is None
        with patch("survng.app.event_store.scene_tracking.time.time", return_value=2100):
            assert store.claim_scene_tracking("gate", "recovery")["lease_owner"] == "recovery"


def test_reused_detection_probe_observes_enqueue(tmp_path):
    store = EventStore(tmp_path)
    with database_polling_session():
        assert store.claim_detection_job("gate", lease_owner="worker") is None
        store.enqueue_detection_job(job_id="new", camera_id="gate", dedupe_key="new", payload={})
        assert store.claim_detection_job("gate", lease_owner="worker")["id"] == "new"

import sqlite3
import threading
import time
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


def test_camera_work_deadline_covers_candidate_and_cover_predicates(tmp_path):
    store = EventStore(tmp_path)
    now = time.time()
    with store._connect() as conn:
        conn.execute(
            "insert into scene_candidate_jobs "
            "(id,camera_id,start_epoch,end_epoch,seed_sample_ids_json,deadline_epoch,"
            "available_at_epoch,state,created_at,updated_at) "
            "values('candidate','gate',?,?, '[]',?,?,'pending',?,?)",
            (now, now + 1, now + 3, now + 2, now, now),
        )
    delay = store.next_camera_work_delay_seconds("gate")
    assert delay is not None and 1.5 <= delay <= 2.0

    with store._connect() as conn:
        conn.execute(
            "update scene_candidate_jobs set available_at_epoch=?,lease_expires_at_epoch=?",
            (now - 1, now + 1.5),
        )
    delay = store.next_camera_work_delay_seconds("gate")
    assert delay is not None and 1.0 <= delay <= 1.5

    event = store.add_event(
        camera_id="cover", kind="motion",
        created_at=datetime.now(timezone.utc).isoformat(), objects_json="[]",
    )
    with store._connect() as conn:
        conn.execute(
            "insert into event_cover_requirements "
            "(event_id,state,deadline_epoch,available_at_epoch,payload_json,created_at,updated_at) "
            "values(?,'pending',?,?, '{}',?,?)",
            (event["id"], now + 4, now + 2, now, now),
        )
    delay = store.next_camera_work_delay_seconds("cover")
    assert delay is not None and 1.5 <= delay <= 2.0
    with store._connect() as conn:
        conn.execute(
            "update event_cover_requirements set attempts=3,lease_expires_at_epoch=? "
            "where event_id=?",
            (now + 1.5, event["id"]),
        )
    delay = store.next_camera_work_delay_seconds("cover")
    assert delay is not None and 1.0 <= delay <= 1.5
    with store._connect() as conn:
        conn.execute(
            "update event_cover_requirements set lease_expires_at_epoch=null,attempts=0,"
            "deadline_epoch=? where event_id=?",
            (now - 1, event["id"]),
        )
    assert store.next_camera_work_delay_seconds("cover") == 0.0


def test_scene_deadline_respects_camera_and_global_demand_leases(tmp_path):
    store = EventStore(tmp_path)
    now = time.time()

    def scene(camera, epoch):
        event = store.add_event(
            camera_id=camera, kind="motion",
            created_at=datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
            objects_json="[]",
        )
        return store.enqueue_scene_tracking(event["id"], epoch, epoch + 1)

    first = scene("gate", now)
    second = scene("gate", now + 1000)
    other = scene("other", now + 2000)
    with store._connect() as conn:
        conn.execute(
            "update scene_analysis_jobs set state='running',lease_expires=? "
            "where episode_id=?",
            (now + 2, first["episode_id"]),
        )
        conn.execute(
            "update scene_analysis_jobs set retry_at=0 where episode_id=?",
            (second["episode_id"],),
        )
    delay = store.next_scene_analysis_delay_seconds("gate")
    assert delay is not None and 1.5 <= delay <= 2.0

    with store._connect() as conn:
        conn.execute(
            "update scene_analysis_jobs set state='complete' where episode_id=?",
            (first["episode_id"],),
        )
        conn.execute(
            "update scene_analysis_jobs set admission='demand',requested_until=?,"
            "request_end_epoch=end_epoch where episode_id=?",
            (now + 30, second["episode_id"]),
        )
        conn.execute(
            "update scene_analysis_jobs set state='running',admission='demand',"
            "requested_until=?,request_end_epoch=end_epoch,lease_expires=? "
            "where episode_id=?",
            (now + 30, now + 1.5, other["episode_id"]),
        )
    delay = store.next_scene_analysis_delay_seconds("gate")
    assert delay is not None and 1.0 <= delay <= 1.5


def test_future_scene_retry_has_its_own_deadline(tmp_path):
    store = EventStore(tmp_path)
    now = time.time()
    event = store.add_event(
        camera_id="gate", kind="motion",
        created_at=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        objects_json="[]",
    )
    job = store.enqueue_scene_tracking(event["id"], now, now + 1)
    with store._connect() as conn:
        conn.execute(
            "update scene_analysis_jobs set retry_at=? where episode_id=?",
            (now + 0.5, job["episode_id"]),
        )
    assert store.next_camera_work_delay_seconds("gate") is None
    delay = store.next_scene_analysis_delay_seconds("gate")
    assert delay is not None and 0.0 < delay <= 0.5

import json
import sqlite3
import threading
from unittest.mock import patch

import pytest

from survng.app.event_store.scene_acquisition import EventStoreSceneAcquisitionMixin


class Ledger(EventStoreSceneAcquisitionMixin):
    def __init__(self, root):
        self.storage_dir=root
        self.db_path=root/"ledger.db"
        self._lock=threading.RLock()
        self._init_scene_acquisition_db()

    def _connect(self):
        conn=sqlite3.connect(self.db_path)
        conn.row_factory=sqlite3.Row
        conn.execute("pragma foreign_keys=on")
        return conn


def observed(confidence=.3):
    return {"label":"chair","confidence":confidence,"frame_source":"live",
            "captured_at_epoch":100,"box":{"x1":1,"y1":2,"x2":3,"y2":4}}


def sample(ledger, key="s", *, observations=None, **extra):
    return ledger.acquire_scene_sample(sample_id=key,camera_id="gate",captured_epoch=100,
        source="discovery",status="complete",observations=[observed()] if observations is None else observations,**extra)


def decision(ledger, key="d", samples=("s",), verdict="supported"):
    return ledger.record_scene_activity_decision(decision_id=key,sample_ids=samples,verdict=verdict,
        activity_epoch=100,reason="measured movement",policy_version="test-v1")


def enqueue(ledger, key="s", **extra):
    return ledger.enqueue_scene_candidate("gate",seed_sample_id=key,start_epoch=95,end_epoch=110,
        deadline_epoch=200,available_at_epoch=100,**extra)


def test_acquisition_without_event_or_incident_and_first_write_replay(tmp_path):
    ledger=Ledger(tmp_path)
    first=sample(ledger)
    replay=sample(ledger,observations=[observed(.9)])
    assert first["created"] and not replay["created"]
    assert replay["observations"][0]["confidence"] == .3
    assert Ledger(tmp_path).scene_sample("s")["observation_ids"] == first["observation_ids"]
    with ledger._connect() as conn:
        assert not conn.execute("select 1 from sqlite_master where name='scene_incidents'").fetchone()
        assert ledger._scene_sample(conn,"s")["id"] == "s"
        assert len(ledger._scene_samples(conn,"gate",90,110)) == 1


def test_complete_empty_and_failed_samples_remain_distinct(tmp_path):
    ledger=Ledger(tmp_path)
    assert sample(ledger,observations=[])["status"] == "complete"
    failed=ledger.acquire_scene_sample(sample_id="failed",camera_id="gate",captured_epoch=101,
        source="discovery",status="failed",observations=[],metadata={"reason":"decoder_unavailable"})
    assert failed["status"] == "failed"
    assert failed["observations"] == []
    with pytest.raises(ValueError,match="collision"):
        ledger.acquire_scene_sample(sample_id="s",camera_id="other",captured_epoch=100,
            source="discovery",status="complete",observations=[])


def test_supported_decision_admits_atomically_and_only_once(tmp_path):
    ledger=Ledger(tmp_path)
    sample(ledger)
    decision(ledger)
    assert len(ledger.scene_pending_activity_decisions("gate")) == 1
    with pytest.raises(RuntimeError):
        with ledger._connect() as conn:
            ledger._admit_scene_activity(conn,decision_id="d",event_id=9,episode_id="episode")
            raise RuntimeError("event transaction failed")
    assert len(ledger.scene_pending_activity_decisions()) == 1
    with ledger._connect() as conn:
        ledger._admit_scene_activity(conn,decision_id="d",event_id=9,episode_id="episode")
        ledger._admit_scene_activity(conn,decision_id="d",event_id=9,episode_id="episode")
    assert ledger.scene_pending_activity_decisions() == []
    decision(ledger,"u",verdict="unsupported")
    with ledger._connect() as conn,pytest.raises(ValueError,match="supported"):
        ledger._admit_scene_activity(conn,decision_id="u",event_id=9,episode_id="episode")


def test_coalesced_candidate_lease_fences_old_worker_and_preserves_new_samples(tmp_path):
    ledger=Ledger(tmp_path)
    sample(ledger)
    sample(ledger,"second")
    with patch("survng.app.event_store.scene_acquisition.time.time",return_value=100):
        queued=enqueue(ledger)
        first=ledger.claim_scene_candidate("gate",lease_owner="worker",lease_seconds=1)
        assert enqueue(ledger,"second")["id"] == queued["id"]
    with patch("survng.app.event_store.scene_acquisition.time.time",return_value=102):
        second=Ledger(tmp_path).claim_scene_candidate("gate",lease_owner="worker",lease_seconds=10)
        assert second["lease_token"] > first["lease_token"]
        decision(ledger,samples=("s","second"))
        assert not ledger.finish_scene_candidate(queued["id"],lease_owner="worker",lease_token=first["lease_token"],decision_id="d")
        with ledger._connect() as conn,pytest.raises(ValueError,match="superseded"):
            ledger.assert_scene_candidate_lease(conn,queued["id"],"worker",first["lease_token"])
        assert ledger.finish_scene_candidate(queued["id"],lease_owner="worker",lease_token=second["lease_token"],decision_id="d")


def test_new_sample_while_running_requeues_followup(tmp_path):
    ledger=Ledger(tmp_path)
    sample(ledger)
    sample(ledger,"second")
    with patch("survng.app.event_store.scene_acquisition.time.time",return_value=100):
        queued=enqueue(ledger)
        job=ledger.claim_scene_candidate("gate",lease_owner="worker")
        enqueue(ledger,"second")
        decision(ledger)
        assert ledger.finish_scene_candidate(queued["id"],lease_owner="worker",lease_token=job["lease_token"],decision_id="d")
        assert ledger.claim_scene_candidate("gate",lease_owner="worker")["attempts"] == 2


def test_expired_verification_is_incomplete_and_keeps_original_evidence(tmp_path):
    ledger=Ledger(tmp_path)
    first=sample(ledger)
    with patch("survng.app.event_store.scene_acquisition.time.time",return_value=100):
        queued=enqueue(ledger)
    with patch("survng.app.event_store.scene_acquisition.time.time",return_value=201):
        assert ledger.claim_scene_candidate("gate",lease_owner="worker") is None
    assert ledger.scene_activity_decision(queued["id"]+":deadline")["verdict"] == "incomplete"
    assert ledger.scene_activity_decision(queued["id"]+":deadline")["activity_epoch"] is None
    assert ledger.scene_sample("s")["observation_ids"] == first["observation_ids"]


def test_expired_snapshot_does_not_resurrect_on_new_sample(tmp_path):
    ledger=Ledger(tmp_path)
    first=sample(ledger,observations=[{**observed(),"snapshot_path":"snapshots/test.jpg"}])
    with ledger._connect() as conn:
        ledger._expire_acquired_snapshots(conn,["snapshots/test.jpg"])
    second=sample(ledger,"second",observations=[{**observed(),"snapshot_path":"snapshots/test.jpg"}])
    assert second["observations"][0]["snapshot_path"] == ""
    assert ledger.scene_acquired_observation(first["observation_ids"][0])["snapshot_path"] == ""


def test_historical_backfill_is_bounded_silent_and_preserves_original_ids(tmp_path):
    ledger=Ledger(tmp_path)
    with ledger._connect() as conn:
        conn.execute("create table scene_observations(id text primary key,event_id integer,camera_id text,captured_epoch real,payload_json text,snapshot_path text,recording_path text)")
        for index in range(3):
            conn.execute("insert into scene_observations values(?,?,?,?,?,?,?)",
                         (f"old-{index}",1,"gate",100+index,json.dumps(observed()),"",""))
    assert ledger.backfill_existing_scene_acquisitions(batch_size=2) == {"processed":2,"complete":False}
    assert Ledger(tmp_path).backfill_existing_scene_acquisitions(batch_size=2) == {"processed":1,"complete":True}
    assert ledger.backfill_existing_scene_acquisitions()["processed"] == 0
    with ledger._connect() as conn:
        assert conn.execute("select count(*) from acquired_observations").fetchone()[0] == 3
        assert [row[0] for row in conn.execute("select source_observation_id from scene_observations order by id")] == ["old-0","old-1","old-2"]
        assert conn.execute("select count(*) from scene_activity_decisions").fetchone()[0] == 0


def test_acquisition_registers_media_without_filesystem_dependency(tmp_path):
    ledger=Ledger(tmp_path)
    with ledger._connect() as conn:
        conn.execute("create table scene_snapshot_assets(snapshot_path text primary key,camera_id text,created_at text,snapshot_size_bytes integer)")
    sample(ledger,snapshot_path="snapshots/frame.jpg",metadata={"snapshot_size_bytes":1234})
    with ledger._connect() as conn:
        asset=conn.execute("select * from scene_snapshot_assets").fetchone()
        assert asset["snapshot_path"] == "snapshots/frame.jpg"
        assert asset["snapshot_size_bytes"] == 1234


def test_candidate_defer_and_event_commit_are_fenced(tmp_path):
    ledger=Ledger(tmp_path)
    sample(ledger)
    with patch("survng.app.event_store.scene_acquisition.time.time",return_value=100):
        queued=enqueue(ledger)
        job=ledger.claim_scene_candidate("gate",lease_owner="worker",lease_seconds=1)
    with patch("survng.app.event_store.scene_acquisition.time.time",return_value=102):
        assert not ledger.defer_scene_candidate(queued["id"],lease_owner="worker",lease_token=job["lease_token"],reason="recording_not_ready")
        with ledger._connect() as conn,pytest.raises(ValueError,match="expired"):
            ledger.assert_scene_candidate_lease(conn,queued["id"],"worker",job["lease_token"])
        current=ledger.claim_scene_candidate("gate",lease_owner="worker")
        assert ledger.defer_scene_candidate(queued["id"],lease_owner="worker",lease_token=current["lease_token"],reason="recording_not_ready",retry_delay_seconds=0)
        assert ledger.claim_scene_candidate("gate",lease_owner="next")["attempts"] == 3


def test_confirmation_includes_prior_empty_discovery_for_arrival_evidence(tmp_path):
    ledger=Ledger(tmp_path)
    ledger.acquire_scene_sample(sample_id='empty-before',camera_id='gate',captured_epoch=90,source='live_discovery',status='complete',observations=[])
    ledger.acquire_scene_sample(sample_id='person-now',camera_id='gate',captured_epoch=100,source='live_discovery',status='complete',observations=[observed()])
    with patch('survng.app.event_store.scene_acquisition.time.time',return_value=100):
        job=ledger.enqueue_scene_candidate('gate',seed_sample_id='person-now',start_epoch=95,end_epoch=105,deadline_epoch=200)
    assert job['start_epoch']==90
    assert job['end_epoch']==105

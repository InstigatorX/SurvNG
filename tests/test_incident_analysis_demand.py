from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from survng.app.config import CameraConfig
from survng.app.event_store import EventStore
from survng.app.object_tracking_lifecycle import ObjectTrackingLifecycle


def bookmark(store, epoch=1000, camera="gate", *, deferred=True):
    event = store.add_event(
        camera_id=camera, kind="motion",
        created_at=datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
        objects_json=json.dumps([{"label": "person", "confidence": .75,
                                 "incident_eligible": False,
                                 "box": {"x1": 10, "y1": 10, "x2": 30, "y2": 70},
                                 "detection_frame_width": 100, "detection_frame_height": 100}]),
    )
    job = store.enqueue_scene_tracking(event["id"], epoch, epoch + 20, deferred=deferred)
    incident = store.scene_incident(event_id=event["id"])
    return event, job, incident["id"]


def checkpoint(store, job, *, state="complete", cursor=1020, reason="", gaps=()):
    tracking = {"scene_analysis_job": {"episode_id": job["episode_id"], "lease_owner": job["lease_owner"]},
                "state": state, "window_end_epoch": job["end_epoch"],
                "scene_cursor_epoch": cursor,
                "analyzed_through": datetime.fromtimestamp(cursor, timezone.utc).isoformat(),
                "completion_reason": reason, "coverage_gaps": list(gaps)}
    with store._lock, store._connect() as conn:
        conn.execute("begin immediate")
        store._checkpoint_scene_tracking(conn, job["event_id"], tracking)


def test_deferred_bookmark_is_searchable_but_get_never_claims(tmp_path):
    store = EventStore(tmp_path)
    _event, _job, incident_id = bookmark(store)
    assert store.scene_incident(incident_id)["labels"] == ["person"]
    for _ in range(2):
        status = store.incident_analysis_status(incident_id)
        assert status["status"] == "deferred"
        assert status["remaining"] == 1
        assert status["episodes"][0]["camera_id"] == "gate"
        assert store.claim_scene_tracking("gate", "worker") is None


def test_default_eager_admission_and_migration_defaults(tmp_path):
    store = EventStore(tmp_path)
    _event, job, _incident = bookmark(store, deferred=False)
    assert job["admission"] == "automatic"
    assert store.claim_scene_tracking("gate", "worker") is not None


def test_pretrial_job_schema_migrates_without_admitting_or_deferring_work():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("""create table scene_analysis_jobs (
            episode_id text primary key,event_id integer,camera_id text,event_epoch real,
            start_epoch real,end_epoch real,cursor_epoch real,state text default 'queued',
            lease_owner text default '',lease_expires real default 0,retry_at real default 0,
            attempts integer default 0,last_error text default '')""")
        conn.execute("insert into scene_analysis_jobs(episode_id,camera_id,state) values('old','gate','queued')")
        EventStore._init_scene_tracking_schema(conn)
        assert conn.execute("select state,admission,requested_until,request_end_epoch from scene_analysis_jobs").fetchone() == (
            "queued", "automatic", 0, None,
        )
    finally:
        conn.close()


def test_concurrent_viewers_share_job_and_completed_results(tmp_path):
    store = EventStore(tmp_path)
    _event, _job, incident = bookmark(store)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.request_incident_analysis(incident), range(8)))
    assert all(result["status"] == "queued" for result in results)
    job = store.claim_scene_tracking("gate", "worker")
    assert store.claim_scene_tracking("gate", "other") is None
    assert store.is_demand_scene_tracking({"episode_id": job["episode_id"], "lease_owner": "worker"})
    assert not store.is_demand_scene_tracking({"episode_id": job["episode_id"], "lease_owner": "other"})
    checkpoint(store, job)
    assert store.request_incident_analysis(incident)["status"] == "complete"
    assert store.claim_scene_tracking("gate", "again") is None
    with store._connect() as conn:
        assert conn.execute("select count(*) from scene_analysis_jobs").fetchone()[0] == 1


def test_global_demand_cap_does_not_reduce_automatic_capacity(tmp_path):
    store = EventStore(tmp_path)
    _, _, first = bookmark(store)
    _, _, second = bookmark(store, camera="drive")
    bookmark(store, camera="porch", deferred=False)
    store.request_incident_analysis(first)
    store.request_incident_analysis(second)
    first_job = store.claim_scene_tracking("gate", "one")
    assert store.claim_scene_tracking("drive", "two") is None
    assert store.claim_scene_tracking("porch", "automatic") is not None
    checkpoint(store, first_job)
    assert store.claim_scene_tracking("drive", "two") is not None


def test_view_expiry_stops_next_chunk_but_retains_cursor(tmp_path, monkeypatch):
    now = [2000.0]
    monkeypatch.setattr("survng.app.event_store.scene_tracking.time.time", lambda: now[0])
    store = EventStore(tmp_path)
    _, _, incident = bookmark(store)
    store.request_incident_analysis(incident)
    job = store.claim_scene_tracking("gate", "one")
    now[0] += 61
    checkpoint(store, job, state="interrupted", cursor=1005, reason="processing_budget_exhausted")
    assert store.incident_analysis_status(incident)["status"] == "deferred"
    assert store.claim_scene_tracking("gate", "two") is None
    store.request_incident_analysis(incident)
    resumed = store.claim_scene_tracking("gate", "two")
    assert resumed["cursor_epoch"] == 1005


def test_queued_expiry_and_crash_recovery_are_lease_safe(tmp_path, monkeypatch):
    now = [2000.0]
    monkeypatch.setattr("survng.app.event_store.scene_tracking.time.time", lambda: now[0])
    store = EventStore(tmp_path)
    _, _, incident = bookmark(store)
    store.request_incident_analysis(incident)
    now[0] += 61
    assert store.claim_scene_tracking("gate", "expired") is None
    store.request_incident_analysis(incident)
    job = store.claim_scene_tracking("gate", "crashed")
    now[0] += 61
    restarted = EventStore(tmp_path)
    restarted.request_incident_analysis(incident)
    recovered = restarted.claim_scene_tracking("gate", "restarted")
    assert recovered["episode_id"] == job["episode_id"]
    assert recovered["lease_owner"] == "restarted"
    assert not restarted.is_demand_scene_tracking({"episode_id": job["episode_id"], "lease_owner": "crashed"})


def test_active_extension_does_not_expand_inflight_request(tmp_path):
    store = EventStore(tmp_path)
    event, original, incident = bookmark(store)
    store.request_incident_analysis(incident)
    job = store.claim_scene_tracking("gate", "worker")
    store.enqueue_scene_tracking(event["id"], 1000, 1040, deferred=True)
    store.request_incident_analysis(incident)
    with store._connect() as conn:
        row = conn.execute("select * from scene_analysis_jobs").fetchone()
        assert row["request_end_epoch"] == 1020
        conn.execute("update scene_episodes set last_activity_epoch=1100")
    checkpoint(store, job)
    assert store.incident_analysis_status(incident)["status"] == "deferred"
    store.request_incident_analysis(incident)
    next_job = store.claim_scene_tracking("gate", "next")
    assert next_job["end_epoch"] == 1040
    assert next_job["cursor_epoch"] == 1020
    assert next_job["episode_id"] == original["episode_id"]


def test_incomplete_terminal_results_are_not_retried_on_every_open(tmp_path):
    store = EventStore(tmp_path)
    _, _, incident = bookmark(store)
    store.request_incident_analysis(incident)
    job = store.claim_scene_tracking("gate", "worker")
    checkpoint(store, job, gaps=[{"start_epoch": 1005, "end_epoch": 1010, "reason": "missing_recording"}])
    assert store.request_incident_analysis(incident)["status"] == "partial"
    assert store.claim_scene_tracking("gate", "again") is None


def test_disabling_trial_does_not_release_deferred_backlog(tmp_path):
    store = EventStore(tmp_path)
    event, _, incident = bookmark(store)
    store.enqueue_scene_tracking(event["id"], 1000, 1040)
    assert store.incident_analysis_status(incident)["status"] == "deferred"
    assert store.claim_scene_tracking("gate", "worker") is None


def test_request_resolves_alias_and_rejects_unknown(tmp_path):
    store = EventStore(tmp_path)
    event, _, incident = bookmark(store)
    alias = f"incident-gate-{event['id']}"
    assert store.request_incident_analysis(alias) == store.incident_analysis_status(incident)
    with pytest.raises(KeyError):
        store.incident_analysis_status("not-found")
    with pytest.raises(KeyError):
        store.request_incident_analysis("not-found")


def test_merged_incident_renews_at_most_eight_outstanding_episodes(tmp_path):
    store = EventStore(tmp_path)
    incidents = [bookmark(store, 1000 + index * 1000)[2] for index in range(11)]
    root = store.scene_incident(incidents[0])
    store.correct_scene_incident(root["id"], root["revision"], {
        "operation": "merge", "incident_ids": incidents[1:],
        "expected_revisions": {key: store.scene_incident(key)["revision"] for key in incidents[1:]},
    })
    for _ in range(3):
        status = store.request_incident_analysis(incidents[-1])
        assert sum(episode["status"] == "queued" for episode in status["episodes"]) == 8
        assert status["remaining"] == 3


def test_automatic_work_is_claimed_before_demand_on_same_camera(tmp_path):
    store = EventStore(tmp_path)
    _, _, deferred = bookmark(store)
    _, automatic, _ = bookmark(store, 2000, deferred=False)
    store.request_incident_analysis(deferred)
    claimed = store.claim_scene_tracking("gate", "worker")
    assert claimed["episode_id"] == automatic["episode_id"]


def test_request_camera_allowlist_does_not_queue_unavailable_camera(tmp_path):
    store = EventStore(tmp_path)
    _, _, first = bookmark(store)
    _, _, second = bookmark(store, camera="drive")
    root = store.scene_incident(first)
    store.correct_scene_incident(first, root["revision"], {
        "operation": "merge", "incident_ids": [second],
        "expected_revisions": {second: store.scene_incident(second)["revision"]},
    })
    status = store.request_incident_analysis(first, camera_ids={"drive"})
    assert {episode["camera_id"]: episode["status"] for episode in status["episodes"]} == {
        "gate": "deferred", "drive": "queued",
    }


def test_expired_unavailable_camera_requests_do_not_block_eligible_episodes(tmp_path, monkeypatch):
    now = [20000.0]
    monkeypatch.setattr("survng.app.event_store.scene_tracking.time.time", lambda: now[0])
    store = EventStore(tmp_path)
    incidents = [bookmark(store, 1000 + index * 1000, "gate" if index < 8 else "drive")[2]
                 for index in range(9)]
    root = store.scene_incident(incidents[0])
    store.correct_scene_incident(root["id"], root["revision"], {
        "operation": "merge", "incident_ids": incidents[1:],
        "expected_revisions": {key: store.scene_incident(key)["revision"] for key in incidents[1:]},
    })
    store.request_incident_analysis(root["id"])
    now[0] += 61
    status = store.request_incident_analysis(root["id"], camera_ids={"drive"})
    assert next(episode for episode in status["episodes"] if episode["camera_id"] == "drive")["status"] == "queued"


def test_long_demand_lease_is_not_shortened_by_progress(tmp_path, monkeypatch):
    now = [2000.0]
    monkeypatch.setattr("survng.app.event_store.scene_tracking.time.time", lambda: now[0])
    store = EventStore(tmp_path)
    _, _, first = bookmark(store)
    _, _, second = bookmark(store, camera="drive")
    store.request_incident_analysis(first)
    store.request_incident_analysis(second)
    job = store.claim_scene_tracking("gate", "slow", demand_lease_seconds=340)
    checkpoint(store, job, state="active", cursor=1005)
    now[0] += 61
    store.request_incident_analysis(second)
    assert store.claim_scene_tracking("drive", "must-wait") is None
    with store._connect() as conn:
        assert conn.execute("select lease_expires from scene_analysis_jobs where episode_id=?",
                            (job["episode_id"],)).fetchone()[0] == 2340


def test_failed_demand_is_terminal_and_identified_as_unavailable(tmp_path):
    store = EventStore(tmp_path)
    _, _, incident = bookmark(store)
    store.request_incident_analysis(incident)
    job = store.claim_scene_tracking("gate", "worker")
    checkpoint(store, job, state="failed", reason="decoder_failed")
    with store._connect() as conn:
        conn.execute("update scene_analysis_jobs set analyzed_epoch=null")
    assert store.request_incident_analysis(incident)["status"] == "unavailable"
    assert store.claim_scene_tracking("gate", "again") is None


def test_on_demand_lifecycle_skips_prewarm_and_preserves_bookmark(tmp_path):
    store = EventStore(tmp_path)
    event, _, incident = bookmark(store)
    session = Mock()
    session.config = SimpleNamespace(enabled=True, analysis_mode="on_demand", max_session_seconds=20)
    session.window_provider = lambda *_: (1000, 1020)
    session.running.return_value = False
    factory, prewarm = Mock(), Mock()
    factory.create.return_value = session
    lifecycle = ObjectTrackingLifecycle(
        camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        factory=factory, frame_provider=Mock(), catchup_frame_provider=Mock(),
        prewarm_frame_provider=prewarm, history=Mock(), accepting=lambda: True,
        lifecycle_lock=threading.RLock(), scene_job_store=store,
    )
    lifecycle._trackable_objects = Mock(return_value=[])
    assert lifecycle.prewarm() is None
    prewarm.assert_not_called()
    assert lifecycle.start_incident(event["id"], datetime.fromtimestamp(1000, timezone.utc), [])
    session.start.assert_not_called()
    assert store.incident_analysis_status(incident)["status"] == "deferred"

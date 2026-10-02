from datetime import datetime, timezone
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from survng.app.config import CameraConfig, ObjectTrackingConfig
from survng.app.event_store import EventStore
from survng.app.object_tracking import ObjectTrackingSession


def person():
    return {"label": "person", "confidence": .75, "incident_eligible": False,
            "box": {"x1": 10, "y1": 10, "x2": 30, "y2": 70},
            "detection_frame_width": 100, "detection_frame_height": 100}


def create_event(store, epoch):
    return store.add_event(camera_id="gate", kind="motion",
                           created_at=datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
                           objects_json=json.dumps([person()]))


def make_session(store, *, interrupt_after_first=False, limiter=None, requests=None, observations=None):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    requests = requests if requests is not None else []

    def frames(start, end, fps, width, *, after_epoch=None):
        requests.append((start, end, after_epoch))
        return [(epoch, frame) for epoch in (1000,1000.5,1001,1001.5,1002)
                if start <= epoch <= end]

    def update(event_id, payload, objects):
        result = store.update_object_tracking(event_id, payload, objects)
        if interrupt_after_first and payload["frames_processed"] == 1:
            session._deadline = time.monotonic() - 1
        return result

    session = ObjectTrackingSession(
        camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        config=ObjectTrackingConfig(capacity_wait_seconds=0, persist_interval_seconds=0),
        detector=SimpleNamespace(config=SimpleNamespace(confidence_threshold=.45, require_incident_zone=False),
                                 detect=lambda *_args, **_kwargs: observations if observations is not None else [person()]),
        frame_provider=lambda: None, catchup_frame_provider=frames,
        update_event=update, publisher=None, limiter=limiter or threading.BoundedSemaphore(1),
    )
    session.set_accepting(True)
    return session


def run_job(session, job, owner, *, resume=None):
    assert session.start(job["event_id"], datetime.fromtimestamp(job["event_epoch"], timezone.utc),
                         [], None, recorded_window=(job["start_epoch"],job["end_epoch"]),
                         resume_after=job["cursor_epoch"],
                         scene_analysis_job={"episode_id":job["episode_id"],"lease_owner":owner,
                                             "analyzed_through_epoch":job.get("analyzed_epoch")},
                         scene_track_resume=resume)
    try:
        assert session.wait_stopped(3)
    finally:
        session.stop()


def test_processing_budget_cursor_survives_restart_and_resumes_exclusively(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    queued = store.enqueue_scene_tracking(event["id"], 1000, 1002)
    first = store.claim_scene_tracking("gate", "worker-one")
    run_job(make_session(store, interrupt_after_first=True), first, "worker-one")
    restarted = EventStore(tmp_path)
    resumed = restarted.claim_scene_tracking("gate", "worker-two")
    assert resumed["episode_id"] == queued["episode_id"]
    assert resumed["cursor_epoch"] == 1000
    requests = []
    run_job(make_session(restarted, requests=requests), resumed, "worker-two")
    assert requests[0][2] == 1000
    assert requests[0][0] > 1000
    assert restarted.claim_scene_tracking("gate", "worker-three") is None
    with restarted._connect() as conn:
        final = conn.execute("select * from scene_analysis_jobs").fetchone()
        assert final["state"] == "complete"
        assert final["cursor_epoch"] == 1002
        times = [row[0] for row in conn.execute("select distinct captured_epoch from scene_observations "
                                               "where captured_epoch between 1000 and 1002 order by captured_epoch")]
    assert times == [1000,1000.5,1001,1001.5,1002]


def test_capacity_saturation_retains_job_for_later_claim(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1002)
    job = store.claim_scene_tracking("gate", "capacity-worker")
    limiter = threading.BoundedSemaphore(1)
    assert limiter.acquire(blocking=False)
    run_job(make_session(store, limiter=limiter), job, "capacity-worker")
    with store._connect() as conn:
        persisted = conn.execute("select * from scene_analysis_jobs").fetchone()
        assert persisted["state"] == "queued"
        assert persisted["cursor_epoch"] is None
        assert persisted["last_error"] == "tracking_capacity"
        conn.execute("update scene_analysis_jobs set retry_at=0")
    limiter.release()
    resumed = store.claim_scene_tracking("gate", "available-worker")
    run_job(make_session(store, limiter=limiter), resumed, "available-worker")
    assert store.claim_scene_tracking("gate", "finished") is None


def test_slow_physical_movement_accumulates_across_tracking_samples(tmp_path):
    store=EventStore(tmp_path)
    event=create_event(store,1000)
    store.enqueue_scene_tracking(event['id'],1000,1002)
    job=store.claim_scene_tracking('gate','slow-motion')
    session=make_session(store)
    pictures=[]
    for index in range(11):
        frame=np.zeros((100,100,3),dtype=np.uint8)
        frame[10:70,10+index:60+index]=220
        pictures.append((1000+index/5,frame))
    session.catchup_frame_provider=lambda *_args,**_kwargs:pictures
    def detect(frame,*_args,**_kwargs):
        left=int(np.nonzero(frame[20,:,0])[0][0])
        return [{**person(),'box':{'x1':left,'y1':10,'x2':left+50,'y2':70}}]
    session.detector.detect=detect
    run_job(session,job,'slow-motion')
    incident=store.scene_incident(event_id=event['id'])
    assert datetime.fromisoformat(incident['episodes'][0]['last_activity_at']).timestamp()>1001
    assert incident['establishment']['reason']=='video_verified_activity'


def test_new_activity_extends_same_episode_job_and_old_lease_cannot_advance_it(tmp_path):
    store = EventStore(tmp_path)
    first_event = create_event(store, 1000)
    first = store.enqueue_scene_tracking(first_event["id"], 995, 1045)
    claimed = store.claim_scene_tracking("gate", "current")
    second_event = create_event(store, 1020)
    extended = store.enqueue_scene_tracking(second_event["id"], 1015, 1065)
    assert extended["episode_id"] == first["episode_id"]
    assert extended["event_id"] == first_event["id"]
    assert extended["end_epoch"] == 1065
    store.update_object_tracking(first_event["id"], {
        "scene_analysis_job":{"episode_id":claimed["episode_id"],"lease_owner":"obsolete"},
        "state":"complete", "window_end_epoch":1065,
        "analyzed_through":datetime.fromtimestamp(1065,timezone.utc).isoformat(),
    })
    with store._connect() as conn:
        persisted = conn.execute("select * from scene_analysis_jobs").fetchone()
    assert persisted["state"] == "running"
    assert persisted["cursor_epoch"] is None


def test_measured_motion_extends_analysis_without_another_trigger(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1045)
    job = store.claim_scene_tracking("gate", "moving-worker")
    observed = {**person(), "captured_at_epoch":1044, "frame_source":"recorded_main",
                "box":{"x1":60,"y1":10,"x2":80,"y2":70}}
    from survng.app.motion_pipeline.object_detection import _RecordedDetectionSample, _scene_sample_records
    earlier = {**person(), "captured_at_epoch":1043, "frame_source":"recorded_main"}
    frames = []
    for item in (earlier, observed):
        frame = np.zeros((100,100,3),dtype=np.uint8)
        b=item["box"]; frame[b["y1"]:b["y2"],b["x1"]:b["x2"]]=220
        frames.append(_RecordedDetectionSample(item["captured_at_epoch"],frame,[item],"main.mp4",exact_timestamp=True))
    measured = _scene_sample_records(frames,[earlier,observed],0,"gate")
    store.update_object_tracking(event["id"], {
        "scene_analysis_job":{"episode_id":job["episode_id"],"lease_owner":"moving-worker"},
        "scene_observations":[earlier,observed], "scene_samples":measured, "state":"complete", "window_end_epoch":1045,
        "analyzed_through":datetime.fromtimestamp(1044,timezone.utc).isoformat(),
    })
    resumed = store.claim_scene_tracking("gate", "continuing-worker")
    assert resumed["episode_id"] == job["episode_id"]
    assert resumed["event_id"] == event["id"]
    assert resumed["end_epoch"] == 1089
    assert resumed["cursor_epoch"] == 1044
    scene=store.scene_incident(event_id=event["id"])
    assert scene["last_epoch"] == 1089
    assert datetime.fromisoformat(scene["episodes"][0]["end_at"]).timestamp() == 1089
    with store._connect() as conn:
        assert conn.execute("select count(*) from events").fetchone()[0] == 1


def _measured_tracking_payload(job, owner, *, analyzed_through):
    from survng.app.motion_pipeline.object_detection import _RecordedDetectionSample, _scene_sample_records
    earlier = {**person(), "captured_at_epoch":1043, "frame_source":"recorded_main"}
    observed = {**person(), "captured_at_epoch":1044, "frame_source":"recorded_main",
                "box":{"x1":60,"y1":10,"x2":80,"y2":70}}
    frames = []
    for item in (earlier, observed):
        frame = np.zeros((100,100,3),dtype=np.uint8)
        b=item["box"]; frame[b["y1"]:b["y2"],b["x1"]:b["x2"]]=220
        frames.append(_RecordedDetectionSample(item["captured_at_epoch"],frame,[item],"main.mp4",exact_timestamp=True))
    return {
        "scene_analysis_job":{"episode_id":job["episode_id"],"lease_owner":owner},
        "scene_observations":[earlier,observed],
        "scene_samples":_scene_sample_records(frames,[earlier,observed],0,"gate"),
        "state":"active", "analyzed_through":datetime.fromtimestamp(analyzed_through,timezone.utc).isoformat(),
    }


def test_tracking_extension_attaches_context_inside_the_new_window(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1045)
    job = store.claim_scene_tracking("gate", "worker")
    store.acquire_scene_sample(sample_id="later-context", camera_id="gate", captured_epoch=1070,
                               source="recorded_main", status="complete", observations=[])
    store.update_object_tracking(event["id"], _measured_tracking_payload(job, "worker", analyzed_through=1044))
    with store._connect() as conn:
        assert conn.execute("select end_epoch from scene_episodes where id=?",
                            (job["episode_id"],)).fetchone()[0] == 1089
        assert conn.execute("select 1 from acquired_sample_episodes where sample_id='later-context' "
                            "and episode_id=?", (job["episode_id"],)).fetchone()


def test_tracking_update_without_extension_ingests_once(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1045)
    job = store.claim_scene_tracking("gate", "worker")
    store.update_object_tracking(event["id"], _measured_tracking_payload(job, "worker", analyzed_through=1044))
    revision = store.scene_incident(event_id=event["id"])["revision"]
    store._scene_ingest = Mock(wraps=store._scene_ingest)
    store.update_object_tracking(event["id"], {
        "scene_analysis_job":{"episode_id":job["episode_id"],"lease_owner":"worker"},
        "state":"active", "analyzed_through":datetime.fromtimestamp(1045,timezone.utc).isoformat(),
    })
    assert store._scene_ingest.call_count == 1
    assert store.scene_incident(event_id=event["id"])["revision"] > revision


def test_existing_scene_observations_gain_the_event_time_index(tmp_path):
    EventStore(tmp_path)
    store = EventStore(tmp_path)
    with store._connect() as conn:
        conn.execute("drop index scene_observation_event_time")
        conn.execute("create index scene_observation_event on scene_observations(event_id)")
    store = EventStore(tmp_path)
    with store._connect() as conn:
        names = {row[0] for row in conn.execute("select name from sqlite_master where tbl_name='scene_observations'")}
        plan = " ".join(str(row[-1]) for row in conn.execute(
            "explain query plan select id from scene_observations where event_id=1 "
            "and captured_epoch between 999.94 and 1000.06 and abs(captured_epoch-1000)<=0.05"))
    assert "scene_observation_event_time" in names and "scene_observation_event" not in names
    assert "scene_observation_event_time (event_id=? AND captured_epoch>? AND captured_epoch<?)" in plan


def test_historical_decisions_backfill_once_per_event(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1045)
    job = store.claim_scene_tracking("gate", "worker")
    lease = {"episode_id":job["episode_id"],"lease_owner":"worker"}
    eligible = {"alert_eligible": True, "alert_reasons": [], "zones": []}
    store.update_object_tracking(event["id"], {
        "scene_analysis_job":lease, "state":"active",
        "scene_observations":[{**person(), **eligible, "captured_at_epoch":1001+i/2} for i in range(4)],
    })
    observation_ids = ("select id from scene_observations where event_id=? "
                       "and json_type(payload_json,'$.alert_eligible') is not null")
    with store._connect() as conn:
        observations = conn.execute(observation_ids, (event["id"],)).fetchall()
        assert len(observations) == 4
        conn.execute("delete from scene_observation_decisions where observation_id in "
                     "(select id from scene_observations where event_id=?)", (event["id"],))
        conn.execute("delete from scene_event_decision_backfill where event_id=?", (event["id"],))

    store.update_object_tracking(event["id"], {"scene_analysis_job":lease, "state":"active"})
    decisions = ("select count(distinct observation_id) from scene_observation_decisions where source='explicit' "
                 "and observation_id in (select id from scene_observations where event_id=?)")
    with store._connect() as conn:
        assert conn.execute(decisions, (event["id"],)).fetchone()[0] == len(observations)

    store._scene_record_decision = Mock(wraps=store._scene_record_decision)
    store.update_object_tracking(event["id"], {
        "scene_analysis_job":lease, "state":"active",
        "scene_observations":[{**person(), **eligible, "captured_at_epoch":1010}],
    })
    recorded = {call.args[1] for call in store._scene_record_decision.call_args_list}
    assert not recorded & {row[0] for row in observations}
    with store._connect() as conn:
        assert conn.execute(decisions, (event["id"],)).fetchone()[0] == len(observations) + 1


def test_tracking_commit_refreshes_alert_decisions_once(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1045)
    job = store.claim_scene_tracking("gate", "worker")
    lease = {"episode_id":job["episode_id"],"lease_owner":"worker"}
    eligible = {"alert_eligible": True, "alert_reasons": [], "zones": []}
    store.update_object_tracking(event["id"], {"scene_analysis_job":lease, "state":"active"})
    refresh = store._scene_refresh_alerts
    store._scene_refresh_alerts = Mock(wraps=refresh)
    store.update_object_tracking(event["id"], {
        "scene_analysis_job":lease, "state":"active",
        "scene_observations":[{**person(), **eligible, "captured_at_epoch":1002}],
        "scene_samples":[{"captured_epoch":1002, "status":"complete", "metadata":{"source":"recorded_main"}}],
    })
    assert store._scene_refresh_alerts.call_count == 1
    alerts = "select payload_json from scene_alert_decisions where event_id=?"
    with store._connect() as conn:
        stored = conn.execute(alerts, (event["id"],)).fetchone()[0]
        assert json.loads(stored)["eligible"] is True
        assert refresh(conn, event["id"]) is False


def test_notification_snapshot_matches_lifecycle_payloads_without_history(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1045)
    job = store.claim_scene_tracking("gate", "worker")
    store.update_object_tracking(event["id"], {
        "scene_analysis_job": {"episode_id": job["episode_id"], "lease_owner": "worker"}, "state": "active",
        "scene_observations": [{**person(), "captured_at_epoch": 1001 + i} for i in range(6)],
    })
    full = store.list_scene_incidents()
    assert max(len(subject["observations"]) for subject in full[0]["scene_objects"]) > 1
    pending = {item["incident_id"]: item["payload"] for item in store.scene_pending_notifications()}
    snapshot = store.list_scene_incident_notifications()
    assert [item["id"] for item in snapshot] == [item["id"] for item in full]
    assert all(len(subject["observations"]) == 1 for subject in snapshot[0]["scene_objects"])
    assert snapshot[0] == pending[snapshot[0]["id"]]


def test_long_activity_continues_the_incident_in_a_bounded_episode(tmp_path):
    from survng.app.event_store.scenes import MAX_SCENE_EPISODE_SECONDS
    store = EventStore(tmp_path)
    events = [create_event(store, 1000 + offset) for offset in range(0, int(MAX_SCENE_EPISODE_SECONDS) + 120, 40)]
    with store._connect() as conn:
        rows = conn.execute("select m.event_id,p.id,p.incident_id,p.start_epoch,p.boundary_locked from scene_event_membership m "
                            "join scene_episodes p on p.id=m.episode_id order by m.event_id").fetchall()
    episodes = list(dict.fromkeys(row["id"] for row in rows))
    assert len(rows) == len(events)
    assert len(episodes) == 2
    assert len({row["incident_id"] for row in rows}) == 1
    first = [row for row in rows if row["id"] == episodes[0]]
    assert max(row["event_id"] for row in first) < min(row["event_id"] for row in rows if row["id"] == episodes[1])
    assert all(row["boundary_locked"] == 1 for row in first)
    assert {row["boundary_locked"] for row in rows if row["id"] == episodes[1]} == {0}
    with store._connect() as conn:
        second_start = conn.execute("select start_epoch from scene_episodes where id=?", (episodes[1],)).fetchone()[0]
    assert second_start - 1000 >= MAX_SCENE_EPISODE_SECONDS


def test_activity_cannot_extend_a_tracking_job_past_the_episode_bound(tmp_path):
    from survng.app.event_store.scenes import MAX_SCENE_EPISODE_SECONDS
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1045)
    job = store.claim_scene_tracking("gate", "worker")
    with store._connect() as conn:
        conn.execute("update scene_episodes set last_activity_epoch=? where id=?", (1000 + 4 * MAX_SCENE_EPISODE_SECONDS, job["episode_id"]))
        conn.commit()
    store.update_object_tracking(event["id"], {
        "scene_analysis_job": {"episode_id": job["episode_id"], "lease_owner": "worker"}, "state": "active",
    })
    with store._connect() as conn:
        end = conn.execute("select end_epoch from scene_analysis_jobs where episode_id=?", (job["episode_id"],)).fetchone()[0]
    assert end == job["start_epoch"] + MAX_SCENE_EPISODE_SECONDS


def _reference_alerts(conn, event_id):
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    policy = conn.execute("select o.id,o.object_id,d.source,d.payload_json "
                          "from scene_observation_decisions d join scene_observations o on o.id=d.observation_id "
                          "where o.event_id=? order by o.captured_epoch,coalesce(o.object_index,2147483647),o.id,d.rowid",
                          (event_id,)).fetchall()
    explicit = {o["object_id"] for o in policy if o["source"] == "explicit"}
    decisions = {}
    for o in policy:
        if o["source"] != "explicit" and o["object_id"] in explicit:
            continue
        decision = {**json.loads(o["payload_json"]), "object_id": o["object_id"], "observation_id": o["id"]}
        decisions[(decision["object_id"], decision["eligible"], canonical(decision["reasons"]),
                   canonical(decision["zones"]))] = decision
    return list(decisions.values())


def test_alert_decisions_keep_first_position_and_latest_observation(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1045)
    job = store.claim_scene_tracking("gate", "worker")
    lease = {"episode_id":job["episode_id"],"lease_owner":"worker"}
    store.update_object_tracking(event["id"], {
        "scene_analysis_job":lease, "state":"active",
        "scene_observations":[{**person(), "alert_eligible": True, "alert_reasons": [], "zones": [],
                               "captured_at_epoch":1001+i} for i in range(3)] + [
                              {**person(), "label": "car", "box": {"x1": 60, "y1": 60, "x2": 95, "y2": 95},
                               "alert_eligible": True, "alert_reasons": [], "zones": [],
                               "captured_at_epoch":1001.5}],
    })
    with store._connect() as conn:
        rows = conn.execute("select id,object_id from scene_observations where event_id=? "
                            "order by captured_epoch", (event["id"],)).fetchall()
        assert len({row["object_id"] for row in rows}) >= 2
        for index, row in enumerate(rows):
            conn.execute("insert into scene_observation_decisions values(?,?,?,?)", (
                f"legacy-{index}", row["id"], "legacy",
                json.dumps({"eligible": index % 2 == 0, "label": "person", "reasons": ["legacy"], "zones": []},
                           sort_keys=True, separators=(",", ":"))))
            conn.execute("insert into scene_observation_decisions values(?,?,?,?)", (
                f"relabel-{index}", row["id"], "explicit",
                json.dumps({"eligible": index < 2, "label": f"label-{index}", "reasons": ["zone"], "zones": ["a"]},
                           sort_keys=True, separators=(",", ":"))))
        conn.commit()
        conn.execute("begin immediate")
        store._scene_refresh_alerts(conn, event["id"])
        stored = json.loads(conn.execute("select payload_json from scene_alert_decisions where event_id=?",
                                         (event["id"],)).fetchone()[0])
        reference = _reference_alerts(conn, event["id"])
        conn.rollback()
    assert stored["objects"] == reference
    assert stored["eligible"] is any(item["eligible"] for item in reference)


def test_slow_progress_commits_back_off_routine_persistence(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1030)
    job = store.claim_scene_tracking("gate", "worker")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pictures = [(1000 + index / 2, frame) for index in range(60)]
    states = []
    session = make_session(store)
    session.config = session.config.model_copy(update={"persist_interval_seconds": 0.01})
    session.catchup_frame_provider = lambda start, end, *_args, **_kwargs: [
        item for item in pictures if start <= item[0] <= end]

    def slow_update(event_id, payload, objects):
        time.sleep(0.05)
        states.append(payload["state"])
        return store.update_object_tracking(event_id, payload, objects)

    session.update_event = slow_update
    run_job(session, job, "worker")
    assert states[-1] == "complete"
    assert states.count("active") < 10


def test_expired_lease_can_be_recovered_after_process_crash(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1002)
    first = store.claim_scene_tracking("gate", "crashed-worker")
    assert store.claim_scene_tracking("gate", "other-worker") is None
    with store._connect() as conn:
        conn.execute("update scene_analysis_jobs set lease_expires=0")
    recovered = EventStore(tmp_path).claim_scene_tracking("gate", "restarted-worker")
    assert recovered["episode_id"] == first["episode_id"]
    assert recovered["cursor_epoch"] is None
    assert recovered["lease_owner"] == "restarted-worker"


def test_classes_excluded_from_tracking_remain_in_full_scene_acquisition(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1002)
    job = store.claim_scene_tracking("gate", "scene-worker")
    run_job(make_session(store, observations=[{**person(), "label":"face"}]), job, "scene-worker")
    with store._connect() as conn:
        observations = [json.loads(row[0]) for row in conn.execute("select payload_json from scene_observations")]
        state = conn.execute("select state from scene_analysis_jobs").fetchone()[0]
    assert sum(item["label"] == "face" for item in observations) == 5
    assert state == "complete"


def test_tracking_confirmation_uses_scene_confidence_not_alert_threshold(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1002)
    job = store.claim_scene_tracking("gate", "independent-worker")
    session = make_session(store)
    session.detector.config.confidence_threshold = .99
    session.config.confirmation_confidence_threshold = .65
    run_job(session, job, "independent-worker")
    objects = json.loads(store.get(event["id"])["objects_json"])
    tracking = next(item["object_tracking"] for item in objects if item.get("status") == "object_tracking")
    assert len(tracking["tracks"]) == 1
    with store._connect() as conn:
        acquired = [json.loads(row[0]) for row in conn.execute("select payload_json from scene_observations")]
    assert any(item.get("scene_confirmation_threshold") == .65 for item in acquired)
    assert all(item.get("alert_eligible") is False for item in acquired if item.get("frame_source") == "tracking")


def test_tracking_low_threshold_does_not_lower_scene_acquisition_floor(tmp_path):
    store=EventStore(tmp_path)
    event=create_event(store,1000)
    store.enqueue_scene_tracking(event["id"],1000,1002)
    job=store.claim_scene_tracking("gate","floor-worker")
    below={**person(),"confidence":.2}
    at_floor={**person(),"confidence":.25,"box":{"x1":60,"y1":10,"x2":80,"y2":70}}
    session=make_session(store,observations=[below,at_floor])
    session.config.low_confidence_threshold=.1
    session.detector.config.event_candidate_confidence_threshold=.25
    run_job(session,job,"floor-worker")
    with store._connect() as conn:
        acquired=[json.loads(row[0]) for row in conn.execute("select payload_json from scene_observations")]
    tracking=[item for item in acquired if item.get("frame_source")=="tracking"]
    assert len(tracking)==5
    assert all(item["confidence"]==.25 for item in tracking)
    assert all(item["temporal_candidate_threshold"]==.25 for item in tracking)


def test_person_arriving_after_cover_can_alert_after_temporal_confirmation(tmp_path):
    store = EventStore(tmp_path)
    event = store.add_event(camera_id="gate",kind="motion",objects_json="[]",
                            created_at=datetime.fromtimestamp(1000,timezone.utc).isoformat())
    store.enqueue_scene_tracking(event["id"],1000,1002)
    job = store.claim_scene_tracking("gate","late-person-worker")
    session = make_session(store)
    session.detector.detect = Mock(side_effect=[[],[person()],[person()],[person()],[person()]])
    run_job(session,job,"late-person-worker")
    with store._connect() as conn:
        acquired = [json.loads(row[0]) for row in conn.execute("select payload_json from scene_observations order by captured_epoch")]
    assert acquired[0]["alert_eligible"] is False
    assert acquired[0]["captured_at_epoch"] == 1000.5
    assert any(item["alert_eligible"] is True for item in acquired[1:])
    incident=store.scene_incident(event_id=event["id"])
    assert len(incident["scene_objects"]) == 1
    assert incident["alert_decisions"][0]["eligible"] is True
    assert any(item["eligible"] for item in incident["alert_decisions"][0]["objects"])


def test_outside_alert_zone_observations_never_gain_alert_eligibility_from_tracking(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store,1000)
    store.enqueue_scene_tracking(event["id"],1000,1002)
    job = store.claim_scene_tracking("gate","outside-worker")
    session = make_session(store)
    session.detector.config.require_incident_zone = True
    session.camera = CameraConfig(id="gate",name="Gate",stream_url="rtsp://example.invalid/main", zones=[{
        "name":"Door","behavior":"incident","points":[{"x":.8,"y":0},{"x":1,"y":0},{"x":1,"y":1},{"x":.8,"y":1}],
    }])
    run_job(session,job,"outside-worker")
    with store._connect() as conn:
        acquired = [json.loads(row[0]) for row in conn.execute("select payload_json from scene_observations")]
    observations = [item for item in acquired if item.get("frame_source") == "tracking"]
    assert len(observations) == 5
    assert all(item["alert_eligible"] is False for item in observations)
    assert observations[-1]["alert_confirmation_observations"] >= 2
    incident=store.scene_incident(event_id=event["id"])
    assert incident["scene_objects"]
    assert incident["alert_decisions"][0]["eligible"] is False


def test_recorded_analysis_window_extends_playback_through_empty_roll(tmp_path):
    store=EventStore(tmp_path)
    event=create_event(store,1000)
    before=store.scene_incident(event_id=event["id"])
    store.enqueue_scene_tracking(event["id"],990,1045)
    incident=store.scene_incident(event_id=event["id"])
    assert incident["revision"] > before["revision"]
    assert incident["start_epoch"] == 990
    assert incident["last_epoch"] == 1045
    assert datetime.fromisoformat(incident["episodes"][0]["start_at"]).timestamp() == 990
    assert datetime.fromisoformat(incident["episodes"][0]["end_at"]).timestamp() == 1045
    with store._connect() as conn:
        episode=conn.execute("select * from scene_episodes").fetchone()
        parent=conn.execute("select * from scene_incidents").fetchone()
        assert (episode["start_epoch"],episode["end_epoch"])==(990,1045)
        assert (parent["start_epoch"],parent["end_epoch"])==(990,1045)
        assert json.loads(episode["coverage_json"])["state"]=="incomplete"


def _track_keys(store):
    with store._connect() as conn:
        rows = conn.execute("select payload_json from scene_observations").fetchall()
    keys = []
    states = []
    for (payload,) in rows:
        item = json.loads(payload)
        if item.get("scene_track_key"):
            keys.append(item["scene_track_key"])
            states.append(item.get("track_state"))
    return keys, states


def _saved_tracking(store, event_id):
    row = store.get(int(event_id))
    for item in json.loads(row["objects_json"] or "[]"):
        if isinstance(item, dict) and item.get("status") == "object_tracking":
            return item["object_tracking"]
    raise AssertionError("tracking summary missing")


def test_abutting_ten_second_files_keep_one_confirmed_track(tmp_path):
    from survng.app.tracking_frames import CameraFrameTimeline
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 999, 1002)
    job = store.claim_scene_tracking("gate", "seam-worker")
    recorder = Mock()
    recorder.ffmpeg_path = "/usr/bin/ffmpeg"
    recorder.timestamp_health.return_value = {
        ("gate", "main"): {"last_rollover_at": "1970-01-01T00:16:40+00:00"}
    }
    recorder.recording_rows_between.return_value = [
        {"start_epoch": 990.0, "end_epoch": 1000.0, "path": "/no/such/a.mp4"},
        {"start_epoch": 1000.0, "end_epoch": 1010.0, "path": "/no/such/b.mp4"},
    ]
    timeline = CameraFrameTimeline(
        camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        capture=Mock(), recorder=recorder, stop_event=threading.Event(), sample_fps=lambda: 2.0,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    timeline.remember(frame, 999.5, source="live")
    timeline.remember(frame, 1000.5, source="live")
    timeline.remember(frame, 1001.5, source="live")
    session = make_session(store)
    session.catchup_frame_provider = timeline.read_recorded_frames
    run_job(session, job, "seam-worker")
    keys, states = _track_keys(store)
    assert len(keys) == 3
    assert len(set(keys)) == 1
    assert "confirmed" in states
    scene = store.scene_incident(event_id=event["id"])
    assert scene["coverage"]["gaps"] == []


def test_open_file_pause_shorter_than_one_sample_keeps_the_track(tmp_path):
    from survng.app.object_track.types import TrackingFrameBatch
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1002)
    job = store.claim_scene_tracking("gate", "pause-worker")
    session = make_session(store)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def frames(start, end, fps, width, *, after_epoch=None):
        if after_epoch is None or after_epoch < 1000.2:
            return TrackingFrameBatch(((1000.0, frame),), 1000.0, "missing_recording", 1000.2)
        return TrackingFrameBatch(
            tuple((t, frame) for t in (1000.5, 1001.0, 1001.5, 1002.0) if start <= t <= end and t > after_epoch),
            1002.0,
        )

    session.catchup_frame_provider = frames
    run_job(session, job, "pause-worker")
    keys, _states = _track_keys(store)
    assert keys
    assert len(set(keys)) == 1
    assert store.scene_incident(event_id=event["id"])["coverage"]["gaps"] == []


def test_hole_longer_than_lost_timeout_starts_a_new_track(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1006)
    job = store.claim_scene_tracking("gate", "hole-worker")
    session = make_session(store)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    session.catchup_frame_provider = lambda start, end, fps, width, after_epoch=None: [
        (epoch, frame) for epoch in (1000.0, 1004.0) if start <= epoch <= end
    ]
    run_job(session, job, "hole-worker")
    keys, _states = _track_keys(store)
    assert len(keys) >= 2
    assert len({key.split(":")[2] for key in keys}) == 2
    gaps = store.scene_incident(event_id=event["id"])["coverage"]["gaps"]
    assert gaps and gaps[0]["reason"] == "missing_recording"


def test_reclaimed_scene_job_reseeds_the_confirmed_track(tmp_path):
    store = EventStore(tmp_path)
    event = create_event(store, 1000)
    store.enqueue_scene_tracking(event["id"], 1000, 1002)
    job = store.claim_scene_tracking("gate", "first-claim")
    session = make_session(store)

    def update(event_id, payload, objects):
        result = store.update_object_tracking(event_id, payload, objects)
        if payload["frames_processed"] >= 2:
            session._deadline = time.monotonic() - 1
        return result

    session.update_event = update
    run_job(session, job, "first-claim")
    resume = store.scene_track_resume(event["id"])
    assert resume["tracks"][0]["track_id"] == 1
    restarted = EventStore(tmp_path)
    resumed = restarted.claim_scene_tracking("gate", "second-claim")
    assert resumed is not None
    second = make_session(restarted)
    run_job(second, resumed, "second-claim", resume=restarted.scene_track_resume(event["id"]))
    keys, _states = _track_keys(restarted)
    assert keys
    assert len(set(keys)) == 1
    assert resume["scene_run_key"] in keys[0]
    saved = _saved_tracking(restarted, event["id"])
    assert datetime.fromisoformat(saved["analyzed_from"]).timestamp() <= 1000.5
    epochs = [sample[0] for sample in saved["tracks"][0]["box_history"]]
    assert min(epochs) <= 1000.5
    assert max(epochs) >= 1001.5


def test_scene_analysis_continues_after_recording_boundary(tmp_path):
    from survng.app.object_track.types import TrackingFrameBatch
    store=EventStore(tmp_path);event=create_event(store,1000)
    store.enqueue_scene_tracking(event['id'],1000,1002)
    job=store.claim_scene_tracking('gate','boundary-worker')
    session=make_session(store)
    frame=np.zeros((100,100,3),dtype=np.uint8)
    def frames(start,end,fps,width,*,after_epoch=None):
        if after_epoch is None or after_epoch<1001:
            return TrackingFrameBatch(((1000,frame),),1000,'recorder_epoch_changed',1001)
        return TrackingFrameBatch(tuple((t,frame) for t in (1001.5,1002) if t>after_epoch),1002)
    session.catchup_frame_provider=frames
    run_job(session,job,'boundary-worker')
    scene=store.scene_incident(event_id=event['id'])
    assert any(o['captured_at'].endswith('16:42+00:00') for s in scene['scene_objects'] for o in s['observations'])
    assert scene['coverage']['state']=='incomplete'
    assert scene['coverage']['gaps'][0]['reason']=='recorder_epoch_changed'
    assert EventStore(tmp_path).scene_incident(event_id=event['id'])['coverage']['gaps']==scene['coverage']['gaps']


def test_unavailable_old_window_is_a_gap_not_analyzed_empty_video(tmp_path):
    store=EventStore(tmp_path);event=create_event(store,1000)
    store.enqueue_scene_tracking(event['id'],990,999)
    job=store.claim_scene_tracking('gate','gap-worker')
    session=make_session(store)
    session.catchup_frame_provider=lambda *_args,**_kwargs: []
    run_job(session,job,'gap-worker')
    scene=store.scene_incident(event_id=event['id'])
    assert scene['coverage']['state']=='incomplete'
    assert scene['coverage']['gaps']
    with store._connect() as conn:
        job=conn.execute('select * from scene_analysis_jobs').fetchone()
        assert job['cursor_epoch']==999
        assert job['state']=='complete'
    assert scene['coverage']['analyzed_through'] is None


def test_restart_after_gap_does_not_call_skipped_frames_analyzed(tmp_path):
    store=EventStore(tmp_path);event=create_event(store,1000)
    store.enqueue_scene_tracking(event['id'],990,999)
    job=store.claim_scene_tracking('gate','first-worker')
    session=make_session(store);session.catchup_frame_provider=lambda *_args,**_kwargs: []
    run_job(session,job,'first-worker')
    restarted=EventStore(tmp_path)
    restarted.enqueue_scene_tracking(event['id'],990,1002)
    next_job=restarted.claim_scene_tracking('gate','second-worker')
    assert next_job['cursor_epoch']==999
    assert next_job['analyzed_epoch'] is None
    session=make_session(restarted);session.catchup_frame_provider=lambda *_args,**_kwargs: []
    run_job(session,next_job,'second-worker')
    scene=restarted.scene_incident(event_id=event['id'])
    assert scene['coverage']['analyzed_through'] is None
    assert scene['coverage']['gaps'][0]['start_at']
    assert min(g['start_epoch'] for g in scene['coverage']['gaps'])==990
    assert max(g['end_epoch'] for g in scene['coverage']['gaps'])==1002


def test_expired_decode_budget_does_not_skip_unread_recording(tmp_path):
    store=EventStore(tmp_path);event=create_event(store,1000)
    store.enqueue_scene_tracking(event['id'],1000,1002)
    job=store.claim_scene_tracking('gate','budget-worker')
    session=make_session(store)
    def exhausted(*_args,**_kwargs):
        session._deadline=time.monotonic()-1
        return []
    session.catchup_frame_provider=exhausted
    run_job(session,job,'budget-worker')
    with store._connect() as conn:
        retained=conn.execute('select * from scene_analysis_jobs').fetchone()
        assert retained['cursor_epoch'] is None
        assert json.loads(retained['coverage_gaps_json'])==[]
        assert retained['state']=='queued'


def test_rewound_preroll_progress_is_not_replaced_by_old_analysis_time(tmp_path):
    store=EventStore(tmp_path);event=create_event(store,1000)
    store.enqueue_scene_tracking(event['id'],990,1002)
    job=store.claim_scene_tracking('gate','rewound-worker')
    store.update_object_tracking(event['id'],{
        'scene_analysis_job':{'episode_id':job['episode_id'],'lease_owner':'rewound-worker'},
        'state':'interrupted','completion_reason':'processing_budget_exhausted',
        'analyzed_through':datetime.fromtimestamp(1000,timezone.utc).isoformat(),
        'scene_cursor_epoch':989.75,
    })
    with store._connect() as conn:
        retained=conn.execute('select * from scene_analysis_jobs').fetchone()
        assert retained['cursor_epoch'] is None
        assert retained['analyzed_epoch']==1000

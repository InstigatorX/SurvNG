"""Acquisition, activity, membership and notification are separate contracts."""
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from survng.app.events import EventStore
from survng.app.motion_pipeline.decision_handler import MotionDecisionHandler
from survng.app.motion_pipeline.object_detection import RecordedDetectionResult, _RecordedDetectionSample, _scene_sample_records
from survng.app.motion_pipeline.scene_evidence import scene_observation


def detection(x=20, **extra):
    return {"label":"person", "confidence":.279, "incident_eligible":False,
            "temporal_candidate_threshold":.25, "detection_frame_width":160, "detection_frame_height":100,
            "box":{"x1":x,"y1":20,"x2":x+20,"y2":70}, **extra}


def picture(x=20):
    image=np.zeros((100,160,3),dtype=np.uint8)
    image[20:70,x:x+20]=220
    return image


def processor(store, result, root):
    def snapshot(frame, at):
        target=root / 'snapshots' / (str(at.timestamp())+'.webp')
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b'captured evidence')
        return str(target)
    return MotionDecisionHandler(camera_id='gate',events=store,detection_provider=lambda _:result,
        snapshot_writer=snapshot,object_serializer=json.dumps)


def acquire(store, root):
    raw=detection()
    observation=scene_observation(raw,captured_at_epoch=1000,frame_source='live_discovery')
    result=RecordedDetectionResult(frame=picture(),objects=[raw,{'status':'scene_observations','observations':[observation]}],
        recording_path='',timings_ms={},frame_captured_at_epoch=1000,frame_source='live_discovery')
    return processor(store,result,root).refine('scene/discovery','',datetime.fromtimestamp(1000,timezone.utc),
        {'scene_discovery':True},existing_event_id=None)


def test_candidate_survives_without_an_event_incident_or_alert_and_restart(tmp_path):
    store=EventStore(tmp_path)
    outcome=acquire(store,tmp_path)
    assert outcome.event_id is None
    assert outcome.rejection_reason=='scene_activity_pending'
    with store._connect() as conn:
        assert conn.execute('select count(*) from events').fetchone()[0]==0
        assert conn.execute('select count(*) from scene_incidents').fetchone()[0]==0
        assert conn.execute('select count(*) from acquired_observations').fetchone()[0]==1
        assert conn.execute('select count(*) from scene_candidate_jobs').fetchone()[0]==1
    assert store.scene_pending_notifications()==[]
    restarted=EventStore(tmp_path)
    review=restarted.scene_observation_reviews(start_epoch=900,end_epoch=1100)
    assert review['total']==1 and review['items'][0]['status']=='pending'
    assert review['items'][0]['observations'][0]['snapshot_available']


def confirmation_result():
    frames=[_RecordedDetectionSample(t,picture(x),[detection(x)],'main.mp4',requested_offset=t,exact_timestamp=True)
            for t,x in [(0,20),(1,40),(2,60)]]
    observations=[scene_observation(detection(x),captured_at_epoch=1000+t,frame_source='recorded_main',scene_track_key='physical-person')
                  for t,x in [(0,20),(1,40),(2,60)]]
    samples=_scene_sample_records(frames,observations,1000,'gate',(0,1,2))
    result=RecordedDetectionResult(frame=picture(60),objects=[detection(60),{'status':'scene_observations','observations':observations,'samples':samples}],
        recording_path='main.mp4',timings_ms={},frame_captured_at_epoch=1002,frame_source='recorded_main')
    return result


def test_supported_activity_attaches_original_context_and_does_not_use_alert_rules(tmp_path):
    store=EventStore(tmp_path); acquire(store,tmp_path)
    with store._connect() as conn:
        conn.execute('update scene_candidate_jobs set available_at_epoch=0')
    job=store.claim_scene_candidate('gate',lease_owner='test')
    result=confirmation_result()
    q={'scene_confirmation':True,'scene_seed_sample_ids':job['seed_sample_ids'],
       'scene_candidate_id':job['id'],'scene_candidate_lease_owner':'test','scene_candidate_lease_token':job['lease_token'],
       'detection_intent_id':'verification:'+job['id']}
    outcome=processor(store,result,tmp_path).refine('scene/confirmation','',datetime.fromtimestamp(1000,timezone.utc),q,existing_event_id=None)
    assert outcome.event_id is not None
    incident=store.scene_incident(event_id=outcome.event_id)
    assert incident['establishment']['status']=='established'
    assert any(item['kind']=='changed_position' for item in incident['activity'])
    assert any(o['frame_source']=='live_discovery' for s in incident['scene_objects'] for o in s['observations'])
    assert not any(a['eligible'] for a in incident['alert_decisions'])
    before=incident['episodes'][0]['last_activity_at']
    store.record_scene_observations(outcome.event_id,[detection(110,confidence=.99,captured_at_epoch=1030)])
    assert store.scene_incident(event_id=outcome.event_id)['episodes'][0]['last_activity_at']==before
    assert store.finish_scene_candidate(job['id'],lease_owner='test',lease_token=job['lease_token'],decision_id=outcome.scene_activity_decision_id)
    assert store.scene_observation_review(job['id'])['status']=='established'
    assert store.scene_observation_review(job['id'])['review_images'][0]['source']=='recorded_main'


def test_restart_after_admission_recovers_without_inference_or_duplicate_event(tmp_path):
    from tests.test_motion_incidents import _service
    from survng.app.motion_pipeline.decision_handler import MotionDecisionOutcome
    store=EventStore(tmp_path); acquire(store,tmp_path)
    with store._connect() as conn:
        conn.execute('update scene_candidate_jobs set available_at_epoch=0')
    job=store.claim_scene_candidate('gate',lease_owner='lost-worker')
    q={'scene_confirmation':True,'scene_seed_sample_ids':job['seed_sample_ids'],
       'scene_candidate_id':job['id'],'scene_candidate_lease_owner':'lost-worker','scene_candidate_lease_token':job['lease_token'],
       'detection_intent_id':'verification:'+job['id']+':1'}
    outcome=processor(store,confirmation_result(),tmp_path).refine('scene/confirmation','',datetime.fromtimestamp(1000,timezone.utc),q,existing_event_id=None)
    with store._connect() as conn:
        conn.execute('update scene_candidate_jobs set lease_expires_at_epoch=0')
    restarted=EventStore(tmp_path)
    service,decision,tracking,_,_= _service(MotionDecisionOutcome(None,'',False),refinement_store=restarted)
    service.scene_analysis_enabled=lambda:True
    assert service._run_scene_candidate()
    decision.refine.assert_not_called()
    tracking.start.assert_called_once()
    assert restarted.scene_candidate_admission(job['id'],1)['id']==outcome.event_id
    assert restarted.scene_observation_review(job['id'])['status']=='established'
    with restarted._connect() as conn:
        assert conn.execute('select count(*) from events').fetchone()[0]==1
        assert conn.execute('select state from scene_candidate_jobs').fetchone()[0]=='complete'


def test_coalesced_generation_admits_later_evidence_without_identity_collision(tmp_path):
    from tests.test_motion_incidents import _service
    from survng.app.motion_pipeline.decision_handler import MotionDecisionOutcome
    store=EventStore(tmp_path); acquire(store,tmp_path)
    with store._connect() as conn:
        conn.execute('update scene_candidate_jobs set available_at_epoch=0')
    job=store.claim_scene_candidate('gate',lease_owner='first')
    import time
    store.acquire_scene_sample(sample_id='later-context',camera_id='gate',captured_epoch=1003,source='live_discovery',status='complete',observations=[detection(80)],
        request_confirmation={'start_epoch':998,'end_epoch':1008,'deadline_epoch':time.time()+300})
    q={'scene_confirmation':True,'scene_seed_sample_ids':job['seed_sample_ids'],
       'scene_candidate_id':job['id'],'scene_candidate_lease_owner':'first','scene_candidate_lease_token':job['lease_token'],
       'detection_intent_id':'scene-confirmation:'+job['id']+':1'}
    outcome=processor(store,confirmation_result(),tmp_path).refine('scene/confirmation','',datetime.fromtimestamp(1000,timezone.utc),q,existing_event_id=None)
    assert store.finish_scene_candidate(job['id'],lease_owner='first',lease_token=job['lease_token'],decision_id=outcome.scene_activity_decision_id)
    service,_,_,_,_= _service(MotionDecisionOutcome(None,'',False),refinement_store=store)
    service.scene_analysis_enabled=lambda:True
    service.decision_processor=processor(store,confirmation_result(),tmp_path)
    assert service._run_scene_candidate()
    second=store.scene_candidate_admission(job['id'],2)
    assert second is not None and second['id']!=outcome.event_id
    assert store.scene_incident(event_id=second['id'])['incident_id']==store.scene_incident(event_id=outcome.event_id)['incident_id']


def test_historical_reclassification_preserves_inventory_and_links_without_alerts(tmp_path):
    store=EventStore(tmp_path)
    event=store.add_event('gate','motion',created_at=datetime.fromtimestamp(1000,timezone.utc).isoformat(),objects_json=json.dumps([detection()]))
    original=store.scene_incident(event_id=event['id'])
    with store._connect() as conn:
        conn.execute("update events set topic='scene/discovery',objects_json=? where id=?",
            (json.dumps([detection(),{'status':'motion_qualification','motion_qualification':{'scene_discovery':True}}]),event['id']))
        conn.execute('delete from scene_activity_admissions')
        conn.execute('delete from scene_event_establishment')
        conn.execute('delete from scene_notification_outbox')
        before=[tuple(r) for r in conn.execute('select id,object_id,payload_json from scene_observations order by id')]
    restarted=EventStore(tmp_path)
    migrated=restarted.scene_incident(event_id=event['id'])
    assert migrated['incident_id']==original['incident_id']
    assert migrated['state']=='unconfirmed'
    assert restarted.scene_incident_facets()["camera_ids"]==[]
    assert restarted.scene_incident_facets()["labels"]==[]
    assert migrated['establishment']['reason']=='historical_activity_unverified'
    assert restarted.scene_observation_reviews(start_epoch=990,end_epoch=1010)['total']==1
    assert restarted.scene_pending_notifications()==[]
    with restarted._connect() as conn:
        assert [tuple(r) for r in conn.execute('select id,object_id,payload_json from scene_observations order by id')]==before
    later=restarted.add_event('gate','motion',created_at=datetime.fromtimestamp(1010,timezone.utc).isoformat(),objects_json=json.dumps([detection()]))
    assert restarted.scene_incident(event_id=later['id'])['incident_id']!=migrated['incident_id']
    assert restarted.scene_incident(event_id=event['id'])['state']=='unconfirmed'


@pytest.mark.parametrize('legacy_sample_ids', [False, True])
def test_reanalysis_of_same_frame_retains_new_detector_geometry(tmp_path, legacy_sample_ids):
    store=EventStore(tmp_path)
    first=confirmation_result()
    event=store.add_event('gate','motion',created_at=datetime.fromtimestamp(1000,timezone.utc).isoformat(),objects_json=json.dumps(first.objects))
    observations=[scene_observation(detection(x+1),captured_at_epoch=1000+t,frame_source='recorded_main',scene_track_key='physical-person')
                  for t,x in [(0,20),(1,40),(2,60)]]
    frames=[_RecordedDetectionSample(t,picture(x),[detection(x+1)],'main.mp4',requested_offset=t,exact_timestamp=True)
            for t,x in [(0,20),(1,40),(2,60)]]
    samples=_scene_sample_records(frames,observations,1000,'gate',(0,1,2))
    if legacy_sample_ids:
        originals=next(item['samples'] for item in first.objects if item.get('status')=='scene_observations')
        for sample, original in zip(samples, originals):
            sample['id']=original['id']
    tracking={'scene_observations':observations,'scene_samples':samples,'state':'complete'}
    store.update_object_tracking(event['id'],tracking)
    with store._connect() as conn:
        boxes=[json.loads(r[0])['box']['x1'] for r in conn.execute("select payload_json from acquired_observations where camera_id='gate'")]
        counts=tuple(conn.execute(f'select count(*) from {table}').fetchone()[0]
                     for table in ('acquired_samples','acquired_observations','scene_observations'))
        assert conn.execute("select count(distinct object_id) from scene_observations where track_key=?",
                            (f"{event['id']}:physical-person",)).fetchone()[0]==1
    assert {20,21,40,41,60,61} <= set(boxes)
    store.update_object_tracking(event['id'],tracking)
    with store._connect() as conn:
        assert counts==tuple(conn.execute(f'select count(*) from {table}').fetchone()[0]
                             for table in ('acquired_samples','acquired_observations','scene_observations'))
        assert conn.execute('pragma foreign_key_check').fetchall()==[]


def test_context_projection_preserves_acquired_observation_timestamp(tmp_path):
    store=EventStore(tmp_path)
    capture=1790613107.0975204
    observed_at=round(capture,6)
    observation=scene_observation(detection(),captured_at_epoch=observed_at,frame_source='live_discovery')
    sample=store.acquire_scene_sample(sample_id='original-capture',camera_id='gate',captured_epoch=capture,
        source='live_discovery',status='complete',observations=[observation])
    event=store.add_event('gate','motion',created_at=datetime.fromtimestamp(capture-1,timezone.utc).isoformat())
    store.associate_scene_samples([sample['id']])
    with store._connect() as conn:
        retained=conn.execute('select source_observation_id,captured_epoch from scene_observations where event_id=?',
                              (event['id'],)).fetchall()
        assert [(r[0],r[1]) for r in retained]==[(sample['observation_ids'][0],observed_at)]
        assert conn.execute('pragma foreign_key_check').fetchall()==[]


def test_context_projection_is_paged_by_a_durable_obligation(tmp_path):
    store = EventStore(tmp_path)
    for index in range(205):
        store.acquire_scene_sample(
            sample_id=f"context-{index:03d}", camera_id="gate",
            captured_epoch=1000, source="live_discovery", status="complete",
            observations=[],
        )
    event = store.add_event(
        "gate", "motion",
        created_at=datetime.fromtimestamp(1000, timezone.utc).isoformat(),
        objects_json=json.dumps([detection(confidence=.8, incident_eligible=True)]),
    )
    incident = store.scene_incident(event_id=event["id"])
    episode_id = incident["episodes"][0]["id"]
    with store._connect() as connection:
        self_count = connection.execute(
            "select count(*) from acquired_sample_episodes where episode_id=?", (episode_id,),
        ).fetchone()[0]
        jobs = connection.execute("select count(*) from scene_context_projection_jobs").fetchone()[0]
    assert self_count == 200
    assert jobs == 1

    assert store.project_pending_scene_context() > 0
    with store._connect() as connection:
        assert connection.execute(
            "select count(*) from acquired_sample_episodes where episode_id=?", (episode_id,),
        ).fetchone()[0] == 206
        assert connection.execute("select count(*) from scene_context_projection_jobs").fetchone()[0] == 0


def test_failed_reanalysis_preserves_original_successful_observations(tmp_path):
    store=EventStore(tmp_path)
    first=confirmation_result()
    event=store.add_event('gate','motion',created_at=datetime.fromtimestamp(1000,timezone.utc).isoformat(),
                          objects_json=json.dumps(first.objects))
    failed=_scene_sample_records([_RecordedDetectionSample(0,None,[],'main.mp4',requested_offset=0)],
                                 [],1000,'gate',(0,))
    store.update_object_tracking(event['id'],{'scene_observations':[],'scene_samples':failed,'state':'complete'})
    with store._connect() as conn:
        assert conn.execute('select count(*) from acquired_observations where camera_id=?',('gate',)).fetchone()[0]==3
        assert conn.execute('select status from acquired_samples where id=?',(failed[0]['id'],)).fetchone()[0]=='failed'
        assert conn.execute('pragma foreign_key_check').fetchall()==[]


@pytest.mark.parametrize("local,background", [(0.082738, 0.072881), (0.085119, 0.079609)])
def test_confirmation_with_background_level_change_keeps_evidence_without_incident(tmp_path, local, background):
    from tests.test_scene_zone_admission import policy, zone

    store = EventStore(tmp_path)
    result = confirmation_result()
    batch = next(item for item in result.objects if item.get("status") == "scene_observations")
    for sample in batch["samples"]:
        for witness in sample["metadata"]["activity_witnesses"]:
            witness.update(kind="localized_motion", witness_version=1,
                           local_change_fraction=local, background_change_fraction=background,
                           normalized_displacement=0.0007)
        for observation in sample["observations"]:
            observation.update(activity_role="indeterminate", confidence=0.93)
    outcome = processor(store, result, tmp_path).refine(
        "scene/confirmation", "", datetime.fromtimestamp(1000, timezone.utc),
        {"scene_confirmation": True,
         "establishment_zone_policy": policy(zone("Entry", "incident", 0, 1), confidence_threshold=0.7)},
        existing_event_id=None,
    )
    assert outcome.event_id is None
    assert outcome.rejection_reason == "scene_activity_unsupported"
    with store._connect() as conn:
        assert conn.execute("select count(*) from acquired_observations").fetchone()[0] == 3
        assert conn.execute("select count(*) from scene_incidents").fetchone()[0] == 0
        assert conn.execute("select count(*) from scene_notification_outbox").fetchone()[0] == 0
    restarted = EventStore(tmp_path)
    with restarted._connect() as conn:
        assert conn.execute("select count(*) from scene_incidents").fetchone()[0] == 0

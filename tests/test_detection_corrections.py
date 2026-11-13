"""Human-review storage and API boundary regressions; no live inference needed."""
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import cv2
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from survng.app.config import AppConfig
from survng.app.events import EventStore
from survng.app.correction_routes import create_correction_router
from survng.app.event_store.corrections import ReviewConflict
from survng.app.manager_access import ManagerAccessCoordinator
from survng.app.security import ApiPrincipal, required_api_scope
from survng.app.training_routes import TrainingRouteDependencies, create_training_router


@pytest.fixture
def context(tmp_path):
    store = EventStore(tmp_path)
    config = AppConfig()
    manager = SimpleNamespace(events=store, storage_dir=tmp_path)
    deps = TrainingRouteDependencies(get_config=lambda:config,get_manager=lambda:manager,
        manager_lock=threading.RLock(),manager_access=ManagerAccessCoordinator())
    app = FastAPI()
    principal = [ApiPrincipal('test','Test',frozenset({'admin'}))]
    @app.middleware('http')
    async def auth(request, call_next):
        request.scope['survng_principal'] = principal[0]
        return await call_next(request)
    app.include_router(create_correction_router(deps)); app.include_router(create_training_router(deps))
    image = tmp_path/'snapshots'/'frame.jpg'; image.parent.mkdir()
    assert cv2.imwrite(str(image), np.full((50,100,3),120,np.uint8))
    objects = [{'label':'person','confidence':.3,'box':{'x1':10,'y1':5,'x2':50,'y2':25},
                'detection_frame_width':100,'detection_frame_height':50,'incident_eligible':False},
               {'label':'dog','confidence':.9,'box':{'x1':55,'y1':10,'x2':90,'y2':40},
                'detection_frame_width':100,'detection_frame_height':50,'incident_eligible':True}]
    event = store.add_event(camera_id='gate',kind='motion',snapshot_path=str(image),objects_json=json.dumps(objects),created_at='2026-11-12T12:00:00+00:00')
    return SimpleNamespace(store=store, client=TestClient(app), principal=principal, event=event, image=image, tmp=tmp_path)


def payload(c, **overrides):
    evidence = c.store.review_evidence('event',c.event['id'])
    return {'evidence_revision':evidence['revision'],'idempotency_key':'request-0001','verdict':'missed',
            'object_key':'object:0',**overrides}


def create(c, **overrides):
    return c.store.create_detection_review('event',c.event['id'],payload(c,**overrides),'test')


def training(c, **kwargs):
    return c.client.get('/api/training/samples',params={'start_at':'2026-11-12T00:00:00Z','end_at':'2026-11-13T00:00:00Z',**kwargs})


def audit(c):
    return c.store.add_motion_audit(event_id=None, camera_id='gate',snapshot_path=str(c.image),created_at='2026-11-12T12:00:00+00:00',mode='shadow',sensitivity='normal',score=.2,threshold=.5,reason='no_object',object_detected=False,trigger_count=1,features={})


def test_source_unchanged_retention_and_review_export(context):
    c=context
    before=c.store.get(c.event['id'])
    row=create(c)
    assert c.store.get(c.event['id']) == before
    assert row['source_kind']=='hidden_candidate'
    assert create(c)['id']==row['id']
    result=training(c).json()
    annotations=result['samples'][0]['annotations']
    assert {a['annotation_state'] for a in annotations} == {'human_reviewed','model_generated'}
    result=training(c,review='reviewed').json()
    assert len(result['samples'][0]['annotations'])==1
    assert result['samples'][0]['annotations'][0]['confidence'] is None
    c.image.unlink()
    with c.store._connect() as conn:
        conn.execute('delete from events where id=?',(c.event['id'],))
    assert training(c,review='reviewed').json()['count']==1
    response=c.client.get(f"/api/detection-corrections/assets/{row['asset_id']}")
    assert response.status_code==200 and response.headers['content-type']=='image/jpeg'
    assert c.store.create_detection_review('event',c.event['id'],{'evidence_revision':row['asset']['source_revision'],'idempotency_key':'request-0001','verdict':'missed','object_key':'object:0'},'test')['id']==row['id']


def test_stale_evidence_and_same_key_different_request(context):
    c=context; values=payload(c)
    with c.store._connect() as conn:
        conn.execute('update events set evidence_revision=evidence_revision+1 where id=?',(c.event['id'],))
    with pytest.raises(ReviewConflict): c.store.create_detection_review('event',c.event['id'],values,'test')
    create(c)
    with pytest.raises(ReviewConflict): create(c, verdict='not_object')


def test_conflict_supersession_and_revoke(context):
    c=context; first=create(c)
    with pytest.raises(ReviewConflict): create(c,idempotency_key='request-0002',verdict='not_object')
    second=create(c,idempotency_key='request-0002',verdict='not_object',supersedes={str(first['id']):1})
    assert second['verdict']=='not_object'
    assert training(c,review='reviewed',eligibility='eligible',minimum_confidence=.99).json()['samples'][0]['annotations'][0]['review_verdict']=='not_object'
    with pytest.raises(ReviewConflict): c.store.revoke_detection_review(first['id'],1,'test')
    c.store.revoke_detection_review(second['id'],1,'test')
    assert training(c,review='reviewed').json()['count']==0
    with c.store._connect() as conn:
        assert conn.execute('select count(*) from detection_review_audit').fetchone()[0]==4


def test_parallel_duplicate_and_conflicting_review(context):
    c=context; values=payload(c)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(lambda _:c.store.create_detection_review('event',c.event['id'],values,'test'),range(2)))
    assert rows[0]['id']==rows[1]['id']
    with c.store._connect() as conn:
        assert conn.execute('select count(*) from detection_reviews').fetchone()[0]==1


def test_negative_conflict_drawn_and_audit_retention(context):
    c=context; item=audit(c); aid=item['id']
    e=c.store.review_evidence('motion_audit',aid)
    negative=c.client.put(f'/api/motion-audits/{aid}/review',json={'evidence_revision':e['revision'],'idempotency_key':'negative-0001','verdict':'verified_negative'})
    assert negative.status_code==200,negative.text
    row=negative.json()
    result=training(c,review='reviewed',sources='motion_audit').json()['samples'][0]
    assert result['verified_negative'] is True and result['assumed_negative'] is False
    values={'evidence_revision':e['revision'],'idempotency_key':'drawn-0001','verdict':'missed','source':'drawn','corrected_label':'raccoon','box':[.1,.1,.5,.5]}
    assert c.client.post(f'/api/motion-audits/{aid}/corrections',json=values).status_code==409
    values['supersedes']={str(row['id']):1}
    response=c.client.post(f'/api/motion-audits/{aid}/corrections',json=values)
    assert response.status_code==200,response.text
    sample=training(c,review='reviewed',sources='motion_audit',sample_kinds='annotated').json()['samples'][0]
    assert sample['sample_kind']=='annotated'
    assert sample['annotations'][0]['annotation_source']=='operator_drawn'
    assert c.client.get('/api/detection-corrections/review-queue').json()['items']==[]
    with c.store._connect() as conn: conn.execute('delete from motion_audits where id=?',(aid,))
    assert training(c,review='reviewed',sources='motion_audit').json()['count']==1


def test_http_admin_gate_and_validation(context):
    c=context
    c.principal[0]=ApiPrincipal('viewer','Viewer',frozenset({'read'}))
    assert c.client.post(f"/api/events/{c.event['id']}/corrections",json=payload(c)).status_code==403
    assert c.client.get(f"/api/events/{c.event['id']}/review-evidence").json()['can_review'] is False
    assert required_api_scope('POST',f"/api/events/{c.event['id']}/corrections")=='admin'
    assert required_api_scope('GET','/api/detection-corrections/assets/1')=='read'
    c.principal[0]=ApiPrincipal('test','Test',frozenset({'admin'}))
    response=c.client.post(f"/api/events/{c.event['id']}/corrections",json=payload(c,source='drawn',box=[0,0,2,1],corrected_label='cat'))
    assert response.status_code==422
    assert c.client.get('/api/detection-corrections/status').json()['mode']=='off'


def test_review_pagination_and_legacy_response_stability(context):
    c=context; baseline=training(c).content
    assert b'correction_id' not in baseline and b'verified_negative' not in baseline
    first=create(c)
    c.store.revoke_detection_review(first['id'],1,'test')
    assert training(c).content==baseline
    row=create(c,idempotency_key='request-0002')
    page=training(c,limit=1).json()
    assert page['next_cursor']=='legacy:'
    final=training(c,limit=1,cursor=page['next_cursor']).json()
    assert final['count']==0 and final['next_cursor']==''
    assert training(c,review='reviewed',cursor='legacy:').status_code==422
    assert row['id'] != first['id']


def test_own_image_keeps_coordinates_after_cover_replaced(context):
    c=context; row=create(c,source='drawn',box=[.1,.2,.3,.6],corrected_label='raccoon')
    replacement=c.tmp/'new.jpg'; assert cv2.imwrite(str(replacement),np.zeros((100,200,3),np.uint8))
    with c.store._connect() as conn: conn.execute('update events set snapshot_path=?,evidence_revision=evidence_revision+1 where id=?',(str(replacement),c.event['id']))
    result=training(c,review='reviewed').json()['samples'][0]
    assert result['image']['width']==100 and result['image']['height']==50
    assert result['annotations'][0]['bbox_xyxy']==[10,10,30,30]
    assert c.client.get(f"/api/detection-corrections/assets/{row['asset_id']}?review_id={row['id']}").status_code==200


def test_orphan_cleanup_and_failed_commit(context, monkeypatch):
    c=context
    original=c.store._review_source; calls=[]
    def changed(*args,**kwargs):
        result=original(*args,**kwargs); calls.append(1)
        if len(calls)==3: # payload read, create evidence read, commit recheck
            return result[0], 'changed', result[2],result[3]
        return result
    monkeypatch.setattr(c.store,'_review_source',changed)
    with pytest.raises(ReviewConflict): create(c)
    with c.store._connect() as conn:
        assert conn.execute('select count(*) from detection_reviews').fetchone()[0]==0
    import os,time
    files=[p for p in (c.tmp/'corrections').iterdir() if p.name != '.review.lock']; assert files
    for f in files: os.utime(f,(time.time()-90000,time.time()-90000))
    with c.store._connect() as conn: c.store._review_cleanup_orphans(conn)
    assert [p.name for p in (c.tmp/'corrections').iterdir()]==['.review.lock']


def test_same_image_new_revision_still_requires_supersession(context):
    c=context; first=create(c)
    with c.store._connect() as conn:
        conn.execute('update events set evidence_revision=evidence_revision+1 where id=?',(c.event['id'],))
    with pytest.raises(ReviewConflict):
        create(c,verdict='not_object',idempotency_key='request-0002')
    second=create(c,verdict='not_object',idempotency_key='request-0002',supersedes={str(first['id']):1})
    assert second['asset_id'] != first['asset_id']
    assert training(c,review='reviewed').json()['count']==1


def test_quota_and_audit_image_revision(context,monkeypatch):
    import survng.app.event_store.corrections as module
    c=context
    monkeypatch.setattr(module,'MAX_REVIEW_BYTES',1)
    with pytest.raises(ValueError,match='quota'): create(c)
    monkeypatch.setattr(module,'MAX_REVIEW_BYTES',2*1024**3)
    item=audit(c); aid=item['id']
    evidence=c.store.review_evidence('motion_audit',aid)
    # A coalesced audit may keep its ID while replacing the snapshot metadata.
    with c.store._connect() as conn:
        conn.execute('update motion_audits set features_json=? where id=?',('{"changed":true}',aid))
    response=c.client.put(f'/api/motion-audits/{aid}/review',json={'evidence_revision':evidence['revision'],'idempotency_key':'audit-stale-1','verdict':'verified_negative'})
    assert response.status_code==409


def test_cross_store_concurrent_idempotency(context):
    c=context; other=EventStore(c.tmp); values=payload(c)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(lambda store:store.create_detection_review('event',c.event['id'],values,'test'),[c.store,other]))
    assert rows[0]['id']==rows[1]['id']


def test_human_labels_are_not_mapped_onto_recording_frames(context):
    c=context
    assert training(c,review='reviewed',sources='motion_audit',image_source='main_recording').status_code==422


def test_review_detail_audit_and_crop_dimensions(context):
    c=context; row=create(c)
    result=c.client.get(f"/api/detection-corrections/{row['id']}")
    assert result.status_code==200
    assert result.json()['audit'][0]['action']=='create'
    response=c.client.get(f"/api/detection-corrections/assets/{row['asset_id']}?review_id={row['id']}")
    image=cv2.imdecode(np.frombuffer(response.content,np.uint8),cv2.IMREAD_COLOR)
    assert image.shape[:2]==(20,40)


def test_same_image_reviews_group_across_metadata_revisions(context):
    c=context; create(c,verdict='not_object')
    with c.store._connect() as conn:
        conn.execute('update events set evidence_revision=evidence_revision+1 where id=?',(c.event['id'],))
    create(c,verdict='wrong_label',object_key='object:1',corrected_label=' CAT ',idempotency_key='request-0002')
    samples=training(c).json()['samples']
    assert len(samples)==1
    annotations=samples[0]['annotations']
    assert len(annotations)==2
    assert {a['annotation_state'] for a in annotations}=={'human_reviewed'}
    assert {a['label'] for a in annotations}=={'person','cat'}


def test_off_cover_observation_uses_own_image_and_raw_identity(context):
    c=context
    own=c.tmp/'snapshots'/'support.jpg'
    assert cv2.imwrite(str(own),np.full((80,160,3),70,np.uint8))
    observation={'label':'cat','confidence':.3,'incident_eligible':False,'snapshot_visible':False,
                 'snapshot_path':str(own),'detection_frame_width':160,'detection_frame_height':80,
                 'box':{'x1':16,'y1':8,'x2':80,'y2':40},'captured_at_epoch':1794484801.0,'frame_source':'recorded_main'}
    c.store.record_scene_observations(c.event['id'],[observation])
    with c.store._connect() as conn:
        row=conn.execute("select id,source_observation_id from scene_observations where event_id=? and json_extract(payload_json,'$.label')='cat'",(c.event['id'],)).fetchone()
    assert row is not None
    evidence=c.store.review_evidence('event',c.event['id'],row['id'])
    assert evidence['width']==160 and evidence['height']==80
    assert evidence['objects'][0]['box']==[.1,.1,.5,.5]
    saved=c.store.create_detection_review('event',c.event['id'],{'evidence_revision':evidence['revision'],'idempotency_key':'offcover-review','observation_id':row['id'],'object_key':row['id'],'verdict':'missed'},'test')
    assert saved['observation_id']==(row['source_observation_id'] or row['id'])
    assert saved['asset']['width']==160


def test_snapshot_retention_and_storage_cleanup_leave_review_assets(context):
    import time,os
    from survng.app.recorder import Recorder
    from survng.app.storage_maintenance import StorageReconciler
    c=context; row=create(c)
    _,owned=c.store.review_asset(row['asset_id'])
    old=time.time()-36000
    os.utime(owned,(old,old))
    c.store.apply_snapshot_retention(time.time()+86400,100)
    assert owned.is_file()
    recorder=Recorder('ffmpeg',c.tmp,segment_seconds=10,index_dir=c.tmp/'index')
    StorageReconciler(c.tmp,c.store.db_path,recorder).run(apply=True,full=True,reclaim_orphans=True,reclaim_dry_run=False,orphan_min_age_seconds=600)
    assert owned.is_file()
    assert training(c,review='reviewed').json()['count']==1


def test_two_reviewed_images_paginate_without_loss(context):
    c=context; first=create(c)
    replacement=c.tmp/'replacement.png'; assert cv2.imwrite(str(replacement),np.zeros((50,100,3),np.uint8))
    with c.store._connect() as conn:
        conn.execute('update events set snapshot_path=?,evidence_revision=evidence_revision+1 where id=?',(str(replacement),c.event['id']))
    second=create(c,idempotency_key='second-image')
    first_page=training(c,review='reviewed',limit=1).json()
    assert first_page['samples'][0]['annotations'][0]['correction_id']==second['id']
    next_page=training(c,review='reviewed',limit=1,cursor=first_page['next_cursor']).json()
    assert next_page['samples'][0]['annotations'][0]['correction_id']==first['id']
    assert next_page['next_cursor']==''


def test_unavailable_volume_does_not_create_review_directory(context,monkeypatch):
    c=context
    monkeypatch.setattr(c.store,'_snapshot_verdict',lambda path:'unavailable')
    response=c.client.post(f"/api/events/{c.event['id']}/corrections",json=payload(c))
    assert response.status_code==503
    assert not (c.tmp/'corrections').exists()


def test_queue_uses_latest_completed_ai_review_and_paginates(context):
    c = context
    older = audit(c)
    with c.store._connect() as conn:
        conn.execute("update motion_audits set created_at='2026-11-12T11:00:00+00:00' where id=?", (older['id'],))
    newer = audit(c)
    assert older['id'] != newer['id']
    review = c.store.create_motion_ai_review('gate', 2)
    c.store.update_motion_ai_review(review['id'], status='completed', result={
        'samples': [{'kind': 'motion_decision', 'audit_id': older['id'],
                     'verdict': 'likely_miss'}]})
    first = c.client.get('/api/detection-corrections/review-queue', params={'limit': 1}).json()
    assert first['items'][0]['id'] == older['id']
    assert first['items'][0]['probable_miss'] is True
    second = c.client.get('/api/detection-corrections/review-queue',
                          params={'limit': 1, 'cursor': first['next_cursor']}).json()
    assert second['items'][0]['id'] == newer['id']
    assert second['items'][0]['probable_miss'] is False
    # A pending review does not hide the last completed analysis.
    latest = c.store.create_motion_ai_review('gate', 2)
    assert c.client.get('/api/detection-corrections/review-queue').json()['items'][0]['id'] == older['id']
    c.store.update_motion_ai_review(latest['id'], status='completed', result={
        'samples': [{'kind': 'motion_decision', 'audit_id': older['id'],
                     'verdict': 'consistent'}]})
    items = c.client.get('/api/detection-corrections/review-queue').json()['items']
    assert [item['id'] for item in items] == [newer['id'], older['id']]
    assert not any(item['probable_miss'] for item in items)
    assert c.store.detection_reviews() == []

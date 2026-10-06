"""Actual additional images, not a broadened detection roster."""
import gc
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from survng.app.config import ImageStorageConfig
from survng.app.evidence_gallery import GalleryImage, encode_gallery
from survng.app.events import EventStore
from survng.app.image_storage import DurableImageWriter, EncodedImage
from survng.app.motion_pipeline.object_detection import _RecordedDetectionSample, RecordedDetectionResult
from survng.app.motion_pipeline.decision_handler import MotionDecisionHandler


def observation(epoch, x=100, label='cat'):
    return dict(label=label, confidence=.52, confidence_threshold=.7, confidence_eligible=False,
        incident_eligible=False, captured_at_epoch=epoch, frame_source='recorded_main',
        box=dict(x1=x,y1=100,x2=x+80,y2=200), detection_frame_width=640,
        detection_frame_height=360, temporal_consensus=True, activity_role='active')


def samples_and_observations():
    samples = [_RecordedDetectionSample(offset=i, frame=np.full((360,640,3), i*25, np.uint8),
        objects=[], recording_path='recordings/animal.mp4', exact_timestamp=True) for i in (0,1,3,8)]
    return samples, [observation(1000+i,100+i*40) for i in (0,1,3)]


def batch_for(samples, observations):
    return encode_gallery(samples,samples[-1],observations,camera_id='gate',event_epoch=1000,
                          request={'remaining':2},timing={})


def test_encodes_only_two_distinct_existing_frames_and_keeps_weak_boxes():
    samples, observations = samples_and_observations()
    batch = batch_for(samples, observations)
    try:
        assert len(batch.images) == 2
        for image in batch.images:
            decoded = cv2.imdecode(np.frombuffer(image.image.data,np.uint8), cv2.IMREAD_COLOR)
            assert decoded.shape == (360,640,3)
            assert image.metadata['objects'][0]['confidence'] == .52
            assert not image.metadata['objects'][0]['incident_eligible']
            assert image.metadata['frame_timestamp_exact']
        assert all(sample.frame is not None for sample in samples)
    finally:
        batch.close()


def test_near_duplicate_and_cover_are_not_retained_again():
    samples, observations = samples_and_observations()
    samples[1].offset = .01
    observations[1] = observation(1000.01)
    samples[2].objects = []
    batch = encode_gallery(samples,samples[2],observations,camera_id='gate',event_epoch=1000,
                           request={'remaining':2},timing={})
    try:
        assert len(batch.images) == 1
        assert batch.images[0].captured_epoch in (1000,1000.01)
    finally:
        batch.close()


def test_optional_capacity_is_nonblocking_and_disposal_releases_it():
    samples, observations = samples_and_observations()
    first, second = batch_for(samples, observations), batch_for(samples, observations)
    try:
        timing = {}
        assert encode_gallery(samples,samples[-1],observations,camera_id='gate',event_epoch=1000,
            request={'remaining':2},timing=timing) is None
        assert timing['gallery_capacity_skips'] == 1
        first.close()
        third = batch_for(samples, observations)
        assert third is not None
        del third
        gc.collect()
        fourth = batch_for(samples, observations)
        assert fourth is not None
        fourth.close()
    finally:
        first.close()
        second.close()


def test_full_gallery_and_oversize_images_do_not_queue_work():
    samples, observations = samples_and_observations()
    with patch.object(DurableImageWriter,'_encode') as encode:
        assert encode_gallery(samples,samples[-1],observations,camera_id='gate',event_epoch=1000,
            request={'remaining':0},timing={}) is None
        encode.assert_not_called()
    with patch('survng.app.evidence_gallery.MAX_IMAGE_BYTES', 1):
        assert batch_for(samples, observations) is None


def seed(store):
    return store.add_event('gate','motion',created_at=datetime.fromtimestamp(1000,timezone.utc).isoformat(),
        objects_json=json.dumps([{'label':'dog','confidence':.8,'incident_eligible':True,
            'confidence_threshold':.7,'box':{'x1':100,'y1':100,'x2':180,'y2':200},
            'detection_frame_width':640,'detection_frame_height':360}]))


def writer_for(tmp_path):
    writer = DurableImageWriter(ImageStorageConfig())
    calls = []
    def write(image, at):
        calls.append(image)
        return str(writer.write(tmp_path/'snapshots', f'image-{at.timestamp()}', image) or '')
    return write, calls


def gallery_image(key, epoch=1001):
    frame = np.full((360,640,3),100,np.uint8)
    data = DurableImageWriter._encode('jpeg',frame,85)
    return GalleryImage(key,epoch,EncodedImage(data),{
        'camera_id':'gate','captured_epoch':epoch,'width':640,'height':360,
        'frame_timestamp_exact':True,'objects':[observation(epoch)],'role':'gallery'})


def test_persistence_is_capped_across_events_and_retries_without_promoting_cover(tmp_path):
    store=EventStore(tmp_path)
    event=seed(store)
    before=store.scene_incident(event_id=event['id'])
    before_notifications=store.scene_pending_notifications()
    write,calls=writer_for(tmp_path)
    images=[gallery_image('first'),gallery_image('second',1004)]
    assert store.retain_scene_evidence_images(event['id'],images,write)==2
    assert store.retain_scene_evidence_images(event['id'],images,write)==0
    assert store.retain_scene_evidence_images(event['id'],[gallery_image('third',1005)],write)==0
    assert len(calls)==2
    assert store.scene_evidence_capacity(event['id'])['remaining']==0
    after=store.scene_incident(event_id=event['id'])
    assert len(after['evidence_images'])==2
    for field in ('snapshot_path','snapshot_url','snapshot_observation_id','labels','scene_objects',
                  'alert_decisions','establishment','summary','objects'):
        assert after.get(field)==before.get(field),field
    assert all('evidence_images' not in row['payload'] for row in store.scene_pending_notifications())
    # No new notification obligation was created by the gallery writes.
    assert len(store.scene_pending_notifications())==len(before_notifications)
    for image in after['evidence_images']:
        assert image['objects'][0]['confidence']==.52
        assert image['objects'][0]['snapshot_visible']
        assert store.scene_evidence_image(image['id'])['snapshot_path']
    plan=store.snapshot_retention_plan(2000)
    assert plan['file_count']==2
    # Related events share the incident-wide cap.
    second=store.add_event('gate','motion',created_at=datetime.fromtimestamp(1005,timezone.utc).isoformat(),
        objects_json=json.dumps([observation(1005)]))
    assert store.scene_incident(event_id=second['id'])['id']==after['id']
    assert store.scene_evidence_capacity(second['id'])['remaining']==0


def test_write_failure_releases_reservation_and_expiry_does_not_refill_slots(tmp_path):
    store=EventStore(tmp_path);event=seed(store);image=gallery_image('first')
    with pytest.raises(OSError):
        store.retain_scene_evidence_images(event['id'],[image],Mock(side_effect=OSError('disk unavailable')))
    assert store.scene_evidence_capacity(event['id'])['remaining']==2
    write,calls=writer_for(tmp_path)
    assert store.retain_scene_evidence_images(event['id'],[image],write)==1
    row=store.scene_incident(event_id=event['id'])['evidence_images'][0]
    path=store.scene_evidence_image(row['id'])['snapshot_path']
    store._delete_snapshot_if_unreferenced(path)
    assert (tmp_path/path).exists()
    with store._lock,store._connect() as conn:
        store._clear_snapshot_references(conn,[path])
    assert store.scene_incident(event_id=event['id'])['evidence_images']==[]
    assert store.scene_evidence_image(row['id']) is None
    assert store.scene_evidence_capacity(event['id'])['remaining']==1
    assert store.retain_scene_evidence_images(event['id'],[image],write)==0
    assert len(calls)==1


def test_stale_reservation_and_duplicate_frame_are_recovered(tmp_path):
    store=EventStore(tmp_path);event=seed(store);image=gallery_image('first')
    with store._connect() as conn:
        conn.execute('insert into scene_evidence_images values(?,?,?,?,?,?,?,?,?)',
            ('abandoned',event['id'],'first',1001,'',json.dumps(image.metadata),'pending','old',time.time()-1))
    write,calls=writer_for(tmp_path)
    assert store.retain_scene_evidence_images(event['id'],[image,image],write)==1
    assert len(calls)==1


def test_decision_handler_adopts_compressed_gallery_and_releases_capacity(tmp_path):
    store=EventStore(tmp_path);event=seed(store)
    samples,observations=samples_and_observations();batch=batch_for(samples,observations)
    result=RecordedDetectionResult(frame=samples[-1].frame,objects=[{'status':'scene_observations',
        'observations':observations}],recording_path='recordings/animal.mp4',timings_ms={},gallery=batch)
    provider=Mock(return_value=result)
    write,calls=writer_for(tmp_path)
    handler=MotionDecisionHandler(camera_id='gate',events=store,detection_provider=provider,
        snapshot_writer=write,object_serializer=json.dumps)
    handler.refine('motion','motion',datetime.fromtimestamp(1000,timezone.utc),{},existing_event_id=event['id'])
    assert len(store.scene_incident(event_id=event['id'])['evidence_images'])==2
    assert batch.images==[]
    assert provider.call_args.args[1]['evidence_gallery']['remaining']==2


def test_full_subject_is_preferred_over_a_larger_clipped_near_duplicate():
    samples, observations = samples_and_observations()
    observations = [observation(1000, x=5), observation(1001, x=0, label='dog')]
    observations[1]['box']['x2'] = 95
    batch = batch_for(samples, observations)
    try:
        assert len(batch.images) == 1
        assert batch.images[0].captured_epoch == 1000
    finally:
        batch.close()


def test_failed_gallery_transaction_removes_unadopted_file(tmp_path):
    store=EventStore(tmp_path);event=seed(store)
    write,calls=writer_for(tmp_path)
    with patch.object(store,'_evidence_outbox',side_effect=RuntimeError('commit failed')):
        with pytest.raises(RuntimeError):
            store.retain_scene_evidence_images(event['id'],[gallery_image('failed')],write)
    assert store.scene_evidence_capacity(event['id'])['remaining']==2
    assert list((tmp_path/'snapshots').iterdir())==[]


def test_recorded_result_encodes_before_releasing_source_frames_without_inference():
    from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector
    samples, observations=samples_and_observations()
    for sample, item in zip(samples,observations):
        sample.objects=[{**item,'temporal_candidate_eligible':True}]
    backend=SimpleNamespace(config=SimpleNamespace(),detect=Mock())
    detector=RecordedMotionObjectDetector(SimpleNamespace(id='gate'),backend,SimpleNamespace(),lambda:None)
    result=detector._recorded_result(samples[-1],[],samples,{},time.monotonic(),
        refinement_pending=False,event_epoch=1000,gallery_request={'remaining':2})
    try:
        assert result.gallery and result.gallery.images
        assert all(sample.frame is None for sample in samples[:-1])
        assert result.frame is samples[-1].frame
        assert all(isinstance(image.image.data,bytes) for image in result.gallery.images)
        backend.detect.assert_not_called()
    finally:
        if result.gallery:
            result.gallery.close()


def test_gallery_endpoint_serves_real_image_without_exposing_it_as_observation(tmp_path):
    import threading
    from survng.app.incident_queries import IncidentQueryDependencies, IncidentQueryService, create_incident_query_router
    store=EventStore(tmp_path);event=seed(store);write,_=writer_for(tmp_path)
    store.retain_scene_evidence_images(event['id'],[gallery_image('actual')],write)
    image=store.scene_incident(event_id=event['id'])['evidence_images'][0]
    manager=SimpleNamespace(events=store,storage_dir=tmp_path,media_storage=None)
    bundle=create_incident_query_router(IncidentQueryDependencies(get_manager=lambda:manager,
        manager_lock=threading.RLock()),IncidentQueryService())
    endpoint=next(r.endpoint for r in bundle.router.routes if r.path=='/api/incidents/evidence/{observation_id}/snapshot')
    response=endpoint(image['id'])
    assert cv2.imread(str(response.path)).shape==(360,640,3)
    assert store.scene_observation(image['id']) is None


def test_concurrent_refinements_reserve_only_two_slots(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    store=EventStore(tmp_path);event=seed(store)
    barrier=threading.Barrier(2)
    writer,_=writer_for(tmp_path)
    def blocked_write(image,at):
        barrier.wait(timeout=3)
        return writer(image,at)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(store.retain_scene_evidence_images,event['id'],
            [gallery_image(f'{i}-a',1001+i),gallery_image(f'{i}-b',1010+i)],blocked_write) for i in (0,1)]
        assert sum(f.result(timeout=5) for f in futures)==2
    assert len(store.scene_incident(event_id=event['id'])['evidence_images'])==2
    assert len(list((tmp_path/'snapshots').iterdir()))==2

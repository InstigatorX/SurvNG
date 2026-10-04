"""Defects reproduced during review of canonical scene history."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from survng.app.events import EventStore
from survng.app.incident_queries import IncidentQueryService


def observation(at, label='person', **extra):
    return {'label':label, 'confidence':.8,'captured_at_epoch':at,
            'box':{'x1':10,'y1':10,'x2':30,'y2':60},**extra}


def add(store, at, objects):
    return store.add_event('gate','motion',created_at=datetime.fromtimestamp(at,timezone.utc).isoformat(),objects_json=json.dumps(objects))


def test_incident_hides_below_threshold_evidence_without_deleting_it(tmp_path):
    store = EventStore(tmp_path)
    event = add(store, 1000, [
        observation(1000, confidence=.9, confidence_threshold=.7, confidence_eligible=True, temporal_observations=2),
        observation(1000, 'robot_lawnmower', confidence=.5464, confidence_threshold=.8, confidence_eligible=False),
        # An explicit false flag is not necessary: the saved threshold suffices.
        observation(1000, 'dog', confidence=.69, confidence_threshold=.7),
        # Confidence is distinct from notification/zone eligibility.
        observation(1000, 'car', confidence=.7, confidence_threshold=.7, incident_eligible=False, temporal_observations=2),
    ])
    scene = store.scene_incident(event_id=event['id'])
    assert scene['labels'] == ['car', 'person']
    assert {obj['label'] for obj in scene['scene_objects']} == {'car', 'person'}
    assert scene['summary'] == 'Car and person observed'
    for row in [scene, *scene['events']]:
        assert not any(obj.get('label') in {'robot_lawnmower', 'dog'} for obj in row['objects'])
    assert all(entry['label'] in {'car', 'person'} for entry in scene['activity'])
    cards = store.list_scene_incident_cards()
    assert cards[0]['labels'] == ['car', 'person']
    assert store.list_scene_incident_cards(object_label='robot_lawnmower') == []
    assert store.list_scene_incident_cards(object_label='car')
    with store._connect() as connection:
        assert connection.execute('select count(*) from scene_observations').fetchone()[0] == 4


def test_incident_summary_does_not_count_unresolved_sightings(tmp_path):
    store = EventStore(tmp_path)
    event = add(store, 1000, [observation(1000, scene_track_key='one', temporal_observations=2)])
    store.record_scene_observations(event['id'], [observation(1020, scene_track_key='two', temporal_observations=2)])
    scene = store.scene_incident(event_id=event['id'])
    assert len(scene['scene_objects']) == 2
    assert scene['summary'] == 'Person observed'
    assert scene['continuity_uncertain']


def test_shared_frame_is_retained_in_every_episode_that_analyzed_it(tmp_path):
    store=EventStore(tmp_path)
    left=add(store,1000,[observation(1000)])
    right=add(store,1100,[observation(1100)])
    shared=observation(1050,'dog',frame_source='recorded_main',frame_timestamp_exact=True,recording_path='recordings/shared.mp4')
    store.record_scene_observations(left['id'],[shared])
    store.record_scene_observations(right['id'],[shared])
    for row in (left,right):
        assert 'dog' in store.scene_incident(event_id=row['id'])['labels']
    before=store.scene_incident(event_id=right['id'])
    store.record_scene_observations(right['id'],[shared])
    assert store.scene_incident(event_id=right['id'])==before


def test_legacy_tracks_absent_from_cover_are_imported(tmp_path):
    store=EventStore(tmp_path)
    event=add(store,1000,[observation(1000),{'status':'object_tracking','object_tracking':{
        'state':'complete','frame_width':100,'frame_height':100,'tracks':[{
            'track_id':7,'label':'dog','state':'confirmed','observations':3,'max_confidence':.88,
            'first_seen':'1970-01-01T00:16:41+00:00','last_seen':'1970-01-01T00:16:42+00:00',
            'box_history':[[1001,40,10,60,30],[1002,42,10,62,30]],
        }]}}])
    scene=store.scene_incident(event_id=event['id'])
    assert scene['labels']==['dog','person']
    dog=next(s for s in scene['scene_objects'] if s['label']=='dog')
    assert len(dog['observations'])==2
    assert all(not o['snapshot_available'] for o in dog['observations'])
    assert all(o['confidence_provenance']=='track_summary' for o in dog['observations'])


def test_duplicate_raw_batches_still_attach_the_cover(tmp_path):
    store=EventStore(tmp_path)
    image=tmp_path/'cover.webp';image.write_bytes(b'image')
    raw=observation(1000)
    event=store.add_event('gate','motion',created_at=datetime.fromtimestamp(1000,timezone.utc).isoformat(),
        snapshot_path=str(image),objects_json=json.dumps([raw,{'status':'scene_observations','observations':[raw,raw]}]))
    scene=store.scene_incident(event_id=event['id'])
    assert scene['scene_objects'][0]['observations'][0]['snapshot_available']


def test_rare_filter_does_not_hydrate_unmatched_history(tmp_path):
    store=EventStore(tmp_path)
    for i in range(4):add(store,1000+i*100,[observation(1000+i*100)])
    with patch.object(store,'_scene_payload', wraps=store._scene_payload) as hydrate:
        page, more, _=IncidentQueryService.recent_filtered_summaries(SimpleNamespace(events=store),
            limit=18,offset=0,gap_seconds=45,event_type='all',object_label='elephant')
    assert page==[] and not more
    assert hydrate.call_count==0


def test_legacy_track_upgrade_repairs_already_imported_rows_silently(tmp_path):
    store=EventStore(tmp_path)
    row=add(store,1000,[observation(1000)])
    original=store.scene_incident(event_id=row['id'])
    legacy={'status':'object_tracking','object_tracking':{'state':'complete','tracks':[{
        'label':'dog','track_id':7,'state':'confirmed','max_confidence':.8,
        'box_history':[[1001,40,10,60,30]],
    }]}}
    with store._connect() as conn:
        conn.execute('update events set objects_json=? where id=?',(json.dumps([observation(1000),legacy]),row['id']))
        conn.execute('delete from scene_notification_outbox')
    upgraded=EventStore(tmp_path)
    current=upgraded.scene_incident(event_id=row['id'])
    assert current['id']==original['id']
    assert current['labels']==['dog','person']
    assert not upgraded.scene_pending_notifications()
    assert EventStore(tmp_path).scene_incident(event_id=row['id'])==current


def test_merge_coalesces_shared_evidence_without_losing_episode_context(tmp_path):
    store=EventStore(tmp_path)
    left=add(store,1000,[]);right=add(store,1100,[])
    shared=observation(1050,'dog')
    for event in (left,right):store.record_scene_observations(event['id'],[shared])
    a,b=[store.scene_incident(event_id=e['id']) for e in (left,right)]
    merged=store.correct_scene_incident(a['id'],a['revision'],{'operation':'merge','incident_ids':[b['id']],'expected_revisions':{b['id']:b['revision']}})
    assert len(merged['scene_objects'])==1
    assert len(merged['scene_objects'][0]['observations'])==2
    store.correct_scene_incident(merged['id'],merged['revision'],{'operation':'split','episode_ids':[b['episodes'][0]['id']]})
    for row in (left,right):
        assert store.scene_incident(event_id=row['id'])['labels']==['dog']


def test_nearby_distinct_frames_do_not_both_inherit_cover(tmp_path):
    store=EventStore(tmp_path)
    image=tmp_path/'cover.webp';image.write_bytes(b'image')
    event=store.add_event('gate','motion',created_at=datetime.fromtimestamp(1000,timezone.utc).isoformat(),
        snapshot_path=str(image),objects_json=json.dumps([observation(1000),{'status':'scene_observations','observations':[observation(999.98),observation(1000.02)]}]))
    scene=store.scene_incident(event_id=event['id'])
    assert not any(o['snapshot_available'] for s in scene['scene_objects'] for o in s['observations'])


def test_detector_box_movement_alone_cannot_establish_camera_transition(tmp_path):
    store=EventStore(tmp_path)
    observations=[]
    for i in range(21):
        item=observation(1000+i*.5,scene_track_key='walker')
        item['box']={'x1':10+i*2,'y1':10,'x2':30+i*2,'y2':60}
        observations.append(item)
    first=add(store,1000,[{'status':'scene_observations','observations':observations}])
    second=store.add_event('drive','motion',created_at=datetime.fromtimestamp(1020,timezone.utc).isoformat(),
        objects_json=json.dumps([observation(1020,temporal_newly_appeared=True)]))
    identity=[{'identity_id':1,'status':'confirmed'}]
    store.update_scene_identities(first['id'],identity)
    store.update_scene_identities(second['id'],identity)
    assert store.scene_incident(event_id=first['id'])['id']!=store.scene_incident(event_id=second['id'])['id']


def test_missing_geometry_is_not_evidence_of_movement(tmp_path):
    store=EventStore(tmp_path)
    observations=[{'label':'person','confidence':.8,'captured_at_epoch':1000+i,'scene_track_key':'legacy'} for i in range(2)]
    first=add(store,1000,[{'status':'scene_observations','observations':observations}])
    second=store.add_event('drive','motion',created_at=datetime.fromtimestamp(1020,timezone.utc).isoformat(),
        objects_json=json.dumps([observation(1020,temporal_newly_appeared=True)]))
    identity=[{'identity_id':1,'status':'confirmed'}]
    store.update_scene_identities(first['id'],identity)
    store.update_scene_identities(second['id'],identity)
    assert store.scene_incident(event_id=first['id'])['id']!=store.scene_incident(event_id=second['id'])['id']


def test_legacy_cover_and_track_id_share_one_subject(tmp_path):
    store=EventStore(tmp_path)
    row=add(store,1000,[observation(1000,track_id=7),{'status':'object_tracking','object_tracking':{
        'state':'complete','tracks':[{'track_id':7,'label':'person','state':'confirmed','max_confidence':.8,
            'box_history':[[1001,40,10,60,60],[1002,60,10,80,60]]}],
    }}])
    scene=store.scene_incident(event_id=row['id'])
    assert len(scene['scene_objects'])==1
    assert len(scene['scene_objects'][0]['observations'])==3
    assert EventStore(tmp_path).scene_incident(event_id=row['id'])==scene

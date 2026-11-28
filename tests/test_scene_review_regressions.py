"""Defects reproduced during review of canonical scene history."""
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from survng.app.events import EventStore
from survng.app.incident_queries import IncidentQueryService, _filter_incident_summaries


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


def test_duplicate_raw_batches_still_attach_the_cover(tmp_path):
    store=EventStore(tmp_path)
    image=tmp_path/'cover.webp';image.write_bytes(b'image')
    raw=observation(1000)
    event=store.add_event('gate','motion',created_at=datetime.fromtimestamp(1000,timezone.utc).isoformat(),
        snapshot_path=str(image),objects_json=json.dumps([raw,{'status':'scene_observations','observations':[raw,raw]}]))
    scene=store.scene_incident(event_id=event['id'])
    assert scene['scene_objects'][0]['observations'][0]['snapshot_available']


def test_confidence_floor_filters_incident_search(tmp_path):
    store = EventStore(tmp_path)
    add(store, 1000, [observation(1000, confidence=.9, temporal_observations=2),
                      observation(1000, 'car', confidence=.55, temporal_observations=2)])
    add(store, 2000, [observation(2000, 'car', confidence=.6, temporal_observations=2)])
    add(store, 3000, [])
    assert store.count_scene_incidents(event_type='object') == 2
    assert store.count_scene_incidents(event_type='object', min_confidence=.8) == 1
    assert store.count_scene_incidents(event_type='object', min_confidence=.95) == 0
    # A label and a floor must hold for the same observation.
    assert store.count_scene_incidents(object_label='car', min_confidence=.58) == 1
    assert store.count_scene_incidents(object_label='car', min_confidence=.8) == 0
    assert len(store.list_scene_incident_cards(min_confidence=.8)) == 1
    assert len(store.list_scene_incidents(min_confidence=.8, summary_only=True)) == 1
    # Several labels: any one of them matches, and blanks are ignored.
    add(store, 4000, [observation(4000, 'dog', confidence=.7, temporal_observations=2)])
    assert store.count_scene_incidents(object_label='car') == 2
    assert store.count_scene_incidents(object_label='dog') == 1
    assert store.count_scene_incidents(object_label='car,dog') == 3
    assert store.count_scene_incidents(object_label=' dog , ,car,dog') == 3
    assert store.count_scene_incidents(object_label='person,elephant') == 1
    assert store.count_scene_incidents(object_label='car,dog', min_confidence=.65) == 1
    assert len(store.list_scene_incident_cards(object_label='car,dog')) == 3
    assert len(_filter_incident_summaries([{'labels': ['car']}, {'labels': ['dog']}, {'labels': ['person']}], 'all', object_label='car,dog')) == 2
    # Motion-only incidents have no detection to compare.
    assert store.count_scene_incidents(event_type='motion', min_confidence=.8) == store.count_scene_incidents(event_type='motion')
    manager = SimpleNamespace(events=store)
    day = datetime.fromtimestamp(1000, timezone.utc).strftime('%Y-%m-%d')
    found = IncidentQueryService.search(manager, day=day, time_zone='UTC', event_type='object', min_confidence=.8, limit=10)
    assert found['total'] == 1 and len(found['items']) == 1
    assert IncidentQueryService.search(manager, day=day, time_zone='UTC', event_type='object', limit=10)['total'] == 3


def test_camera_zone_and_person_filters_take_several_values(tmp_path):
    store = EventStore(tmp_path)
    def seen(camera, at, zone):
        return store.add_event(camera, 'motion', created_at=datetime.fromtimestamp(at, timezone.utc).isoformat(),
                               objects_json=json.dumps([observation(at, zones=[zone], temporal_observations=2)]))
    gate, door = seen('gate', 1000, 'Road'), seen('door', 2000, 'Porch')
    seen('yard', 3000, 'Lawn')
    assert store.count_scene_incidents() == 3
    # Any chosen value matches within one filter; blanks and repeats are ignored.
    assert store.count_scene_incidents(camera_id='gate') == 1
    assert store.count_scene_incidents(camera_id='gate,door') == 2
    assert store.count_scene_incidents(camera_id=' door , ,gate,door') == 2
    assert store.count_scene_incidents(camera_id='gate,nowhere') == 1
    assert store.count_scene_incidents(zone='Road,Lawn') == 2
    assert store.count_scene_incidents(zone='Porch') == 1
    # Different filters still narrow each other.
    assert store.count_scene_incidents(camera_id='gate,door', zone='Porch,Lawn') == 1
    assert len(store.list_scene_incident_cards(camera_id='gate,yard')) == 2
    summaries = [{'camera_id': 'gate', 'zones': ['Road']}, {'camera_ids': ['door', 'yard'], 'zones': ['Porch']}, {'camera_id': 'yard', 'zones': []}]
    assert len(_filter_incident_summaries(summaries, 'all', camera_id='gate,door')) == 2
    assert len(_filter_incident_summaries(summaries, 'all', zone='Road,Porch')) == 2
    assert len(_filter_incident_summaries(summaries, 'all', camera_id='yard', zone='Porch')) == 1

    faces = [{'event_id': gate['id'], 'person_id': 7}, {'event_id': door['id'], 'person_id': 9}]
    manager = SimpleNamespace(events=store, face_identity_observations=lambda ids: [face for face in faces if face['event_id'] in ids])
    day = datetime.fromtimestamp(1000, timezone.utc).strftime('%Y-%m-%d')
    def people(value):
        return IncidentQueryService.search(manager, day=day, time_zone='UTC', event_type='all', person_id=value, limit=10)['total']
    assert people(7) == 1 and people('7') == 1
    assert people('7,9') == 2
    assert people('9, ,bad,404') == 1
    assert people('') == 3 and people(0) == 3


def test_rare_filter_does_not_hydrate_unmatched_history(tmp_path):
    store=EventStore(tmp_path)
    for i in range(4):add(store,1000+i*100,[observation(1000+i*100)])
    with patch.object(store,'_scene_payload', wraps=store._scene_payload) as hydrate:
        page, more, _=IncidentQueryService.recent_filtered_summaries(SimpleNamespace(events=store),
            limit=18,offset=0,gap_seconds=45,event_type='all',object_label='elephant')
    assert page==[] and not more
    assert hydrate.call_count==0


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


def test_unconfirmed_incident_is_not_reactivated_by_fresh_activity(tmp_path):
    store = EventStore(tmp_path)
    now = time.time()
    event = add(store, now, [observation(now, temporal_observations=2)])
    incident_id = store.scene_incident(event_id=event["id"])["incident_id"]
    with store._connect() as conn:
        conn.execute("update scene_incidents set state='unconfirmed' where id=?", (incident_id,))
    store.record_scene_observations(
        event["id"],
        [observation(now + 1, scene_track_key="fresh", temporal_observations=2)],
    )
    # Fresh activity must not jump unconfirmed → active. Establishment may
    # still restore unconfirmed → complete when a supported verdict exists.
    assert store.scene_incident(event_id=event["id"])["state"] != "active"


def test_merged_incident_is_not_reactivated_by_fresh_activity(tmp_path):
    store = EventStore(tmp_path)
    now = time.time()
    event = add(store, now, [observation(now, temporal_observations=2)])
    incident_id = store.scene_incident(event_id=event["id"])["incident_id"]
    with store._connect() as conn:
        conn.execute("update scene_incidents set state='merged' where id=?", (incident_id,))
        before = conn.execute(
            "select state from scene_incidents where id=?", (incident_id,)
        ).fetchone()[0]
        assert before == "merged"
        # Exercise the same guarded write used by _scene_project.
        conn.execute(
            "update scene_incidents set state='active',historical=0 "
            "where id=? and state in ('complete','active')",
            (incident_id,),
        )
        after = conn.execute(
            "select state from scene_incidents where id=?", (incident_id,)
        ).fetchone()[0]
    assert after == "merged"


def test_complete_incident_reactivates_on_fresh_activity(tmp_path):
    store = EventStore(tmp_path)
    now = time.time()
    event = add(store, now, [observation(now, temporal_observations=2)])
    incident_id = store.scene_incident(event_id=event["id"])["incident_id"]
    with store._connect() as conn:
        conn.execute("update scene_incidents set state='complete' where id=?", (incident_id,))
    store.record_scene_observations(
        event["id"],
        [observation(now + 1, scene_track_key="reopen", temporal_observations=2)],
    )
    assert store.scene_incident(event_id=event["id"])["state"] == "active"


def test_a_filter_value_may_contain_a_comma(tmp_path):
    from survng.app.event_store.scenes import filter_values
    assert filter_values('gate, door,,gate') == ['gate', 'door', 'gate, door,,gate']
    assert filter_values('Driveway%2C left,Porch') == ['Driveway, left', 'Porch']
    assert filter_values('50%252C off') == ['50%2C off', '50%252C off'] and filter_values('50%25 off,Porch') == ['50% off', 'Porch']
    assert filter_values('gate') == ['gate'] and filter_values('') == [] and filter_values(None) == []
    store = EventStore(tmp_path)
    add(store, 1000, [observation(1000, temporal_observations=2, zones=['Driveway, left'])])
    add(store, 2000, [observation(2000, temporal_observations=2, zones=['Porch'])])
    assert 'Driveway, left' in store.scene_incident_facets()['zones']
    # Sent whole, as older callers do, and escaped beside another value, as the gallery does.
    assert store.count_scene_incidents(zone='Driveway, left') == 1
    assert store.count_scene_incidents(zone='Driveway%2C left') == 1
    assert store.count_scene_incidents(zone='Driveway%2C left,Porch') == 2
    summaries = [{'camera_id': 'gate', 'zones': ['Driveway, left']}, {'camera_id': 'gate', 'zones': ['Porch']}]
    assert len(_filter_incident_summaries(summaries, 'all', zone='Driveway, left')) == 1
    assert len(_filter_incident_summaries(summaries, 'all', zone='Driveway%2C left,Porch')) == 2


def test_a_later_detection_does_not_hide_queued_recorded_analysis(tmp_path):
    store = EventStore(tmp_path)
    event = add(store, 1000, [observation(1000, temporal_observations=2)])
    def reason():
        return store.scene_incident(event_id=event['id'])['episodes'][0]['coverage']['reason']
    assert reason().startswith('sampled detector frames')
    store.enqueue_scene_tracking(event['id'], 995, 1030)
    assert reason() == 'recorded analysis queued'
    add(store, 1005, [observation(1005, temporal_observations=2)])
    assert reason() == 'recorded analysis queued'

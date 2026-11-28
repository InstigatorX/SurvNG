"""Defects reproduced during review of canonical scene history."""
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

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


def test_zone_filter_starts_from_each_incidents_own_episodes(tmp_path):
    store = EventStore(tmp_path)
    add(store, 1000, [observation(1000, zones=['Road'], temporal_observations=2)])
    where, args = store._scene_query_where(zone='Road,Porch')
    with store._connect() as conn:
        plan = [row[3] for row in conn.execute('explain query plan select count(*) from scene_incidents i where ' + where, args)]
    # The zone rows are looked up by episode. Looked up by zone name alone, every
    # row of that zone is visited once per incident.
    zone_step = next(step for step in plan if step.startswith('SEARCH z '))
    assert 'episode_id=?' in zone_step, plan
    assert store.count_scene_incidents(zone='Road') == 1 and store.count_scene_incidents(zone='Porch') == 0


def test_card_cover_choice_is_reused_until_an_event_changes(tmp_path):
    from survng.app import incident_presenter
    store = EventStore(tmp_path)
    events = [add(store, 1000 + step * 8, [observation(1000 + step * 8, confidence=.6 + step / 20, temporal_observations=2)]) for step in range(5)]
    with patch.object(incident_presenter, '_event_row', wraps=incident_presenter._event_row) as present:
        first = store.list_scene_incident_cards()
        presented_once = present.call_count
        assert presented_once == 5
        again = store.list_scene_incident_cards()
        assert present.call_count == presented_once, 'listing the same incident again reads no event'
        assert again == first
        # The kept choice is a copy: a caller changing its card cannot change the next one.
        first[0]['events'][0]['snapshot_captured_at'] = 'changed'
        assert store.list_scene_incident_cards() == again
        # A changed event is a different key, so the choice is made afresh.
        with store._lock, store._connect() as conn:
            conn.execute('update events set objects_json=? where id=?',
                         (json.dumps([observation(1000, confidence=.99, temporal_observations=2)]), events[0]['id']))
        store.list_scene_incident_cards()
        assert present.call_count == 2 * presented_once


def test_compact_search_leaves_out_the_per_event_list(tmp_path):
    store = EventStore(tmp_path)
    for step in range(6):                                   # one incident, six events
        add(store, 1000 + step * 8, [observation(1000 + step * 8, temporal_observations=2)])
    manager = SimpleNamespace(events=store)
    day = datetime.fromtimestamp(1000, timezone.utc).strftime('%Y-%m-%d')
    full = IncidentQueryService.search(manager, day=day, time_zone='UTC', event_type='all', limit=10)['items'][0]
    compact = IncidentQueryService.search(manager, day=day, time_zone='UTC', event_type='all', limit=10, compact=True)['items'][0]
    assert len(full['events']) == 6 and 'events' not in compact
    # Everything a gallery tile draws, and the event membership, is still there.
    assert compact['event_ids'] == full['event_ids'] and len(compact['event_ids']) == 6
    for key in ('id', 'incident_id', 'revision', 'camera_id', 'start_at', 'labels', 'objects', 'representative_event_id', 'review_verdict'):
        assert compact[key] == full[key], key
    assert len(json.dumps(compact)) < len(json.dumps(full)) / 2


def test_cards_carry_the_review_verdict(tmp_path):
    store = EventStore(tmp_path)
    first = add(store, 1000, [observation(1000, temporal_observations=2)])
    add(store, 2000, [observation(2000, temporal_observations=2)])
    reviewed = store.scene_incident(event_id=first['id'])['id']
    assert [card['review_verdict'] for card in store.list_scene_incident_cards()] == [None, None]
    store.set_incident_review(reviewed, 'reviewed')
    by_id = {card['id']: card['review_verdict'] for card in store.list_scene_incident_cards()}
    assert by_id[reviewed] == 'reviewed' and list(by_id.values()).count(None) == 1
    day = datetime.fromtimestamp(1000, timezone.utc).strftime('%Y-%m-%d')
    items = IncidentQueryService.search(SimpleNamespace(events=store), day=day, time_zone='UTC', event_type='all', limit=10)['items']
    assert {item['id']: item['review_verdict'] for item in items} == by_id
    store.set_incident_review(reviewed, None)
    assert [card['review_verdict'] for card in store.list_scene_incident_cards()] == [None, None]


def test_search_filters_by_review_state_and_marks_many_at_once(tmp_path):
    store = EventStore(tmp_path)
    ids = [store.scene_incident(event_id=add(store, 1000 * step, [observation(1000 * step, temporal_observations=2)])['id'])['id']
           for step in range(1, 6)]
    store.set_incident_review(ids[0], 'reviewed')
    store.set_incident_review(ids[1], 'nothing_here')
    store.set_incident_review(ids[2], 'keep')              # kept for later: still to look at
    assert store.count_scene_incidents(review='reviewed') == 2
    assert store.count_scene_incidents(review='unreviewed') == 3
    assert store.count_scene_incidents(review='') == 5
    assert {card['id'] for card in store.list_scene_incident_cards(review='unreviewed')} == set(ids[2:])
    manager = SimpleNamespace(events=store)
    day = datetime.fromtimestamp(1000, timezone.utc).strftime('%Y-%m-%d')
    def total(**extra):
        return IncidentQueryService.search(manager, day=day, time_zone='UTC', event_type='all', limit=10, **extra)['total']
    assert (total(review='unreviewed'), total(review='reviewed'), total(review='all'), total()) == (3, 2, 5, 5)
    with pytest.raises(HTTPException):
        total(review='maybe')

    marked = store.set_incident_reviews(ids[2:], 'reviewed')
    assert marked == {'verdict': 'reviewed', 'incident_ids': ids[2:]}
    assert store.count_scene_incidents(review='unreviewed') == 0
    assert store.incident_review(ids[2])['verdict'] == 'reviewed'
    store.set_incident_reviews([ids[0], ids[0], ids[4]], None)   # repeats are one request
    assert store.count_scene_incidents(review='unreviewed') == 2
    # All or nothing: one unknown incident leaves every mark as it was.
    with pytest.raises(LookupError):
        store.set_incident_reviews([ids[0], 'incident-missing'], 'reviewed')
    assert store.incident_review(ids[0])['verdict'] is None
    for bad in ([], [''], [f'incident-{n}' for n in range(501)]):
        with pytest.raises(ValueError):
            store.set_incident_reviews(bad, 'reviewed')
    with pytest.raises(ValueError):
        store.set_incident_reviews(ids[:1], 'maybe')


def test_search_names_the_version_of_each_cover_file(tmp_path):
    from survng.app.image_cache import image_version
    picture = tmp_path / 'snapshots' / 'cover.jpg'
    picture.parent.mkdir()
    picture.write_bytes(b'first')
    store = EventStore(tmp_path)
    def add_pictured(at, path):
        return store.add_event('gate', 'motion', created_at=datetime.fromtimestamp(at, timezone.utc).isoformat(),
                               objects_json=json.dumps([observation(at, temporal_observations=2)]), snapshot_path=path)
    seen = add_pictured(1000, 'snapshots/cover.jpg')
    add_pictured(2000, 'snapshots/missing.jpg')
    add(store, 3000, [observation(3000, temporal_observations=2)])
    reviewed = store.scene_incident(event_id=seen['id'])['id']
    store.set_incident_review(reviewed, 'reviewed')
    faces = [{'event_id': seen['id'], 'person_id': 7}]
    manager = SimpleNamespace(events=store, storage_dir=tmp_path, media_storage=None,
                              face_identity_observations=lambda ids: [face for face in faces if face['event_id'] in ids])
    day = datetime.fromtimestamp(1000, timezone.utc).strftime('%Y-%m-%d')
    def search(**extra):
        return IncidentQueryService.search(manager, day=day, time_zone='UTC', event_type='all', limit=10, **extra)['items']
    # The stored cards say only that a picture exists; the version is of the file itself.
    items = search()
    assert [item['snapshot_path'] for item in items] == ['', 'available', 'available']
    assert [item.get('image_version') for item in items] == [None, None, image_version(picture)]
    assert all('cover_file' not in item for item in items)
    # A People filter reads full incidents; they carry the same version and review mark.
    (person,) = search(person_id=7)
    assert person['id'] == reviewed and person['image_version'] == image_version(picture)
    assert person['review_verdict'] == 'reviewed' and 'cover_file' not in person
    before = items[2]['image_version']
    picture.write_bytes(b'second, longer')
    assert search()[2]['image_version'] != before


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

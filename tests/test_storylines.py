import json
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from survng.app.config import AppConfig
from survng.app.manager_access import ManagerAccessCoordinator
from survng.app.storylines import StorylineStore, StoryConflict
from survng.app.story_replay import build_replay_plan, crop_at, crop_path
from survng.app.storyline_routes import StorylineDependencies, create_storyline_router, _ai_montage, StoryAiReview
from survng.app.security import required_api_scope


def incident(name, camera='gate', start=100, end=105, revision=1):
    from datetime import datetime, timezone
    iso = lambda n: datetime.fromtimestamp(n, timezone.utc).isoformat()
    return {'id': name, 'revision': revision, 'camera_id': camera, 'camera_ids': [camera], 'start_at': iso(start), 'end_at': iso(end), 'representative_event_id': 1,
            'episodes': [{'id': 'ep-'+name, 'camera_id': camera, 'start_at': iso(start), 'end_at': iso(end)}], 'scene_objects': [], 'events': []}


def rows(camera, start, end):
    return [{'start_epoch': start, 'end_epoch': end}]


def test_store_persists_and_optimistic_writes(tmp_path):
    store = StorylineStore(tmp_path/'events.db')
    s = store.create({'title': 'Arrival', 'summary': '', 'members': [{'incident_id': 'i1'}, {'incident_id': 'i2'}]})
    reloaded = StorylineStore(tmp_path/'events.db')
    assert reloaded.get(s['id']) == s
    updated = store.update(s['id'], 1, {'title': 'Delivery'})
    assert updated['revision'] == 2
    with pytest.raises(StoryConflict):
        store.update(s['id'], 1, {'title': 'Lost edit'})
    assert store.get(s['id'])['title'] == 'Delivery'
    with pytest.raises(StoryConflict):
        store.delete(s['id'], 1)
    assert store.list()['total'] == 1


def test_atomic_merge_split_and_invalidation(tmp_path):
    store = StorylineStore(tmp_path/'events.db')
    a = store.create({'title': 'A', 'summary': '', 'members': [{'incident_id': 'i1'}, {'incident_id': 'i2'}]})
    a = store.update(a['id'], 1, {'ai_review': {'summary': 'Old evidence'}})
    with pytest.raises(ValueError):
        store.split(a['id'], 2, ['i1', 'missing'])
    assert store.get(a['id'])['revision'] == 2
    result = store.split(a['id'], 2, ['i2'])
    assert result['story']['members'] == [{'incident_id': 'i1'}]
    assert result['story']['ai_review'] is None
    b = result['separated']
    with pytest.raises(StoryConflict):
        store.merge(a['id'], 3, b['id'], 2)
    assert store.list()['total'] == 2
    merged = store.merge(a['id'], 3, b['id'], 1)
    assert len(merged['members']) == 2
    assert store.list()['total'] == 1
    with pytest.raises(LookupError):
        store.get(b['id'])


def test_concurrent_edits_only_one_wins(tmp_path):
    store = StorylineStore(tmp_path/'events.db')
    story = store.create({'title': 'A', 'members': [{'incident_id': 'i'}]})
    barrier, results = threading.Barrier(2), []
    def edit(title):
        barrier.wait()
        try:
            store.update(story['id'], 1, {'title': title})
            results.append('saved')
        except StoryConflict:
            results.append('conflict')
    threads = [threading.Thread(target=edit, args=(t,)) for t in ('a', 'b')]
    for t in threads: t.start()
    for t in threads: t.join()
    assert sorted(results) == ['conflict', 'saved']


def test_replay_joins_cameras_and_marks_elapsed_time():
    plan = build_replay_plan([incident('a', start=100, end=105), incident('b', 'porch', 120, 125)], rows, padding=0)
    assert [s['kind'] for s in plan['shots']] == ['video', 'gap', 'video']
    assert plan['shots'][1]['elapsed_seconds'] == 15
    assert plan['shots'][2]['offset'] == 7
    assert plan['duration'] == 12


def test_split_screen_preserves_simultaneous_activity_and_no_duplicates():
    plan = build_replay_plan([incident('a', start=100, end=110), incident('b', 'porch', 105, 115), incident('c', 'gate', 106, 108)], rows, padding=0)
    overlap = next(s for s in plan['shots'] if s['start'] == 105)
    assert {v['camera_id'] for v in overlap['views']} == {'gate', 'porch'}
    assert sum(s['duration'] for s in plan['shots']) == 15
    single = build_replay_plan([incident('a', start=100, end=110), incident('b', 'porch', 105, 115)], rows, split_screen=False, padding=0)
    assert all(len(s['views']) == 1 for s in single['shots'])


def test_missing_recordings_are_not_replayed_as_continuous_video():
    partial = lambda *_: [{'start_epoch': 100, 'end_epoch': 103}, {'start_epoch': 107, 'end_epoch': 110}]
    plan = build_replay_plan([incident('a', start=98, end=112)], partial, padding=0)
    assert [(g['start'], g['end']) for g in plan['missing_coverage']] == [(98, 100), (103, 107), (110, 112)]
    assert [s['kind'] for s in plan['shots']] == ['video', 'gap', 'video']
    assert plan['shots'][1]['elapsed_seconds'] == 4
    assert build_replay_plan([incident('a')], lambda *_: [])['shots'] == []


def test_manual_order_labels_backward_time_jump():
    plan = build_replay_plan([incident('later', start=200, end=205), incident('earlier', start=100, end=105)], rows, padding=0, order='member')
    assert plan['shots'][0]['views'][0]['incident_id'] == 'later'
    assert plan['shots'][1]['elapsed_seconds'] == -105


def test_crop_keeps_all_subjects_and_uses_only_supported_time():
    i = incident('a', start=100, end=110)
    def o(at, box):
        return {'captured_at': at, 'camera_id': 'gate', 'detection_frame_width': 100, 'detection_frame_height': 100, 'box': dict(zip(('x1','y1','x2','y2'), box))}
    i['scene_objects'] = [{'id': 'p1', 'observations': [o(100, (10,20,20,40)), o(101, (20,20,30,40))]}, {'id': 'p2', 'observations': [o(100, (70,20,90,40)), o(101, (70,20,90,40))]}]
    points = crop_path(i, 'gate', 100, 110)
    assert points[0]['size'] == 1
    one = crop_path(i, 'gate', 100, 110, ['p1'])
    assert one[0]['size'] == .5
    assert crop_at(one, 5)['size'] == 1
    assert crop_at(one, -1)['size'] == 1
    assert crop_at(one, .5)['size'] == 1  # Establishing view.
    i['scene_objects'][0]['observations'][1]['captured_at'] = 109
    assert crop_path(i, 'gate', 100, 110, ['p1']) == []


@pytest.fixture
def api(tmp_path):
    details = {name: incident(name, start=100+10*n, end=105+10*n) for n, name in enumerate(['a','b','c'])}
    recording = tmp_path/'segment.mp4'; recording.write_bytes(b'fixture')
    config = AppConfig(cameras=[])
    store = StorylineStore(tmp_path/'events.db')
    manager = SimpleNamespace(storylines=store, events=SimpleNamespace(scene_incident=lambda id: details.get(id), get=lambda id: None), config=config,
                              recorder=SimpleNamespace(recording_rows_between=lambda c,s,e,*a,**k: [{'start_epoch': s,'end_epoch': e,'path': str(recording)}]), storage_dir=tmp_path, media_storage=None)
    exported = []
    deps = StorylineDependencies(lambda: manager, threading.RLock(), ManagerAccessCoordinator(), SimpleNamespace(), lambda: SimpleNamespace(create=lambda p: exported.append(p) or {'id': 'job'}), lambda: threading.Semaphore(1))
    app = FastAPI(); app.include_router(create_storyline_router(deps))
    return TestClient(app), manager, details, exported


def create(client, members=('a','b')):
    response = client.post('/api/storylines', json={'title': 'Delivery', 'members': [{'incident_id': n} for n in members]})
    assert response.status_code == 201, response.text
    return response.json()


def test_api_crud_expired_sources_and_validation(api):
    client, manager, details, _ = api
    story = create(client)
    id = story['id']
    response = client.put('/api/storylines/'+id, json={'title': 'Edited', 'revision': 1, 'members': [{'incident_id': 'a'}]})
    assert response.status_code == 200
    assert client.put('/api/storylines/'+id, json={'title': 'Lost', 'revision': 1, 'members': [{'incident_id': 'a'}]}).status_code == 409
    assert client.post('/api/storylines', json={'members': [{'incident_id': 'a'}, {'incident_id': 'a'}]}).status_code == 422
    assert client.post('/api/storylines', json={'members': [{'incident_id': 'missing'}]}).status_code == 404
    details.pop('a')
    read = client.get('/api/storylines/'+id).json()
    assert read['missing_incidents'] == ['a']
    assert client.get('/api/storylines').json()['total'] == 1
    assert client.delete('/api/storylines/'+id+'?revision=2').status_code == 200
    assert client.get('/api/storylines/'+id).status_code == 404


def test_api_replay_export_and_ai_disabled(api):
    client, _, _, exported = api
    story = create(client)
    url = '/api/storylines/'+story['id']
    assert client.post(url+'/replay', json={'revision': 99}).status_code == 409
    plan = client.post(url+'/replay', json={'revision': 1}).json()
    assert plan['shots'] and plan['evidence_fingerprint']
    assert client.post(url+'/export', json={'revision': 1}).status_code == 202
    assert exported[0]['kind'] == 'storyline'
    assert 'path' not in json.dumps(exported[0]['options'])
    assert client.post(url+'/ai', json={'revision': 1}).status_code == 503


def test_suggestion_versioning_dismissal_and_no_automatic_merging(api):
    client, _, _, _ = api
    story = create(client, ('a',))
    url = '/api/storylines/'+story['id']
    suggestion = {'incident': {'id': 'b'}, 'evidence_fingerprint': 'a'*64, 'reasons': ['Nearby person; identity not established']}
    with patch('survng.app.storyline_routes._suggestions', return_value={'items': [suggestion]}):
        body = {'revision': 1, 'incident_id': 'b', 'evidence_fingerprint': 'b'*64, 'decision': 'accept'}
        assert client.post(url+'/suggestions', json=body).status_code == 409
        body['evidence_fingerprint'] = 'a'*64; body['decision'] = 'reject'
        rejected = client.post(url+'/suggestions', json=body).json()
        assert len(rejected['members']) == 1
        assert rejected['dismissed']['b'] == 'a'*64
        body.update(revision=2, decision='accept')
        added = client.post(url+'/suggestions', json=body).json()
        assert added['members'][1]['relationship'] == 'related_event'


def test_ai_uses_configured_transport_and_rejects_unknown_citations(api):
    client, manager, details, _ = api
    manager.config.audit_ai.enabled = True; manager.config.audit_ai.api_key = 'test-only'
    manager.config.audit_ai.assistant_reasoning_model = 'existing-deep-model'
    story = create(client)
    url = '/api/storylines/'+story['id']+'/ai'
    review = StoryAiReview(title='Delivery visit', summary='Two selected incidents.', actions=[{'description':'A person is visible.', 'incident_ids':['a'], 'certainty':'observed'}], suggested_relationships=[])
    montage = [{'incident_id': 'a'}, {'incident_id': 'b'}]
    with patch('survng.app.storyline_routes._ai_montage', return_value=montage), patch('survng.app.storyline_routes.AuditAiAdvisor.analyze_structured', return_value=review) as transport:
        response = client.post(url, json={'revision': 1})
        assert response.status_code == 200, response.text
        assert transport.call_args.kwargs['model_override'] == 'existing-deep-model'
        assert json.loads(transport.call_args.args[1])['operator_context']['title'] == story['title']
        assert 'test-only' not in transport.call_args.args[1]
        stored = response.json(); assert stored['ai_review']['provider'] == 'openai'
        assert len(stored['members']) == 2  # AI does not alter associations.
        details['a']['revision'] += 1
        assert client.get('/api/storylines/'+story['id']).json()['ai_review_stale']
        review.actions[0].incident_ids = ['invented']
        assert client.post(url, json={'revision': 2}).status_code == 502


def test_viewer_can_prepare_replay_but_not_edit_or_incur_ai_cost():
    path = '/api/storylines/story-'+'a'*32
    assert required_api_scope('POST', path+'/replay') == 'read'
    for suffix in ('/ai', '/export', '/suggestions', '/merge', '/split', ''):
        assert required_api_scope('POST', path+suffix) == 'admin'


def test_suggestion_strength_route_is_context_not_identity(api):
    from survng.app.storyline_routes import _suggestions
    from survng.app.config import CameraTransitionRoute
    client, manager, details, _ = api
    details['b']['camera_id'] = 'porch'
    manager.config.detector.tracking.camera_transition_routes = [CameraTransitionRoute(from_camera='gate',to_camera='porch',min_seconds=0,max_seconds=20)]
    story = create(client, ('a',))
    queries = SimpleNamespace(resolve_event=lambda m,n: details['b'],hydrate=lambda *a: [],with_faces=lambda *a: [])
    trace = {'matches':[{'event_id':2,'match_strength':'context_candidate','reasons':['Nearby person only']}]}
    with patch('survng.app.cross_camera_trace.build_cross_camera_trace',return_value=trace):
        result = _suggestions(manager, story, queries)
        candidate = result['items'][0]
        assert candidate['route'] is not None
        assert candidate['confidence'] == 'low'
        trace['matches'][0]['match_strength'] = 'confirmed_identity'
        assert _suggestions(manager, story, queries)['items'][0]['confidence'] == 'high'
        story['dismissed'] = {'b':candidate['evidence_fingerprint']}
        assert _suggestions(manager, story, queries)['items'][0]['dismissed']
        details['b']['revision'] += 1
        assert not _suggestions(manager, story, queries)['items'][0]['dismissed']


def test_corrected_aliases_remain_readable_without_duplicate_replay(api):
    client, manager, details, _ = api
    story = create(client)
    details['b'] = details['a']  # Existing scene correction redirects b to a.
    read = client.get('/api/storylines/'+story['id']).json()
    assert len(read['incidents']) == 1
    assert read['incidents'][0]['story_member_ids'] == ['a','b']
    replay = client.post('/api/storylines/'+story['id']+'/replay',json={'revision':1}).json()
    assert len([s for s in replay['shots'] if s['kind']=='video']) == 1


def test_recording_format_change_creates_replay_boundary():
    recording_rows = lambda *_: [{'start_epoch':100,'end_epoch':105,'stream_fingerprint':'h264'},{'start_epoch':105,'end_epoch':110,'stream_fingerprint':'h265'}]
    plan = build_replay_plan([incident('a',start=100,end=110)], recording_rows,padding=0)
    assert [(s['start'],s['end']) for s in plan['shots']] == [(100,105),(105,110)]


def test_ai_manager_lease_and_limiter_release_on_provider_failure(api):
    from survng.app.audit_ai import AuditAiError
    client, manager, _, _ = api
    manager.config.audit_ai.enabled = True; manager.config.audit_ai.api_key = 'test-only'
    story = create(client)
    with patch('survng.app.storyline_routes._ai_montage',return_value=[{'incident_id':'a'}]), patch('survng.app.storyline_routes.AuditAiAdvisor.analyze_structured',side_effect=AuditAiError('Provider unavailable')):
        for _ in range(2):
            assert client.post('/api/storylines/'+story['id']+'/ai',json={'revision':1}).status_code == 502


def test_manual_export_metadata_uses_source_bounds_not_sequence_endpoints(api):
    client, _, _, exported = api
    story = create(client, ('b','a'))
    assert client.post('/api/storylines/'+story['id']+'/export',json={'revision':1,'order':'member','padding':0}).status_code == 202
    assert exported[0]['start_epoch'] == 100
    assert exported[0]['end_epoch'] == 115
    assert exported[0]['options']['replay_plan']['shots'][0]['views'][0]['incident_id'] == 'b'


def _retained_gallery(store, root, event_id, key, at, color=80, camera='gate'):
    import cv2
    import numpy as np
    from datetime import datetime, timezone
    from survng.app.evidence_gallery import GalleryImage
    from survng.app.image_storage import EncodedImage
    image = np.full((180, 320, 3), color, dtype=np.uint8)
    encoded = EncodedImage(cv2.imencode('.jpg', image)[1].tobytes())
    def write(payload, captured):
        path = root/'snapshots'/f'{key}.jpg'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload.data)
        return str(path)
    store.retain_scene_evidence_images(event_id, [GalleryImage(key, at, encoded, {
        'camera_id': camera, 'captured_epoch': at, 'width':320, 'height':180,
        'frame_timestamp_exact': True, 'objects':[{'label':'cat', 'confidence':.52,
            'confidence_eligible':False, 'incident_eligible':False}], 'role':'gallery'})], write)
    return store.scene_incident(event_id=event_id)


def _gallery_scene(root, cover=False):
    import cv2
    import numpy as np
    from datetime import datetime, timezone
    from survng.app.events import EventStore
    store = EventStore(root)
    path = root/'snapshots'/'cover.jpg'
    path.parent.mkdir(exist_ok=True)
    if cover:
        cv2.imwrite(str(path), np.full((180,320,3),20,dtype=np.uint8))
    event = store.add_event('gate','motion',created_at=datetime.fromtimestamp(1000,timezone.utc).isoformat(),
        snapshot_path=str(path) if cover else '', objects_json=json.dumps([{
            'label':'dog','confidence':.8,'incident_eligible':True,
            'box':{'x1':10,'y1':10,'x2':100,'y2':150},
            'detection_frame_width':320,'detection_frame_height':180}]))
    manager = SimpleNamespace(events=store,storage_dir=root,media_storage=None)
    return manager, event


def test_ai_montage_includes_cover_and_actual_gallery_camera_time(tmp_path):
    manager, event = _gallery_scene(tmp_path, cover=True)
    detail = _retained_gallery(manager.events, tmp_path, event['id'], 'porch-frame',1003,camera='porch')
    evidence = _ai_montage(manager,[detail],tmp_path/'montage.png')
    assert [item['kind'] for item in evidence] == ['cover','gallery']
    assert evidence[1]['camera_id'] == 'porch'
    assert evidence[1]['captured_at'] == '1970-01-01T00:16:43+00:00'
    assert evidence[1]['frame_timestamp_exact'] is True
    assert evidence[1]['detections'][0]['incident_eligible'] is False
    assert evidence[1]['image_id'] == detail['evidence_images'][0]['id']
    assert 'snapshot_path' not in json.dumps(evidence)
    assert 'snapshot_url' not in json.dumps(evidence)


def test_ai_gallery_only_scene_succeeds_through_real_route(api,tmp_path):
    import cv2
    client, manager, _, _ = api
    real, event = _gallery_scene(tmp_path/'real')
    detail = _retained_gallery(real.events,tmp_path/'real',event['id'],'gallery-only',1002)
    manager.events = real.events; manager.storage_dir = real.storage_dir
    manager.config.audit_ai.enabled = True; manager.config.audit_ai.api_key = 'test-only'
    story = create(client,(detail['id'],))
    review = StoryAiReview(title='Animal visit',summary='Retained evidence.',actions=[],suggested_relationships=[])
    def analyze(image,prompt,**kwargs):
        assert cv2.imread(str(image)).shape == (300,1440,3)
        frames = json.loads(prompt)['selected_incidents']
        assert len(frames) == 1 and frames[0]['kind'] == 'gallery'
        return review
    with patch('survng.app.storyline_routes.AuditAiAdvisor.analyze_structured',side_effect=analyze):
        response = client.post('/api/storylines/'+story['id']+'/ai',json={'revision':1})
    assert response.status_code == 200, response.text
    result = response.json()['ai_review']
    assert result['reviewed_incident_ids'] == [detail['id']]
    assert result['reviewed_images'][0]['image_id'] == detail['evidence_images'][0]['id']
    assert result['reviewed_images'][0]['camera_id'] == 'gate'


def test_ai_skips_corrupt_cover_and_missing_gallery_then_uses_retained_frame(tmp_path):
    manager,event = _gallery_scene(tmp_path,cover=True)
    (tmp_path/'snapshots'/'cover.jpg').write_bytes(b'corrupt')
    _retained_gallery(manager.events,tmp_path,event['id'],'expired',1001)
    detail = _retained_gallery(manager.events,tmp_path,event['id'],'retained',1002,color=120)
    (tmp_path/'snapshots'/'expired.jpg').unlink()
    evidence = _ai_montage(manager,[detail],tmp_path/'montage.png')
    assert len(evidence) == 1
    assert evidence[0]['image_id'] == detail['evidence_images'][1]['id']


def test_ai_montage_deduplicates_frames_and_shares_twelve_image_budget(tmp_path):
    import cv2
    import numpy as np
    paths = []
    for n in range(13):
        path=tmp_path/f'{n}.jpg'
        cv2.imwrite(str(path),np.full((180,320,3),n*10,dtype=np.uint8)); paths.append(str(path))
    events = SimpleNamespace(get=lambda n:{'id':n,'snapshot_path':paths[n],'camera_id':'gate','created_at':100+n},
        scene_evidence_image=lambda key:{'snapshot_path':paths[12],'camera_id':'gate','captured_epoch':200})
    manager=SimpleNamespace(events=events,storage_dir=tmp_path,media_storage=None)
    details=[incident(str(n)) | {'representative_event_id':n,
        'evidence_images':[{'id':'extra','frame_timestamp_exact':True,'objects':[]}]} for n in range(12)]
    evidence=_ai_montage(manager,details,tmp_path/'montage.png')
    assert len(evidence) == 12
    assert [item['incident_id'] for item in evidence] == [str(n) for n in range(12)]
    # Different paths containing identical frames must not spend extra slots.
    duplicate=tmp_path/'copy.jpg'; duplicate.write_bytes(Path(paths[0]).read_bytes())
    events.scene_evidence_image=lambda key:{'snapshot_path':str(duplicate),'camera_id':'gate','captured_epoch':200}
    assert len(_ai_montage(manager,details[:1],tmp_path/'single.png')) == 1


def test_gallery_change_during_ai_review_rejects_stale_result(api,tmp_path):
    client,manager,_,_=api
    real,event=_gallery_scene(tmp_path/'real')
    detail=_retained_gallery(real.events,tmp_path/'real',event['id'],'first',1001)
    manager.events=real.events;manager.storage_dir=real.storage_dir
    manager.config.audit_ai.enabled=True;manager.config.audit_ai.api_key='test-only'
    story=create(client,(detail['id'],))
    def analyze(*args,**kwargs):
        _retained_gallery(real.events,tmp_path/'real',event['id'],'new-evidence',1003,color=120)
        return StoryAiReview(title='Stale',summary='Old frame.',actions=[],suggested_relationships=[])
    with patch('survng.app.storyline_routes.AuditAiAdvisor.analyze_structured',side_effect=analyze):
        response=client.post('/api/storylines/'+story['id']+'/ai',json={'revision':1})
    assert response.status_code == 409,response.text
    assert manager.storylines.get(story['id'])['ai_review'] is None


def test_identical_views_of_distinct_incidents_keep_both_citations(tmp_path):
    import cv2
    import numpy as np
    path=tmp_path/'same-view.jpg'
    cv2.imwrite(str(path),np.full((180,320,3),100,dtype=np.uint8))
    events=SimpleNamespace(get=lambda n:{'id':n,'snapshot_path':str(path),'camera_id':'gate','created_at':100+n})
    manager=SimpleNamespace(events=events,storage_dir=tmp_path,media_storage=None)
    details=[incident(str(n)) | {'representative_event_id':n} for n in range(2)]
    assert [item['incident_id'] for item in _ai_montage(manager,details,tmp_path/'montage.png')] == ['0','1']

from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

from survng.app.config import AppConfig, CameraConfig, DetectorConfig
from survng.app.native_activity import NativeActivity
from survng.app.native_admission import NativeAdmission, context_crop
from survng.app.native_evidence import Candidate
from survng.app.native_main_frame import NativeMainFrameVerifier
from survng.app.live_detections import DetectionSnapshot


def admission_from_evidence(evidence):
    config = getattr(
        evidence,
        "config",
        AppConfig(
            cameras=[
                {
                    "id": "front",
                    "name": "Front",
                    "stream_url": "rtsp://unused.invalid",
                }
            ]
        ),
    )
    main_frames = NativeMainFrameVerifier(
        lambda: config,
        Mock(),
        Counter(),
    )
    main_frames.read_frame = evidence.read_frame
    main_frames.project_main = evidence.project_main
    main_frames.detector = evidence.verifier
    return NativeAdmission(main_frames)


def setup_activity():
    events = Mock(); events.add_event.side_effect = [{'id': 1}, {'id': 2}]
    a = NativeActivity(CameraConfig(id='front', name='Front', stream_url='rtsp://unused.invalid'),
                       DetectorConfig(), events, Mock(), Mock(return_value='preview'))
    a.admission = Mock(); a.admission.poll.return_value = None
    a.nominate = Mock(); a.verified_snapshot = Mock(return_value='main.webp')
    return a


def feed(a, seq, objects=None):
    obj = {'label': 'dog', 'confidence': .8, 'box': {'x1': 20, 'y1': 20, 'x2': 40, 'y2': 60},
           'native_track_id': 9, 'detection_provenance': 'native_fresh_detection'}
    now = 100+seq/5
    a.consume(DetectionSnapshot(seq/5,seq,100,100,tuple([obj] if objects is None else objects),'session', 'native_fresh_detection',now), now=now, epoch=1000+seq/5)


def test_no_event_or_notification_before_verification_and_retains_departed_object():
    a = setup_activity()
    for seq in range(1, 8): feed(a, seq)
    a.events.add_event.assert_not_called(); a.publish.assert_not_called()
    assert len(a._verification_pending) == 1
    for seq in range(8, 70): feed(a, seq, [])
    assert not a.tracks  # Awaiting verification outlives live tracking context.
    a.admission.poll.return_value = {'status': 'confirmed', 'votes': ['confirmed','confirmed','negative'],
                                   'cover': (None, {'label':'dog','box':{}},1000.4)}
    a.tick(now=114)
    assert a.events.add_event.call_count == 1
    kwargs = a.events.add_event.call_args.kwargs
    assert kwargs['snapshot_path'] == 'main.webp'
    assert kwargs['created_at'].startswith('1970-01-01T00:16:40.')
    histories = [call.args[1] for call in a.events.update_native_incident_state.call_args_list]
    assert len(histories[0]['tracks'][0]['box_history']) == 7
    assert histories[-1]['state'] == 'complete'
    assert a.counts['verification_confirmed'] == 1


@pytest.mark.parametrize('status', ['rejected', 'unverified'])
def test_negative_or_unavailable_never_alerts(status):
    a=setup_activity(); feed(a,1); feed(a,2)
    a.admission.poll.return_value={'status':status}
    a.tick(now=101)
    for seq in range(3,12): feed(a,seq)
    a.events.add_event.assert_not_called(); a.publish.assert_not_called()
    assert a.counts['verification_'+status] == 1


def test_reset_cancels_pending_and_late_results_cannot_create_incident():
    a=setup_activity(); feed(a,1); feed(a,2)
    a.finish('policy_changed',now=101)
    a.admission.cancel.assert_called_once()
    a.admission.poll.return_value={'status':'confirmed'}
    a.tick(now=102)
    a.events.add_event.assert_not_called()


def test_verifier_requires_multiple_clear_views_and_spatial_match():
    rng=np.random.default_rng(7)
    main=rng.integers(0,255,(600,800,3),dtype=np.uint8)
    box={'x1':300,'y1':200,'x2':340,'y2':250}
    obj={'label':'dog','box':box,'confidence':.8}
    config=AppConfig(cameras=[{'id':'front','name':'Front','stream_url':'rtsp://unused.invalid'}])
    evidence=SimpleNamespace(config=config,read_frame=Mock(return_value=main),project_main=Mock(return_value=[obj]), verifier=Mock())
    service=admission_from_evidence(evidence)
    samples=[Candidate(i,main[::2,::2], [obj],0) for i in range(3)]
    crop,left,top=context_crop(main,box)
    detection=dict(obj,box={k:v-(left if k.startswith('x') else top) for k,v in box.items()})
    evidence.verifier.detect.side_effect=[[detection],[detection],[]]
    assert service.verify('front',samples)['status']=='confirmed'
    assert evidence.read_frame.call_count == 1
    evidence.verifier.detect.assert_called_once()
    evidence.verifier.detect.side_effect=[[] for _ in range(15)]
    assert service.verify('front',samples)['status']=='rejected'
    evidence.verifier.detect.side_effect=[[detection],[],[]]
    assert service.verify('front',samples)['status']=='confirmed'
    evidence.verifier.detect.side_effect=[[dict(detection,confidence=.4)] for _ in range(15)]
    assert service.verify('front',samples)['status']=='unverified'
    fragment = dict(detection, confidence=.95, box={**detection['box'], 'x2': detection['box']['x1']+12, 'y2': detection['box']['y1']+15})
    evidence.verifier.detect.side_effect=[[fragment] for _ in range(15)]
    assert service.verify('front',samples)['status']=='unverified'
    evidence.verifier.detect.side_effect=None
    evidence.read_frame.return_value=None
    assert service.verify('front',samples)['status']=='unverified'
    evidence.read_frame.return_value=main
    evidence.project_main.return_value=[]
    assert service.verify('front',samples)['status']=='unverified'
    assert crop.shape[0] >= 192 and crop.shape[1] >=192


def test_queue_is_bounded_and_cancel_discards_inflight_result():
    import threading
    service=NativeAdmission(None)
    for i in range(40): service.offer(str(i),'front',100,None,{},(100,100))
    assert len(service.jobs)==32
    for i in range(40): service.cancel(str(i))
    entered, release=threading.Event(), threading.Event()
    def verify(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return {'status':'rejected','reason':'test'}
    service.verify=verify
    service.offer('old','front',100,None,{},(100,100))
    service.jobs['old']['due']=0
    service.start()
    assert entered.wait(2)
    service.cancel('old')
    release.set()
    service.stop()
    assert service.poll('old') is None


def test_cancelled_nomination_does_not_start_inference_after_decode():
    import threading
    cancelled = threading.Event()
    main = np.zeros((20, 20, 3), np.uint8)
    def read(*args):
        cancelled.set()
        return main
    evidence = SimpleNamespace(read_frame=Mock(side_effect=read), project_main=Mock(), verifier=Mock())
    service = admission_from_evidence(evidence)
    samples = [Candidate(i, main[::2, ::2], [], 0) for i in range(3)]
    result = service.verify('front', samples, cancelled=cancelled)
    assert result['status'] == 'unverified'
    evidence.read_frame.assert_called_once()
    evidence.project_main.assert_not_called()
    evidence.verifier.detect.assert_not_called()


def test_nomination_requires_temporally_distinct_samples():
    service=NativeAdmission(None)
    obj={'box':{'x1':10,'x2':20,'y1':10,'y2':20}}
    image=np.zeros((100,100,3),np.uint8)
    for t in (1,1,1.1,1.2,1.5,2,3): service.offer('track','front',t,image,obj,(100,100))
    assert [s.epoch for s in service.jobs['track']['samples']]==[1,1.5,3]


def test_nomination_shares_immutable_frames_but_owns_mutable_inputs():
    service = NativeAdmission(None)
    obj = {'box': {'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}}
    writable = np.ones((20, 20, 3), np.uint8)
    service.offer('mutable', 'front', 1, writable, obj, (20, 20))
    saved = service.jobs['mutable']['samples'][0].image
    writable[:] = 0
    assert saved.all() and not saved.flags.writeable
    immutable = np.ones_like(writable)
    immutable.setflags(write=False)
    for token in ('one', 'two'):
        service.offer(token, 'front', 1, immutable, obj, (20, 20))
        assert service.jobs[token]['samples'][0].image is immutable


def test_rejected_location_can_be_reverified_after_object_moves():
    a=setup_activity(); feed(a,1); feed(a,2)
    a.admission.poll.return_value={'status':'rejected'}
    a.tick(now=101)
    a.admission.poll.return_value=None
    key = next(key for key, track in a.registry.tracks.items()
               if track.get('native_track_id') == 9)
    old = a._activity_states[key]['_verification_token']
    for seq in range(3,6): feed(a,seq)
    assert not a._verification_pending
    obj={'label':'dog','confidence':.8,'box':{'x1':60,'y1':20,'x2':80,'y2':60},
         'native_track_id':9,'detection_provenance':'native_fresh_detection'}
    feed(a,6,[obj])
    assert len(a._verification_pending)==1
    assert a._activity_states[key]['_verification_token'] != old


@pytest.mark.parametrize('offset', [.5, -.5, 1., -1.])
def test_verification_finds_time_skewed_main_pose_and_preserves_frame_time(offset):
    main = np.random.default_rng(9).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    obj = {'label': 'person', 'confidence': .8,
           'box': {'x1': 300, 'y1': 200, 'x2': 340, 'y2': 280}}
    config = AppConfig(cameras=[{'id': 'front', 'name': 'Front', 'stream_url': 'rtsp://unused.invalid'}])
    selected = main.copy()
    evidence = SimpleNamespace(
        config=config,
        read_frame=Mock(side_effect=lambda camera, epoch, source: selected if epoch == 100 + offset else main),
        project_main=Mock(side_effect=lambda candidate, frame: [obj] if frame is selected else []),
        verifier=Mock())
    _, left, top = context_crop(main, obj['box'])
    evidence.verifier.detect.return_value = [dict(obj, box={
        k: v - (left if k.startswith('x') else top) for k, v in obj['box'].items()})]
    service = admission_from_evidence(evidence)
    result = service.verify('front', [Candidate(100, main[::2, ::2], [obj], 0)])
    assert result['status'] == 'confirmed'
    assert result['votes'] == ['confirmed']
    assert result['cover'][0] is selected
    assert result['cover'][1]['frame_captured_at_epoch'] == 100 + offset
    assert result['cover'][2] == 100 + offset
    evidence.verifier.detect.assert_called_once()
    assert evidence.read_frame.call_count <= 5


def test_time_window_does_not_bypass_geometry_alignment():
    main = np.random.default_rng(9).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    evidence = SimpleNamespace(read_frame=Mock(return_value=main),
                               project_main=Mock(return_value=[]), verifier=Mock())
    result = admission_from_evidence(evidence).verify('front', [Candidate(100, main[::2, ::2], [], 0)])
    assert result['status'] == 'unverified'
    assert result['votes'] == ['unaligned']
    assert [call.args[1] for call in evidence.read_frame.call_args_list] == [100, 100.5, 99.5, 101, 99]
    evidence.verifier.detect.assert_not_called()


def test_cancel_during_time_window_decode_stops_before_matching_or_inference():
    import threading
    cancelled = threading.Event()
    main = np.ones((40, 40, 3), np.uint8)
    def read(camera, epoch, source):
        if epoch != 100:
            cancelled.set()
        return main
    evidence = SimpleNamespace(read_frame=Mock(side_effect=read),
                               project_main=Mock(return_value=[]), verifier=Mock())
    result = admission_from_evidence(evidence).verify(
        'front', [Candidate(100, main[::2, ::2], [], 0)], cancelled=cancelled)
    assert result == {'status': 'unverified', 'reason': 'stopped'}
    assert evidence.read_frame.call_count == 2
    evidence.project_main.assert_called_once()
    evidence.verifier.detect.assert_not_called()


def nominate_walk_with_id_change(a, *, second_start=8):
    for seq in range(1, 8):
        feed(a, seq)
    for seq in range(8, second_start):
        feed(a, seq, [])
    replacement = {'label': 'dog', 'confidence': .8,
                   'box': {'x1': 40, 'y1': 20, 'x2': 60, 'y2': 60},
                   'native_track_id': 10, 'detection_provenance': 'native_fresh_detection'}
    for seq in range(second_start, second_start + 8):
        feed(a, seq, [replacement])
    tokens = list(a._verification_pending)
    for seq in range(second_start + 8, second_start + 70):
        feed(a, seq, [])
    return tokens


def test_delayed_verified_id_change_continues_one_incident():
    a = setup_activity()
    first, second = nominate_walk_with_id_change(a)
    results = {first: {'status': 'confirmed'}}
    a.admission.poll.side_effect = lambda token: results.pop(token, None)
    a.tick(now=116)
    assert a.event_id == 1  # Await the adjacent track even after activity timeout.
    assert a.events.add_event.call_count == 1
    assert [t['native_track_id'] for t in a.inventory.tracking_tracks()] == [9]
    results[second] = {'status': 'confirmed'}
    a.tick(now=117)
    assert a.events.add_event.call_count == 1
    final = a.events.update_native_incident_state.call_args.args[1]
    assert final['state'] == 'complete'
    assert [t['native_track_id'] for t in final['tracks']] == [9, 10]
    assert final['tracks'][0]['first_seen'] < final['tracks'][1]['first_seen']
    assert a.event_id is None


@pytest.mark.parametrize('status', ['rejected', 'unverified', 'deadline'])
def test_failed_pending_continuation_closes_without_extending_confirmed_history(status):
    a = setup_activity()
    first, second = nominate_walk_with_id_change(a)
    results = {first: {'status': 'confirmed'}}
    a.admission.poll.side_effect = lambda token: results.pop(token, None)
    a.tick(now=116)
    assert a.event_id == 1
    if status == 'deadline':
        a.tick(now=300)
        a.admission.cancel.assert_called_with(second)
    else:
        results[second] = {'status': status}
        a.tick(now=117)
    assert a.events.add_event.call_count == 1
    final = a.events.update_native_incident_state.call_args.args[1]
    assert [t['native_track_id'] for t in final['tracks']] == [9]
    assert a.event_id is None


def test_later_verification_result_waits_for_earlier_activity():
    a = setup_activity()
    first, second = nominate_walk_with_id_change(a)
    results = {second: {'status': 'confirmed'}}
    a.admission.poll.side_effect = lambda token: results.pop(token, None)
    a.tick(now=116)
    a.events.add_event.assert_not_called()
    assert second in results
    results[first] = {'status': 'confirmed'}
    a.tick(now=117)
    assert a.events.add_event.call_count == 1
    final = a.events.update_native_incident_state.call_args.args[1]
    assert [t['native_track_id'] for t in final['tracks']] == [9, 10]


def test_separated_activity_stays_separate_even_when_results_arrive_together():
    a = setup_activity()
    first, second = nominate_walk_with_id_change(a, second_start=50)
    results = {first: {'status': 'confirmed'}, second: {'status': 'confirmed'}}
    a.admission.poll.side_effect = lambda token: results.pop(token, None)
    a.tick(now=125)
    assert a.events.add_event.call_count == 2
    completed = [call.args[1] for call in a.events.update_native_incident_state.call_args_list
                 if call.args[1]['state'] == 'complete']
    assert [[t['native_track_id'] for t in episode['tracks']] for episode in completed] == [[9], [10]]


def test_renewed_candidate_cannot_hold_an_inactive_incident_forever():
    a = setup_activity()
    first, second = nominate_walk_with_id_change(a)
    results = {first: {'status': 'confirmed'}}
    a.admission.poll.side_effect = lambda token: results.pop(token, None)
    a.tick(now=116)
    assert a.event_id == 1
    # A fresh retry of an old track must not restart the episode's hold limit.
    a._verification_pending[second]['started'] = 240
    a.last_fresh = 252
    a.health = 'healthy'
    a.tick(now=252)
    assert a.event_id is None
    assert second in a._verification_pending
    final = a.events.update_native_incident_state.call_args.args[1]
    assert final['state'] == 'complete'
    assert [t['native_track_id'] for t in final['tracks']] == [9]


def test_recent_clear_view_can_confirm_after_early_negative_views():
    main = np.random.default_rng(8).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    obj = {'label': 'person', 'confidence': .9,
           'box': {'x1': 300, 'y1': 200, 'x2': 340, 'y2': 280}}
    config = AppConfig(cameras=[{'id': 'front', 'name': 'Front', 'stream_url': 'rtsp://unused.invalid'}])
    evidence = SimpleNamespace(config=config, read_frame=Mock(return_value=main),
                               project_main=Mock(return_value=[obj]), verifier=Mock())
    _, left, top = context_crop(main, obj['box'])
    detected = dict(obj, box={k: v - (left if k.startswith('x') else top) for k, v in obj['box'].items()})
    service = admission_from_evidence(evidence)
    for epoch in range(1, 8):
        service.offer('track', 'front', epoch, main[::2, ::2], obj, (400, 300))
    evidence.verifier.detect.side_effect = [[] for _ in range(10)] + [[detected]]
    samples = service.jobs['track']['samples']
    assert [s.epoch for s in samples] == [1, 2, 7]
    result = service.verify('front', samples)
    assert result['status'] == 'confirmed'
    assert result['votes'] == ['negative', 'negative', 'confirmed']
    assert result['cover'][2] == 7


def test_new_view_arriving_during_negative_verification_is_not_discarded():
    import threading
    entered, release = threading.Event(), threading.Event()
    service = NativeAdmission(None)
    image = np.ones((20, 20, 3), np.uint8)
    obj = {'box': {'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}}
    for epoch in (1, 2, 3):
        service.offer('track', 'front', epoch, image, obj, (20, 20))
    def verify(camera, samples, **kwargs):
        if samples[-1].epoch == 3:
            entered.set()
            assert release.wait(2)
            return {'status': 'rejected', 'reason': 'test', 'votes': ['negative'] * 3}
        return {'status': 'confirmed', 'reason': 'test', 'votes': ['confirmed']}
    service.verify = verify
    service.jobs['track']['due'] = 0
    service.start()
    try:
        assert entered.wait(2)
        service.offer('track', 'front', 4, image, obj, (20, 20))
        with service.condition:
            service.jobs['track']['due'] = 0
        release.set()
        # A condition wait releases the worker lock; avoid polling/sleep races.
        import time
        deadline = time.monotonic() + 2
        with service.condition:
            while 'track' not in service.results and time.monotonic() < deadline:
                service.condition.wait(.01)
        assert service.poll('track')['status'] == 'confirmed'
    finally:
        release.set()
        service.stop()


def test_ambiguous_pose_checks_nearby_time_before_deciding():
    main = np.random.default_rng(8).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    obj = {'label': 'person', 'confidence': .9,
           'box': {'x1': 300, 'y1': 200, 'x2': 340, 'y2': 280}}
    config = AppConfig(cameras=[{'id': 'front', 'name': 'Front', 'stream_url': 'rtsp://unused.invalid'}])
    evidence = SimpleNamespace(config=config, read_frame=Mock(return_value=main),
                               project_main=Mock(return_value=[obj]), verifier=Mock())
    _, left, top = context_crop(main, obj['box'])
    actual = dict(obj, box={k: v - (left if k.startswith('x') else top) for k, v in obj['box'].items()})
    fragment = dict(actual, box={**actual['box'], 'x2': actual['box']['x1'] + 5})
    evidence.verifier.detect.side_effect = [[fragment], [], [actual]]
    result = admission_from_evidence(evidence).verify('front', [Candidate(100, main[::2, ::2], [obj], 0)])
    assert result['status'] == 'confirmed'
    assert result['votes'] == ['confirmed']
    assert result['checks'] == [{'epoch': 100, 'votes': ['ambiguous', 'negative', 'confirmed']}]
    assert result['cover'][2] == 99.5

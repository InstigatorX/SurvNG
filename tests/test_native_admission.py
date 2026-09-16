from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

from survng.app.config import AppConfig, CameraConfig, DetectorConfig
from survng.app.native_activity import NativeActivity
from survng.app.native_admission import NativeAdmission, context_crop
from survng.app.native_evidence import Candidate
from survng.app.live_detections import DetectionSnapshot


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
    histories = [call.args[1] for call in a.events.update_object_tracking.call_args_list]
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
    evidence=SimpleNamespace(config=config,read_frame=Mock(return_value=main),match_main=Mock(return_value=[obj]), verifier=Mock())
    service=NativeAdmission(evidence)
    samples=[Candidate(i,main[::2,::2], [obj],0) for i in range(3)]
    crop,left,top=context_crop(main,box)
    detection=dict(obj,box={k:v-(left if k.startswith('x') else top) for k,v in box.items()})
    evidence.verifier.detect.side_effect=[[detection],[detection],[]]
    assert service.verify('front',samples)['status']=='confirmed'
    evidence.verifier.detect.side_effect=[[],[],[]]
    assert service.verify('front',samples)['status']=='rejected'
    evidence.verifier.detect.side_effect=[[detection],[],[]]
    assert service.verify('front',samples)['status']=='confirmed'
    evidence.verifier.detect.side_effect=[[dict(detection,confidence=.4)],[],[]]
    assert service.verify('front',samples)['status']=='unverified'
    evidence.verifier.detect.side_effect=None
    evidence.read_frame.return_value=None
    assert service.verify('front',samples)['status']=='unverified'
    evidence.read_frame.return_value=main
    evidence.match_main.return_value=[]
    assert service.verify('front',samples)['status']=='unverified'
    assert crop.shape[0] >= 192 and crop.shape[1] >=192


def test_queue_is_bounded_and_cancel_discards_inflight_result():
    import threading
    service=NativeAdmission(None)
    for i in range(40): service.offer(str(i),'front',100,None,{},(100,100))
    assert len(service.jobs)==32
    for i in range(40): service.cancel(str(i))
    entered, release=threading.Event(), threading.Event()
    def verify(*args):
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


def test_nomination_requires_temporally_distinct_samples():
    service=NativeAdmission(None)
    obj={'box':{'x1':10,'x2':20,'y1':10,'y2':20}}
    image=np.zeros((100,100,3),np.uint8)
    for t in (1,1,1.1,1.2,1.5,2,3): service.offer('track','front',t,image,obj,(100,100))
    assert [s.epoch for s in service.jobs['track']['samples']]==[1,1.5,2]


def test_rejected_location_can_be_reverified_after_object_moves():
    a=setup_activity(); feed(a,1); feed(a,2)
    a.admission.poll.return_value={'status':'rejected'}
    a.tick(now=101)
    a.admission.poll.return_value=None
    old=a.tracks[(9,'dog')]['_verification_token']
    for seq in range(3,6): feed(a,seq)
    assert not a._verification_pending
    obj={'label':'dog','confidence':.8,'box':{'x1':60,'y1':20,'x2':80,'y2':60},
         'native_track_id':9,'detection_provenance':'native_fresh_detection'}
    feed(a,6,[obj])
    assert len(a._verification_pending)==1
    assert a.tracks[(9,'dog')]['_verification_token'] != old

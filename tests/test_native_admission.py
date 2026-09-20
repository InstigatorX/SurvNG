import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock
import time
import numpy as np
import pytest

from survng.app.config import AppConfig, CameraConfig, DetectorConfig
from survng.app.native_activity import NativeActivity
from survng.app.native_admission import NativeAdmission, context_crop
from survng.app.native_evidence import Candidate
from survng.app.native_main_frame import NativeMainFrameVerifier, _VerifierScheduler
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


def setup_activity(**detector_kwargs):
    events = Mock()
    events.add_event.side_effect = [{"id": 1}, {"id": 2}]
    events.open_incident = Mock(
        side_effect=[{"id": 10, "observation_count": 1}, {"id": 11, "observation_count": 1}]
    )
    a = NativeActivity(
        CameraConfig(id="front", name="Front", stream_url="rtsp://unused.invalid"),
        DetectorConfig(**detector_kwargs),
        events,
        Mock(),
        Mock(return_value="preview"),
    )
    a.admission = Mock()
    a.nominate = Mock()
    a.verified_snapshot = Mock(return_value="main.webp")
    return a


def feed(a, seq, objects=None):
    obj = {
        "label": "dog",
        "confidence": 0.8,
        "box": {"x1": 20, "y1": 20, "x2": 40, "y2": 60},
        "detection_provenance": "native_fresh_detection",
    }
    now = 100 + seq / 5
    a.consume(
        DetectionSnapshot(
            seq / 5,
            seq,
            100,
            100,
            tuple([obj] if objects is None else objects),
            "session",
            "native_fresh_detection",
            now,
        ),
        now=now,
        epoch=1000 + seq / 5,
    )


def test_scene_activity_opens_incident_on_first_fresh_detection():
    a = setup_activity()
    feed(a, 1)
    assert a.events.add_event.call_count == 1
    assert a.event_id == 1
    assert a.incident_id == 10
    assert a.status()["active"] is True


def test_tracking_classes_filter_scene_activity():
    a = setup_activity(native={"tracking_classes": ["person"]})
    feed(a, 1)  # dog
    a.events.add_event.assert_not_called()
    person = {
        "label": "person",
        "confidence": 0.9,
        "box": {"x1": 10, "y1": 10, "x2": 30, "y2": 50},
        "detection_provenance": "native_fresh_detection",
    }
    feed(a, 2, [person])
    assert a.events.add_event.call_count == 1


def test_multi_object_scene_joins_one_incident():
    a = setup_activity()
    feed(
        a,
        1,
        [
            {
                "label": "dog",
                "confidence": 0.9,
                "box": {"x1": 20, "y1": 20, "x2": 40, "y2": 60},
                "detection_provenance": "native_fresh_detection",
            },
            {
                "label": "person",
                "confidence": 0.88,
                "box": {"x1": 70, "y1": 10, "x2": 90, "y2": 80},
                "detection_provenance": "native_fresh_detection",
            },
        ],
    )
    assert a.events.add_event.call_count == 1
    participants = a.events.update_native_incident_state.call_args.args[2]
    assert {item["label"] for item in participants} == {"dog", "person"}


def test_inactivity_closes_scene_incident():
    a = setup_activity()
    feed(a, 1)
    assert a.event_id == 1
    for seq in range(2, 40):
        feed(a, seq, [])
    assert a.event_id is None
    final = a.events.update_native_incident_state.call_args.args[1]
    assert final["state"] == "complete"


def test_policy_reset_clears_open_scene_incident():
    a = setup_activity()
    feed(a, 1)
    assert a.event_id == 1
    a.finish("policy_changed", now=101)
    assert a.event_id is None


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


def test_admission_waiter_preempts_cover_waiter():
    import threading
    scheduler = _VerifierScheduler()
    order = []
    cover_started = threading.Event()
    admission_started = threading.Event()

    def cover():
        cover_started.set()
        with scheduler.lease("cover"):
            order.append("cover")

    def admission():
        admission_started.set()
        with scheduler.lease("admission"):
            order.append("admission")

    with scheduler.lease("cover"):
        cover_thread = threading.Thread(target=cover)
        admission_thread = threading.Thread(target=admission)
        cover_thread.start()
        assert cover_started.wait(1)
        admission_thread.start()
        assert admission_started.wait(1)
        with scheduler._condition:
            deadline = time.monotonic() + 1
            while scheduler._admission_waiters < 1 and time.monotonic() < deadline:
                scheduler._condition.wait(.01)
            assert scheduler._admission_waiters == 1

    admission_thread.join(1)
    cover_thread.join(1)
    assert order == ["admission", "cover"]


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

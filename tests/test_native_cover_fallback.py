"""Promotion regressions: exact live evidence survives until a safe main cover exists."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import pytest
from fastapi import HTTPException

from survng.app.config import CameraConfig
from survng.app.native_admission import context_crop
from survng.app.native_evidence import Candidate, event_tracking
from survng.app.native_routes import create_native_router
from tests.test_native_evidence import fixture


def setup(tmp_path, *, state='complete', aligned=False):
    service, events, event, live, obj, tracking = fixture(tmp_path)
    service.config.cameras = [CameraConfig(id='test', name='Test', stream_url='rtsp://unused.invalid',
                                           native_same_field_of_view=True, record_sub=False)]
    service.config.detector.native.verification_enabled = False
    tracking.update(state=state, native_session='session')
    tracking['tracks'][0]['label'] = 'person'
    if aligned:
        tracking['recording_alignment'] = {'source': 'main', 'verified': True,
                                          'offset_seconds': .4, 'mean_iou': .9}
    events.update_object_tracking(event['id'], tracking)
    main = cv2.resize(live, (1280, 720))
    projected = dict(obj, box={k: v*2 for k, v in obj['box'].items()})
    service.project_main = Mock(return_value=[projected])
    service.read_frame = Mock(side_effect=lambda camera, epoch, source: main if source == 'main' else None)
    _, left, top = context_crop(main, projected['box'])
    actual = {k: v + (8 if k.startswith('x') else 4) for k, v in projected['box'].items()}
    detected = dict(obj, box={k: v-(left if k.startswith('x') else top) for k, v in actual.items()})
    service.verifier.detect = Mock(return_value=[detected])
    return service, events, event, Candidate(100, live, [obj], 5), main, actual, tracking


@pytest.mark.parametrize('state', ['active', 'complete'])
def test_short_incident_can_promote_without_clock_alignment(tmp_path, state):
    service, events, event, candidate, main, actual, tracking = setup(tmp_path, state=state)
    result = service.process(event['id'], [candidate], recorded_history=True)
    assert result['status'] == 'promoted'
    assert result['reason'] == 'main_frame_verified'
    assert result['retained_live_candidates'] == 1
    after = events.get(event['id'])
    stored, saved_tracking = event_tracking(after)
    chosen = next(x for x in stored if x.get('snapshot_visible') is True)
    assert chosen['box'] == actual
    assert chosen['box'] != {k: v*2 for k, v in candidate.objects[0]['box'].items()}
    assert chosen['box_provenance'] == 'detected_in_main'
    assert chosen['native_cover_verified'] is True
    assert chosen['native_alignment']['method'] == 'main_frame_verified'
    assert 'recording_offset_seconds' not in chosen['native_alignment']
    assert saved_tracking == tracking
    assert 'recording_alignment' not in saved_tracking
    assert not service.config.detector.native.verification_enabled
    assert chosen['detection_frame_width'] == main.shape[1]
    assert chosen['detection_frame_height'] == main.shape[0]
    assert not any(call.args[2] == 'live' for call in service.read_frame.call_args_list)
    attempts = events.evidence_attempts(event['id'])
    assert attempts[-1]['source'] == 'native_cover'
    assert attempts[-1]['reason'] == 'main_frame_verified'


def test_fallback_box_and_image_use_the_same_neighboring_timestamp(tmp_path):
    service, events, event, candidate, main, actual, _ = setup(tmp_path)
    selected = main.copy()
    service.read_frame = Mock(side_effect=lambda camera, epoch, source: selected if epoch == 100.5 else main)
    service.verifier.detect.side_effect = [[], service.verifier.detect.return_value]
    writer = service.image_writer.write
    written = []
    def write(directory, name, image):
        written.append(image)
        return writer(directory, name, image)
    service.image_writer.write = write
    assert service.process(event['id'], [candidate])['status'] == 'promoted'
    assert len(written) == 1 and written[0] is selected
    stored, tracking = event_tracking(events.get(event['id']))
    chosen = next(x for x in stored if x.get('native_cover_verified'))
    assert chosen['box'] == actual
    assert chosen['frame_captured_at_epoch'] == 100.5
    assert chosen['native_alignment']['sample_offset_seconds'] == .5
    assert 'recording_alignment' not in tracking


@pytest.mark.parametrize('failure', ['negative', 'fragment', 'unrelated_geometry'])
def test_unverified_main_never_replaces_substream(tmp_path, failure):
    service, events, event, candidate, _, _, _ = setup(tmp_path)
    if failure == 'negative':
        service.verifier.detect.return_value = []
    elif failure == 'fragment':
        result = deepcopy(service.verifier.detect.return_value[0])
        result['box']['x2'] = result['box']['x1'] + 9
        result['box']['y2'] = result['box']['y1'] + 9
        service.verifier.detect.return_value = [result]
    else:
        service.project_main.return_value = []
    result = service.process(event['id'], [candidate])
    assert result['status'] == 'no_usable_candidate'
    assert result['reason'] == 'main_verification_failed'
    assert events.get(event['id'])['snapshot_path'] == event['snapshot_path']
    assert service.verifier.detect.call_count <= 5


def test_missing_recording_then_retry_uses_retained_exact_live_frame(tmp_path):
    service, events, event, candidate, main, _, _ = setup(tmp_path)
    service.read_frame.return_value = None
    service.read_frame.side_effect = None
    result = service.process(event['id'], [candidate], recorded_history=True)
    assert result['status'] == 'recording_pending'
    assert result['reason'] == 'main_recording_unavailable'
    assert events.get(event['id'])['snapshot_path'] == event['snapshot_path']
    service.read_frame.side_effect = lambda camera, epoch, source: main if source == 'main' else None
    assert service.process(event['id'], [candidate], recorded_history=True)['status'] == 'promoted'
    assert len(events.evidence_attempts(event['id'])) == 2


def test_valid_alignment_still_uses_corrected_timestamp_without_inference(tmp_path):
    service, events, event, candidate, main, _, tracking = setup(tmp_path, aligned=True)
    service.verifier.detect = Mock(side_effect=AssertionError('unnecessary inference'))
    service.admission.verify = Mock(side_effect=AssertionError('unnecessary fallback'))
    result = service.process(event['id'], [candidate], recorded_history=True)
    assert result['status'] == 'promoted'
    assert result['reason'] == 'same_fov_timestamp_aligned'
    service.read_frame.assert_called_once_with('test', 99.6, 'main')
    assert event_tracking(events.get(event['id']))[1] == tracking


def test_recorded_history_supplements_retained_candidates(tmp_path):
    service, _, event, candidate, _, _, _ = setup(tmp_path)
    additional = Candidate(102, candidate.image, candidate.objects, 6)
    service.recorded_candidates = Mock(return_value=([additional], False))
    result = service.process(event['id'], [candidate], recorded_history=True)
    assert result['cover_candidates'] == 2
    assert result['retained_live_candidates'] == 1
    service.recorded_candidates.assert_called_once()


def test_completion_during_preview_preserves_job_and_shortlist(tmp_path):
    service, _, event, candidate, _, _, _ = setup(tmp_path)
    service.offer(event['id'], candidate.epoch, candidate.image, candidate.objects, (640, 360))
    service._pending[event['id']]['due'] = 0
    calls = []
    def process(event_id, candidates, *, recorded_history=False):
        calls.append((candidates, recorded_history))
        assert candidates[0].image is candidate.image
        if len(calls) == 1:
            service.enqueue(event_id)
            service._pending[event_id]['due'] = 0
            return {'status': 'recording_pending'}
        service._closed = True
        return {'status': 'promoted'}
    service.process = process
    service._run()
    assert [history for _, history in calls] == [False, True]
    assert [c.epoch for c in calls[0][0]] == [c.epoch for c in calls[1][0]]


def test_successful_preview_remains_available_for_later_completion(tmp_path, monkeypatch):
    service, _, event, candidate, _, _, _ = setup(tmp_path)
    service.offer(event['id'], candidate.epoch, candidate.image, candidate.objects, (640, 360))
    service._pending[event['id']]['due'] = 0
    calls = []
    def process(event_id, candidates, *, recorded_history=False):
        calls.append(recorded_history)
        assert candidates[0].image is candidate.image
        if recorded_history:
            service._closed = True
        return {'status': 'promoted'}
    service.process = process
    def wait(timeout):
        assert service._pending[event['id']]['due'] == float('inf')
        service.enqueue(event['id'])
        service._pending[event['id']]['due'] = 0
    monkeypatch.setattr(service._condition, 'wait', wait)
    service._run()
    assert calls == [False, True]


def test_fallback_attempts_are_bounded_and_do_not_touch_live_incident_admission(tmp_path):
    service, _, event, candidate, _, _, _ = setup(tmp_path)
    service.verifier.detect.return_value = []
    service.admission.offer = Mock(side_effect=AssertionError('must not create admission work'))
    service.admission.poll = Mock(side_effect=AssertionError('must not consume admission results'))
    candidates = [Candidate(100+i, candidate.image, candidate.objects, 5) for i in range(20)]
    result = service.process(event['id'], candidates)
    assert result['status'] == 'no_usable_candidate'
    assert service.read_frame.call_count <= 15
    assert service.verifier.detect.call_count <= 15


def test_better_existing_cover_and_diagnostic_history_are_preserved(tmp_path):
    service, events, event, candidate, _, _, _ = setup(tmp_path)
    assert service.process(event['id'], [candidate])['status'] == 'promoted'
    before = events.get(event['id'])
    for _ in range(5):
        result = service.process(event['id'], [candidate])
        assert result['status'] == 'kept_better_cover'
        assert result['reason'] == 'better_cover_retained'
    assert events.get(event['id'])['snapshot_path'] == before['snapshot_path']
    assert events.get(event['id'])['evidence_revision'] == before['evidence_revision']
    assert len(events.evidence_attempts(event['id'])) == 3


def test_native_evidence_endpoint_reads_only_persisted_native_attempts(tmp_path):
    service, events, event, candidate, _, _, _ = setup(tmp_path)
    service.process(event['id'], [candidate])
    router = create_native_router(lambda: SimpleNamespace(events=events))
    endpoint = next(route.endpoint for route in router.routes if route.path.endswith('/native-evidence'))
    result = endpoint(event['id'])
    assert result['camera_id'] == 'test'
    assert result['attempts'][-1]['reason'] == 'main_frame_verified'
    with pytest.raises(HTTPException) as exc:
        endpoint(999999)
    assert exc.value.status_code == 404

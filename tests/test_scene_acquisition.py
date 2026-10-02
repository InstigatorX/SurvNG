from datetime import datetime, timezone
import json
import time
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
import pytest

from survng.app.motion_pipeline.decision_handler import MotionDecisionHandler
from survng.app.motion_pipeline.object_detection import (
    RecordedDetectionResult, RecordedMotionObjectDetector, TimestampedLiveFrame, _RecordedDetectionSample,
    resolve_recorded_refinement_plan,
)
from survng.app.motion_pipeline.scene_evidence import scene_batches, scene_observation
from survng.app.motion_incidents import MotionIncidentService
from survng.app.config import CameraConfig, DetectionZone


def person(x=10, **values):
    return {"label": "person", "confidence": .75,
            "box": {"x1": x, "y1": 10, "x2": x + 20, "y2": 70},
            "detection_frame_width": 100, "detection_frame_height": 100,
            "temporal_candidate_threshold": .25, "temporal_candidate_eligible": True,
            "incident_eligible": False, **values}


def test_recorded_ledger_retains_objects_absent_from_cover_and_outside_alert_zone():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    samples = [
        _RecordedDetectionSample(-1, frame, [person()], "before.mp4", exact_timestamp=True),
        _RecordedDetectionSample(0, frame, [], "cover.mp4", exact_timestamp=True),
        _RecordedDetectionSample(1, frame, [person(70)], "after.mp4", exact_timestamp=True),
    ]
    detector = RecordedMotionObjectDetector.__new__(RecordedMotionObjectDetector)
    result = detector._recorded_result(
        samples[1], [person(70, snapshot_visible=False,
                            temporal_sample_offset_seconds=1, recording_path="after.mp4")],
        samples, {}, time.monotonic(),
        refinement_pending=False, event_epoch=1000,
    )
    observations = scene_batches(result.objects)
    assert len(observations) == 2
    assert {item["captured_at_epoch"] for item in observations} == {999, 1001}
    assert {item["recording_path"] for item in observations} == {"before.mp4", "after.mp4"}
    assert all(item["snapshot_visible"] is False for item in observations)
    assert all(item["incident_eligible"] is False for item in observations)
    assert all(item["scene_track_key"] for item in observations)
    assert all("snapshot_path" not in item for item in observations)
    assert result.objects[0]["frame_captured_at_epoch"] == 1001
    assert result.objects[0]["recording_path"] == "after.mp4"


def test_observation_floor_is_distinct_from_alert_admission_and_has_stable_identity():
    low = person(confidence=.3, confidence_eligible=False)
    first = scene_observation(low, captured_at_epoch=1000, frame_source="recorded_main")
    second = scene_observation(low, captured_at_epoch=1000, frame_source="recorded_main")
    assert first == second
    assert first["incident_eligible"] is False
    assert scene_observation(person(confidence=.2), captured_at_epoch=1000,
                             frame_source="recorded_main") is None


def handler(events, objects, frame=None):
    result = RecordedDetectionResult(
        frame=np.zeros((100, 100, 3), dtype=np.uint8) if frame is None else frame,
        objects=objects, recording_path="source.mp4", timings_ms={},
        frame_captured_at_epoch=1000, frame_source="recorded_main", frame_timestamp_exact=True,
    )
    return MotionDecisionHandler(
        camera_id="gate", events=events, detection_provider=lambda _: result,
        snapshot_writer=lambda *_: "cover.webp", object_serializer=json.dumps,
    )


def test_paused_recorded_pass_decides_and_persists_nothing():
    events = Mock()
    paused = RecordedDetectionResult(
        frame=None, objects=[{"status": "refinement_resumes_later"}], recording_path="",
        timings_ms={}, resume_stage=2, resume_at=1234.5,
    )
    processor = MotionDecisionHandler(
        camera_id="gate", events=events, detection_provider=lambda _: paused,
        snapshot_writer=lambda *_: "cover.webp", object_serializer=json.dumps,
    )
    outcome = processor.refine("motion", "", datetime.fromtimestamp(1000, timezone.utc),
                               {}, existing_event_id=None)
    assert outcome.object_detected is None
    assert (outcome.refinement_resume_stage, outcome.refinement_resume_at) == (2, 1234.5)
    assert events.mock_calls == []


def test_cover_only_pass_persists_scene_evidence_before_presentation_rejection():
    events = Mock()
    processor = handler(events, [person()])
    processor.refine("motion", "", datetime.fromtimestamp(1000, timezone.utc),
                     {"cover_only": True}, existing_event_id=42)
    observations = events.record_scene_observations.call_args.args[1]
    assert observations[0]["confidence"] == .75
    assert observations[0]["incident_eligible"] is False
    assert observations[0]["recording_path"] == "source.mp4"


def test_discovery_persists_observations_without_creating_an_incident(tmp_path):
    from survng.app.events import EventStore
    events = EventStore(tmp_path)
    processor = handler(events, [person()])
    outcome = processor.refine("scene/discovery", "", datetime.fromtimestamp(1000, timezone.utc),
                               {"scene_discovery": True}, existing_event_id=None)
    assert outcome.event_id is None
    assert outcome.rejection_reason == "scene_activity_pending"
    assert events.list_scene_incidents() == []
    assert len(events.scene_acquired_observations("gate", 999, 1001)) == 1


def test_discovery_is_one_bounded_full_scene_sample():
    stages, retry, settle, *_ = resolve_recorded_refinement_plan(
        qualification={"scene_discovery": True},
    )
    assert stages == ((0.0,),)
    assert retry == 2
    assert settle == 0


def discovery_service():
    import queue
    import threading
    service = MotionIncidentService.__new__(MotionIncidentService)
    service.camera_id = "gate"
    service._status_lock = threading.RLock()
    service._refinement_queue = queue.Queue(maxsize=1)
    service._refinement_accepting = True
    service._refinement_stop = threading.Event()
    service._security_work_pending = threading.Event()
    service._pending_scene_discovery = None
    service._scene_discovery_counts = {"offered": 0, "superseded": 0, "stale": 0, "completed": 0, "failed": 0}
    service._last_scene_discovery_failure_log = float("-inf")
    service._queue_refinement = Mock(side_effect=AssertionError("discovery must not use the durable ledger"))
    service._record_timing = Mock()
    service._handoff = Mock()
    service.decision_processor = Mock()
    service.decision_processor.refine.return_value = SimpleNamespace(event_id=None)
    return service


def test_discovery_offer_keeps_only_newest_sample_without_durable_job():
    service = discovery_service()
    older = datetime.fromtimestamp(time.time() - 2, timezone.utc)
    newer = datetime.fromtimestamp(time.time() - 1, timezone.utc)
    assert service.offer_scene_discovery(older) == "queued"
    assert service.offer_scene_discovery(newer) == "superseded"
    service.decision_processor.refine.assert_not_called()
    assert service._run_scene_discovery() is True
    assert service._run_scene_discovery() is False
    service.decision_processor.refine.assert_called_once()
    args = service.decision_processor.refine.call_args
    assert args.args[0] == "scene/discovery"
    assert args.args[2] == newer
    assert args.args[3]["scene_discovery"] is True
    assert args.kwargs["require_eligible_object"] is False
    assert service._scene_discovery_counts == {"offered": 2, "superseded": 1, "stale": 0, "completed": 1, "failed": 0}


def test_discovery_yields_to_security_work_and_drops_stale_samples():
    service = discovery_service()
    service.offer_scene_discovery(datetime.fromtimestamp(time.time(), timezone.utc))
    service._security_work_pending.set()
    assert service._run_scene_discovery() is False
    assert service._pending_scene_discovery is not None
    service._security_work_pending.clear()
    service.offer_scene_discovery(datetime.fromtimestamp(time.time() - 120, timezone.utc))
    assert service._run_scene_discovery() is True
    service.decision_processor.refine.assert_not_called()
    assert service._scene_discovery_counts["stale"] == 1


def test_discovery_does_not_relax_configured_alert_confirmation_count():
    detector = SimpleNamespace(
        config=SimpleNamespace(confidence_threshold=.45, event_confirmation_frames=2,
                               require_incident_zone=False),
        detect=lambda *_args, **_kwargs: [person(confidence=.9)],
    )
    recorder = SimpleNamespace(recording_at=lambda *_args: {"path":"scene.mp4","start_epoch":1000})
    backend = RecordedMotionObjectDetector(
        CameraConfig(id="gate",name="Gate",stream_url="rtsp://example.invalid/main"),
        detector,recorder,lambda:None,
    )
    backend._read_recorded_frame = Mock(return_value=np.zeros((100,100,3),dtype=np.uint8))
    result = backend.detect(datetime.fromtimestamp(1000,timezone.utc), {"scene_discovery":True})
    detected = next(item for item in result.objects if item.get("label"))
    assert detected["temporal_required_observations"] == 2
    assert detected["incident_eligible"] is False
    assert len(scene_batches(result.objects)) == 1


def discovery_backend(objects):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sample = TimestampedLiveFrame(frame, 1000.1, 50.0, 4, 2, 3)
    detector = SimpleNamespace(
        config=SimpleNamespace(confidence_threshold=.45, event_confirmation_frames=2,
                               event_class_confirmation_frames={'dog': 3}, require_incident_zone=False),
        detect=Mock(side_effect=AssertionError('discovery must use shared refinement inference')),
        detect_refinement=Mock(return_value=objects),
    )
    backend = RecordedMotionObjectDetector(
        CameraConfig(id='gate', name='Gate', stream_url='rtsp://example.invalid/main'),
        detector, Mock(), lambda: None, timestamped_live_frame_provider=Mock(return_value=sample),
    )
    backend._detect = Mock(side_effect=AssertionError('fresh discovery must not wait for recordings'))
    return backend, sample


def test_discovery_uses_fresh_live_scene_before_recordings_finalize():
    backend, sample = discovery_backend([person(confidence=.9), person(70, label='dog', confidence=.3)])
    with patch('survng.app.motion_pipeline.object_detection.time.time', return_value=1000.2):
        result = backend.detect(datetime.fromtimestamp(1000, timezone.utc), {'scene_discovery': True})
    backend.detector.detect_refinement.assert_called_once()
    assert backend.detector.detect_refinement.call_args.kwargs['confidence_threshold'] == .25
    backend.recorder.recording_at.assert_not_called()
    assert result.frame is sample.frame
    assert result.frame_captured_at_epoch == 1000.1
    assert result.frame_source == 'live_discovery'
    assert not result.frame_timestamp_exact and not result.recording_path
    observed = scene_batches(result.objects)
    assert {item['label'] for item in observed} == {'person', 'dog'}
    assert all(item['captured_at_epoch'] == 1000.1 for item in observed)
    assert all(item['frame_source'] == 'live_discovery' for item in observed)
    detected = {item['label']: item for item in result.objects if item.get('label')}
    assert detected['person']['temporal_required_observations'] == 2
    assert detected['dog']['temporal_required_observations'] == 3
    assert all(item['incident_eligible'] is False for item in detected.values())


def test_empty_live_discovery_is_successful_analysis():
    backend, sample = discovery_backend([])
    with patch('survng.app.motion_pipeline.object_detection.time.time', return_value=1000.2):
        result = backend.detect(datetime.fromtimestamp(1000, timezone.utc), {'scene_discovery': True})
    assert result.frame is sample.frame
    assert scene_batches(result.objects) == []
    assert not any(item.get('status') == 'no_recorded_frame' for item in result.objects)


@pytest.mark.parametrize('changes', [
    {'captured_at_epoch': 990}, {'captured_at_epoch': 1005}, {'sequence': 0},
    {'capture_generation': 0}, {'camera_generation': 0}, {'source': 'main'},
    {'captured_at_monotonic': float('nan')}, {'captured_at_epoch': float('nan')},
])
def test_discovery_rejects_stale_or_invalid_live_evidence(changes):
    from dataclasses import replace
    backend, sample = discovery_backend([])
    backend.timestamped_live_frame_provider.return_value = replace(sample, **changes)
    backend._detect.side_effect = None
    with patch('survng.app.motion_pipeline.object_detection.time.time', return_value=1000.2):
        backend.detect(datetime.fromtimestamp(1000, timezone.utc), {'scene_discovery': True})
    backend._detect.assert_called_once()
    backend.detector.detect_refinement.assert_not_called()


def test_delayed_discovery_cannot_substitute_a_fresh_frame_for_old_event():
    backend, _ = discovery_backend([])
    backend._detect.side_effect = None
    with patch('survng.app.motion_pipeline.object_detection.time.time', return_value=1000.2):
        backend.detect(datetime.fromtimestamp(980, timezone.utc), {'scene_discovery': True})
    backend._detect.assert_called_once()
    backend.detector.detect_refinement.assert_not_called()


def test_live_discovery_keeps_untrusted_zone_geometry_out_of_alerts():
    from dataclasses import replace
    backend, sample = discovery_backend([person(confidence=.9)])
    backend.detector.config.event_confirmation_frames = 1
    backend.camera.zones = [DetectionZone(name='yard', points=[
        {'x': 0, 'y': 0}, {'x': 1, 'y': 0}, {'x': 1, 'y': 1}, {'x': 0, 'y': 1},
    ])]
    backend.timestamped_live_frame_provider.return_value = replace(sample, geometry_trusted=False)
    with patch('survng.app.motion_pipeline.object_detection.time.time', return_value=1000.2):
        result = backend.detect(datetime.fromtimestamp(1000, timezone.utc), {'scene_discovery': True})
    assert len(scene_batches(result.objects)) == 1
    detected = next(item for item in result.objects if item.get('label'))
    assert detected['incident_eligible'] is False
    assert detected['fast_geometry_untrusted'] is True


def test_live_discovery_inference_failure_is_retained_as_incomplete(tmp_path):
    backend, _ = discovery_backend([{'status': 'inference_error', 'error': 'capacity unavailable'}])
    with patch('survng.app.motion_pipeline.object_detection.time.time', return_value=1000.2):
        result = backend.detect(datetime.fromtimestamp(1000, timezone.utc), {'scene_discovery': True})
    from survng.app.events import EventStore
    events = EventStore(tmp_path)
    processor = MotionDecisionHandler(
        camera_id='gate', events=events, detection_provider=lambda _: result,
        snapshot_writer=lambda *_: 'cover.webp', object_serializer=json.dumps,
    )
    outcome = processor.refine('scene/discovery', '', datetime.fromtimestamp(1000, timezone.utc),
                               {'scene_discovery': True}, existing_event_id=None)
    assert outcome.rejection_reason == 'scene_activity_incomplete'
    assert events.list_scene_incidents() == []
    with events._connect() as conn:
        assert conn.execute('select status from acquired_samples').fetchone()[0] == 'failed'


def test_queued_discovery_uses_its_retained_capture_not_a_later_scene():
    backend, sample = discovery_backend([person()])
    backend.remember_scene_frame(sample)
    sample.frame[:] = 255  # A producer reusing its array cannot rewrite evidence.
    backend.timestamped_live_frame_provider.return_value = None
    with patch('survng.app.motion_pipeline.object_detection.time.time', return_value=1012):
        result = backend.detect(datetime.fromtimestamp(1000.1, timezone.utc), {'scene_discovery': True})
    assert result.frame_captured_at_epoch == 1000.1
    assert not result.frame.any()
    assert result.frame_source == 'live_discovery'
    backend._detect.assert_not_called()


def test_evicted_discovery_frame_uses_recorded_evidence():
    from dataclasses import replace
    backend, sample = discovery_backend([])
    for i in range(5):
        backend.remember_scene_frame(replace(sample, captured_at_epoch=1000.1+i*10))
    backend.timestamped_live_frame_provider.return_value = None
    backend._detect.side_effect = None
    with patch('survng.app.motion_pipeline.object_detection.time.time', return_value=1041):
        backend.detect(datetime.fromtimestamp(1000.1, timezone.utc), {'scene_discovery': True})
    backend._detect.assert_called_once()
    backend.detector.detect_refinement.assert_not_called()

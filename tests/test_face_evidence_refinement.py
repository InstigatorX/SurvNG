from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from survng.app.config import CameraConfig, DetectorConfig
from survng.app.evidence_work import EvidenceWorkPreempted
from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector, _RecordedDetectionSample


@pytest.fixture
def evidence():
    config = DetectorConfig(face_recognition_enabled=True, face_evidence_max_extra_frames=2)
    backend = RecordedMotionObjectDetector(
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid"),
        SimpleNamespace(config=config), Mock(), lambda: None,
    )
    frame = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    person = {"label": "person", "incident_eligible": True,
              "box": {"x1": 10, "y1": 10, "x2": 150, "y2": 100}}
    face = {"label": "face", "confidence": .9,
            "box": {"x1": 30, "y1": 20, "x2": 80, "y2": 70}}
    samples = [_RecordedDetectionSample(0., frame, [person], "recording.mp4")]
    backend._enrich_selected_faces = Mock(return_value=[person, face])
    backend._detect_objects = Mock(return_value=[person, face])
    sampler = Mock()
    sampler.recording_at.return_value = {"path": "recording.mp4", "start_epoch": 90.}

    def frames_at(_path, offsets, *, deadline):
        assert deadline <= 14.
        return {offset: SimpleNamespace(frame=frame, actual_offset=offset, exact_timestamp=True)
                for offset in offsets}, 1, 0

    sampler.frames_at.side_effect = frames_at
    return backend, samples, sampler, defaultdict(float)


def refine(evidence, deadline=20.):
    backend, samples, sampler, timing = evidence
    with patch("survng.app.motion_pipeline.object_detection.time.monotonic", return_value=10.):
        backend._refine_face_evidence(samples, sampler, 100., deadline, timing)


def test_refinement_obeys_extra_frame_cap_and_keeps_actual_timestamps(evidence):
    backend, samples, sampler, timing = evidence
    refine(evidence)
    assert sampler.frames_at.call_count == 2
    assert len(samples) == 3
    assert [round(item.offset, 2) for item in samples] == [0., -.4, .4]
    assert all(item.exact_timestamp for item in samples[1:])
    assert backend._detect_objects.call_count == 2
    assert timing["recording_samples_requested"] == 2
    assert timing["face_evidence_samples"] == 3


@pytest.mark.parametrize("disabled", ["face_recognition_enabled", "face_evidence_enabled"])
def test_disabled_refinement_does_no_work(evidence, disabled):
    backend, samples, sampler, _ = evidence
    setattr(backend.detector.config, disabled, False)
    refine(evidence)
    backend._enrich_selected_faces.assert_not_called()
    sampler.frames_at.assert_not_called()
    assert len(samples) == 1


def test_expired_budget_starts_no_decode_or_inference(evidence):
    backend, _, sampler, timing = evidence
    refine(evidence, deadline=9.)
    backend._enrich_selected_faces.assert_not_called()
    sampler.frames_at.assert_not_called()
    assert timing["face_evidence_deadline_reached"] == 1


def test_inexact_frames_cannot_be_counted_as_independent_evidence(evidence):
    backend, samples, sampler, timing = evidence
    sampler.frames_at.side_effect = lambda _path, offsets, **_kw: (
        {offset: SimpleNamespace(exact_timestamp=False) for offset in offsets}, 0, 1,
    )
    refine(evidence)
    assert len(samples) == 1
    backend._detect_objects.assert_not_called()
    assert timing["face_evidence_inexact_or_missing"] == 2


def test_decode_failure_preserves_existing_samples_and_reports_failure(evidence):
    _, samples, sampler, timing = evidence
    sampler.frames_at.side_effect = RuntimeError("decode failed")
    refine(evidence)
    assert len(samples) == 1
    assert timing["face_evidence_failed"] == 1


def test_cancellation_is_not_swallowed_as_optional_failure(evidence):
    with patch("survng.app.motion_pipeline.object_detection.check_evidence_cancellation",
               side_effect=EvidenceWorkPreempted):
        with pytest.raises(EvidenceWorkPreempted):
            refine(evidence)


def test_evidence_does_not_change_returned_event_objects(evidence):
    backend, samples, sampler, timing = evidence
    selected = samples[0]
    event_objects = [dict(selected.objects[0], temporal_consensus=True)]
    # Let the base enrichment retain only the qualified object. The optional
    # path can add faces to samples without changing the event's object list.
    backend._enrich_selected_faces = Mock(side_effect=lambda _frame, objects, **_kw: objects)
    backend._enrich_depth = Mock(side_effect=lambda _frame, objects, *_args, **_kw: objects)
    with patch("survng.app.motion_pipeline.object_detection.time.monotonic", return_value=10.):
        result = backend._recorded_result(
            selected, event_objects, samples, timing, 9., refinement_pending=False,
            event_epoch=100., face_sampler=sampler, face_deadline=20.,
        )
    assert result.objects == event_objects
    assert all(item["label"] == "person" for item in result.objects)
    assert result.face_candidates
    assert all(item.frame is None for item in samples[1:])

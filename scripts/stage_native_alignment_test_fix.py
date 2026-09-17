from pathlib import Path

path = Path("tests/test_native_replay_alignment.py")
text = path.read_text()
old = '''def test_terminal_evidence_reuses_detector_results_for_alignment(tmp_path):
    from unittest.mock import Mock
    from survng.app.config import CameraConfig
    from survng.app.native_evidence import Candidate
    service, events, event, image, obj, _ = fixture(tmp_path)
    tracking, observations = samples()
    events.update_object_tracking(event['id'], tracking)
    service.config.cameras = [CameraConfig(id='test',name='Test',stream_url='rtsp://example.test/main',native_same_field_of_view=True)]
    service.read_frame = Mock(return_value=image)
    service.match_main = Mock(return_value=[])
    service.verifier.detect = Mock(side_effect=[frame['objects'] for frame in observations])
    candidates = [Candidate(frame['epoch'],image,[],1) for frame in observations]
    service.process(event['id'],candidates)
    _, saved = event_tracking(events.get(event['id']))
    assert saved['recording_alignment']['offset_seconds'] == 1.2
    assert service.verifier.detect.call_count == len(candidates)
    assert service.counts['replay_aligned'] == 1
'''
new = '''def test_terminal_evidence_calibrates_from_track_history_not_cover_candidates(tmp_path):
    from unittest.mock import Mock
    from survng.app.config import CameraConfig
    from survng.app.native_evidence import Candidate, calibration_epochs

    service, events, event, image, obj, _ = fixture(tmp_path)
    tracking, _ = samples()
    events.update_object_tracking(event['id'], tracking)
    service.config.cameras = [CameraConfig(
        id='test', name='Test', stream_url='rtsp://example.test/main', native_same_field_of_view=True
    )]
    service.read_frame = Mock(return_value=image)
    service.match_main = Mock(return_value=[])

    epochs = calibration_epochs(tracking)
    offset = 1.2
    detector_results = []
    for epoch in epochs:
        t = epoch - 100 + offset
        x = 100 + 25 * t
        detector_results.append([{
            'label': 'person',
            'confidence': .9,
            'box': {'x1': x, 'y1': 60, 'x2': x + 40, 'y2': 180},
        }])
    service.verifier.detect = Mock(side_effect=detector_results)

    # Cover selection is deliberately smaller than the timing-calibration set.
    candidates = [Candidate(epoch, image, [], 1) for epoch in (103, 107, 111)]
    service.process(event['id'], candidates)

    _, saved = event_tracking(events.get(event['id']))
    assert saved['recording_alignment']['offset_seconds'] == 1.2
    assert len(epochs) > len(candidates)
    assert service.verifier.detect.call_count == len(epochs)
    assert service.counts['replay_aligned'] == 1
'''
if old not in text:
    raise SystemExit("terminal alignment test anchor not found")
path.write_text(text.replace(old, new, 1))

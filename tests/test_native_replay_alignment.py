import json
from copy import deepcopy

import pytest

from survng.app.native_replay_alignment import estimate_replay_alignment
from survng.app.native_evidence import event_tracking
from tests.test_native_evidence import fixture


def samples(offset=1.2, stationary=False):
    def box(t):
        x = 100 if stationary else 100+25*t
        return [x, 60, x+40, 180]
    history = [[100+t, *box(t)] for t in [i/5 for i in range(81)]]
    tracking = {"implementation":"gvatrack", "state":"complete", "native_session":"session",
                "frame_width":640,"frame_height":360,
                "tracks":[{"label":"person","box_history":history}]}
    observations = [{"epoch":100+t,"objects":[{"label":"person","confidence":.9,
                     "box":dict(zip(("x1","y1","x2","y2"),box(t+offset)))}]} for t in range(3,12)]
    return tracking, observations


@pytest.mark.parametrize("offset", [-1.4, 0, 1.2])
def test_estimates_recording_clock_offset_in_both_directions(offset):
    tracking, observations = samples(offset)
    aligned = estimate_replay_alignment(tracking, observations)
    assert aligned["offset_seconds"] == pytest.approx(offset)
    assert aligned["source"] == "main"
    assert aligned["mean_iou"] > .99


def test_stationary_sparse_and_out_of_range_evidence_is_not_calibrated():
    tracking, observations = samples(stationary=True)
    assert estimate_replay_alignment(tracking, observations) is None
    tracking, observations = samples()
    assert estimate_replay_alignment(tracking, observations[:4]) is None
    tracking, observations = samples(4)
    assert estimate_replay_alignment(tracking, observations) is None


def test_conflicting_frame_offsets_are_not_calibrated():
    tracking, observations = samples()
    _, reversed_observations = samples(-1.2)
    observations[::2] = reversed_observations[::2]
    assert estimate_replay_alignment(tracking, observations) is None


def test_alignment_commit_preserves_tracks_covers_and_guards_session(tmp_path):
    service, events, event, image, obj, old_tracking = fixture(tmp_path)
    tracking, observations = samples()
    events.update_object_tracking(event['id'], tracking)
    before = events.get(event['id'])
    alignment = estimate_replay_alignment(tracking, observations)
    assert events.update_native_replay_alignment(event['id'], 'stale-session', alignment) is None
    assert events.update_native_replay_alignment(event['id'], 'session', alignment)
    after = events.get(event['id'])
    _, saved = event_tracking(after)
    assert saved['recording_alignment'] == alignment
    del saved['recording_alignment']
    assert saved == tracking
    assert before['snapshot_path'] == after['snapshot_path']
    assert json.loads(before['objects_json'])[0] == json.loads(after['objects_json'])[0]
    assert after['evidence_revision'] == before['evidence_revision']
    with events._connect() as conn:
        assert conn.execute("select 1 from event_evidence_outbox where event_id=? and kind='incident_metadata_updated'", (event['id'],)).fetchone()
    assert events.update_native_replay_alignment(event['id'], 'session', alignment) is None
    tracking['state'] = 'active'
    events.update_object_tracking(event['id'], tracking)
    assert events.update_native_replay_alignment(event['id'], 'session', alignment) is None


def test_terminal_evidence_reuses_detector_results_for_alignment(tmp_path):
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


def test_fragmented_stationary_ids_are_not_timing_evidence():
    tracking = {"tracks": []}
    observations = []
    for i in range(6):
        box = [i*100, 0, i*100+50, 100]
        tracking["tracks"].append({"label":"person", "box_history":[[i+1.05,*box],[i+1.15,*box]]})
        observations.append({"epoch":i,"objects":[{"label":"person","confidence":.9,
                             "box":dict(zip(("x1","y1","x2","y2"),box))}]})
    assert estimate_replay_alignment(tracking, observations) is None


def test_similar_paths_with_competing_time_offsets_are_ambiguous():
    tracking, observations = samples(1.2)
    second = deepcopy(tracking["tracks"][0])
    for sample in second["box_history"]:
        sample[0] += 1.4
    tracking["tracks"].append(second)
    assert estimate_replay_alignment(tracking, observations) is None

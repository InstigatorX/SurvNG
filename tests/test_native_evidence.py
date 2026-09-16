import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from survng.app.config import AppConfig
from survng.app.event_store import EventStore
from survng.app.image_storage import DurableImageWriter
from survng.app.media_storage import MediaStorageRegistry
from survng.app.native_activity import compact_history
from survng.app.native_evidence import Candidate, NativeEvidenceService, image_quality, shortlist
from survng.app.stream_alignment import estimate_stream_alignment


def fixture(tmp_path):
    config = AppConfig(storage_dir=str(tmp_path))
    media = MediaStorageRegistry(tmp_path, config.media_storage)
    events = EventStore(tmp_path, media_storage=media)
    service = NativeEvidenceService(config, events, Mock(), DurableImageWriter(config.image_storage), media)
    image = np.random.default_rng(7).integers(0, 256, (360, 640, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    obj = {"label":"person", "confidence":0.9,"track_id":1,"box":{"x1":100,"y1":80,"x2":240,"y2":300},"incident_eligible":True}
    old = tmp_path/"snapshots"/"old.jpg"
    old.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(old),image)
    tracking={"implementation":"gvatrack","frame_width":640,"frame_height":360,"tracks":[{"track_id":1,"box_history":[[100,100,80,240,300]]}]}
    event=events.add_event(camera_id="test",kind="motion",topic="native/object-presence",message="",created_at=datetime.fromtimestamp(100,timezone.utc).isoformat(),snapshot_path=str(old),objects_json=json.dumps([obj,{"status":"object_tracking","object_tracking":tracking}]))
    return service,events,event,image,obj,tracking


def test_registration_and_quality_reject_uniform_frames():
    assert image_quality(np.full((360,640,3),128,np.uint8)) is None
    image=np.random.default_rng(7).integers(0,256,(360,640,3),dtype=np.uint8)
    estimate=estimate_stream_alignment(image,cv2.resize(image,(1280,720)))
    assert estimate is not None
    assert abs(estimate[0]-1)<0.02
    assert abs(estimate[2])<0.02


def test_cover_promotion_preserves_tracks_and_archives_exact_boxes(tmp_path):
    service,events,event,image,obj,tracking=fixture(tmp_path)
    service.read_frame=Mock(return_value=cv2.resize(image,(1280,720)))
    service.verifier.detect=Mock(return_value=[dict(obj,box={key:value*2 for key,value in obj["box"].items()})])
    result=service.process(event["id"],[Candidate(100,image,[obj],5)])
    assert result["status"]=="promoted"
    updated=events.get(event["id"])
    objects=json.loads(updated["objects_json"])
    assert next(x["object_tracking"] for x in objects if x.get("status")=="object_tracking")==tracking
    chosen=next(x for x in objects if x.get("label"))
    assert chosen["detection_frame_width"]==1280
    assert abs(chosen["box"]["x1"]-200)<5
    assert chosen["frame_captured_at_epoch"]==100
    assert updated["evidence_revision"]>event["evidence_revision"]
    assert len(events.source_observations(event["id"]))>=2
    previous=updated["snapshot_path"]
    service.read_frame=Mock(return_value=np.full((720,1280,3),128,np.uint8))
    assert service.process(event["id"],[Candidate(100,image,[obj],5)])["status"]=="no_verified_candidate"
    assert events.get(event["id"])["snapshot_path"]==previous


def test_long_history_keeps_start_end_and_remains_bounded():
    history=[]
    for index in range(10000):
        history=compact_history([*history,[index,1,2,3,4]])
    assert len(history)<=1024
    assert history[0][0]==0
    assert history[-1][0]==9999


def test_shortlist_is_bounded_and_spaced():
    selected=shortlist([Candidate(t,None,[],score) for t,score in [(1,5),(1.2,6),(3,4),(5,3),(7,2)]])
    assert [x.epoch for x in selected]==[1.2,3,5]


def test_failed_registration_does_not_project_boxes(tmp_path):
    service,events,event,image,obj,tracking=fixture(tmp_path)
    unrelated=np.random.default_rng(19).integers(0,256,(720,1280,3),dtype=np.uint8)
    assert service.match_main(Candidate(100,image,[obj],5),unrelated)==[]


def test_registration_alone_cannot_promote_an_empty_background(tmp_path):
    service, events, event, image, obj, _ = fixture(tmp_path)
    service.read_frame = Mock(return_value=cv2.resize(image, (1280, 720)))
    service.verifier.detect = Mock(return_value=[])
    result = service.process(event["id"], [Candidate(100, image, [obj], 5)])
    assert result["status"] == "no_verified_candidate"
    assert events.get(event["id"])["snapshot_path"] == event["snapshot_path"]


def test_cover_cannot_replace_verified_subject_with_small_same_class_fragment(tmp_path):
    service, events, event, image, obj, _ = fixture(tmp_path)
    main = cv2.resize(image, (1280, 720))
    service.read_frame = Mock(return_value=main)
    expected = dict(obj, box={k: v*2 for k, v in obj['box'].items()})
    service.match_main = Mock(return_value=[expected])
    fragment = dict(expected, box={**expected['box'], 'x2': 240, 'y2': 200})
    service.verifier.detect = Mock(return_value=[fragment])
    assert service.process(event['id'], [Candidate(100, image, [obj], 5)])['status'] == 'no_verified_candidate'
    assert events.get(event['id'])['snapshot_path'] == event['snapshot_path']


def test_empty_candidates_skip_image_quality_work(monkeypatch):
    from survng.app.native_evidence import candidate_score
    quality = Mock(side_effect=AssertionError('unnecessary pixel work'))
    monkeypatch.setattr('survng.app.native_evidence.image_quality', quality)
    assert candidate_score(None, []) is None
    assert candidate_score(None, [{'incident_eligible': False}]) is None


def test_recorded_frame_uses_lossless_uncompressed_ipc(tmp_path, monkeypatch):
    service, _, _, image, _, _ = fixture(tmp_path)
    service.recorder.recording_at.return_value = {'path': str(tmp_path/'video.mp4'), 'start_epoch': 99}
    encoded = cv2.imencode('.bmp', image)[1].tobytes()
    from types import SimpleNamespace
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=encoded))
    monkeypatch.setattr('survng.app.native_evidence.subprocess.run', run)
    actual = service.read_frame('test', 100, 'main')
    assert np.array_equal(actual, image)
    command = run.call_args.args[0]
    assert command[command.index('-c:v')+1] == 'bmp'
    assert command[command.index('-pix_fmt')+1] == 'bgr24'


def test_archived_images_survive_cleanup_after_promotion(tmp_path):
    service, events, event, image, obj, _ = fixture(tmp_path)
    service.read_frame = Mock(return_value=cv2.resize(image, (1280, 720)))
    service.verifier.detect = Mock(return_value=[dict(obj, box={k:v*2 for k,v in obj["box"].items()})])
    result = service.process(event["id"], [Candidate(100, image, [obj], 5)])
    assert result["status"] == "promoted"
    old = tmp_path / "snapshots" / "old.jpg"
    events._delete_snapshot_if_unreferenced(str(old), preserve_archive=True)
    assert old.exists()
    assert len(list((tmp_path / "snapshots").rglob("*.webp"))) >= 1


def test_completion_rescans_history_after_inflight_preview(tmp_path):
    service, _, event, image, obj, _ = fixture(tmp_path)
    event_id = event["id"]
    service._active.add(event_id)
    service.enqueue(event_id)
    assert service._pending[event_id]["recorded_history"]
    # A later preview must not downgrade the terminal full-history request.
    service.offer(event_id, 101, image, [obj], (640, 360))
    assert service._pending[event_id]["recorded_history"]
    service._active.clear()
    service._pending[event_id]["due"] = 0
    def process(eid, candidates):
        assert eid == event_id
        assert candidates is None
        service._closed = True
        return {"status": "promoted"}
    service.process = process
    service._run()
    assert service.counts["promoted"] == 1


def test_completion_upgrades_existing_preview_shortlist(tmp_path):
    service, _, event, image, obj, _ = fixture(tmp_path)
    service.offer(event["id"], 101, image, [obj], (640, 360))
    assert service._pending[event["id"]]["candidates"]
    service.enqueue(event["id"])
    assert service._pending[event["id"]]["recorded_history"]


def test_disabled_validation_promotes_aligned_main_without_detector_and_keeps_better_cover(tmp_path):
    service, events, event, image, obj, tracking = fixture(tmp_path)
    service.config.detector.native.verification_enabled = False
    service.read_frame = Mock(return_value=cv2.resize(image, (1280, 720)))
    service.verifier.detect = Mock(side_effect=AssertionError('image promotion does not require detection'))
    result = service.process(event['id'], [Candidate(100, image, [obj], 5)])
    assert result['status'] == 'promoted'
    service.verifier.detect.assert_not_called()
    updated = events.get(event['id'])
    stored = json.loads(updated['objects_json'])
    cover = next(o for o in stored if o.get('label'))
    assert cover['native_cover_verified'] is False
    assert cover['box_provenance'] == 'projected_from_substream'
    assert cover['verification']['source'] == 'substream'
    assert cover['detection_frame_width'] == 1280
    assert abs(cover['box']['x1']-200) < 5
    assert next(o['object_tracking'] for o in stored if o.get('status') == 'object_tracking') == tracking
    assert service.process(event['id'], [Candidate(100, image, [obj], 5)])['status'] == 'kept_better_cover'
    assert events.get(event['id'])['snapshot_path'] == updated['snapshot_path']


def test_disabled_validation_retains_substream_for_unusable_or_unaligned_main(tmp_path):
    service, events, event, image, obj, _ = fixture(tmp_path)
    service.config.detector.native.verification_enabled = False
    for main in (None, np.full((720,1280,3),128,np.uint8), np.random.default_rng(19).integers(0,256,(720,1280,3),dtype=np.uint8)):
        service.read_frame = Mock(return_value=main)
        result = service.process(event['id'], [Candidate(100, image, [obj], 5)])
        assert result['status'] in {'recording_pending', 'no_usable_candidate'}
        assert events.get(event['id'])['snapshot_path'] == event['snapshot_path']


def test_optional_calibration_failure_does_not_block_disabled_mode_promotion(tmp_path):
    from survng.app.config import CameraConfig
    service, events, event, image, obj, tracking = fixture(tmp_path)
    service.config.detector.native.verification_enabled = False
    service.config.cameras = [CameraConfig(id='test', name='Test', stream_url='rtsp://unused.invalid', native_same_field_of_view=True)]
    tracking.update(state='complete', native_session='test-session')
    events.update_object_tracking(event['id'], tracking)
    service.read_frame = Mock(return_value=cv2.resize(image,(1280,720)))
    service.verifier.detect = Mock(side_effect=RuntimeError('unavailable'))
    assert service.process(event['id'], [Candidate(100,image,[obj],5), Candidate(101,image,[obj],5)])['status'] == 'promoted'
    service.verifier.detect.assert_called_once()
    assert service.counts['calibration_unavailable'] == 1


def test_verification_projection_survives_changed_object_appearance(tmp_path):
    service, _, _, image, obj, _ = fixture(tmp_path)
    main = cv2.resize(image, (1280, 720))
    main[160:600, 200:480] = 100  # Room geometry matches; subject appearance does not.
    candidate = Candidate(100, image, [obj], 5)
    assert service.match_main(candidate, main) == []
    projected = service.project_main(candidate, main)
    assert len(projected) == 1
    for key, value in obj['box'].items():
        assert abs(projected[0]['box'][key] - value * 2) < 5
    assert 'native_cover_verified' not in projected[0]
    unrelated = np.random.default_rng(19).integers(0, 256, main.shape, dtype=np.uint8)
    assert service.project_main(candidate, unrelated) == []

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch
from collections import defaultdict
import time

import numpy as np

from survng.app.scene_activity import evaluate_scene_activity
from survng.app.motion_pipeline.object_detection import (
    RecordedDetectionResult, RecordedMotionObjectDetector, _RecordedDetectionSample, _DecodedRecordedFrame,
    _scene_sample_records,
)
from survng.app.motion_pipeline.scene_evidence import scene_observation
from survng.app.config import CameraConfig


def detection(x=20, **extra):
    return {"label":"person","confidence":.279,"incident_eligible":False,
            "temporal_candidate_threshold":.25,
            "box":{"x1":x,"y1":20,"x2":x+20,"y2":70},
            "detection_frame_width":160,"detection_frame_height":100,**extra}


def frame(x=None, background=20):
    result=np.full((100,160,3),background,dtype=np.uint8)
    if x is not None:
        result[20:70,x:x+20]=220
    return result


def records(frames, detections, *, offsets=None):
    offsets=offsets or list(range(len(frames)))
    samples=[_RecordedDetectionSample(t,image,objects,"main.mp4",requested_offset=t,exact_timestamp=True)
             for t,image,objects in zip(offsets,frames,detections)]
    observations=[scene_observation(item,captured_at_epoch=1000+s.offset,frame_source="recorded_main",recording_path="main.mp4")
                  for s in samples for item in s.objects if item.get("label")]
    return _scene_sample_records(samples,[o for o in observations if o],1000,"gate",tuple(offsets))


def test_furniture_label_flicker_and_box_jitter_do_not_establish_activity():
    image=frame(20)
    samples=records([image]*4,[[],[detection()],[detection(22,confidence=.339)],[detection(20,confidence=.95)]])
    decision=evaluate_scene_activity(samples)
    assert decision["status"]=="unsupported"
    assert decision["supporting_observation_ids"]==[]


def test_real_low_confidence_crossing_is_activity_independent_of_alert_policy():
    samples=records([frame(20),frame(35),frame(50)],[[detection(20)],[detection(35)],[detection(50)]])
    decision=evaluate_scene_activity(samples)
    assert decision["status"]=="supported"
    assert decision["reason"]=="video_verified_activity"
    assert decision["activity_epoch"]==1002
    assert decision["supporting_observation_ids"]
    assert all(o["incident_eligible"] is False for s in samples for o in s["observations"])


def test_brief_arrival_has_physical_support_even_when_only_one_frame_has_person():
    samples=records([frame(),frame(20),frame()],[[],[detection()],[]])
    assert evaluate_scene_activity(samples)["status"]=="supported"


def test_unavailable_confirmation_stays_incomplete_not_empty_or_rejected():
    samples=_scene_sample_records([],[],1000,"gate",(-1,0,1))
    decision=evaluate_scene_activity(samples)
    assert decision["status"]=="incomplete"
    assert len(samples)==3
    assert all(s["status"]=="failed" for s in samples)


def test_retries_of_one_capture_cannot_supply_temporal_support():
    samples=records([frame(20),frame(35)],[[detection(20)],[detection(35)]],offsets=[0,0])
    assert evaluate_scene_activity(samples)["status"]=="incomplete"


def test_analysis_variants_are_durable_but_not_independent_frames():
    original=records([frame(20)],[[detection(20)]])
    refined=records([frame(20)],[[detection(21),detection(70)]])
    reordered=records([frame(20)],[[detection(70),detection(21)]])
    assert original[0]['id']!=refined[0]['id']
    assert refined[0]['id']==reordered[0]['id']
    assessment=evaluate_scene_activity(original+refined)
    assert assessment['diagnostics']['complete_sample_count']==1
    assert assessment['status']=='incomplete'


def test_failed_analysis_is_distinct_from_successful_empty_analysis_of_same_frame():
    empty=records([frame()], [[]])
    failed=records([frame()], [[{'status':'inference_error'}]])
    assert empty[0]['id']!=failed[0]['id']
    assert empty[0]['status']=='complete'
    assert failed[0]['status']=='failed'
    assert evaluate_scene_activity(empty+failed)['reason']=='source_evidence_unavailable'


def test_global_exposure_change_does_not_validate_detector_arrival():
    samples=records([frame(background=20),frame(background=80)],[[],[detection()]])
    assert evaluate_scene_activity(samples)["status"]=="unsupported"


def test_physical_movement_inside_stationary_box_is_not_lost():
    before,after=frame(20),frame(20)
    after[35:55,20:40]=20
    assert evaluate_scene_activity(records([before,after],[[detection()],[detection()]]))["status"]=="supported"


def test_widespread_camera_shift_does_not_validate_local_object_movement():
    rng=np.random.default_rng(42)
    before=rng.integers(0,256,(100,160,3),dtype=np.uint8)
    after=np.roll(before,8,axis=1)
    assert evaluate_scene_activity(records([before,after],[[detection(20)],[detection(28)]]))["status"]=="unsupported"


def test_empty_successful_samples_finish_without_inventing_activity():
    assert evaluate_scene_activity(records([frame(),frame()],[[],[]]))["status"]=="unsupported"


def test_camera_reported_notice_is_not_a_video_witness():
    sample={"id":"a","camera_id":"gate","captured_epoch":1000,"status":"complete","observations":[],
            "metadata":{"validated_motion":True,"motion_source":"camera_reported"}}
    assert evaluate_scene_activity([sample])["status"]=="pending"
    sample["metadata"]["motion_source"]="measured_pixels"
    assert evaluate_scene_activity([sample])["reason"]=="measured_motion"


def test_confirmation_bypasses_live_and_honors_bounded_requested_window():
    detector=RecordedMotionObjectDetector.__new__(RecordedMotionObjectDetector)
    detector.camera=SimpleNamespace(id="gate")
    detector.detector=SimpleNamespace(config=SimpleNamespace())
    detector._live_discovery_result=Mock(side_effect=AssertionError("live confirmation forbidden"))
    detector._detect=Mock(return_value=RecordedDetectionResult(None,[],"",{}))
    qualification={"scene_discovery":True,"scene_confirmation":True,
                   "scene_confirmation_window":{"start_epoch":997,"end_epoch":1003}}
    result=detector.detect(datetime.fromtimestamp(1000,timezone.utc),qualification)
    kwargs=detector._detect.call_args.kwargs
    offsets=kwargs["stages"][0]
    assert min(offsets)==-3 and max(offsets)==3
    assert len(offsets)<=8
    assert kwargs["scene_confirmation"] is True
    assert kwargs["allow_representative_refinement"] is False
    batch=next(item for item in result.objects if item.get("status")=="scene_observations")
    assert len(batch["samples"])==len(offsets)
    assert evaluate_scene_activity(batch["samples"])["status"]=="incomplete"


def test_native_recorded_review_image_keeps_its_own_time_and_source():
    detector=RecordedMotionObjectDetector.__new__(RecordedMotionObjectDetector)
    detector.camera=SimpleNamespace(id="gate")
    images=[frame(20),frame(35)]
    samples=[_RecordedDetectionSample(t,image,[detection(x)],"main.mp4",requested_offset=t,exact_timestamp=True)
             for t,image,x in zip((-1,1),images,(20,35))]
    result=detector._recorded_result(samples[1],samples[1].objects,samples,{},time.monotonic(),
                                     refinement_pending=False,event_epoch=1000,confirmation_offsets=(-1,1))
    assert result.frame is images[1]
    assert result.review_image=={"role":"additional_review","source":"recorded_main",
                                 "captured_at_epoch":1001,"width":160,"height":100,"analyzed_frame":True,
                                 "frame_timestamp_exact":True,"recording_path":"main.mp4"}
    batch=next(item for item in result.objects if item.get("status")=="scene_observations")
    assert [s["captured_epoch"] for s in batch["samples"]]==[999,1001]
    assert evaluate_scene_activity(batch["samples"])["status"]=="supported"


def test_recorded_confirmation_acquires_whole_plan_despite_semantic_consensus():
    detector=RecordedMotionObjectDetector(
        CameraConfig(id="gate",name="Gate",stream_url="rtsp://example.invalid/main"),
        SimpleNamespace(config=SimpleNamespace(confidence_threshold=.65,event_confirmation_frames=2,
                                               event_class_confirmation_frames={},recorded_adaptive_sampling=True)),
        SimpleNamespace(recording_at=lambda *_:{"path":"main.mp4","start_epoch":990}),lambda:None,
    )
    detector._read_recorded_frames=Mock(side_effect=lambda path,offsets,**kwargs:(
        {t:_DecodedRecordedFrame(frame(20+round((t-9)*10)),t,True) for t in offsets},1))
    detector._detect_objects=Mock(side_effect=lambda image,**kwargs:[detection(int(np.flatnonzero(image[25,:,0]>200)[0]))])
    detector._enrich_selected_faces=lambda image,objects,**kwargs:objects
    detector._enrich_depth=lambda image,objects,*args,**kwargs:objects
    detector._refine_face_evidence=Mock()
    offsets=(-1,-.5,0,.5,1)
    with patch("survng.app.motion_pipeline.object_detection._refinement_early_exit_ready",side_effect=AssertionError("semantic early exit forbidden")):
        result=detector._detect_with_budget(datetime.fromtimestamp(1000,timezone.utc),stages=(offsets,),
            retry_seconds=1,allow_representative_refinement=False,refinement_pending=False,
            workflow_started=time.monotonic(),timing=defaultdict(float),event_epoch=1000,
            deadline=time.monotonic()+1,planned_offsets=offsets,prefetched_rows=[],scene_confirmation=True,settle_seconds=0)
    batch=next(item for item in result.objects if item.get("status")=="scene_observations")
    assert len(batch["samples"])==5
    assert all(s["status"]=="complete" for s in batch["samples"])
    assert detector._detect_objects.call_count==5
    assert result.review_image["source"]=="recorded_main"
    assert evaluate_scene_activity(batch["samples"])["status"]=="supported"


def test_background_level_pixel_change_is_not_localized_activity():
    # A stable box with scattered changes throughout the image: the old
    # independent cutoffs admitted 8.0247% local versus 6.9505% background.
    before = frame(20)
    after = before.copy()
    after[7::11, 7::11] = 255 - after[7::11, 7::11]
    after[30, 25:27] = 255 - after[30, 25:27]
    samples = records([before, after], [[detection()], [detection()]])
    assert samples[-1]["metadata"]["activity_witnesses"] == []
    assert evaluate_scene_activity(samples)["status"] == "unsupported"


def test_persisted_near_background_witnesses_do_not_establish_activity():
    from copy import deepcopy
    from survng.app.scene_zone_admission import evaluate_scene_establishment
    from tests.test_scene_zone_admission import policy, zone
    # Measurements retained by the two reported front-door confirmations.
    for local, background, displacement in (
        (0.082738, 0.072881, 0.000434),
        (0.085119, 0.079609, 0.000742),
    ):
        samples = records([frame(20), frame(35)], [[detection(20)], [detection(35)]])
        witness = samples[-1]["metadata"]["activity_witnesses"][0]
        witness.update(kind="localized_motion", witness_version=1,
                       local_change_fraction=local, background_change_fraction=background,
                       normalized_displacement=displacement)
        for sample in samples:
            for item in sample["observations"]:
                item.update(activity_role="indeterminate", confidence=0.93)
        original = deepcopy(samples)
        decision = evaluate_scene_establishment(samples, policy=policy(zone("Entry", "incident", 0, 1)))
        assert decision["status"] == "unsupported"
        assert decision["supporting_observation_ids"] == []
        assert decision["diagnostics"]["localized_change_policy_version"] == 2
        assert samples == original


def test_localization_measurements_must_be_valid_and_exceed_background():
    from survng.app.scene_activity import localized_change_supported
    for local, background in ((float("nan"), 0), (0.5, float("inf")), (1.1, 0),
                              (0.2, -0.1), (0.5, 0.081), (None, 0), (0.1, 0.07)):
        assert not localized_change_supported(local, background)
    assert localized_change_supported(0.08, 0)
    assert localized_change_supported(0.2, 0.07)
    assert localized_change_supported(0.15, 0.07)
    assert not localized_change_supported(0.149999, 0.07)

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from survng.app.config import CameraConfig, MotionQualificationConfig
from survng.app.motion_decisions import MotionDecisionOrchestrator, priority_motion_topic
from survng.app.motion_events import MotionEventCoordinator, MotionTrigger, MotionTriggerBatch
from survng.app.motion_pipeline.context import MotionContext
from survng.app.motion_pipeline.evidence import MotionEvidenceRepository
from survng.app.motion_pipeline.evidence_stages import OnvifEventEvidenceStage
from survng.app.motion_pipeline.runtime import MotionRuntimeState
from survng.app.motion_topics import normalize_motion_topic, semantic_motion_kind
from survng.app.onvif_events import OnvifEventListener


@pytest.mark.parametrize("suffix,kind", [
    ("PeopleDetect", "person"),
    ("VehicleDetect", "vehicle"),
    ("DogCatDetect", "animal"),
    ("FaceDetect", "face"),
    ("SmartMotionHuman", "person"),
    ("SmartMotionVehicle", "vehicle"),
])
def test_vendor_semantics_agree_across_parser_evidence_and_decision(suffix, kind):
    topic = f"tns1:RuleEngine/tns2:MyRuleDetector/{suffix}"
    assert semantic_motion_kind(topic) == kind
    assert priority_motion_topic(topic)
    listener = OnvifEventListener(
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid"),
        Mock(),
    )
    assert listener._motion_event_state(topic, '<tt:SimpleItem Name="State" Value="true"/>') is True
    assert listener._motion_event_state(topic, '<tt:SimpleItem Name="State" Value="false"/>') is False

    repository = MotionEvidenceRepository("gate")
    stage = OnvifEventEvidenceStage("onvif", repository)
    context = MotionContext(
        camera_id="gate", captured_at=10.0, original_frame=None,
        configuration={"observation_kind": "motion_event", "event_topic": topic},
        runtime=MotionRuntimeState("gate"),
    )
    evidence = stage.process(context).source_evidence["onvif"]
    assert evidence["semantic_kind"] == kind
    assert evidence["priority"] is True
    assert evidence["score"] == 0.95


def test_dogcat_camera_notice_bypasses_visual_nuisance_qualification():
    qualification = Mock()
    qualification.qualify_burst.side_effect = AssertionError("semantic notice must bypass EMA")
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate", events=MotionEventCoordinator(queue_size=4, retry_limit=2), audit_recorder=Mock(),
        config=MotionQualificationConfig(mode="camera"), qualification=qualification,
        incidents=Mock(), media=Mock(), analysis=Mock(), state=Mock(),
    )
    now = datetime.now(timezone.utc)
    topic = "tns1:RuleEngine/MyRuleDetector/DogCatDetect"
    result, _ = orchestrator._qualification_result(
        triggers=MotionTriggerBatch((MotionTrigger(topic, "", now, now.timestamp()),)),
        event_at=now, received_at=now.timestamp(), mode="camera", sensitivity="balanced",
        priority=priority_motion_topic(topic), adaptive_only=False,
    )
    assert result.accepted
    assert result.reason == "priority_topic"
    qualification.qualify_burst.assert_not_called()


def test_generic_motion_and_unknown_topics_do_not_gain_semantic_bypass():
    assert normalize_motion_topic(" tns1:RuleEngine/tns2:CellMotionDetector/Motion ") == "ruleengine/cellmotiondetector/motion"
    for topic in ("tns1:VideoSource/MotionAlarm", "tns1:Device/Status", ""):
        assert semantic_motion_kind(topic) is None
        assert not priority_motion_topic(topic)
    assert not priority_motion_topic(
        "RuleEngine/PersonDetect",
        '<SimpleItem Name="State" Value="false"/>',
    )
    assert semantic_motion_kind("manual") == "manual"


def test_generic_alarm_with_explicit_class_is_an_active_semantic_notice():
    listener = OnvifEventListener(
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid"),
        Mock(),
    )
    message = "{'Name': 'ObjectType', 'Value': 'dog'}"
    assert listener._motion_event_state("RuleEngine/MyRuleDetector/Alarm", message) is True
    assert priority_motion_topic("RuleEngine/MyRuleDetector/Alarm", message)


def test_custom_evidence_keywords_remain_supported():
    repository = MotionEvidenceRepository("gate")
    stage = OnvifEventEvidenceStage("onvif", repository, priority_keywords=("tripwire",))
    context = MotionContext(
        camera_id="gate", captured_at=10.0, original_frame=None,
        configuration={"observation_kind": "motion_event", "event_topic": "rule/Tripwire"},
        runtime=MotionRuntimeState("gate"),
    )
    evidence = stage.process(context).source_evidence["onvif"]
    assert evidence["priority"] is True
    assert evidence["semantic_kind"] is None


def test_manual_evidence_remains_priority() -> None:
    repository = MotionEvidenceRepository("gate")
    stage = OnvifEventEvidenceStage("onvif", repository)
    context = MotionContext(
        camera_id="gate", captured_at=10.0, original_frame=None,
        configuration={"observation_kind": "motion_event", "event_topic": "manual"},
        runtime=MotionRuntimeState("gate"),
    )
    evidence = stage.process(context).source_evidence["onvif"]
    assert evidence["priority"] is True
    assert evidence["score"] == 0.95


def _onvif_evidence(topic: str, message: str = "", **options: object) -> dict:
    stage = OnvifEventEvidenceStage("onvif", MotionEvidenceRepository("gate"), **options)  # type: ignore[arg-type]
    context = MotionContext(
        camera_id="gate", captured_at=10.0, original_frame=None,
        configuration={"observation_kind": "motion_event", "event_topic": topic, "event_message": message},
        runtime=MotionRuntimeState("gate"),
    )
    return stage.process(context).source_evidence["onvif"]


@pytest.mark.parametrize(("topic", "message", "reason"), [
    ("RuleEngine/MyRuleDetector/PeopleDetect", '<tt:SimpleItem Name="State" Value="unknown"/>', "state_not_affirmative"),
    ("RuleEngine/MyRuleDetector/PeopleDetect", '<tt:SimpleItem Name="State" Value="false"/>', "state_not_affirmative"),
    ("RuleEngine/MyRuleDetector/PeopleDetect", "<tt:Message><broken", "unparsed_payload"),
    # The default configuration no longer matches taxonomy words as substrings.
    ("tns1:RuleEngine/CellMotionDetector/Motion", "{'Name': 'Rule', 'Value': 'person zone'}", "no_semantic_claim"),
])
def test_evidence_does_not_call_unconfirmed_notices_priority(topic: str, message: str, reason: str) -> None:
    evidence = _onvif_evidence(topic, message)
    assert evidence["priority"] is False
    assert evidence["priority_reason"] is None
    assert evidence["semantic_kind"] is None
    assert evidence["semantic_admission_reason"] == reason
    assert evidence["score"] == 0.55


def test_evidence_priority_reason_matches_the_decision_predicate() -> None:
    semantic = _onvif_evidence(
        "RuleEngine/MyRuleDetector/PeopleDetect", '<tt:SimpleItem Name="State" Value="true"/>'
    )
    assert (semantic["priority"], semantic["priority_reason"], semantic["semantic_kind"]) == (
        True, "semantic", "person"
    )
    assert _onvif_evidence("manual")["priority_reason"] == "manual"
    custom = _onvif_evidence("rule/Tripwire", priority_keywords=("tripwire",))
    assert (custom["priority"], custom["priority_reason"]) == (True, "keyword")
    # A configured keyword never overrides a cleared semantic topic.
    cleared = _onvif_evidence(
        "RuleEngine/PersonDetect", '<tt:SimpleItem Name="State" Value="false"/>',
        priority_keywords=("person",),
    )
    assert cleared["priority"] is False

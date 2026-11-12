"""Speech as an incident trigger: provenance and pipeline isolation."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from survng.app.ema_v2 import (
    CameraNotice,
    EmaQualified,
    EpisodeDecisionReason,
    MotionEpisodeController,
    MotionSource,
)
from survng.app.incident_presenter import _event_row
from survng.app.motion import MotionQualificationResult
from survng.app.motion_decisions import MotionDecisionOrchestrator
from survng.app.motion_ingress import MotionEventIngressService
from survng.app.motion_topics import SPEECH_TOPIC, is_speech_topic, semantic_motion_kind
from survng.app.scene_zone_admission import evaluate_scene_establishment

SPOKEN_AT = datetime(2026, 11, 12, 15, 0, 0, tzinfo=timezone.utc)


def _service(mode: str = "camera") -> tuple[MotionEventIngressService, Mock]:
    owned = Mock()
    owned.events.enqueue.return_value = True
    owned.events.episode_controller = MotionEpisodeController("gate")
    owned.events.episode_controller.start_generation(1)
    owned.state.begin_ingress.return_value = 1
    owned.state.publish_event = owned.publish_event
    owned.state.set_last_motion_at = owned.set_last_motion_at
    owned.qualification.settings.return_value = (mode, "balanced", 320)
    owned.qualification.observe_event = owned.observe_event
    service = MotionEventIngressService(
        camera_id="gate",
        events=owned.events,
        qualification=owned.qualification,
        state=owned.state,
        epoch_now=lambda: SPOKEN_AT.timestamp() + 2.0,
    )
    return service, owned


def _ema() -> EmaQualified:
    return EmaQualified(
        camera_id="gate",
        captured_at=SPOKEN_AT.timestamp(),
        observed_monotonic=time.monotonic(),
        result=MotionQualificationResult(
            accepted=True, score=0.8, threshold=0.48, reason="credible_motion",
            frame_count=3, features={}, telemetry={},
        ),
        required_score=0.65,
        qualifying_samples=3,
        window_samples=4,
        candidate_started_at=SPOKEN_AT.timestamp() - 1.0,
    )


def test_speech_topic_is_ours_and_carries_no_camera_semantics() -> None:
    assert is_speech_topic(SPEECH_TOPIC)
    assert not is_speech_topic("onvif/motion")
    assert semantic_motion_kind(SPEECH_TOPIC) is None


@pytest.mark.parametrize("mode", ["camera", "camera_rescue", "adaptive", "enforce"])
def test_speech_queues_a_check_without_touching_motion_state(mode: str) -> None:
    service, owned = _service(mode)

    assert service.handle_speech(SPOKEN_AT) is True

    # Not camera motion: no Motion sensor event, no last-motion time, no
    # ONVIF evidence, and the adaptive mode does not drop it.
    owned.publish_event.assert_not_called()
    owned.set_last_motion_at.assert_not_called()
    owned.observe_event.assert_not_called()
    trigger = owned.events.enqueue.call_args.args[0]
    assert trigger.topic == SPEECH_TOPIC
    assert trigger.event_at == SPOKEN_AT  # SurvNG's clock, not the camera model
    assert trigger.event_timing.selection_reason == "speech_time"
    assert trigger.camera_semantics is None
    assert service.event_clock.status()["samples"] == 0
    intent = owned.events.episode_controller.intent(trigger.detection_intent_id)
    assert intent.primary_source is MotionSource.SPEECH
    assert intent.speech_notice is not None and intent.camera_notice is None
    owned.state.end_ingress.assert_called_once_with(1)


def test_speech_is_ignored_while_ingress_is_closed() -> None:
    service, owned = _service()
    owned.state.begin_ingress.return_value = None
    assert service.handle_speech(SPOKEN_AT) is False
    owned.events.enqueue.assert_not_called()


def test_speech_merges_into_a_running_check_and_keeps_the_camera_notice() -> None:
    service, owned = _service()
    service.handle("onvif/motion", "motion")
    camera_trigger = owned.events.enqueue.call_args.args[0]
    owned.events.enqueue.reset_mock()

    assert service.handle_speech(SPOKEN_AT) is False  # merged, not a new job

    owned.events.enqueue.assert_not_called()
    intent = owned.events.episode_controller.intent(camera_trigger.detection_intent_id)
    assert intent.camera_notice is not None and intent.camera_notice.topic == "onvif/motion"
    assert intent.speech_notice is not None
    assert {MotionSource.CAMERA, MotionSource.SPEECH} <= set(intent.sources)


def test_failed_ema_admission_falls_back_to_pending_speech() -> None:
    controller = MotionEpisodeController("gate")
    controller.start_generation(1)
    reserved = controller.observe_ema(_ema(), generation=1)
    assert reserved.reason is EpisodeDecisionReason.REQUEST_RESERVED
    merged = controller.observe_speech(
        CameraNotice(
            camera_id="gate", event_at=SPOKEN_AT.timestamp(),
            observed_monotonic=time.monotonic(), topic=SPEECH_TOPIC,
        ),
        generation=1,
    )
    assert merged.reason is EpisodeDecisionReason.MERGED_WITH_REQUEST

    fallback = controller.acknowledge_admission(
        reserved.intent.intent_id, admitted=False, occurred_monotonic=time.monotonic(),
    )

    assert fallback.reason is EpisodeDecisionReason.REQUEST_RESERVED
    assert fallback.intent.primary_source is MotionSource.SPEECH
    assert fallback.intent.event_at == SPOKEN_AT.timestamp()


def test_speech_is_accepted_without_visual_motion() -> None:
    result, _ = MotionDecisionOrchestrator._qualification_result(
        SimpleNamespace(),
        triggers=[],
        event_at=SPOKEN_AT,
        received_at=SPOKEN_AT.timestamp(),
        mode="enforce",
        sensitivity="balanced",
        priority=False,
        adaptive_only=False,
        speech=True,
    )
    assert result.accepted is True
    assert result.reason == "speech_trigger"
    assert result.features["primary_motion_source"] == "speech"


def test_unzoned_camera_speech_notice_reads_as_speech() -> None:
    result = evaluate_scene_establishment([], policy=None, notice={"source": "speech", "epoch": 10.0})
    assert result["status"] == "supported"
    assert result["reason"] == "admitted_speech_notice"
    assert result["summary"] == "The camera heard speech."
    assert result["evidence_kind"] == "speech_reported"
    camera = evaluate_scene_establishment([], policy=None, notice={"source": "camera", "epoch": 10.0})
    assert camera["reason"] == "admitted_camera_notice"
    assert camera["evidence_kind"] == "camera_reported"


@pytest.mark.parametrize(
    ("trigger_source", "topic", "expected"),
    [("speech", SPEECH_TOPIC, "speech"), ("camera", "onvif/motion", "camera"), ("visual_backup", "adaptive/visual_backup", "ema")],
)
def test_incident_events_report_their_trigger(trigger_source: str, topic: str, expected: str) -> None:
    event = _event_row({
        "id": 1,
        "camera_id": "gate",
        "topic": topic,
        "objects_json": json.dumps([{"status": "motion_qualification", "motion_qualification": {"trigger_source": trigger_source}}]),
        "created_at": SPOKEN_AT.isoformat(),
    })
    assert event["trigger_source"] == expected

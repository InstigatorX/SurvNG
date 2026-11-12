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


# Layer 1: speech attached to incidents ------------------------------------

from survng.app.event_store import EventStore
from survng.app.speech.attach import ATTACH_PAD_SECONDS, incident_speech
from survng.app.speech.store import SpeechStore


def _incident(store: EventStore, incident_id: str, state: str, episodes: list[tuple[str, float, float]]) -> None:
    with store._connect() as conn:
        conn.execute(
            "insert into scene_incidents (id, revision, start_epoch, end_epoch, state) values (?, 1, ?, ?, ?)",
            (incident_id, min(e[1] for e in episodes), max(e[2] for e in episodes), state),
        )
        for index, (camera_id, start, end) in enumerate(episodes):
            conn.execute(
                "insert into scene_episodes (id, incident_id, camera_id, start_epoch, end_epoch, last_activity_epoch, coverage_json) "
                "values (?, ?, ?, ?, ?, ?, '{}')",
                (f"{incident_id}-{index}", incident_id, camera_id, start, end, end),
            )


def _speech(store: SpeechStore, camera_id: str, start: float, end: float, text: str = "") -> int:
    segment_id = store.insert_segment(
        camera_id=camera_id, started_at=start, ended_at=end, peak_probability=0.9, status="queued",
    )
    if text:
        store.set_transcript(segment_id, status="done", text=text, language="en", score=None, model="base.en", device="CPU")
    return segment_id


def test_episode_windows_skip_merged_and_unconfirmed_incidents(tmp_path) -> None:
    events = EventStore(tmp_path)
    _incident(events, "live", "complete", [("gate", 100.0, 130.0), ("yard", 120.0, 150.0)])
    _incident(events, "merged", "merged", [("gate", 200.0, 210.0)])
    _incident(events, "pending", "unconfirmed", [("gate", 300.0, 310.0)])

    windows = events.scene_episode_windows(["live", "merged", "pending", "missing"])

    assert windows["live"] == [("gate", 100.0, 130.0), ("yard", 120.0, 150.0)]
    assert windows["merged"] == [] and windows["pending"] == [] and windows["missing"] == []
    assert events.scene_episode_windows_overlapping("gate", 125.0, 400.0) == [(100.0, 130.0)]
    assert events.scene_incident_ids_overlapping("gate", 0.0, 1000.0) == ["live"]


def test_speech_counts_only_the_incidents_own_camera_and_padding(tmp_path) -> None:
    speech = SpeechStore(tmp_path)
    inside = _speech(speech, "gate", 110.0, 114.0, text="hello")
    straddle = _speech(speech, "gate", 128.0, 140.0)  # starts inside, ends after the episode
    early = _speech(speech, "gate", 100.0 - ATTACH_PAD_SECONDS - 3.0, 100.0 - ATTACH_PAD_SECONDS + 1.0)  # within pad
    _speech(speech, "gate", 160.0, 165.0)  # outside the padded gate window
    _speech(speech, "yard", 110.0, 112.0)  # another camera, outside the yard episode
    _speech(speech, "porch", 110.0, 112.0)  # camera not in the incident

    summary = incident_speech(speech, {"live": [("gate", 100.0, 130.0), ("yard", 120.0, 150.0)], "quiet": [("gate", 500.0, 510.0)]})

    assert set(summary) == {"live"}
    live = summary["live"]
    assert live["segments"] == 3 and live["transcribed"] == 1
    assert live["seconds"] == round(4.0 + 12.0 + 4.0, 1)
    assert live["windows"] == [{
        "camera_id": "gate",
        "start_at": datetime.fromtimestamp(100.0 - ATTACH_PAD_SECONDS, timezone.utc).isoformat(),
        "end_at": datetime.fromtimestamp(130.0 + ATTACH_PAD_SECONDS, timezone.utc).isoformat(),
    }]
    # Read-scope incident payloads never carry transcript text.
    assert "hello" not in json.dumps(summary)
    assert {inside, straddle, early}


def test_manager_attaches_counts_and_survives_speech_failures(tmp_path) -> None:
    from survng.app.manager import AppManager

    events = EventStore(tmp_path)
    _incident(events, "live", "complete", [("gate", 100.0, 130.0)])
    speech = SpeechStore(tmp_path)
    _speech(speech, "gate", 110.0, 112.0, text="private words")
    manager = AppManager.__new__(AppManager)
    manager.events = events
    manager.speech = SimpleNamespace(incident_speech=lambda windows: incident_speech(speech, windows))

    items = manager.attach_incident_speech([{"id": "live"}, {"id": "other"}])

    assert items[0]["speech"]["segments"] == 1
    assert "speech" not in items[1]
    assert "private words" not in json.dumps(items)

    def broken(_windows):
        raise RuntimeError("speech db locked")

    manager.speech = SimpleNamespace(incident_speech=broken)
    assert manager.attach_incident_speech([{"id": "live"}]) == [{"id": "live"}]


def test_transcript_rows_link_to_their_incident(tmp_path) -> None:
    from survng.app.manager import AppManager

    events = EventStore(tmp_path)
    _incident(events, "live", "complete", [("gate", 100.0, 130.0)])
    manager = AppManager.__new__(AppManager)
    manager.events = events
    rows = [
        {"camera_id": "gate", "started_at": datetime.fromtimestamp(126.0, timezone.utc).isoformat(), "ended_at": datetime.fromtimestamp(133.0, timezone.utc).isoformat()},
        {"camera_id": "gate", "started_at": datetime.fromtimestamp(400.0, timezone.utc).isoformat(), "ended_at": datetime.fromtimestamp(401.0, timezone.utc).isoformat()},
    ]
    manager.speech_incident_ids(rows)
    assert [row["incident_id"] for row in rows] == ["live", None]

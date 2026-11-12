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


# Transcript retention (option B) -------------------------------------------

import sqlite3

from survng.app.speech.retention import apply_speech_retention


def test_retention_deletes_unattached_and_expires_attached_without_recordings(tmp_path) -> None:
    speech = SpeechStore(tmp_path)
    unattached = _speech(speech, "gate", 10.0, 12.0, text="nobody cares")
    recorded = _speech(speech, "gate", 101.0, 104.0, text="package at the door")
    unrecorded = _speech(speech, "gate", 201.0, 204.0, text="secret plans")
    young = _speech(speech, "gate", 5_000.0, 5_002.0, text="still fresh")
    episodes = {"gate": [(100.0, 110.0), (200.0, 210.0)]}

    result = apply_speech_retention(
        speech,
        cutoff=1_000.0,
        episode_windows=lambda camera, low, high: [w for w in episodes.get(camera, []) if w[1] >= low and w[0] <= high],
        recording_ranges=lambda camera, low, high: [(90.0, 150.0)],
        batch_size=2,
    )

    assert (result.deleted, result.expired, result.kept) == (1, 1, 1)
    assert speech.get(unattached) is None
    assert speech.get(recorded)["text"] == "package at the door"
    expired = speech.get(unrecorded)
    assert expired["text"] == "" and expired["content_expired"] is True
    assert speech.list_segments(query="secret") == []  # search entry removed too
    assert speech.get(young)["text"] == "still fresh"
    # The incident still shows that speech happened, with no transcript.
    summary = incident_speech(speech, {"live": [("gate", 200.0, 210.0)]})
    assert summary["live"]["segments"] == 1 and summary["live"]["transcribed"] == 0

    # Finalized rows are not rescanned; recorded ones are checked again later.
    again = apply_speech_retention(
        speech, cutoff=1_000.0,
        episode_windows=lambda camera, low, high: episodes["gate"],
        recording_ranges=lambda camera, low, high: [],
    )
    assert (again.deleted, again.expired, again.kept) == (0, 1, 0)
    assert speech.get(recorded)["content_expired"] is True


def test_retention_keeps_attached_transcripts_when_recordings_cannot_be_read(tmp_path) -> None:
    speech = SpeechStore(tmp_path)
    attached = _speech(speech, "gate", 101.0, 104.0, text="keep me")

    def broken(*_args):
        raise RuntimeError("recording index unavailable")

    result = apply_speech_retention(
        speech, cutoff=1_000.0,
        episode_windows=lambda camera, low, high: [(100.0, 110.0)],
        recording_ranges=broken,
    )
    assert result.kept == 1
    assert speech.get(attached)["text"] == "keep me"


def test_existing_speech_database_is_migrated(tmp_path) -> None:
    with sqlite3.connect(tmp_path / "speech.sqlite3") as conn:
        conn.execute(
            "create table speech_segments (id integer primary key autoincrement, camera_id text not null, "
            "started_at real not null, ended_at real not null, peak_probability real not null default 0, "
            "status text not null, text text not null default '', language text not null default '', score real, "
            "model text not null default '', device text not null default '', reason text not null default '', "
            "created_at real not null, updated_at real not null)"
        )
        conn.execute(
            "insert into speech_segments (camera_id, started_at, ended_at, status, text, created_at, updated_at) "
            "values ('gate', 1, 2, 'done', 'old row', 0, 0)"
        )
    store = SpeechStore(tmp_path)
    row = store.list_segments()[0]
    assert row["text"] == "old row" and row["content_expired"] is False


def test_retention_keeps_running_after_speech_is_turned_off(tmp_path) -> None:
    from survng.app.config import AppConfig
    from survng.app.speech.lifecycle import SpeechLifecycle

    config = AppConfig.model_validate({"cameras": []})
    off = SpeechLifecycle(config, database_dir=tmp_path, camera_enabled=lambda camera_id: False)
    off.start()
    assert off._thread is None  # no database: nothing to expire, nothing created
    assert not (tmp_path / "speech.sqlite3").exists()

    SpeechStore(tmp_path)
    off = SpeechLifecycle(config, database_dir=tmp_path, camera_enabled=lambda camera_id: False)
    off.start()
    try:
        assert off.status()["state"] == "disabled"
        assert off._thread is not None and off._thread.is_alive()
        assert off.store is not None
    finally:
        off.stop()


# Speech confirmation and trigger throttling -------------------------------

import threading as _threading

from survng.app.speech.listener import ListenerEvent
from survng.app.speech.segmenter import FRAME_SAMPLES, SpeechConfirmed, SpeechSegmenter
from survng.app.speech.triggers import TriggerDispatcher, TriggerThrottle


def _segmenter_events(probabilities, **kwargs):
    segmenter = SpeechSegmenter(**kwargs)
    events = []
    for value in probabilities:
        events.extend(segmenter.push(None, value))
    return events


def test_confirmation_needs_enough_voiced_audio_and_fires_once_per_utterance() -> None:
    settings = dict(min_speech_ms=100, hangover_ms=320, pre_roll_ms=128, max_segment_seconds=5.0, confirm_ms=1500)
    short = _segmenter_events([0.0] * 10 + [0.9] * 30 + [0.0] * 20, **settings)  # about 1 s
    assert not any(isinstance(event, SpeechConfirmed) for event in short)

    long = _segmenter_events([0.0] * 10 + [0.9] * 400 + [0.0] * 20, **settings)  # about 13 s, split at 5 s
    confirmed = [event for event in long if isinstance(event, SpeechConfirmed)]
    assert len(confirmed) == 1
    # Dated to where the utterance began (pre-roll included), not to confirmation.
    assert confirmed[0].start_sample == 6 * FRAME_SAMPLES

    two = _segmenter_events(([0.9] * 60 + [0.0] * 20) * 2, **settings)
    assert sum(isinstance(event, SpeechConfirmed) for event in two) == 2

    disabled = _segmenter_events([0.9] * 100, min_speech_ms=100)
    assert not any(isinstance(event, SpeechConfirmed) for event in disabled)


def test_throttle_applies_cooldown_and_rolling_budget() -> None:
    now = [0.0]
    throttle = TriggerThrottle(cooldown_seconds=30.0, max_per_window=3, clock=lambda: now[0])
    assert throttle.allow() is True
    now[0] = 10.0
    assert throttle.allow() is False  # cooldown
    allowed = []
    for moment in (40.0, 80.0, 120.0, 160.0):
        now[0] = moment
        allowed.append(throttle.allow())
    assert allowed == [True, True, False, False]  # budget of 3 per 5 minutes
    now[0] = 301.0
    assert throttle.allow() is True  # the first check left the window
    assert throttle.status() == {"incident_checks": 4, "incident_checks_suppressed": 3}


def test_trigger_cameras_request_checks_off_the_audio_thread(tmp_path) -> None:
    from survng.app.speech.lifecycle import SpeechLifecycle

    config = AppConfig_for_speech({"gate": "trigger", "yard": "attach"})
    calls = []
    done = _threading.Event()

    def request(camera_id, started_at):
        calls.append((camera_id, started_at, _threading.current_thread().name))
        done.set()
        return True

    lifecycle = SpeechLifecycle(
        config,
        database_dir=tmp_path,
        camera_enabled=lambda camera_id: True,
        vad_factory=lambda path: object(),
        listener_factory=lambda camera, **kwargs: SimpleNamespace(start=lambda: None, stop=lambda: None, status=lambda: {"state": "listening"}),
        request_incident_check=request,
    )
    lifecycle.start()
    try:
        lifecycle._listener_event(ListenerEvent("yard", SpeechConfirmed(0), 50.0, None))
        lifecycle._listener_event(ListenerEvent("gate", SpeechConfirmed(0), 100.0, None))
        lifecycle._listener_event(ListenerEvent("gate", SpeechConfirmed(0), 105.0, None))  # within cooldown
        assert done.wait(5.0)
        assert [(camera, at) for camera, at, _ in calls] == [("gate", 100.0)]
        assert calls[0][2] == "speech-incident-trigger"
        gate = lifecycle.status()["cameras"]["gate"]
        assert gate["incidents"] == "trigger"
        assert (gate["incident_checks"], gate["incident_checks_suppressed"]) == (1, 1)
        assert "incident_checks" not in lifecycle.status()["cameras"]["yard"]
    finally:
        lifecycle.stop()


def AppConfig_for_speech(modes: dict[str, str]):
    from survng.app.config import AppConfig

    return AppConfig.model_validate({
        "speech": {"enabled": True, "vad_model_path": "/models/vad.onnx"},
        "cameras": [
            {"id": camera_id, "name": camera_id, "stream_url": f"rtsp://camera.local/{camera_id}", "speech": {"enabled": True, "incidents": mode}}
            for camera_id, mode in modes.items()
        ],
    })


def test_manager_requests_checks_only_for_active_cameras() -> None:
    from survng.app.manager import AppManager

    seen = []
    worker = SimpleNamespace(handle_speech=lambda when: seen.append(when) or True)
    switches = SimpleNamespace(camera_enabled=lambda camera_id: True, detection_enabled=lambda camera_id: True)
    manager = AppManager.__new__(AppManager)
    manager.camera_fleet = switches
    manager.camera_controls = switches
    manager.workers = {"gate": worker}

    assert manager._speech_incident_check("gate", SPOKEN_AT.timestamp()) is True
    assert seen == [SPOKEN_AT]
    assert manager._speech_incident_check("unknown", SPOKEN_AT.timestamp()) is False
    switches.detection_enabled = lambda camera_id: False
    assert manager._speech_incident_check("gate", SPOKEN_AT.timestamp()) is False
    assert len(seen) == 1


def test_dispatcher_drops_when_full_instead_of_blocking() -> None:
    from survng.app.speech import triggers

    blocker = _threading.Event()
    dispatcher = TriggerDispatcher(lambda camera_id, at: blocker.wait(5.0))
    dispatcher.start()
    try:
        results = [dispatcher.submit("gate", float(index)) for index in range(triggers.QUEUE_LIMIT + 5)]
        assert results.count(False) >= 4 and dispatcher.dropped == results.count(False)
    finally:
        blocker.set()
        dispatcher.stop()


# Speech establishment on cameras with zones ------------------------------

from tests.test_scene_zone_admission import box, physical, restricting

SPEECH_NOTICE = {"source": "speech", "epoch": 1000.0}


def _finished(samples):
    for sample in samples:
        sample["metadata"]["confirmation_complete"] = True
    return samples


def _empty_frames(count=2):
    return [{
        "id": f"empty-{index}", "camera_id": "gate", "captured_epoch": 1000 + index, "status": "complete",
        "observations": [], "metadata": {"frame_width": 160, "frame_height": 100, "activity_witnesses": [], "confirmation_complete": True},
    } for index in range(count)]


def _speech_decision(samples):
    return evaluate_scene_establishment(samples, policy=restricting(), notice=dict(SPEECH_NOTICE))


def test_zoned_camera_speech_with_nobody_visible_establishes() -> None:
    decision = _speech_decision(_empty_frames())
    assert decision["status"] == "supported"
    assert decision["reason"] == "admitted_speech_notice"
    assert decision["evidence_kind"] == "speech_reported"
    assert decision["zone_interpretation"]["establishment_eligible"] is True
    # The same notice from the camera (not speech) stays zone-restricted.
    camera = evaluate_scene_establishment(_empty_frames(), policy=restricting(), notice={"source": "camera", "epoch": 1000.0})
    assert camera["status"] != "supported"


def test_speech_is_explained_away_only_by_people_in_ignore_zones() -> None:
    pedestrian = _speech_decision(_finished(physical(["person", "person"], [box(20, 40), box(40, 40)])))
    assert pedestrian["status"] == "unsupported"
    assert pedestrian["reason"] == "speech_attributed_to_ignored_subject"

    parked_car = _speech_decision(_finished(physical(["car", "car"], [box(20, 40), box(40, 40)])))
    assert parked_car["status"] == "supported"
    assert parked_car["reason"] == "admitted_speech_notice"


def test_person_in_an_eligible_zone_supports_speech() -> None:
    visitor = _speech_decision(physical(["person", "person"], [box(20, 80), box(40, 80)]))
    assert visitor["status"] == "supported"
    assert visitor["reason"] == "speech_with_subject"
    assert visitor["supporting_observation_ids"]

    both = physical(["person", "person"], [box(20, 40), box(40, 80)])
    assert _speech_decision(both)["reason"] == "speech_with_subject"


def test_stationary_resident_in_an_eligible_zone_still_counts_as_present() -> None:
    samples = physical(["person", "person"], [box(20, 80), box(20, 80)])
    for sample in samples:
        for observation in sample["observations"]:
            observation["activity_role"] = "scene_context"
    zoned = restricting()
    zoned["stationary_subject_presence"] = "ignore"
    zoned["object_activity_attribution"] = "enforce"
    interpretations = evaluate_scene_establishment(samples, policy=zoned)["zone_interpretation"]["observations"]
    assert {item["admission_reason"] for item in interpretations} == {"stationary_scene_context"}
    # A resident sitting still does not open an incident on video alone...
    assert evaluate_scene_establishment(samples, policy=zoned)["status"] != "supported"
    # ...but is present, so speech heard while they are there establishes.
    decision = evaluate_scene_establishment(samples, policy=zoned, notice=dict(SPEECH_NOTICE))
    assert decision["status"] == "supported"
    assert decision["reason"] == "speech_with_subject"


def test_unverifiable_video_marks_speech_unverified_and_undecided_video_waits() -> None:
    failed = _speech_decision([{"id": "failed", "camera_id": "gate", "captured_epoch": 1000, "status": "failed", "observations": [], "metadata": {}}])
    assert failed["status"] == "supported"
    assert failed["evidence_kind"] == "speech_unverified"
    assert _speech_decision([])["evidence_kind"] == "speech_unverified"

    sampling = physical(["car"], [box(20, 40)])  # one frame, detection still running
    assert _speech_decision(sampling)["status"] == "pending"


def test_speech_establishment_is_stable_on_replay() -> None:
    samples = _finished(physical(["person", "person"], [box(20, 40), box(40, 40)]))
    first = _speech_decision(samples)
    again = _speech_decision(json.loads(json.dumps(samples)))
    assert (first["status"], first["reason"]) == (again["status"], again["reason"])

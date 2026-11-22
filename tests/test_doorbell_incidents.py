"""A recognized doorbell press opens a doorbell incident and notifies."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from survng.app.events import EventStore
from survng.app.motion_pipeline.object_detection import RecordedDetectionResult
from survng.app.motion_topics import DOORBELL_TOPIC
from survng.app.scene_zone_admission import evaluate_scene_establishment
from tests.test_scene_establishment import one_look, picture, processor
from tests.test_scene_zone_admission import policy, restricting, zone

SOURCE_TOPIC = "tns1:RuleEngine/MyRuleDetector/Visitor"


def doorbell_notice() -> dict:
    return {"source": "doorbell", "epoch": 1000.0, "features": {},
            "camera_admission": {"version": 1, "trigger_class": "doorbell", "source_topics": [SOURCE_TOPIC]}}


def empty_look() -> list[dict]:
    return [{"id": "s", "camera_id": "front-door", "captured_epoch": 1000, "status": "complete",
             "observations": [], "metadata": {}}]


@pytest.mark.parametrize("zones", ["unrestricted", "ignore_everything", "required_zone_elsewhere"])
def test_press_establishes_without_a_detection_whatever_the_zones(zones: str) -> None:
    zone_policy = {
        "unrestricted": None,
        "ignore_everything": policy(zone("Porch", "ignore", 0, 1), require=False),
        "required_zone_elsewhere": policy(zone("Driveway", "incident", .9, 1)),
    }[zones]

    decision = evaluate_scene_establishment(empty_look(), policy=zone_policy, notice=doorbell_notice())

    assert decision["status"] == "supported"
    assert decision["reason"] == "admitted_doorbell_press"
    assert decision["evidence_kind"] == "doorbell_press"
    assert decision["activity_epoch"] == 1000.0
    assert decision["summary"] == "Someone rang the doorbell."


def test_eligible_visitor_is_recorded_as_supporting_evidence_only() -> None:
    visitor = evaluate_scene_establishment(one_look(y2=90, eligible=True), policy=restricting(), notice=doorbell_notice())
    assert visitor["supporting_observation_ids"] == ["look-0"]
    # A person only in an ignore zone neither supports nor vetoes the press.
    ignored = evaluate_scene_establishment(one_look(y2=40, eligible=True), policy=restricting(), notice=doorbell_notice())
    assert ignored["status"] == "supported"
    assert ignored["supporting_observation_ids"] == []
    car = evaluate_scene_establishment(one_look("car", y2=90, eligible=True), policy=restricting(), notice=doorbell_notice())
    assert car["supporting_observation_ids"] == []


def test_press_event_opens_a_doorbell_incident_that_survives_restart(tmp_path) -> None:
    store = EventStore(tmp_path)
    nobody = RecordedDetectionResult(frame=picture(), objects=[], recording_path="", timings_ms={},
                                     frame_captured_at_epoch=1000.0, frame_source="recorded_main")
    qualification = {"trigger_source": "doorbell", "features": {},
                     "camera_notice_admission": doorbell_notice()["camera_admission"]}

    outcome = processor(store, nobody, tmp_path).handle(
        DOORBELL_TOPIC, "<press/>", datetime.fromtimestamp(1000, timezone.utc), dict(qualification))

    assert outcome.event_id is not None
    incident = EventStore(tmp_path).scene_incident(event_id=outcome.event_id)
    assert incident is not None
    assert incident["establishment"]["status"] == "established"
    assert incident["trigger_source"] == "doorbell"


def _alert_manager(exclude_motion: bool, notify: str):
    from survng.app.config import AppConfig
    from survng.app.manager import AppManager

    manager = AppManager.__new__(AppManager)
    manager.config = AppConfig.model_validate({
        "integration_notifications": {"exclude_motion": exclude_motion},
        "cameras": [{
            "id": "front-door", "name": "Front door", "stream_url": "rtsp://camera.local/door",
            "doorbell": {"enabled": True, "notify": notify},
        }],
    })
    return manager


@pytest.mark.parametrize("exclude_motion", [False, True])
@pytest.mark.parametrize("notify", ["always", "motion", "never"])
def test_doorbell_notification_matrix(exclude_motion: bool, notify: str) -> None:
    manager = _alert_manager(exclude_motion, notify)
    payload = {"camera_id": "front-door", "trigger_source": "doorbell",
               "events": [{"id": 1, "camera_id": "front-door", "trigger_source": "doorbell"}]}
    nobody = {"event_id": 1, "objects": []}
    ignored = {"event_id": 1, "objects": [{"label": "person", "eligible": False, "zones": ["Sidewalk"]}]}
    visitor = {"event_id": 1, "objects": [{"label": "person", "eligible": True, "zones": []}]}

    always = notify == "always"
    assert manager._scene_alert_enabled(nobody, payload) is {
        "always": True, "motion": not exclude_motion, "never": False}[notify]
    assert manager._scene_alert_enabled(ignored, payload) is always
    # An eligible object notifies under every setting, as for any incident.
    assert manager._scene_alert_enabled(visitor, payload) is True


def test_camera_notification_switch_still_wins_over_doorbell_always() -> None:
    manager = _alert_manager(False, "always")
    manager.config.cameras[0].incident_notifications_enabled = False
    payload = {"camera_id": "front-door", "trigger_source": "doorbell",
               "events": [{"id": 1, "camera_id": "front-door", "trigger_source": "doorbell"}]}

    assert manager._scene_alert_enabled({"event_id": 1, "objects": []}, payload) is False


def test_presenter_and_label_name_the_doorbell() -> None:
    from survng.app.incident_presenter import _event_row

    event = _event_row({
        "id": 1, "camera_id": "front-door", "topic": DOORBELL_TOPIC, "message": "",
        "created_at": "2026-11-22T16:45:43+00:00", "snapshot_path": "", "recording_path": "",
        "objects_json": '[{"status": "motion_qualification", "motion_qualification": {"trigger_source": "doorbell"}}]',
    })
    assert event["trigger_source"] == "doorbell"

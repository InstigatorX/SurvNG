from datetime import datetime, timezone

from survng.app.onvif_inspector import OnvifInspector


def test_classifies_reolink_topics():
    inspector = OnvifInspector()
    assert inspector.classify_topic(
        "ruleengine/myruledetector/vehicledetect"
    ) == "vehicle"
    assert inspector.classify_topic(
        "ruleengine/myruledetector/dogcatdetect"
    ) == "animal"
    assert inspector.classify_topic(
        "ruleengine/myruledetector/peopledetect"
    ) == "person"
    assert inspector.classify_topic(
        "videosource/motionalarm"
    ) == "motion"


def test_records_state_transitions():
    inspector = OnvifInspector()
    now = datetime.now(timezone.utc)

    first = inspector.record(
        camera_id="gate",
        topic="tns1:RuleEngine/MyRuleDetector/VehicleDetect",
        normalized_topic="ruleengine/myruledetector/vehicledetect",
        active=False,
        simple_items=(("state", "false"),),
        received_at=now,
    )
    repeated = inspector.record(
        camera_id="gate",
        topic="tns1:RuleEngine/MyRuleDetector/VehicleDetect",
        normalized_topic="ruleengine/myruledetector/vehicledetect",
        active=False,
        simple_items=(("state", "false"),),
        received_at=now,
    )
    activated = inspector.record(
        camera_id="gate",
        topic="tns1:RuleEngine/MyRuleDetector/VehicleDetect",
        normalized_topic="ruleengine/myruledetector/vehicledetect",
        active=True,
        simple_items=(("state", "true"),),
        received_at=now,
    )

    assert first.changed is True
    assert repeated.changed is False
    assert activated.changed is True

    state = inspector.state_snapshot()
    assert state["cameras"]["gate"]["classes"]["vehicle"]["active"] is True


def test_filters_changes_only():
    inspector = OnvifInspector()
    now = datetime.now(timezone.utc)

    for value in (False, False, True):
        inspector.record(
            camera_id="gate",
            topic="tns1:VideoSource/MotionAlarm",
            normalized_topic="videosource/motionalarm",
            active=value,
            received_at=now,
        )

    payload = inspector.events_after(0, changes_only=True)
    assert [event["active"] for event in payload["events"]] == [False, True]


def test_clear_keeps_monotonic_cursor():
    inspector = OnvifInspector()
    inspector.record(
        camera_id="gate",
        topic="tns1:VideoSource/MotionAlarm",
        normalized_topic="videosource/motionalarm",
        active=True,
    )
    cleared = inspector.clear()
    assert cleared["cleared"] == 1

    new_event = inspector.record(
        camera_id="gate",
        topic="tns1:VideoSource/MotionAlarm",
        normalized_topic="videosource/motionalarm",
        active=False,
    )
    assert new_event.seq > cleared["next"]

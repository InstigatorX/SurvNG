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


PACKAGE_XML = (
    '<ns0:Message xmlns:ns0="http://docs.oasis-open.org/wsn/b-2" xmlns:ns1="http://www.onvif.org/ver10/schema">'
    '<ns1:Message UtcTime="2026-11-22T05:38:34Z" PropertyOperation="Initialized">'
    '<ns1:Source><ns1:SimpleItem Name="Source" Value="000" /></ns1:Source>'
    '<ns1:Data><ns1:SimpleItem Name="State" Value="false" /></ns1:Data></ns1:Message></ns0:Message>'
)


def test_classification_uses_the_shared_camera_taxonomy():
    inspector = OnvifInspector()
    # Dahua Smart Motion names a person, even though "motion" is in the topic.
    assert inspector.classify_topic("ruleengine/smartmotionhuman") == "person"
    assert inspector.classify_topic("ruleengine/smartmotionvehicle") == "vehicle"
    assert inspector.classify_topic("ruleengine/myruledetector/petdetect") == "animal"
    assert inspector.classify_topic("ruleengine/myruledetector/facedetect") == "face"
    assert inspector.classify_topic("ruleengine/cellmotiondetector/motion") == "motion"
    # An explicit payload class names the object, whatever the state.
    assert inspector.classify_topic(
        "ruleengine/fielddetector/objectsinside",
        simple_items=(("isinside", "false"), ("objecttype", "human")),
    ) == "person"
    # Reolink doorbell, counters and device health stay unclassified.
    for topic in ("ruleengine/myruledetector/package", "ruleengine/myruledetector/visitor",
                  "ruleengine/countaggregation/counter", "monitoring/operatingtime/lastreboot",
                  "device/trigger/digitalinput", "manual"):
        assert inspector.classify_topic(topic) is None


def test_reported_state_is_kept_apart_from_motion_state():
    inspector = OnvifInspector()
    package = inspector.record(
        camera_id="front-side",
        topic="tns1:RuleEngine/MyRuleDetector/Package",
        normalized_topic="ruleengine/myruledetector/package",
        active=None,
        simple_items=(("source", "000"), ("state", "false")),
        message_xml=PACKAGE_XML,
    )
    assert package.active is None
    assert package.reported_state is False
    assert package.recognized is False
    assert package.changed is False
    # The reported value never feeds class or topic state.
    assert inspector.state_snapshot()["cameras"].get("front-side", {}).get("classes", {}) == {}

    # Without raw items the XML itself is read.
    from_xml = inspector.record(
        camera_id="front-side", topic="tns1:RuleEngine/MyRuleDetector/Visitor",
        normalized_topic="ruleengine/myruledetector/visitor", active=None,
        message_xml=PACKAGE_XML.replace('Value="false"', 'Value="true"'),
    )
    assert from_xml.reported_state is True
    counter = inspector.record(
        camera_id="upper-garage", topic="tns1:RuleEngine/CountAggregation/Counter",
        normalized_topic="ruleengine/countaggregation/counter", active=None,
        simple_items=(("count", "3"),),
    )
    assert counter.reported_state is None
    payload = inspector.events_after(0)["events"][0]
    assert payload["reported_state"] is False

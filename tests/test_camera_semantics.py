from survng.app.camera_semantics import camera_semantic_reports
from survng.app.motion_events import MotionTrigger
from datetime import datetime, timezone


def test_namespace_topic_maps_broad_vehicle_to_configured_taxonomy() -> None:
    reports = camera_semantic_reports(
        "tns1:RuleEngine/tns2:VehicleDetector/tns3:VehicleDetect",
        '<tt:SimpleItem Name="State" Value="true"/>',
        ["person", "car", "truck", "robot_lawnmower"],
    )
    assert reports == [{
        "topic": "tns1:RuleEngine/tns2:VehicleDetector/tns3:VehicleDetect",
        "category": "vehicle",
        "candidate_model_classes": ["car", "truck"],
    }]


def test_generic_rule_uses_explicit_zeep_object_type() -> None:
    reports = camera_semantic_reports(
        "RuleEngine/MyRuleDetector/Alarm",
        "{'Name': 'ObjectType', 'Value': 'dog'}",
        ["cat", "dog", "horse"],
    )
    assert reports[0]["category"] == "animal"
    assert reports[0]["reported_class"] == "dog"
    assert reports[0]["candidate_model_classes"] == ["dog"]


def test_full_namespaced_soap_payload_reads_explicit_class() -> None:
    reports = camera_semantic_reports(
        "RuleEngine/MyRuleDetector/Alarm",
        '<env:Envelope xmlns:env="urn:env" xmlns:tt="urn:onvif"><env:Body>'
        '<tt:SimpleItem Name="ObjectClass" Value="delivery_truck"/>'
        '</env:Body></env:Envelope>',
        ["car", "delivery_truck"],
    )
    assert reports[0]["category"] == "vehicle"
    assert reports[0]["candidate_model_classes"] == ["delivery_truck"]


def test_inactive_semantic_payload_does_not_report_a_claim() -> None:
    assert camera_semantic_reports(
        "RuleEngine/PersonDetect",
        '<SimpleItem Name="State" Value="false"/>',
        ["person"],
    ) == []


def test_unknown_or_configuration_topics_are_not_guessed() -> None:
    labels = ["person", "robot_lawnmower"]
    assert camera_semantic_reports("RuleEngine/PersonCount", "<broken", labels) == []
    assert camera_semantic_reports("Vendor/Analytics", "malformed", labels) == []


def test_unknown_explicit_class_does_not_broaden_a_known_topic() -> None:
    reports = camera_semantic_reports(
        "RuleEngine/VehicleDetect",
        "{'Name': 'ObjectType', 'Value': 'robot_lawnmower'}",
        ["car", "truck", "robot_lawnmower"],
    )
    assert reports[0]["reported_class"] == "robot_lawnmower"
    assert reports[0]["candidate_model_classes"] == ["robot_lawnmower"]


def test_dogcat_topic_only_suggests_configured_dog_and_cat() -> None:
    reports = camera_semantic_reports(
        "RuleEngine/MyRuleDetector/DogCatDetect", "", ["dog", "cat", "horse"]
    )
    assert reports[0]["category"] == "animal"
    assert "reported_class" not in reports[0]
    assert reports[0]["candidate_model_classes"] == ["dog", "cat"]


def test_camera_semantics_survive_durable_trigger_round_trip() -> None:
    now = datetime.now(timezone.utc)
    semantics = {"reports": [{
        "topic": "RuleEngine/PersonDetect",
        "category": "person",
        "reported_class": "person",
        "candidate_model_classes": ["person"],
    }]}
    trigger = MotionTrigger(
        "RuleEngine/PersonDetect", "", now, now.timestamp(),
        camera_semantics=semantics,
    )

    restored = MotionTrigger.from_durable_payload(trigger.durable_payload(), "job-7")

    assert restored.camera_semantics == semantics
    assert restored.delivery_job_id == "job-7"


def test_notice_admission_accepts_only_affirmative_semantic_claims() -> None:
    from survng.app.camera_semantics import camera_notice_admission, semantic_label_compatible

    labels = ["person", "car", "truck", "dog", "cat", "horse"]
    active = '<tt:SimpleItem Name="State" Value="true"/>'
    person = camera_notice_admission("tns1:RuleEngine/MyRuleDetector/PeopleDetect", active, labels)
    assert person["kind"] == "semantic"
    assert person["compatible_labels"] == ["person"]
    # A specific known alias admits its configured family; an unknown class stays exact.
    car = camera_notice_admission("Vendor/Alarm", "{'Name': 'ObjectType', 'Value': 'car'}", labels)
    assert car["compatible_labels"] == ["car", "truck"]
    mower = camera_notice_admission(
        "RuleEngine/VehicleDetect", "{'Name': 'ObjectType', 'Value': 'robot_lawnmower'}",
        [*labels, "robot_lawnmower"],
    )
    assert mower["compatible_labels"] == ["robot_lawnmower"]
    assert camera_notice_admission("RuleEngine/DogCatDetect", "", labels)["compatible_labels"] == ["dog", "cat"]
    # Plain text carries no state; the semantic topic decides, as at ingest.
    assert camera_notice_admission("RuleEngine/PersonDetect", "person", labels)["kind"] == "semantic"
    assert semantic_label_compatible("Truck", car["compatible_labels"])
    assert not semantic_label_compatible("person", car["compatible_labels"])

    for topic, message, reason in [
        ("tns1:RuleEngine/CellMotionDetector/Motion", '<tt:SimpleItem Name="IsMotion" Value="true"/>', "no_semantic_claim"),
        ("RuleEngine/PersonDetect", '<tt:SimpleItem Name="State" Value="false"/>', "state_not_affirmative"),
        ("RuleEngine/PersonDetect", '<tt:SimpleItem Name="State" Value="pending"/>', "state_not_affirmative"),
        ("RuleEngine/PersonDetect", "<tt:Message><broken", "unparsed_payload"),
        # Truncated mid-document: whole SimpleItem tags must not be harvested.
        ("RuleEngine/PersonDetect", '<tt:Message><tt:Data><tt:SimpleItem Name="State" Value="true"/>', "unparsed_payload"),
        ("RuleEngine/PersonDetect", "<tt:Message><tt:Data/></tt:Message>", "unparsed_payload"),
        ("RuleEngine/PersonCount", active, "no_semantic_claim"),
    ]:
        admission = camera_notice_admission(topic, message, labels)
        assert admission == {"version": 1, "kind": "generic", "reason": reason}


def test_undeclared_prefix_fragment_reads_its_state() -> None:
    assert camera_semantic_reports(
        "RuleEngine/PersonDetect", '<tt:SimpleItem Name="State" Value="false"/>', ["person"]
    ) == []


def test_dahua_ivs_cleared_intrusion_is_not_a_semantic_claim() -> None:
    from survng.app.camera_semantics import camera_notice_admission

    topic = "tns1:RuleEngine/FieldDetector/ObjectsInside"

    def payload(inside: str) -> str:
        return (
            '<tt:Data><tt:SimpleItem Name="IsInside" Value="' + inside + '"/>'
            '<tt:SimpleItem Name="ObjectType" Value="Human"/></tt:Data>'
        )

    entered = camera_notice_admission(topic, payload("true"), ["person", "car"])
    assert entered["kind"] == "semantic"
    assert entered["compatible_labels"] == ["person"]
    assert camera_semantic_reports(topic, payload("false"), ["person"]) == []
    assert camera_notice_admission(topic, payload("false"), ["person"]) == {
        "version": 1, "kind": "generic", "reason": "state_not_affirmative",
    }


def test_dahua_smart_motion_topics_are_semantic() -> None:
    from survng.app.camera_semantics import camera_notice_admission

    active = '<tt:SimpleItem Name="State" Value="true"/>'
    labels = ["person", "car", "truck"]
    human = camera_notice_admission("tns1:RuleEngine/SmartMotionHuman", active, labels)
    vehicle = camera_notice_admission("tns1:RuleEngine/SmartMotionVehicle", active, labels)
    assert (human["kind"], human["compatible_labels"]) == ("semantic", ["person"])
    assert (vehicle["kind"], vehicle["compatible_labels"]) == ("semantic", ["car", "truck"])
    cleared = camera_notice_admission(
        "tns1:RuleEngine/SmartMotionHuman", '<tt:SimpleItem Name="State" Value="false"/>', labels
    )
    assert cleared["kind"] == "generic"

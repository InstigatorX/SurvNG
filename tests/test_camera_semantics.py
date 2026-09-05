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

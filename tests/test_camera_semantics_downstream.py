import json

from survng.app.incident_presenter import _event_row, _incident_list_payload, _incident_row
from survng.app.incident_payload import canonical_incident_payload


def _row(event_id: int = 7) -> dict:
    qualification = {
        "status": "motion_qualification",
        "motion_qualification": {"camera_semantics": {"reports": [{
            "topic": "RuleEngine/VehicleDetect",
            "category": "vehicle",
            "candidate_model_classes": ["car", "truck"],
        }]}},
    }
    return {
        "id": event_id,
        "camera_id": "gate",
        "kind": "motion",
        "topic": "RuleEngine/VehicleDetect",
        "message": "",
        "created_at": "2026-09-05T12:00:00+00:00",
        "snapshot_path": "",
        "recording_path": "",
        "objects_json": json.dumps([qualification]),
    }


def test_api_incident_preserves_event_and_aggregated_semantic_provenance() -> None:
    event = _event_row(_row())
    incident = _incident_row("gate", [event])
    compact = _incident_list_payload(incident)

    assert event["labels"] == []
    assert event["has_objects"] is False
    assert compact["events"][0]["camera_semantics"]["reports"][0]["category"] == "vehicle"
    report = compact["camera_semantics"]["reports"][0]
    assert report["source_event_id"] == 7
    assert report["source_created_at"] == "2026-09-05T12:00:00+00:00"


def test_mqtt_incident_exposes_semantics_without_creating_classes() -> None:
    event = _row()
    pending = _incident_row("gate", [_event_row(event)])
    payload = canonical_incident_payload(pending)

    assert payload["has_objects"] is False
    assert payload["classes"] == []
    assert payload["camera_semantics"]["reports"][0]["source_event_id"] == 7

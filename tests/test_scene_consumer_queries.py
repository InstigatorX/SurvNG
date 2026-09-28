"""Assistant queries use canonical incident scope rather than current covers."""
from types import SimpleNamespace
from unittest.mock import patch

from survng.app import main
from survng.app.assistant import AssistantToolCall


def incident():
    return {
        "id": "scene-1", "incident_id": "scene-1", "camera_id": "gate",
        "camera_ids": ["gate", "drive"], "representative_event_id": 7,
        "start_at": "2026-09-12T12:00:00+00:00", "end_at": "2026-09-12T12:01:00+00:00",
        "labels": ["person"], "zones": [], "has_objects": True,
        "scene_objects": [{"label": "person", "confidence": .75, "incident_eligible": False}],
        "events": [{"id": 7, "camera_id": "gate", "objects": []}],
    }


def test_assistant_confidence_filter_includes_observation_missing_from_current_cover():
    scene = incident()
    manager = SimpleNamespace(events=SimpleNamespace(list_scene_incidents=lambda **kwargs: [scene]))
    service = main._intelligence_route_bundle.service
    with (patch.object(service.deps.incident_queries, "hydrate", return_value=[scene]),
          patch.object(service.deps.incident_queries, "with_faces", return_value=[scene])):
        evidence = service._assistant_search_incidents(
            AssistantToolCall(name="search_incidents", minimum_confidence=.7), "UTC", manager,
        )
    assert evidence[0].data["returned"] == 1
    assert evidence[1].data["scene_objects"][0]["confidence"] == .75
    assert evidence[1].href == "/incidents/scene-1"


def test_assistant_counts_all_cameras_in_canonical_incident():
    manager = SimpleNamespace(events=SimpleNamespace(list_scene_incidents=lambda **kwargs: [incident()]))
    evidence = main._intelligence_route_bundle.service._assistant_recent_activity_summary(
        AssistantToolCall(name="summarize_recent_activity"), "UTC", manager,
    )
    assert evidence.data["incident_count"] == 1
    assert evidence.data["camera_counts"] == {"drive": 1, "gate": 1}
    assert "2 cameras" in evidence.summary

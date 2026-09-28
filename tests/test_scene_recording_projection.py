"""Recording lanes preserve canonical incident identity and camera geometry."""
from types import SimpleNamespace
from unittest.mock import Mock

from survng.app.incident_presenter import _event_row, _incident_row
from survng.app.recording_routes import _identity_hydrated_recording_incidents


def test_camera_lanes_resolve_full_incident_but_use_local_media_and_timing():
    events = [_event_row({
        "id": event_id, "camera_id": camera, "created_at": at,
        "snapshot_path": f"{event_id}.jpg", "objects_json": '[{"label":"person","confidence":0.8}]',
    }) for event_id, camera, at in [
        (1, "gate", "2026-09-12T12:00:00+00:00"),
        (2, "drive", "2026-09-12T12:01:00+00:00"),
    ]]
    canonical = {**_incident_row("gate", events), "id": "scene-1", "incident_id": "scene-1",
                 "camera_ids": ["gate", "drive"], "episodes": [
                     {"camera_id": "gate", "start_at": events[0]["created_at"], "end_at": "2026-09-12T12:00:20+00:00"},
                     {"camera_id": "drive", "start_at": events[1]["created_at"], "end_at": "2026-09-12T12:01:30+00:00"},
                 ]}
    store = SimpleNamespace(scene_incident=Mock(return_value=canonical))
    lanes = _identity_hydrated_recording_incidents(SimpleNamespace(events=store), events, include_identities=False)
    assert len(lanes) == 2
    assert {lane["id"] for lane in lanes} == {"scene-1"}
    assert {lane["camera_id"]: lane["representative_event_id"] for lane in lanes} == {"gate": 1, "drive": 2}
    assert {lane["camera_id"]: lane["duration_seconds"] for lane in lanes} == {"gate": 20, "drive": 30}
    assert all(lane["camera_ids"] == ["gate", "drive"] for lane in lanes)
    store.scene_incident.assert_called_once_with(event_id=1)

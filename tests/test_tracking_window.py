from datetime import datetime, timezone

from survng.app.tracking_window import recorded_tracking_window


def test_leadin_includes_motion_evidence_and_tail_is_independent_of_cover_time():
    at = datetime(2026, 9, 24, 20, 25, 45, tzinfo=timezone.utc)
    event = {"objects": [{"status": "motion_qualification", "motion_qualification": {
        "features": {"persistence_seconds": 4.3945}}}]}
    start, end = recorded_tracking_window(event, at, before=5, after=5, activity_seconds=15)
    assert abs(start - (at.timestamp() - 9.3945)) < .00001
    assert end == at.timestamp() + 20


def test_invalid_motion_history_cannot_make_unbounded_or_nonfinite_window():
    at = datetime.now(timezone.utc)
    for value in [float('nan'), float('inf'), None, 'bad', -1]:
        event = {"objects": [{"status": "motion_qualification", "motion_qualification": {
            "features": {"persistence_seconds": value}}}]}
        assert recorded_tracking_window(event, at, before=5, after=5, activity_seconds=15)[0] == at.timestamp() - 5


def test_snapshot_track_identity_can_be_assigned_after_leadin_progress(tmp_path):
    import json
    from survng.app.events import EventStore
    store = EventStore(tmp_path)
    box = {"x1": 10, "y1": 10, "x2": 40, "y2": 80}
    event = store.add_event("gate", "motion", objects_json=json.dumps([
        {"label": "person", "confidence": .9, "incident_eligible": True, "box": box}
    ]))
    store.update_object_tracking(event['id'], {"state": "active", "tracks": [], "snapshot_track_assignments": []})
    # Track 1 belongs to a different early object; the snapshot person is 2.
    assignment = {"label": "person", "box": box, "track_id": 2, "track_state": "confirmed", "track_observations": 10}
    updated = store.update_object_tracking(event['id'], {"state": "active", "tracks": [], "snapshot_track_assignments": [assignment]})
    assert json.loads(updated['objects_json'])[0]['track_id'] == 2
    # A later cover with different geometry must not inherit this old mapping.
    assignment['box'] = {**box, "x1": 20}
    assignment['track_id'] = 3
    updated = store.update_object_tracking(event['id'], {"state": "active", "tracks": [], "snapshot_track_assignments": [assignment]})
    assert json.loads(updated['objects_json'])[0]['track_id'] == 2

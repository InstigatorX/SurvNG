"""Native incident lifecycle and transport-independent recovery."""
from unittest.mock import Mock

from survng.app.incident_lifecycle import IncidentLifecycle


def evidence(event_id=41, **changes):
    return {"id": event_id, "camera_id": "gate", "created_at": "2026-09-12T12:00:00+00:00",
            "snapshot_path": "snapshots/gate.jpg",
            "objects": [{"label": "person", "confidence": 0.9, "zones": ["porch"]}], **changes}


def test_native_lifecycle_and_late_identity_without_mqtt(tmp_path):
    published = []
    lifecycle = IncidentLifecycle(published.append, tmp_path / "incidents.json")
    lifecycle.start()
    try:
        lifecycle.track_incident(evidence(), "Gate", "/survng")
        lifecycle.track_incident(evidence(snapshot_path="snapshots/better.jpg"), "Gate", "/survng", allow_new=False)
        key = published[0]["incident_id"]
        lifecycle._groups[key]["settle_at"] = 0
        lifecycle._settle(key)
        lifecycle.track_incident(evidence(identities=[{"identity_id": 1, "name": "Alex", "confidence": 0.95}]),
                                 "Gate", "/survng", allow_new=False)
        assert [item["state"] for item in published] == ["new", "updated", "complete", "complete"]
        assert [item["revision"] for item in published] == [1, 2, 3, 4]
        assert len({item["incident_id"] for item in published}) == 1
        assert published[-1]["summary"] == "Alex detected at Gate."
        assert published[-1]["people"] == ["Alex"]
        assert published[-1]["completed_at"] == published[-2]["completed_at"]
        assert "image" in published[1]["changed_fields"]
        assert published[0]["objects"][0]["observation_count"] == 1
    finally:
        lifecycle.close()


def test_restart_retains_revisions_and_does_not_complete_on_shutdown(tmp_path):
    path = tmp_path / "incidents.json"
    lifecycle = IncidentLifecycle(Mock(), path)
    lifecycle.start()
    lifecycle.track_incident(evidence(), "Gate")
    lifecycle.close()
    published = []
    restored = IncidentLifecycle(published.append, path)
    assert restored.snapshot()[0]["state"] == "new"
    assert restored.snapshot()[0]["revision"] == 1
    restored.start()
    try:
        key = restored.snapshot()[0]["incident_id"]
        restored._groups[key]["settle_at"] = 0
        restored._settle(key)
        assert published[0]["state"] == "complete"
        assert published[0]["revision"] == 2
    finally:
        restored.close()


def test_refinement_does_not_extend_settlement_or_open_unknown_incident():
    lifecycle = IncidentLifecycle(Mock())
    lifecycle.start()
    try:
        lifecycle.track_incident(evidence(), "Gate", allow_new=False)
        assert not lifecycle.snapshot()
        lifecycle.track_incident(evidence(), "Gate")
        key = lifecycle.snapshot()[0]["incident_id"]
        deadline = lifecycle._groups[key]["settle_at"]
        lifecycle.track_incident(evidence(), "Gate", allow_new=False)
        assert lifecycle._groups[key]["settle_at"] == deadline
        lifecycle._settle(key)  # An obsolete timer may race with an extension.
        assert lifecycle.snapshot()[0]["state"] == "updated"
    finally:
        lifecycle.close()


def test_recovery_snapshot_is_detached_from_live_state():
    lifecycle = IncidentLifecycle(Mock())
    lifecycle.start()
    try:
        lifecycle.track_incident(evidence(), "Gate")
        snapshot = lifecycle.snapshot()
        snapshot[0]["objects"].clear()
        assert lifecycle.snapshot()[0]["objects"]
    finally:
        lifecycle.close()


def test_gap_completes_old_incident_and_preserves_old_refinement_identity():
    published = []
    lifecycle = IncidentLifecycle(published.append)
    lifecycle.start()
    try:
        lifecycle.track_incident(evidence(), "Gate")
        lifecycle.track_incident(evidence(42, created_at="2026-09-12T12:02:00+00:00"), "Gate")
        assert [item["state"] for item in published] == ["new", "complete", "new"]
        lifecycle.track_incident(evidence(identities=[{"identity_id": 2, "name": "Sam"}]), "Gate", allow_new=False)
        assert published[-1]["incident_id"] == published[0]["incident_id"]
        assert published[-1]["state"] == "complete"
        assert lifecycle.snapshot()[0]["incident_id"] != lifecycle.snapshot()[1]["incident_id"]
    finally:
        lifecycle.close()


def test_missing_image_is_explicit_and_close_cancels_timers():
    lifecycle = IncidentLifecycle(Mock())
    lifecycle.start()
    lifecycle.track_incident(evidence(snapshot_path=""), "Gate")
    payload = lifecycle.snapshot()[0]
    assert payload["snapshot_url"] is None
    assert payload["image_available"] is False
    lifecycle.close()
    assert not lifecycle._timers
    lifecycle.track_incident(evidence(42), "Gate")
    assert len(lifecycle.snapshot()) == 1

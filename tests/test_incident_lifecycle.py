"""Canonical incident revisions are the sole notification membership source."""
from copy import deepcopy
from unittest.mock import Mock

from survng.app.incident_lifecycle import CanonicalIncidentLifecycle, IncidentLifecycle


def scene(revision=1):
    return {
        "id": "scene-1", "incident_id": "scene-1", "revision": revision,
        "state": "active", "camera_id": "gate", "camera_ids": ["gate", "drive"],
        "start_at": "2026-09-12T12:00:00+00:00", "end_at": "2026-09-12T12:00:20+00:00",
        "summary": "Two people observed across Gate and Drive.",
        "scene_objects": [
            {"id": "person-1", "label": "person", "confidence": .94},
            {"id": "person-2", "label": "person", "confidence": .75,
             "incident_eligible": False, "zone_admission_reason": "outside_incident_zone"},
        ],
        "episodes": [{"camera_id": "gate"}, {"camera_id": "drive"}],
        "events": [{"id": 80913}, {"id": 80914}],
        "alert_decisions": [{"event_id": 80913, "eligible": True}],
    }


class Scenes:
    def __init__(self, *snapshots):
        self.pending = [dict(incident_id=item["incident_id"], revision=item["revision"], payload=deepcopy(item))
                        for item in snapshots]
        self.current = deepcopy(snapshots[-1]) if snapshots else scene()
        self.acknowledged = []

    def scene_pending_notifications(self):
        return deepcopy(self.pending)

    def settle_scene_incidents(self):
        pass

    def acknowledge_scene_notification(self, incident_id, revision):
        self.acknowledged.append((incident_id, revision))
        self.pending = [item for item in self.pending if (item["incident_id"], item["revision"]) != (incident_id, revision)]

    def list_scene_incidents(self):
        return [deepcopy(self.current)]

    def scene_incident(self, incident_id=None, event_id=None):
        return deepcopy(self.current) if incident_id == self.current["incident_id"] else None


def test_canonical_payload_contains_every_subject_and_camera():
    store = Scenes(scene())
    published = []
    lifecycle = IncidentLifecycle(store, published.append)
    assert isinstance(lifecycle, CanonicalIncidentLifecycle)
    assert lifecycle.run_once() == 1
    payload = published[0]
    assert payload["schema_version"] == 3
    assert [item["confidence"] for item in payload["objects"]] == [.94, .75]
    assert payload["camera_ids"] == ["gate", "drive"]
    assert payload["event_ids"] == [80913, 80914]
    assert payload["summary"] == store.current["summary"]
    assert payload["alert_decisions"] == store.current["alert_decisions"]
    assert store.acknowledged == [("scene-1", 1)]


def test_replay_publishes_snapshot_revision_without_substituting_latest_state():
    first, latest = scene(1), scene(2)
    latest["scene_objects"].append({"id": "car-1", "label": "car"})
    store = Scenes(first, latest)
    published = []
    assert IncidentLifecycle(store, published.append).run_once() == 2
    assert [item["revision"] for item in published] == [1, 2]
    assert [len(item["objects"]) for item in published] == [2, 3]


def test_publication_failure_stops_ordered_replay_and_retries_after_restart():
    store = Scenes(scene(1), scene(2))
    failed = IncidentLifecycle(store, Mock(side_effect=RuntimeError("offline")))
    assert failed.run_once() == 0
    assert not store.acknowledged
    failed.close()
    published = []
    restored = IncidentLifecycle(store, published.append)
    restored.start()
    try:
        assert [item["revision"] for item in published] == [1, 2]
        assert not store.pending
    finally:
        restored.close()


def test_acknowledgement_failure_permits_at_least_once_delivery():
    store = Scenes(scene())
    acknowledge = store.acknowledge_scene_notification
    store.acknowledge_scene_notification = Mock(side_effect=RuntimeError("db unavailable"))
    published = []
    lifecycle = IncidentLifecycle(store, published.append)
    assert lifecycle.run_once() == 0
    assert len(store.pending) == 1
    store.acknowledge_scene_notification = acknowledge
    assert lifecycle.run_once() == 1
    assert [(item["incident_id"], item["revision"]) for item in published] == [("scene-1", 1)] * 2


def test_invalid_snapshot_revision_is_not_published_or_acknowledged():
    store = Scenes(scene())
    store.pending[0]["payload"]["revision"] = 2
    publish = Mock()
    assert IncidentLifecycle(store, publish).run_once() == 0
    publish.assert_not_called()
    assert not store.acknowledged


def test_legacy_outbox_purge_runs_in_bounded_slices_before_publication():
    store = Scenes(scene())
    calls = []
    store.purge_legacy_scene_notifications = lambda: calls.append(len(calls)) or (10 if len(calls) < 3 else 0)
    published = []
    assert IncidentLifecycle(store, published.append).run_once() == 1
    assert len(calls) == 3
    assert [item["revision"] for item in published] == [1]


def test_reads_return_detached_canonical_state_without_notifying():
    store = Scenes()
    publish = Mock()
    lifecycle = IncidentLifecycle(store, publish)
    returned = lifecycle.snapshot()[0]
    returned["objects"].clear()
    assert len(lifecycle.get("scene-1")["objects"]) == 2
    assert lifecycle.get("missing") is None
    publish.assert_not_called()


def test_wakeup_never_changes_membership_or_completes_incident_on_close():
    store = Scenes()
    published = []
    lifecycle = IncidentLifecycle(store, published.append)
    lifecycle.start()
    try:
        lifecycle.track_incident({"id": 999, "camera_id": "new-camera"}, "New Camera", allow_new=True)
        assert lifecycle.snapshot()[0]["event_ids"] == [80913, 80914]
        assert not published
    finally:
        lifecycle.close()
    assert store.current["state"] == "active"
    lifecycle.track_incident({"id": 1000}, allow_new=False)
    assert not published

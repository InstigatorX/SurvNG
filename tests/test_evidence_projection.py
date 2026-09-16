"""Outbox delivery waits for applied evidence and retries without duplicate work."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest

from survng.app.evidence_projection import EvidenceProjection
from survng.app.state_events import StateEventBroker


class Events:
    def __init__(self, rows=None):
        self.rows = rows or [{"id": 1, "event_id": 7, "evidence_revision": 2, "kind": "evidence_updated"}]
        self.event = {"id": 7, "camera_id": "gate", "evidence_revision": 2,
                      "snapshot_path": "new.png", "objects": [{"label": "person"}]}
        self.fail_ack = False

    def pending_evidence_updates(self, *, limit, after_id=0):
        return copy.deepcopy([row for row in self.rows if row["id"] > after_id][:limit])

    def get(self, event_id):
        return copy.deepcopy(self.event)

    def mark_evidence_publication(self, outbox_ids):
        for row in self.rows:
            if row["id"] in outbox_ids:
                row["publication_done"] = True
        return True

    def acknowledge_evidence_update(self, outbox_id):
        if self.fail_ack:
            raise RuntimeError("temporary acknowledgement failure")
        before = len(self.rows)
        self.rows = [row for row in self.rows if row["id"] != outbox_id]
        return len(self.rows) != before


def setup_projection(events=None):
    events = events or Events()
    semantic = SimpleNamespace(
        config=SimpleNamespace(enabled=True), index=SimpleNamespace(delete_event=Mock()),
        projection_current=Mock(return_value=False), projection_pending=Mock(return_value=False),
        queue_event=Mock(return_value=True),
    )
    broker = StateEventBroker()
    subscriber = broker.subscribe()
    notification = Mock()
    worker = EvidenceProjection(events, lambda: semantic, broker, notification, retry_seconds=5)
    return worker, events, semantic, subscriber, notification


def test_ack_waits_for_applied_projection_and_coalesces_revisions():
    events = Events([
        {"id": 1, "event_id": 7, "evidence_revision": 1, "kind": "evidence_updated"},
        {"id": 2, "event_id": 7, "evidence_revision": 2, "kind": "evidence_updated"},
        {"id": 3, "event_id": 7, "evidence_revision": 1, "kind": "cover_required"},
    ])
    worker, events, semantic, subscriber, notification = setup_projection(events)
    assert worker.run_once() == 1  # Requirement is durable independently of its wakeup.
    assert semantic.queue_event.call_count == 1
    assert semantic.queue_event.call_args.args[0]["evidence_revision"] == 2
    notification.assert_called_once_with("gate", 7)
    message = subscriber.get_nowait()
    assert message.type == "incident" and message.data["evidence_revision"] == 2
    semantic.projection_pending.return_value = True
    for _ in range(3):
        assert worker.run_once() == 0
    assert semantic.queue_event.call_count == 1
    semantic.projection_current.return_value = True
    assert worker.run_once() == 2
    assert not events.rows
    notification.assert_called_once_with("gate", 7)
    assert subscriber.empty()
    assert worker.run_once() == 0 and subscriber.empty()


def test_failed_enqueue_is_retried_with_backoff():
    worker, events, semantic, subscriber, notification = setup_projection()
    semantic.queue_event.return_value = False
    with patch("survng.app.evidence_projection.time.monotonic", return_value=100):
        assert worker.run_once() == 0
        assert worker.run_once() == 0
    assert semantic.queue_event.call_count == 1
    with patch("survng.app.evidence_projection.time.monotonic", return_value=106):
        assert worker.run_once() == 0
    assert semantic.queue_event.call_count == 2
    assert len(events.rows) == 1 and notification.call_count == 1
    assert subscriber.get_nowait().type == "incident" and subscriber.empty()


def test_ack_failure_does_not_repeat_published_revision_in_same_process():
    worker, events, semantic, subscriber, notification = setup_projection()
    semantic.projection_current.return_value = True
    events.fail_ack = True
    assert worker.run_once() == 0
    assert subscriber.get_nowait().type == "incident"
    events.fail_ack = False
    assert worker.run_once() == 1
    notification.assert_called_once()
    assert subscriber.empty()


def test_notification_failure_retains_outbox_and_retries():
    worker, events, semantic, subscriber, notification = setup_projection()
    semantic.projection_current.return_value = True
    notification.side_effect = RuntimeError("temporarily unavailable")
    assert worker.run_once() == 0
    assert subscriber.empty() and len(events.rows) == 1
    notification.side_effect = None
    assert worker.run_once() == 1
    assert subscriber.get_nowait().data["evidence_revision"] == 2


def test_disabled_and_no_object_events_do_not_wait_for_model():
    worker, events, semantic, subscriber, notification = setup_projection()
    semantic.config.enabled = False
    assert worker.run_once() == 1
    semantic.queue_event.assert_not_called()


def test_no_admitted_objects_clear_old_vectors_without_waiting_for_model():
    worker, events, semantic, _, _ = setup_projection()
    events.event["objects"] = [{"label": "car", "incident_eligible": False}]
    assert worker.run_once() == 1
    semantic.index.delete_event.assert_called_once_with(7, expected_event=events.event)
    semantic.queue_event.assert_not_called()
    worker, events, semantic, subscriber, notification = setup_projection()
    events.event["objects"] = []
    assert worker.run_once() == 1
    semantic.index.delete_event.assert_called_once_with(7, expected_event=events.event)
    semantic.queue_event.assert_not_called()


def test_changed_event_during_projection_waits_for_its_new_revision():
    worker, events, semantic, subscriber, notification = setup_projection()
    def advanced(_):
        events.event["evidence_revision"] = 3
        return True
    semantic.projection_current.side_effect = advanced
    assert worker.run_once() == 0
    assert notification.call_count == 0 and subscriber.empty()
    semantic.projection_current.side_effect = None
    semantic.projection_current.return_value = True
    assert worker.run_once() == 1
    assert subscriber.get_nowait().data["evidence_revision"] == 3


def test_requirement_only_transition_updates_clients_without_reindex():
    events = Events([{"id": 1, "event_id": 7, "evidence_revision": 2, "kind": "cover_requirement_updated"}])
    events.event["cover_requirement"] = {"state": "exhausted", "reason": "association_ambiguous", "attempts": 2}
    worker, events, semantic, subscriber, notification = setup_projection(events)
    assert worker.run_once() == 1
    semantic.queue_event.assert_not_called()
    semantic.projection_current.assert_not_called()
    assert subscriber.get_nowait().data["cover_requirement"]["state"] == "exhausted"


def test_restart_reconciles_durable_row_and_thread_stops():
    worker, events, semantic, subscriber, notification = setup_projection()
    assert worker.run_once() == 0
    subscriber.get_nowait()
    restarted = EvidenceProjection(events, lambda: semantic, worker.state_events, notification, poll_seconds=.1)
    semantic.projection_current.return_value = True
    restarted.start()
    first_thread = restarted._thread
    restarted.start()
    assert restarted._thread is first_thread
    try:
        # Publication was already durably marked before the simulated restart.
        first_thread.join(.15)
    finally:
        restarted.stop()
    assert not first_thread.is_alive()
    assert not events.rows
    assert subscriber.empty()


def test_unavailable_index_does_not_starve_later_outbox_pages():
    events = Events([
        {"id": number, "event_id": number, "evidence_revision": 2, "kind": "evidence_updated"}
        for number in range(1, 5)
    ])
    events.get = lambda event_id: {**events.event, "id": event_id}
    worker, events, semantic, subscriber, notification = setup_projection(events)
    worker.batch_size = 2
    assert worker.run_once() == 0
    assert worker.run_once() == 0
    assert [call.args[1] for call in notification.call_args_list] == [1, 2, 3, 4]
    assert len(events.rows) == 4
    assert worker.run_once() == 0  # Wrap and retry old work without repeating delivery.
    assert notification.call_count == 4


def test_stop_failure_keeps_thread_and_dependency_ownership():
    worker, _, _, _, _ = setup_projection()
    thread = Mock()
    thread.is_alive.return_value = True
    worker._thread = thread
    with pytest.raises(RuntimeError, match="dependencies must remain open"):
        worker.stop()
    assert worker._thread is thread


def test_semantic_failure_still_invalidates_committed_image():
    worker, events, semantic, subscriber, notification = setup_projection()
    semantic.projection_current.side_effect = RuntimeError("index database unavailable")
    assert worker.run_once() == 0
    assert subscriber.get_nowait().data["reason"] == "evidence_updated"
    assert len(events.rows) == 1


def test_later_face_metadata_refreshes_notification_without_visual_revision():
    worker, events, semantic, subscriber, notification = setup_projection()
    semantic.projection_current.return_value = True
    assert worker.run_once() == 1
    subscriber.get_nowait()
    events.rows.append({"id": 2, "event_id": 7, "evidence_revision": 2, "kind": "incident_metadata_updated"})
    assert worker.run_once() == 1
    assert notification.call_count == 2
    assert subscriber.get_nowait().data["evidence_revision"] == 2
    semantic.queue_event.assert_not_called()


def test_real_store_cover_commit_converges_after_projection_restart(tmp_path):
    import json
    import cv2
    import numpy as np
    from survng.app.config import SemanticSearchConfig
    from survng.app.events import EventStore
    from survng.app.semantic_search import SemanticIndex, SemanticModelIdentity, SemanticSearchService

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    for name, color in (("old.png", 0), ("new.png", 255)):
        assert cv2.imwrite(str(snapshots / name), np.full((20, 20, 3), color, dtype=np.uint8))
    events = EventStore(tmp_path)
    objects = json.dumps([{"label": "person", "box": {"x1": 1, "y1": 1, "x2": 19, "y2": 19}}])
    event = events.add_event(camera_id="gate", kind="motion", snapshot_path="snapshots/old.png", objects_json=objects)
    identity = SemanticModelIdentity("test", "model", "prep", 2)
    semantic = SemanticSearchService(SemanticSearchConfig(enabled=True), SemanticIndex(events.db_path), tmp_path, {})
    semantic._storage_dir = tmp_path
    semantic._event_store = events
    semantic.encoder = SimpleNamespace(identity=identity, encode_images=lambda images: np.array([[1, 0] for _ in images]))
    broker = StateEventBroker()
    worker = EvidenceProjection(events, lambda: semantic, broker, Mock())
    assert worker.run_once() == 0
    _, _, queued = semantic._queue.get_nowait()
    assert semantic.index_event(queued) == 2
    assert worker.run_once() == 1
    events.refine_event_evidence(event["id"], snapshot_path="snapshots/new.png", recording_path="", objects_json=objects)
    reopened = EventStore(tmp_path)
    restarted = EvidenceProjection(reopened, lambda: semantic, broker, Mock())
    assert restarted.run_once() == 0
    _, _, queued = semantic._queue.get_nowait()
    assert semantic.index_event(queued) == 2
    assert restarted.run_once() == 1
    assert not reopened.pending_evidence_updates()
    current = reopened.get(event["id"])
    assert semantic.projection_current(current)
    assert {hit.image_path for hit in semantic.index.search([1, 0], identity)} == {"snapshots/new.png"}


def test_publication_ack_survives_bounded_cache_eviction_and_restart():
    events = Events([
        {"id": number, "event_id": number, "evidence_revision": 2, "kind": "evidence_updated"}
        for number in range(1, 301)
    ])
    events.get = lambda event_id: {**events.event, "id": event_id}
    worker, events, semantic, subscriber, notification = setup_projection(events)
    for _ in range(5):
        assert worker.run_once() == 0
    assert notification.call_count == 300
    assert len(worker._delivery) == 256
    assert all(row.get("publication_done") for row in events.rows)
    restarted = EvidenceProjection(events, lambda: semantic, worker.state_events, notification)
    for _ in range(5):
        assert restarted.run_once() == 0
    assert notification.call_count == 300


def test_expired_requirements_settle_without_camera_worker_and_are_rate_limited():
    worker, events, semantic, subscriber, notification = setup_projection()
    events.expire_cover_requirements = Mock()
    with patch("survng.app.evidence_projection.time.monotonic", return_value=100):
        worker.run_once()
        worker.run_once()
    assert events.expire_cover_requirements.call_count == 1
    with patch("survng.app.evidence_projection.time.monotonic", return_value=131):
        worker.run_once()
    assert events.expire_cover_requirements.call_count == 2

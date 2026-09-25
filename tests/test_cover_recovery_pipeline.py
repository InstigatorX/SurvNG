from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app.events import EventStore
from survng.app.motion_incidents import MotionIncidentService
from survng.app.motion_pipeline.decision_handler import MotionDecisionHandler
from survng.app.motion_pipeline.object_detection import RecordedDetectionResult


def make_pipeline(tmp_path, *, objects=None):
    store = EventStore(tmp_path)
    event = store.add_event(
        camera_id="gate", kind="motion", snapshot_path="live.webp",
        created_at=datetime.fromtimestamp(1000, timezone.utc).isoformat(),
        objects_json=json.dumps([{
            "label": "person", "confidence": .8, "incident_eligible": True,
            "provisional_detection": True, "frame_source": "live_fast_path",
            "frame_captured_at_epoch": 1000., "detection_frame_width": 640,
            "detection_frame_height": 360,
            "box": {"x1": 100, "y1": 100, "x2": 140, "y2": 200},
        }, {"status": "face_evidence_pending"}]),
    )
    provider = Mock(return_value=RecordedDetectionResult(
        frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
        objects=objects if objects is not None else [{
            "label": "person", "confidence": .9, "temporal_consensus": True,
            "box": {"x1": 300, "y1": 300, "x2": 500, "y2": 800},
        }], recording_path="main.mp4", frame_source="recorded_main",
        frame_captured_at_epoch=1002., frame_timestamp_exact=True, timings_ms={},
    ))
    publish, tracking = Mock(), Mock()
    handler = MotionDecisionHandler(
        camera_id="gate", events=store, detection_provider=provider,
        snapshot_writer=lambda *_: "main.webp", object_serializer=json.dumps,
        refinement_cover_promoter=store.promote_refinement_cover, event_callback=publish,
    )
    service = MotionIncidentService(
        camera_id="gate", decision_processor=handler, tracking_enabled=lambda: False,
        has_trackable_objects=lambda _: False, start_tracking=tracking,
        prewarm_tracking=Mock(), image_reader=Mock(), refinement_store=store,
    )
    return store, event, service, provider, publish, tracking


def due(store):
    with store._connect() as conn:
        conn.execute("update event_cover_requirements set available_at_epoch=0")


def test_cover_recovery_survives_restart_without_security_job_or_tracking(tmp_path):
    store, event, service, provider, publish, tracking = make_pipeline(tmp_path)
    service.refinement_store = EventStore(tmp_path)
    due(store)
    assert service._run_cover_requirement()
    assert store.cover_requirement(event["id"])["state"] == "satisfied"
    updated = store.get(event["id"])
    assert updated["snapshot_path"] == "main.webp"
    assert updated["evidence_revision"] == 2
    assert len(store.recent()) == 1
    tracking.assert_not_called()
    assert all(call.args[0] != "object" for call in publish.call_args_list)
    assert provider.call_args.args[1]["evidence_sampling"]["minimum_last_offset_seconds"] == 4
    assert store.evidence_attempts(event["id"])


def test_cover_recovery_exhausts_with_reason_and_preserves_original(tmp_path):
    store, event, service, provider, publish, tracking = make_pipeline(tmp_path, objects=[])
    for _ in range(3):
        due(store)
        assert service._run_cover_requirement()
    requirement = store.cover_requirement(event["id"])
    assert requirement["state"] == "exhausted"
    assert requirement["reason"]
    assert requirement["attempts"] == 3
    assert store.get(event["id"])["snapshot_path"] == "live.webp"
    assert not service._run_cover_requirement()
    tracking.assert_not_called()
    assert len(store.evidence_attempts(event["id"])) == 3


def test_delayed_refinement_cannot_overwrite_concurrent_cover(tmp_path):
    from survng.app.event_store.store import EventSnapshotChangedError
    store, event, service, provider, publish, tracking = make_pipeline(tmp_path)
    original_result = provider.return_value

    def race(*args):
        store.refine_event_evidence(event["id"], snapshot_path="newer.webp",
                                    recording_path="main.mp4", objects_json="[]")
        return original_result

    provider.side_effect = race
    with pytest.raises(EventSnapshotChangedError):
        service.decision_processor.refine(
            "motion", "", datetime.fromtimestamp(1000, timezone.utc), {},
            existing_event_id=event["id"], require_motion_correlation=False,
        )
    assert store.get(event["id"])["snapshot_path"] == "newer.webp"
    publish.assert_not_called()


def test_duplicate_cover_refresh_does_not_publish_another_notification():
    from survng.app.incident_lifecycle import IncidentLifecycle
    published = Mock()
    lifecycle = IncidentLifecycle(published)
    event = {"id": 1, "camera_id": "gate", "created_at": datetime.now(timezone.utc).isoformat(),
             "snapshot_path": "cover.webp", "objects_json": "[]", "evidence_revision": 2}
    lifecycle.start()
    try:
        lifecycle.track_incident(event, "Gate")
        lifecycle.track_incident(dict(event), "Gate", allow_new=False)
        assert published.call_count == 1
        lifecycle.track_incident({**event, "evidence_revision": 3}, "Gate", allow_new=False)
        assert published.call_count == 2
    finally:
        lifecycle.close()


def test_projection_shutdown_failure_keeps_shared_dependencies_alive():
    from tests.test_manager_lifecycle import manager_with_mocks
    from survng.app.manager import ManagerShutdownIncompleteError
    manager = manager_with_mocks()
    manager.evidence_projection.stop.side_effect = RuntimeError("worker still active")
    with pytest.raises(ManagerShutdownIncompleteError):
        manager.stop_all()
    assert not manager._closed
    manager.inference.close.assert_not_called()
    manager.state_events.close.assert_not_called()


def test_security_admission_preempts_recovery_without_spending_attempt(tmp_path):
    import threading
    from survng.app.evidence_work import check_evidence_cancellation
    from survng.app.motion_incidents import _RefinementJob
    from survng.app.motion_pipeline.decision_handler import MotionDecisionOutcome

    store, event, service, provider, publish, _ = make_pipeline(tmp_path)
    due(store)
    deadline = store.cover_requirement(event['id'])['deadline_epoch']
    entered = threading.Event()
    errors = []

    def waiting_for_media(*args):
        entered.set()
        for _ in range(200):
            check_evidence_cancellation()
            threading.Event().wait(.01)
        pytest.fail('cover work did not yield to security admission')

    provider.side_effect = waiting_for_media
    def recover():
        try:
            service._run_cover_requirement()
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=recover)
    worker.start()
    assert entered.wait(2)
    probe = _RefinementJob('motion', '', datetime.now(timezone.utc), {}, None,
                           True, True, None, None, MotionDecisionOutcome(None, '', False))
    service._queue_refinement(probe)
    worker.join(2)
    assert not worker.is_alive() and not errors
    requirement = store.cover_requirement(event['id'])
    assert requirement['state'] == 'pending'
    assert requirement['attempts'] == 0 and requirement['lease_owner'] == ''
    assert requirement['deadline_epoch'] == deadline
    assert store.get(event['id'])['snapshot_path'] == 'live.webp'
    publish.assert_not_called()
    claimed = store.claim_detection_job('gate', lease_owner='security')
    assert claimed['id'] == probe.job_id('gate')
    # A later retry is a new cancellation scope and can still complete.
    service._security_work_pending.clear()
    provider.side_effect = None
    due(store)
    assert service._run_cover_requirement()
    assert store.cover_requirement(event['id'])['state'] == 'satisfied'


def test_due_durable_security_work_preempts_without_an_in_memory_wakeup(tmp_path):
    import time
    from survng.app.evidence_work import check_evidence_cancellation
    store, event, service, provider, *_ = make_pipeline(tmp_path)
    due(store)
    def media_wait(*args):
        store.enqueue_detection_job(job_id='retry', camera_id='gate', dedupe_key='retry',
                                    payload={'existing_event_id': None})
        end = time.monotonic() + 2
        while time.monotonic() < end:
            check_evidence_cancellation()
            time.sleep(.01)
        pytest.fail('durable security work was not noticed')
    provider.side_effect = media_wait
    assert service._run_cover_requirement()
    assert store.cover_requirement(event['id'])['attempts'] == 0
    assert store.claim_detection_job('gate', lease_owner='security')['id'] == 'retry'


def test_completed_job_duplicate_does_not_preempt_cover_recovery(tmp_path):
    from survng.app.motion_incidents import _RefinementJob
    from survng.app.motion_pipeline.decision_handler import MotionDecisionOutcome
    store, event, service, provider, *_ = make_pipeline(tmp_path)
    job = _RefinementJob('motion', '', datetime.now(timezone.utc), {}, event['id'],
                         True, True, None, None, MotionDecisionOutcome(None, '', False))
    assert service._queue_refinement(job) == 'queued'
    claimed = store.claim_detection_job('gate', lease_owner='security')
    assert store.complete_detection_job(claimed['id'], event['id'], lease_owner='security')
    service._security_work_pending.clear()
    result = provider.return_value
    def duplicate(*args):
        assert service._queue_refinement(job) == 'coalesced'
        assert not store.has_due_detection_job('gate')
        return result
    provider.side_effect = duplicate
    due(store)
    assert service._run_cover_requirement()
    assert store.cover_requirement(event['id'])['state'] == 'satisfied'
    assert not service._security_work_pending.is_set()


def test_transient_cover_write_failure_retries_without_readmitting_incident(tmp_path):
    store, event, service, provider, publish, tracking = make_pipeline(tmp_path)
    promoter = service.decision_processor.refinement_cover_promoter
    service.decision_processor.refinement_cover_promoter = Mock(side_effect=OSError("storage busy"))
    due(store)
    assert service._run_cover_requirement()
    requirement = store.cover_requirement(event["id"])
    assert requirement["state"] == "pending" and requirement["attempts"] == 1
    assert requirement["reason"] == "refinement_cover_promotion_failed"
    assert store.get(event["id"])["snapshot_path"] == "live.webp"
    service.decision_processor.refinement_cover_promoter = promoter
    due(store)
    assert service._run_cover_requirement()
    assert store.cover_requirement(event["id"])["state"] == "satisfied"
    assert len(store.recent()) == 1
    assert all(call.args[0] == "incident_update" for call in publish.call_args_list)
    tracking.assert_not_called()


def test_cover_recovery_does_not_extend_security_inference_freshness(tmp_path):
    from survng.app.event_store.jobs import DETECTION_EVENT_JOB_MAXIMUM_AGE_SECONDS
    assert DETECTION_EVENT_JOB_MAXIMUM_AGE_SECONDS == 60.0
    store, event, *_ = make_pipeline(tmp_path)
    requirement = store.cover_requirement(event["id"])
    assert requirement["deadline_epoch"] - requirement["created_at"] == 300.0

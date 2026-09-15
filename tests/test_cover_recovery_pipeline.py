from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app.config import MainEvidenceConfig
from survng.app.events import EventStore
from survng.app.main_evidence_lifecycle import MainEvidenceFleet
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


def test_insufficient_association_requests_one_bounded_continuation(tmp_path, monkeypatch):
    store, event, service, provider, publish, tracking = make_pipeline(tmp_path)
    provider.return_value.objects[0]["temporal_last_observation_offset_seconds"] = 0.5
    monkeypatch.setattr("survng.app.motion_pipeline.decision_handler.motion_correlated_objects",
                        lambda *args, **kwargs: ([], {"reason": "insufficient_span"}))
    result = service.decision_processor.refine(
        "motion", "", datetime.fromtimestamp(1000, timezone.utc), {},
        existing_event_id=event["id"], require_eligible_object=True,
        require_motion_correlation=True,
    )
    assert provider.call_count == 2
    request = provider.call_args.args[1]
    assert request["association_continuation"] is True
    assert request["evidence_sampling"]["minimum_last_offset_seconds"] == 4
    assert 0 < request["evidence_sampling"]["timeout_seconds"] <= 24
    assert result.object_detected is False
    assert not any(item.get("status") == "face_evidence_pending"
                   for item in json.loads(store.get(event["id"])["objects_json"]))


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


class FakeProvider:
    def __init__(self, camera_id, source, budget, **kwargs):
        self.running = False
        self.max_bytes = kwargs["max_bytes"]
        self.starts = self.stops = 0

    def start(self):
        self.running = True
        self.starts += 1

    def stop(self):
        self.running = False
        self.stops += 1

    def refresh_status(self):
        pass


def test_buffer_fleet_defaults_off_and_obeys_aggregate_quota_and_camera_state():
    cameras = [SimpleNamespace(id=name, source_url=lambda _: "rtsp://example/main") for name in ("a", "b")]
    enabled = {"a": True, "b": False}
    fleet = MainEvidenceFleet(MainEvidenceConfig(), cameras, None, enabled.get, provider_factory=FakeProvider)
    assert fleet.providers == {}
    fleet = MainEvidenceFleet(MainEvidenceConfig(enabled=True, camera_ids=["a", "b"],
                              total_max_bytes=8 * 1024**2), cameras, None, enabled.get,
                              provider_factory=FakeProvider)
    assert sum(p.max_bytes for p in fleet.providers.values()) <= 8 * 1024**2
    fleet.reconcile()
    assert fleet.providers["a"].running and not fleet.providers["b"].running
    enabled["a"], enabled["b"] = False, True
    fleet.reconcile()
    assert not fleet.providers["a"].running and fleet.providers["b"].running
    fleet.stop()
    assert not any(p.running for p in fleet.providers.values())


def test_buffer_config_rejects_unbounded_or_zero_clock_allowance():
    with pytest.raises(ValueError):
        MainEvidenceConfig(max_timestamp_uncertainty_seconds=0)
    with pytest.raises(ValueError):
        MainEvidenceConfig(camera_max_bytes=0)


def test_camera_buffer_switch_reconfigures_without_restarting_other_collectors():
    from survng.app.config import CameraConfig
    config = MainEvidenceConfig(enabled=True, camera_ids=["a"])
    cameras = [CameraConfig(id=name, name=name, stream_url=f"rtsp://{name}/main") for name in ("a", "b")]
    fleet = MainEvidenceFleet(config, cameras, None, lambda _: True, provider_factory=FakeProvider)
    fleet.reconcile()
    original = fleet.providers["a"]
    cameras[1].main_evidence_enabled = True
    fleet.reconfigure(config, cameras)
    fleet.reconcile()
    added = fleet.providers["b"]
    assert added.running
    assert fleet.providers["a"] is original and original.starts == 1 and original.stops == 0
    cameras[1].main_evidence_enabled = False
    fleet.reconfigure(config, cameras)
    assert "b" not in fleet.providers and not added.running
    assert original.running and original.stops == 0
    # An explicit per-camera off overrides membership in the legacy global list.
    cameras[0].main_evidence_enabled = False
    fleet.reconfigure(config, cameras)
    assert fleet.providers == {} and not original.running
    fleet.stop()


def test_existing_detector_resolves_newly_enabled_buffer():
    from survng.app.config import CameraConfig, DetectorConfig
    from survng.app.main_evidence import MainEvidenceBatch
    from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetectorFactory
    import time
    providers = {}
    camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://gate/main")
    factory = RecordedMotionObjectDetectorFactory(
        SimpleNamespace(config=DetectorConfig()), Mock(),
        main_evidence_provider=lambda item: providers.get(item.id),
    )
    detector = factory.create(camera, lambda: None)
    def attempt():
        return detector._detect_buffered(event_epoch=100, stages=((0., .5),),
                                        deadline=time.monotonic() + 1, timing={},
                                        workflow_started=time.monotonic(), minimum_last_offset_seconds=None)
    assert attempt() is None
    provider = Mock()
    provider.frames_at.return_value = MainEvidenceBatch("pending")
    providers["gate"] = provider
    assert attempt() is None
    provider.frames_at.assert_called_once()
    providers.clear()
    assert attempt() is None
    assert provider.frames_at.call_count == 1


def test_observability_exposes_only_allowlisted_buffer_state():
    from survng.app.config import AppConfig
    from survng.app.local_observability import _detector_snapshot
    result = _detector_snapshot(AppConfig(), {"main_evidence": {"gate": {
        "running": True, "bytes": 123, "state": "ready",
        "source_url": "rtsp://secret:password@example/main", "raw_error": "secret",
        "outcomes": {"ready": 2, "arbitrary_secret": 42},
    }}})["main_evidence"]["gate"]
    assert result["bytes"] == 123 and result["outcomes"]["ready"] == 2
    assert "secret" not in json.dumps(result)


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
    assert not any(item.get('status') == 'face_evidence_pending'
                   for item in json.loads(store.get(event['id'])['objects_json']))


def test_face_deadline_survives_crash_after_cover_commit(tmp_path):
    store, event, service, provider, _, _ = make_pipeline(tmp_path)
    # Crash boundary: cover committed, face persistence/settlement did not run.
    store.settle_cover_requirement(event['id'], state='satisfied', reason='cover_committed')
    with store._connect() as conn:
        conn.execute('update event_face_requirements set deadline_epoch=0')
    recovered = EventStore(tmp_path)
    assert recovered.expire_face_requirements() == 1
    row = recovered.get(event['id'])
    assert row['cover_requirement']['state'] == 'satisfied'
    assert row['evidence_revision'] == 1
    statuses = json.loads(row['objects_json'])
    terminal = next(item for item in statuses if item.get('status') == 'face_evidence_complete')
    assert terminal['face_evidence']['state'] == 'exhausted'
    assert recovered.expire_face_requirements() == 0
    assert any(row['kind'] == 'incident_metadata_updated' for row in recovered.pending_evidence_updates())


def test_face_completion_is_independent_of_failed_cover(tmp_path):
    store, event, service, *_ = make_pipeline(tmp_path, objects=[])
    due(store)
    assert service._run_cover_requirement()
    assert store.cover_requirement(event['id'])['state'] == 'pending'
    assert not any(item.get('status') == 'face_evidence_pending'
                   for item in json.loads(store.get(event['id'])['objects_json']))
    with store._connect() as conn:
        assert conn.execute('select state from event_face_requirements').fetchone()[0] == 'not_applicable'


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


def test_face_requirement_migration_is_idempotent_and_terminal_state_cannot_reopen(tmp_path):
    store, event, *_ = make_pipeline(tmp_path)
    original = store.get(event['id'])['objects_json']
    with store._connect() as conn:
        conn.execute('delete from event_face_requirements')
        conn.execute("delete from survng_metadata where key='face_requirements_migrated'")
    recovered = EventStore(tmp_path)
    with recovered._connect() as conn:
        deadline = conn.execute('select deadline_epoch from event_face_requirements').fetchone()[0]
    restarted = EventStore(tmp_path)
    with restarted._connect() as conn:
        assert conn.execute('select deadline_epoch from event_face_requirements').fetchone()[0] == deadline
    restarted.settle_face_evidence(event['id'], state='complete', reason='candidates_processed')
    restarted.update_objects(event['id'], original)
    objects = json.loads(restarted.get(event['id'])['objects_json'])
    assert not any(item.get('status') == 'face_evidence_pending' for item in objects)
    terminal = next(item for item in objects if item.get('status') == 'face_evidence_complete')
    assert terminal['face_evidence'] == {'state': 'complete', 'reason': 'candidates_processed'}


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

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from survng.app.config import MotionQualificationConfig
from survng.app.ema_v2 import CameraNotice, EmaQualified
from survng.app.events import EventStore
from survng.app.motion import MotionQualificationResult
from survng.app.motion_decisions import MotionDecisionOrchestrator
from survng.app.motion_events import MotionEventCoordinator, MotionTrigger, MotionTriggerBatch
from survng.app.motion_ingress import MotionEventIngressService
from survng.app.motion_trigger_authority import merge_trigger_authority


@pytest.mark.parametrize("empty_first", [False, True])
def test_empty_semantic_snapshot_survives_merge_into_legacy_none(empty_first: bool) -> None:
    empty = {"camera_semantics": {"reports": []}}
    legacy = {"camera_semantics": None}
    payload, authority = (empty, legacy) if empty_first else (legacy, empty)
    assert merge_trigger_authority(payload, authority)["camera_semantics"] == {"reports": []}


@pytest.mark.parametrize("payload", [{}, {"camera_semantics": None}])
def test_legacy_absence_does_not_become_a_parsed_semantic_snapshot(payload: dict) -> None:
    merged = merge_trigger_authority(payload, {})
    assert merged.get("camera_semantics") is None
    assert ("camera_semantics" in merged) == ("camera_semantics" in payload)


def _coordinator(store: EventStore) -> MotionEventCoordinator:
    return MotionEventCoordinator(
        camera_id="gate", queue_size=4, retry_limit=2, durable_store=store
    )

def _reserve_ema(events: MotionEventCoordinator) -> MotionTrigger:
    now = time.time()
    observed = time.monotonic()
    result = MotionQualificationResult(
        True, 0.9, 0.5, "ema_v2_qualified", 4, {"ema_v2": True}
    )
    decision = events.episode_controller.observe_ema(
        EmaQualified("gate", now, observed, result, 0.6, 3, 3, now - 1),
        generation=0,
    )
    assert decision.intent is not None
    return MotionTrigger(
        topic="adaptive/visual_backup", message="EMA",
        event_at=datetime.fromtimestamp(now, timezone.utc), received_at=now,
        prequalified=result, detection_intent_id=decision.intent.intent_id,
        episode_id=decision.intent.episode_id,
    )

def _enqueue(events: MotionEventCoordinator, trigger: MotionTrigger) -> None:
    assert events.enqueue(trigger)
    events.episode_controller.acknowledge_admission(
        trigger.detection_intent_id, admitted=True, occurred_monotonic=time.monotonic()
    )

def _qualification(mode: str = "camera_rescue") -> Mock:
    qualification = Mock()
    qualification.settings.return_value = (mode, "balanced", 320)
    qualification.rescue_settings.return_value = (False, 0.0)
    qualification.suppression_verification_rate.return_value = 0.0
    qualification.with_pipeline_telemetry.side_effect = lambda value: value
    return qualification


def _camera_ingress(
    events: MotionEventCoordinator, *, mode: str = "camera_rescue",
    labels: tuple[str, ...] = ("car", "truck"), topic: str = "RuleEngine/VehicleDetect",
) -> None:
    state = Mock()
    state.begin_ingress.return_value = 0
    ingress = MotionEventIngressService(
        camera_id="gate", events=events, qualification=_qualification(mode), state=state,
        model_labels=lambda: list(labels),
    )
    ingress.handle(topic, '<SimpleItem Name="State" Value="true"/>')


def _process(events: MotionEventCoordinator, trigger: MotionTrigger) -> tuple[dict, dict]:
    incidents = Mock()
    incidents.process.return_value = Mock(as_dict=Mock(return_value={
        "event_id": 9, "object_detected": False, "snapshot_path": "",
    }))
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate", events=events, audit_recorder=Mock(),
        config=MotionQualificationConfig(), qualification=_qualification(),
        incidents=incidents, media=Mock(), analysis=Mock(), state=Mock(),
        # Reloading the model must not reinterpret a persisted camera claim.
        model_labels=lambda: ["bus"],
    )
    orchestrator._process_batch(MotionTriggerBatch((trigger,)), threading.Event())
    call = incidents.process.call_args
    return call.args[3], call.kwargs


@pytest.mark.parametrize("merge_at", ["before_enqueue", "before_claim", "after_claim"])
def test_merged_camera_authority_survives_reconstruction(tmp_path: Path, merge_at: str) -> None:
    store = EventStore(tmp_path)
    events = _coordinator(store)
    trigger = _reserve_ema(events)
    if merge_at == "before_enqueue":
        _camera_ingress(events)
    _enqueue(events, trigger)
    if merge_at == "before_claim":
        _camera_ingress(events)
    if merge_at == "after_claim":
        claimed = events.next_trigger(timeout=0.1)
        assert claimed is not None
        assert claimed.admitted_sources == ("ema",)
        _camera_ingress(events)
        events.release_deliveries(MotionTriggerBatch((claimed,)))

    restored = _coordinator(EventStore(tmp_path))
    restored.episode_controller.start_generation(5)
    recovered = restored.next_trigger(timeout=0.1)
    assert recovered is not None
    assert restored.episode_controller.intent(recovered.detection_intent_id) is None
    qualification, kwargs = _process(restored, recovered)

    assert qualification["trigger_source"] == "camera"
    assert qualification["reason"] == "camera_primary_fast_path"
    assert kwargs["require_eligible_object"] is False
    assert kwargs["require_motion_correlation"] is False
    reports = qualification["camera_semantics"]["reports"]
    assert len(reports) == 1
    assert reports[0]["candidate_model_classes"] == ["car", "truck"]


def test_decision_refreshes_camera_merge_after_claim(tmp_path: Path) -> None:
    events = _coordinator(EventStore(tmp_path))
    trigger = _reserve_ema(events)
    _enqueue(events, trigger)
    claimed = events.next_trigger(timeout=0.1)
    assert claimed is not None
    _camera_ingress(events)
    # Exercise the durable refresh rather than the controller shortcut.
    events.episode_controller.start_generation(1)
    qualification, kwargs = _process(events, claimed)
    assert qualification["trigger_source"] == "camera"
    assert kwargs["require_eligible_object"] is False


@pytest.mark.parametrize("labels", [(), ("car", "truck")])
def test_frozen_reports_match_with_or_without_runtime_controller(
    tmp_path: Path, labels: tuple[str, ...]
) -> None:
    snapshots = []
    for restart in (False, True):
        directory = tmp_path / str(restart)
        store = EventStore(directory)
        events = _coordinator(store)
        trigger = _reserve_ema(events)
        _enqueue(events, trigger)
        claimed = events.next_trigger(timeout=0.1)
        assert claimed is not None
        _camera_ingress(events, labels=labels)
        _camera_ingress(events, labels=labels, topic="RuleEngine/PeopleDetect")
        if restart:
            events.release_deliveries(MotionTriggerBatch((claimed,)))
            events = _coordinator(EventStore(directory))
            claimed = events.next_trigger(timeout=0.1)
            assert claimed is not None
        qualification, kwargs = _process(events, claimed)
        snapshots.append((
            qualification["camera_semantics"], qualification["trigger_source"],
            kwargs["require_eligible_object"], kwargs["require_motion_correlation"],
        ))
    assert snapshots[0] == snapshots[1]
    reports = snapshots[0][0]["reports"]
    assert {report["category"] for report in reports} == {"vehicle", "person"}
    vehicle = next(report for report in reports if report["category"] == "vehicle")
    assert vehicle["candidate_model_classes"] == list(labels)


def test_camera_merge_serializes_with_initial_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EventStore(tmp_path)
    events = _coordinator(store)
    trigger = _reserve_ema(events)
    write_started = threading.Event()
    finish_write = threading.Event()
    merge_started = threading.Event()
    merge_finished = threading.Event()
    original_enqueue = store.enqueue_motion_trigger

    def blocked_enqueue(**kwargs: object) -> bool:
        write_started.set()
        assert finish_write.wait(timeout=5)
        return original_enqueue(**kwargs)

    def merge() -> None:
        merge_started.set()
        _camera_ingress(events)
        merge_finished.set()

    monkeypatch.setattr(store, "enqueue_motion_trigger", blocked_enqueue)
    with ThreadPoolExecutor(max_workers=2) as pool:
        enqueuing = pool.submit(_enqueue, events, trigger)
        assert write_started.wait(timeout=5)
        merging = pool.submit(merge)
        try:
            assert merge_started.wait(timeout=5)
            assert not merge_finished.wait(timeout=0.05)
        finally:
            finish_write.set()
        enqueuing.result(timeout=5)
        merging.result(timeout=5)
    assert store.motion_trigger_authority(
        camera_id="gate", job_id=trigger.delivery_job_id
    )["admitted_sources"] == ["camera", "ema"]


@pytest.mark.parametrize("order", ["merge_first", "checkpoint_first", "concurrent"])
def test_stale_retry_checkpoint_cannot_erase_camera_merge(
    tmp_path: Path, order: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EventStore(tmp_path)
    events = _coordinator(store)
    trigger = _reserve_ema(events)
    _enqueue(events, trigger)
    claimed = events.next_trigger(timeout=0.1)
    assert claimed is not None
    events.episode_controller.mark_running(
        trigger.detection_intent_id, occurred_monotonic=time.monotonic()
    )
    stale_payload = claimed.durable_payload()
    stale_payload["retry_diagnostics"] = {"windows_evaluated": 2}
    # Separate store instances exercise SQLite transaction ordering, not only
    # the process-local jobs lock.
    checkpoint_store = EventStore(tmp_path)

    def checkpoint() -> None:
        assert checkpoint_store.fail_motion_trigger(
            claimed.delivery_job_id, "retry test", payload=stale_payload
        ) is True

    if order == "concurrent":
        barrier = threading.Barrier(2)

        def merge() -> None:
            barrier.wait(timeout=5)
            _camera_ingress(events)

        def concurrent_checkpoint() -> None:
            barrier.wait(timeout=5)
            checkpoint()

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(merge), pool.submit(concurrent_checkpoint)]
            for future in futures:
                future.result(timeout=5)
    elif order == "merge_first":
        _camera_ingress(events)
        checkpoint()
    else:
        checkpoint()
        _camera_ingress(events)

    # Advance past backoff without editing the persisted row or sleeping.
    replay_epoch = time.time() + 10
    monkeypatch.setattr("survng.app.event_store.jobs.time.time", lambda: replay_epoch)
    restored = _coordinator(EventStore(tmp_path))
    recovered = restored.next_trigger(timeout=0.1)
    assert recovered is not None
    assert recovered.retry_diagnostics == {"windows_evaluated": 2}
    qualification, kwargs = _process(restored, recovered)
    assert qualification["trigger_source"] == "camera"
    assert kwargs["require_motion_correlation"] is False
    assert qualification["camera_semantics"]["reports"][0]["category"] == "vehicle"


def test_new_generation_and_adaptive_mode_do_not_merge_camera_authority(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    events = _coordinator(store)
    trigger = _reserve_ema(events)
    _enqueue(events, trigger)
    _camera_ingress(events, mode="adaptive")
    assert store.motion_trigger_authority(
        camera_id="gate", job_id=trigger.delivery_job_id
    )["admitted_sources"] == ["ema"]

    events.episode_controller.start_generation(1)
    _camera_ingress(events)  # Its old generation is no longer admitted.
    events.observe_camera(
        CameraNotice("gate", time.time(), time.monotonic(), "manual", manual=True),
        generation=1,
    )
    store.merge_motion_trigger_authority(
        camera_id="gate", job_id=trigger.delivery_job_id, lifecycle_generation=1,
        authority={"admitted_sources": ["camera"]},
    )
    assert store.motion_trigger_authority(
        camera_id="gate", job_id=trigger.delivery_job_id
    )["admitted_sources"] == ["ema"]


def test_completed_delivery_is_not_resurrected_by_camera_merge(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    events = _coordinator(store)
    trigger = _reserve_ema(events)
    _enqueue(events, trigger)
    claimed = events.next_trigger(timeout=0.1)
    assert claimed is not None
    events.complete_deliveries(MotionTriggerBatch((claimed,)))
    _camera_ingress(events)
    assert store.motion_trigger_status("gate") == {}

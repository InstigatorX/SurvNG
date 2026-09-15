from __future__ import annotations

import json
import sqlite3

import pytest

from survng.app.events import EventStore
from survng.app.event_store.store import EventSnapshotChangedError


def provisional(**extra):
    return {"label": "person", "confidence": .8, "incident_eligible": True,
            "provisional_detection": True, "frame_source": "live_fast_path",
            "frame_captured_at_epoch": 1000., "detection_frame_width": 640,
            "detection_frame_height": 360,
            "box": {"x1": 100, "y1": 100, "x2": 140, "y2": 200}, **extra}


def add(store, **extra):
    return store.add_event(camera_id="gate", kind="motion", snapshot_path="live.webp",
                           objects_json=json.dumps([provisional(),
                               {"status": "motion_qualification", "motion_qualification": {"accepted": True}}]),
                           **extra)


def promote(store, event_id, **extra):
    candidate = {"label": "person", "confidence": .9, "temporal_consensus": True,
                 "frame_source": "recorded_main", "box": {"x1": 300, "y1": 300, "x2": 500, "y2": 800}}
    kwargs = dict(snapshot_path="main.webp", recording_path="main.mp4", captured_at=1002.,
                  frame_width=1920, frame_height=1080, cover_objects=[candidate],
                  source="recorded_main", timestamp_exact=True)
    kwargs.update(extra)
    return store.promote_refinement_cover(event_id, **kwargs)


def test_admission_atomically_owns_recoverable_requirement(tmp_path):
    store = EventStore(tmp_path)
    event = add(store, detection_intent_id="intent-1")
    recovered = EventStore(tmp_path)
    requirement = recovered.cover_requirement(event["id"])
    assert requirement["state"] == "pending"
    assert requirement["deadline_epoch"] - requirement["created_at"] == 300
    assert requirement["payload"]["qualification"] == {"accepted": True}
    assert requirement["payload"]["objects"][0]["box"] == provisional()["box"]
    assert {row["kind"] for row in recovered.pending_evidence_updates()} == {"cover_required", "evidence_updated"}
    assert recovered.recent_compact()[0]["evidence_revision"] == 1
    # Replayed admission cannot reset the obligation or enqueue a second dispatch.
    add(recovered, detection_intent_id="intent-1", created_at=event["created_at"])
    assert recovered.cover_requirement(event["id"]) == requirement
    assert len(recovered.pending_evidence_updates()) == 2


def test_admission_rolls_back_event_if_obligation_outbox_fails(tmp_path, monkeypatch):
    store = EventStore(tmp_path)
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("simulated crash before commit")
    monkeypatch.setattr(store, "_evidence_outbox", fail)
    with pytest.raises(sqlite3.OperationalError):
        add(store)
    assert store.recent() == []
    assert store.pending_evidence_updates() == []
    with store._connect() as conn:
        assert conn.execute("select count(*) from event_cover_requirements").fetchone()[0] == 0


def test_legacy_migration_does_not_schedule_historical_evidence(tmp_path):
    # A pre-revision database with old provisional evidence remains historical.
    with sqlite3.connect(tmp_path / "survng.sqlite3") as conn:
        conn.execute("create table events(id integer primary key, camera_id text, kind text, topic text, message text, "
                     "snapshot_path text, recording_path text, objects_json text, created_at text)")
        conn.execute("insert into events values(1,'gate','motion','','','old.webp','',?,'2026-09-01T00:00:00+00:00')",
                     (json.dumps([provisional()]),))
    store = EventStore(tmp_path)
    assert store.get(1)["evidence_revision"] == 0
    assert store.cover_requirement(1) is None
    assert store.pending_evidence_updates() == []


def test_cover_commit_revision_cas_and_noop_are_atomic(tmp_path):
    store = EventStore(tmp_path)
    event = add(store)
    updated = store.update_objects(event["id"], json.dumps([provisional(confidence=.85)]), expected_revision=1)
    assert updated["evidence_revision"] == 2
    store.update_objects(event["id"], json.dumps([provisional(confidence=.85)], indent=2), expected_revision=2)
    assert store.get(event["id"])["evidence_revision"] == 2
    with pytest.raises(EventSnapshotChangedError):
        promote(EventStore(tmp_path), event["id"], expected_revision=1)
    assert store.get(event["id"])["snapshot_path"] == "live.webp"
    assert store.cover_requirement(event["id"])["state"] == "pending"
    assert len(store.pending_evidence_updates()) == 3


def test_cover_promotion_preserves_confidence_and_clears_stale_mask(tmp_path):
    store = EventStore(tmp_path)
    original = provisional(mask_polygon=[[100, 100], [140, 100], [140, 200]])
    event = store.add_event(camera_id="gate", kind="motion", snapshot_path="live.webp", objects_json=json.dumps([original]))
    updated = promote(store, event["id"], expected_revision=1)
    person = json.loads(updated["objects_json"])[0]
    assert "mask_polygon" not in person
    assert person["box"] != original["box"]
    assert person["confidence"] == original["confidence"]
    assert updated["evidence_revision"] == 2
    assert store.cover_requirement(event["id"])["state"] == "satisfied"


@pytest.mark.parametrize("visible", [False, True])
def test_off_frame_geometry_and_temporal_identity_ambiguity_cannot_promote(tmp_path, visible):
    store = EventStore(tmp_path)
    event = add(store)
    candidate = {"label": "person", "temporal_consensus": True, "snapshot_visible": False,
                 "box": {"x1": 300, "y1": 300, "x2": 500, "y2": 800}}
    candidates = [candidate]
    if visible:
        candidates.append({**candidate, "snapshot_visible": True})
    assert promote(store, event["id"], cover_objects=candidates) is None
    assert store.get(event["id"])["evidence_revision"] == 1
    assert store.cover_requirement(event["id"])["state"] == "pending"


def test_leased_attempt_recovers_and_concurrent_satisfaction_wins(tmp_path, monkeypatch):
    now = [1000.]
    monkeypatch.setattr("survng.app.event_store.evidence.time.time", lambda: now[0])
    store = EventStore(tmp_path)
    event = add(store)
    assert store.claim_cover_requirement("gate", lease_owner="a") is None
    now[0] += 16
    first = store.claim_cover_requirement("gate", lease_owner="a")
    assert first["attempts"] == 1
    assert first["latest_event"]["evidence_revision"] == 1
    assert store.claim_cover_requirement("gate", lease_owner="b") is None
    now[0] += 61
    recovered = EventStore(tmp_path).claim_cover_requirement("gate", lease_owner="b")
    assert recovered["attempts"] == 2
    assert not store.finish_cover_attempt(event["id"], lease_owner="a", reason="stale_owner")
    promote(store, event["id"], expected_revision=1)
    assert not store.finish_cover_attempt(event["id"], lease_owner="b", reason="late_failure")
    assert store.cover_requirement(event["id"])["state"] == "satisfied"


def test_attempts_and_deadline_are_bounded(tmp_path, monkeypatch):
    now = [1000.]
    monkeypatch.setattr("survng.app.event_store.evidence.time.time", lambda: now[0])
    store = EventStore(tmp_path)
    event = add(store)
    now[0] += 16
    for attempt in range(1, 4):
        assert store.claim_cover_requirement("gate", lease_owner="owner")["attempts"] == attempt
        store.finish_cover_attempt(event["id"], lease_owner="owner", reason="no_associated_candidate", retry_delay_seconds=0)
    assert store.cover_requirement(event["id"])["state"] == "exhausted"
    assert store.claim_cover_requirement("gate", lease_owner="owner") is None
    other = add(store)
    now[0] += 301
    assert store.claim_cover_requirement("gate", lease_owner="owner") is None
    assert store.cover_requirement(other["id"])["reason"] == "deadline_expired"


def test_outbox_ack_does_not_erase_later_revision(tmp_path):
    store = EventStore(tmp_path)
    event = add(store)
    initial = store.pending_evidence_updates()
    first = initial[0]
    store.acknowledge_evidence_update(initial[1]["id"])
    promote(store, event["id"])
    assert store.acknowledge_evidence_update(first["id"])
    assert not store.acknowledge_evidence_update(first["id"])
    pending = EventStore(tmp_path).pending_evidence_updates()
    assert {item["kind"] for item in pending} == {"evidence_updated", "cover_requirement_updated"}
    assert all(item["payload"]["evidence_revision"] == 2 for item in pending)


def test_manual_annotations_share_revision_commit(tmp_path):
    store = EventStore(tmp_path)
    event = add(store)
    updated = store.replace_detected_objects(event["id"], json.dumps([{"label": "car"}]),
                                              expected_snapshot_path="live.webp", expected_revision=1)
    assert updated["evidence_revision"] == 2
    with pytest.raises(EventSnapshotChangedError):
        store.replace_detected_objects(event["id"], "[]", expected_snapshot_path="live.webp", expected_revision=1)
    assert json.loads(store.get(event["id"])["objects_json"])[0]["label"] == "car"


def test_reclaimed_cover_lease_fences_old_worker_even_before_any_revision_change(tmp_path, monkeypatch):
    now = [1000.]
    monkeypatch.setattr("survng.app.event_store.evidence.time.time", lambda: now[0])
    store = EventStore(tmp_path)
    event = add(store)
    now[0] += 16
    store.claim_cover_requirement("gate", lease_owner="old")
    now[0] += 61
    store.claim_cover_requirement("gate", lease_owner="new")
    with pytest.raises(EventSnapshotChangedError):
        promote(store, event["id"], expected_revision=1, requirement_lease_owner="old")
    assert store.get(event["id"])["evidence_revision"] == 1
    assert promote(store, event["id"], expected_revision=1, requirement_lease_owner="new")


def test_cover_commit_rolls_back_if_outbox_write_fails(tmp_path, monkeypatch):
    store = EventStore(tmp_path)
    event = add(store)
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("simulated outbox failure")
    monkeypatch.setattr(store, "_evidence_outbox", fail)
    with pytest.raises(sqlite3.OperationalError):
        promote(store, event["id"])
    assert store.get(event["id"])["snapshot_path"] == "live.webp"
    assert store.get(event["id"])["evidence_revision"] == 1
    assert store.cover_requirement(event["id"])["state"] == "pending"


def test_attempt_diagnostics_are_bounded(tmp_path):
    store = EventStore(tmp_path)
    event = store.add_event(camera_id="gate", kind="motion", objects_json=json.dumps([
        provisional(), {"status": "face_evidence_pending"}]))
    for i in range(5):
        store.record_evidence_attempt(event["id"], {"attempt": i, "reason": "off_frame_candidate"})
    assert [item["attempt"] for item in store.evidence_attempts(event["id"])] == [2, 3, 4]
    assert store.get(event["id"])["evidence_revision"] == 1


def test_requirement_state_is_exposed_without_private_payload(tmp_path):
    store = EventStore(tmp_path)
    event = add(store)
    assert store.get(event["id"])["cover_requirement"]["state"] == "pending"
    store.settle_cover_requirement(event["id"], state="exhausted", reason="media_unavailable")
    visible = store.recent_compact()[0]["cover_requirement"]
    assert visible["reason"] == "media_unavailable"
    assert "payload" not in visible and "lease_owner" not in visible
    assert store.pending_evidence_updates()[-1]["kind"] == "cover_requirement_updated"


def test_tracking_progress_does_not_churn_visual_revision(tmp_path):
    store = EventStore(tmp_path)
    event = add(store)
    for observation_count in range(20):
        store.update_object_tracking(event["id"], {"state": "active", "observations": observation_count})
    assert store.get(event["id"])["evidence_revision"] == 1
    assert [row["kind"] for row in store.pending_evidence_updates()] == ["evidence_updated", "cover_required"]


def test_accepted_refinement_cannot_downgrade_an_already_improved_cover(tmp_path):
    store = EventStore(tmp_path)
    event = add(store)
    updated = promote(store, event["id"])
    # Simulate a restarted/refreshed inference with a CURRENT revision but
    # smaller image: CAS alone cannot detect this semantic downgrade.
    smaller = provisional(provisional_detection=False, frame_source="recorded_main")
    result = store.refine_event_evidence(event["id"], snapshot_path="smaller.webp",
                                         recording_path="main.mp4", objects_json=json.dumps([smaller]),
                                         expected_revision=updated["evidence_revision"])
    assert result is None
    assert store.get(event["id"])["snapshot_path"] == "main.webp"
    assert store.get(event["id"])["evidence_revision"] == updated["evidence_revision"]
    assert store.cover_requirement(event["id"])["state"] == "satisfied"


def test_metadata_outbox_ack_cannot_erase_concurrent_same_revision_state(tmp_path):
    store = EventStore(tmp_path)
    event = add(store)
    first_state = [{"status": "face_evidence_pending"}, provisional()]
    store.update_objects(event["id"], json.dumps(first_state))
    first = store.pending_evidence_updates()[-1]
    assert first["kind"] == "incident_metadata_updated"
    store.update_objects(event["id"], json.dumps([provisional(), {"status": "refinement_complete"}]))
    store.acknowledge_evidence_update(first["id"])
    newer = store.pending_evidence_updates(after_id=first["id"])
    assert len(newer) == 1
    assert newer[0]["kind"] == "incident_metadata_updated"
    assert newer[0]["evidence_revision"] == first["evidence_revision"]


def test_concurrent_cover_writers_have_one_revision_winner(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    first_store = EventStore(tmp_path)
    second_store = EventStore(tmp_path)
    event = add(first_store)
    barrier = threading.Barrier(2)
    def write(store, path):
        barrier.wait(timeout=2)
        try:
            return promote(store, event["id"], snapshot_path=path, expected_revision=1)["snapshot_path"]
        except EventSnapshotChangedError:
            return "stale"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(write, first_store, "main-a.webp")
        second = pool.submit(write, second_store, "main-b.webp")
        results = [first.result(timeout=5), second.result(timeout=5)]
    assert results.count("stale") == 1
    assert first_store.get(event["id"])["evidence_revision"] == 2
    assert len([row for row in first_store.pending_evidence_updates()
                if row["kind"] == "evidence_updated" and row["evidence_revision"] == 2]) == 1


def test_publication_checkpoint_only_marks_observed_ids(tmp_path):
    store = EventStore(tmp_path)
    event = add(store)
    observed_ids = [row["id"] for row in store.pending_evidence_updates()]
    promote(store, event["id"])
    assert store.mark_evidence_publication(observed_ids)
    rows = EventStore(tmp_path).pending_evidence_updates()
    assert all(bool(row["publication_done"]) == (row["id"] in observed_ids) for row in rows)
    assert any(not row["publication_done"] for row in rows)


def test_global_expiry_handles_stopped_cameras_and_preserves_live_leases(tmp_path, monkeypatch):
    now = [1000.]
    monkeypatch.setattr("survng.app.event_store.evidence.time.time", lambda: now[0])
    store = EventStore(tmp_path)
    idle = add(store)
    leased = add(store)
    now[0] += 280
    with store._connect() as conn:
        conn.execute("update event_cover_requirements set available_at_epoch=2000 where event_id=?", (idle["id"],))
    assert store.claim_cover_requirement("gate", lease_owner="live")["event_id"] == leased["id"]
    now[0] += 21
    assert store.expire_cover_requirements() == 1
    assert store.cover_requirement(idle["id"])["state"] == "exhausted"
    assert store.cover_requirement(leased["id"])["state"] == "pending"
    now[0] += 40
    assert store.expire_cover_requirements() == 1
    assert store.cover_requirement(leased["id"])["state"] == "exhausted"
    assert store.expire_cover_requirements() == 0

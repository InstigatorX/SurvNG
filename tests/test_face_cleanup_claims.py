from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from survng.app.event_store import EventStore
from survng.app.face_candidates import FaceCandidate
from survng.app.faces import FaceStore
from survng.app.motion_pipeline import MotionDecisionHandler
from survng.app.motion_pipeline.object_detection import RecordedDetectionResult


STAMP = "2025-01-01T00:00:00+00:00"
BOX = {"x1": 1, "y1": 1, "x2": 40, "y2": 40}


def stores(root, shared_mutex=True):
    lock = threading.RLock()
    events = EventStore(root, database_write_lock=lock)
    faces = FaceStore(root, start_recognition=False,
                      database_write_lock=lock if shared_mutex else threading.RLock())
    return events, faces


def snapshot(root, name):
    path = root / "snapshots" / "gate" / f"{name}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"snapshot")
    return path


def audit(events, path, **kwargs):
    return events.add_motion_audit(
        camera_id="gate", snapshot_path=str(path), created_at=STAMP,
        mode="ema", sensitivity="medium", score=1, threshold=0.5,
        reason="motion", object_detected=True, trigger_count=1, features={},
        **kwargs,
    )


def face(faces, path, event_id=1):
    with faces._connect() as connection:
        return connection.execute(
            """insert into face_observations
                (event_id, object_index, camera_id, snapshot_path, box_json, observed_at, created_at)
                values (?, 0, 'gate', ?, '{}', ?, ?)""",
            (event_id, str(path), STAMP, STAMP),
        ).lastrowid


@pytest.mark.parametrize("shared_mutex", [True, False])
def test_blocked_cleanup_allows_writes_but_refuses_claimed_path_adoption(tmp_path, shared_mutex):
    # Separate mutexes exercise the durable SQLite claim, rather than relying
    # on the in-process lock shared by AppManager's stores.
    events, faces = stores(tmp_path, shared_mutex)
    crop = snapshot(tmp_path, "orphan")
    retained = snapshot(tmp_path, "retained")
    unrelated = snapshot(tmp_path, "unrelated")
    existing = events.add_event(
        camera_id="gate", kind="object", snapshot_path=str(retained),
        objects_json=json.dumps([{"label": "face", "track_id": "track", "box": BOX}]),
    )
    old_audit = audit(events, retained, decision_id="retained-audit")
    entered, release = threading.Event(), threading.Event()
    original_unlink = Path.unlink

    def blocked_unlink(path, *args, **kwargs):
        if path == crop:
            entered.set()
            assert release.wait(10), "test did not release cleanup"
        return original_unlink(path, *args, **kwargs)

    def write_during_cleanup():
        event = events.add_event(camera_id="gate", kind="object", snapshot_path=str(unrelated))
        assert event["snapshot_path"] == unrelated.relative_to(tmp_path).as_posix()
        assert audit(events, unrelated)["snapshot_path"] == event["snapshot_path"]

        claimed_event = events.add_event(camera_id="gate", kind="object", snapshot_path=str(crop))
        assert claimed_event["snapshot_path"] == ""
        assert claimed_event["snapshot_size_bytes"] == 0
        assert audit(events, crop)["snapshot_path"] == ""
        assert audit(events, crop, decision_id="retained-audit")["snapshot_path"] == old_audit["snapshot_path"]

        assert faces.ingest_candidates(claimed_event["id"], "gate", STAMP, [{
            "snapshot_path": str(crop), "box": BOX, "track_id": "new-track", "rank": 1,
        }]) == 0
        assert faces.ingest_events([{
            "id": claimed_event["id"], "camera_id": "gate", "snapshot_path": str(crop),
            "objects_json": json.dumps([{"label": "face", "confidence": 0.9, "box": BOX}]),
        }]) == 0

        assert events.refine_event_evidence(
            existing["id"], snapshot_path=str(crop), recording_path="", objects_json="[]",
        ) is None
        assert events.promote_tracking_cover(
            existing["id"], snapshot_path=str(crop), captured_at=1, frame_width=100,
            frame_height=100, tracked_objects=[{"track_id": "track", "box": BOX}], cover_metrics={},
        ) is None
        assert events.promote_refinement_cover(
            existing["id"], snapshot_path=str(crop), recording_path="", captured_at=1,
            frame_width=100, frame_height=100,
            cover_objects=[{"label": "face", "box": BOX, "temporal_consensus": True}],
            source="test", timestamp_exact=True,
        ) is None
        current = events.get(existing["id"])
        assert current["snapshot_path"] == existing["snapshot_path"]
        assert current["objects_json"] == existing["objects_json"]

        with faces._connect() as connection:
            for table in ("events", "motion_audits", "face_observations"):
                assert connection.execute(
                    f"select count(*) from {table} where snapshot_path in (?, ?)",
                    (str(crop), crop.relative_to(tmp_path).as_posix()),
                ).fetchone()[0] == 0
            assert connection.execute("select count(*) from media_deletion_claims").fetchone()[0] == 1

    with patch.object(Path, "unlink", blocked_unlink), ThreadPoolExecutor(max_workers=2) as pool:
        cleanup = pool.submit(faces._delete_face_snapshots, [crop], "test")
        assert entered.wait(3)
        try:
            pool.submit(write_during_cleanup).result(timeout=3)
        finally:
            release.set()
        cleanup.result(timeout=3)
    assert not crop.exists()
    assert retained.exists()
    with faces._connect() as connection:
        assert connection.execute("select count(*) from media_deletion_claims").fetchone()[0] == 0


def test_cleanup_path_resolution_does_not_hold_database_writer(tmp_path):
    from survng.app.face_store import store as face_store_module

    events, faces = stores(tmp_path)
    crop = snapshot(tmp_path, "orphan")
    entered, release = threading.Event(), threading.Event()
    portable = face_store_module.portable_media_path

    def blocked_resolution(*args):
        entered.set()
        assert release.wait(10), "test did not release path resolution"
        return portable(*args)

    with patch.object(face_store_module, "portable_media_path", blocked_resolution), ThreadPoolExecutor(max_workers=2) as pool:
        cleanup = pool.submit(faces._delete_face_snapshots, [crop], "test")
        assert entered.wait(3)
        try:
            pool.submit(events.add_event, camera_id="gate", kind="object").result(timeout=3)
        finally:
            release.set()
        cleanup.result(timeout=3)


@pytest.mark.parametrize("owner", ["face", "event", "audit"])
@pytest.mark.parametrize("absolute", [False, True])
def test_cleanup_preserves_retained_path_aliases(tmp_path, owner, absolute):
    events, faces = stores(tmp_path)
    crop = snapshot(tmp_path, "retained")
    raw_path = str(crop) if absolute else crop.relative_to(tmp_path).as_posix()
    if owner == "face":
        face(faces, raw_path)
    else:
        if owner == "event":
            row = events.add_event(camera_id="gate", kind="object")
            table = "events"
        else:
            row = audit(events, "")
            table = "motion_audits"
        # Preserve a legacy absolute reference, bypassing insertion normalization.
        with events._connect() as connection:
            connection.execute(f"update {table} set snapshot_path = ? where id = ?", (raw_path, row["id"]))
    faces._delete_face_snapshots([crop], "test")
    assert crop.exists()
    with faces._connect() as connection:
        assert connection.execute("select count(*) from media_deletion_claims").fetchone()[0] == 0


@pytest.mark.parametrize("absolute", [False, True])
def test_cleanup_does_not_steal_existing_claim(tmp_path, absolute):
    _events, faces = stores(tmp_path)
    crop = snapshot(tmp_path, "claimed")
    raw_path = str(crop) if absolute else crop.relative_to(tmp_path).as_posix()
    with faces._connect() as connection:
        connection.execute(
            "insert into media_deletion_claims values (?, 'snapshot', ?)", (raw_path, STAMP),
        )
    faces._delete_face_snapshots([crop], "test")
    assert crop.exists()
    with faces._connect() as connection:
        assert tuple(connection.execute("select * from media_deletion_claims").fetchone()) == (raw_path, "snapshot", STAMP)


@pytest.mark.parametrize("owner", ["face", "claim"])
def test_cleanup_preserves_original_absolute_alias_after_portable_migration(tmp_path, owner):
    root = tmp_path / "current-mount"
    _events, faces = stores(root)
    snapshot(root, "retained")
    original = snapshot(tmp_path / "other-mount", "retained")
    if owner == "face":
        face(faces, str(original))
    else:
        with faces._connect() as connection:
            connection.execute(
                "insert into media_deletion_claims values (?, 'snapshot', ?)", (str(original), STAMP),
            )
    # portable_media_path can map an old absolute mount to the equivalent
    # current subtree; the original absolute reference must still protect it.
    faces._delete_face_snapshots([original], "test")
    assert original.exists()


def test_failed_unlink_releases_only_its_claim_and_can_retry(tmp_path):
    _events, faces = stores(tmp_path)
    crop = snapshot(tmp_path, "failed")
    with patch.object(Path, "unlink", side_effect=PermissionError("unavailable storage")):
        faces._delete_face_snapshots([crop], "test")
    assert crop.exists()
    with faces._connect() as connection:
        assert connection.execute("select count(*) from media_deletion_claims").fetchone()[0] == 0
    faces._delete_face_snapshots([crop], "retry")
    assert not crop.exists()


def test_cleanup_does_not_release_a_replaced_claim(tmp_path):
    _events, faces = stores(tmp_path)
    crop = snapshot(tmp_path, "reclaimed")

    def replace_claim(*args, **kwargs):
        with faces._connect() as connection:
            connection.execute("update media_deletion_claims set claimed_at = ?", (STAMP,))
        raise OSError("unlink failed")

    with patch.object(Path, "unlink", replace_claim):
        faces._delete_face_snapshots([crop], "test")
    assert crop.exists()
    with faces._connect() as connection:
        assert connection.execute("select claimed_at from media_deletion_claims").fetchone()[0] == STAMP


@pytest.mark.parametrize("deleted", [False, True])
def test_crop_claim_uses_existing_snapshot_restart_recovery(tmp_path, deleted):
    _events, faces = stores(tmp_path)
    crop = snapshot(tmp_path, "interrupted")
    # Model a process crash after the claim commits, before or after unlink.
    with faces._connect() as connection:
        connection.execute(
            "insert into media_deletion_claims values (?, 'snapshot', ?)",
            (crop.relative_to(tmp_path).as_posix(), STAMP),
        )
    if deleted:
        crop.unlink()
    recovered = EventStore(tmp_path)
    with recovered._connect() as connection:
        assert connection.execute("select count(*) from media_deletion_claims").fetchone()[0] == 0
    assert crop.exists() is not deleted


def claimed_snapshot_handler(tmp_path, events, faces):
    claimed = snapshot(tmp_path, "claimed-cover")
    independent = snapshot(tmp_path, "independent-face")
    with events._connect() as connection:
        connection.execute(
            "insert into media_deletion_claims values (?, 'snapshot', ?)",
            (claimed.relative_to(tmp_path).as_posix(), STAMP),
        )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = RecordedDetectionResult(
        frame=frame,
        objects=[{"label": "person", "confidence": 0.9, "incident_eligible": True, "box": BOX}],
        recording_path="recordings/gate/new.mp4", timings_ms={"detector_ms": 1.0},
        face_candidates=(FaceCandidate(
            track_id="face-track", rank=1, offset_seconds=1.0, frame=frame, box=BOX,
            confidence=0.9, quality_score=0.8, sharpness_score=0.8, exposure_score=0.8,
            edge_clearance_ratio=0.8, detection_source="test",
        ),),
    )
    published = []
    writer = Mock(side_effect=[str(claimed), str(independent)])
    sink = Mock(wraps=faces.ingest_candidates)
    handler = MotionDecisionHandler(
        camera_id="gate", events=events, detection_provider=lambda _at: result,
        snapshot_writer=writer, object_serializer=json.dumps,
        event_callback=lambda kind, payload: published.append((kind, payload)),
        face_candidate_sink=sink,
    )
    return handler, published, writer, sink, independent


def test_claimed_refinement_preserves_live_event_without_replacement_side_effects(tmp_path):
    events, faces = stores(tmp_path)
    old = snapshot(tmp_path, "old-cover")
    existing = events.add_event(
        camera_id="gate", kind="motion", snapshot_path=str(old),
        recording_path="recordings/gate/old.mp4",
        objects_json=json.dumps([{"label": "car", "box": BOX}]),
    )
    before = events.get(existing["id"])
    handler, published, writer, sink, _independent = claimed_snapshot_handler(tmp_path, events, faces)
    qualification = {}

    outcome = handler.refine(
        "onvif/motion", "motion", datetime.fromisoformat(STAMP), qualification,
        existing_event_id=existing["id"],
    )

    assert events.get(existing["id"]) == before
    assert old.exists()
    assert qualification["refinement_evidence_preserved"] is True
    assert outcome.event_id == existing["id"]
    assert outcome.snapshot_path == ""
    assert outcome.object_detected is None
    assert outcome.detected_objects == ()
    assert outcome.rejection_reason == "refinement_unavailable_preserved"
    assert outcome.refinement_pending is False
    assert outcome.processing_timing["phases_ms"] == {"detector_ms": 1.0}
    assert published == []
    sink.assert_not_called()
    assert writer.call_count == 1
    assert faces.observation_count() == 0


def test_claimed_new_event_publishes_empty_snapshot_without_tracking_seeds(tmp_path):
    events, faces = stores(tmp_path)
    handler, published, writer, sink, independent = claimed_snapshot_handler(tmp_path, events, faces)

    outcome = handler.handle("onvif/motion", "motion", datetime.fromisoformat(STAMP), {})

    stored = events.get(outcome.event_id)
    assert stored["snapshot_path"] == ""
    assert stored["snapshot_size_bytes"] == 0
    assert stored["recording_path"] == "recordings/gate/new.mp4"
    assert next(item for item in json.loads(stored["objects_json"]) if item.get("label"))["label"] == "person"
    assert outcome.snapshot_path == ""
    assert outcome.object_detected is True
    assert outcome.detected_objects == ()
    assert [kind for kind, _payload in published] == ["incident", "object"]
    assert published[1][1]["snapshot_path"] == ""
    assert published[1][1]["objects"][0]["label"] == "person"
    sink.assert_called_once()
    assert writer.call_count == 2
    with faces._connect() as connection:
        row = connection.execute("select event_id, snapshot_path from face_observations").fetchone()
        assert row["event_id"] == outcome.event_id
        assert row["snapshot_path"] == independent.relative_to(tmp_path).as_posix()

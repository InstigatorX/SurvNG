from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import threading

import numpy as np
import pytest

from survng.app.event_store import EventStore
from survng.app.faces import FaceStore


OLD = "2020-01-01T00:00:00+00:00"
NEW = "2025-01-01T00:00:00+00:00"
CUTOFF = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()


def observation(store, event_id, *, path="face.jpg", person_id=None, canonical=1,
                duplicate_of=None, track="", status=None, vector=None, observed_at=OLD):
    with store._connect() as connection:
        return int(connection.execute(
            """insert into face_observations (
                event_id, object_index, camera_id, snapshot_path, box_json,
                observed_at, created_at, person_id, review_status, canonical,
                duplicate_of_observation_id, candidate_track_id, embedding_blob,
                embedding_model, quality_score, recognition_pending
            ) values (?, 0, 'gate', ?, '{}', ?, ?, ?, ?, ?, ?, ?, ?, 'model', 0.8, 0)""",
            (event_id, path, observed_at, observed_at, person_id,
             status or ("confirmed" if person_id else "unknown"), canonical,
             duplicate_of, track,
             np.asarray(vector, dtype=np.float32).tobytes() if vector is not None else None),
        ).lastrowid)


def stores(tmp_path):
    events = EventStore(tmp_path)
    faces = FaceStore(tmp_path, max_observations=100, start_recognition=False, recognizer=SimpleNamespace(
        enabled=False, config=SimpleNamespace(),
        status=lambda: {"model_fingerprint": "model"},
    ))
    return events, faces


def event_face(events, faces, *, stamp=OLD, person_id=None, vector=None):
    directory = faces.storage_dir / "snapshots" / "gate"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"event-{len(list(directory.glob('event-*.jpg')))}.jpg"
    path.write_bytes(b"snapshot")
    event = events.add_event(camera_id="gate", kind="object", snapshot_path=str(path), created_at=stamp)
    face_id = observation(faces, event["id"], path=path.relative_to(faces.storage_dir).as_posix(), person_id=person_id, vector=vector)
    return face_id, path


@pytest.mark.parametrize("mutation", ["create", "bootstrap", "enrich", "pin"])
def test_pin_mutations_interleaved_after_real_retention_claim(tmp_path, mutation):
    events, faces = stores(tmp_path)
    person = None if mutation == "create" else faces.create_person("Alice")["id"]
    face_id, path = event_face(events, faces, person_id=person, vector=[1, 0])
    original = events._snapshot_path_for_retention
    called = []

    def during_claim(raw_path):
        called.append(raw_path)
        if mutation == "create":
            with pytest.raises(RuntimeError, match="being removed"):
                faces.create_person("Alice", face_id)
            assert faces.people() == []
            assert faces.observation(face_id)["person_id"] is None
        elif mutation == "pin":
            with pytest.raises(RuntimeError, match="being removed"):
                faces.set_reference_pinned(face_id, True)
        elif mutation == "bootstrap":
            assert faces.bootstrap_person_references(person, seed_observation_id=face_id) == []
        else:
            assert faces.enrich_person_gallery(person)["added"] == 0
        assert not faces.observation(face_id)["reference_pinned"]
        return original(raw_path)

    with patch.object(events, "_snapshot_path_for_retention", side_effect=during_claim):
        result = events.apply_snapshot_retention(CUTOFF, 10)
    assert len(called) == 1
    assert result["deleted_files"] == 1
    assert not path.exists()
    assert faces.observation(face_id)["snapshot_path"] == ""
    if person:
        with pytest.raises(RuntimeError, match="unavailable"):
            faces.set_reference_pinned(face_id, True)
        assert faces.bootstrap_person_references(person) == []


def gallery(events, faces):
    person = faces.create_person("Alice")["id"]
    ids = [event_face(events, faces, stamp=NEW, person_id=person, vector=vector)[0]
           for vector in ([1, 0], [1, 0.1], [0, 1], [0.1, 1])]
    other = faces.create_person("Bob")["id"]
    event_face(events, faces, stamp=NEW, person_id=other, vector=[0, -1])
    with faces._connect() as connection:
        connection.execute("update face_observations set reference_pinned = 1, reference_auto_pinned = 1 where id = ?", (ids[0],))
    return person, ids


def test_optimizer_claim_failure_rolls_back_gallery_changes(tmp_path):
    events, faces = stores(tmp_path)
    person, ids = gallery(events, faces)
    preview = faces.optimize_person_gallery(person, max_references=2)
    assert preview["improved"]
    target = next(item for item in preview["optimized_reference_ids"] if item != ids[0])
    with faces._connect() as connection:
        connection.execute("update events set created_at = ? where id = (select event_id from face_observations where id = ?)", (OLD, target))
    original = events._snapshot_path_for_retention

    def during_claim(raw_path):
        with pytest.raises(RuntimeError, match="being removed"):
            faces.optimize_person_gallery(person, max_references=2, apply=True)
        assert faces.observation(ids[0])["reference_pinned"]
        assert not faces.observation(target)["reference_pinned"]
        return original(raw_path)

    with patch.object(events, "_snapshot_path_for_retention", side_effect=during_claim):
        assert events.apply_snapshot_retention(CUTOFF, 10)["deleted_files"] == 1


def test_optimizer_revalidates_operator_pin_changes(tmp_path):
    events, faces = stores(tmp_path)
    person, ids = gallery(events, faces)
    median = np.median
    changed = False

    def operator_pin(values):
        nonlocal changed
        if not changed:
            changed = True
            faces.set_reference_pinned(ids[0], True)
        return median(values)

    with patch("survng.app.face_store.people.np.median", side_effect=operator_pin):
        with pytest.raises(RuntimeError, match="gallery changed"):
            faces.optimize_person_gallery(person, max_references=2, apply=True)
    assert faces.observation(ids[0])["reference_pinned"]
    assert not faces.observation(ids[0])["reference_auto_pinned"]


def test_optimizer_preserves_explicit_pin_ownership(tmp_path):
    events, faces = stores(tmp_path)
    person, ids = gallery(events, faces)
    faces.set_reference_pinned(ids[0], True)
    assert faces.optimize_person_gallery(person, max_references=2, apply=True)["applied"]
    assert faces.observation(ids[0])["reference_pinned"]
    assert not faces.observation(ids[0])["reference_auto_pinned"]


def test_seed_pin_precedes_retention_and_temporal_crop_has_separate_ownership(tmp_path):
    events, faces = stores(tmp_path)
    face_id, path = event_face(events, faces)
    faces.create_person("Alice", face_id)
    assert events.apply_snapshot_retention(CUTOFF, 10)["deleted_files"] == 0
    assert path.exists()
    temporal_id, event_path = event_face(events, faces)
    crop = tmp_path / "temporal.jpg"
    crop.write_bytes(b"crop")
    with faces._connect() as connection:
        connection.execute("update face_observations set snapshot_path = ?, candidate_track_id = 'track' where id = ?", (crop.name, temporal_id))
    original = events._snapshot_path_for_retention

    def during_claim(raw_path):
        faces.create_person("Bob", temporal_id)
        return original(raw_path)

    with patch.object(events, "_snapshot_path_for_retention", side_effect=during_claim):
        assert events.apply_snapshot_retention(CUTOFF, 10)["deleted_files"] == 1
    assert not event_path.exists()
    assert crop.exists()
    assert faces.observation(temporal_id)["reference_pinned"]


def fill_limit(faces):
    for event_id in range(10, 110):
        observation(faces, event_id, observed_at=NEW)


def test_prune_removes_cross_event_duplicate_tree_and_owned_crops(tmp_path):
    events, faces = stores(tmp_path)
    legacy = tmp_path / "legacy.jpg"
    legacy.write_bytes(b"event-owned")
    source = observation(faces, 1, path=legacy.name, vector=[1, 0])
    hidden = observation(faces, 2, vector=[1, 0])
    with faces._connect() as connection:
        assert faces._mark_exact_embedding_duplicate_locked(connection, hidden) == source
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"crop")
    temporal = observation(faces, 3, path=crop.name, canonical=0, duplicate_of=hidden, track="track")
    fill_limit(faces)
    with faces._lock, faces._connect() as connection:
        assert faces._prune_locked(connection) == 3
        assert connection.execute("select count(*) from face_observations").fetchone()[0] == 100
    assert all(faces.observation(item) is None for item in (source, hidden, temporal))
    assert legacy.exists()
    assert not crop.exists()


@pytest.mark.parametrize("protection", ["reviewed", "pinned", "visible"])
def test_prune_preserves_source_when_duplicate_group_has_protected_evidence(tmp_path, protection):
    _events, faces = stores(tmp_path)
    source = observation(faces, 1)
    person = faces.create_person("Alice")["id"] if protection == "reviewed" else None
    hidden = observation(faces, 2, canonical=int(protection == "visible"), duplicate_of=source,
                         person_id=person, observed_at="2026-01-01T00:00:00+00:00")
    if protection == "pinned":
        with faces._connect() as connection:
            connection.execute("update face_observations set reference_pinned = 1 where id = ?", (hidden,))
    fill_limit(faces)
    with faces._lock, faces._connect() as connection:
        faces._prune_locked(connection)
    assert faces.observation(source) is not None
    assert faces.observation(hidden)["duplicate_of_observation_id"] == source
    assert faces.observation_count() == 100


def test_prune_cleans_existing_unreviewed_orphan_below_limit(tmp_path):
    _events, faces = stores(tmp_path)
    hidden = observation(faces, 1, canonical=0, duplicate_of=999)
    with faces._lock, faces._connect() as connection:
        assert faces._prune_locked(connection) == 1
    assert faces.observation(hidden) is None


def test_repeated_candidate_payload_does_not_delete_retained_crop(tmp_path):
    _events, faces = stores(tmp_path)
    path = tmp_path / "crop.jpg"
    path.write_bytes(b"crop")
    payload = {"snapshot_path": str(path), "box": {"x1": 1, "y1": 1, "x2": 30, "y2": 30},
               "track_id": "track", "rank": 1, "offset_seconds": 1, "quality_score": 0.8}
    assert faces.ingest_candidates(1, "gate", OLD, [payload]) == 1
    assert faces.ingest_candidates(1, "gate", OLD, [payload]) == 0
    assert path.exists()
    discarded = tmp_path / "discarded.jpg"
    discarded.write_bytes(b"unused")
    assert faces.ingest_candidates(1, "gate", OLD, [dict(payload, snapshot_path=str(discarded))]) == 0
    assert not discarded.exists()
    assert path.exists()


def test_crop_cleanup_preserves_event_owned_path(tmp_path):
    events, faces = stores(tmp_path)
    path = tmp_path / "shared.jpg"
    path.write_bytes(b"event")
    events.add_event(camera_id="gate", kind="object", snapshot_path=str(path), created_at=NEW)
    source = observation(faces, 2, path=path.name, track="track")
    fill_limit(faces)
    with faces._lock, faces._connect() as connection:
        assert faces._prune_locked(connection) == 1
    assert faces.observation(source) is None
    assert path.exists()


def test_enrichment_rechecks_concurrent_explicit_pin(tmp_path):
    events, faces = stores(tmp_path)
    person = faces.create_person("Alice")["id"]
    face_id, _path = event_face(events, faces, person_id=person, vector=[1, 0])
    candidates = faces.gallery_candidates

    def operator_pin(*args, **kwargs):
        result = candidates(*args, **kwargs)
        faces.set_reference_pinned(face_id, True)
        return result

    with patch.object(faces, "gallery_candidates", side_effect=operator_pin):
        result = faces.enrich_person_gallery(person)
    assert result["selected_observation_ids"] == []
    assert faces.observation(face_id)["reference_pinned"]
    assert not faces.observation(face_id)["reference_auto_pinned"]


@pytest.mark.parametrize("mutation", ["create", "pin"])
def test_reference_retention_conflicts_reach_http_boundary(tmp_path, mutation):
    from fastapi import HTTPException
    from survng.app.face_routes import (
        FacePersonCreate, FaceReferenceUpdate, FaceRouteDependencies, create_face_router,
    )

    events, faces = stores(tmp_path)
    person = None if mutation == "create" else faces.create_person("Alice")["id"]
    face_id, _path = event_face(events, faces, person_id=person)
    with faces._connect() as connection:
        connection.execute(
            """insert into media_deletion_claims (path, role, claimed_at)
            select snapshot_path, 'snapshot', ? from face_observations where id = ?""",
            (NEW, face_id),
        )
    bundle = create_face_router(FaceRouteDependencies(
        get_manager=lambda: SimpleNamespace(faces=faces), manager_lock=threading.RLock(),
        start_observation_sync=lambda: None,
    ))
    with pytest.raises(HTTPException) as error:
        if mutation == "create":
            bundle.handlers["create_face_person"](FacePersonCreate(name="Alice", observation_id=face_id))
        else:
            bundle.handlers["update_face_reference"](face_id, FaceReferenceUpdate(pinned=True))
    assert error.value.status_code == 409
    assert "being removed" in error.value.detail

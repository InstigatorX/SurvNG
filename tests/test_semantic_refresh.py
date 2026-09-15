"""Concurrent semantic cover refresh must supersede prior evidence."""
import threading
import sqlite3
from pathlib import Path
from types import SimpleNamespace
import gc
import queue
import pytest
import cv2
import numpy as np
from survng.app.config import SemanticSearchConfig
from survng.app.semantic_search import SemanticSearchService, SemanticIndex, SemanticModelIdentity


def test_cover_refresh_supersedes_inflight_and_queued_evidence(tmp_path):
    entered, release = threading.Event(), threading.Event()
    identity = SemanticModelIdentity('audit', 'model', 'preprocessing', 2)
    class Encoder:
        def __init__(self):
            self.identity = identity
            self.calls = 0
        def encode_images(self, images):
            self.calls += 1
            entered.set()
            assert release.wait(3)
            return np.array([[1, 0] for _ in images], dtype=np.float32)

    old_path, new_path = tmp_path / 'old.png', tmp_path / 'new.png'
    cv2.imwrite(str(old_path), np.zeros((20, 20, 3), dtype=np.uint8))
    cv2.imwrite(str(new_path), np.full((20, 20, 3), 255, dtype=np.uint8))
    database_path = tmp_path / 'semantic.sqlite3'
    with sqlite3.connect(database_path) as connection:
        connection.execute('create table events (id integer primary key)')
        connection.execute('insert into events values (1)')
    index = SemanticIndex(database_path)
    service = SemanticSearchService(SemanticSearchConfig(enabled=True, index_object_crops=True), index, tmp_path, {})
    service._storage_dir = tmp_path
    service.encoder = Encoder()
    old = {'id': 1, 'camera_id': 'gate', 'created_at': '2026-09-01T00:00:00+00:00', 'snapshot_path': str(old_path), 'objects': [{'label':'person','confidence':.9,'box':{'x1':1,'y1':1,'x2':19,'y2':19}}]}
    new = {**old, 'snapshot_path': str(new_path)}
    stored = {"event": old}
    service._event_store = SimpleNamespace(get=lambda event_id: stored["event"])
    errors = []
    def run_old():
        try: service.index_event(old)
        except BaseException as exc: errors.append(exc)
    worker = threading.Thread(target=run_old)
    worker.start()
    try:
        assert entered.wait(3)
        assert service.queue_event(old)
        stored["event"] = new
        assert service.refresh_event(new)
        # An older notification arriving later must resolve the canonical cover.
        assert service.queue_event(old)
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and not errors
    _, _, stale = service._queue.get_nowait()
    assert service.index_event(stale) == 0
    _, _, queued = service._queue.get_nowait()
    assert service.index_event(queued) == 2
    # The stale cover stops before its crop; the replacement encodes cover
    # and crop separately so production can be admitted between them.
    assert service.encoder.calls == 3
    assert service._queue.empty()  # The delayed notification shares the queued revision.
    assert service.encoder.calls == 3
    with index._connect() as connection:
        rows = connection.execute('select image_path from semantic_embeddings where event_id=1').fetchall()
    assert len(rows) == 2
    assert {row['image_path'] for row in rows} == {str(new_path)}


def test_revision_registry_does_not_retain_completed_incidents(tmp_path):
    index = SemanticIndex(tmp_path / "semantic.sqlite3")
    service = SemanticSearchService(SemanticSearchConfig(enabled=True), index, tmp_path, {})
    for event_id in range(1000):
        queued = service._revision_event({"id": event_id, "snapshot_path": "cover.png"})
        assert len(service._event_revisions) == 1
    del queued
    gc.collect()
    assert len(service._event_revisions) == 0


def _revision_service(tmp_path):
    database_path = tmp_path / "revision.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("create table events (id integer primary key, evidence_revision integer, snapshot_path text)")
        connection.execute("insert into events values (1, 1, 'old.png')")
    for name, color in (("old.png", 0), ("new.png", 255)):
        assert cv2.imwrite(str(tmp_path / name), np.full((20, 20, 3), color, dtype=np.uint8))
    identity = SemanticModelIdentity("audit", "model", "preprocessing", 2)

    class Encoder:
        def __init__(self):
            self.identity = identity
            self.fail = False
            self.after_encode = lambda: None

        def encode_images(self, images):
            if self.fail:
                raise RuntimeError("temporary encoding failure")
            self.after_encode()
            return np.array([[1, 0] if image.mean() < 128 else [0, 1] for image in images], dtype=np.float32)

    index = SemanticIndex(database_path)
    service = SemanticSearchService(SemanticSearchConfig(enabled=True), index, tmp_path, {})
    service._storage_dir = tmp_path
    service.encoder = Encoder()
    event = {
        "id": 1, "camera_id": "gate", "created_at": "2026-09-13T20:00:00+00:00",
        "snapshot_path": "old.png", "evidence_revision": 1,
        "objects": [{"label": "person", "box": {"x1": 1, "y1": 1, "x2": 19, "y2": 19}}],
    }
    service._event_store = SimpleNamespace(get=lambda _: dict(event))

    def change_cover():
        event.update(snapshot_path="new.png", evidence_revision=2)
        with sqlite3.connect(database_path) as connection:
            connection.execute("update events set snapshot_path = 'new.png', evidence_revision = 2 where id = 1")

    return service, event, change_cover


def test_ordinary_queue_replaces_full_frame_and_unchanged_crop_geometry(tmp_path):
    service, event, change_cover = _revision_service(tmp_path)
    assert service.index_event(event) == 2
    change_cover()
    assert service.queue_event(event)  # Accepted refinement uses this path.
    _, _, queued = service._queue.get_nowait()
    assert service.index_event(queued) == 2
    rows = service.index.search([0, 1], service.encoder.identity)
    assert len(rows) == 2
    assert {row.image_path for row in rows} == {"new.png"}
    assert all(row.score > .99 for row in rows)
    with service.index._connect() as connection:
        assert {row[0] for row in connection.execute("select evidence_revision from semantic_embeddings")} == {2}
    assert service.index_event(event) == 0


def test_refresh_keeps_searchable_vectors_on_queue_and_encoding_failure(tmp_path):
    service, event, change_cover = _revision_service(tmp_path)
    assert service.index_event(event) == 2
    change_cover()
    service._queue = queue.PriorityQueue(maxsize=1)
    service._queue.put_nowait((0, -1, {}))
    assert not service.refresh_event(event)
    assert {hit.image_path for hit in service.index.search([1, 0], service.encoder.identity)} == {"old.png"}
    service._queue.get_nowait()
    assert service.refresh_event(event)
    _, _, queued = service._queue.get_nowait()
    service.encoder.fail = True
    with pytest.raises(RuntimeError, match="temporary encoding failure"):
        service.index_event(queued)
    assert {hit.image_path for hit in service.index.search([1, 0], service.encoder.identity)} == {"old.png"}


def test_commit_guard_rejects_old_inflight_result_before_notification_arrives(tmp_path):
    service, event, change_cover = _revision_service(tmp_path)
    service.encoder.after_encode = change_cover
    assert service.index_event(event) == 0
    assert service.index.coverage(service.encoder.identity)["evidence_count"] == 0
    service.encoder.after_encode = lambda: None
    assert service.index_event(event) == 2


def test_revision_change_reencodes_even_when_image_path_is_unchanged(tmp_path):
    service, event, _ = _revision_service(tmp_path)
    assert service.index_event(event) == 2
    event["evidence_revision"] = 2
    with service.index._connect() as connection:
        connection.execute("update events set evidence_revision = 2 where id = 1")
    assert service.queue_event(event)
    _, _, queued = service._queue.get_nowait()
    assert service.index_event(queued) == 2


def test_crop_only_projection_removes_last_obsolete_crop(tmp_path):
    service, event, _ = _revision_service(tmp_path)
    service.config.index_full_frame = False
    assert service.index_event(event) == 1
    event.update(evidence_revision=2, objects=[{"label": "person"}])
    with service.index._connect() as connection:
        connection.execute("update events set evidence_revision = 2 where id = 1")
    assert not service.projection_current(event)
    assert service.index_event(event) == 0
    assert service.projection_current(event)
    assert service.index.coverage(service.encoder.identity)["evidence_count"] == 0


@pytest.mark.parametrize('full_frame', [True, False])
def test_unencodable_crops_have_durable_revision_scoped_completion(tmp_path, full_frame):
    service, event, change_cover = _revision_service(tmp_path)
    service.config.index_full_frame = full_frame
    event['objects'][0]['box'] = {'x1': 21, 'y1': 1, 'x2': 29, 'y2': 19}
    assert service.index_event(event) == int(full_frame)
    assert service.projection_current(event)
    # Completion includes an explicit skip, including when no vectors exist.
    receipt = service.index.projection_receipt(event, service.encoder.identity,
                                               service._projection_plan_key(event['objects']))
    assert list(receipt['skipped_crops'].values()) == ['empty_after_clipping']
    service.index = SemanticIndex(tmp_path / 'revision.sqlite3')
    assert service.projection_current(event)
    service.encoder.fail = True
    assert service.index_event(event) == 0  # No repeated image/model work.
    change_cover()
    assert not service.projection_current(event)
    event['objects'][0]['box'] = {'x1': 1, 'y1': 1, 'x2': 19, 'y2': 19}
    service.encoder.fail = False
    assert service.index_event(event) == 1 + int(full_frame)
    assert service.projection_current(event)


def test_projection_receipt_rolls_back_with_stale_encoded_result(tmp_path):
    service, event, change_cover = _revision_service(tmp_path)
    old = dict(event)
    service.encoder.after_encode = change_cover
    assert service.index_event(old) == 0
    assert service.index.projection_receipt(old, service.encoder.identity,
                                           service._projection_plan_key(old['objects'])) is None
    assert not service.projection_current(event)


def test_missing_image_does_not_become_a_completed_skip(tmp_path):
    service, event, _ = _revision_service(tmp_path)
    (tmp_path / 'old.png').unlink()
    assert service.index_event(event) == 0
    assert not service.projection_current(event)
    assert service.index.projection_receipt(event, service.encoder.identity,
                                           service._projection_plan_key(event['objects'])) is None


@pytest.mark.parametrize('operation', ['refresh', 'queue', 'index', 'backfill'])
def test_negative_correction_has_one_policy_across_index_entrypoints(tmp_path, operation):
    service, event, _ = _revision_service(tmp_path)
    assert service.index_event(event) == 2
    event['objects'] = [{**event['objects'][0], 'incident_eligible': False}]
    event['evidence_revision'] = 2
    with sqlite3.connect(tmp_path / 'revision.sqlite3') as conn:
        conn.execute('update events set evidence_revision=2 where id=1')
    service.encoder.fail = True  # Deletion cannot depend on a working encoder.
    if operation == 'backfill':
        service._backfill(SimpleNamespace(recent_compact=lambda *args: [dict(event)]))
    else:
        getattr(service, operation + '_event')(event)
    assert not service.index.event_indexed(1, service.encoder.identity)
    assert service._queue.empty()
    assert service.projection_current(event)
    # A late callback or direct replay cannot resurrect the negative evidence.
    assert service.queue_event(event)
    assert service.index_event(event) == 0
    assert not service.index.event_indexed(1, service.encoder.identity)
    service.encoder.fail = False
    event['objects'] = [{**event['objects'][0], 'incident_eligible': True}]
    event['evidence_revision'] = 3
    with sqlite3.connect(tmp_path / 'revision.sqlite3') as conn:
        conn.execute('update events set evidence_revision=3 where id=1')
    assert service.index_event(event) == 2


def test_negative_deletion_blocks_unnotified_inflight_positive_commit(tmp_path):
    from survng.app.evidence_projection import EvidenceProjection
    service, event, _ = _revision_service(tmp_path)
    assert service.queue_event(event)
    _, _, queued = service._queue.get_nowait()
    def delete_during_encoding():
        event['objects'] = [{**event['objects'][0], 'incident_eligible': False}]
        event['evidence_revision'] = 2
        with sqlite3.connect(tmp_path / 'revision.sqlite3') as conn:
            conn.execute('update events set evidence_revision=2 where id=1')
        projection = EvidenceProjection(None, lambda: service, None, lambda *_: None)
        assert projection._semantic_current(service, event, 0)
    service.encoder.after_encode = delete_during_encoding
    assert service.index_event(queued) == 0
    assert not service.index.event_indexed(1, service.encoder.identity)
    assert service.queue_event(event)
    assert service._queue.empty()

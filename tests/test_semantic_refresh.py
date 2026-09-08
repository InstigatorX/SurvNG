"""Concurrent semantic cover refresh must supersede prior evidence."""
import threading
import sqlite3
from pathlib import Path
from types import SimpleNamespace
import gc
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
    assert service.encoder.calls == 2
    _, _, delayed = service._queue.get_nowait()
    assert service.index_event(delayed) == 0
    assert service.encoder.calls == 2
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

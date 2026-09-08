"""Regression coverage for audited workflow failures using real stores/queues."""
import json
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from survng.app.appearance_backfill import DeferredAppearanceBackfill
from survng.app.appearance_index import AppearanceIndex
from survng.app.config import ObjectTrackingConfig
from survng.app.face_store import FaceStore
from survng.app.inference import InferenceUnavailable
from survng.app.security import required_api_scope
from survng.app.support_bundle import _safe_value


def test_text_search_is_read_only_but_mutations_remain_admin():
    assert required_api_scope('POST', '/api/semantic-search') == 'read'
    for method, path in [('DELETE', '/api/semantic-search'), ('POST', '/api/semantic-search/reindex'), ('PUT', '/api/tls')]:
        assert required_api_scope(method, path) == 'admin'


@pytest.mark.parametrize('value', [
    '/Users/example/models/person.xml', '/private/tmp/session/database.sqlite3',
    '/Volumes/Recordings/gate/clip.mp4', '/custom-root/video.mp4',
    'open failed: "/Users/Example Person/private/model.xml"',
    r'failed: C:\Users\Example\model.xml', r'failed: \\server\private\model.xml',
])
def test_support_paths_are_redacted_across_platforms(value):
    sanitized = _safe_value({'message': value, 'model_xml': value})
    assert 'example' not in json.dumps(sanitized).lower()
    assert 'model.xml' not in json.dumps(sanitized)
    assert 'clip.mp4' not in json.dumps(sanitized)
    assert sanitized['model_xml'] == '[redacted]'
    assert '[redacted-path]' in sanitized['message']


def test_both_compose_examples_forward_go2rtc_settings():
    root = Path(__file__).resolve().parents[1]
    documented = (root / 'README.install.docker.md').read_text().split('cat > /tmp/survng.compose.yaml <<EOF\n', 1)[1].split('\nEOF', 1)[0]
    for text in [(root / 'compose.yaml').read_text(), documented.replace('\\$', '$')]:
        env = yaml.safe_load(text)['services']['survng']['environment']
        assert env['SURVNG_GO2RTC'] == '${SURVNG_GO2RTC:-1}'
        assert env['SURVNG_GO2RTC_CONFIG'] == '${SURVNG_GO2RTC_CONFIG:-/config/go2rtc.yaml}'
        # Exercise default, disabled bundled process, and a custom mounted path.
        import re
        for selected in ({}, {'SURVNG_GO2RTC': '0', 'SURVNG_GO2RTC_CONFIG': '/config/external.yaml'}):
            resolved = {key: re.sub(r'\$\{([^:}]+):-([^}]+)\}', lambda m: selected.get(m[1]) or m[2], value) for key, value in env.items() if key.startswith('SURVNG_GO2RTC')}
            assert resolved['SURVNG_GO2RTC'] == selected.get('SURVNG_GO2RTC', '1')
            assert resolved['SURVNG_GO2RTC_CONFIG'] == selected.get('SURVNG_GO2RTC_CONFIG', '/config/go2rtc.yaml')


def test_face_suitability_uses_finite_populated_quality(tmp_path):
    store = FaceStore(tmp_path, start_recognition=False)
    with store._connect() as con:
        for event_id, size in enumerate([0.7, 0.3, 'invalid', float('nan'), float('inf'), None], 1):
            con.execute('''insert into face_observations(event_id, object_index, camera_id, snapshot_path, box_json, observed_at, created_at, quality_json)
                values (?,0,'gate','','{}','now','now',?)''', (event_id, json.dumps({'size': size})))
    result = store.camera_suitability()[0]
    assert result['observations'] == 6
    assert result['average_face_size'] == 0.5
    store.close()


def test_face_shutdown_joins_when_real_ingestion_fills_queue(tmp_path, monkeypatch):
    snapshot = tmp_path / 'candidate.jpg'
    snapshot.write_bytes(b'candidate')
    recognizer = SimpleNamespace(enabled=True, config=SimpleNamespace(face_auto_identify_enabled=False), status=lambda: {'ready': True, 'model_fingerprint': 'test'})
    store = FaceStore(tmp_path, max_observations=100, recognizer=recognizer, start_recognition=False)
    entered, release = threading.Event(), threading.Event()
    def recognize(_):
        entered.set()
        assert release.wait(10)
        return False
    monkeypatch.setattr(store, '_recognize_observation', recognize)
    store.start()
    thread = store._recognition_thread
    try:
        candidates = [{'snapshot_path': str(snapshot), 'box': {'x1': 1, 'y1': 1, 'x2': 30, 'y2': 30}, 'confidence': .9, 'track_id': 'track', 'rank': rank, 'offset_seconds': float(rank), 'quality_score': .8} for rank in range(1, 5)]
        for event_id in range(1, 27):
            store.ingest_candidates(event_id, 'gate', '2026-09-07T00:00:00+00:00', candidates)
            assert entered.wait(1)
        assert store._recognition_queue.full()
        put = store._recognition_queue.put_nowait
        def put_and_release(value):
            try:
                return put(value)
            finally:
                if value is None:
                    release.set()
        monkeypatch.setattr(store._recognition_queue, 'put_nowait', put_and_release)
        store.close()
        assert not thread.is_alive()
        assert store._recognition_thread is None
    finally:
        store._recognition_stop.set()
        release.set()
        thread.join(3)


@pytest.mark.parametrize('failure', ['unready', 'identity', 'shed', 'partial'])
def test_backfill_retains_retryable_work_until_model_recovers(tmp_path, monkeypatch, failure):
    database = tmp_path / 'events.db'
    with sqlite3.connect(database) as con:
        con.execute('create table events(id integer primary key, created_at text not null)')
        con.execute("insert into events values(7, 'now')")
    (tmp_path / 'snapshots').mkdir()
    assert cv2.imwrite(str(tmp_path / 'snapshots/event.jpg'), np.full((100, 200, 3), 127, np.uint8))
    detected = {'label': 'car', 'box': {'x1': 0, 'y1': 0, 'x2': 200, 'y2': 100}}
    event = {'id': 7, 'camera_id': 'gate', 'created_at': '2026-09-07T00:00:00+00:00', 'snapshot_path': 'snapshots/event.jpg', 'objects_json': json.dumps([detected, detected])}
    class Encoder:
        recovering = True
        calls = 0
        def supports_label(self, label):
            return not (self.recovering and failure == 'unready')
        def model_identity_for_label(self, label):
            if self.recovering and failure == 'identity':
                return None
            return dict(model_kind='vehicle', model_fingerprint='test', embedding_size=2, match_threshold=.8)
        def embed_for_label(self, label, crop):
            self.calls += 1
            if self.recovering and (failure == 'shed' or (failure == 'partial' and self.calls % 2 == 0)):
                raise InferenceUnavailable('shed for incident')
            return np.array([.6, .8], np.float32)
    encoder = Encoder()
    index = AppearanceIndex(database)
    service = DeferredAppearanceBackfill(database, tmp_path, ObjectTrackingConfig(vehicle_reid_enabled=True, vehicle_reid_model_path='vehicle.xml'), SimpleNamespace(get=lambda _: event), index, encoder)
    service.enqueue(7, 'gate', delay_seconds=0)
    # Run the actual worker through a deferred result, then stop after it persists.
    retry = service._retry
    def retry_then_stop(*args, **kwargs):
        retry(*args, **kwargs)
        service._stop.set()
    monkeypatch.setattr(service, '_retry', retry_then_stop)
    service._run()
    latest = service.status()['latest']
    assert latest['state'] == 'queued'
    assert latest['available_at'] > time.time()
    assert not index.has_event(7)  # no partial append may poison has_event
    # Temporary unavailability remains queued even after the normal retry budget.
    retry(7, 'recovering', 20, deferred=True)
    assert service.status()['latest']['state'] == 'queued'
    encoder.recovering = False
    service.enqueue(7, 'gate', delay_seconds=0)
    assert service._claim() is not None
    state, count, reason = service.process_event(7)
    service._finish(7, state, reason, count)
    assert state == 'completed' and count == 2
    assert index.has_event(7)
    assert service._claim() is None


def test_ineligible_narrow_crop_does_not_suppress_valid_backfill(tmp_path):
    database = tmp_path / 'events.db'
    with sqlite3.connect(database) as con:
        con.execute('create table events(id integer primary key, created_at text not null)')
        con.execute("insert into events values(7, 'now')")
    (tmp_path / 'snapshots').mkdir()
    assert cv2.imwrite(str(tmp_path / 'snapshots/event.jpg'), np.full((1000, 200, 3), 127, np.uint8))
    event = {'id': 7, 'camera_id': 'gate', 'snapshot_path': 'snapshots/event.jpg', 'objects_json': json.dumps([
        {'label': 'car', 'box': {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}},
        {'label': 'car', 'box': {'x1': 0, 'y1': 0, 'x2': 7, 'y2': 1000}},
    ])}
    calls = []
    def embed(label, crop):
        calls.append(crop.shape)
        if min(crop.shape[:2]) < 8:
            raise ValueError('Vehicle crop is too small for ReID.')
        return np.array([.6, .8], np.float32)
    encoder = SimpleNamespace(supports_label=lambda _: True, model_identity_for_label=lambda _: dict(model_kind='vehicle', model_fingerprint='test', embedding_size=2, match_threshold=.8), embed_for_label=embed)
    index = AppearanceIndex(database)
    service = DeferredAppearanceBackfill(database, tmp_path, ObjectTrackingConfig(vehicle_reid_enabled=True, vehicle_reid_model_path='vehicle.xml'), SimpleNamespace(get=lambda _: event), index, encoder)
    assert service.process_event(7)[:2] == ('completed', 1)
    assert calls == [(100, 100, 3)]

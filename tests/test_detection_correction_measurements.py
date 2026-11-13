"""Guard measurement validity and the read-only deployment boundary."""
import importlib.util
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "correction_measurements", Path(__file__).parents[1] / "scripts/measure-detection-corrections.py"
)
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "index.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("""create table semantic_embeddings (
            id integer primary key, event_id integer, camera_id text, source_kind text,
            object_label text, implementation text, model_fingerprint text,
            preprocessing_fingerprint text, embedding_size integer, embedding_blob blob)""")
        for key, label, vector, model in [(1, 'person', [1, 0], 'a'), (2, 'person', [1, 1], 'a'),
                                         (3, 'dog', [0, 1], 'a'), (4, 'person', [1, 0], 'b')]:
            conn.execute("insert into semantic_embeddings values(?,?,'gate','object_crop',?,'test',?,'p',2,?)",
                         (key, key, label, model, np.array(vector, dtype=np.float16).tobytes()))
    return path


def test_read_only_generations_and_reviewed_margins(database):
    before = database.read_bytes()
    report = measure.index_measurements(database, {
        "pairs": [{"a": 1, "b": 2, "category": "same_object", "reviewer": "test"}],
        "triplets": [{"query": 1, "positive": 2, "opposing": 3, "reviewer": "test"}],
    })
    assert database.read_bytes() == before
    assert len(report['generations']) == 2
    assert report['reviewed']['threshold_recommendation'] is None
    assert next(iter(report['reviewed']['winning_minus_opposing'].values()))['p50'] == pytest.approx(2 ** -.5)
    assert all('same_object' not in key for generation in report['generations'] for key in generation['unreviewed_proxies'])


@pytest.mark.parametrize('pair', [
    {"a": 1, "b": 4, "category": "same_object", "reviewer": "test"},
    {"a": 1, "b": 1, "category": "same_object", "reviewer": "test"},
    {"a": 1, "b": 2, "category": "same_object"},
    {"a": 1, "b": 99, "category": "same_object", "reviewer": "test"},
])
def test_invalid_evidence_is_not_silently_used(database, pair):
    with pytest.raises(ValueError):
        measure.index_measurements(database, {"pairs": [pair]})


def test_corrupt_vector_is_reported(database):
    with sqlite3.connect(database) as conn:
        conn.execute("update semantic_embeddings set embedding_blob=? where id=1",
                     (np.array([np.nan, 0], dtype=np.float16).tobytes(),))
    with pytest.raises(ValueError, match='invalid embedding'):
        measure.index_measurements(database, {})


def test_crop_rescales_and_clips_without_mutating_source():
    frame = np.arange(10 * 20 * 3).reshape(10, 20, 3)
    result = measure.crop(frame, [-2, 2, 20, 10], [40, 20])
    assert np.array_equal(result, frame[1:5, :10])
    result[:] = 0
    assert frame[1, 0, 0] != 0
    with pytest.raises(ValueError):
        measure.crop(frame, [2, 2, 1, 1], [20, 10])


def test_encoder_failures_propagate_from_contention_worker():
    calls = []
    def encode(images):
        calls.append(len(images))
        if len(calls) == 3:
            raise RuntimeError('worker failure')
        return np.ones((len(images), 2))
    with pytest.raises(RuntimeError, match='worker failure'):
        measure.benchmark_encoder(SimpleNamespace(encode_images=encode), [np.zeros((2, 2, 3))], 1, 2, 10)

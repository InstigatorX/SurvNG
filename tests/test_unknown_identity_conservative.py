import numpy as np

from survng.app.unknown_identity import (
    cluster_unknown_embeddings,
    unknown_cluster_cohesion,
)


def _blob(values):
    vector = np.asarray(values, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return vector.tobytes()


def _row(observation_id, values, quality=0.8):
    return {
        "id": observation_id,
        "embedding_blob": _blob(values),
        "quality_score": quality,
    }


def test_conservative_cluster_keeps_clear_identity_together():
    rows = [
        _row(10, [1.0, 0.00]),
        _row(11, [0.99, 0.06]),
        _row(12, [0.98, -0.08]),
    ]
    result = cluster_unknown_embeddings(rows, threshold=0.80)
    assert result[10] == result[11] == result[12]


def test_bridge_does_not_chain_distant_population():
    rows = [
        _row(10, [1.00, 0.00, 0.00]),
        _row(11, [0.99, 0.08, 0.00]),
        _row(12, [0.97, 0.14, 0.00]),
        _row(20, [0.72, 0.69, 0.00]),
        _row(30, [0.10, 0.99, 0.00]),
        _row(31, [0.04, 1.00, 0.00]),
    ]
    result = cluster_unknown_embeddings(rows, threshold=0.75)
    assert result[10] == result[11] == result[12]
    assert result[30] == result[31]
    assert result[30] != result[10]


def test_low_quality_unknown_stays_out():
    rows = [_row(10, [1.0, 0.0], quality=0.2)]
    assert cluster_unknown_embeddings(rows, threshold=0.55) == {}


def test_cohesion_reports_radius():
    rows = [
        _row(10, [1.0, 0.0]),
        _row(11, [0.99, 0.05]),
        _row(12, [0.98, -0.05]),
    ]
    membership = cluster_unknown_embeddings(rows, threshold=0.8)
    metrics = unknown_cluster_cohesion(rows, membership)
    assert metrics[10]["centroid_min_similarity"] > 0.95

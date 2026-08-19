import numpy as np
from survng.app.unknown_identity import cluster_unknown_embeddings

def _blob(values):
    vector = np.asarray(values, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return vector.tobytes()

def test_similar_unknowns_share_stable_anchor():
    rows = [
        {"id": 10, "embedding_blob": _blob([1.0, 0.0]), "quality_score": 0.8},
        {"id": 14, "embedding_blob": _blob([0.98, 0.1]), "quality_score": 0.8},
        {"id": 20, "embedding_blob": _blob([0.0, 1.0]), "quality_score": 0.8},
    ]
    result = cluster_unknown_embeddings(rows, threshold=0.8)
    assert result[10] == 10
    assert result[14] == 10
    assert result[20] == 20

def test_low_quality_unknown_is_not_clustered():
    rows = [{"id": 10, "embedding_blob": _blob([1.0, 0.0]), "quality_score": 0.2}]
    assert cluster_unknown_embeddings(rows) == {}

"""Disabled recognition must not admit work, even with stale ready status."""
from unittest.mock import Mock, patch

import numpy as np
import pytest

from survng.app.config import DetectorConfig
from survng.app.face_store import FaceStore
from survng.app.inference import InferenceSupervisor, InferenceUnavailable


def test_disabled_recognizer_does_not_queue_captured_faces(tmp_path):
    recognizer = Mock(enabled=False)
    recognizer.status.return_value = {"ready": False, "enabled": False}
    (tmp_path / "face.jpg").write_bytes(b"snapshot")
    store = FaceStore(tmp_path, recognizer=recognizer, start_recognition=False)
    assert store.ingest_events([{
        "id": 1,
        "camera_id": "gate",
        "snapshot_path": "face.jpg",
        "objects_json": '[{"label":"face","confidence":0.9,"box":{"x1":0,"y1":0,"x2":100,"y2":100}}]',
    }]) == 1
    assert store._recognition_queue.empty()
    assert not store._recognition_pending
    # Keep durable observations for recognition when the operator enables it.
    assert store.recognition_status()["pending"] == 1
    store.start()
    assert store._recognition_thread is None
    recognizer.enabled = True
    store._queue_pending_recognition()
    assert store._recognition_queue.get_nowait() == 1


def test_disabled_recognizer_ignores_stale_ready_status(tmp_path):
    recognizer = Mock(enabled=False)
    recognizer.status.return_value = {"ready": True, "model_fingerprint": "stale"}
    store = FaceStore(tmp_path, recognizer=recognizer, start_recognition=False)
    with patch.object(store, "_connect", side_effect=AssertionError("must not read faces")):
        assert store._recognize_observation(1) is False
    recognizer.status.assert_not_called()
    recognizer.embed.assert_not_called()


def test_disabled_face_embedding_never_admits_or_dispatches_work():
    supervisor = InferenceSupervisor(DetectorConfig(enabled=False, face_recognition_enabled=False))
    with patch.object(supervisor, "_enter_device_workload") as admit, patch.object(supervisor._face, "request") as request:
        with pytest.raises(InferenceUnavailable, match="disabled"):
            supervisor.embed(np.zeros((32, 32, 3), dtype=np.uint8))
    admit.assert_not_called()
    request.assert_not_called()

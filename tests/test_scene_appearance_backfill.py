"""Appearance indexing adds missing retained subjects instead of skipping events."""
import json

import cv2
import numpy as np

from survng.app.appearance_backfill import DeferredAppearanceBackfill
from survng.app.appearance_index import AppearanceIndex
from survng.app.config import ObjectTrackingConfig
from survng.app.events import EventStore
from tests.test_appearance_backfill import _Encoder


def test_existing_appearance_vector_does_not_hide_second_subject_outside_alert_zone(tmp_path):
    store = EventStore(tmp_path)
    assert cv2.imwrite(str(tmp_path / "old.png"), np.full((100, 200, 3), 127, dtype=np.uint8))
    event = store.add_event("gate", "motion", snapshot_path="old.png", objects_json=json.dumps([
        {"label": "car", "confidence": .94, "track_id": 1,
         "box": {"x1": 0, "y1": 0, "x2": 80, "y2": 90}},
        {"label": "car", "confidence": .75, "incident_eligible": False,
         "box": {"x1": 100, "y1": 0, "x2": 180, "y2": 90}},
    ]))
    index = AppearanceIndex(store.db_path)
    index.replace_event(event["id"], "gate", [{
        "track_id": 1, "label": "car", "model_kind": "vehicle", "model_fingerprint": "vehicle-test",
        "embedding": np.array([.6, .8]), "observation_count": 5,
    }])
    service = DeferredAppearanceBackfill(store.db_path, tmp_path, ObjectTrackingConfig(
        vehicle_reid_enabled=True, vehicle_reid_model_path="vehicle.xml", deferred_reid_min_crop_pixels=256,
    ), store, index, _Encoder())
    assert service.process_event(event["id"])[:2] == ("completed", 1)
    assert service.process_event(event["id"])[0] == "skipped"
    with index._connect() as connection:
        rows = connection.execute("select * from appearance_embeddings where event_id=?", (event["id"],)).fetchall()
    assert len(rows) == 2
    retained = next(row for row in rows if row["observation_id"])
    assert store.scene_observation(retained["observation_id"])["object_id"] == retained["scene_object_id"]
    # A later tracking update must not erase the independently retained subject.
    index.replace_event(event["id"], "gate", [{
        "track_id": 1, "label": "car", "model_kind": "vehicle", "model_fingerprint": "vehicle-test",
        "embedding": np.array([.6, .8]), "observation_count": 6,
    }])
    with index._connect() as connection:
        assert connection.execute("select count(*) from appearance_embeddings where event_id=?", (event["id"],)).fetchone()[0] == 2

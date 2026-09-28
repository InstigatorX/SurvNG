"""Retained observations stay searchable independently of an event cover."""
import json

import cv2
import numpy as np

from survng.app.config import SemanticSearchConfig
from survng.app.events import EventStore
from survng.app.semantic_search import SemanticIndex, SemanticModelIdentity, SemanticSearchService


class Encoder:
    identity = SemanticModelIdentity("test", "scene-model", "scene-preprocessing", 2)

    def encode_images(self, images):
        return np.array([[1., 0.] for _ in images], dtype=np.float32)


def make_service(tmp_path):
    store = EventStore(tmp_path)
    assert cv2.imwrite(str(tmp_path / "old.png"), np.full((20, 20, 3), 127, dtype=np.uint8))
    assert cv2.imwrite(str(tmp_path / "new.png"), np.full((20, 20, 3), 255, dtype=np.uint8))
    event = store.add_event("gate", "motion", snapshot_path="old.png", objects_json=json.dumps([{
        "label": "person", "confidence": .75, "incident_eligible": False,
        "box": {"x1": 1, "y1": 1, "x2": 19, "y2": 19},
    }]))
    index = SemanticIndex(store.db_path)
    service = SemanticSearchService(SemanticSearchConfig(enabled=True, backfill_pause_seconds=.01), index, tmp_path, {})
    service._event_store = store
    service._storage_dir = tmp_path
    service.encoder = Encoder()
    return store, event, service


def test_retained_object_search_survives_cover_replacement_and_empty_current_detection(tmp_path):
    store, event, service = make_service(tmp_path)
    assert service.index_event(event) == 3
    observation = store.scene_search_observations(event_id=event["id"])[0]
    changed = store.refine_event_evidence(event["id"], snapshot_path="new.png", recording_path="", objects_json="[]")
    assert service.semantic_searchable(changed)
    assert service.index_event(changed) == 0
    hits = service.index.search([1., 0.], service.encoder.identity)
    assert len(hits) == 1
    assert hits[0].observation_id == observation["id"]
    assert hits[0].image_path == "old.png"
    assert (tmp_path / "old.png").exists()


def test_new_index_backfills_old_observation_even_when_current_cover_has_no_objects(tmp_path):
    store, event, service = make_service(tmp_path)
    changed = store.refine_event_evidence(event["id"], snapshot_path="new.png", recording_path="", objects_json="[]")
    service._backfill(store)
    _, _, queued = service._queue.get_nowait()
    assert queued["id"] == event["id"]
    assert service.index_event(queued) == 1
    assert service.projection_current(changed)


def test_current_projection_checks_observation_ids_without_loading_payloads(tmp_path):
    store, event, service = make_service(tmp_path)
    service.index_event(event)
    calls = {"payloads": 0, "search": 0}
    original_payloads = store.scene_observation_payloads
    original_search = store.scene_search_observations

    def payloads(ids):
        calls["payloads"] += 1
        return original_payloads(ids)

    def search(*args, **kwargs):
        calls["search"] += 1
        return original_search(*args, **kwargs)

    store.scene_observation_payloads = payloads
    store.scene_search_observations = search
    assert service.projection_current(event)
    assert calls["payloads"] == 0
    assert calls["search"] == 0


def test_unindexed_observation_without_a_box_does_not_keep_projection_pending(tmp_path):
    store, event, service = make_service(tmp_path)
    service.index_event(event)
    observation = store.scene_search_observations(event_id=event["id"])[0]
    payload = json.loads(observation["payload_json"])
    payload.pop("box", None)
    with store._connect() as connection:
        connection.execute(
            "insert into scene_observations("
            "id,object_id,episode_id,event_id,camera_id,captured_epoch,track_key,object_index,"
            "snapshot_path,recording_path,payload_json) "
            "select 'extra', object_id, episode_id, event_id, camera_id, captured_epoch, track_key, "
            "object_index, snapshot_path, recording_path, ? from scene_observations where id=?",
            (json.dumps(payload), observation["id"]),
        )
    calls = {"payloads": []}
    original_payloads = store.scene_observation_payloads

    def payloads(ids):
        calls["payloads"].append(list(ids))
        return original_payloads(ids)

    store.scene_observation_payloads = payloads
    assert service.projection_current(event)
    assert calls["payloads"] == [["extra"]]


def test_missing_observation_image_does_not_claim_searchable_vector(tmp_path):
    store, event, service = make_service(tmp_path)
    (tmp_path / "old.png").unlink()
    assert service.index_event(event) == 0
    assert service.index.search([1., 0.], service.encoder.identity) == []
    assert not service.projection_current(event)


def test_scene_label_correction_changes_search_filter_without_reencoding(tmp_path):
    store, event, service = make_service(tmp_path)
    service.index_event(event)
    observation = store.scene_search_observations(event_id=event["id"])[0]
    with store._connect() as connection:
        connection.execute("update scene_objects set label_override='dog' where id=?", (observation["object_id"],))
    hits = service.index.search([1., 0.], service.encoder.identity, object_labels=["dog"])
    assert len(hits) == 1
    assert hits[0].object_label == "dog"
    assert hits[0].observation_id == observation["id"]
    assert service.index.search([1., 0.], service.encoder.identity, object_labels=["person"]) == []

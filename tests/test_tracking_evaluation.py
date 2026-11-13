import copy

import pytest

from survng.app.config import ObjectTrackingConfig
from survng.app.tracking_evaluation import identity_metrics, replay_digest, selected_frames, validate_labels, validate_replay


def detection(x=10):
    return {"label": "person", "confidence": .9, "incident_eligible": True,
            "box": {"x1": x, "y1": 10, "x2": x+30, "y2": 80}}


def replay_fixture(count=4):
    replay = {"schema_version": 1, "camera_id": "gate", "tracking_config": ObjectTrackingConfig(min_confirmations=1).model_dump(mode="json"),
              "high_confidence_threshold": .7, "timestamp_source": "source_pts", "source_pts_frames": count,
              "appearance_source": "shared_supplied_embeddings",
              "frames": [{"frame_index": i, "captured_at": 100+i*.5, "width": 120, "height": 100, "detections": [detection()]} for i in range(count)]}
    replay["replay_id"] = replay_digest(replay)
    return replay


def truth(replay):
    return {"schema_version": 1, "replay_id": replay["replay_id"], "frames": [
        {"frame_index": f["frame_index"], "objects": [{"identity": "person-A", "label": "person", "box": detection()["box"]}]} for f in replay["frames"]]}


def observations(ids):
    return [{"frame_index": i, "objects": [] if value is None else [{"track_id": value, "label": "person", "box": detection()["box"]}]} for i, value in enumerate(ids)]


def test_metrics_perfect_switch_fragment_and_false_merge():
    replay = replay_fixture()
    labels = truth(replay)
    perfect = identity_metrics(observations([1,1,1,1]), labels, replay)
    assert perfect["idf1"] == 1
    assert perfect["id_switches"] == perfect["fragmentations"] == perfect["false_merges"] == 0
    split = identity_metrics(observations([1,1,2,2]), labels, replay)
    assert split["idf1"] == .5
    assert split["id_switches"] == 1
    assert split["fragmentations"] == 0
    gap = identity_metrics(observations([1,None,1,1]), labels, replay)
    assert gap["fragmentations"] == 1 and gap["idfn"] == 1
    for f in labels["frames"][2:]:
        f["objects"][0]["identity"] = "person-B"
    merged = identity_metrics(observations([1,1,1,1]), labels, replay)
    assert merged["false_merges"] == 1 and merged["idf1"] == .5


def test_idf1_uses_global_one_to_one_assignment():
    replay = replay_fixture()
    labels = truth(replay)
    # A and B swap tracker IDs: a global mapping cannot credit both identities
    # with both IDs, even though frame-level geometry is always perfect.
    obs = observations([1,1,2,2])
    for i, frame in enumerate(labels["frames"]):
        frame["objects"].append({"identity": "person-B", "label": "person", "box": detection(70)["box"]})
        obs[i]["objects"].append({"track_id": 2 if i < 2 else 1, "label": "person", "box": detection(70)["box"]})
    score = identity_metrics(obs, labels, replay)
    assert score["idtp"] == 4 and score["idf1"] == .5
    assert score["id_switches"] == 2 and score["false_merges"] == 2


def test_labels_require_explicit_coverage_and_matching_input():
    replay = replay_fixture()
    labels = truth(replay)
    labels["frames"].pop()
    with pytest.raises(ValueError, match="every evaluated frame"):
        identity_metrics(observations([1,1,1,1]), labels, replay)
    labels = truth(replay)
    labels["replay_id"] = "different"
    with pytest.raises(ValueError, match="exact replay"):
        identity_metrics(observations([1,1,1,1]), labels, replay)


def test_replay_checksum_chronology_and_limits():
    replay = replay_fixture()
    validate_replay(replay)
    replay["frames"][1]["captured_at"] = 99
    with pytest.raises(ValueError, match="checksum"):
        validate_replay(replay)
    replay["replay_id"] = replay_digest(replay)
    with pytest.raises(ValueError, match="increasing"):
        validate_replay(replay)


def test_sampling_keeps_source_indexes_and_deliberate_gaps():
    replay = replay_fixture(60)
    sparse = selected_frames(replay, "sparse_gaps")
    assert sparse[0]["frame_index"] == 0
    assert all(not (5 <= f["captured_at"]-100 < 9 or 15 <= f["captured_at"]-100 < 21) for f in sparse)
    slow = selected_frames(replay, "fixed_075fps")
    assert len(slow) < len(replay["frames"])
    assert slow[1]["frame_index"] == 3
    assert replay["frames"][3]["captured_at"] == slow[1]["captured_at"]


def test_idf1_preserves_all_admissible_edges_before_global_assignment():
    replay = replay_fixture(2)
    labels = truth(replay)
    obs = observations([1, 1])
    for i, frame in enumerate(labels["frames"]):
        frame["objects"].append({"identity": "person-B", "label": "person", "box": detection(14)["box"]})
        obs[i]["objects"].append({"track_id": 2, "label": "person", "box": detection(14)["box"]})
    # In frame two spatially optimal assignment swaps IDs, but both globally
    # consistent identity pairs still clear the 0.5 IoU threshold in both frames.
    obs[1]["objects"][0]["box"], obs[1]["objects"][1]["box"] = (
        obs[1]["objects"][1]["box"], obs[1]["objects"][0]["box"],
    )
    metrics = identity_metrics(obs, labels, replay)
    assert metrics["idtp"] == 4 and metrics["idf1"] == 1
    assert metrics["id_switches"] == 2


def test_overlap_candidates_do_not_themselves_count_as_false_merges():
    replay = replay_fixture(2)
    labels = truth(replay)
    obs = observations([1, 1])
    for i, frame in enumerate(labels["frames"]):
        frame["objects"].append({"identity": "person-B", "label": "person", "box": detection(14)["box"]})
        obs[i]["objects"].append({"track_id": 2, "label": "person", "box": detection(14)["box"]})
    metrics = identity_metrics(obs, labels, replay)
    assert metrics["idf1"] == 1 and metrics["false_merges"] == 0


def test_scoring_rejects_duplicate_predictions_and_repeated_frames():
    replay = replay_fixture(2)
    labels = truth(replay)
    obs = observations([1, 1])
    obs[0]["objects"].append(copy.deepcopy(obs[0]["objects"][0]))
    with pytest.raises(ValueError, match="unique track IDs"):
        identity_metrics(obs, labels, replay)
    obs = observations([1, 1])
    obs[1]["frame_index"] = 0
    with pytest.raises(ValueError, match="unique increasing"):
        identity_metrics(obs, labels, replay)


def test_labels_require_explicit_objects_and_selected_frame_coverage():
    replay = replay_fixture(2)
    labels = truth(replay)
    del labels["frames"][0]["objects"]
    with pytest.raises(ValueError, match="explicit objects"):
        validate_labels(labels, replay, [0, 1])
    labels = truth(replay)
    labels["frames"].pop()
    assert 0 in validate_labels(labels, replay, [0])
    with pytest.raises(ValueError, match="every evaluated frame"):
        validate_labels(labels, replay, [0, 1])


def test_replay_rejects_resolution_changes_and_missing_provenance():
    replay = replay_fixture(2)
    replay["frames"][1]["width"] = 240
    replay["replay_id"] = replay_digest(replay)
    with pytest.raises(ValueError, match="dimensions must remain constant"):
        validate_replay(replay)
    replay = replay_fixture(2)
    del replay["timestamp_source"]
    replay["replay_id"] = replay_digest(replay)
    with pytest.raises(ValueError, match="provenance"):
        validate_replay(replay)


def test_sampling_handles_long_recording_gap_without_iterating_empty_slots():
    replay = replay_fixture(3)
    replay["frames"][1]["captured_at"] = 1_000_000_100
    replay["frames"][2]["captured_at"] = 1_000_000_100.5
    replay["replay_id"] = replay_digest(replay)
    validate_replay(replay)
    assert [f["frame_index"] for f in selected_frames(replay, "fixed_2fps")] == [0, 1, 2]


def test_class_mismatch_and_true_absence_have_explicit_metric_effects():
    replay = replay_fixture(3)
    labels = truth(replay)
    obs = observations([1, None, 1])
    labels["frames"][1]["objects"] = []
    score = identity_metrics(obs, labels, replay)
    assert score["idf1"] == 1 and score["fragmentations"] == 0
    obs[0]["objects"][0]["label"] = "car"
    score = identity_metrics(obs, labels, replay)
    assert score["idtp"] == 1 and score["idfp"] == 1 and score["idfn"] == 1


@pytest.mark.parametrize("implementation", ["survng_hybrid_candidate", "ultralytics_tracktrack", "ultralytics_botsort"])
def test_evaluation_engines_cannot_be_selected_by_production_config(implementation):
    with pytest.raises(ValueError):
        ObjectTrackingConfig(implementation=implementation)


def test_sparse_identity_remains_selectable_for_production():
    assert ObjectTrackingConfig(implementation="survng_sparse_identity").implementation == "survng_sparse_identity"

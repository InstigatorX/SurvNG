from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from survng.app.face_evaluation import evaluate, metrics, validate_manifest


@pytest.fixture
def corpus(tmp_path):
    samples = []
    for index, (split, person) in enumerate([
        ("enrollment", "Alice"), ("calibration", "Alice"),
        ("calibration", None), ("held_out", "Alice"), ("held_out", None),
    ]):
        path = tmp_path / f"{index}.png"
        image = np.full((64, 64, 3), index + 1, np.uint8)
        assert cv2.imwrite(str(path), image)
        samples.append(dict(id=str(index), path=path.name, day=split,
                            visit=str(index), track="face-1", person=person, split=split))
    return {"version": 1, "samples": samples}


def recognizer(vectors):
    return SimpleNamespace(
        config=SimpleNamespace(face_min_size=32, face_embedding_profile="adaface",
                               face_match_threshold=.5),
        model_fingerprint="test-model", embed=lambda image: vectors[int(image[0, 0, 0])-1],
    )


def test_calibration_does_not_consult_held_out_labels_or_change_settings(corpus, tmp_path):
    engine = recognizer([[1., 0.], [.8, .6], [.6, .8], [.8, .6], [.6, .8]])
    report = evaluate(corpus, tmp_path, engine)
    assert .6 < report["calibrated_suggestion_threshold"] <= .8
    assert report["baseline_held_out"]["unknown_false_names"] == 1
    assert report["held_out"]["correct_names"] == 1
    assert report["held_out"]["wrong_names"] == 0
    assert report["held_out"]["unresolved"] == 1
    assert engine.config.face_match_threshold == .5
    changed = deepcopy(corpus)
    changed["samples"][3]["person"] = None
    changed["samples"][4]["person"] = "Alice"
    altered = evaluate(changed, tmp_path, engine)
    assert altered["calibrated_suggestion_threshold"] == report["calibrated_suggestion_threshold"]
    assert altered["held_out"]["wrong_names"] == 1
    assert altered["images_digest"] == report["images_digest"]
    assert altered["manifest_digest"] != report["manifest_digest"]


@pytest.mark.parametrize("field", ["day", "visit", "path"])
def test_split_leakage_is_rejected(corpus, tmp_path, field):
    corpus["samples"][1][field] = corpus["samples"][0][field]
    with pytest.raises(ValueError, match="must not cross"):
        validate_manifest(corpus, tmp_path)


def test_unknowns_must_be_labeled_explicitly(corpus, tmp_path):
    del corpus["samples"][2]["person"]
    with pytest.raises(ValueError, match="explicit"):
        validate_manifest(corpus, tmp_path)


def test_reencoded_duplicate_is_rejected(corpus, tmp_path):
    image = cv2.imread(str(tmp_path / "0.png"))
    assert cv2.imwrite(str(tmp_path / "duplicate.bmp"), image)
    corpus["samples"][1]["path"] = "duplicate.bmp"
    with pytest.raises(ValueError, match="must not cross"):
        validate_manifest(corpus, tmp_path)


def test_no_separating_threshold_produces_no_recommendation(corpus, tmp_path):
    report = evaluate(corpus, tmp_path, recognizer([[1., 0.]] * 5))
    assert report["calibrated_suggestion_threshold"] is None
    assert report["held_out"]["wrong_names"] == 1


def test_failed_probes_remain_in_coverage_denominator(corpus, tmp_path):
    engine = recognizer([[1., 0.], [.8, .6], [.6, .8], [0., 0.], [.6, .8]])
    report = evaluate(corpus, tmp_path, engine)
    assert report["failed_samples"] == [{"id": "3", "reason": "Invalid embedding"}]
    assert report["held_out"]["known_coverage"] == 0
    assert report["held_out"]["tracks"] == 2


def test_metrics_do_not_count_unknown_abstention_as_correct_identification():
    result = metrics([{"expected": None, "predicted": None},
                      {"expected": "Alice", "predicted": None}])
    assert result["precision"] is None
    assert result["correct_names"] == 0
    assert result["known_coverage"] == 0
    assert result["unknown_false_name_rate"] == 0

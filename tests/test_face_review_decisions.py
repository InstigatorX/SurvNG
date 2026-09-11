from __future__ import annotations

from unittest.mock import patch

import pytest

from tests import test_face_identity_reconciliation as identity_tests
from tests.test_face_reference_retention import CUTOFF, OLD, gallery, stores


@pytest.mark.parametrize("bulk", [False, True], ids=["single", "bulk"])
def test_manual_clear_survives_refresh_and_refinement_until_reassigned(bulk):
    case = identity_tests.FaceIdentityReconciliationTest()
    case.setUp()
    try:
        ids = case.candidates([case.alice, case.alice])
        for observation_id in ids:
            case.recognize(observation_id)
        canonical_id = case.canonical()["id"]
        case.store.assign(canonical_id, case.alice)
        assert case.canonical()["review_status"] == "confirmed"
        case.updates.clear()

        if bulk:
            case.store.bulk_review([canonical_id], action="unassign")
        else:
            result = case.store.assign(canonical_id, None)
            assert result["person_id"] is None
        case.store._refresh_unknown_recognition()
        row = case.canonical()
        assert row["id"] == canonical_id
        assert row["person_id"] is None
        assert row["review_status"] == "rejected"
        assert row["rejected_person_id"] == case.alice
        assert [(item["action"], item["current_person_id"]) for item in case.updates] == [("cleared", None)]
        assert case.updates[0]["source"] == ("manual_bulk" if bulk else "manual")
        assert case.updates[0]["previous_person_id"] == case.alice
        assert case.updates[0]["identity_status"] == "confirmed"

        # A later crop with the same strong evidence must not undo the clear.
        refined_ids = case.candidates([case.alice], first_rank=3)
        for observation_id in refined_ids:
            case.recognize(observation_id)
        case.store._refresh_unknown_recognition()
        assert case.canonical()["id"] == canonical_id
        assert case.canonical()["person_id"] is None
        assert case.canonical()["candidate_person_id"] is None
        assert len(case.updates) == 1

        if bulk:
            case.store.bulk_review([canonical_id], action="assign", person_id=case.alice)
        else:
            case.store.assign(canonical_id, case.alice)
        case.store._refresh_unknown_recognition()
        assert case.canonical()["person_id"] == case.alice
        assert case.canonical()["review_status"] == "confirmed"
        with case.store._connect() as connection:
            assert connection.execute(
                "select count(*) from face_rejections where observation_id = ? and person_id = ?",
                (canonical_id, case.alice),
            ).fetchone()[0] == 0
        assert [(item["action"], item["current_person_id"]) for item in case.updates] == [
            ("cleared", None), ("assigned", case.alice),
        ]
    finally:
        case.doCleanups()


@pytest.mark.parametrize("completed", [False, True], ids=["claimed", "deleted"])
def test_optimizer_uses_available_alternative_and_keeps_expired_held_out_sample(tmp_path, completed):
    events, faces = stores(tmp_path)
    person, ids = gallery(events, faces)
    preview = faces.optimize_person_gallery(person, max_references=2)
    target = next(item for item in preview["optimized_reference_ids"] if item != ids[0])
    with faces._connect() as connection:
        connection.execute(
            "update events set created_at = ? where id = (select event_id from face_observations where id = ?)",
            (OLD, target),
        )

    def optimize():
        result = faces.optimize_person_gallery(person, max_references=2, apply=True)
        assert result["applied"]
        assert result["sample_count"] == len(ids)
        assert result["baseline"]["trials"] == len(ids) - 1
        assert result["optimized"]["trials"] == len(ids) - 2
        assert target not in result["optimized_reference_ids"]
        assert len(result["optimized_reference_ids"]) == 2
        assert not faces.observation(target)["reference_pinned"]
        for observation_id in result["optimized_reference_ids"]:
            assert faces.observation(observation_id)["reference_pinned"]
            assert faces.observation(observation_id)["snapshot_path"]

    original = events._snapshot_path_for_retention

    def during_claim(raw_path):
        optimize()
        return original(raw_path)

    if completed:
        assert events.apply_snapshot_retention(CUTOFF, 10)["deleted_files"] == 1
        assert faces.observation(target)["snapshot_path"] == ""
        optimize()
    else:
        with patch.object(events, "_snapshot_path_for_retention", side_effect=during_claim):
            assert events.apply_snapshot_retention(CUTOFF, 10)["deleted_files"] == 1


def test_optimizer_without_available_media_reports_no_references(tmp_path):
    events, faces = stores(tmp_path)
    person, ids = gallery(events, faces)
    with faces._connect() as connection:
        connection.execute("update face_observations set reference_pinned = 0, reference_auto_pinned = 0")
        connection.execute("update events set created_at = ?", (OLD,))
    assert events.apply_snapshot_retention(CUTOFF, 10)["deleted_files"] == len(ids) + 1
    result = faces.optimize_person_gallery(person, max_references=2, apply=True)
    assert result["sample_count"] == len(ids)
    assert not result["applied"]
    assert result["reason"] == "no_available_references"

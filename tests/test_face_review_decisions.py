from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

import pytest

from tests import test_face_identity_reconciliation as identity_tests
from tests.test_face_reference_retention import CUTOFF, OLD, gallery, stores


@pytest.mark.parametrize("confirm", [False, True], ids=["reject", "confirm"])
def test_review_does_not_wait_for_entire_background_refresh(confirm):
    case = identity_tests.FaceIdentityReconciliationTest()
    case.setUp()
    try:
        case.recognizer.config.face_auto_identify_enabled = False
        ids = [case.candidates([case.alice], event_id=event)[0] for event in range(1, 41)]
        for observation_id in ids:
            case.recognize(observation_id)
        matching = Event()
        reviewed = Event()
        matched_after_review = []

        def slow_match(_connection, observation_id, *_args):
            matching.set()
            # Represent a large queue without depending on machine-specific
            # gallery throughput. The entire sweep takes about two seconds.
            time.sleep(0.05)
            if reviewed.is_set():
                matched_after_review.append(observation_id)
            return case.matches[observation_id]

        def review():
            result = case.store.assign(ids[0], case.alice if confirm else None)
            reviewed.set()
            return result

        with (
            patch.object(case.store, "_match_result", side_effect=slow_match),
            patch.object(case.store, "request_match_refresh"),
            patch.object(case.store, "bootstrap_person_references"),
            patch.object(case.store, "_queue_recognition"),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            refresh = executor.submit(case.store._refresh_unknown_recognition)
            assert matching.wait(2)
            decision = executor.submit(review)
            assert reviewed.wait(1), "review blocked behind the whole background sweep"
            assert not refresh.done()
            assert decision.result()["review_status"] == ("confirmed" if confirm else "rejected")
            refresh.result(timeout=10)
        assert matched_after_review
        assert case.canonical(1)["review_status"] == ("confirmed" if confirm else "rejected")
        if confirm:
            assert ids[0] not in matched_after_review
    finally:
        case.doCleanups()


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

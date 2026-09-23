from __future__ import annotations

from survng.app.person_identity import FusionPolicy, fuse_identity_match


def test_fuse_prefers_strong_face() -> None:
    decision = fuse_identity_match(
        face_person_id=1,
        face_score=0.82,
        face_runner_up=0.40,
        face_reference_ids=(10, 11, 12),
        face_reference_scores=(0.84, 0.81, 0.80),
        body_person_id=None,
        body_score=None,
        body_runner_up=None,
        quality_score=0.7,
        auto_identify_enabled=True,
    )
    assert decision.decision == "auto"
    assert decision.match.modality == "face"
    assert decision.match.person_id == 1


def test_fuse_body_only_suggests_never_autos() -> None:
    decision = fuse_identity_match(
        face_person_id=None,
        face_score=None,
        face_runner_up=None,
        body_person_id=7,
        body_score=0.88,
        body_runner_up=0.50,
        body_reference_ids=(21, 22),
        body_reference_scores=(0.90, 0.86),
        quality_score=0.8,
        auto_identify_enabled=True,
        policy=FusionPolicy(body_suggest_threshold=0.70),
    )
    assert decision.decision == "suggest"
    assert decision.reason == "body_only"
    assert decision.match.modality == "body"
    assert decision.match.person_id == 7


def test_fuse_agreement_can_auto_identify() -> None:
    decision = fuse_identity_match(
        face_person_id=3,
        face_score=0.70,
        face_runner_up=0.40,
        face_reference_ids=(1, 2, 3),
        face_reference_scores=(0.72, 0.70, 0.68),
        body_person_id=3,
        body_score=0.80,
        body_runner_up=0.45,
        body_reference_ids=(4, 5),
        body_reference_scores=(0.82, 0.78),
        quality_score=0.6,
        auto_identify_enabled=True,
    )
    assert decision.match.modality == "fused"
    assert decision.decision == "auto"
    assert decision.match.person_id == 3


def test_fuse_blocks_close_disagreement() -> None:
    decision = fuse_identity_match(
        face_person_id=1,
        face_score=0.74,
        face_runner_up=0.20,
        face_reference_ids=(1, 2, 3),
        face_reference_scores=(0.75, 0.74, 0.73),
        body_person_id=2,
        body_score=0.76,
        body_runner_up=0.20,
        body_reference_ids=(8, 9),
        body_reference_scores=(0.77, 0.75),
        quality_score=0.7,
        auto_identify_enabled=True,
    )
    assert decision.decision == "none"
    assert decision.reason == "face_body_disagreement"
    assert decision.match.person_id is None


def test_fuse_prefers_clearer_face_when_modalities_disagree() -> None:
    decision = fuse_identity_match(
        face_person_id=1,
        face_score=0.90,
        face_runner_up=0.20,
        face_reference_ids=(1, 2, 3),
        face_reference_scores=(0.91, 0.90, 0.89),
        body_person_id=2,
        body_score=0.71,
        body_runner_up=0.20,
        body_reference_ids=(8, 9),
        body_reference_scores=(0.72, 0.70),
        quality_score=0.7,
        auto_identify_enabled=False,
    )
    assert decision.decision == "suggest"
    assert decision.match.modality == "face"
    assert decision.match.person_id == 1

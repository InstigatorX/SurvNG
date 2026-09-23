"""Face + body fusion policy for named people matching.

Design rules:
- Strong face evidence can suggest or auto-identify.
- Body evidence can suggest or reinforce, but cannot auto-identify alone.
- Face/body disagreement blocks automatic assignment.
- Clothing-heavy body matches stay reviewable suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Modality = Literal["none", "face", "body", "fused"]
Decision = Literal["none", "suggest", "auto"]


@dataclass(frozen=True, slots=True)
class FusionPolicy:
    """Thresholds for fused people identity matching."""

    face_suggest_threshold: float = 0.30
    face_auto_threshold: float = 0.55
    face_auto_margin: float = 0.12
    body_suggest_threshold: float = 0.70
    body_reinforce_threshold: float = 0.75
    fused_suggest_threshold: float = 0.45
    fused_auto_threshold: float = 0.62
    fused_auto_margin: float = 0.10
    disagreement_gap: float = 0.12
    face_weight: float = 0.65
    body_weight: float = 0.35
    min_face_refs_for_auto: int = 3
    min_body_refs_for_suggest: int = 2
    min_quality_for_auto: float = 0.45


@dataclass(frozen=True, slots=True)
class FusionMatch:
    person_id: int | None
    score: float | None
    runner_up_score: float | None
    margin: float | None
    face_score: float | None
    body_score: float | None
    modality: Modality
    face_reference_ids: tuple[int, ...]
    body_reference_ids: tuple[int, ...]
    face_reference_scores: tuple[float, ...]
    body_reference_scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FusionDecision:
    match: FusionMatch
    decision: Decision
    reason: str


def _clip(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, min(1.0, float(value))), 4)


def fuse_identity_match(
    *,
    face_person_id: int | None,
    face_score: float | None,
    face_runner_up: float | None,
    face_reference_ids: tuple[int, ...] = (),
    face_reference_scores: tuple[float, ...] = (),
    body_person_id: int | None,
    body_score: float | None,
    body_runner_up: float | None,
    body_reference_ids: tuple[int, ...] = (),
    body_reference_scores: tuple[float, ...] = (),
    quality_score: float = 0.0,
    auto_identify_enabled: bool = False,
    policy: FusionPolicy | None = None,
) -> FusionDecision:
    """Combine independent face/body rankings into one identity decision."""
    policy = policy or FusionPolicy()
    face_score = _clip(face_score)
    body_score = _clip(body_score)
    face_runner_up = _clip(face_runner_up)
    body_runner_up = _clip(body_runner_up)

    face_ok = (
        face_person_id is not None
        and face_score is not None
        and face_score >= policy.face_suggest_threshold
    )
    body_ok = (
        body_person_id is not None
        and body_score is not None
        and body_score >= policy.body_suggest_threshold
        and len(body_reference_ids) >= policy.min_body_refs_for_suggest
    )

    # Disagreement: face and body name different people with meaningful scores.
    if (
        face_ok
        and body_ok
        and face_person_id != body_person_id
        and abs(float(face_score) - float(body_score)) < policy.disagreement_gap
    ):
        empty = FusionMatch(
            None,
            max(face_score or 0.0, body_score or 0.0),
            None,
            None,
            face_score,
            body_score,
            "none",
            face_reference_ids,
            body_reference_ids,
            face_reference_scores,
            body_reference_scores,
        )
        return FusionDecision(empty, "none", "face_body_disagreement")

    if face_ok and body_ok and face_person_id == body_person_id:
        fused = (
            policy.face_weight * float(face_score)
            + policy.body_weight * float(body_score)
        )
        runner_up_candidates = [
            value
            for value in (face_runner_up, body_runner_up)
            if value is not None
        ]
        runner_up = max(runner_up_candidates) if runner_up_candidates else None
        margin = fused - runner_up if runner_up is not None else fused
        match = FusionMatch(
            int(face_person_id),
            _clip(fused),
            runner_up,
            _clip(margin),
            face_score,
            body_score,
            "fused",
            face_reference_ids,
            body_reference_ids,
            face_reference_scores,
            body_reference_scores,
        )
        if (
            auto_identify_enabled
            and fused >= policy.fused_auto_threshold
            and margin is not None
            and margin >= policy.fused_auto_margin
            and len(face_reference_ids) >= policy.min_face_refs_for_auto
            and float(body_score) >= policy.body_reinforce_threshold
            and quality_score >= policy.min_quality_for_auto
        ):
            return FusionDecision(match, "auto", "fused_agreement")
        if fused >= policy.fused_suggest_threshold:
            return FusionDecision(match, "suggest", "fused_agreement")
        return FusionDecision(
            FusionMatch(
                None,
                match.score,
                match.runner_up_score,
                match.margin,
                face_score,
                body_score,
                "fused",
                face_reference_ids,
                body_reference_ids,
                face_reference_scores,
                body_reference_scores,
            ),
            "none",
            "fused_below_threshold",
        )

    # Prefer face when only one modality wins, or when face clearly dominates.
    if face_ok and (not body_ok or face_person_id == body_person_id or float(face_score) >= float(body_score or 0.0)):
        face_margin = (
            float(face_score) - float(face_runner_up)
            if face_runner_up is not None
            else float(face_score)
        )
        match = FusionMatch(
            int(face_person_id),
            face_score,
            face_runner_up,
            _clip(face_margin),
            face_score,
            body_score,
            "face",
            face_reference_ids,
            body_reference_ids,
            face_reference_scores,
            body_reference_scores,
        )
        if (
            auto_identify_enabled
            and float(face_score) >= policy.face_auto_threshold
            and face_margin >= policy.face_auto_margin
            and len(face_reference_ids) >= policy.min_face_refs_for_auto
            and quality_score >= policy.min_quality_for_auto
        ):
            return FusionDecision(match, "auto", "strong_face")
        return FusionDecision(match, "suggest", "face_only")

    if body_ok:
        body_margin = (
            float(body_score) - float(body_runner_up)
            if body_runner_up is not None
            else float(body_score)
        )
        match = FusionMatch(
            int(body_person_id),
            body_score,
            body_runner_up,
            _clip(body_margin),
            face_score,
            body_score,
            "body",
            face_reference_ids,
            body_reference_ids,
            face_reference_scores,
            body_reference_scores,
        )
        # Body-only never auto-identifies: clothing changes and lookalikes.
        return FusionDecision(match, "suggest", "body_only")

    return FusionDecision(
        FusionMatch(
            None,
            face_score if face_score is not None else body_score,
            None,
            None,
            face_score,
            body_score,
            "none",
            face_reference_ids,
            body_reference_ids,
            face_reference_scores,
            body_reference_scores,
        ),
        "none",
        "no_match",
    )

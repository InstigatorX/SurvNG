from __future__ import annotations

from survng.app.active_motion_followup import (
    ActiveMotionFollowupAction,
    ActiveMotionFollowupPolicy,
)
from survng.app.motion import MotionQualificationResult


def _result(
    *,
    event_key: str = "gate:100000",
    transition: str,
    track_id: int,
    region: tuple[float, float, float, float],
    score: float = 0.8,
) -> MotionQualificationResult:
    return MotionQualificationResult(
        accepted=transition == "activation_threshold",
        score=score,
        threshold=0.48,
        reason="qualified" if transition == "activation_threshold" else "event_state_active",
        frame_count=3,
        features={
            "event_state_phase": "active",
            "event_state_key": event_key,
            "event_state_transition": transition,
            "motion_region_track_id": track_id,
            "motion_regions": [list(region)],
        },
    )


def test_distinct_credible_track_becomes_one_followup_candidate() -> None:
    policy = ActiveMotionFollowupPolicy()
    baseline = policy.consider(
        _result(
            transition="activation_threshold",
            track_id=4,
            region=(0.1, 0.05, 0.3, 0.18),
        ),
        100.0,
        credible_motion=True,
    )
    candidate_result = _result(
        transition="active_confirmed",
        track_id=9,
        region=(0.55, 0.55, 0.85, 0.9),
    )
    candidate = policy.consider(
        candidate_result,
        103.0,
        credible_motion=True,
    )

    assert baseline.action is ActiveMotionFollowupAction.BASELINE
    assert candidate.action is ActiveMotionFollowupAction.CANDIDATE
    assert candidate.anchor_index == 1
    assert policy.commit(
        candidate,
        103.0,
        candidate_result.features["motion_regions"],
    )
    duplicate = policy.consider(
        candidate_result,
        104.0,
        credible_motion=True,
    )
    assert duplicate.action is ActiveMotionFollowupAction.DUPLICATE


def test_tracker_id_churn_in_same_area_is_deduplicated() -> None:
    policy = ActiveMotionFollowupPolicy()
    policy.consider(
        _result(
            transition="activation_threshold",
            track_id=4,
            region=(0.1, 0.05, 0.3, 0.18),
        ),
        100.0,
        credible_motion=True,
    )

    decision = policy.consider(
        _result(
            transition="active_confirmed",
            track_id=5,
            region=(0.12, 0.06, 0.31, 0.19),
        ),
        103.0,
        credible_motion=True,
    )

    assert decision.action is ActiveMotionFollowupAction.DUPLICATE


def test_followups_are_bounded_per_episode_and_reset_for_next_event() -> None:
    policy = ActiveMotionFollowupPolicy(maximum_anchors=2)
    policy.consider(
        _result(
            transition="activation_threshold",
            track_id=1,
            region=(0.0, 0.0, 0.1, 0.1),
        ),
        100.0,
        credible_motion=True,
    )
    for timestamp, track_id, region in (
        (103.0, 2, (0.3, 0.3, 0.4, 0.4)),
        (106.0, 3, (0.6, 0.6, 0.7, 0.7)),
    ):
        result = _result(
            transition="active_confirmed",
            track_id=track_id,
            region=region,
        )
        decision = policy.consider(result, timestamp, credible_motion=True)
        assert decision.admitted
        assert policy.commit(decision, timestamp, result.features["motion_regions"])

    limited = policy.consider(
        _result(
            transition="active_confirmed",
            track_id=4,
            region=(0.85, 0.1, 0.95, 0.2),
        ),
        109.0,
        credible_motion=True,
    )
    assert limited.action is ActiveMotionFollowupAction.EPISODE_LIMIT

    reset = policy.consider(
        _result(
            event_key="gate:200000",
            transition="activation_threshold",
            track_id=4,
            region=(0.85, 0.1, 0.95, 0.2),
        ),
        200.0,
        credible_motion=True,
    )
    assert reset.action is ActiveMotionFollowupAction.BASELINE


def test_rejected_or_global_change_motion_cannot_create_followup() -> None:
    policy = ActiveMotionFollowupPolicy()
    policy.consider(
        _result(
            transition="activation_threshold",
            track_id=1,
            region=(0.0, 0.0, 0.1, 0.1),
        ),
        100.0,
        credible_motion=True,
    )

    decision = policy.consider(
        _result(
            transition="active_confirmed",
            track_id=2,
            region=(0.6, 0.6, 0.8, 0.8),
            score=0.2,
        ),
        103.0,
        credible_motion=False,
    )

    assert decision.action is ActiveMotionFollowupAction.NOT_CREDIBLE

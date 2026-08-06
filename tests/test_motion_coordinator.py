from __future__ import annotations

from typing import Any

from survng.app.motion import MotionQualificationResult
from survng.app.motion_coordinator import (
    VisualBackupAction,
    VisualBackupCoordinator,
    VisualBackupPolicy,
    VisualBackupReplaySample,
    replay_visual_backup,
)


def _policy(**overrides: Any) -> VisualBackupPolicy:
    values: dict[str, Any] = {
        "warmup_seconds": 0.0,
        "grace_seconds": 1.0,
        "minimum_score": 0.7,
        "score_margin": 0.1,
        "minimum_consecutive": 3,
        "cooldown_seconds": 15.0,
        "maximum_triggers_5m": 2,
        "sample_fps": 2.0,
        "background_fps": 2.0,
    }
    values.update(overrides)
    return VisualBackupPolicy(**values)


def _result(
    *,
    accepted: bool,
    score: float,
    reason: str,
    threshold: float = 0.5,
) -> MotionQualificationResult:
    return MotionQualificationResult(
        accepted=accepted,
        score=score,
        threshold=threshold,
        reason=reason,
        frame_count=4,
        features={},
        telemetry={},
    )


def _stable(at: float) -> VisualBackupReplaySample:
    return VisualBackupReplaySample(
        captured_at=at,
        result=_result(accepted=False, score=0.0, reason="no_motion_blobs"),
    )


def _credible(at: float, **kwargs: Any) -> VisualBackupReplaySample:
    return VisualBackupReplaySample(
        captured_at=at,
        result=_result(accepted=True, score=0.82, reason="credible_motion"),
        **kwargs,
    )


def test_replay_promotes_only_after_scene_readiness_and_persistence() -> None:
    decisions = replay_visual_backup(
        _policy(),
        (
            _stable(90.0),
            _stable(90.75),
            _stable(91.5),
            _credible(100.0),
            _credible(100.5),
            _credible(101.0),
        ),
    )

    assert [decision.action for decision in decisions[-3:]] == [
        VisualBackupAction.ACCUMULATING,
        VisualBackupAction.ACCUMULATING,
        VisualBackupAction.READY,
    ]
    assert decisions[-1].consecutive == 3
    assert decisions[-1].required_score == 0.7


def test_replay_is_deterministic_and_does_not_mutate_inputs() -> None:
    samples = (
        _stable(90.0),
        _stable(90.75),
        _stable(91.5),
        _credible(100.0),
        _credible(100.5),
        _credible(101.0),
    )

    assert replay_visual_backup(_policy(), samples) == replay_visual_backup(
        _policy(), samples
    )


def test_replay_timeline_may_begin_at_zero() -> None:
    decisions = replay_visual_backup(
        _policy(),
        (
            _stable(0.0),
            _stable(0.75),
            _stable(1.5),
            _credible(2.0),
            _credible(2.5),
            _credible(3.0),
        ),
    )

    assert decisions[-1].action == VisualBackupAction.READY


def test_candidate_gap_resets_persistence_sequence() -> None:
    decisions = replay_visual_backup(
        _policy(),
        (
            _stable(90.0),
            _stable(90.75),
            _stable(91.5),
            _credible(100.0),
            _credible(102.0),
            _credible(102.5),
        ),
    )

    assert decisions[-1].action == VisualBackupAction.ACCUMULATING
    assert decisions[-1].consecutive == 2


def test_recent_camera_notice_suppresses_ready_visual_backup_once() -> None:
    samples = (
        _stable(90.0),
        _stable(90.75),
        _stable(91.5),
        _credible(100.0, camera_motion_times=(99.0,)),
        _credible(100.5, camera_motion_times=(99.0,)),
        _credible(101.0, camera_motion_times=(99.0,)),
        _credible(102.0, camera_motion_times=(99.0,)),
        _credible(102.5, camera_motion_times=(99.0,)),
        _credible(103.0, camera_motion_times=(99.0,)),
    )
    decisions = replay_visual_backup(_policy(), samples)

    assert decisions[5].action == VisualBackupAction.CAMERA_NOTICE
    assert decisions[5].new_camera_match is True
    assert decisions[-1].action == VisualBackupAction.CAMERA_NOTICE
    assert decisions[-1].new_camera_match is False


def test_known_noise_never_accumulates_as_visual_backup() -> None:
    noise = VisualBackupReplaySample(
        captured_at=100.0,
        result=_result(
            accepted=True,
            score=0.95,
            reason="stationary_foreground",
        ),
    )
    decisions = replay_visual_backup(
        _policy(),
        (_stable(90.0), _stable(90.75), _stable(91.5), noise),
    )

    assert decisions[-1].action == VisualBackupAction.IGNORED
    assert decisions[-1].count_nonpromotion is True


def test_disabled_detection_clears_partial_candidate() -> None:
    coordinator = VisualBackupCoordinator()
    policy = _policy()
    stable = _result(accepted=False, score=0.0, reason="no_motion_blobs")
    credible = _result(accepted=True, score=0.82, reason="credible_motion")
    for captured_at in (90.0, 90.75, 91.5):
        coordinator.evaluate(
            stable,
            captured_at,
            policy,
            detection_enabled=True,
        )
    assert coordinator.evaluate(
        credible,
        100.0,
        policy,
        detection_enabled=True,
    ).action == VisualBackupAction.ACCUMULATING

    decision = coordinator.evaluate(
        credible,
        100.5,
        policy,
        detection_enabled=False,
    )

    assert decision.action == VisualBackupAction.DISABLED
    assert coordinator.consecutive == 0


def test_trigger_reservation_honors_cooldown_pending_and_rolling_limit() -> None:
    coordinator = VisualBackupCoordinator()
    policy = _policy()

    assert coordinator.reserve_trigger(
        100.0,
        policy,
        trigger_pending=False,
        last_completed_at=0.0,
    )
    assert not coordinator.reserve_trigger(
        100.0,
        policy,
        trigger_pending=True,
        last_completed_at=0.0,
    )
    assert not coordinator.reserve_trigger(
        110.0,
        policy,
        trigger_pending=False,
        last_completed_at=100.0,
    )
    coordinator.record_trigger(100.0)
    coordinator.record_trigger(120.0)
    assert not coordinator.reserve_trigger(
        140.0,
        policy,
        trigger_pending=False,
        last_completed_at=0.0,
    )
    assert coordinator.reserve_trigger(
        421.0,
        policy,
        trigger_pending=False,
        last_completed_at=0.0,
    )

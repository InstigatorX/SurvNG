from __future__ import annotations

from survng.app.ema_v2 import (
    CameraNotice,
    EmaSignalAction,
    EmaSignalConditioner,
    EpisodeDecisionReason,
    MotionEpisodeController,
    EmaPolicy,
)
from survng.app.motion import MotionQualificationResult


def _policy(**changes: object) -> EmaPolicy:
    values = {
        "warmup_seconds": 1.0,
        "grace_seconds": 1.0,
        "minimum_score": 0.65,
        "score_margin": 0.1,
        "minimum_consecutive": 3,
        "cooldown_seconds": 15.0,
        "maximum_triggers_5m": 2,
        "sample_fps": 2.0,
        "background_fps": 2.0,
    }
    values.update(changes)
    return EmaPolicy(**values)


def _result(score: float, *, accepted: bool = True, reason: str = "credible_motion") -> MotionQualificationResult:
    return MotionQualificationResult(
        accepted=accepted,
        score=score,
        threshold=0.48,
        reason=reason,
        frame_count=3,
        features={},
        telemetry={},
    )


def _qualified(camera: str = "gate"):
    conditioner = EmaSignalConditioner(camera)
    policy = _policy(warmup_seconds=0.0)
    for at in (1.0, 1.5, 2.0):
        conditioner.evaluate(
            _result(0.0, accepted=False, reason="no_motion_blobs"),
            at,
            at + 1000.0,
            policy,
            detection_enabled=True,
        )
    decisions = [
        conditioner.evaluate(_result(0.8), at, at + 1000.0, policy, detection_enabled=True)
        for at in (10.0, 10.5, 11.0)
    ]
    assert decisions[-1].qualified is not None
    return decisions[-1].qualified


def test_busy_scene_readiness_is_not_coupled_to_rejected_motion() -> None:
    conditioner = EmaSignalConditioner("gate")
    policy = _policy()
    decisions = [
        conditioner.evaluate(_result(0.9), at, at, policy, detection_enabled=True)
        for at in (0.0, 0.5, 1.0)
    ]

    assert decisions[0].action is EmaSignalAction.LEARNING
    assert decisions[-1].scene_ready is True


def test_one_borderline_dropout_does_not_erase_persistent_motion() -> None:
    conditioner = EmaSignalConditioner("gate")
    policy = _policy(warmup_seconds=0.0)
    for at in (1.0, 1.5, 2.0):
        conditioner.evaluate(
            _result(0.0, accepted=False, reason="no_motion_blobs"),
            at,
            at,
            policy,
            detection_enabled=True,
        )
    decisions = [
        conditioner.evaluate(_result(score), at, at, policy, detection_enabled=True)
        for at, score in ((10.0, 0.8), (10.5, 0.60), (11.0, 0.82), (11.5, 0.84))
    ]

    assert decisions[-1].action is EmaSignalAction.QUALIFIED
    assert decisions[-1].qualifying_samples == 3
    assert decisions[-1].window_samples == 4


def test_known_nuisance_cannot_accumulate() -> None:
    conditioner = EmaSignalConditioner("gate")
    policy = _policy(warmup_seconds=0.0)
    decisions = [
        conditioner.evaluate(
            _result(0.95, reason="stationary_foreground"),
            at,
            at,
            policy,
            detection_enabled=True,
        )
        for at in (1.0, 1.5, 2.0, 2.5)
    ]

    assert all(item.action is not EmaSignalAction.QUALIFIED for item in decisions)


def test_raw_camera_notice_does_not_hide_ema_after_enqueue_failure() -> None:
    controller = MotionEpisodeController("gate")
    controller.start_generation(4)
    camera = controller.observe_camera(
        CameraNotice("gate", 10.0, 1010.0, "RuleEngine/CellMotionDetector/Motion"),
        generation=4,
    )
    assert camera.reason is EpisodeDecisionReason.REQUEST_RESERVED
    assert camera.intent is not None
    controller.acknowledge_admission(
        camera.intent.intent_id,
        admitted=False,
        occurred_monotonic=1010.01,
    )

    ema = controller.observe_ema(_qualified("gate"), generation=4)

    assert ema.reason is EpisodeDecisionReason.REQUEST_RESERVED
    assert ema.intent is not None
    assert set(ema.intent.sources) == {"camera", "ema"}


def test_camera_and_ema_merge_only_after_request_is_admitted() -> None:
    controller = MotionEpisodeController("gate")
    controller.start_generation(8)
    ema = controller.observe_ema(_qualified("gate"), generation=8)
    assert ema.intent is not None
    controller.acknowledge_admission(
        ema.intent.intent_id,
        admitted=True,
        occurred_monotonic=1011.1,
    )

    camera = controller.observe_camera(
        CameraNotice("gate", 11.2, 1011.2, "motion"),
        generation=8,
    )

    assert camera.reason is EpisodeDecisionReason.MERGED_WITH_REQUEST
    assert camera.intent is not None
    assert set(camera.intent.sources) == {"camera", "ema"}


def test_stale_generation_cannot_reserve_work() -> None:
    controller = MotionEpisodeController("gate")
    controller.start_generation(2)

    decision = controller.observe_ema(_qualified("gate"), generation=1)

    assert decision.reason is EpisodeDecisionReason.STALE_GENERATION
    assert decision.intent is None


def test_cooldown_starts_only_after_completed_detection() -> None:
    controller = MotionEpisodeController("gate", cooldown_seconds=10.0)
    controller.start_generation(1)
    first = controller.observe_ema(_qualified("gate"), generation=1)
    assert first.intent is not None
    controller.acknowledge_admission(
        first.intent.intent_id,
        admitted=True,
        occurred_monotonic=1011.1,
    )
    controller.complete(first.intent.intent_id, occurred_monotonic=1012.0)

    during = controller.observe_camera(
        CameraNotice("gate", 13.0, 1013.0, "motion"),
        generation=1,
    )

    # It remains the same completed request, so the observation merges rather
    # than manufacturing another request or treating raw notice as completion.
    assert during.reason is EpisodeDecisionReason.MERGED_WITH_REQUEST


def test_episode_transitions_explain_every_request_boundary() -> None:
    controller = MotionEpisodeController("gate")
    controller.start_generation(3)
    decision = controller.observe_ema(_qualified("gate"), generation=3)
    assert decision.intent is not None
    controller.acknowledge_admission(
        decision.intent.intent_id,
        admitted=True,
        occurred_monotonic=1011.1,
    )
    controller.mark_running(decision.intent.intent_id, occurred_monotonic=1011.2)
    controller.complete(decision.intent.intent_id, occurred_monotonic=1012.0)

    assert [item.reason for item in controller.transitions()] == [
        EpisodeDecisionReason.REQUEST_RESERVED,
        EpisodeDecisionReason.REQUEST_ADMITTED,
        EpisodeDecisionReason.REQUEST_RUNNING,
        EpisodeDecisionReason.REQUEST_COMPLETED,
    ]

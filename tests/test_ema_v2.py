from __future__ import annotations

from dataclasses import replace

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


def _qualified_region(
    *,
    camera: str = "gate",
    track_id: int,
    region: tuple[float, float, float, float],
    started_at: float,
):
    result = _result(0.8)
    result = MotionQualificationResult(
        accepted=result.accepted,
        score=result.score,
        threshold=result.threshold,
        reason=result.reason,
        frame_count=result.frame_count,
        features={
            "motion_region_track_id": track_id,
            "motion_regions": [list(region)],
        },
        telemetry={},
    )
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
        conditioner.evaluate(result, at, at + 1000.0, policy, detection_enabled=True)
        for at in (started_at, started_at + 0.5, started_at + 1.0)
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


def test_long_grace_period_retains_enough_samples_to_qualify() -> None:
    conditioner = EmaSignalConditioner("gate")
    policy = _policy(
        warmup_seconds=0.0,
        grace_seconds=5.0,
        minimum_consecutive=3,
    )
    for at in (1.0, 1.5, 2.0):
        conditioner.evaluate(
            _result(0.0, accepted=False, reason="no_motion_blobs"),
            at,
            at,
            policy,
            detection_enabled=True,
        )

    decisions = [
        conditioner.evaluate(_result(0.8), at, at, policy, detection_enabled=True)
        for at in (10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0)
    ]

    assert decisions[-1].action is EmaSignalAction.QUALIFIED
    assert decisions[-1].qualifying_samples == 11


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
    assert set(ema.intent.sources) == {"ema"}
    snapshot = controller.snapshot()
    assert set(snapshot["sources"]) == {"camera", "ema"}
    assert snapshot["admitted_sources"] == ()


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
    assert controller.intent(ema.intent.intent_id) == camera.intent


def test_route_security_verification_is_not_dropped_by_cooldown_or_rate_limit() -> None:
    controller = MotionEpisodeController(
        "gate",
        episode_gap_seconds=5.0,
        cooldown_seconds=60.0,
        maximum_ema_requests_5m=1,
    )
    controller.start_generation(1)
    first = controller.observe_ema(_qualified("gate"), generation=1)
    assert first.intent is not None
    controller.acknowledge_admission(
        first.intent.intent_id,
        admitted=True,
        occurred_monotonic=1011.1,
    )
    controller.complete(first.intent.intent_id, occurred_monotonic=1011.2)
    verified = _qualified("gate")
    verified = replace(
        verified,
        captured_at=20.0,
        observed_monotonic=1020.0,
        result=MotionQualificationResult(
            accepted=True,
            score=0.55,
            threshold=0.48,
            reason="credible_motion",
            frame_count=3,
            features={
                "security_verification": True,
                "security_verification_bypass_limits": True,
            },
            telemetry={},
        ),
    )

    second = controller.observe_ema(verified, generation=1)

    assert second.reason is EpisodeDecisionReason.REQUEST_RESERVED
    assert second.intent is not None
    assert second.intent.intent_id != first.intent.intent_id


def test_persistent_security_verification_still_obeys_configured_budget() -> None:
    controller = MotionEpisodeController(
        "gate",
        episode_gap_seconds=5.0,
        cooldown_seconds=60.0,
        maximum_ema_requests_5m=1,
    )
    controller.start_generation(1)
    first = controller.observe_ema(_qualified("gate"), generation=1)
    assert first.intent is not None
    controller.acknowledge_admission(
        first.intent.intent_id,
        admitted=True,
        occurred_monotonic=1011.1,
    )
    controller.complete(first.intent.intent_id, occurred_monotonic=1011.2)
    verified = _qualified("gate")
    verified = replace(
        verified,
        captured_at=20.0,
        observed_monotonic=1020.0,
        result=replace(
            verified.result,
            features={"security_verification": True},
        ),
    )

    second = controller.observe_ema(verified, generation=1)

    assert second.reason is EpisodeDecisionReason.COOLDOWN_ACTIVE
    assert second.intent is None


def test_in_flight_intent_cannot_be_orphaned_by_episode_rollover() -> None:
    controller = MotionEpisodeController("gate", episode_gap_seconds=30.0)
    controller.start_generation(8)
    first = controller.observe_ema(_qualified("gate"), generation=8)
    assert first.intent is not None
    controller.acknowledge_admission(
        first.intent.intent_id, admitted=True, occurred_monotonic=1011.1
    )
    controller.mark_running(first.intent.intent_id, occurred_monotonic=1011.2)

    later = controller.observe_camera(
        CameraNotice("gate", 60.0, 1060.0, "motion"), generation=8
    )

    assert later.reason is EpisodeDecisionReason.MERGED_WITH_REQUEST
    assert later.intent is not None
    assert later.intent.intent_id == first.intent.intent_id
    controller.complete(first.intent.intent_id, occurred_monotonic=1061.0)


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


def test_completed_camera_request_cools_down_only_ema_rescue() -> None:
    controller = MotionEpisodeController(
        "gate", episode_gap_seconds=0.5, cooldown_seconds=10.0
    )
    controller.start_generation(1)
    first = controller.observe_camera(
        CameraNotice("gate", 10.0, 1010.0, "motion"), generation=1
    )
    assert first.intent is not None
    controller.acknowledge_admission(
        first.intent.intent_id, admitted=True, occurred_monotonic=1010.1
    )
    controller.complete(first.intent.intent_id, occurred_monotonic=1010.2)

    ema = controller.observe_ema(
        replace(_qualified("gate"), observed_monotonic=1011.0), generation=1
    )
    camera = controller.observe_camera(
        CameraNotice("gate", 11.1, 1011.1, "motion"), generation=1
    )

    assert ema.reason is EpisodeDecisionReason.COOLDOWN_ACTIVE
    assert camera.reason is EpisodeDecisionReason.REQUEST_RESERVED


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
    assert controller.snapshot()["decision_counts"] == {
        reason.value: (1 if reason in {
            EpisodeDecisionReason.REQUEST_RESERVED,
            EpisodeDecisionReason.REQUEST_ADMITTED,
            EpisodeDecisionReason.REQUEST_RUNNING,
            EpisodeDecisionReason.REQUEST_COMPLETED,
        } else 0)
        for reason in EpisodeDecisionReason
    }


def test_controller_replacement_namespaces_same_generation_and_sequence() -> None:
    first_controller = MotionEpisodeController("gate", incarnation_id="runtime-a")
    replacement = MotionEpisodeController("gate", incarnation_id="runtime-b")
    first_controller.start_generation(1)
    replacement.start_generation(1)

    first = first_controller.observe_ema(_qualified("gate"), generation=1)
    second = replacement.observe_ema(_qualified("gate"), generation=1)

    assert first.intent is not None
    assert second.intent is not None
    assert first.intent.episode_id == "gate:iruntime-a:g1:e1"
    assert second.intent.episode_id == "gate:iruntime-b:g1:e1"
    assert first.intent.intent_id != second.intent.intent_id
    assert first_controller.snapshot()["incarnation_id"] == "runtime-a"
    assert replacement.snapshot()["incarnation_id"] == "runtime-b"


def test_controller_incarnation_is_unique_by_default() -> None:
    first = MotionEpisodeController("gate")
    replacement = MotionEpisodeController("gate")

    assert first.incarnation_id != replacement.incarnation_id


def test_terminal_detector_failure_is_distinct_from_policy_abort() -> None:
    controller = MotionEpisodeController("gate")
    decision = controller.observe_ema(_qualified("gate"), generation=0)
    assert decision.intent is not None
    controller.acknowledge_admission(
        decision.intent.intent_id, admitted=True, occurred_monotonic=1011.1
    )
    controller.mark_running(decision.intent.intent_id, occurred_monotonic=1011.2)

    failed = controller.fail(
        decision.intent.intent_id, occurred_monotonic=1012.0
    )

    assert failed.reason is EpisodeDecisionReason.DETECTOR_FAILED
    snapshot = controller.snapshot()
    assert snapshot["request_status"] == "failed"
    assert snapshot["decision_counts"]["detector_failed"] == 1


def test_distinct_later_track_reserves_bounded_followup_in_same_episode() -> None:
    controller = MotionEpisodeController("gate", minimum_followup_interval_seconds=1.0)
    controller.start_generation(1)
    first = controller.observe_ema(
        _qualified_region(track_id=1, region=(0.05, 0.1, 0.2, 0.4), started_at=10.0),
        generation=1,
    )
    assert first.intent is not None
    controller.acknowledge_admission(
        first.intent.intent_id, admitted=True, occurred_monotonic=1011.1
    )
    controller.complete(first.intent.intent_id, occurred_monotonic=1012.0)

    followup = controller.observe_ema(
        _qualified_region(track_id=2, region=(0.70, 0.1, 0.9, 0.5), started_at=14.0),
        generation=1,
    )

    assert followup.reason is EpisodeDecisionReason.FOLLOWUP_RESERVED
    assert followup.intent is not None
    assert followup.intent.episode_id == first.intent.episode_id
    assert followup.intent.intent_id.endswith(":request:2")


def test_aborted_followup_refunds_followup_budget() -> None:
    controller = MotionEpisodeController(
        "gate", maximum_followups=1, minimum_followup_interval_seconds=0.0
    )
    controller.start_generation(1)
    first = controller.observe_ema(
        _qualified_region(track_id=1, region=(0.05, 0.1, 0.2, 0.4), started_at=10.0),
        generation=1,
    )
    assert first.intent is not None
    controller.acknowledge_admission(
        first.intent.intent_id, admitted=True, occurred_monotonic=1011.1
    )
    controller.complete(first.intent.intent_id, occurred_monotonic=1012.0)
    rejected = controller.observe_ema(
        _qualified_region(track_id=2, region=(0.70, 0.1, 0.9, 0.5), started_at=14.0),
        generation=1,
    )
    assert rejected.intent is not None
    controller.acknowledge_admission(
        rejected.intent.intent_id, admitted=False, occurred_monotonic=1015.1
    )

    retry = controller.observe_ema(
        _qualified_region(track_id=2, region=(0.70, 0.1, 0.9, 0.5), started_at=16.0),
        generation=1,
    )

    assert retry.reason is EpisodeDecisionReason.FOLLOWUP_RESERVED
    assert retry.intent is not None
    assert controller.snapshot()["followup_count"] == 1


def test_configured_five_minute_limit_applies_only_to_ema_admissions() -> None:
    controller = MotionEpisodeController(
        "gate", episode_gap_seconds=5.0, maximum_ema_requests_5m=2
    )
    controller.start_generation(1)
    qualified = _qualified("gate")
    for index, observed in enumerate((1011.0, 1051.0), start=1):
        decision = controller.observe_ema(
            replace(
                qualified,
                captured_at=observed - 1000.0,
                observed_monotonic=observed,
            ),
            generation=1,
        )
        assert decision.intent is not None
        controller.acknowledge_admission(
            decision.intent.intent_id,
            admitted=True,
            occurred_monotonic=observed + 0.1,
        )
        controller.complete(
            decision.intent.intent_id, occurred_monotonic=observed + 0.2
        )

    limited = controller.observe_ema(
        replace(qualified, captured_at=91.0, observed_monotonic=1091.0),
        generation=1,
    )
    camera = controller.observe_camera(
        CameraNotice("gate", 92.0, 1092.0, "motion"), generation=1
    )

    assert limited.reason is EpisodeDecisionReason.EMA_RATE_LIMITED
    assert camera.reason is EpisodeDecisionReason.REQUEST_RESERVED
    assert camera.intent is not None
    assert camera.intent.primary_source.value == "camera"


def test_same_track_cannot_create_followup_request() -> None:
    controller = MotionEpisodeController("gate", minimum_followup_interval_seconds=0.0)
    controller.start_generation(1)
    first = controller.observe_ema(
        _qualified_region(track_id=1, region=(0.05, 0.1, 0.2, 0.4), started_at=10.0),
        generation=1,
    )
    assert first.intent is not None
    controller.acknowledge_admission(
        first.intent.intent_id, admitted=True, occurred_monotonic=1011.1
    )
    controller.complete(first.intent.intent_id, occurred_monotonic=1012.0)

    duplicate = controller.observe_ema(
        _qualified_region(track_id=1, region=(0.08, 0.1, 0.23, 0.4), started_at=14.0),
        generation=1,
    )

    assert duplicate.reason is EpisodeDecisionReason.FOLLOWUP_DUPLICATE

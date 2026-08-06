"""Pure coordination policy for camera-triggered visual motion backup."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Sequence

from .motion import MotionQualificationResult


VISUAL_BACKUP_EXCLUDED_REASONS = frozenset({
    "global_illumination_change",
    "illumination_change",
    "insect_like_motion",
    "persistent_scene_motion",
    "stationary_foreground",
    "stationary_region",
})
UNSTABLE_BASELINE_REASONS = frozenset({
    "global_illumination_change",
    "illumination_change",
    "insufficient_frames",
    "validation_unavailable_fail_open",
})


@dataclass(frozen=True, slots=True)
class VisualBackupPolicy:
    warmup_seconds: float
    grace_seconds: float
    minimum_score: float
    score_margin: float
    minimum_consecutive: int
    cooldown_seconds: float
    maximum_triggers_5m: int
    sample_fps: float
    background_fps: float


class VisualBackupAction(StrEnum):
    DISABLED = "disabled"
    IGNORED = "ignored"
    NOT_READY = "not_ready"
    ACCUMULATING = "accumulating"
    CAMERA_NOTICE = "camera_notice"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class VisualBackupDecision:
    action: VisualBackupAction
    result: MotionQualificationResult
    required_score: float
    consecutive: int = 0
    scene_ready: bool = False
    new_camera_match: bool = False
    readiness_audit_needed: bool = False
    count_nonpromotion: bool = False
    illumination_probe: bool = False


@dataclass(frozen=True, slots=True)
class VisualBackupReplaySample:
    captured_at: float
    result: MotionQualificationResult
    detection_enabled: bool = True
    camera_motion_times: tuple[float, ...] = ()
    illumination_probe_allowed: bool = False


class VisualBackupCoordinator:
    """Own visual-backup state and return side-effect-free decisions."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_matched_camera_at = 0.0
        self.analysis_started_at: float | None = None
        self.scene_ready = False
        self.stable_since: float | None = None
        self.stable_samples = 0
        self.readiness_audited = False
        self.candidate_since: float | None = None
        self.last_candidate_at: float | None = None
        self.consecutive = 0
        self.trigger_times: deque[float] = deque(maxlen=64)

    def reset_candidate(self) -> None:
        self.candidate_since = None
        self.last_candidate_at = None
        self.consecutive = 0

    def readiness(
        self,
        result: MotionQualificationResult,
        captured_at: float,
        policy: VisualBackupPolicy,
    ) -> bool:
        if self.analysis_started_at is None or captured_at < self.analysis_started_at:
            self.analysis_started_at = captured_at
            self.scene_ready = False
            self.stable_since = None
            self.stable_samples = 0
        if self.scene_ready:
            return True
        if captured_at - self.analysis_started_at < policy.warmup_seconds:
            self.stable_since = None
            self.stable_samples = 0
            return False
        stable_result = bool(
            not result.accepted and result.reason not in UNSTABLE_BASELINE_REASONS
        )
        if not stable_result:
            self.stable_since = None
            self.stable_samples = 0
            return False
        if self.stable_since is None:
            self.stable_since = captured_at
        self.stable_samples += 1
        if (
            self.stable_samples >= max(3, policy.minimum_consecutive)
            and captured_at - self.stable_since >= max(1.5, policy.grace_seconds)
        ):
            self.scene_ready = True
            self.reset_candidate()
        return self.scene_ready

    def evaluate(
        self,
        result: MotionQualificationResult,
        captured_at: float,
        policy: VisualBackupPolicy,
        *,
        detection_enabled: bool,
        camera_motion_times: Sequence[float] = (),
        illumination_probe_allowed: bool = False,
    ) -> VisualBackupDecision:
        if not detection_enabled:
            self.reset_candidate()
            return VisualBackupDecision(
                VisualBackupAction.DISABLED, result, policy.minimum_score
            )

        scene_ready = self.readiness(result, captured_at, policy)
        illumination_probe = bool(
            result.reason == "illumination_change"
            and result.features.get("illumination_would_reject")
            and illumination_probe_allowed
        )
        effective_result = result
        if illumination_probe:
            effective_result = MotionQualificationResult(
                accepted=True,
                score=result.score,
                threshold=result.threshold,
                reason="illumination_verification_probe",
                frame_count=result.frame_count,
                features={**result.features, "illumination_verification_probe": True},
                telemetry=dict(result.telemetry),
            )
        required_score = max(
            policy.minimum_score,
            float(effective_result.threshold) + policy.score_margin,
        )
        strong_candidate = bool(
            effective_result.accepted
            and effective_result.score >= required_score
            and effective_result.reason not in VISUAL_BACKUP_EXCLUDED_REASONS
        )
        if not strong_candidate:
            count_nonpromotion = bool(effective_result.accepted and scene_ready)
            self.reset_candidate()
            return VisualBackupDecision(
                VisualBackupAction.IGNORED,
                effective_result,
                required_score,
                scene_ready=scene_ready,
                count_nonpromotion=count_nonpromotion,
                illumination_probe=illumination_probe,
            )
        if not scene_ready:
            audit_needed = not self.readiness_audited
            self.readiness_audited = True
            self.reset_candidate()
            return VisualBackupDecision(
                VisualBackupAction.NOT_READY,
                effective_result,
                required_score,
                readiness_audit_needed=audit_needed,
                illumination_probe=illumination_probe,
            )

        expected_interval = 1.0 / max(
            0.5, min(policy.sample_fps, policy.background_fps)
        )
        if (
            self.last_candidate_at is not None
            and captured_at - self.last_candidate_at > expected_interval * 2.5
        ):
            self.reset_candidate()
        if self.candidate_since is None:
            self.candidate_since = captured_at
        self.last_candidate_at = captured_at
        self.consecutive += 1
        if (
            self.consecutive < policy.minimum_consecutive
            or captured_at - self.candidate_since < policy.grace_seconds
        ):
            return VisualBackupDecision(
                VisualBackupAction.ACCUMULATING,
                effective_result,
                required_score,
                consecutive=self.consecutive,
                scene_ready=True,
                illumination_probe=illumination_probe,
            )

        matched_camera_at = max(
            (
                observed_at
                for observed_at in camera_motion_times
                if 0.0 <= captured_at - observed_at <= policy.cooldown_seconds
            ),
            default=0.0,
        )
        if matched_camera_at > 0.0:
            new_match = matched_camera_at > self.last_matched_camera_at
            if new_match:
                self.last_matched_camera_at = matched_camera_at
            self.reset_candidate()
            return VisualBackupDecision(
                VisualBackupAction.CAMERA_NOTICE,
                effective_result,
                required_score,
                scene_ready=True,
                new_camera_match=new_match,
                illumination_probe=illumination_probe,
            )
        return VisualBackupDecision(
            VisualBackupAction.READY,
            effective_result,
            required_score,
            consecutive=self.consecutive,
            scene_ready=True,
            illumination_probe=illumination_probe,
        )

    def reserve_trigger(
        self,
        captured_at: float,
        policy: VisualBackupPolicy,
        *,
        trigger_pending: bool,
        last_completed_at: float,
    ) -> bool:
        cutoff = captured_at - 300.0
        while self.trigger_times and self.trigger_times[0] < cutoff:
            self.trigger_times.popleft()
        return not bool(
            trigger_pending
            or (
                last_completed_at > 0.0
                and captured_at - last_completed_at < policy.cooldown_seconds
            )
            or len(self.trigger_times) >= policy.maximum_triggers_5m
        )

    def record_trigger(self, captured_at: float) -> None:
        self.trigger_times.append(captured_at)

    def record_camera_match(self, observed_at: float) -> bool:
        if observed_at <= self.last_matched_camera_at:
            return False
        self.last_matched_camera_at = observed_at
        return True

    def snapshot(self) -> dict[str, object]:
        return {
            "scene_ready": self.scene_ready,
            "stable_samples": self.stable_samples,
            "candidate_consecutive": self.consecutive,
            "recent_trigger_count": len(self.trigger_times),
        }


def replay_visual_backup(
    policy: VisualBackupPolicy,
    samples: Iterable[VisualBackupReplaySample],
) -> tuple[VisualBackupDecision, ...]:
    """Replay policy inputs without camera threads, files, clocks, or inference."""
    coordinator = VisualBackupCoordinator()
    return tuple(
        coordinator.evaluate(
            sample.result,
            sample.captured_at,
            policy,
            detection_enabled=sample.detection_enabled,
            camera_motion_times=sample.camera_motion_times,
            illumination_probe_allowed=sample.illumination_probe_allowed,
        )
        for sample in samples
    )

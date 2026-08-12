"""EMAv2 signal qualification and unified motion-episode admission.

The image pipeline answers whether visual motion is credible.  This module owns
the two policy boundaries that follow that answer:

* :class:`EmaSignalConditioner` converts sampled scores into one qualified edge.
* :class:`MotionEpisodeController` merges qualified EMA and camera notices into
  one episode and reserves at most one object-detection request at a time.

Neither class owns a thread, queue, detector, or incident store.  Callers must
explicitly acknowledge whether a reserved request was admitted so an observed
ONVIF notice can never hide an EMA rescue whose work was actually lost.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable

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
class EmaPolicy:
    warmup_seconds: float
    grace_seconds: float
    minimum_score: float
    score_margin: float
    minimum_consecutive: int
    cooldown_seconds: float
    maximum_triggers_5m: int
    sample_fps: float
    background_fps: float


class EmaSignalAction(StrEnum):
    DISABLED = "disabled"
    LEARNING = "learning"
    REJECTED = "rejected"
    ACCUMULATING = "accumulating"
    QUALIFIED = "qualified"


@dataclass(frozen=True, slots=True)
class EmaQualified:
    camera_id: str
    captured_at: float
    observed_monotonic: float
    result: MotionQualificationResult
    required_score: float
    qualifying_samples: int
    window_samples: int
    candidate_started_at: float


@dataclass(frozen=True, slots=True)
class EmaSignalDecision:
    action: EmaSignalAction
    result: MotionQualificationResult
    required_score: float
    scene_ready: bool
    qualifying_samples: int = 0
    window_samples: int = 0
    qualified: EmaQualified | None = None
    readiness_audit_needed: bool = False
    count_nonpromotion: bool = False


@dataclass(frozen=True, slots=True)
class _SignalSample:
    captured_at: float
    strong: bool


class EmaSignalConditioner:
    """Turn raw EMA scores into a tolerant, deterministic qualified edge.

    Readiness is based on elapsed learning time and observed samples rather than
    requiring a quiet scene.  Persistence uses a bounded k-of-n window so one
    borderline frame cannot erase an otherwise continuous real-world episode.
    """

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.reset()

    def reset(self) -> None:
        self.analysis_started_at: float | None = None
        self.observation_count = 0
        self.scene_ready = False
        self.readiness_audited = False
        self._samples: deque[_SignalSample] = deque()

    def evaluate(
        self,
        result: MotionQualificationResult,
        captured_at: float,
        observed_monotonic: float,
        policy: EmaPolicy,
        *,
        detection_enabled: bool,
    ) -> EmaSignalDecision:
        self._validate_clock(captured_at, "captured_at")
        self._validate_clock(observed_monotonic, "observed_monotonic")
        required_score = max(
            policy.minimum_score,
            float(result.threshold) + policy.score_margin,
        )
        if not detection_enabled:
            self._samples.clear()
            return EmaSignalDecision(
                EmaSignalAction.DISABLED,
                result,
                required_score,
                self.scene_ready,
            )

        if self.analysis_started_at is None or captured_at < self.analysis_started_at:
            self.reset()
            self.analysis_started_at = captured_at
        self.observation_count += 1
        minimum_learning_samples = max(3, policy.minimum_consecutive)
        if (
            captured_at - self.analysis_started_at >= policy.warmup_seconds
            and self.observation_count >= minimum_learning_samples
        ):
            self.scene_ready = True

        strong = bool(
            result.accepted
            and result.score >= required_score
            and result.reason not in VISUAL_BACKUP_EXCLUDED_REASONS
        )
        if not self.scene_ready:
            audit_needed = bool(strong and not self.readiness_audited)
            self.readiness_audited = self.readiness_audited or audit_needed
            self._samples.clear()
            return EmaSignalDecision(
                EmaSignalAction.LEARNING,
                result,
                required_score,
                False,
                readiness_audit_needed=audit_needed,
            )

        expected_interval = 1.0 / max(
            0.5, min(policy.sample_fps, policy.background_fps)
        )
        if (
            self._samples
            and captured_at - self._samples[-1].captured_at > expected_interval * 2.5
        ):
            self._samples.clear()
        self._samples.append(_SignalSample(captured_at, strong))
        maximum_samples = max(policy.minimum_consecutive + 1, 3)
        while len(self._samples) > maximum_samples:
            self._samples.popleft()

        qualifying = sum(sample.strong for sample in self._samples)
        strong_times = [sample.captured_at for sample in self._samples if sample.strong]
        persistence = (
            strong_times[-1] - strong_times[0] if len(strong_times) >= 2 else 0.0
        )
        if (
            qualifying >= policy.minimum_consecutive
            and persistence >= policy.grace_seconds
        ):
            qualified = EmaQualified(
                camera_id=self.camera_id,
                captured_at=captured_at,
                observed_monotonic=observed_monotonic,
                result=result,
                required_score=required_score,
                qualifying_samples=qualifying,
                window_samples=len(self._samples),
                candidate_started_at=strong_times[0],
            )
            self._samples.clear()
            return EmaSignalDecision(
                EmaSignalAction.QUALIFIED,
                result,
                required_score,
                True,
                qualifying_samples=qualifying,
                window_samples=qualified.window_samples,
                qualified=qualified,
            )
        if strong or any(sample.strong for sample in self._samples):
            return EmaSignalDecision(
                EmaSignalAction.ACCUMULATING,
                result,
                required_score,
                True,
                qualifying_samples=qualifying,
                window_samples=len(self._samples),
            )
        # Preserve only a short weak sample as dropout context; a known nuisance
        # or unstable baseline must not accumulate toward a later qualification.
        if (
            result.reason in VISUAL_BACKUP_EXCLUDED_REASONS
            or result.reason in UNSTABLE_BASELINE_REASONS
        ):
            self._samples.clear()
        return EmaSignalDecision(
            EmaSignalAction.REJECTED,
            result,
            required_score,
            True,
            window_samples=len(self._samples),
            count_nonpromotion=bool(result.accepted),
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "scene_ready": self.scene_ready,
            "learning_observations": self.observation_count,
            "candidate_samples": len(self._samples),
            "candidate_qualifying_samples": sum(
                sample.strong for sample in self._samples
            ),
        }

    @staticmethod
    def _validate_clock(value: float, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")


class MotionSource(StrEnum):
    CAMERA = "camera"
    EMA = "ema"
    MANUAL = "manual"


class DetectionRequestStatus(StrEnum):
    RESERVED = "reserved"
    ADMITTED = "admitted"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"


class EpisodeDecisionReason(StrEnum):
    REQUEST_RESERVED = "request_reserved"
    MERGED_WITH_REQUEST = "merged_with_request"
    COOLDOWN_ACTIVE = "cooldown_active"
    REQUEST_ABORTED = "request_aborted"
    REQUEST_ADMITTED = "request_admitted"
    REQUEST_RUNNING = "request_running"
    REQUEST_COMPLETED = "request_completed"
    FOLLOWUP_RESERVED = "followup_reserved"
    FOLLOWUP_DUPLICATE = "followup_duplicate"
    FOLLOWUP_RATE_LIMITED = "followup_rate_limited"
    FOLLOWUP_LIMIT_REACHED = "followup_limit_reached"
    STALE_GENERATION = "stale_generation"


@dataclass(frozen=True, slots=True)
class CameraNotice:
    camera_id: str
    event_at: float
    observed_monotonic: float
    topic: str
    message: str = ""
    manual: bool = False


@dataclass(frozen=True, slots=True)
class DetectionIntent:
    intent_id: str
    episode_id: str
    camera_id: str
    generation: int
    event_at: float
    created_monotonic: float
    primary_source: MotionSource
    sources: tuple[MotionSource, ...]
    ema: EmaQualified | None = None
    camera_notice: CameraNotice | None = None


@dataclass(frozen=True, slots=True)
class EpisodeTransition:
    episode_id: str
    generation: int
    occurred_monotonic: float
    reason: EpisodeDecisionReason
    source: MotionSource
    intent_id: str | None = None


@dataclass(slots=True)
class _Episode:
    episode_id: str
    sequence: int
    generation: int
    started_monotonic: float
    updated_monotonic: float
    sources: set[MotionSource] = field(default_factory=set)
    ema: EmaQualified | None = None
    camera_notice: CameraNotice | None = None
    intent: DetectionIntent | None = None
    status: DetectionRequestStatus | None = None
    completed_monotonic: float | None = None
    request_count: int = 0
    followup_count: int = 0
    last_request_monotonic: float = 0.0
    known_track_ids: set[int] = field(default_factory=set)
    covered_regions: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )


@dataclass(frozen=True, slots=True)
class EpisodeDecision:
    reason: EpisodeDecisionReason
    episode_id: str
    intent: DetectionIntent | None = None


class MotionEpisodeController:
    """Sole owner of cross-source episode identity and detector admission."""

    def __init__(
        self,
        camera_id: str,
        *,
        episode_gap_seconds: float = 30.0,
        cooldown_seconds: float = 0.0,
        transition_limit: int = 256,
        maximum_followups: int = 2,
        minimum_followup_interval_seconds: float = 1.5,
        followup_maximum_overlap: float = 0.10,
        followup_minimum_center_distance: float = 0.10,
    ) -> None:
        self.camera_id = camera_id
        self.episode_gap_seconds = max(0.1, float(episode_gap_seconds))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.maximum_followups = max(0, int(maximum_followups))
        self.minimum_followup_interval_seconds = max(
            0.0, float(minimum_followup_interval_seconds)
        )
        self.followup_maximum_overlap = min(
            1.0, max(0.0, float(followup_maximum_overlap))
        )
        self.followup_minimum_center_distance = min(
            1.0, max(0.0, float(followup_minimum_center_distance))
        )
        self._lock = threading.RLock()
        self._generation = 0
        self._sequence = 0
        self._episode: _Episode | None = None
        self._last_completed_monotonic: float | None = None
        self._transitions: deque[EpisodeTransition] = deque(
            maxlen=max(16, transition_limit)
        )

    def start_generation(self, generation: int) -> None:
        with self._lock:
            if generation < self._generation:
                raise ValueError("motion generation cannot move backwards")
            if generation != self._generation:
                self._generation = generation
                self._episode = None
                self._last_completed_monotonic = None

    def observe_camera(
        self, notice: CameraNotice, *, generation: int
    ) -> EpisodeDecision:
        if notice.camera_id != self.camera_id:
            raise ValueError("camera notice belongs to another camera")
        source = MotionSource.MANUAL if notice.manual else MotionSource.CAMERA
        return self._observe(
            source,
            event_at=notice.event_at,
            observed_monotonic=notice.observed_monotonic,
            generation=generation,
            camera_notice=notice,
        )

    def observe_ema(
        self, qualified: EmaQualified, *, generation: int
    ) -> EpisodeDecision:
        if qualified.camera_id != self.camera_id:
            raise ValueError("EMA evidence belongs to another camera")
        return self._observe(
            MotionSource.EMA,
            event_at=qualified.captured_at,
            observed_monotonic=qualified.observed_monotonic,
            generation=generation,
            ema=qualified,
        )

    def _observe(
        self,
        source: MotionSource,
        *,
        event_at: float,
        observed_monotonic: float,
        generation: int,
        ema: EmaQualified | None = None,
        camera_notice: CameraNotice | None = None,
    ) -> EpisodeDecision:
        with self._lock:
            if generation != self._generation:
                return self._decision(
                    EpisodeDecisionReason.STALE_GENERATION,
                    source,
                    observed_monotonic,
                )
            episode = self._episode_for(observed_monotonic)
            episode.sources.add(source)
            episode.updated_monotonic = max(
                episode.updated_monotonic, observed_monotonic
            )
            if ema is not None:
                episode.ema = ema
            if camera_notice is not None:
                episode.camera_notice = camera_notice
            if episode.status in {
                DetectionRequestStatus.RESERVED,
                DetectionRequestStatus.ADMITTED,
                DetectionRequestStatus.RUNNING,
            }:
                # Refresh the immutable request so downstream diagnostics see
                # every source associated with this same episode.
                if episode.intent is not None:
                    episode.intent = DetectionIntent(
                        intent_id=episode.intent.intent_id,
                        episode_id=episode.intent.episode_id,
                        camera_id=episode.intent.camera_id,
                        generation=episode.intent.generation,
                        event_at=min(episode.intent.event_at, event_at),
                        created_monotonic=episode.intent.created_monotonic,
                        primary_source=episode.intent.primary_source,
                        sources=tuple(sorted(episode.sources, key=str)),
                        ema=episode.ema,
                        camera_notice=episode.camera_notice,
                    )
                return self._decision(
                    EpisodeDecisionReason.MERGED_WITH_REQUEST,
                    source,
                    observed_monotonic,
                    episode,
                    episode.intent,
                )
            if episode.status is DetectionRequestStatus.COMPLETED:
                if ema is None:
                    return self._decision(
                        EpisodeDecisionReason.MERGED_WITH_REQUEST,
                        source,
                        observed_monotonic,
                        episode,
                        episode.intent,
                    )
                followup_reason = self._followup_reason(
                    episode, ema, observed_monotonic
                )
                if followup_reason is not EpisodeDecisionReason.FOLLOWUP_RESERVED:
                    return self._decision(
                        followup_reason,
                        source,
                        observed_monotonic,
                        episode,
                        episode.intent,
                    )
            if (
                self._last_completed_monotonic is not None
                and observed_monotonic - self._last_completed_monotonic
                < self.cooldown_seconds
            ):
                return self._decision(
                    EpisodeDecisionReason.COOLDOWN_ACTIVE,
                    source,
                    observed_monotonic,
                    episode,
                )
            episode.request_count += 1
            followup = episode.request_count > 1
            intent = DetectionIntent(
                intent_id=f"{episode.episode_id}:request:{episode.request_count}",
                episode_id=episode.episode_id,
                camera_id=self.camera_id,
                generation=generation,
                event_at=event_at,
                created_monotonic=observed_monotonic,
                primary_source=source,
                sources=tuple(sorted(episode.sources, key=str)),
                ema=episode.ema,
                camera_notice=episode.camera_notice,
            )
            episode.intent = intent
            episode.status = DetectionRequestStatus.RESERVED
            episode.last_request_monotonic = observed_monotonic
            if followup:
                episode.followup_count += 1
            if ema is not None:
                self._remember_ema(episode, ema)
            return self._decision(
                (
                    EpisodeDecisionReason.FOLLOWUP_RESERVED
                    if followup
                    else EpisodeDecisionReason.REQUEST_RESERVED
                ),
                source,
                observed_monotonic,
                episode,
                intent,
            )

    def acknowledge_admission(
        self,
        intent_id: str,
        *,
        admitted: bool,
        occurred_monotonic: float,
    ) -> EpisodeDecision:
        with self._lock:
            episode = self._require_intent(intent_id)
            if admitted:
                episode.status = DetectionRequestStatus.ADMITTED
                reason = EpisodeDecisionReason.REQUEST_ADMITTED
            else:
                episode.status = DetectionRequestStatus.ABORTED
                episode.intent = None
                episode.request_count = max(0, episode.request_count - 1)
                reason = EpisodeDecisionReason.REQUEST_ABORTED
            return self._decision(
                reason,
                MotionSource.EMA
                if episode.ema is not None
                else MotionSource.CAMERA,
                occurred_monotonic,
                episode,
                episode.intent,
            )

    def abort(
        self, intent_id: str, *, occurred_monotonic: float
    ) -> EpisodeDecision:
        with self._lock:
            episode = self._require_intent(intent_id)
            source = episode.intent.primary_source
            episode.status = DetectionRequestStatus.ABORTED
            episode.intent = None
            episode.request_count = max(0, episode.request_count - 1)
            return self._decision(
                EpisodeDecisionReason.REQUEST_ABORTED,
                source,
                occurred_monotonic,
                episode,
            )

    def mark_running(
        self, intent_id: str, *, occurred_monotonic: float
    ) -> EpisodeDecision:
        with self._lock:
            episode = self._require_intent(intent_id)
            episode.status = DetectionRequestStatus.RUNNING
            return self._decision(
                EpisodeDecisionReason.REQUEST_RUNNING,
                episode.intent.primary_source,
                occurred_monotonic,
                episode,
                episode.intent,
            )

    def complete(
        self, intent_id: str, *, occurred_monotonic: float
    ) -> EpisodeDecision:
        with self._lock:
            episode = self._require_intent(intent_id)
            episode.status = DetectionRequestStatus.COMPLETED
            episode.completed_monotonic = occurred_monotonic
            self._last_completed_monotonic = occurred_monotonic
            return self._decision(
                EpisodeDecisionReason.REQUEST_COMPLETED,
                episode.intent.primary_source,
                occurred_monotonic,
                episode,
                episode.intent,
            )

    def transitions(self) -> tuple[EpisodeTransition, ...]:
        with self._lock:
            return tuple(self._transitions)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            episode = self._episode
            return {
                "generation": self._generation,
                "episode_id": episode.episode_id if episode else None,
                "sources": (
                    tuple(sorted(source.value for source in episode.sources))
                    if episode
                    else ()
                ),
                "request_status": (
                    episode.status.value if episode and episode.status else None
                ),
                "intent_id": (
                    episode.intent.intent_id if episode and episode.intent else None
                ),
                "transition_count": len(self._transitions),
                "request_count": episode.request_count if episode else 0,
                "followup_count": episode.followup_count if episode else 0,
            }

    def reset(self) -> None:
        with self._lock:
            self._episode = None
            self._last_completed_monotonic = None
            self._transitions.clear()

    def _episode_for(self, observed_monotonic: float) -> _Episode:
        episode = self._episode
        if (
            episode is None
            or observed_monotonic < episode.started_monotonic
            or observed_monotonic - episode.updated_monotonic
            > self.episode_gap_seconds
        ):
            self._sequence += 1
            episode = _Episode(
                episode_id=f"{self.camera_id}:g{self._generation}:e{self._sequence}",
                sequence=self._sequence,
                generation=self._generation,
                started_monotonic=observed_monotonic,
                updated_monotonic=observed_monotonic,
            )
            self._episode = episode
        return episode

    def _require_intent(self, intent_id: str) -> _Episode:
        episode = self._episode
        if (
            episode is None
            or episode.intent is None
            or episode.intent.intent_id != intent_id
        ):
            raise ValueError("unknown or stale detection intent")
        return episode

    def _followup_reason(
        self,
        episode: _Episode,
        ema: EmaQualified,
        observed_monotonic: float,
    ) -> EpisodeDecisionReason:
        if episode.followup_count >= self.maximum_followups:
            self._remember_ema(episode, ema)
            return EpisodeDecisionReason.FOLLOWUP_LIMIT_REACHED
        if (
            observed_monotonic - episode.last_request_monotonic
            < self.minimum_followup_interval_seconds
        ):
            return EpisodeDecisionReason.FOLLOWUP_RATE_LIMITED
        track_id = self._track_id(ema.result.features.get("motion_region_track_id"))
        regions = self._regions(ema.result.features.get("motion_regions"))
        if track_id is not None and track_id in episode.known_track_ids:
            self._remember_ema(episode, ema)
            return EpisodeDecisionReason.FOLLOWUP_DUPLICATE
        if not regions:
            return EpisodeDecisionReason.FOLLOWUP_DUPLICATE
        candidate = regions[-1]
        overlap = max(
            (self._intersection_over_union(candidate, known) for known in episode.covered_regions),
            default=0.0,
        )
        center_distance = min(
            (self._center_distance(candidate, known) for known in episode.covered_regions),
            default=1.0,
        )
        if episode.covered_regions and (
            overlap > self.followup_maximum_overlap
            or center_distance < self.followup_minimum_center_distance
        ):
            self._remember_ema(episode, ema)
            return EpisodeDecisionReason.FOLLOWUP_DUPLICATE
        return EpisodeDecisionReason.FOLLOWUP_RESERVED

    @classmethod
    def _remember_ema(cls, episode: _Episode, ema: EmaQualified) -> None:
        track_id = cls._track_id(ema.result.features.get("motion_region_track_id"))
        if track_id is not None:
            episode.known_track_ids.add(track_id)
        episode.covered_regions.extend(
            cls._regions(ema.result.features.get("motion_regions"))
        )
        if len(episode.covered_regions) > 32:
            del episode.covered_regions[:-32]

    @staticmethod
    def _track_id(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            track_id = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return None
        return track_id if track_id > 0 else None

    @staticmethod
    def _regions(value: object) -> list[tuple[float, float, float, float]]:
        if not isinstance(value, (list, tuple)):
            return []
        regions: list[tuple[float, float, float, float]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(component) for component in item)
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(component) for component in (x1, y1, x2, y2)):
                continue
            x1, x2 = sorted((min(1.0, max(0.0, x1)), min(1.0, max(0.0, x2))))
            y1, y2 = sorted((min(1.0, max(0.0, y1)), min(1.0, max(0.0, y2))))
            if x2 > x1 and y2 > y1:
                regions.append((x1, y1, x2, y2))
        return regions

    @staticmethod
    def _intersection_over_union(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> float:
        intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
            0.0, min(left[3], right[3]) - max(left[1], right[1])
        )
        if intersection <= 0.0:
            return 0.0
        left_area = (left[2] - left[0]) * (left[3] - left[1])
        right_area = (right[2] - right[0]) * (right[3] - right[1])
        return intersection / max(1e-9, left_area + right_area - intersection)

    @staticmethod
    def _center_distance(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> float:
        return math.dist(
            ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0),
            ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0),
        )

    def _decision(
        self,
        reason: EpisodeDecisionReason,
        source: MotionSource,
        occurred_monotonic: float,
        episode: _Episode | None = None,
        intent: DetectionIntent | None = None,
    ) -> EpisodeDecision:
        episode_id = episode.episode_id if episode is not None else ""
        self._transitions.append(EpisodeTransition(
            episode_id=episode_id,
            generation=self._generation,
            occurred_monotonic=occurred_monotonic,
            reason=reason,
            source=source,
            intent_id=intent.intent_id if intent is not None else None,
        ))
        return EpisodeDecision(reason, episode_id, intent)


def replay_ema_signal(
    camera_id: str,
    policy: EmaPolicy,
    samples: Iterable[tuple[float, float, MotionQualificationResult]],
) -> tuple[EmaSignalDecision, ...]:
    conditioner = EmaSignalConditioner(camera_id)
    return tuple(
        conditioner.evaluate(
            result,
            captured_at,
            observed_monotonic,
            policy,
            detection_enabled=True,
        )
        for captured_at, observed_monotonic, result in samples
    )

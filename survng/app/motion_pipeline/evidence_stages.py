from __future__ import annotations

import math
import threading
from typing import Any, Mapping

from ..motion import BackgroundMotionTracker, aggregate_mog2_evidence
from .context import MotionContext
from .evidence import MotionEvidenceRepository, MotionEvidenceSample
from .registry import (
    MotionStageDependencies,
    MotionStageOption,
    MotionStageRegistration,
    MotionStageRegistry,
)


EVIDENCE_REPOSITORY_SERVICE = "motion_evidence_repository"


def _finite_float(value: object, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _safe_unit_score(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(0.0, min(1.0, number))


class Mog2EvidenceSourceStage:
    observation_kinds = frozenset({"frame"})

    def __init__(
        self,
        stage_id: str,
        repository: MotionEvidenceRepository,
        *,
        enabled: bool,
        sample_fps: float,
        history_seconds: float,
    ) -> None:
        self._stage_id = stage_id
        self.repository = repository
        self.enabled = bool(enabled)
        self.sample_fps = max(1.0, float(sample_fps))
        self.history_seconds = max(1.0, float(history_seconds))
        self._update_lock = threading.Lock()
        repository.configure_source(
            "mog2",
            enabled=self.enabled,
            implementation="opencv_mog2",
            sample_fps=self.sample_fps,
            history_seconds=self.history_seconds,
        )

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        if not self.enabled or context.original_frame is None:
            return context
        tracker = context.runtime.state_for(
            self.stage_id,
            lambda: BackgroundMotionTracker(
                sample_fps=self.sample_fps,
                history_seconds=self.history_seconds,
            ),
        )
        with self._update_lock:
            evidence = tracker.update(context.original_frame)
        self.repository.append("mog2", context.captured_at, evidence)
        context.source_evidence["mog2"] = dict(evidence)
        return context


class OnvifEventEvidenceStage:
    observation_kinds = frozenset({"motion_event"})

    def __init__(
        self,
        stage_id: str,
        repository: MotionEvidenceRepository,
        *,
        enabled: bool = True,
        base_score: float = 0.55,
        priority_score: float = 0.95,
        priority_keywords: tuple[str, ...] = (
            "manual",
            "person",
            "people",
            "human",
            "vehicle",
            "animal",
            "face",
        ),
    ) -> None:
        self._stage_id = stage_id
        self.repository = repository
        self.enabled = bool(enabled)
        self.base_score = max(0.0, min(1.0, float(base_score)))
        self.priority_score = max(0.0, min(1.0, float(priority_score)))
        self.priority_keywords = tuple(
            keyword.strip().lower()
            for keyword in priority_keywords
            if keyword.strip()
        )
        repository.configure_source(
            "onvif",
            enabled=self.enabled,
            implementation="onvif_event",
            base_score=self.base_score,
            priority_score=self.priority_score,
        )

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        if not self.enabled or context.configuration.get("observation_kind") != "motion_event":
            return context
        topic = str(context.configuration.get("event_topic") or "")
        message = str(context.configuration.get("event_message") or "")
        event_source = str(context.configuration.get("event_source") or "onvif")
        searchable = f"{topic} {message}".lower()
        priority = any(keyword in searchable for keyword in self.priority_keywords)
        evidence = {
            "warmed": 1.0,
            "score": self.priority_score if priority else self.base_score,
            "priority": priority,
            "topic": topic[:300],
            "message": message[:500],
            "event_source": event_source,
            "received_at": context.captured_at,
            "event_at": float(
                context.configuration.get("event_at", context.captured_at)
            ),
        }
        self.repository.append("onvif", context.captured_at, evidence)
        context.source_evidence["onvif"] = dict(evidence)
        return context


class BufferedMotionFusionStage:
    """Aggregates independent sources and optionally applies an explicit policy."""

    def __init__(
        self,
        stage_id: str,
        repository: MotionEvidenceRepository,
        sources: tuple[str, ...] = ("mog2",),
        *,
        policy: str = "audit",
        source_thresholds: Mapping[str, float] | None = None,
        source_weights: Mapping[str, float] | None = None,
        weighted_threshold: float | None = None,
        minimum_sources: int = 1,
        require_warmed: bool = True,
        include_primary: bool = True,
        fail_open: bool = True,
    ) -> None:
        self._stage_id = stage_id
        self.repository = repository
        self.sources = sources
        normalized_policy = policy.strip().lower()
        if normalized_policy not in {"audit", "bypass", "any", "all", "weighted"}:
            raise ValueError(f"unsupported motion fusion policy: {policy}")
        self.policy = normalized_policy
        self.source_thresholds = {
            str(source): max(0.0, min(1.0, _finite_float(
                threshold, label=f"source threshold for {source}"
            )))
            for source, threshold in (source_thresholds or {}).items()
        }
        self.source_weights = {
            str(source): max(0.0, _finite_float(
                weight, label=f"source weight for {source}"
            ))
            for source, weight in (source_weights or {}).items()
        }
        self.weighted_threshold = (
            None
            if weighted_threshold is None
            else max(0.0, min(1.0, _finite_float(
                weighted_threshold, label="weighted threshold"
            )))
        )
        self.minimum_sources = max(0, int(minimum_sources))
        self.require_warmed = bool(require_warmed)
        self.include_primary = bool(include_primary)
        self.fail_open = bool(fail_open)

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
        started_at = float(context.configuration.get("evidence_started_at", context.captured_at))
        ended_at = float(context.configuration.get("evidence_ended_at", context.captured_at))
        for source in self.sources:
            samples = self.repository.window(source, started_at, ended_at)
            if source == "mog2":
                aggregate = aggregate_mog2_evidence(
                    [dict(sample.values) for sample in samples]
                )
            else:
                aggregate = self._aggregate_generic_source(source, samples)
            context.source_evidence[source] = aggregate
            context.scoring.features.update(aggregate)
        self._apply_policy(context)
        return context

    @staticmethod
    def _aggregate_generic_source(
        source: str,
        samples: tuple[MotionEvidenceSample, ...],
    ) -> dict[str, Any]:
        if not samples:
            return {
                f"{source}_warmed": 0.0,
                f"{source}_sample_count": 0,
            }
        best = max(
            samples,
            key=lambda sample: _safe_unit_score(sample.values.get("score", 0.0)),
        )
        values = dict(best.values)
        aggregate = {
            f"{source}_warmed": _safe_unit_score(values.get("warmed", 1.0)),
            f"{source}_score": _safe_unit_score(values.get("score", 0.0)),
            f"{source}_sample_count": len(samples),
        }
        aggregate.update(
            {
                f"{source}_{key}": value
                for key, value in values.items()
                if key not in {"warmed", "score"}
            }
        )
        return aggregate

    def _apply_policy(self, context: MotionContext) -> None:
        if (
            bool(context.configuration.get("require_primary_trigger"))
            and not context.scoring.accepted
        ):
            context.scoring.features.update({
                "fusion_policy": self.policy,
                "fusion_applied": False,
                "fusion_primary_required": True,
            })
            context.scoring.reason = "primary_trigger_rejected"
            return
        if self.policy == "audit":
            return
        if self.policy == "bypass":
            context.scoring.accepted = True
            context.scoring.score = 1.0
            context.scoring.reason = "validation_disabled"
            context.scoring.features.update({
                "fusion_policy": self.policy,
                "fusion_applied": True,
                "fusion_primary_included": False,
            })
            return
        source_scores: dict[str, float] = {}
        source_votes: dict[str, bool] = {}
        for source in self.sources:
            aggregate = context.source_evidence.get(source, {})
            if not isinstance(aggregate, Mapping):
                continue
            score_value = aggregate.get(f"{source}_score", aggregate.get("score"))
            if score_value is None:
                continue
            warmed = aggregate.get(f"{source}_warmed", aggregate.get("warmed", 1.0))
            if self.require_warmed and _safe_unit_score(warmed) < 1.0:
                continue
            score = _safe_unit_score(score_value)
            source_scores[source] = score
            source_votes[source] = score >= self.source_thresholds.get(source, 0.5)

        features = context.scoring.features
        features["fusion_policy"] = self.policy
        features["fusion_sources_considered"] = sorted(source_scores)
        features["fusion_primary_accepted"] = context.scoring.accepted
        features["fusion_primary_included"] = self.include_primary
        if len(source_scores) < self.minimum_sources:
            features["fusion_applied"] = False
            features["fusion_reason"] = "insufficient_sources"
            if self.fail_open:
                context.scoring.accepted = True
                context.scoring.score = 1.0
                context.scoring.reason = "validation_unavailable_fail_open"
            else:
                context.scoring.accepted = False
                context.scoring.reason = "validation_unavailable_fail_closed"
            return

        primary_score = context.scoring.score
        considered_scores = list(source_scores.values())
        considered_votes = list(source_votes.values())
        if self.include_primary:
            considered_scores.insert(0, primary_score)
            considered_votes.insert(0, context.scoring.accepted)
        if self.policy == "any":
            fused_score = max(considered_scores, default=primary_score)
            accepted = any(considered_votes)
        elif self.policy == "all":
            fused_score = min(considered_scores, default=primary_score)
            accepted = bool(considered_votes) and all(considered_votes)
        else:
            primary_weight = self.source_weights.get("primary", 1.0) if self.include_primary else 0.0
            weighted_total = primary_score * primary_weight
            total_weight = primary_weight
            for source, score in source_scores.items():
                weight = self.source_weights.get(source, 1.0)
                weighted_total += score * weight
                total_weight += weight
            fused_score = weighted_total / total_weight if total_weight > 0 else primary_score
            threshold = (
                self.weighted_threshold
                if self.weighted_threshold is not None
                else context.scoring.threshold
            )
            accepted = fused_score >= threshold

        context.scoring.accepted = accepted
        context.scoring.score = round(fused_score, 4)
        context.scoring.reason = f"fusion_{self.policy}_{'accepted' if accepted else 'rejected'}"
        features["fusion_applied"] = True
        features["fusion_score"] = context.scoring.score
        features["fusion_source_votes"] = dict(source_votes)


def _repository(dependencies: MotionStageDependencies) -> MotionEvidenceRepository:
    repository = dependencies.services.get(EVIDENCE_REPOSITORY_SERVICE)
    if not isinstance(repository, MotionEvidenceRepository):
        raise ValueError(
            f"motion stage requires {EVIDENCE_REPOSITORY_SERVICE!r} dependency"
        )
    return repository


def _build_mog2_source(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> Mog2EvidenceSourceStage:
    return Mog2EvidenceSourceStage(
        stage_id,
        _repository(dependencies),
        enabled=bool(options.get("enabled", True)),
        sample_fps=float(options.get("sample_fps", 5.0)),
        history_seconds=float(options.get("history_seconds", 30.0)),
    )


def _build_onvif_source(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> OnvifEventEvidenceStage:
    raw_keywords = options.get(
        "priority_keywords",
        ("manual", "person", "people", "human", "vehicle", "animal", "face"),
    )
    if isinstance(raw_keywords, str):
        priority_keywords = (raw_keywords,)
    elif isinstance(raw_keywords, (list, tuple)):
        priority_keywords = tuple(str(keyword) for keyword in raw_keywords)
    else:
        raise ValueError("ONVIF priority_keywords must be a list of strings")
    return OnvifEventEvidenceStage(
        stage_id,
        _repository(dependencies),
        enabled=bool(options.get("enabled", True)),
        base_score=float(options.get("base_score", 0.55)),
        priority_score=float(options.get("priority_score", 0.95)),
        priority_keywords=priority_keywords,
    )


def _build_buffered_fusion(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> BufferedMotionFusionStage:
    raw_sources = options.get("sources", ("mog2",))
    if isinstance(raw_sources, str):
        source_values = (raw_sources,)
    elif isinstance(raw_sources, (list, tuple)):
        source_values = raw_sources
    else:
        raise ValueError("motion fusion sources must be a list of source names")
    sources = tuple(dict.fromkeys(
        normalized
        for source in source_values
        if (normalized := str(source).strip().lower())
    ))
    raw_thresholds = options.get("source_thresholds", {})
    raw_weights = options.get("source_weights", {})
    if not isinstance(raw_thresholds, Mapping):
        raise ValueError("motion fusion source_thresholds must be an object")
    if not isinstance(raw_weights, Mapping):
        raise ValueError("motion fusion source_weights must be an object")
    weighted_threshold = options.get("weighted_threshold")
    return BufferedMotionFusionStage(
        stage_id,
        _repository(dependencies),
        sources,
        policy=str(options.get("policy", "audit")),
        source_thresholds={
            str(source): float(threshold)
            for source, threshold in raw_thresholds.items()
        },
        source_weights={
            str(source): float(weight)
            for source, weight in raw_weights.items()
        },
        weighted_threshold=(
            None if weighted_threshold is None else float(weighted_threshold)
        ),
        minimum_sources=int(options.get("minimum_sources", 1)),
        require_warmed=bool(options.get("require_warmed", True)),
        include_primary=bool(options.get("include_primary", True)),
        fail_open=bool(options.get("fail_open", True)),
    )


def register_evidence_stages(registry: MotionStageRegistry) -> None:
    registry.register(
        MotionStageRegistration(
            implementation="opencv_mog2_evidence",
            builder=_build_mog2_source,
            requires=frozenset({"original_frame"}),
            provides=frozenset({"source_evidence"}),
            graph="observation",
            category="background",
            observation_kinds=frozenset({"frame"}),
            display_name="Adaptive background (MOG2)",
            description="Learns the normal scene and reports foreground changes.",
            options=(
                MotionStageOption("enabled", "Enabled", "boolean", True),
                MotionStageOption("sample_fps", "Samples per second", "number", 5.0, minimum=0.1, maximum=30),
                MotionStageOption("history_seconds", "Learning time", "number", 30.0, minimum=1, maximum=600),
            ),
        )
    )
    registry.register(
        MotionStageRegistration(
            implementation="onvif_event_evidence",
            builder=_build_onvif_source,
            requires=frozenset({"configuration"}),
            provides=frozenset({"source_evidence"}),
            graph="observation",
            category="camera_signal",
            observation_kinds=frozenset({"motion_event"}),
            display_name="Camera motion signal (ONVIF)",
            description="Uses motion and object notifications sent by the camera.",
            options=(
                MotionStageOption("enabled", "Enabled", "boolean", True),
                MotionStageOption("base_score", "Motion confidence", "number", 0.55, minimum=0, maximum=1),
                MotionStageOption("priority_score", "Priority confidence", "number", 0.95, minimum=0, maximum=1),
                MotionStageOption("priority_keywords", "Priority event words", "string_list", [], advanced=True),
            ),
        )
    )
    registry.register(
        MotionStageRegistration(
            implementation="buffered_evidence_fusion",
            builder=_build_buffered_fusion,
            requires=frozenset({"scoring"}),
            provides=frozenset({"source_evidence", "scoring"}),
            graph="fusion",
            category="fusion",
            display_name="Buffered evidence fusion",
            description="Combines the normal motion score with recent independent evidence.",
            options=(
                MotionStageOption("sources", "Extra sources", "string_list", ["mog2", "onvif"]),
                MotionStageOption("policy", "Decision style", "string", "audit", choices=("audit", "bypass", "any", "all", "weighted")),
                MotionStageOption("source_thresholds", "Source confidence levels", "object", {}, advanced=True),
                MotionStageOption("source_weights", "Source importance", "object", {}, advanced=True),
                MotionStageOption("weighted_threshold", "Combined confidence", "number", 0.5, minimum=0, maximum=1, advanced=True),
                MotionStageOption("minimum_sources", "Minimum available sources", "integer", 1, minimum=0, advanced=True),
                MotionStageOption("require_warmed", "Require learned background", "boolean", True, advanced=True),
                MotionStageOption("include_primary", "Include adaptive analysis", "boolean", True, advanced=True),
                MotionStageOption("fail_open", "Run detection when validation is unavailable", "boolean", True, advanced=True),
            ),
        )
    )

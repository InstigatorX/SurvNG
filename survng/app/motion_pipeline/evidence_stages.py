from __future__ import annotations

from typing import Any, Mapping

from ..camera_semantics import active_camera_semantic_kind
from ..motion_topics import semantic_motion_kind

from .context import MotionContext
from .evidence import MotionEvidenceRepository
from .registry import (
    MotionStageDependencies,
    MotionStageOption,
    MotionStageRegistration,
    MotionStageRegistry,
)


EVIDENCE_REPOSITORY_SERVICE = "motion_evidence_repository"


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
        semantic_kind = active_camera_semantic_kind(topic, message)
        topic_semantic_kind = semantic_motion_kind(topic)
        priority = topic_semantic_kind == "manual" or semantic_kind is not None or (
            topic_semantic_kind is None
            and any(keyword in searchable for keyword in self.priority_keywords)
        )
        evidence = {
            "warmed": 1.0,
            "score": self.priority_score if priority else self.base_score,
            "priority": priority,
            "semantic_kind": semantic_kind,
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
    """Apply the EMA validation switch to the primary qualification result.

    ``audit`` keeps the EMA decision. ``bypass`` disables EMA validation so a
    camera notice runs object detection. No independent evidence source feeds
    this stage; configuration validation rejects sources and other policies.
    """

    def __init__(self, stage_id: str, *, policy: str = "audit") -> None:
        self._stage_id = stage_id
        normalized_policy = policy.strip().lower()
        if normalized_policy not in {"audit", "bypass"}:
            raise ValueError(f"unsupported motion fusion policy: {policy}")
        self.policy = normalized_policy

    @property
    def stage_id(self) -> str:
        return self._stage_id

    def process(self, context: MotionContext) -> MotionContext:
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
            return context
        if self.policy == "bypass":
            context.scoring.accepted = True
            context.scoring.score = 1.0
            context.scoring.reason = "validation_disabled"
            context.scoring.features.update({
                "fusion_policy": self.policy,
                "fusion_applied": True,
                "fusion_primary_included": False,
            })
        return context


def _repository(dependencies: MotionStageDependencies) -> MotionEvidenceRepository:
    repository = dependencies.services.get(EVIDENCE_REPOSITORY_SERVICE)
    if not isinstance(repository, MotionEvidenceRepository):
        raise ValueError(
            f"motion stage requires {EVIDENCE_REPOSITORY_SERVICE!r} dependency"
        )
    return repository




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
    # Saved configurations written by the UI still carry the former
    # source_thresholds/source_weights/weighted_threshold/minimum_sources/
    # require_warmed/include_primary/fail_open keys. They are accepted and
    # ignored; ``include_primary`` is read by the qualification service.
    del dependencies
    raw_sources = options.get("sources", ())
    sources = (raw_sources,) if isinstance(raw_sources, str) else raw_sources
    if not isinstance(sources, (list, tuple)):
        raise ValueError("motion fusion sources must be a list of source names")
    if any(str(source).strip() for source in sources):
        raise ValueError("motion fusion has no supported evidence sources")
    return BufferedMotionFusionStage(stage_id, policy=str(options.get("policy", "audit")))


def register_evidence_stages(registry: MotionStageRegistry) -> None:
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
            description="Turns EMA validation on (audit) or off (bypass) for camera notices.",
            options=(
                MotionStageOption("sources", "Extra sources", "string_list", [], advanced=True),
                MotionStageOption("policy", "Decision style", "string", "audit", choices=("audit", "bypass")),
                MotionStageOption("source_thresholds", "Source confidence levels", "object", {}, advanced=True),
                MotionStageOption("source_weights", "Source importance", "object", {}, advanced=True),
                MotionStageOption("weighted_threshold", "Combined confidence", "number", 0.5, minimum=0, maximum=1, advanced=True),
                MotionStageOption("minimum_sources", "Minimum available sources", "integer", 1, minimum=0, advanced=True),
                MotionStageOption("require_warmed", "Require learned background", "boolean", True, advanced=True),
                MotionStageOption("include_primary", "Include EMA", "boolean", True, advanced=True),
                MotionStageOption("fail_open", "Run detection when validation is unavailable", "boolean", True, advanced=True),
            ),
        )
    )

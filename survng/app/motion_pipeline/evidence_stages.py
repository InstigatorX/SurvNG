from __future__ import annotations

from typing import Any, Mapping

from ..motion import BackgroundMotionTracker, aggregate_mog2_evidence
from .context import MotionContext
from .evidence import MotionEvidenceRepository
from .registry import MotionStageDependencies, MotionStageRegistration, MotionStageRegistry


EVIDENCE_REPOSITORY_SERVICE = "motion_evidence_repository"


class Mog2EvidenceSourceStage:
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
        evidence = tracker.update(context.original_frame)
        self.repository.append("mog2", context.captured_at, evidence)
        context.source_evidence["mog2"] = dict(evidence)
        return context


class BufferedMotionFusionStage:
    """Adds buffered source evidence without changing the primary decision policy."""

    def __init__(
        self,
        stage_id: str,
        repository: MotionEvidenceRepository,
        sources: tuple[str, ...] = ("mog2",),
    ) -> None:
        self._stage_id = stage_id
        self.repository = repository
        self.sources = sources

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
                aggregate = {
                    f"{source}_sample_count": len(samples),
                }
            context.source_evidence[source] = aggregate
            context.scoring.features.update(aggregate)
        return context


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


def _build_buffered_fusion(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> BufferedMotionFusionStage:
    raw_sources = options.get("sources", ("mog2",))
    if isinstance(raw_sources, str):
        sources = (raw_sources,)
    elif isinstance(raw_sources, (list, tuple)):
        sources = tuple(str(source) for source in raw_sources)
    else:
        raise ValueError("motion fusion sources must be a list of source names")
    return BufferedMotionFusionStage(stage_id, _repository(dependencies), sources)


def register_evidence_stages(registry: MotionStageRegistry) -> None:
    registry.register(
        MotionStageRegistration(
            implementation="opencv_mog2_evidence",
            builder=_build_mog2_source,
            requires=frozenset({"original_frame"}),
            provides=frozenset({"source_evidence"}),
        )
    )
    registry.register(
        MotionStageRegistration(
            implementation="buffered_evidence_fusion",
            builder=_build_buffered_fusion,
            requires=frozenset({"scoring"}),
            provides=frozenset({"source_evidence", "scoring"}),
        )
    )

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..motion_types import MotionBlob, MotionFrameBlobs, MotionTrack
from .runtime import MotionRuntimeState


Frame: TypeAlias = NDArray[np.uint8]


@dataclass(slots=True)
class MotionScoring:
    accepted: bool = False
    score: float = 0.0
    threshold: float = 0.0
    reason: str = "unscored"
    frame_count: int = 0
    features: dict[str, Any] = field(default_factory=dict)


class MotionEventPhase(StrEnum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    REJECTED = "rejected"
    COOLDOWN = "cooldown"


@dataclass(slots=True)
class MotionEventState:
    phase: MotionEventPhase = MotionEventPhase.IDLE
    event_key: str = ""
    started_at: float | None = None
    updated_at: float | None = None


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    run_object_detection: bool
    reason: str
    score: float
    evidence_sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StageTiming:
    stage_id: str
    duration_ms: float
    succeeded: bool


@dataclass(slots=True)
class MotionDebugData:
    enabled: bool = False
    values: dict[str, Any] = field(default_factory=dict)
    images: dict[str, Frame] = field(default_factory=dict)


@dataclass(slots=True)
class MotionContext:
    camera_id: str
    captured_at: float
    original_frame: Frame | None
    configuration: Mapping[str, Any]
    runtime: MotionRuntimeState

    frame_history: tuple[Frame, ...] = ()
    processed_frame_history: tuple[Frame, ...] = ()
    difference_history: tuple[Frame, ...] = ()
    threshold_mask_history: tuple[Frame, ...] = ()
    motion_mask_history: tuple[Frame, ...] = ()
    raw_blob_history: tuple[MotionFrameBlobs, ...] = ()
    filtered_blob_history: tuple[MotionFrameBlobs, ...] = ()
    dominant_track: MotionTrack | None = None
    processed_frame: Frame | None = None
    background_image: Frame | None = None
    difference_image: Frame | None = None
    binary_motion_mask: Frame | None = None
    blobs: list[MotionBlob] = field(default_factory=list)
    tracked_objects: list[MotionTrack] = field(default_factory=list)
    scoring: MotionScoring = field(default_factory=MotionScoring)
    event_state: MotionEventState = field(default_factory=MotionEventState)
    decision: TriggerDecision | None = None
    source_evidence: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, StageTiming] = field(default_factory=dict)
    debug: MotionDebugData = field(default_factory=MotionDebugData)

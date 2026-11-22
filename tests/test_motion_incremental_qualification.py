"""Continuous EMA analysis processes each frame once.

``MotionAnalysisService.analyze_continuous`` used to pass its whole frame ring
to the stateful qualification graph on every cycle. It now passes the newest
already-processed frame as an anchor plus newer frames. These tests replay the
same frame sequence through the incremental path and through a full-window
replay on the current code, and require the same accept/reason/score.

That is not identity with the code before the change: the illumination stage
now keeps each pair's mask from when its newer frame was analysed, where a
full replay recomputed it against a background that had already advanced.
Scores can differ in the fourth decimal on lighting scenes; the intended
behaviour is pinned in test_illumination_pair_mask_is_frozen_at_first_analysis.
"""

from __future__ import annotations

import threading
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from survng.app.config import CameraConfig, MotionQualificationConfig
from survng.app.motion_analysis import FairMotionAnalysisLimiter
from survng.app.motion_analysis_service import MotionAnalysisService
from survng.app.motion_events import MotionEventCoordinator
from survng.app.motion_pipeline import (
    MotionDebugSnapshotStore,
    MotionPipelineFactory,
    adaptive_motion_stage_configs,
    build_builtin_motion_registry,
)
from survng.app.motion_qualification_service import MotionQualificationService

WIDTH, HEIGHT = 320, 180
FPS = 5.0
RING = 36


def _qualification(config: MotionQualificationConfig) -> MotionQualificationService:
    pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
        "gate",
        adaptive_motion_stage_configs(),
        required_artifacts={"scoring"},
    )
    observation = Mock()
    observation.handles_observation.return_value = False
    observation.stage_configuration = []
    fusion = Mock()
    fusion.stage_configuration = []
    return MotionQualificationService(
        camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        config=config,
        qualification_pipeline=pipeline,
        observation_pipeline=observation,
        fusion_pipeline=fusion,
        pipeline_origins={"qualification": "default", "observation": "default", "fusion": "default"},
        debug_store=MotionDebugSnapshotStore(),
        stop_event=threading.Event(),
        state=Mock(),
    )


def _analysis(qualification: MotionQualificationService, config: MotionQualificationConfig) -> MotionAnalysisService:
    state = Mock()
    state.detection_enabled.return_value = True
    state.lifecycle_generation.return_value = 1
    return MotionAnalysisService(
        camera_id="gate",
        frame_lock=threading.Lock(),
        analysis_lock=threading.Lock(),
        ring_size=RING,
        queue_size=1,
        limiter=FairMotionAnalysisLimiter(1),
        events=MotionEventCoordinator(queue_size=4, retry_limit=2, camera_id="gate"),
        evidence=Mock(),
        audit_recorder=Mock(),
        debug_store=MotionDebugSnapshotStore(),
        config=config,
        qualification=qualification,
        media=Mock(),
        state=state,
    )


def _scene(kind: str):
    rng = np.random.default_rng(11)
    base = rng.integers(40, 200, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)

    def frame(index: int) -> np.ndarray:
        image = base.astype(np.int16)
        image += rng.integers(-5, 6, size=image.shape, dtype=np.int16)
        if kind == "lighting" and 30 <= index < 60:
            image += (index - 29) * 2
        image = np.clip(image, 0, 255).astype(np.uint8)
        if kind == "object" and index >= 30:
            x = 10 + ((index - 30) * 5) % (WIDTH - 50)
            cv2.rectangle(image, (x, 60), (x + 26, 120), (240, 235, 230), -1)
        return image

    return frame


def _decisions(*, incremental: bool, kind: str, illumination: bool) -> list[tuple[bool, float, str]]:
    config = MotionQualificationConfig(mode="camera", illumination_filter_enabled=illumination)
    qualification = _qualification(config)
    analysis = _analysis(qualification, config)
    frame = _scene(kind)
    decisions: list[tuple[bool, float, str]] = []
    try:
        for index in range(110):
            captured_at = 1_000.0 + index / FPS
            analysis._preprocess_frame(frame(index), captured_at)
            if incremental:
                analysis.analyze_continuous(captured_at)
                result = analysis.last_continuous_result
            else:
                # The former behaviour: the whole ring on every cycle.
                with analysis.frame_lock:
                    color = list(analysis.color_frames)
                    gray = dict(analysis.frames)
                    processed = dict(analysis.processed_frames)
                if len(color) < 2:
                    continue
                result = qualification.run_pipeline(
                    [image for _at, image in color],
                    "balanced",
                    captured_at,
                    [at for at, _image in color],
                    isolated=False,
                    capture_debug=False,
                    include_telemetry=False,
                    processed_frames=[processed[at] for at, _image in color],
                    processed_frame_implementation="gray_blur",
                    luminance_frames=[gray[at] for at, _image in color],
                )
            if result is not None:
                decisions.append((result.accepted, round(result.score, 6), result.reason))
    finally:
        qualification.qualification_pipeline.close()
    return decisions


@pytest.mark.parametrize(
    ("kind", "illumination"),
    [("object", False), ("object", True), ("lighting", True), ("static", False)],
)
def test_incremental_analysis_matches_full_ring(kind: str, illumination: bool) -> None:
    full = _decisions(incremental=False, kind=kind, illumination=illumination)
    incremental = _decisions(incremental=True, kind=kind, illumination=illumination)

    assert incremental == full
    if kind == "object":
        assert any(accepted for accepted, _score, _reason in incremental)


def test_incremental_analysis_passes_only_new_frames_after_warmup() -> None:
    config = MotionQualificationConfig(mode="camera")
    qualification = _qualification(config)
    analysis = _analysis(qualification, config)
    frame = _scene("object")
    calls: list[int] = []
    original = qualification.run_pipeline

    def recording_run_pipeline(frames, *args, **kwargs):
        calls.append(len(frames))
        result = original(frames, *args, **kwargs)
        assert result.frame_count == kwargs["history_frame_count"]
        return result

    qualification.run_pipeline = recording_run_pipeline  # type: ignore[method-assign]
    try:
        for index in range(50):
            captured_at = 2_000.0 + index / FPS
            analysis._preprocess_frame(frame(index), captured_at)
            analysis.analyze_continuous(captured_at)
        assert calls[0] == 2
        assert set(calls[1:]) == {2}

        # A runtime reset (new generation) replays the retained ring once.
        qualification.qualification_pipeline.runtime.reset()
        captured_at = 2_000.0 + 50 / FPS
        analysis._preprocess_frame(frame(50), captured_at)
        analysis.analyze_continuous(captured_at)
        assert calls[-1] == len(analysis.color_frames) > 2
    finally:
        qualification.qualification_pipeline.close()


def test_illumination_pair_mask_is_frozen_at_first_analysis() -> None:
    """A later call with a different mask for the same frame keeps the first."""
    from types import SimpleNamespace

    from survng.app.motion_pipeline.adaptive_stages import (
        IlluminationChangeFilterStage,
        _IlluminationRuntime,
    )

    rng = np.random.default_rng(3)
    base = rng.integers(40, 160, size=(90, 160, 3), dtype=np.uint8)
    brighter = np.clip(base.astype(np.int16) + 50, 0, 255).astype(np.uint8)
    first_mask = np.zeros((90, 160), np.uint8)
    first_mask[10:60, 10:90] = 255
    later_mask = np.zeros((90, 160), np.uint8)
    later_mask[60:88, 100:158] = 255
    stage = IlluminationChangeFilterStage("illumination")
    expected = stage._pair_evidence(base, brighter, first_mask)
    recomputed = stage._pair_evidence(base, brighter, later_mask)
    assert expected != recomputed  # the masks are distinguishable

    def context(timestamps, frames, masks):
        return SimpleNamespace(
            frame_history=frames, motion_mask_history=masks, frame_timestamps=timestamps,
            configuration={"history_started_at": 1.0},
        )

    state = _IlluminationRuntime()
    gray = (None, None, None)
    first = stage._windowed_pair_evidence(
        state, context((1.0, 2.0), (base, brighter), [first_mask]), gray, evaluate=True,
    )
    # The background has since advanced: the context now offers another mask
    # for the frame at 2.0. The pair (1.0, 2.0) keeps the one it was scored with.
    later = stage._windowed_pair_evidence(
        state,
        context((1.0, 2.0, 3.0), (base, brighter, brighter), [later_mask, later_mask]),
        gray,
        evaluate=True,
    )
    assert first == [expected]
    assert later[0] == expected

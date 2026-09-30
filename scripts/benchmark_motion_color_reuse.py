#!/usr/bin/env python3
"""Time motion preprocessing for an analysis-sized BGR frame.

Live capture stays on the Phase 1 software bgr24 pipe. This measures the
Python step that turns that frame into the gray motion buffer. The baseline
always resizes, including when the frame is already the analysis size. The
current path reuses that frozen color buffer.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from survng.app.config import MotionQualificationConfig
from survng.app.motion_analysis import FairMotionAnalysisLimiter
from survng.app.motion_analysis_service import MotionAnalysisService
from survng.app.motion_events import MotionEventCoordinator
from survng.app.motion_pipeline import MotionDebugSnapshotStore


def _service(frame_width: int) -> MotionAnalysisService:
    qualification = Mock()
    qualification.settings.return_value = ("adaptive", "balanced", frame_width)
    qualification.preprocessor_implementation.return_value = "gray_blur"
    return MotionAnalysisService(
        camera_id="bench",
        frame_lock=threading.Lock(),
        analysis_lock=threading.Lock(),
        ring_size=8,
        queue_size=1,
        limiter=FairMotionAnalysisLimiter(1),
        events=MotionEventCoordinator(queue_size=4, retry_limit=2, camera_id="bench"),
        evidence=Mock(),
        audit_recorder=Mock(),
        debug_store=MotionDebugSnapshotStore(),
        config=MotionQualificationConfig(sample_fps=5.0, frame_width=frame_width),
        qualification=qualification,
        media=Mock(),
        state=Mock(),
    )


def _frozen(height: int, width: int) -> np.ndarray:
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    frame.setflags(write=False)
    return frame


def _always_resize(frame: np.ndarray, width: int, height: int) -> None:
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(gray, (5, 5), 0)


def _reuse_color(frame: np.ndarray) -> None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(gray, (5, 5), 0)


def _median_ms(work, repeats: int = 9, frames: int = 400) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        for _warmup in range(20):
            work()
        started = time.perf_counter()
        for _frame in range(frames):
            work()
        samples.append((time.perf_counter() - started) * 1000.0 / frames)
    return statistics.median(samples)


def main() -> None:
    analysis = _frozen(360, 640)
    larger = _frozen(720, 1280)
    service = _service(640)
    epoch = 1_000.0

    def current_analysis() -> None:
        nonlocal epoch
        epoch += 1.0
        prepared = service._preprocess_frame(analysis, epoch)
        if prepared is None or prepared[1] is not analysis:
            raise RuntimeError("analysis-sized color buffer was copied")

    def current_larger() -> None:
        nonlocal epoch
        epoch += 1.0
        prepared = service._preprocess_frame(larger, epoch)
        if prepared is None or prepared[1].shape != (360, 640, 3):
            raise RuntimeError("larger frame was not downscaled")

    print(f"640x360 always resize  {_median_ms(lambda: _always_resize(analysis, 640, 360)):.3f} ms")
    print(f"640x360 reuse color    {_median_ms(lambda: _reuse_color(analysis)):.3f} ms")
    print(f"640x360 service reuse  {_median_ms(current_analysis):.3f} ms")
    print(f"1280x720 downscale     {_median_ms(lambda: _always_resize(larger, 640, 360)):.3f} ms")
    print(f"1280x720 service       {_median_ms(current_larger):.3f} ms")
    telemetry = service.telemetry_snapshot()
    print(
        "identity reuse",
        telemetry["identity_color_reuse_count"],
        "derived bytes",
        telemetry["derived_frame_bytes"],
    )


if __name__ == "__main__":
    main()

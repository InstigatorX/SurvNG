from __future__ import annotations

import threading
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app.config import CameraConfig, DetectionZone, MotionQualificationConfig
from survng.app.motion import MotionQualificationResult
from survng.app.motion_pipeline import MotionDebugSnapshotStore
from survng.app.motion_qualification_service import (
    MotionQualificationService,
)


def _service(
    *,
    camera: CameraConfig | None = None,
    config: MotionQualificationConfig | None = None,
    increment_stat: Mock | None = None,
) -> MotionQualificationService:
    qualification = Mock()
    qualification.continuous_analysis = True
    qualification.stage_configuration = []
    observation = Mock()
    observation.handles_observation.return_value = False
    observation.stage_configuration = []
    fusion = Mock()
    fusion.stage_configuration = []
    return MotionQualificationService(
        camera=camera
        or CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://example.invalid/main",
        ),
        config=config or MotionQualificationConfig(),
        qualification_pipeline=qualification,
        observation_pipeline=observation,
        fusion_pipeline=fusion,
        pipeline_origins={
            "qualification": "default",
            "observation": "default",
            "fusion": "default",
        },
        debug_store=MotionDebugSnapshotStore(),
        stop_event=threading.Event(),
        state=Mock(increment_stat=increment_stat or Mock()),
    )


def test_camera_overrides_are_resolved_without_worker_knowledge() -> None:
    camera = CameraConfig.model_validate({
        "id": "gate",
        "name": "Gate",
        "stream_url": "rtsp://example.invalid/main",
        "motion_qualification": {
            "mode": "adaptive",
            "sensitivity": "high",
            "frame_width": 480,
            "stationary_object_tolerance": "high",
            "borderline_rescue_enabled": False,
            "borderline_margin": 0.01,
        },
    })
    service = _service(camera=camera)

    assert service.settings() == ("adaptive", "high", 480)
    assert service.trigger_mode() == "adaptive"
    assert service.stationary_object_tolerance() == "high"
    assert service.rescue_settings() == (False, 0.01)


def test_hot_zone_update_refreshes_pipeline_configuration_snapshot() -> None:
    service = _service()
    exclusion = DetectionZone.model_validate({
        "name": "Tree",
        "exclude_from_ema": True,
        "points": [
            {"x": 0.0, "y": 0.0},
            {"x": 0.5, "y": 0.0},
            {"x": 0.5, "y": 0.5},
        ],
    })

    service.update_zones([exclusion])
    assert service._pipeline_configuration["motion_zones"][0]["name"] == "Tree"
    assert service._pipeline_configuration["motion_zones"][0]["exclude_from_ema"] is True

    service.update_zones([])
    assert service._pipeline_configuration["motion_zones"] == []


def test_validation_failure_preserves_rejected_primary_when_required() -> None:
    increment_stat = Mock()
    service = _service(increment_stat=increment_stat)
    rejected = MotionQualificationResult(False, 0.3, 0.48, "low_score", 4, {})

    result = service.validation_fail_open_result(
        "motion fusion pipeline",
        RuntimeError("unavailable"),
        rejected,
        allow_detection=False,
    )

    assert not result.accepted
    assert result.reason == "primary_trigger_rejected"
    assert result.features["validation_unavailable"] is True
    increment_stat.assert_any_call("validation_failures", 1)
    increment_stat.assert_any_call("validation_fail_opens", 0)


def test_continuous_schedule_uses_resolved_mode_and_background_rate() -> None:
    config = MotionQualificationConfig(
        mode="camera_rescue",
        sample_fps=5.0,
        camera_mode_background_fps=2.0,
    )
    service = _service(config=config)

    assert service.continuous_primary_required()
    assert not service.continuous_primary_due(10.3, 10.0)
    assert service.continuous_primary_due(10.5, 10.0)


def test_runtime_reset_clears_all_pipeline_state_and_fusion_clock() -> None:
    service = _service()
    service._fusion_last_at = 100.0

    service.reset_runtime()

    service.qualification_pipeline.runtime.reset.assert_called_once_with()
    service.observation_pipeline.runtime.reset.assert_called_once_with()
    service.fusion_pipeline.runtime.reset.assert_called_once_with()
    assert service._fusion_last_at == 0.0


def test_observation_reset_waits_for_inflight_frame_observation() -> None:
    service = _service()
    service.observation_pipeline.handles_observation.return_value = True
    processing = threading.Event()
    release = threading.Event()
    reset_called = threading.Event()
    reset_order: list[str] = []

    def reset_observation_runtime() -> None:
        reset_order.append("runtime")
        reset_called.set()

    service.observation_pipeline.runtime.reset.side_effect = reset_observation_runtime

    def process(_context: object) -> None:
        processing.set()
        assert release.wait(timeout=1.0)

    service.observation_pipeline.process.side_effect = process
    observer = threading.Thread(
        target=service.observe_frame,
        args=(np.zeros((8, 8), dtype=np.uint8), 100.0),
    )
    resetter = threading.Thread(
        target=lambda: service.reset_runtime(
            clear_observation_evidence=lambda: reset_order.append("evidence")
        )
    )

    observer.start()
    assert processing.wait(timeout=1.0)
    resetter.start()
    assert not reset_called.wait(timeout=0.05)
    assert reset_order == []
    release.set()
    observer.join(timeout=1.0)
    resetter.join(timeout=1.0)

    assert not observer.is_alive()
    assert not resetter.is_alive()
    service.observation_pipeline.runtime.reset.assert_called_once_with()
    assert reset_order == ["evidence", "runtime"]


def test_pipeline_rejects_misaligned_preprocessed_derivatives() -> None:
    service = _service()
    frames = [np.zeros((10, 10), dtype=np.uint8) for _index in range(2)]

    with pytest.raises(ValueError, match="one derivative for each source frame"):
        service.run_pipeline(
            frames,
            "balanced",
            10.0,
            processed_frames=[frames[0]],
        )


def test_cached_derivatives_are_used_only_for_matching_preprocessor() -> None:
    service = _service()
    service.qualification_pipeline.stage_configuration = [
        {"stage_id": "preprocess", "implementation": "future_gpu"}
    ]
    service.qualification_pipeline.process.side_effect = lambda context: context
    frames = [np.zeros((10, 10), dtype=np.uint8) for _index in range(2)]
    cached = [np.ones((10, 10), dtype=np.uint8) for _index in range(2)]

    service.run_pipeline(
        frames,
        "balanced",
        10.0,
        isolated=False,
        capture_debug=False,
        include_telemetry=False,
        processed_frames=cached,
        processed_frame_implementation="gray_blur",
    )

    context = service.qualification_pipeline.process.call_args.args[0]
    assert context.processed_frame_history == ()
    assert context.processed_frame is None


def test_cached_derivatives_reach_matching_configured_preprocessor() -> None:
    service = _service()
    service.qualification_pipeline.stage_configuration = [
        {"stage_id": "preprocess", "implementation": "gray_blur"}
    ]
    service.qualification_pipeline.process.side_effect = lambda context: context
    frames = [np.zeros((10, 10), dtype=np.uint8) for _index in range(2)]
    cached = [np.ones((10, 10), dtype=np.uint8) for _index in range(2)]

    service.run_pipeline(
        frames,
        "balanced",
        10.0,
        isolated=False,
        capture_debug=False,
        include_telemetry=False,
        processed_frames=cached,
        processed_frame_implementation="gray_blur",
    )

    context = service.qualification_pipeline.process.call_args.args[0]
    assert context.processed_frame_history == tuple(cached)
    assert context.processed_frame is cached[-1]

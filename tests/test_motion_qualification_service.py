from __future__ import annotations

import threading
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app.config import CameraConfig, MotionQualificationConfig
from survng.app.motion import MotionQualificationResult
from survng.app.motion_pipeline import MotionDebugSnapshotStore
from survng.app.motion_qualification_service import (
    MotionQualificationHooks,
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
        hooks=MotionQualificationHooks(
            samples_since=lambda _captured_at: [],
            increment_stat=increment_stat or Mock(),
        ),
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

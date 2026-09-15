from __future__ import annotations

import pytest

from survng.app.camera_capture import CaptureOpenLimiter
from survng.app.config import DetectorConfig
from survng.app.dlstreamer_capture import DlStreamerCaptureBackend, DlStreamerCaptureOptions
from survng.dlstreamer_live import _normalize_gva_objects, _parser


def test_native_live_pipeline_defaults_are_off() -> None:
    config = DetectorConfig()
    assert config.live_pipeline_inference_enabled is False
    assert config.live_pipeline_inference_interval == 1
    assert config.live_pipeline_tracking == "off"


def test_backend_command_carries_native_tracking_policy() -> None:
    backend = DlStreamerCaptureBackend(
        CaptureOpenLimiter(1),
        DlStreamerCaptureOptions(
            detect_enabled=True,
            model_path="/models/yolo.xml",
            inference_interval=3,
            native_tracking="short-term-imageless",
        ),
    )
    command = backend.command()
    assert command[command.index("--inference-interval") + 1] == "3"
    assert command[command.index("--native-tracking") + 1] == "short-term-imageless"
    assert "--no-detect" not in command


def test_backend_rejects_interval_beyond_short_term_tracker_contract() -> None:
    with pytest.raises(ValueError, match="inference_interval"):
        DlStreamerCaptureBackend(
            CaptureOpenLimiter(1),
            DlStreamerCaptureOptions(inference_interval=6),
        )


def test_live_parser_accepts_native_tracking_policy() -> None:
    args = _parser().parse_args(
        ["--inference-interval", "3", "--native-tracking", "short-term-imageless"]
    )
    assert args.inference_interval == 3
    assert args.native_tracking == "short-term-imageless"


def test_gva_normalization_preserves_native_track_id_as_diagnostic_metadata() -> None:
    objects = _normalize_gva_objects(
        {
            "objects": [
                {
                    "x": 10,
                    "y": 20,
                    "w": 30,
                    "h": 40,
                    "object_id": 17,
                    "detection": {"label": "person", "confidence": 0.91},
                }
            ]
        }
    )
    assert objects == [
        {
            "label": "person",
            "confidence": 0.91,
            "box": {"x1": 10, "y1": 20, "x2": 40, "y2": 60},
            "native_track_id": 17,
        }
    ]

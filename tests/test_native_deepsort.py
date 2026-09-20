from types import SimpleNamespace

import pytest

from survng.app.camera_capture import CaptureOpenLimiter
from survng.app.config import DetectorConfig
from survng.app.dlstreamer_capture import DlStreamerCaptureBackend, DlStreamerCaptureOptions
from survng.dlstreamer_live import _parser
from survng.native_deepsort import DEFAULT_DEEP_SORT_CONFIG, resolve_native_tracking


def detector(*, implementation="off", enabled=False, path="", interval=1, classes=None, device="GPU"):
    tracking = SimpleNamespace(
        mode=(
            "deep-sort"
            if implementation in {"deep-sort", "dlstreamer_deep_sort"}
            else "off"
        ),
        reid_enabled=enabled,
        reid_model_path=path,
        reid_device=device,
        deep_sort_config=DEFAULT_DEEP_SORT_CONFIG,
        resolved_reid_device=lambda: device,
    )
    native = SimpleNamespace(
        inference_interval=interval,
        tracking_classes=classes,
        tracking=tracking,
    )
    return SimpleNamespace(native=native)


def test_default_native_tracking_is_off():
    plan = resolve_native_tracking(detector(classes=["person", "car"]))
    assert plan.mode == "off"
    assert plan.tracking_classes == ("person", "car")
    assert plan.reid_model_path == ""


def test_deep_sort_is_rejected_without_gvatrack():
    with pytest.raises(ValueError, match="without gvatrack"):
        resolve_native_tracking(detector(
            implementation="dlstreamer_deep_sort",
            enabled=True,
            path="/models/mars_small128_fp32.xml",
            classes=None,
        ))


def test_legacy_short_term_config_normalizes_to_off():
    config = DetectorConfig.model_validate({
        "native": {
            "tracking": {"mode": "short-term-imageless"},
        },
    })
    assert config.native.tracking.mode == "off"
    assert resolve_native_tracking(config).mode == "off"


def test_legacy_deep_sort_config_still_loads_but_resolve_rejects():
    config = DetectorConfig.model_validate({
        "native": {"inference_interval": 1},
        "tracking": {
            "implementation": "deep-sort",
            "reid_enabled": True,
            "reid_model_path": "/models/mars.xml",
            "reid_device": "GPU",
        },
    })
    assert config.native.tracking.mode == "deep-sort"
    with pytest.raises(ValueError, match="without gvatrack"):
        resolve_native_tracking(config)


def test_capture_command_passes_tracking_off():
    backend = DlStreamerCaptureBackend(
        CaptureOpenLimiter(1),
        DlStreamerCaptureOptions(
            model_path="/models/person.xml",
            detect_enabled=True,
            native_tracking="off",
            tracking_classes=("person",),
        ),
    )
    command = backend.command()
    assert "--native-tracking" in command
    assert command[command.index("--native-tracking") + 1] == "off"


def test_parser_defaults_native_tracking_off():
    args = _parser().parse_args([])
    assert args.native_tracking == "off"

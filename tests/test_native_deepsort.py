from types import SimpleNamespace

from survng.app.camera_capture import CaptureOpenLimiter
from survng.app.config import DetectorConfig
from survng.app.dlstreamer_capture import DlStreamerCaptureBackend, DlStreamerCaptureOptions
from survng.dlstreamer_live import _parser
from survng.native_deepsort import resolve_native_tracking


def detector(*, classes=None):
    tracking = SimpleNamespace(mode="off")
    native = SimpleNamespace(
        inference_interval=1,
        tracking_classes=classes,
        tracking=tracking,
    )
    return SimpleNamespace(native=native)


def test_default_native_tracking_is_off():
    plan = resolve_native_tracking(detector(classes=["person", "car"]))
    assert plan.mode == "off"
    assert plan.tracking_classes == ("person", "car")
    assert plan.reid_model_path == ""


def test_legacy_short_term_config_normalizes_to_off():
    config = DetectorConfig.model_validate({
        "native": {
            "tracking": {"mode": "short-term-imageless"},
        },
    })
    assert config.native.tracking.mode == "off"
    assert resolve_native_tracking(config).mode == "off"


def test_legacy_deep_sort_config_normalizes_to_off():
    config = DetectorConfig.model_validate({
        "native": {"inference_interval": 1},
        "tracking": {
            "implementation": "deep-sort",
            "reid_enabled": True,
            "reid_model_path": "/models/mars.xml",
            "reid_device": "GPU",
        },
    })
    assert config.native.tracking.mode == "off"
    assert resolve_native_tracking(config).mode == "off"


def test_explicit_deep_sort_mode_normalizes_to_off():
    config = DetectorConfig.model_validate({
        "native": {
            "tracking": {
                "mode": "deep-sort",
                "reid_enabled": True,
                "reid_model_path": "/models/mars.xml",
            },
        },
    })
    assert config.native.tracking.mode == "off"
    assert resolve_native_tracking(config).mode == "off"


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

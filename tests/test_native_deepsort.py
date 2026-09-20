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


def test_resolve_native_tracking_exposes_classes_and_forces_off():
    plan = resolve_native_tracking(detector(classes=["person", "car"]))
    assert plan.mode == "off"
    assert plan.tracking_classes == ("person", "car")


def test_legacy_tracker_modes_normalize_to_off():
    for mode in ("short-term-imageless", "deep-sort", "gvatrack"):
        config = DetectorConfig.model_validate({
            "native": {"tracking": {"mode": mode}},
        })
        assert config.native.tracking.mode == "off"
        assert resolve_native_tracking(config).mode == "off"


def test_legacy_detector_tracking_implementation_migrates_to_off():
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
    assert "--reid-model" not in command
    assert "--deep-sort-config" not in command


def test_capture_rejects_non_off_tracking():
    try:
        DlStreamerCaptureBackend(
            CaptureOpenLimiter(1),
            DlStreamerCaptureOptions(native_tracking="deep-sort"),
        )
    except ValueError as error:
        assert "must be off" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_parser_defaults_native_tracking_off():
    args = _parser().parse_args([])
    assert args.native_tracking == "off"

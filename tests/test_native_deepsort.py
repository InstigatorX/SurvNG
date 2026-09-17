from types import SimpleNamespace

import pytest

from survng.app.camera_capture import CaptureOpenLimiter
from survng.app.dlstreamer_capture import DlStreamerCaptureBackend, DlStreamerCaptureOptions
from survng.dlstreamer_live import _parser
from survng.native_deepsort import DEFAULT_DEEP_SORT_CONFIG, resolve_native_tracking


def detector(*, implementation="survng_hybrid", enabled=False, path="", interval=1, classes=None, device="GPU"):
    tracking = SimpleNamespace(
        implementation=implementation,
        reid_enabled=enabled,
        reid_model_path=path,
        reid_device=device,
        resolved_reid_device=lambda: device,
    )
    native = SimpleNamespace(
        inference_interval=interval,
        tracking_classes=classes,
    )
    return SimpleNamespace(tracking=tracking, native=native)


def test_default_native_tracking_remains_imageless():
    plan = resolve_native_tracking(detector(classes=["person", "car"]))
    assert plan.mode == "short-term-imageless"
    assert plan.tracking_classes == ("person", "car")
    assert plan.reid_model_path == ""


def test_deep_sort_resolves_person_reid_plan():
    plan = resolve_native_tracking(detector(
        implementation="dlstreamer_deep_sort",
        enabled=True,
        path="/models/mars_small128_fp32.xml",
        classes=None,
    ))
    assert plan.mode == "deep-sort"
    assert plan.tracking_classes == ("person",)
    assert plan.reid_model_path.endswith("mars_small128_fp32.xml")
    assert plan.reid_device == "GPU"
    assert "max_cosine_distance=0.2" in plan.deep_sort_config


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"implementation": "deep-sort", "enabled": False, "path": "/m.xml"}, "reid_enabled"),
        ({"implementation": "deep-sort", "enabled": True, "path": ""}, "reid_model_path"),
        ({"implementation": "deep-sort", "enabled": True, "path": "/m.xml", "interval": 2}, "inference_interval"),
        ({"implementation": "deep-sort", "enabled": True, "path": "/m.xml", "classes": ["person", "car"]}, "person-only"),
    ],
)
def test_deep_sort_rejects_unsafe_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_native_tracking(detector(**kwargs))


def test_capture_command_passes_deep_sort_reid_contract():
    backend = DlStreamerCaptureBackend(
        CaptureOpenLimiter(1),
        DlStreamerCaptureOptions(
            model_path="/models/yolo.xml",
            detect_enabled=True,
            native_tracking="deep-sort",
            tracking_classes=("person",),
            reid_model_path="/models/mars_small128_fp32.xml",
            reid_device="GPU",
            deep_sort_config=DEFAULT_DEEP_SORT_CONFIG,
        ),
    )
    command = backend.command()
    assert command[command.index("--native-tracking") + 1] == "deep-sort"
    assert command[command.index("--reid-model") + 1].endswith("mars_small128_fp32.xml")
    assert command[command.index("--reid-device") + 1] == "GPU"
    assert command[command.index("--tracking-classes") + 1] == '["person"]'
    assert command[command.index("--deep-sort-config") + 1] == DEFAULT_DEEP_SORT_CONFIG


def test_child_parser_accepts_deep_sort_arguments():
    args = _parser().parse_args([
        "--native-tracking", "deep-sort",
        "--reid-model", "/models/mars.xml",
        "--reid-device", "GPU",
        "--tracking-classes", '["person"]',
    ])
    assert args.native_tracking == "deep-sort"
    assert args.reid_model == "/models/mars.xml"
    assert args.reid_device == "GPU"
    assert args.tracking_classes == ["person"]

from types import SimpleNamespace

import pytest

from survng.app.camera_capture import CaptureOpenLimiter
from survng.app.config import DetectorConfig
from survng.app.dlstreamer_capture import DlStreamerCaptureBackend, DlStreamerCaptureOptions
from survng.dlstreamer_live import _NativeInferenceEvidence, _NativeReidEvidence, _parser
from survng.native_deepsort import DEFAULT_DEEP_SORT_CONFIG, resolve_native_tracking


def detector(*, implementation="short-term-imageless", enabled=False, path="", interval=1, classes=None, device="GPU"):
    tracking = SimpleNamespace(
        mode=(
            "deep-sort"
            if implementation in {"deep-sort", "dlstreamer_deep_sort"}
            else "short-term-imageless"
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
    assert "max_age=60" in plan.deep_sort_config
    assert "max_cosine_distance=0.3" in plan.deep_sort_config
    assert "object_class=person" in plan.deep_sort_config
    assert "reid_max_age=30" in plan.deep_sort_config


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


def test_legacy_deep_sort_config_migrates_into_native_namespace():
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
    assert config.native.tracking.reid_model_path == "/models/mars.xml"
    assert resolve_native_tracking(config).mode == "deep-sort"


def test_explicit_native_tracking_wins_over_legacy_settings():
    config = DetectorConfig.model_validate({
        "native": {
            "tracking": {"mode": "short-term-imageless"},
        },
        "tracking": {
            "implementation": "deep-sort",
            "reid_enabled": True,
            "reid_model_path": "/models/legacy.xml",
        },
    })
    assert resolve_native_tracking(config).mode == "short-term-imageless"


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


class _FakeTensor:
    def __init__(self, *, name="inference_layer_name:output", layer="output", size=128):
        self._name = name
        self._layer = layer
        self._data = SimpleNamespace(size=size)

    def name(self):
        return self._name

    def layer_name(self):
        return self._layer

    def data(self):
        return self._data


class _FakeRegion:
    def __init__(self, label, tensors=()):
        self._label = label
        self._tensors = list(tensors)

    def label(self):
        return self._label

    def tensors(self):
        return list(self._tensors)

    def confidence(self):
        return 0.9

    def rect(self):
        return SimpleNamespace(x=1, y=2, w=3, h=4)


class _FakeFrame:
    def __init__(self, _buffer, *, caps=None, regions=()):
        del _buffer, caps
        self._regions = list(regions)

    def regions(self):
        return list(self._regions)


def _frame_type(regions):
    return lambda buffer, caps=None: _FakeFrame(buffer, caps=caps, regions=regions)


def test_reid_evidence_accepts_tracker_compatible_128d_tensor():
    evidence = _NativeReidEvidence()
    evidence.observe(object(), object(), _frame_type([
        _FakeRegion("person", [_FakeTensor()]),
    ]))
    status = evidence.status()
    assert status["native_reid_feature_health"] == "healthy"
    assert status["native_reid_features_valid"] == 1
    assert status["native_reid_features_missing"] == 0


def test_reid_evidence_surfaces_missing_or_misnamed_tensor():
    evidence = _NativeReidEvidence()
    evidence.observe(object(), object(), _frame_type([
        _FakeRegion("person", [_FakeTensor(name="classification_layer_name:embedding", layer="embedding")]),
    ]))
    status = evidence.status()
    assert status["native_reid_feature_health"] == "missing_features"
    assert status["native_reid_features_valid"] == 0
    assert status["native_reid_features_missing"] == 1
    assert status["native_reid_last_missing_tensors"][0]["size"] == 128


def test_native_evidence_ignores_spatial_inference_roi():
    from survng.native_spatial import ROI_LABEL

    evidence = _NativeInferenceEvidence(1, tracking=True)
    buffer = SimpleNamespace(pts=1)
    evidence.observe(buffer, object(), _frame_type([
        _FakeRegion(ROI_LABEL),
        _FakeRegion("person"),
    ]))
    provenance, objects = evidence.pop(1)
    assert provenance == "native_fresh_detection"
    assert [item["label"] for item in objects] == ["person"]

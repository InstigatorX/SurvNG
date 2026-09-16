from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from survng.app.config import AppConfig, CameraConfig
from survng.app.config_application import manager_owned_config
from survng.app.camera_capture import CaptureOpenLimiter
from survng.app.dlstreamer_capture import DlStreamerCaptureBackend, DlStreamerCaptureOptions, _SharedLiveProcess
from survng.app.native_camera import NativeCaptureBinding
from survng.dlstreamer_live import _configure_h264_decoder


def camera(**kwargs):
    return CameraConfig(id="test", name="Test", stream_url="rtsp://example.invalid/live", **kwargs)


def test_compliance_default_and_validation():
    assert camera().h264_decoder_compliance == "auto"
    for value in ("auto", "strict", "normal", "flexible"):
        original = camera(h264_decoder_compliance=value)
        assert CameraConfig.model_validate_json(original.model_dump_json()).h264_decoder_compliance == value
    for value in ("invalid", 3, None):
        with pytest.raises(ValidationError):
            camera(h264_decoder_compliance=value)


def test_compliance_change_recreates_capture():
    original = AppConfig(cameras=[camera()])
    changed = original.model_copy(deep=True)
    changed.cameras[0].h264_decoder_compliance = "flexible"
    assert manager_owned_config(original) != manager_owned_config(changed)


@pytest.mark.parametrize("role", ["live", "main"])
def test_camera_compliance_reaches_shared_command(monkeypatch, role):
    backend = DlStreamerCaptureBackend(CaptureOpenLimiter(1), DlStreamerCaptureOptions())
    shared = _SharedLiveProcess([], read_timeout_ms=1000)
    commands = []
    monkeypatch.setattr(shared, "_send", commands.append)
    monkeypatch.setattr(shared, "is_running", lambda: True)
    backend._shared = shared
    binding = NativeCaptureBinding(backend, lambda: True, lambda: "flexible")
    handle = binding.create_handle()
    handle.set_source_role(role)
    monkeypatch.setattr(handle, "prefetch", lambda *args: True)
    assert backend._open_shared(handle, "rtsp://example.invalid/live", lambda: False, timeout_ms=1000)
    assert commands[0]["h264_decoder_compliance"] == "flexible"
    assert commands[0]["source_role"] == role
    # Another camera using the same process keeps its own default.
    other = NativeCaptureBinding(backend, lambda: True).create_handle()
    monkeypatch.setattr(other, "prefetch", lambda *args: True)
    assert backend._open_shared(other, "rtsp://example.invalid/other", lambda: False, timeout_ms=1000)
    assert commands[1]["h264_decoder_compliance"] == "auto"


@pytest.mark.parametrize("factory,property_exists", [("vah265dec", True), ("avdec_h264", False)])
def test_compliance_leaves_other_decoders_unchanged(factory, property_exists):
    updates = []
    decoder = SimpleNamespace(
        get_factory=lambda: SimpleNamespace(get_name=lambda: factory),
        find_property=lambda name: object() if property_exists else None,
        set_property=lambda *args: updates.append(args),
    )
    _configure_h264_decoder(decoder, "flexible")
    assert updates == []

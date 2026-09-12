"""Native install preflight must fail clearly before cameras are started."""

from pathlib import Path
import runpy
import re
import sys
from types import SimpleNamespace

import pytest

from survng import dlstreamer_live as native


CHECK = Path(__file__).resolve().parents[1] / "scripts/check-native-runtime.py"


@pytest.fixture(autouse=True)
def check_cli_arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(CHECK)])


def test_native_driver_pins_match_docker():
    root = CHECK.parents[1]
    docker = dict(re.findall(r"^ARG (\w+_VERSION)=(\S+)$", (root / "Dockerfile").read_text(), re.M))
    native = dict(re.findall(r"^(\w+_version)=(\S+)$", (root / "scripts/install-native-runtime.sh").read_text(), re.M))
    assert native
    for name, version in native.items():
        assert version == docker[name.upper()], name


@pytest.mark.parametrize("failure", ["Namespace Gst not available", "required GStreamer element is unavailable: gvadetect"])
def test_missing_runtime_prints_install_remedy(monkeypatch, capsys, failure):
    monkeypatch.setattr(native, "_apply_dlstreamer_env", lambda: None)

    def fail():
        raise RuntimeError(failure)

    monkeypatch.setattr(native, "_load_gstreamer", fail)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(CHECK), run_name="__main__")
    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert failure in error
    assert "sudo bash scripts/install-native-runtime.sh" in error
    assert "/usr/bin/python3 as the service user" in error


@pytest.mark.parametrize("missing", [None, "appsink"])
def test_capture_element_check(monkeypatch, capsys, missing):
    monkeypatch.setattr(native, "_apply_dlstreamer_env", lambda: None)
    gst = SimpleNamespace(
        version_string=lambda: "GStreamer fixture",
        ElementFactory=SimpleNamespace(find=lambda name: None if name == missing else object()),
    )
    monkeypatch.setattr(native, "_load_gstreamer", lambda: gst)
    monkeypatch.setattr(native, "_require_detection_plugin", lambda gst: None)
    if missing:
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(CHECK), run_name="__main__")
        assert exc.value.code == 1
        assert "missing GStreamer elements: appsink" in capsys.readouterr().err
    else:
        runpy.run_path(str(CHECK), run_name="__main__")
        assert "Native capture runtime OK" in capsys.readouterr().out


@pytest.mark.parametrize("gpu_available", [False, True])
def test_gpu_preflight_checks_render_access(monkeypatch, capsys, gpu_available):
    monkeypatch.setattr(sys, "argv", [str(CHECK), "--intel-gpu"])
    monkeypatch.setattr(native, "_apply_dlstreamer_env", lambda: None)
    gst = SimpleNamespace(version_string=lambda: "fixture", ElementFactory=SimpleNamespace(find=lambda name: object()))
    monkeypatch.setattr(native, "_load_gstreamer", lambda: gst)
    monkeypatch.setattr(native, "_require_detection_plugin", lambda gst: None)

    def context(gst):
        if not gpu_available:
            raise RuntimeError("could not open shared VA display")
        return SimpleNamespace(get_structure=lambda: SimpleNamespace(get_string=lambda name: "/dev/dri/renderD129"))

    monkeypatch.setattr(native, "_create_shared_va_context", context)
    if gpu_available:
        runpy.run_path(str(CHECK), run_name="__main__")
        assert "Shared VA context OK: /dev/dri/renderD129" in capsys.readouterr().out
    else:
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(CHECK), run_name="__main__")
        assert exc.value.code == 1
        output = capsys.readouterr()
        assert "could not open shared VA display" in output.err
        assert "runtime OK" not in output.out

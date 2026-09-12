"""Native install preflight must fail clearly before cameras are started."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from survng import dlstreamer_live as native


CHECK = Path(__file__).resolve().parents[1] / "scripts/check-native-runtime.py"


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

"""Check native capture dependencies using the production runtime selection.

Run with /usr/bin/python3, also as the service user before first startup.
This does not connect to cameras or change service/configuration state.
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survng.dlstreamer_live import _apply_dlstreamer_env, _load_gstreamer, _require_detection_plugin


def check_runtime():
    gst = _load_gstreamer()
    _require_detection_plugin(gst)
    required = ("uridecodebin3", "videoconvert", "videoscale", "videorate", "appsink", "jpegenc")
    missing = [name for name in required if gst.ElementFactory.find(name) is None]
    if missing:
        raise RuntimeError("missing GStreamer elements: " + ", ".join(missing))
    print(f"Native capture runtime OK: {gst.version_string()}, gvadetect available")


if __name__ == "__main__":
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    _apply_dlstreamer_env()
    if os.environ.get("LD_LIBRARY_PATH", "") != previous:
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    try:
        check_runtime()
    except Exception as exc:
        print(f"Native capture runtime unavailable: {exc}\n"
              "On Ubuntu 24.04 amd64, run: sudo bash scripts/install-native-runtime.sh\n"
              "Then rerun this check with /usr/bin/python3 as the service user.", file=sys.stderr)
        raise SystemExit(1)

"""Check native capture dependencies using the production runtime selection.

Run with /usr/bin/python3, also as the service user before first startup.
This does not connect to cameras or change service/configuration state.
"""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survng.dlstreamer_live import _apply_dlstreamer_env, _create_shared_va_context, _load_gstreamer, _require_detection_plugin


def check_runtime(*, intel_gpu=False):
    gst = _load_gstreamer()
    _require_detection_plugin(gst)
    required = ("uridecodebin3", "videoconvert", "videoscale", "videorate", "appsink", "jpegenc")
    missing = [name for name in required if gst.ElementFactory.find(name) is None]
    if missing:
        raise RuntimeError("missing GStreamer elements: " + ", ".join(missing))
    if intel_gpu:
        context = _create_shared_va_context(gst)
        device = context.get_structure().get_string("path")
        print(f"Shared VA context OK: {device}")
    print(f"Native capture runtime OK: {gst.version_string()}, gvadetect available")


if __name__ == "__main__":
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    _apply_dlstreamer_env()
    if os.environ.get("LD_LIBRARY_PATH", "") != previous:
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intel-gpu", action="store_true", help="also verify VA introspection and render-device access")
    args = parser.parse_args()
    try:
        check_runtime(intel_gpu=args.intel_gpu)
    except Exception as exc:
        install_option = " --intel-gpu" if args.intel_gpu else ""
        print(f"Native capture runtime unavailable: {exc}\n"
              f"On Ubuntu 24.04 amd64, run: sudo bash scripts/install-native-runtime.sh{install_option}\n"
              "Then rerun this check with /usr/bin/python3 as the service user.", file=sys.stderr)
        if args.intel_gpu:
            print("Verify that the Intel render device is present and the service user can access it.", file=sys.stderr)
        raise SystemExit(1)

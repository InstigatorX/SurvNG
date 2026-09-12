"""Bounded real-VA context/reconnect check; synthetic frames, no cameras/models.

Run with /usr/bin/python3 as the service owner on an Intel GPU host.
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survng.dlstreamer_live import _apply_dlstreamer_env, _create_shared_va_context, _load_gstreamer


def main():
    gst = _load_gstreamer()
    import gi
    gi.require_version("GstVa", "1.0")
    from gi.repository import GstVa

    context = _create_shared_va_context(gst)
    display = context.get_structure().get_value("gst-display")
    pipelines = []

    def start(name):
        pipeline = gst.Pipeline.new(name)
        pipelines.append(pipeline)
        pipeline.set_context(context)
        branch = gst.parse_bin_from_description(
            "videotestsrc is-live=true ! "
            "video/x-raw,format=NV12,width=96,height=64,framerate=5/1 ! "
            "vapostproc ! video/x-raw(memory:VAMemory) ! "
            "appsink name=frames max-buffers=1 drop=true sync=false", False,
        )
        pipeline.add(branch)
        if pipeline.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"{name} failed to start")
        return pipeline, branch.get_by_name("frames")

    def sample(sink):
        frame = sink.emit("try-pull-sample", 5 * gst.SECOND)
        assert frame is not None, "VA pipeline stopped delivering frames"
        assert GstVa.va_memory_peek_display(frame.get_buffer().peek_memory(0)) == display
        return frame.get_buffer().pts

    try:
        live, live_sink = start("live")
        last_pts = sample(live_sink)
        for index in range(3):
            main_pipeline, main_sink = start(f"main-{index}")
            sample(main_sink)
            main_pipeline.set_state(gst.State.NULL)
            pipelines.remove(main_pipeline)
            for _ in range(5):
                pts = sample(live_sink)
                assert pts > last_pts, "live PTS must advance across main teardown"
                last_pts = pts
        print("PASS: VA buffers share one display across three main reconnects; live frames continue")
    finally:
        for pipeline in pipelines:
            pipeline.set_state(gst.State.NULL)


if __name__ == "__main__":
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    _apply_dlstreamer_env()
    if os.environ.get("LD_LIBRARY_PATH", "") != previous:
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())])
    main()

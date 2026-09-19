"""Private native gvadetect process for bounded incident-cover verification."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

from survng.dlstreamer_live import _apply_dlstreamer_env, _load_gstreamer, _require_detection_plugin, _detection_metadata
from survng.dlstreamer_model import configured_model


def main():
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    _apply_dlstreamer_env()
    if previous != os.environ.get("LD_LIBRARY_PATH", ""):
        os.execv(sys.executable, [sys.executable, "-m", "survng.native_evidence_verify", *sys.argv[1:]])
    Gst = _load_gstreamer()
    _require_detection_plugin(Gst)
    from gstgva import VideoFrame
    from gi.repository import GstVideo
    config = json.loads(Path(sys.argv[1]).read_text())
    with configured_model(Path(config["model"]), config["model_proc"], config["nms"]) as (model, model_proc):
        graph = Gst.parse_launch('appsrc name=input format=time is-live=true ! gvadetect name=detector device=CPU batch-size=1 nireq=1 inference-interval=1 pre-process-backend=opencv ! gvametaconvert add-empty-results=true ! appsink name=output sync=false')
        detector = graph.get_by_name("detector")
        detector.set_property("model", str(model))
        detector.set_property("threshold", config["threshold"])
        detector.set_property("ie-config", "NUM_STREAMS=1,INFERENCE_NUM_THREADS=2")
        if model_proc:
            detector.set_property("model-proc", str(model_proc))
        if config["labels_path"]:
            detector.set_property("labels-file", config["labels_path"])
        elif config["labels"]:
            detector.set_property("labels", ",".join(config["labels"]))
        source, sink, bus = graph.get_by_name("input"), graph.get_by_name("output"), graph.get_bus()
        sequence = 0
        try:
            for line in sys.stdin:
                request = json.loads(line)
                try:
                    sequence += 1
                    caps = Gst.Caps.from_string(f'video/x-raw,format=BGR,width={request["width"]},height={request["height"]},framerate=1/1')
                    source.set_property("caps", caps)
                    graph.set_state(Gst.State.PLAYING)
                    pixels = Path(request["pixels"]).read_bytes()
                    info = GstVideo.VideoInfo.new_from_caps(caps)
                    row_bytes = request["width"] * 3
                    if len(pixels) != row_bytes * request["height"]:
                        raise ValueError("invalid packed BGR frame size")
                    if info.stride[0] != row_bytes:
                        padding = bytes(info.stride[0] - row_bytes)
                        pixels = b"".join(pixels[y*row_bytes:(y+1)*row_bytes] + padding for y in range(request["height"]))
                    buffer = Gst.Buffer.new_wrapped(pixels)
                    buffer.pts = sequence * Gst.SECOND
                    buffer.duration = Gst.SECOND
                    if source.emit("push-buffer", buffer) != Gst.FlowReturn.OK:
                        raise RuntimeError("native evidence input rejected")
                    sample = sink.emit("try-pull-sample", 30 * Gst.SECOND)
                    if sample is None:
                        message = bus.pop_filtered(Gst.MessageType.ERROR)
                        raise RuntimeError(str(message.parse_error()) if message else "native evidence inference timed out")
                    result = _detection_metadata(sample, VideoFrame, inference_sequence=sequence, gst_second=Gst.SECOND, clock_time_none=Gst.CLOCK_TIME_NONE)
                except Exception as error:
                    result = {"error": str(error)}
                output = Path(request["result"])
                temporary = output.with_suffix(".tmp")
                temporary.write_text(json.dumps(result))
                temporary.replace(output)
        finally:
            graph.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()

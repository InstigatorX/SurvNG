"""Recognition check using a real local recording and the native runtime.

Unlike the synthetic plumbing smoke test, this fails if the configured model
silently returns empty detections. Run inside the Intel image on a GPU host.
No camera, application database, network, or recording writes are required.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survng.dlstreamer_live import (
    _apply_dlstreamer_env, _detection_metadata, _load_gstreamer,
    _prefer_decoder, _require_detection_plugin,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--expect-label", default="person")
    parser.add_argument("--min-positive-frames", type=int, default=3)
    parser.add_argument("--negative-after", type=float)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.timeout <= 0 or args.min_positive_frames < 1:
        parser.error("timeout and minimum positive frames must be positive")
    for path in (args.video, args.model, args.labels):
        if path is not None and not path.is_file():
            parser.error(f"file not found: {path}")
    old_path = os.environ.get("LD_LIBRARY_PATH", "")
    _apply_dlstreamer_env()
    if os.environ.get("LD_LIBRARY_PATH", "") != old_path:
        os.execv(sys.executable, [sys.executable, __file__, *sys.argv[1:]])
    Gst = _load_gstreamer()
    _prefer_decoder(Gst, "va")
    _require_detection_plugin(Gst)
    from gstgva import VideoFrame

    gpu = args.device == "GPU"
    conversion = ("vapostproc ! video/x-raw(memory:VAMemory),format=NV12" if gpu
                  else "videoconvert ! video/x-raw,format=BGR")
    labels = f" labels-file={json.dumps(str(args.labels))}" if args.labels else ""
    graph = (
        f"uridecodebin3 uri={json.dumps(args.video.resolve().as_uri())} ! "
        "videorate drop-only=true ! video/x-raw(ANY),framerate=1/1 ! "
        f"{conversion} ! gvadetect model={json.dumps(str(args.model))} "
        f"device={args.device} pre-process-backend={'va-surface-sharing' if gpu else 'opencv'} "
        "batch-size=1 nireq=1 inference-interval=1 threshold=0.25 "
        f"ie-config=PERFORMANCE_HINT=LATENCY,NUM_STREAMS=1{labels} ! "
        "queue ! gvametaconvert add-empty-results=true ! "
        "appsink name=results sync=false max-buffers=2"
    )
    pipeline = Gst.parse_launch(graph)
    sink, bus = pipeline.get_by_name("results"), pipeline.get_bus()
    frames = positives = negative_frames = false_positives = 0
    started = time.monotonic()
    eos = False
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("recognition pipeline failed to start")
        while time.monotonic() - started < args.timeout:
            sample = sink.emit("try-pull-sample", 200 * Gst.MSECOND)
            if sample is not None:
                frames += 1
                result = _detection_metadata(sample, VideoFrame, inference_sequence=frames,
                    gst_second=Gst.SECOND, clock_time_none=Gst.CLOCK_TIME_NONE)
                present = any(obj["label"] == args.expect_label for obj in result["objects"])
                positives += present
                if args.negative_after is not None and result["source_pts"] >= args.negative_after:
                    negative_frames += 1
                    false_positives += present
            message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message is not None:
                if message.type == Gst.MessageType.ERROR:
                    raise RuntimeError(str(message.parse_error()))
                eos = True
                break
    finally:
        pipeline.set_state(Gst.State.NULL)
    report = {"device": args.device, "frames": frames, "positive_frames": positives,
              "negative_frames": negative_frames, "negative_false_positives": false_positives,
              "eos": eos, "elapsed_seconds": round(time.monotonic() - started, 3)}
    print(json.dumps(report), flush=True)
    if not eos or positives < args.min_positive_frames:
        raise SystemExit("FAIL: recording incomplete or expected object not recognized")
    if args.negative_after is not None and (not negative_frames or false_positives):
        raise SystemExit("FAIL: negative window missing or contains unexpected objects")


if __name__ == "__main__":
    main()

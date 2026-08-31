"""Standalone GStreamer GPU-capture probe for SurvNG deployments."""

from __future__ import annotations

import argparse
import json
import math
import re
import resource
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit


DECODERS = {
    "va": ("vah264dec", "vah265dec"),
    "qsv": ("qsvh264dec", "qsvh265dec"),
}
_CREDENTIAL_URL_RE = re.compile(
    r"(\b(?:rtsp|rtsps|http|https)://)([^:/@\s]+):([^@\s]+)@",
    re.IGNORECASE,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure GStreamer capture through an appsink without exposing a "
            "credentialed camera URL in process arguments."
        )
    )
    parser.add_argument(
        "--url-file",
        default="-",
        help="file containing the camera URL, or - to read one line from stdin",
    )
    parser.add_argument(
        "--decoder",
        choices=("auto", "va", "qsv"),
        default="va",
        help="preferred hardware decoder family",
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--open-timeout", type=float, default=10.0)
    parser.add_argument(
        "--test-source",
        action="store_true",
        help="use a generated source to validate the probe without a camera",
    )
    return parser


def _read_camera_url(path: str, stdin: TextIO = sys.stdin) -> str:
    if path == "-":
        value = stdin.readline().strip()
    else:
        value = Path(path).read_text(encoding="utf-8").splitlines()[0].strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"rtsp", "rtsps", "http", "https"}:
        raise ValueError("camera URL must use RTSP, RTSPS, HTTP, or HTTPS")
    if not parsed.hostname:
        raise ValueError("camera URL must include a host")
    return value


def _frame_rate(value: float) -> Fraction:
    if not math.isfinite(value):
        raise ValueError("fps must be finite")
    return Fraction(min(10.0, max(0.5, value))).limit_denominator(1000)


def _positive_seconds(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _redact_error(value: object) -> str:
    return _CREDENTIAL_URL_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}:***@",
        str(value),
    )


def _load_gstreamer():
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


def _prefer_decoder(Gst, family: str) -> None:
    if family == "auto":
        preferred = (*DECODERS["va"], *DECODERS["qsv"])
    else:
        preferred = DECODERS[family]
    registry = Gst.Registry.get()
    for index, name in enumerate(preferred):
        feature = registry.find_feature(name, Gst.ElementFactory)
        if feature is not None:
            feature.set_rank(int(Gst.Rank.PRIMARY) + 100 - index)


def _element(Gst, factory: str, name: str):
    element = Gst.ElementFactory.make(factory, name)
    if element is None:
        raise RuntimeError(f"required GStreamer element is unavailable: {factory}")
    return element


def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    Gst = _load_gstreamer()
    rate = _frame_rate(args.fps)
    duration = _positive_seconds(args.duration, "duration")
    open_timeout = _positive_seconds(args.open_timeout, "open timeout")
    _prefer_decoder(Gst, args.decoder)

    pipeline = Gst.Pipeline.new("survng-gstreamer-capture-probe")
    if pipeline is None:
        raise RuntimeError("could not create GStreamer pipeline")
    source = _element(
        Gst,
        "videotestsrc" if args.test_source else "uridecodebin3",
        "source",
    )
    if args.test_source:
        source.set_property("is-live", True)
        source.set_property("pattern", "ball")
    else:
        source.set_property("uri", _read_camera_url(args.url_file))

    queue = _element(Gst, "queue", "capture-queue")
    queue.set_property("max-size-buffers", 1)
    queue.set_property("leaky", 2)
    videorate = _element(Gst, "videorate", "drop-only-rate")
    videorate.set_property("drop-only", True)
    convert = _element(Gst, "videoconvert", "system-memory-bgr")
    capsfilter = _element(Gst, "capsfilter", "capture-caps")
    capsfilter.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw,format=BGR,framerate={rate.numerator}/{rate.denominator}"
        ),
    )
    sink = _element(Gst, "appsink", "capture-sink")
    sink.set_property("emit-signals", False)
    sink.set_property("max-buffers", 1)
    sink.set_property("drop", True)
    sink.set_property("sync", False)

    for element in (source, queue, videorate, convert, capsfilter, sink):
        pipeline.add(element)
    for left, right in (
        (queue, videorate),
        (videorate, convert),
        (convert, capsfilter),
        (capsfilter, sink),
    ):
        if not left.link(right):
            raise RuntimeError(
                f"could not link {left.get_name()} to {right.get_name()}"
            )

    linked = False

    def link_video_pad(_source, pad) -> None:
        nonlocal linked
        if linked:
            return
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() < 1:
            return
        if not caps.get_structure(0).get_name().startswith("video/"):
            return
        result = pad.link(queue.get_static_pad("sink"))
        linked = result == Gst.PadLinkReturn.OK

    if args.test_source:
        if not source.link(queue):
            raise RuntimeError("could not link generated source")
        linked = True
    else:
        source.connect("pad-added", link_video_pad)

    decoder_elements: set[str] = set()

    def remember_element(_pipeline, _sub_bin, element) -> None:
        factory = element.get_factory()
        if factory is None:
            return
        name = factory.get_name()
        if "dec" in name:
            decoder_elements.add(name)

    pipeline.connect("deep-element-added", remember_element)
    bus = pipeline.get_bus()
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    started = time.monotonic()
    cpu_started = time.process_time()
    first_frame_at: float | None = None
    last_frame_at: float | None = None
    frames = 0
    mapped_bytes = 0
    width = 0
    height = 0
    error = ""
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer pipeline failed to enter PLAYING state")
        while not stop_requested:
            now = time.monotonic()
            if first_frame_at is None and now - started >= open_timeout:
                raise TimeoutError("GStreamer first frame timed out")
            if first_frame_at is not None and now - first_frame_at >= duration:
                break
            message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message is not None:
                if message.type == Gst.MessageType.ERROR:
                    parsed_error, debug = message.parse_error()
                    error = str(parsed_error)
                    if debug:
                        error = f"{error}: {debug[-400:]}"
                    raise RuntimeError(error)
                break
            sample = sink.emit("try-pull-sample", 200 * Gst.MSECOND)
            if sample is None:
                continue
            sample_at = time.monotonic()
            if (
                first_frame_at is not None
                and sample_at - first_frame_at > duration
            ):
                break
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            mapped, info = buffer.map(Gst.MapFlags.READ)
            if not mapped:
                raise RuntimeError("could not map GStreamer frame")
            try:
                mapped_bytes += int(info.size)
                if info.size:
                    _ = info.data[0]
            finally:
                buffer.unmap(info)
            frames += 1
            if first_frame_at is None:
                first_frame_at = sample_at
            last_frame_at = sample_at
    finally:
        pipeline.set_state(Gst.State.NULL)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    finished = time.monotonic()
    selected = sorted(decoder_elements)
    hardware_selected = (
        args.test_source
        or args.decoder == "auto"
        or any(name.startswith(args.decoder) for name in selected)
    )
    return {
        "ok": bool(frames and hardware_selected),
        "decoder_requested": args.decoder,
        "decoder_elements": selected,
        "hardware_decoder_selected": hardware_selected,
        "frames": frames,
        "width": width,
        "height": height,
        "first_frame_ms": (
            round((first_frame_at - started) * 1000.0, 3)
            if first_frame_at is not None
            else None
        ),
        "sample_seconds": round(
            max(0.0, (last_frame_at or finished) - (first_frame_at or finished)),
            3,
        ),
        "output_fps": round(
            max(0, frames - 1)
            / max(
                0.001,
                (last_frame_at or finished) - (first_frame_at or started),
            ),
            3,
        ),
        "mapped_bytes": mapped_bytes,
        "cpu_seconds": round(time.process_time() - cpu_started, 3),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "gstreamer_version": Gst.version_string(),
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = _run_probe(args)
    except Exception as exc:
        result = {"ok": False, "error": _redact_error(exc)}
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

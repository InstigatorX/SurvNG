"""Isolated GStreamer / DL Streamer live pipeline. URL is read from stdin."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import signal
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit


DECODERS = {
    "va": ("vah264dec", "vah265dec"),
    "auto": ("vah264dec", "vah265dec", "avdec_h264", "avdec_h265"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode a camera URL with GStreamer, optionally run gvadetect on "
            "VAMemory, and emit BGR frames for SurvNG. The URL is read from "
            "stdin so it never appears in process arguments."
        )
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--open-timeout", type=float, default=3.0)
    parser.add_argument("--rtsp-transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--decoder", choices=("auto", "va"), default="va")
    parser.add_argument("--model", default="", help="OpenVINO IR XML for gvadetect")
    parser.add_argument("--device", default="GPU")
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="emit frames only; do not attach gvadetect",
    )
    parser.add_argument(
        "--test-source",
        action="store_true",
        help="use videotestsrc instead of a camera URL",
    )
    return parser


def _read_camera_url(stdin: TextIO = sys.stdin) -> str:
    value = stdin.readline().strip()
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


def _set_process_name() -> None:
    name = "survng-dls"
    try:
        Path("/proc/self/comm").write_text(name, encoding="utf-8")
    except OSError:
        pass


def _disable_core_dumps() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        pass


def _apply_dlstreamer_env() -> None:
    root = Path("/opt/intel/dlstreamer")
    if not root.is_dir():
        return
    plugin_dirs = [
        str(root / "lib"),
        str(root / "gstreamer/lib/gstreamer-1.0"),
        str(root / "gstreamer/lib"),
        "/usr/lib/x86_64-linux-gnu/gstreamer-1.0",
    ]
    current = os.environ.get("GST_PLUGIN_PATH", "")
    os.environ["GST_PLUGIN_PATH"] = ":".join(
        part for part in (*plugin_dirs, current) if part
    )
    lib_dirs = [
        str(root / "gstreamer/lib"),
        str(root / "lib"),
        str(root / "lib/gstreamer-1.0"),
    ]
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        part for part in (*lib_dirs, current_ld) if part
    )
    os.environ.setdefault("LIBVA_DRIVER_NAME", "iHD")
    os.environ.setdefault("GST_VA_ALL_DRIVERS", "1")


def _load_gstreamer():
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


def _prefer_decoder(Gst, family: str) -> None:
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


def _factory_available(Gst, name: str) -> bool:
    return Gst.ElementFactory.find(name) is not None


def _link_tee(Gst, tee, sink) -> None:
    pad = tee.get_request_pad("src_%u")
    if pad is None:
        raise RuntimeError("could not request GStreamer tee pad")
    result = pad.link(sink.get_static_pad("sink"))
    if result != Gst.PadLinkReturn.OK:
        raise RuntimeError(
            f"could not link tee to {sink.get_name()}: {int(result)}"
        )


def _normalize_gva_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for item in payload.get("objects") or ():
        if not isinstance(item, dict):
            continue
        detection = item.get("detection") if isinstance(item.get("detection"), dict) else item
        label = str(detection.get("label") or item.get("label") or "").strip()
        try:
            confidence = float(detection.get("confidence", item.get("confidence", 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        x = item.get("x", item.get("x_min"))
        y = item.get("y", item.get("y_min"))
        width = item.get("w", item.get("width"))
        height = item.get("h", item.get("height"))
        if None in (x, y, width, height):
            box = item.get("box")
            if isinstance(box, dict):
                x1, y1, x2, y2 = box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")
            else:
                continue
        else:
            try:
                x1 = float(x)
                y1 = float(y)
                x2 = x1 + float(width)
                y2 = y1 + float(height)
            except (TypeError, ValueError):
                continue
        try:
            objects.append(
                {
                    "label": label or "object",
                    "confidence": round(confidence, 4),
                    "box": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    },
                }
            )
        except (TypeError, ValueError):
            continue
    return objects


def _write(stdout, message: bytes) -> None:
    stdout.write(message)
    stdout.flush()


def run(argv: list[str] | None = None) -> int:
    from survng.app.dlstreamer_protocol import (
        TYPE_DETECTIONS,
        TYPE_STATUS,
        encode_frame,
        encode_json,
    )

    _disable_core_dumps()
    _set_process_name()
    _apply_dlstreamer_env()
    args = _parser().parse_args(argv)
    rate = _frame_rate(args.fps)
    open_timeout = _positive_seconds(args.open_timeout, "open timeout")
    stdout = sys.stdout.buffer
    Gst = _load_gstreamer()
    _prefer_decoder(Gst, args.decoder)

    model_path = Path(args.model).expanduser() if args.model else None
    detect = (
        not args.no_detect
        and model_path is not None
        and model_path.is_file()
        and _factory_available(Gst, "gvadetect")
    )

    pipeline = Gst.Pipeline.new("survng-dlstreamer-live")
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
        source.set_property("uri", _read_camera_url())

        def configure_rtsp(_bin, element) -> None:
            if element.find_property("protocols") is not None:
                try:
                    element.set_property("protocols", args.rtsp_transport)
                except Exception:
                    if args.rtsp_transport == "tcp":
                        element.set_property("protocols", 4)

        source.connect("source-setup", configure_rtsp)

    tee = _element(Gst, "tee", "branches")
    frame_queue = _element(Gst, "queue", "frame-queue")
    frame_queue.set_property("max-size-buffers", 1)
    frame_queue.set_property("leaky", 2)
    videorate = _element(Gst, "videorate", "drop-only-rate")
    videorate.set_property("drop-only", True)
    convert = _element(Gst, "videoconvert", "system-memory-bgr")
    capsfilter = _element(Gst, "capsfilter", "frame-caps")
    capsfilter.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw,format=BGR,framerate={rate.numerator}/{rate.denominator}"
        ),
    )
    sink = _element(Gst, "appsink", "frame-sink")
    sink.set_property("emit-signals", False)
    sink.set_property("max-buffers", 1)
    sink.set_property("drop", True)
    sink.set_property("sync", False)

    elements = [source, tee, frame_queue, videorate, convert, capsfilter, sink]
    meta_sink = None
    detect_queue = None
    detector = None
    meta_convert = None
    if detect:
        detect_queue = _element(Gst, "queue", "detect-queue")
        detect_queue.set_property("max-size-buffers", 2)
        detector = _element(Gst, "gvadetect", "detect")
        detector.set_property("model", str(model_path))
        detector.set_property("device", args.device)
        preprocess = "va" if args.decoder == "va" else "opencv"
        try:
            detector.set_property("pre-process-backend", preprocess)
        except Exception:
            pass
        elements.extend([detect_queue, detector])
        if _factory_available(Gst, "gvametaconvert"):
            meta_convert = _element(Gst, "gvametaconvert", "detect-meta")
            try:
                meta_convert.set_property("format", "json")
            except Exception:
                pass
            meta_sink = _element(Gst, "appsink", "meta-sink")
            meta_sink.set_property("emit-signals", False)
            meta_sink.set_property("max-buffers", 1)
            meta_sink.set_property("drop", True)
            meta_sink.set_property("sync", False)
            elements.extend([meta_convert, meta_sink])
        else:
            fake = _element(Gst, "fakesink", "detect-sink")
            fake.set_property("sync", False)
            elements.append(fake)

    for element in elements:
        pipeline.add(element)

    for left, right in (
        (frame_queue, videorate),
        (videorate, convert),
        (convert, capsfilter),
        (capsfilter, sink),
    ):
        if not left.link(right):
            raise RuntimeError(
                f"could not link {left.get_name()} to {right.get_name()}"
            )
    if detect and detect_queue is not None and detector is not None:
        if not detect_queue.link(detector):
            raise RuntimeError("could not link detect queue")
        tail = detector
        if meta_convert is not None and meta_sink is not None:
            if not detector.link(meta_convert) or not meta_convert.link(meta_sink):
                raise RuntimeError("could not link detection metadata branch")
        else:
            fake = pipeline.get_by_name("detect-sink")
            if fake is None or not detector.link(fake):
                raise RuntimeError("could not link detection sink")

    linked = False

    def link_decoded_pad(_source, pad) -> None:
        nonlocal linked
        if linked:
            return
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() < 1:
            return
        if not caps.get_structure(0).get_name().startswith("video/"):
            return
        result = pad.link(tee.get_static_pad("sink"))
        if result != Gst.PadLinkReturn.OK:
            return
        _link_tee(Gst, tee, frame_queue)
        if detect_queue is not None:
            _link_tee(Gst, tee, detect_queue)
        linked = True

    if args.test_source:
        if not source.link(tee):
            raise RuntimeError("could not link generated source")
        _link_tee(Gst, tee, frame_queue)
        if detect_queue is not None:
            _link_tee(Gst, tee, detect_queue)
        linked = True
    else:
        source.connect("pad-added", link_decoded_pad)

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
    latest_objects: list[dict[str, Any]] = []
    objects_lock = threading.Lock()
    started = time.monotonic()
    first_frame_at: float | None = None
    sequence = 0
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer pipeline failed to enter PLAYING state")
        while not stop_requested:
            now = time.monotonic()
            if first_frame_at is None and now - started >= open_timeout:
                raise TimeoutError("DL Streamer first frame timed out")
            message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message is not None:
                if message.type == Gst.MessageType.ERROR:
                    parsed_error, debug = message.parse_error()
                    error = str(parsed_error)
                    if debug:
                        error = f"{error}: {debug[-400:]}"
                    raise RuntimeError(error)
                break
            if meta_sink is not None:
                meta_sample = meta_sink.emit("try-pull-sample", 0)
                if meta_sample is not None:
                    buffer = meta_sample.get_buffer()
                    mapped, info = buffer.map(Gst.MapFlags.READ)
                    if mapped:
                        try:
                            payload = json.loads(bytes(info.data).decode("utf-8"))
                            if isinstance(payload, dict):
                                with objects_lock:
                                    latest_objects[:] = _normalize_gva_objects(payload)
                        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                            pass
                        finally:
                            buffer.unmap(info)
            sample = sink.emit("try-pull-sample", 200 * Gst.MSECOND)
            if sample is None:
                continue
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            mapped, info = buffer.map(Gst.MapFlags.READ)
            if not mapped:
                raise RuntimeError("could not map GStreamer frame")
            try:
                pixels = bytes(info.data)
            finally:
                buffer.unmap(info)
            expected = width * height * 3
            if len(pixels) < expected:
                raise RuntimeError("GStreamer frame was truncated")
            pixels = pixels[:expected]
            if first_frame_at is None:
                first_frame_at = time.monotonic()
                _write(
                    stdout,
                    encode_json(
                        TYPE_STATUS,
                        {
                            "ok": True,
                            "detect": detect,
                            "decoder_elements": sorted(decoder_elements),
                        },
                    ),
                )
            sequence += 1
            _write(
                stdout,
                encode_frame(
                    width=width,
                    height=height,
                    sequence=sequence,
                    pts=time.monotonic() - started,
                    pixels=pixels,
                ),
            )
            with objects_lock:
                objects = list(latest_objects)
            if objects:
                _write(
                    stdout,
                    encode_json(
                        TYPE_DETECTIONS,
                        {
                            "objects": objects,
                            "decoder_elements": sorted(decoder_elements),
                        },
                    ),
                )
    finally:
        pipeline.set_state(Gst.State.NULL)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


def main(argv: list[str] | None = None) -> int:
    from survng.app.dlstreamer_protocol import TYPE_STATUS, encode_json
    from survng.app.security import redact_secret_text

    try:
        return run(argv)
    except Exception as exc:
        sys.stdout.buffer.write(
            encode_json(TYPE_STATUS, {"ok": False, "error": redact_secret_text(exc)})
        )
        sys.stdout.buffer.flush()
        print(redact_secret_text(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

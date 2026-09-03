"""Isolated GStreamer / DL Streamer live pipelines. URLs are read from stdin."""

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
            "Decode camera URLs with GStreamer, optionally run gvadetect on "
            "VAMemory, and emit a 320-wide grayscale qualifier plus JPEG "
            "preview frames. URLs are read from stdin so they never appear "
            "in process arguments. Supervisor mode hosts every camera in one "
            "process so gvadetect shares model-instance-id."
        )
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument(
        "--detect-fps",
        type=float,
        default=5.0,
        help="maximum gvadetect input rate; independent from EMA qualification FPS",
    )
    parser.add_argument("--open-timeout", type=float, default=3.0)
    parser.add_argument("--rtsp-transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--decoder", choices=("auto", "va"), default="va")
    parser.add_argument("--model", default="", help="OpenVINO IR XML for gvadetect")
    parser.add_argument("--model-proc", default="", help="optional gvadetect model-proc JSON")
    parser.add_argument("--labels", default="", help="optional gvadetect labels file")
    parser.add_argument(
        "--model-instance-id",
        default="",
        help="gvadetect model-instance-id shared across supervisor streams",
    )
    parser.add_argument(
        "--supervisor",
        action="store_true",
        help="host multiple camera pipelines and share gvadetect",
    )
    parser.add_argument("--device", default="GPU")
    parser.add_argument(
        "--frame-width",
        type=int,
        default=320,
        help="grayscale qualifier width in pixels",
    )
    parser.add_argument(
        "--jpeg-fps",
        type=float,
        default=1.0,
        help="JPEG preview rate; 0 disables the JPEG branch",
    )
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
    return _validate_camera_url(stdin.readline().strip())


def _validate_camera_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"rtsp", "rtsps", "http", "https"}:
        raise ValueError("camera URL must use RTSP, RTSPS, HTTP, or HTTPS")
    if not parsed.hostname:
        raise ValueError("camera URL must include a host")
    return value


def model_instance_id(model_path: str, device: str, explicit: str = "") -> str:
    raw = str(explicit or "").strip()
    if not raw:
        stem = Path(model_path).stem if str(model_path or "").strip() else "detect"
        raw = f"survng-{stem}-{device or 'GPU'}"
    cleaned = "".join(
        ch if ch.isalnum() or ch in "-_" else "-" for ch in raw
    ).strip("-_")
    return (cleaned or "survng-detect")[:96]


def _pipeline_name(stream_id: str) -> str:
    suffix = "".join(ch if ch.isalnum() else "-" for ch in stream_id).strip("-")
    return f"survng-dls-{suffix[:24] or 'live'}"


def _set_optional_property(element, name: str, value: object) -> None:
    try:
        element.set_property(name, value)
    except Exception:
        pass


def _qualifier_width(value: int) -> int:
    return int(min(960, max(240, value)))


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


_SYSTEM_GST_PLUGINS = "/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
_URI_SOURCE_FACTORIES = ("uridecodebin3", "uridecodebin")


def _colon_path(*groups: str) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        for part in group.split(":"):
            if not part or part in seen:
                continue
            seen.add(part)
            ordered.append(part)
    return ":".join(ordered)


def _existing_dirs(*paths: Path | str) -> tuple[str, ...]:
    found: list[str] = []
    for path in paths:
        resolved = Path(path)
        if resolved.is_dir():
            found.append(str(resolved))
    return tuple(found)


def _drop_paths(value: str, *unwanted: str) -> str:
    skip = {part for part in unwanted if part}
    return _colon_path(*(part for part in value.split(":") if part and part not in skip))


def _set_gst_search_path(name: str, value: str) -> None:
    os.environ[name] = value
    os.environ[f"{name}_1_0"] = value


def _apply_dlstreamer_env() -> None:
    """Expose gvadetect without hiding Ubuntu's uridecodebin3.

    Intel DL Streamer's bundled libgstreamer compiles in a private system
    plugin path. If that tree is on LD_LIBRARY_PATH at all, python3-gi loads
    Intel's Gst before /usr/lib and Ubuntu playback plugins never register.
    Keep the distro plugin dir on GST_PLUGIN_SYSTEM_PATH, drop the nested
    Intel GStreamer lib dir, and force the 1.0-suffixed search variables
    that otherwise override the unsuffixed ones.
    """
    system_plugins = _colon_path(
        _SYSTEM_GST_PLUGINS,
        os.environ.get("GST_PLUGIN_SYSTEM_PATH_1_0", ""),
        os.environ.get("GST_PLUGIN_SYSTEM_PATH", ""),
    )
    _set_gst_search_path("GST_PLUGIN_SYSTEM_PATH", system_plugins)
    for scanner in (
        "/usr/lib/x86_64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner",
        "/usr/libexec/gstreamer-1.0/gst-plugin-scanner",
    ):
        if Path(scanner).is_file():
            os.environ["GST_PLUGIN_SCANNER"] = scanner
            break
    os.environ.setdefault("LIBVA_DRIVER_NAME", "iHD")
    os.environ.setdefault("GST_VA_ALL_DRIVERS", "1")
    root = Path("/opt/intel/dlstreamer")
    if not root.is_dir():
        return
    plugin_path = _colon_path(
        *_existing_dirs(
            root / "lib",
            root / "gstreamer/lib/gstreamer-1.0",
            Path(_SYSTEM_GST_PLUGINS),
        ),
        os.environ.get("GST_PLUGIN_PATH_1_0", ""),
        os.environ.get("GST_PLUGIN_PATH", ""),
    )
    _set_gst_search_path("GST_PLUGIN_PATH", plugin_path)
    intel_gst_lib = str(root / "gstreamer/lib")
    os.environ["LD_LIBRARY_PATH"] = _colon_path(
        _drop_paths(os.environ.get("LD_LIBRARY_PATH", ""), intel_gst_lib),
        *_existing_dirs(root / "lib", root / "lib/gstreamer-1.0"),
    )


def _load_gstreamer():
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    registry = Gst.Registry.get()
    if Path(_SYSTEM_GST_PLUGINS).is_dir():
        registry.scan_path(_SYSTEM_GST_PLUGINS)
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


def _make_live_source(Gst, *, test_source: bool):
    if test_source:
        return _element(Gst, "videotestsrc", "source"), "videotestsrc"
    missing: list[str] = []
    for name in _URI_SOURCE_FACTORIES:
        element = Gst.ElementFactory.make(name, "source")
        if element is not None:
            return element, name
        missing.append(name)
    raise RuntimeError(
        "required GStreamer element is unavailable: " + " or ".join(missing)
    )


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


def _packed_gray(pixels: bytes, width: int, height: int) -> bytes:
    expected = width * height
    if len(pixels) == expected:
        return pixels
    if height <= 0 or len(pixels) < expected:
        raise RuntimeError("GStreamer grayscale frame was truncated")
    stride = len(pixels) // height
    if stride < width:
        raise RuntimeError("GStreamer grayscale frame was truncated")
    return b"".join(pixels[row * stride : row * stride + width] for row in range(height))


def _write(
    stdout,
    message: bytes,
    *,
    lock: threading.Lock | None = None,
) -> None:
    if lock is None:
        stdout.write(message)
        stdout.flush()
        return
    with lock:
        stdout.write(message)
        stdout.flush()


def run(argv: list[str] | None = None) -> int:
    from survng.app.dlstreamer_protocol import (
        TYPE_DETECTIONS,
        TYPE_STATUS,
        encode_frame,
        encode_jpeg,
        encode_json,
    )

    _disable_core_dumps()
    _set_process_name()
    _apply_dlstreamer_env()
    args = _parser().parse_args(argv)
    rate = _frame_rate(args.fps)
    detect_rate = _frame_rate(args.detect_fps)
    qualifier_width = _qualifier_width(args.frame_width)
    jpeg_rate = _frame_rate(args.jpeg_fps) if args.jpeg_fps > 0 else None
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
    instance_id = (
        model_instance_id(str(model_path or ""), args.device, args.model_instance_id)
        if detect
        else ""
    )
    if args.supervisor:
        return _run_supervisor(
            Gst,
            args,
            detect=detect,
            model_path=model_path,
            instance_id=instance_id,
            rate=rate,
            detect_rate=detect_rate,
            qualifier_width=qualifier_width,
            jpeg_rate=jpeg_rate,
            open_timeout=open_timeout,
            stdout=stdout,
        )
    url = "" if args.test_source else _read_camera_url()
    return _pump_pipeline(
        Gst,
        args,
        url=url,
        stream_id="",
        detect=detect,
        model_path=model_path,
        instance_id=instance_id,
        rate=rate,
        detect_rate=detect_rate,
        qualifier_width=qualifier_width,
        jpeg_rate=jpeg_rate,
        open_timeout=open_timeout,
        stdout=stdout,
        stdout_lock=None,
        stop_event=None,
        encode_frame=encode_frame,
        encode_jpeg=encode_jpeg,
        encode_json=encode_json,
        TYPE_DETECTIONS=TYPE_DETECTIONS,
        TYPE_STATUS=TYPE_STATUS,
    )


def _run_supervisor(
    Gst,
    args,
    *,
    detect: bool,
    model_path: Path | None,
    instance_id: str,
    rate: Fraction,
    detect_rate: Fraction,
    qualifier_width: int,
    jpeg_rate: Fraction | None,
    open_timeout: float,
    stdout,
) -> int:
    from survng.app.dlstreamer_protocol import (
        TYPE_DETECTIONS,
        TYPE_STATUS,
        encode_frame,
        encode_jpeg,
        encode_json,
    )
    from survng.app.redact import redact_secret_text

    stdout_lock = threading.Lock()
    stop_all = threading.Event()
    workers: dict[str, tuple[threading.Event, threading.Thread]] = {}
    workers_lock = threading.Lock()
    print(
        f"survng-dls supervisor model_instance_id={instance_id or 'none'}",
        file=sys.stderr,
        flush=True,
    )

    def request_stop(_signum=None, _frame=None) -> None:
        stop_all.set()
        with workers_lock:
            for event, _thread in workers.values():
                event.set()

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def stop_stream(stream_id: str) -> None:
        with workers_lock:
            worker = workers.pop(stream_id, None)
        if worker is None:
            return
        event, thread = worker
        event.set()
        thread.join(timeout=2.0)

    def start_stream(stream_id: str, url: str) -> None:
        stop_stream(stream_id)
        event = threading.Event()

        def target() -> None:
            try:
                _pump_pipeline(
                    Gst,
                    args,
                    url=url,
                    stream_id=stream_id,
                    detect=detect,
                    model_path=model_path,
                    instance_id=instance_id,
                    rate=rate,
                    detect_rate=detect_rate,
                    qualifier_width=qualifier_width,
                    jpeg_rate=jpeg_rate,
                    open_timeout=open_timeout,
                    stdout=stdout,
                    stdout_lock=stdout_lock,
                    stop_event=event,
                    encode_frame=encode_frame,
                    encode_jpeg=encode_jpeg,
                    encode_json=encode_json,
                    TYPE_DETECTIONS=TYPE_DETECTIONS,
                    TYPE_STATUS=TYPE_STATUS,
                    install_signals=False,
                    test_source=False,
                )
            except Exception as exc:
                _write(
                    stdout,
                    encode_json(
                        TYPE_STATUS,
                        {"ok": False, "error": redact_secret_text(exc)},
                        stream_id=stream_id,
                    ),
                    lock=stdout_lock,
                )

        thread = threading.Thread(
            target=target,
            name=f"survng-dls-{stream_id[:16]}",
            daemon=True,
        )
        with workers_lock:
            workers[stream_id] = (event, thread)
        thread.start()

    try:
        while not stop_all.is_set():
            line = sys.stdin.buffer.readline()
            if not line:
                break
            try:
                command = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(command, dict):
                continue
            operation = str(command.get("op") or "").strip()
            stream_id = str(command.get("stream_id") or "").strip()
            if not stream_id:
                continue
            if operation == "remove":
                stop_stream(stream_id)
                continue
            if operation != "add":
                continue
            try:
                url = _validate_camera_url(str(command.get("url") or ""))
            except ValueError as exc:
                _write(
                    stdout,
                    encode_json(
                        TYPE_STATUS,
                        {"ok": False, "error": redact_secret_text(exc)},
                        stream_id=stream_id,
                    ),
                    lock=stdout_lock,
                )
                continue
            start_stream(stream_id, url)
    finally:
        request_stop()
        with workers_lock:
            remaining = list(workers.items())
            workers.clear()
        for _stream_id, (event, thread) in remaining:
            event.set()
            thread.join(timeout=2.0)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


def _pump_pipeline(
    Gst,
    args,
    *,
    url: str,
    stream_id: str,
    detect: bool,
    model_path: Path | None,
    instance_id: str,
    rate: Fraction,
    detect_rate: Fraction,
    qualifier_width: int,
    jpeg_rate: Fraction | None,
    open_timeout: float,
    stdout,
    stdout_lock: threading.Lock | None,
    stop_event: threading.Event | None,
    encode_frame,
    encode_jpeg,
    encode_json,
    TYPE_DETECTIONS,
    TYPE_STATUS,
    install_signals: bool = True,
    test_source: bool | None = None,
) -> int:
    use_test_source = args.test_source if test_source is None else test_source
    pipeline = Gst.Pipeline.new(_pipeline_name(stream_id))
    if pipeline is None:
        raise RuntimeError("could not create GStreamer pipeline")

    source, source_factory = _make_live_source(Gst, test_source=use_test_source)
    print(
        f"survng-dls source_element={source_factory}"
        + (f" stream_id={stream_id}" if stream_id else ""),
        file=sys.stderr,
        flush=True,
    )
    if use_test_source:
        source.set_property("is-live", True)
        source.set_property("pattern", "ball")
    else:
        source.set_property("uri", url)

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
    convert = _element(Gst, "videoconvert", "qualifier-gray")
    scale = _element(Gst, "videoscale", "qualifier-scale")
    capsfilter = _element(Gst, "capsfilter", "frame-caps")
    capsfilter.set_property(
        "caps",
        Gst.Caps.from_string(
            "video/x-raw,format=GRAY8,width="
            f"{qualifier_width},framerate={rate.numerator}/{rate.denominator}"
        ),
    )
    sink = _element(Gst, "appsink", "frame-sink")
    sink.set_property("emit-signals", False)
    sink.set_property("max-buffers", 1)
    sink.set_property("drop", True)
    sink.set_property("sync", False)

    elements = [source, tee, frame_queue, videorate, convert, scale, capsfilter, sink]
    jpeg_queue = None
    jpeg_rate_el = None
    jpeg_convert = None
    jpeg_encoder = None
    jpeg_sink = None
    if jpeg_rate is not None and _factory_available(Gst, "jpegenc"):
        jpeg_queue = _element(Gst, "queue", "jpeg-queue")
        jpeg_queue.set_property("max-size-buffers", 1)
        jpeg_queue.set_property("leaky", 2)
        jpeg_rate_el = _element(Gst, "videorate", "jpeg-rate")
        jpeg_rate_el.set_property("drop-only", True)
        jpeg_convert = _element(Gst, "videoconvert", "jpeg-convert")
        jpeg_caps = _element(Gst, "capsfilter", "jpeg-caps")
        jpeg_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw,format=I420,framerate="
                f"{jpeg_rate.numerator}/{jpeg_rate.denominator}"
            ),
        )
        jpeg_encoder = _element(Gst, "jpegenc", "jpeg-preview")
        try:
            jpeg_encoder.set_property("quality", 80)
        except Exception:
            pass
        jpeg_sink = _element(Gst, "appsink", "jpeg-sink")
        jpeg_sink.set_property("emit-signals", False)
        jpeg_sink.set_property("max-buffers", 1)
        jpeg_sink.set_property("drop", True)
        jpeg_sink.set_property("sync", False)
        elements.extend(
            [jpeg_queue, jpeg_rate_el, jpeg_convert, jpeg_caps, jpeg_encoder, jpeg_sink]
        )
    meta_sink = None
    detect_queue = None
    detect_rate_el = None
    detect_rate_caps = None
    detector = None
    detect_output_queue = None
    meta_convert = None
    va_caps = None
    preprocess = ""
    if detect:
        detect_queue = _element(Gst, "queue", "detect-queue")
        detect_queue.set_property("max-size-buffers", 1)
        detect_queue.set_property("max-size-bytes", 0)
        detect_queue.set_property("max-size-time", 0)
        detect_queue.set_property("leaky", 2)
        detect_rate_el = _element(Gst, "videorate", "detect-rate")
        detect_rate_el.set_property("drop-only", True)
        detect_rate_caps = _element(Gst, "capsfilter", "detect-rate-caps")
        detect_rate_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw"
                + ("(memory:VAMemory)" if args.decoder == "va" and not use_test_source else "")
                + ",framerate="
                f"{detect_rate.numerator}/{detect_rate.denominator}"
            ),
        )
        detector = _element(Gst, "gvadetect", "detect")
        detector.set_property("model", str(model_path))
        detector.set_property("device", args.device)
        detector.set_property("inference-interval", 1)
        preprocess = "va" if args.decoder == "va" and not use_test_source else "opencv"
        _set_optional_property(detector, "pre-process-backend", preprocess)
        if instance_id:
            _set_optional_property(detector, "model-instance-id", instance_id)
            _set_optional_property(detector, "scheduling-policy", "latency")
        if args.model_proc:
            _set_optional_property(detector, "model-proc", args.model_proc)
        if args.labels:
            for property_name in ("labels", "labels-file"):
                try:
                    detector.set_property(property_name, args.labels)
                    break
                except Exception:
                    continue
        detect_output_queue = _element(Gst, "queue", "detect-output-queue")
        detect_output_queue.set_property("max-size-buffers", 1)
        detect_output_queue.set_property("max-size-bytes", 0)
        detect_output_queue.set_property("max-size-time", 0)
        detect_output_queue.set_property("leaky", 2)
        elements.extend([detect_queue, detect_rate_el, detect_rate_caps, detector, detect_output_queue])
        if preprocess == "va":
            va_caps = _element(Gst, "capsfilter", "detect-va-memory")
            va_caps.set_property(
                "caps",
                Gst.Caps.from_string("video/x-raw(memory:VAMemory)"),
            )
            elements.append(va_caps)
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
        (convert, scale),
        (scale, capsfilter),
        (capsfilter, sink),
    ):
        if not left.link(right):
            raise RuntimeError(
                f"could not link {left.get_name()} to {right.get_name()}"
            )
    if (
        jpeg_queue is not None
        and jpeg_rate_el is not None
        and jpeg_convert is not None
        and jpeg_encoder is not None
        and jpeg_sink is not None
    ):
        jpeg_caps = pipeline.get_by_name("jpeg-caps")
        if jpeg_caps is None:
            raise RuntimeError("could not find JPEG caps")
        for left, right in (
            (jpeg_queue, jpeg_rate_el),
            (jpeg_rate_el, jpeg_convert),
            (jpeg_convert, jpeg_caps),
            (jpeg_caps, jpeg_encoder),
            (jpeg_encoder, jpeg_sink),
        ):
            if not left.link(right):
                raise RuntimeError(
                    f"could not link {left.get_name()} to {right.get_name()}"
                )
    if detect and detect_queue is not None and detector is not None:
        if va_caps is not None:
            if (
                detect_rate_el is None
                or detect_rate_caps is None
                or not detect_queue.link(detect_rate_el)
                or not detect_rate_el.link(detect_rate_caps)
                or not detect_rate_caps.link(va_caps)
                or not va_caps.link(detector)
            ):
                raise RuntimeError("could not link VAMemory detect caps")
        elif (
            detect_rate_el is None
            or detect_rate_caps is None
            or not detect_queue.link(detect_rate_el)
            or not detect_rate_el.link(detect_rate_caps)
            or not detect_rate_caps.link(detector)
        ):
            raise RuntimeError("could not link detect queue")
        if detect_output_queue is None or not detector.link(detect_output_queue):
            raise RuntimeError("could not link gvadetect output queue")
        if meta_convert is not None and meta_sink is not None:
            if not detect_output_queue.link(meta_convert) or not meta_convert.link(meta_sink):
                raise RuntimeError("could not link detection metadata branch")
        else:
            fake = pipeline.get_by_name("detect-sink")
            if fake is None or not detect_output_queue.link(fake):
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
        if jpeg_queue is not None:
            _link_tee(Gst, tee, jpeg_queue)
        linked = True

    if use_test_source:
        if not source.link(tee):
            raise RuntimeError("could not link generated source")
        _link_tee(Gst, tee, frame_queue)
        if detect_queue is not None:
            _link_tee(Gst, tee, detect_queue)
        if jpeg_queue is not None:
            _link_tee(Gst, tee, jpeg_queue)
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
    local_stop = stop_event or threading.Event()
    previous_handlers: dict[int, object] = {}

    def request_stop(_signum, _frame) -> None:
        local_stop.set()

    if install_signals:
        previous_handlers = {
            signum: signal.signal(signum, request_stop)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
    inference_sequence = 0
    started = time.monotonic()
    first_frame_at: float | None = None
    sequence = 0
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer pipeline failed to enter PLAYING state")
        while not local_stop.is_set():
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
                                caps = meta_sample.get_caps()
                                structure = caps.get_structure(0)
                                width = int(structure.get_value("width") or 0)
                                height = int(structure.get_value("height") or 0)
                                pts = float(buffer.pts) / float(Gst.SECOND)
                                if (
                                    width > 0
                                    and height > 0
                                    and buffer.pts != Gst.CLOCK_TIME_NONE
                                    and math.isfinite(pts)
                                ):
                                    inference_sequence += 1
                                    _write(
                                        stdout,
                                        encode_json(
                                            TYPE_DETECTIONS,
                                            {
                                                "schema_version": 1,
                                                "source_pts": pts,
                                                "inference_sequence": inference_sequence,
                                                "width": width,
                                                "height": height,
                                                # Empty is authoritative; it must clear
                                                # a preceding positive snapshot.
                                                "objects": _normalize_gva_objects(payload),
                                            },
                                            stream_id=stream_id,
                                        ),
                                        lock=stdout_lock,
                                    )
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
            pixels = _packed_gray(pixels, width, height)
            jpeg_bytes = b""
            jpeg_width = 0
            jpeg_height = 0
            if jpeg_sink is not None:
                jpeg_sample = jpeg_sink.emit("try-pull-sample", 0)
                if jpeg_sample is not None:
                    jpeg_buffer = jpeg_sample.get_buffer()
                    jpeg_caps = jpeg_sample.get_caps()
                    jpeg_structure = jpeg_caps.get_structure(0)
                    jpeg_width = int(jpeg_structure.get_value("width") or 0)
                    jpeg_height = int(jpeg_structure.get_value("height") or 0)
                    mapped, info = jpeg_buffer.map(Gst.MapFlags.READ)
                    if mapped:
                        try:
                            jpeg_bytes = bytes(info.data)
                        finally:
                            jpeg_buffer.unmap(info)
            if first_frame_at is None:
                first_frame_at = time.monotonic()
                selected = sorted(decoder_elements)
                _write(
                    stdout,
                    encode_json(
                        TYPE_STATUS,
                        {
                            "ok": True,
                            "detect": detect,
                            "decoder_elements": selected,
                            "source_element": source_factory,
                            "hardware_decoder_selected": any(
                                name.startswith("va") for name in selected
                            ),
                            "preprocess_backend": preprocess,
                            "first_frame_ms": round(
                                (first_frame_at - started) * 1000.0,
                                3,
                            ),
                            "qualifier_format": "GRAY8",
                            "qualifier_width": qualifier_width,
                            "detect_fps": float(detect_rate),
                            "inference_interval": 1,
                            "jpeg_preview": jpeg_sink is not None,
                            "model_instance_id": instance_id,
                            "shared_detect": bool(instance_id and stream_id),
                        },
                        stream_id=stream_id,
                    ),
                    lock=stdout_lock,
                )
            sequence += 1
            _write(
                stdout,
                encode_frame(
                    width=width,
                    height=height,
                    sequence=sequence,
                    pts=(
                        float(buffer.pts) / float(Gst.SECOND)
                        if buffer.pts != Gst.CLOCK_TIME_NONE
                        else float("nan")
                    ),
                    pixels=pixels,
                    stream_id=stream_id,
                ),
                lock=stdout_lock,
            )
            if jpeg_bytes:
                _write(
                    stdout,
                    encode_jpeg(
                        width=max(1, jpeg_width),
                        height=max(1, jpeg_height),
                        sequence=sequence,
                        pts=(
                            float(jpeg_buffer.pts) / float(Gst.SECOND)
                            if jpeg_buffer.pts != Gst.CLOCK_TIME_NONE
                            else float("nan")
                        ),
                        jpeg=jpeg_bytes,
                        stream_id=stream_id,
                    ),
                    lock=stdout_lock,
                )
    finally:
        pipeline.set_state(Gst.State.NULL)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


def main(argv: list[str] | None = None) -> int:
    from survng.app.dlstreamer_protocol import TYPE_STATUS, encode_json
    from survng.app.redact import redact_secret_text

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

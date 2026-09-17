"""Isolated GStreamer / DL Streamer live pipelines. URLs are read from stdin."""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from contextlib import ExitStack
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

from survng.openvino_config import GPU_COMPILATION_NUM_THREADS

DECODERS = {
    "va": ("vah264dec", "vah265dec"),
    "auto": ("vah264dec", "vah265dec", "avdec_h264", "avdec_h265"),
}

STREAM_STOP_TIMEOUT_SECONDS = 2.0


class StreamShutdownError(RuntimeError):
    """The native process must be replaced; admitting another graph is unsafe."""


def _tracking_classes_argument(value):
    try:
        labels = json.loads(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tracking classes must be a JSON array") from exc
    if (not isinstance(labels, list) or len(labels) > 256
            or any(not isinstance(label, str) or not label.strip() or len(label.strip()) > 128 for label in labels)):
        raise argparse.ArgumentTypeError("tracking classes must be an array of nonempty labels")
    return list(dict.fromkeys(label.strip().lower() for label in labels))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode camera URLs with GStreamer, optionally run gvadetect on "
            "VAMemory, and emit sampled color evidence plus JPEG "
            "preview frames. URLs are read from stdin so they never appear "
            "in process arguments. Supervisor mode hosts every camera in one "
            "process so gvadetect shares model-instance-id."
        )
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--main-fps", type=float, default=5.0)
    parser.add_argument("--source-role", choices=("live", "main"), default="live")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="retain low-confidence candidates for two-pass tracking")
    parser.add_argument("--nms-threshold", type=float, default=None,
                        help="override native model NMS IoU; final-output models remain NMS-free")
    parser.add_argument(
        "--detect-fps",
        type=float,
        default=5.0,
        help="maximum gvadetect input rate; independent from preview FPS",
    )
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=1,
        help="run gvadetect every Nth detect-branch frame (1-5)",
    )
    parser.add_argument(
        "--native-tracking",
        choices=("off", "short-term-imageless", "deep-sort"),
        default="off",
        help="optional DL Streamer ROI tracking between detector frames",
    )
    parser.add_argument("--reid-model", default="", help="OpenVINO person ReID IR XML for Deep SORT")
    parser.add_argument("--reid-device", default="CPU", help="OpenVINO device for Deep SORT ReID inference")
    parser.add_argument("--deep-sort-config", default="max_iou_distance=0.7,max_age=30,n_init=3,max_cosine_distance=0.2,nn_budget=100")
    parser.add_argument("--tracking-classes", type=_tracking_classes_argument, default=None,
                        help="JSON array of classes admitted to gvatrack; omitted means all, [] means none")
    parser.add_argument("--batch-size", type=int, default=1, choices=range(1, 5),
                        help="shared inference batch size; 1 disables batching")
    parser.add_argument("--open-timeout", type=float, default=3.0)
    parser.add_argument("--rtsp-transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--decoder", choices=("auto", "va"), default="va")
    parser.add_argument("--model", default="", help="OpenVINO IR XML for gvadetect")
    parser.add_argument("--model-proc", default="", help="optional gvadetect model-proc JSON")
    parser.add_argument("--labels", default="", help="optional gvadetect labels file")
    parser.add_argument("--labels-list", default="", help="comma-separated labels when no labels file is configured")
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
    parser.add_argument("--inference-requests", type=int, choices=range(1, 17), default=4)
    parser.add_argument("--inference-streams", type=int, choices=range(1, 9), default=2)
    parser.add_argument(
        "--frame-width",
        type=int,
        default=320,
        help="color evidence frame width in pixels; zero preserves native dimensions",
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
    return 0 if value == 0 else int(min(960, max(240, value)))


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
    """Keep libraries, introspection data, plugins and scanner in one runtime."""
    root = Path("/opt/intel/dlstreamer")
    bundle = root / "gstreamer"
    os.environ.setdefault("LIBVA_DRIVER_NAME", "iHD")
    os.environ.setdefault("GST_VA_ALL_DRIVERS", "1")
    if (bundle / "lib").is_dir():
        # Intel plugins can require APIs newer than Ubuntu's GStreamer.
        # Never put a distro scanner or core library in front of this bundle.
        _set_gst_search_path("GST_PLUGIN_SYSTEM_PATH", str(bundle / "lib/gstreamer-1.0"))
        _set_gst_search_path("GST_PLUGIN_PATH", str(root / "lib"))
        scanner = bundle / "bin/gstreamer-1.0/gst-plugin-scanner"
        if scanner.is_file():
            os.environ["GST_PLUGIN_SCANNER"] = str(scanner)
            os.environ["GST_PLUGIN_SCANNER_1_0"] = str(scanner)
        os.environ["LD_LIBRARY_PATH"] = _colon_path(
            str(bundle / "lib"), str(root / "lib"),
            # The APT package uses /opt/opencv; Intel's reference image also
            # supports dependencies nested under the DL Streamer prefix.
            *_existing_dirs(root / "opencv/lib", root / "rdkafka/lib",
                            "/opt/opencv", "/opt/rdkafka", "/opt/librealsense"),
            os.environ.get("LD_LIBRARY_PATH", ""),
        )
        os.environ["GI_TYPELIB_PATH"] = _colon_path(
            str(bundle / "lib/girepository-1.0"), str(root / "lib/girepository-1.0"),
            os.environ.get("GI_TYPELIB_PATH", ""),
        )
        python_dirs = _existing_dirs(root / "python", bundle / "lib/python3/dist-packages")
        os.environ["PYTHONPATH"] = _colon_path(*python_dirs, os.environ.get("PYTHONPATH", ""))
        for directory in reversed(python_dirs):
            if directory not in sys.path:
                sys.path.insert(0, directory)
    else:
        _set_gst_search_path("GST_PLUGIN_SYSTEM_PATH", _SYSTEM_GST_PLUGINS)
        if (root / "lib").is_dir():
            _set_gst_search_path("GST_PLUGIN_PATH", str(root / "lib"))
        if (root / "python").is_dir():
            sys.path.append(str(root / "python"))


def _load_gstreamer():
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    registry = Gst.Registry.get()
    if not Path("/opt/intel/dlstreamer/gstreamer/lib").is_dir() and Path(_SYSTEM_GST_PLUGINS).is_dir():
        registry.scan_path(_SYSTEM_GST_PLUGINS)
    return Gst


def _create_shared_va_context(Gst):
    """Own one VA display for the lifetime of the shared inference pool.

    Separate per-camera displays force DL Streamer to export/import surfaces
    between driver contexts. Main-stream teardown must not change the display
    used by the shared live model. Use the VA decoder's selected render device.
    """
    import gi

    gi.require_version("GstVa", "1.0")
    from gi.repository import GstVa

    decoder = None
    for name in ("vah264dec", "vah265dec"):
        # Gst's Python overrides can raise for missing factories. Discover
        # availability first so an absent H.264 decoder still permits H.265.
        if Gst.ElementFactory.find(name) is not None:
            decoder = Gst.ElementFactory.make(name)
            if decoder is not None:
                break
    if decoder is None:
        raise RuntimeError("shared VA capture requires a GStreamer VA decoder")
    try:
        if decoder.set_state(Gst.State.READY) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("could not initialize VA decoder to select render device")
        device_path = decoder.get_property("device-path")
        if not device_path:
            raise RuntimeError("VA decoder did not select a render device")
        display = GstVa.VaDisplayDrm.new_from_path(device_path)
        if display is None:
            raise RuntimeError(f"could not open shared VA display on {device_path}")
        context = Gst.Context.new(GstVa.VA_DISPLAY_HANDLE_CONTEXT_TYPE_STR, True)
        # The context owns a reference to the display, beyond the probe decoder.
        GstVa.context_set_va_display(context, display)
        return context
    finally:
        decoder.set_state(Gst.State.NULL)


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


class InferencePipelineError(RuntimeError):
    """A native error in the shared model, rather than an RTSP source failure."""


def _negotiated_memory(element, pad_name: str = "sink") -> str | None:
    """Report actual negotiated memory, not merely the requested backend."""
    if element is None:
        return None
    caps = element.get_static_pad(pad_name).get_current_caps()
    if caps is None or caps.get_size() < 1:
        return None
    return caps.get_features(0).to_string()


def _require_detection_plugin(Gst) -> None:
    if _factory_available(Gst, "gvadetect"):
        return
    plugin = Path("/opt/intel/dlstreamer/lib/libgstvideoanalytics.so")
    if plugin.is_file():
        # Retry a previously blacklisted plugin, and expose loader failures
        # (e.g. missing OpenCV) instead of hiding them behind "unavailable".
        Gst.Plugin.load_file(str(plugin))
    if not _factory_available(Gst, "gvadetect"):
        raise RuntimeError("required GStreamer element is unavailable: gvadetect")


H264_DECODER_COMPLIANCE = {"auto": 0, "strict": 1, "normal": 2, "flexible": 3}


def _configure_h264_decoder(element, compliance: str) -> None:
    factory = element.get_factory()
    if (factory is not None and "h264" in factory.get_name()
            and element.find_property("compliance") is not None):
        element.set_property("compliance", H264_DECODER_COMPLIANCE[compliance])


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
            normalized = {
                "label": label or "object",
                "confidence": confidence,
                "box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
            }
            # Native IDs are authoritative only within a capture session.
            native_track_id = item.get("id", item.get("object_id"))
            if native_track_id is None:
                native_track_id = detection.get("object_id")
            if native_track_id is not None:
                native_track_id = int(native_track_id)
                if native_track_id >= 0:
                    normalized["native_track_id"] = native_track_id
            if "zone_violations" in item:
                ids = item["zone_violations"]
                if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
                    raise ValueError("invalid native zone membership")
                normalized["native_zone_ids"] = list(ids)
            objects.append(normalized)
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


class _NativeInferenceEvidence:
    """Bounded pre-tracker evidence, before the leaky output queue.

    gvadetect's no-block=false contract runs the first buffer and
    every inference-interval buffer thereafter, emitting buffers in order.
    ROI mode supplies exactly one region on every input, including sweeps.
    Count here, never at appsink (which drops buffers). Capture ROIs here too:
    gvatrack can append predictions even on frames with fresh detections.
    See DL Streamer inference_impl.cpp::TransformFrameIp and tracker.cpp::track.
    """

    def __init__(self, interval: int, *, tracking: bool = False) -> None:
        self.interval = interval
        self.tracking = tracking
        self.sequence = 0
        self.results = OrderedDict()
        self.lock = threading.Lock()
        self.invalid = 0
        self.last_pts = None
        self.identity_valid = True
        self.starts = OrderedDict()
        self.latencies_ms = deque(maxlen=100)

    def begin(self, pts):
        with self.lock:
            self.starts[pts] = time.monotonic()
            while len(self.starts) > 128:
                self.starts.popitem(last=False)

    def timing_status(self):
        with self.lock:
            values = sorted(self.latencies_ms)
        return {
            "native_detector_average_ms": round(sum(values) / len(values), 2) if values else None,
            "native_detector_p95_ms": round(values[min(len(values) - 1, int(len(values) * .95))], 2) if values else None,
            "native_detector_timing_samples": len(values),
        }

    def observe(self, buffer, caps, video_frame_type) -> None:
        self.sequence += 1
        fresh = (self.sequence - 1) % self.interval == 0
        with self.lock:
            started = self.starts.pop(buffer.pts, None)
            if fresh and started is not None:
                self.latencies_ms.append((time.monotonic() - started) * 1000)
        objects = []
        if self.last_pts is not None and buffer.pts <= self.last_pts:
            # PTS reuse can collide with metadata still queued after this probe.
            # Fail closed until pipeline recreation establishes a new identity.
            self.identity_valid = False
            self.invalid += 1
            with self.lock:
                self.results.clear()
        self.last_pts = buffer.pts
        try:
            if not self.identity_valid:
                result = ("unknown", [])
            elif fresh:
                for region in video_frame_type(buffer, caps=caps).regions():
                    rect = region.rect()
                    objects.append({
                        "label": region.label(), "confidence": region.confidence(),
                        "box": {"x1": rect.x, "y1": rect.y,
                                "x2": rect.x + rect.w, "y2": rect.y + rect.h},
                    })
            if self.identity_valid:
                result = ("native_fresh_detection" if fresh else (
                    "native_tracked_prediction" if self.tracking else "unknown"
                ), objects)
        except Exception:
            # Do not turn an adapter failure into authoritative empty evidence.
            # Report the bounded counter in status; this frame cannot admit activity.
            self.invalid += 1
            result = ("unknown", [])
        with self.lock:
            self.results[buffer.pts] = result
            while len(self.results) > 128:
                self.results.popitem(last=False)

    def pop(self, pts):
        with self.lock:
            return self.results.pop(pts, ("unknown", []))


def _filter_tracking_regions(buffer, caps, video_frame_type, allowed_classes):
    """Remove excluded ROI metadata before tracking, without mapping pixel memory."""
    from survng.native_spatial import ROI_LABEL
    from gi.repository import GstAnalytics, GLib
    frame = video_frame_type(buffer, caps=caps)
    regions = list(frame.regions())
    selected = [region for region in regions if region.label() != ROI_LABEL
                and (allowed_classes is None or region.label().strip().lower() in allowed_classes)]
    if len(selected) == len(regions):
        return
    kept = [(region.rect(), region.label(), region.confidence(), region.label_id()) for region in selected]
    for region in regions:
        frame.remove_region(region)
    # DL Streamer 2026 also stores detections in analytics relation metadata;
    # its public API has no individual-record removal. Rebuild the selected
    # bounding-box detections in both representations, retaining class IDs.
    meta = buffer.get_meta(GstAnalytics.relation_meta_api_get_type())
    if meta is not None and not buffer.remove_meta(meta):
        raise RuntimeError("could not replace tracking analytics metadata")
    for rect, label, confidence, label_id in kept:
        roi = frame.add_region(rect.x, rect.y, rect.w, rect.h, label, confidence)
        relation = GstAnalytics.buffer_get_analytics_relation_meta(buffer)
        quarks = [0] * (label_id + 1)
        scores = [0.0] * (label_id + 1)
        quarks[label_id], scores[label_id] = GLib.quark_from_string(label), confidence
        success, classification = relation.add_cls_mtd(scores, quarks)
        if not success or not relation.set_relation(GstAnalytics.RelTypes.RELATE_TO, roi.meta().id, classification.id):
            raise RuntimeError("could not retain tracking class identity")


def _detection_metadata(sample, video_frame_type, *, inference_sequence: int, gst_second: int,
                        clock_time_none: int, native_result=None, spatial_plan=None):
    """Read GstGVAJSONMeta; mapping a video buffer yields pixels, not JSON."""
    from survng.app.live_detections import DetectionSnapshot
    buffer = sample.get_buffer()
    if buffer.pts == clock_time_none:
        raise ValueError("inferred frame has no source PTS")
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    messages = video_frame_type(buffer, caps=caps).messages()
    if len(messages) != 1:
        raise ValueError("expected one authoritative inference message")
    payload = json.loads(messages[0])
    if not isinstance(payload, dict):
        raise ValueError("invalid inference metadata")
    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("invalid inference objects")
    normalized = _normalize_gva_objects(payload)
    if len(normalized) != len(objects):
        raise ValueError("incomplete inference metadata")
    if spatial_plan is not None:
        for obj in normalized:
            obj.setdefault("native_zone_ids", [])
            obj["native_zone_revision"] = spatial_plan["revision"]
    provenance, fresh_objects = native_result or ("unknown", [])
    # gvatrack preserves detector ROIs and appends unassociated predictions.
    # Transfer IDs only on an exact, unambiguous detector ROI match. Never
    # promote an appended prediction (which may carry confidence=1) to evidence.
    remaining = list(fresh_objects) if provenance == "native_fresh_detection" else []
    for obj in normalized:
        matches = [item for item in remaining
                   if item["label"] == obj["label"] and item["box"] == obj["box"]
                   and abs(item["confidence"] - obj["confidence"]) < 1e-5]
        obj["detection_provenance"] = "native_tracked_prediction"
        if len(matches) == 1:
            obj["detection_provenance"] = "native_fresh_detection"
            remaining.remove(matches[0])
    if remaining:
        raise ValueError("tracker metadata lost authoritative detector ROIs")
    snapshot = {
        "schema_version": 1,
        "provenance": provenance,
        "source_pts": float(buffer.pts) / gst_second,
        "inference_sequence": inference_sequence,
        "width": int(structure.get_value("width") or 0),
        "height": int(structure.get_value("height") or 0),
        "objects": normalized,
    }
    if spatial_plan is not None:
        snapshot["zone_revision"] = spatial_plan["revision"]
    DetectionSnapshot.parse(snapshot)
    return snapshot


def _write(
    stdout,
    message: bytes | tuple[bytes, ...],
    *,
    lock: threading.Lock | None = None,
) -> None:
    from survng.app.dlstreamer_protocol import ProtocolWriter

    if isinstance(stdout, ProtocolWriter):
        stdout.send(message)
        return
    if lock is None:
        ProtocolWriter(stdout).send(message)
        return
    with lock:
        ProtocolWriter(stdout).send(message)


def run(argv: list[str] | None = None, *, output=None) -> int:
    with ExitStack() as resources:
        return _run(argv, resources, output=output)


def _run(argv: list[str] | None, resources: ExitStack, *, output=None) -> int:
    from survng.app.dlstreamer_protocol import (
        TYPE_DETECTIONS,
        TYPE_STATUS,
        encode_frame_parts as encode_frame,
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
    stdout = output if output is not None else sys.stdout.buffer
    Gst = _load_gstreamer()
    _prefer_decoder(Gst, args.decoder)

    model_path = Path(args.model).expanduser() if args.model else None
    detect = not args.no_detect
    if detect:
        if model_path is None or not model_path.is_file():
            raise RuntimeError("live detection requested but model file is unavailable")
        _require_detection_plugin(Gst)
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        raise ValueError("detection threshold must be between 0 and 1")
    if not 1 <= args.inference_interval <= 5:
        raise ValueError("inference interval must be between 1 and 5")
    if args.native_tracking == "deep-sort":
        if args.tracking_classes is None:
            args.tracking_classes = ["person"]
        elif args.tracking_classes != ["person"]:
            raise ValueError("Deep SORT experiment requires --tracking-classes [\"person\"]")
        reid_model = Path(args.reid_model).expanduser() if args.reid_model else None
        if reid_model is None or not reid_model.is_file():
            raise RuntimeError("Deep SORT requested but ReID model file is unavailable")
        args.reid_model = str(reid_model)
    instance_id = (
        model_instance_id(str(model_path or ""), args.device, args.model_instance_id)
        if detect
        else ""
    )
    from survng.dlstreamer_model import configured_model

    model_path, args.model_proc = resources.enter_context(configured_model(
        model_path, args.model_proc, args.nms_threshold if detect else None,
    ))
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
        detect=detect and args.source_role == "live",
        model_path=model_path,
        instance_id=instance_id,
        rate=rate if args.source_role == "live" else _frame_rate(args.main_fps),
        detect_rate=detect_rate,
        qualifier_width=qualifier_width if args.source_role == "live" else 640,
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
        source_role=args.source_role,
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
        encode_frame_parts as encode_frame,
        encode_jpeg,
        encode_json,
    )
    from survng.app.redact import redact_secret_text

    stdout_lock = threading.Lock()
    stop_all = threading.Event()
    workers: dict[str, tuple[threading.Event, threading.Thread]] = {}
    workers_lock = threading.Lock()
    # Keep the display alive until all live/main workers have stopped. Every
    # pipeline must inherit it before creating decoders or inference elements.
    va_context = (
        _create_shared_va_context(Gst)
        if detect and args.decoder == "va" and not args.test_source else None
    )
    print(
        f"survng-dls supervisor model_instance_id={instance_id or 'none'}",
        file=sys.stderr,
        flush=True,
    )

    def request_stop(_signum=None, _frame=None) -> None:
        # Signals execute on the command thread, possibly while it owns
        # workers_lock. Cleanup owns worker signalling; never reacquire that
        # non-reentrant lock from a signal handler.
        stop_all.set()

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def stop_stream(stream_id: str) -> None:
        with workers_lock:
            worker = workers.get(stream_id)
        if worker is None:
            return
        event, thread = worker
        event.set()
        thread.join(timeout=STREAM_STOP_TIMEOUT_SECONDS)
        if thread.is_alive():
            raise StreamShutdownError("native stream shutdown timed out; supervisor replacement required")
        with workers_lock:
            workers.pop(stream_id, None)

    def start_stream(stream_id: str, url: str, source_role: str = "live", frame_width: int = qualifier_width, detection_enabled: bool = True, h264_decoder_compliance: str = "auto", spatial_plan: dict | None = None) -> None:
        stop_stream(stream_id)
        event = threading.Event()

        def target() -> None:
            try:
                _pump_pipeline(
                    Gst,
                    args,
                    url=url,
                    stream_id=stream_id,
                    detect=detect and detection_enabled and source_role == "live",
                    h264_decoder_compliance=h264_decoder_compliance,
                    spatial_plan=spatial_plan,
                    model_path=model_path,
                    instance_id=instance_id,
                    rate=rate if source_role == "live" else _frame_rate(args.main_fps),
                    detect_rate=detect_rate,
                    qualifier_width=frame_width if source_role == "live" else 640,
                    jpeg_rate=jpeg_rate if source_role == "live" else None,
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
                    test_source=args.test_source,
                    source_role=source_role,
                    va_context=va_context,
                )
            except Exception as exc:
                _write(
                    stdout,
                    encode_json(
                        TYPE_STATUS,
                        {"ok": False, "error": redact_secret_text(exc),
                         "failure_scope": "inference" if isinstance(exc, InferencePipelineError) else "stream"},
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
                source_role = str(command.get("source_role") or "live")
                if source_role not in {"main", "live"}:
                    raise ValueError("invalid capture source role")
                frame_width = _qualifier_width(command.get("frame_width", qualifier_width))
                compliance = command.get("h264_decoder_compliance", "auto")
                if not isinstance(compliance, str) or compliance not in H264_DECODER_COMPLIANCE:
                    raise ValueError("invalid H.264 decoder compliance")
            except (TypeError, ValueError) as exc:
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
            start_stream(stream_id, url, source_role, frame_width, command.get("detection_enabled") is not False, compliance, command.get("spatial_plan"))
    finally:
        request_stop()
        with workers_lock:
            remaining = list(workers.items())
        deadline = time.monotonic() + STREAM_STOP_TIMEOUT_SECONDS
        for _stream_id, (event, _thread) in remaining:
            event.set()
        for _stream_id, (_event, thread) in remaining:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if any(thread.is_alive() for _stream_id, (_event, thread) in remaining):
            raise StreamShutdownError("native stream shutdown timed out; supervisor replacement required")
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
    source_role: str = "live",
    va_context=None,
    h264_decoder_compliance: str = "auto",
    spatial_plan: dict | None = None,
) -> int:
    if h264_decoder_compliance not in H264_DECODER_COMPLIANCE:
        raise ValueError("invalid H.264 decoder compliance")
    from survng.native_budget import BUDGETS, NativeBudget, MOTION_PROPERTIES
    budget_enabled = bool(detect and spatial_plan and spatial_plan.get("budget", {}).get("enabled"))
    budget = NativeBudget(spatial_plan) if budget_enabled else None
    budget_key = f"{stream_id}:{id(budget)}" if budget is not None else None
    if budget is not None:
        detect_rate = _frame_rate(budget.config["active_fps"])
    detector_interval = 1 if budget_enabled else args.inference_interval
    use_test_source = args.test_source if test_source is None else test_source
    pipeline = Gst.Pipeline.new(_pipeline_name(stream_id))
    if pipeline is None:
        raise RuntimeError("could not create GStreamer pipeline")
    if va_context is not None:
        pipeline.set_context(va_context)

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
    color_frames = True
    va_memory = detect and args.decoder == "va" and not use_test_source
    frame_queue = _element(Gst, "queue", "frame-queue")
    frame_queue.set_property("max-size-buffers", 1)
    frame_queue.set_property("leaky", 2)
    videorate = _element(Gst, "videorate", "drop-only-rate")
    videorate.set_property("drop-only", True)
    # The tee carries VA surfaces when detection uses VA preprocessing. CPU
    # consumers need an explicit download boundary; software videoconvert
    # cannot negotiate that transition. Drop frames BEFORE the VA conversion
    # before mapping the evidence frame into host RAM. Width zero retains
    # negotiated native dimensions; bounded legacy consumers can still resize.
    frame_size = f",width={qualifier_width},pixel-aspect-ratio=1/1" if qualifier_width else ""
    frame_converters = []
    if va_memory:
        download = _element(Gst, "vapostproc", "qualifier-download")
        # Even at identical dimensions, download a distinct surface. Mapping
        # the decoder's shared surface can collide with gvadetect VA rendering.
        download.set_property("disable-passthrough", True)
        download_caps = _element(Gst, "capsfilter", "qualifier-host-caps")
        # Download NV12 at native or explicitly requested size, then convert to BGR.
        # Explicit square pixels preserve geometry when scaling.
        download_caps.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw,format=NV12{frame_size}"
        ))
        frame_converters.extend([download, download_caps])
    frame_converters.append(_element(Gst, "videoconvert", "qualifier-gray"))
    if not va_memory and qualifier_width:
        frame_converters.append(_element(Gst, "videoscale", "qualifier-scale"))
    capsfilter = _element(Gst, "capsfilter", "frame-caps")
    capsfilter.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw,format={'BGR' if color_frames else 'GRAY8'}{frame_size},"
            f"framerate={rate.numerator}/{rate.denominator}"
        ),
    )
    sink = _element(Gst, "appsink", "frame-sink")
    sink.set_property("emit-signals", False)
    sink.set_property("max-buffers", 1)
    sink.set_property("drop", True)
    sink.set_property("sync", False)

    frame_chain = [frame_queue, videorate, *frame_converters, capsfilter, sink]
    elements = [source, tee, *frame_chain]
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
        jpeg_convert = _element(
            Gst, "vapostproc" if va_memory else "videoconvert", "jpeg-convert",
        )
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
    native_tracker = None
    reid_queue = None
    reid = None
    reid_preprocess = ""
    reid_instance_id = ""
    analytics = None
    roi_input = None
    roi_enabled = False
    budget_elements = []
    meta_convert = None
    va_caps = None
    preprocess = ""
    video_frame_type = None
    if detect:
        from gstgva import VideoFrame
        from gstgva.util import GST_PAD_PROBE_INFO_BUFFER
        video_frame_type = VideoFrame
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
        compile_options = "PERFORMANCE_HINT=THROUGHPUT"
        target = args.device.upper().split(".", 1)[0]
        if target in {"CPU", "GPU"}:
            compile_options += f",NUM_STREAMS={args.inference_streams}"
        if target == "GPU":
            # THROUGHPUT may enable OpenVINO automatic batching even though
            # explicit gvadetect batching is configured. Its wrapper cannot bind this
            # model's VA surface inputs ("Input tensor with index 0 is not
            # found"). Keep parallel requests/streams, but disable auto batching.
            compile_options += ",ALLOW_AUTO_BATCHING=NO"
            compile_options += f",COMPILATION_NUM_THREADS={GPU_COMPILATION_NUM_THREADS}"
        detector.set_property("ie-config", compile_options)
        # Explicit batches and parallel requests share one compiled model.
        # Input/output queues remain bounded under overload.
        detector.set_property("batch-size", args.batch_size)
        detector.set_property("nireq", args.inference_requests)
        detector.set_property("inference-interval", detector_interval)
        detector.set_property("no-block", False)
        roi_enabled = bool(spatial_plan and spatial_plan.get("roi", {}).get("enabled"))
        detector.set_property("inference-region", 1 if roi_enabled or budget_enabled else 0)
        if (roi_enabled or budget_enabled) and instance_id:
            # Shared preprocessing is compiled from the first stream's region
            # mode. Mixing full-frame and ROI consumers misprojects batch output.
            # ROI cameras share a separate compiled pool with one another.
            instance_id += "-roi"
        native_evidence = _NativeInferenceEvidence(
            detector_interval, tracking=args.native_tracking != "off",
        )

        selected_classes = getattr(args, "tracking_classes", None)
        allowed_classes = None if selected_classes is None else frozenset(label.strip().lower() for label in selected_classes)

        def capture_native_evidence(pad, info):
            # Intel's context manager lends the probe buffer without the extra
            # Python reference that otherwise makes metadata read-only.
            with GST_PAD_PROBE_INFO_BUFFER(info) as buffer:
                if buffer is not None:
                    try:
                        _filter_tracking_regions(buffer, pad.get_current_caps(), video_frame_type, allowed_classes)
                    except Exception as exc:
                        native_evidence.invalid += 1
                        # A dropped detector output loses the inference-interval
                        # phase. Fail closed until the watchdog recreates capture.
                        native_evidence.identity_valid = False
                        with native_evidence.lock:
                            native_evidence.results.clear()
                        if native_evidence.invalid == 1 or native_evidence.invalid % 100 == 0:
                            print(f"survng-dls tracking class metadata filter failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                        return Gst.PadProbeReturn.DROP
                    native_evidence.observe(buffer, pad.get_current_caps(), video_frame_type)
                    if budget is not None:
                        with native_evidence.lock:
                            provenance, objects = native_evidence.results.get(buffer.pts, ("unknown", []))
                        if provenance == "native_fresh_detection":
                            structure = pad.get_current_caps().get_structure(0)
                            budget.objects(objects, structure.get_value("width"), structure.get_value("height"), buffer.pts / Gst.SECOND)
            return Gst.PadProbeReturn.OK

        spatial_dimensions = None

        def capture_native_start(pad, info):
            nonlocal spatial_dimensions
            buffer = info.get_buffer()
            if buffer is not None:
                try:
                    structure = pad.get_current_caps().get_structure(0)
                    dimensions = (int(structure.get_value("width")), int(structure.get_value("height")))
                    if analytics is not None and spatial_dimensions is None:
                        from survng.native_spatial import analytics_zones
                        analytics.set_property("zones", json.dumps(analytics_zones(spatial_plan, *dimensions)))
                        analytics.set_locked_state(False)
                        if not analytics.sync_state_with_parent():
                            raise RuntimeError("could not start zone analytics")
                        spatial_dimensions = dimensions
                    if analytics is not None and dimensions != spatial_dimensions:
                        raise RuntimeError("zone geometry changed; stream rebuild required")
                    native_evidence.begin(buffer.pts)
                except Exception as exc:
                    native_evidence.invalid += 1
                    native_evidence.identity_valid = False
                    with native_evidence.lock:
                        native_evidence.results.clear()
                    if native_evidence.invalid == 1 or native_evidence.invalid % 100 == 0:
                        print("survng-dls spatial geometry unavailable; stream rebuild required", file=sys.stderr, flush=True)
                    return Gst.PadProbeReturn.DROP
            return Gst.PadProbeReturn.OK

        detector.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, capture_native_start)
        detector.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, capture_native_evidence)
        detector.set_property("threshold", args.threshold)
        preprocess = "opencv"
        if args.decoder == "va" and not use_test_source:
            preprocess = "va-surface-sharing" if "GPU" in args.device.upper() else "va"
        detector.set_property("pre-process-backend", preprocess)
        if preprocess.startswith("va"):
            detector.set_property("pre-process-config", "VAAPI_THREAD_POOL_SIZE=1")
        if instance_id:
            detector.set_property("model-instance-id", instance_id)
            # Each RTSP pipeline has its own running-time origin. Comparing
            # their raw PTS unfairly prioritizes later-started cameras.
            detector.set_property("scheduling-policy", "throughput")
        if args.model_proc:
            detector.set_property("model-proc", args.model_proc)
        if args.labels:
            # `labels` is a comma-separated class list, NOT a filename. Both
            # properties accept strings, so the old fallback silently succeeded.
            detector.set_property("labels-file", args.labels)
        elif args.labels_list:
            detector.set_property("labels", args.labels_list)
        if args.native_tracking == "deep-sort":
            if detector_interval != 1:
                raise InferencePipelineError("Deep SORT requires detector inference interval 1")
            if not _factory_available(Gst, "gvainference"):
                raise InferencePipelineError("Deep SORT requested but gvainference is unavailable")
            reid_queue = _element(Gst, "queue", "reid-queue")
            reid_queue.set_property("max-size-buffers", 2)
            reid_queue.set_property("max-size-bytes", 0)
            reid_queue.set_property("max-size-time", 0)
            reid = _element(Gst, "gvainference", "native-reid")
            reid.set_property("model", args.reid_model)
            reid.set_property("device", args.reid_device)
            reid.set_property("inference-region", 1)
            reid.set_property("object-class", "person")
            reid.set_property("inference-interval", 1)
            reid.set_property("batch-size", 1)
            reid.set_property("nireq", 2)
            reid.set_property("no-block", False)
            reid_preprocess = "opencv"
            if args.decoder == "va" and not use_test_source:
                reid_preprocess = "va-surface-sharing" if "GPU" in args.reid_device.upper() else "va"
            reid.set_property("pre-process-backend", reid_preprocess)
            reid_instance_id = model_instance_id(args.reid_model, args.reid_device)
            reid.set_property("model-instance-id", reid_instance_id)
            reid.set_property("scheduling-policy", "throughput")
        detect_output_queue = _element(Gst, "queue", "detect-output-queue")
        detect_output_queue.set_property("max-size-buffers", 1)
        detect_output_queue.set_property("max-size-bytes", 0)
        detect_output_queue.set_property("max-size-time", 0)
        detect_output_queue.set_property("leaky", 2)
        if args.native_tracking != "off":
            if not _factory_available(Gst, "gvatrack"):
                raise InferencePipelineError(
                    "native tracking requested but gvatrack is unavailable"
                )
            native_tracker = _element(Gst, "gvatrack", "native-track")
            native_tracker.set_property("tracking-type", args.native_tracking)
            if args.native_tracking == "deep-sort":
                native_tracker.set_property("device", "CPU")
                native_tracker.set_property("deepsort-trck-cfg", args.deep_sort_config)
        if spatial_plan is not None:
            analytics = _element(Gst, "gvaanalytics", "zone-analytics")
            analytics.set_property("evaluation-point", 1)
            analytics.set_property("draw-zones", False)
            analytics.set_property("draw-tripwires", False)

            # 2026.2 reads polygons on READY -> PAUSED, not on property writes.
            # Wait for detector input caps before starting this metadata element.
            analytics.set_locked_state(True)
            analytics.set_state(Gst.State.READY)
            elements.append(analytics)
        if budget is not None:
            if budget.config["motion_enabled"]:
                converter = _element(Gst, "vapostproc" if va_memory else "videoconvert", "motion-convert")
                motion_caps = _element(Gst, "capsfilter", "motion-caps")
                motion_caps.set_property("caps", Gst.Caps.from_string(
                    "video/x-raw" + ("(memory:VAMemory)" if va_memory else "") + ",format=NV12"))
                motion_detector = _element(Gst, "gvamotiondetect", "budget-motion")
                for name in MOTION_PROPERTIES:
                    motion_detector.set_property(name.replace("_", "-"), budget.config[name])
                budget_elements.extend([converter, motion_caps, motion_detector])
            gate = _element(Gst, "identity", "budget-gate")
            budget_elements.append(gate)

            def admit_budget(pad, info):
                try:
                    with GST_PAD_PROBE_INFO_BUFFER(info) as buffer:
                        if buffer is None:
                            return Gst.PadProbeReturn.OK
                        caps = pad.get_current_caps()
                        structure = caps.get_structure(0)
                        width, height = structure.get_value("width"), structure.get_value("height")
                        frame = video_frame_type(buffer, caps=caps)
                        motion = [(r.rect().x/width, r.rect().y/height,
                                   (r.rect().x+r.rect().w)/width, (r.rect().y+r.rect().h)/height)
                                  for r in frame.regions() if r.label() == "motion"]
                        # Motion metadata (confidence 1) must never reach inference
                        # evidence or tracking, including on skipped frames.
                        _filter_tracking_regions(buffer, caps, video_frame_type, frozenset())
                        if not budget.select(buffer.pts / Gst.SECOND, motion):
                            return Gst.PadProbeReturn.DROP
                except Exception as exc:
                    native_evidence.invalid += 1
                    if native_evidence.invalid == 1 or native_evidence.invalid % 100 == 0:
                        print(f"survng-dls inference budget metadata failed: {type(exc).__name__}", file=sys.stderr, flush=True)
                    return Gst.PadProbeReturn.DROP
                return Gst.PadProbeReturn.OK

            gate.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, admit_budget)
            elements.extend(budget_elements)
        if roi_enabled or budget_enabled:
            roi_input = _element(Gst, "gvapython", "inference-region")
            roi_input.set_property("module", str(Path(__file__).with_name("native_spatial.py")))
            roi_input.set_property("class", "RoiInput")
            roi_input.set_property("kwarg", json.dumps({"plan": spatial_plan, "interval": detector_interval, "budget_key": budget_key}))
            elements.append(roi_input)
        elements.extend([detect_queue, detect_rate_el, detect_rate_caps, detector, detect_output_queue])
        if reid_queue is not None and reid is not None:
            elements.extend([reid_queue, reid])
        if native_tracker is not None:
            elements.append(native_tracker)
        if preprocess.startswith("va"):
            va_caps = _element(Gst, "capsfilter", "detect-va-memory")
            va_caps.set_property(
                "caps",
                Gst.Caps.from_string("video/x-raw(memory:VAMemory)"),
            )
            elements.append(va_caps)
        if _factory_available(Gst, "gvametaconvert"):
            meta_convert = _element(Gst, "gvametaconvert", "detect-meta")
            meta_convert.set_property("add-empty-results", True)
            try:
                meta_convert.set_property("format", "json")
            except Exception:
                pass
            meta_sink = _element(Gst, "appsink", "meta-sink")
            meta_sink.set_property("emit-signals", False)
            # These are full video buffers (possibly VA surfaces), unlike the
            # parent's small metadata-only history. Release them promptly.
            meta_sink.set_property("max-buffers", 4)
            meta_sink.set_property("drop", True)
            meta_sink.set_property("sync", False)
            # Sparse inference output must not gate preview startup.
            meta_sink.set_property("async", False)
            elements.extend([meta_convert, meta_sink])
        else:
            raise RuntimeError("gvametaconvert is required for authoritative live detections")

    for element in elements:
        pipeline.add(element)

    for left, right in zip(frame_chain, frame_chain[1:]):
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
        input_chain = [detect_queue, detect_rate_el, detect_rate_caps]
        if va_caps is not None:
            input_chain.append(va_caps)
        input_chain.extend(budget_elements)
        if roi_input is not None:
            input_chain.append(roi_input)
        input_chain.append(detector)
        for left, right in zip(input_chain, input_chain[1:]):
            if not left.link(right):
                raise RuntimeError(f"could not link native detection input: {left.get_name()} to {right.get_name()}")
        tracked_source = detector
        if reid_queue is not None and reid is not None:
            if not detector.link(reid_queue) or not reid_queue.link(reid):
                raise RuntimeError("could not link Deep SORT ReID inference")
            tracked_source = reid
        if native_tracker is not None:
            if not tracked_source.link(native_tracker):
                raise RuntimeError("could not link gvatrack")
            tracked_source = native_tracker
        if analytics is not None:
            if not tracked_source.link(analytics):
                raise RuntimeError("could not link native zone analytics")
            tracked_source = analytics
        # Track every completed detection before shedding metadata delivery.
        if detect_output_queue is None or not tracked_source.link(detect_output_queue):
            raise RuntimeError("could not link native output queue")
        metadata_source = detect_output_queue
        if meta_convert is not None and meta_sink is not None:
            if not metadata_source.link(meta_convert) or not meta_convert.link(meta_sink):
                raise RuntimeError("could not link detection metadata branch")
        else:
            fake = pipeline.get_by_name("detect-sink")
            if fake is None or not metadata_source.link(fake):
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
        _configure_h264_decoder(element, h264_decoder_compliance)
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
    if budget is not None:
        BUDGETS[budget_key] = budget
    started = time.monotonic()
    first_frame_at: float | None = None
    last_status_at: float | None = None
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
                        error = f"{error}: {debug}"
                    from survng.app.redact import redact_diagnostic_text
                    error = redact_diagnostic_text(error)
                    if (
                        (detector is not None and message.src == detector)
                        or (reid is not None and message.src == reid)
                        or (native_tracker is not None and message.src == native_tracker)
                    ):
                        raise InferencePipelineError(error)
                    raise RuntimeError(error)
                break
            if meta_sink is not None:
                for _ in range(32):
                    meta_sample = meta_sink.emit("try-pull-sample", 0)
                    if meta_sample is None:
                        break
                    inference_sequence += 1
                    try:
                        payload = _detection_metadata(
                            meta_sample, video_frame_type,
                            inference_sequence=inference_sequence,
                            gst_second=Gst.SECOND, clock_time_none=Gst.CLOCK_TIME_NONE,
                            native_result=native_evidence.pop(meta_sample.get_buffer().pts),
                            spatial_plan=spatial_plan,
                        )
                    except Exception:
                        # Malformed metadata must leave video available for
                        # a second detector. Surface failures via status.
                        native_evidence.invalid += 1
                        continue
                    _write(stdout, encode_json(TYPE_DETECTIONS, payload, stream_id=stream_id), lock=stdout_lock)
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
            pixels = _packed_gray(pixels, width * (3 if color_frames else 1), height)
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
            if last_status_at is None or time.monotonic() - last_status_at >= 1.0:
                last_status_at = time.monotonic()
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
                            "h264_decoder_compliance": h264_decoder_compliance,
                            "preprocess_backend": preprocess,
                            "decoded_memory": _negotiated_memory(tee),
                            "detection_memory": _negotiated_memory(detector),
                            "qualifier_memory": _negotiated_memory(sink),
                            "first_frame_ms": round(
                                (first_frame_at - started) * 1000.0,
                                3,
                            ),
                            "qualifier_format": "BGR" if color_frames else "GRAY8",
                            "source_role": source_role,
                            "metadata_contract": "GstGVAJSONMeta-v1" if detect else "disabled",
                            "detection_threshold": args.threshold if detect else None,
                            "requested_nms_threshold": args.nms_threshold if detect else None,
                            "qualifier_width": qualifier_width,
                            "evidence_width": width, "evidence_height": height,
                            "evidence_sample_fps": float(rate),
                            "detect_fps": float(detect_rate),
                            "native_evidence_invalid": native_evidence.invalid if detect else 0,
                            **(native_evidence.timing_status() if detect else {}),
                            "batch_size": args.batch_size if detect else None,
                            "inference_interval": detector_interval if detect else None,
                            "effective_inference_fps": (
                                round(float(detect_rate) / detector_interval, 3)
                                if detect else 0.0
                            ),
                            "native_tracking": args.native_tracking if detect else "off",
                            "native_tracking_classes": getattr(args, "tracking_classes", None) if detect else None,
                            "native_reid_enabled": bool(reid is not None),
                            "native_reid_model": Path(args.reid_model).name if reid is not None else None,
                            "native_reid_device": args.reid_device if reid is not None else None,
                            "native_reid_preprocess_backend": reid_preprocess if reid is not None else None,
                            "native_reid_model_instance_id": reid_instance_id if reid is not None else None,
                            "native_reid_memory": _negotiated_memory(reid) if reid is not None else None,
                            "native_zone_revision": spatial_plan.get("revision") if spatial_plan and detect else None,
                            "native_budget": budget.status() if budget is not None else {"mode": "disabled"},
                            "native_roi_enabled": bool(spatial_plan and spatial_plan.get("roi", {}).get("enabled") and detect),
                            "native_zone_count": sum(z.get("enabled", True) and len(z.get("points", [])) >= 3 for z in spatial_plan.get("zones", [])) if spatial_plan and detect else 0,
                            "native_roi_full_frame_interval": spatial_plan.get("roi", {}).get("full_frame_interval", 5) if spatial_plan and roi_enabled else 0,
                            "native_inference_requests": args.inference_requests if detect else 0,
                            "native_inference_streams": args.inference_streams if detect else 0,
                            "native_tracking_authoritative": native_tracker is not None,
                            "native_tracking_memory": _negotiated_memory(native_tracker),
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
        if analytics is not None:
            analytics.set_locked_state(False)
        pipeline.set_state(Gst.State.NULL)
        if budget_key is not None:
            BUDGETS.pop(budget_key, None)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


def main(argv: list[str] | None = None) -> int:
    from survng.app.dlstreamer_protocol import (
        TYPE_FATAL, TYPE_STATUS, ProtocolWriter, encode_json, open_protocol_output,
    )
    from survng.app.redact import redact_secret_text

    output = open_protocol_output()
    writer = ProtocolWriter(output)
    try:
        return run(argv, output=writer)
    except Exception as exc:
        print(redact_secret_text(exc), file=sys.stderr, flush=True)
        # A surviving worker may own a blocked write. Parent termination is
        # authoritative; never wait indefinitely to report a fatal shutdown.
        if not isinstance(exc, StreamShutdownError):
            try:
                writer.send(encode_json(
                    TYPE_FATAL if "--supervisor" in (sys.argv[1:] if argv is None else argv) else TYPE_STATUS,
                    {"ok": False, "error": redact_secret_text(exc)},
                ))
            except (OSError, ValueError):
                pass  # The parent has closed the connection; stderr holds the cause.
        return 1
    finally:
        if output is not sys.stdout.buffer:
            output.close()


if __name__ == "__main__":
    # dlopen consults the process-start library path. Re-exec once before GI
    # imports if selecting Intel's bundle changed it.
    previous_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    _apply_dlstreamer_env()
    if os.environ.get("LD_LIBRARY_PATH", "") != previous_library_path:
        os.execv(sys.executable, [sys.executable, "-m", "survng.dlstreamer_live", *sys.argv[1:]])
    raise SystemExit(main())

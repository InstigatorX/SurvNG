from __future__ import annotations

import json
import hashlib
import logging
import math
import mimetypes
import asyncio
import os
import re
import platform
import shutil
import time
import socket
import struct
import subprocess
import tempfile
import threading
import weakref
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import Request as UrlRequest, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import cv2
import numpy as np

from .baichuan_native import ffmpeg_input_args, is_native_baichuan, start_ffmpeg_pipe
from .config import AppConfig, CameraConfig, DetectionZone, camera_by_id, load_config, save_config, slugify_camera_id
from .detector import objects_to_json
from .manager import AppManager
from .zones import apply_detection_zones, detection_threshold

config = load_config()
manager = AppManager(config)
active_mse_streams: set[str] = set()
LOG_LINES: deque[dict] = deque(maxlen=1000)
SECRET_URL_RE = re.compile(r"(\b(?:rtsp|rtmp|http|https|reolink)://)([^:/@\s]+):([^@\s]+)@", re.IGNORECASE)
RECORDING_LOOKUP_LIMIT = 20000
RECORDING_FMP4_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
RECORDING_FMP4_LOCKS_GUARD = threading.Lock()
EVENT_CLIP_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
EVENT_CLIP_LOCKS_GUARD = threading.Lock()
RECORDING_DAY_CACHE: dict[tuple[str, str, int, int], tuple[float, list[dict]]] = {}
RECORDING_DAY_CACHE_LOCK = threading.Lock()
RECORDING_DAY_CACHE_SECONDS = 30.0
RECORDING_CACHE_MAINTENANCE_LOCK = threading.Lock()
RECORDING_CACHE_LAST_MAINTENANCE = 0.0
RECORDING_PREWARM_STOP = threading.Event()
RECORDING_PREWARM_THREAD: threading.Thread | None = None
GO2RTC_COMPAT_STREAMS: set[str] = set()
GO2RTC_COMPAT_LOCK = threading.Lock()
FACE_OBSERVATIONS_SYNCED = False


class MemoryLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        LOG_LINES.append({
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_log_message(message),
        })


def redact_log_message(message: str) -> str:
    return SECRET_URL_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}:***@", str(message))


def install_memory_log_handler() -> None:
    root = logging.getLogger()
    if any(getattr(handler, "name", "") == "survng-memory-log" for handler in root.handlers):
        return
    handler = MemoryLogHandler()
    handler.name = "survng-memory-log"
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    root.setLevel(min(root.level or logging.INFO, logging.INFO))


install_memory_log_handler()



def _ffmpeg_sibling_tool(name: str) -> str:
    ffmpeg = Path(config.ffmpeg_path)
    if ffmpeg.name == "ffmpeg":
        sibling = ffmpeg.with_name(name)
        if sibling.exists():
            return str(sibling)
    return name


def _ffprobe_path() -> str:
    return _ffmpeg_sibling_tool("ffprobe")


def _ffplay_path() -> str:
    return _ffmpeg_sibling_tool("ffplay")

def normalize_source(source: str) -> str:
    return "main" if source == "main" else "live"


def _ensure_go2rtc_h264(host: str, stream_name: str) -> str:
    compat_name = "survng_" + re.sub(r"[^a-zA-Z0-9_-]+", "_", stream_name) + "_h264"
    with GO2RTC_COMPAT_LOCK:
        if compat_name in GO2RTC_COMPAT_STREAMS:
            return compat_name
        params = f"name={quote(compat_name, safe='')}&src={quote(f'ffmpeg:{stream_name}#video=h264#width=1920', safe='')}"
        request = UrlRequest(f"http://{host}:1984/api/streams?{params}", method="PUT")
        with urlopen(request, timeout=5) as response:
            if response.status >= 300:
                raise RuntimeError(f"go2rtc returned HTTP {response.status}")
        GO2RTC_COMPAT_STREAMS.add(compat_name)
    return compat_name


def _go2rtc_ws_url(camera_id: str, source: str, compatibility: str = "native") -> str:
    camera = next((item for item in config.cameras if item.id == camera_id), None)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    parsed = urlsplit(camera.source_url(normalize_source(source)))
    stream_name = parsed.path.strip("/")
    if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname or not stream_name:
        raise HTTPException(status_code=409, detail="camera source is not a go2rtc RTSP restream")
    if compatibility == "h264":
        stream_name = _ensure_go2rtc_h264(parsed.hostname, stream_name)
    return f"ws://{parsed.hostname}:1984/api/ws?src={quote(stream_name, safe='')}"


def recording_source(source: str = "main") -> str:
    return "live" if source == "live" else "main"


def reload_manager(next_config: AppConfig) -> None:
    global config, manager, FACE_OBSERVATIONS_SYNCED
    _stop_recording_prewarmer()
    manager.stop_all()
    with RECORDING_DAY_CACHE_LOCK:
        RECORDING_DAY_CACHE.clear()
    config = next_config
    manager = AppManager(config)
    FACE_OBSERVATIONS_SYNCED = False
    manager.start_all()
    _start_recording_prewarmer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.start_all()
    _start_recording_prewarmer()
    yield
    _stop_recording_prewarmer()
    manager.stop_all()


app = FastAPI(title="SurvNG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="survng/static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("survng/static/index.html", headers={"Cache-Control": "no-store"})


@app.get("/recordings")
def recordings_page() -> FileResponse:
    return FileResponse("survng/static/recordings.html", headers={"Cache-Control": "no-store"})


@app.get("/config")
def config_page() -> FileResponse:
    return FileResponse("survng/static/config.html", headers={"Cache-Control": "no-store"})


@app.get("/incidents")
def incidents_page() -> FileResponse:
    return FileResponse("survng/static/index.html", headers={"Cache-Control": "no-store"})


@app.get("/faces")
def faces_page() -> FileResponse:
    return FileResponse("survng/static/index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/cameras")
def cameras() -> list[dict]:
    return manager.statuses()


@app.get("/api/config")
def get_config() -> dict:
    return config.model_dump(mode="json")


@app.get("/api/logs")
def logs(limit: int = 300, level: str = "", q: str = "") -> dict:
    safe_limit = max(1, min(limit, 1000))
    wanted_level = level.strip().upper()
    query = q.strip().lower()
    rows = list(LOG_LINES)[-safe_limit:]
    if wanted_level:
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        try:
            min_index = levels.index(wanted_level)
            allowed = set(levels[min_index:])
            rows = [row for row in rows if row.get("level") in allowed]
        except ValueError:
            rows = [row for row in rows if row.get("level") == wanted_level]
    if query:
        rows = [row for row in rows if query in f"{row.get('level', '')} {row.get('logger', '')} {row.get('message', '')}".lower()]
    return {"lines": rows[-safe_limit:], "total": len(LOG_LINES)}



def _run_ffmpeg_list(args: list[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(
            [config.ffmpeg_path, "-hide_banner", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return result.stdout or ""
    except Exception:
        return ""


def _dri_render_devices() -> list[str]:
    return sorted(str(path) for path in Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").exists() else []


def _ffmpeg_qsv_info() -> dict:
    hwaccels = _run_ffmpeg_list(["-hwaccels"])
    encoders = _run_ffmpeg_list(["-encoders"])
    decoders = _run_ffmpeg_list(["-decoders"])
    render_devices = _dri_render_devices()
    qsv_encoders = sorted({name for name in ("h264_qsv", "hevc_qsv", "av1_qsv", "mjpeg_qsv") if name in encoders})
    qsv_decoders = sorted({name for name in ("h264_qsv", "hevc_qsv", "av1_qsv", "mjpeg_qsv") if name in decoders})
    listed = "qsv" in hwaccels and "h264_qsv" in encoders
    runtime_usable = False
    runtime_error = ""
    if listed:
        probe_args = [config.ffmpeg_path, "-hide_banner", "-v", "error"]
        if render_devices:
            probe_args.extend(["-qsv_device", render_devices[0]])
        probe_args.extend(["-f", "lavfi", "-i", "color=size=64x64:rate=1", "-frames:v", "1", "-c:v", "h264_qsv", "-f", "null", "-"])
        try:
            probe = subprocess.run(
                probe_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
            )
            runtime_usable = probe.returncode == 0
            runtime_error = "" if runtime_usable else (probe.stderr or "QSV runtime probe failed").strip()[-500:]
        except Exception as exc:
            runtime_error = str(exc) or "QSV runtime probe failed"
    return {
        "available": bool(listed and runtime_usable),
        "listed": bool(listed),
        "runtime_usable": runtime_usable,
        "runtime_error": runtime_error,
        "hwaccel_listed": "qsv" in hwaccels,
        "encoders": qsv_encoders,
        "decoders": qsv_decoders,
        "render_devices": render_devices,
    }


def _ffmpeg_vaapi_info() -> dict:
    hwaccels = _run_ffmpeg_list(["-hwaccels"])
    encoders = _run_ffmpeg_list(["-encoders"])
    decoders = _run_ffmpeg_list(["-decoders"])
    filters = _run_ffmpeg_list(["-filters"])
    render_devices = _dri_render_devices()
    vaapi_encoders = sorted({name for name in ("h264_vaapi", "hevc_vaapi", "av1_vaapi", "mjpeg_vaapi", "mpeg2_vaapi", "vp8_vaapi", "vp9_vaapi") if name in encoders})
    vaapi_decoders = sorted({name for name in ("h264_vaapi", "hevc_vaapi", "av1_vaapi", "mjpeg_vaapi", "mpeg2_vaapi", "vp8_vaapi", "vp9_vaapi") if name in decoders})
    vaapi_filters = sorted({name for name in ("hwupload", "scale_vaapi") if name in filters})
    listed = "vaapi" in hwaccels and "h264_vaapi" in encoders and "hwupload" in filters
    runtime_usable = False
    runtime_error = ""
    if listed and render_devices:
        probe_args = [
            config.ffmpeg_path,
            "-hide_banner",
            "-v",
            "error",
            "-vaapi_device",
            render_devices[0],
            "-f",
            "lavfi",
            "-i",
            "color=size=64x64:rate=1",
            "-frames:v",
            "1",
            "-vf",
            "format=nv12,hwupload",
            "-c:v",
            "h264_vaapi",
            "-f",
            "null",
            "-",
        ]
        try:
            probe = subprocess.run(
                probe_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
            )
            runtime_usable = probe.returncode == 0
            runtime_error = "" if runtime_usable else (probe.stderr or "VAAPI runtime probe failed").strip()[-500:]
        except Exception as exc:
            runtime_error = str(exc) or "VAAPI runtime probe failed"
    elif listed:
        runtime_error = "No /dev/dri/renderD* render device found"
    return {
        "available": bool(listed and runtime_usable),
        "listed": bool(listed),
        "runtime_usable": runtime_usable,
        "runtime_error": runtime_error,
        "hwaccel_listed": "vaapi" in hwaccels,
        "encoders": vaapi_encoders,
        "decoders": vaapi_decoders,
        "filters": vaapi_filters,
        "render_devices": render_devices,
        "device": render_devices[0] if render_devices else "",
    }


def _hardware_acceleration_mode() -> str:
    mode = str(getattr(config, "hardware_acceleration", "auto") or "auto").lower()
    return mode if mode in {"auto", "vaapi", "qsv", "off"} else "auto"


def _probe_video_codec(path: Path) -> str:
    try:
        result = subprocess.run(
            [
                _ffprobe_path(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
        )
        return (result.stdout or "").strip().lower()
    except Exception:
        return ""


def _mp4_boxes(data: bytes | bytearray, start: int = 0, end: int | None = None):
    limit = len(data) if end is None else min(end, len(data))
    cursor = start
    while cursor + 8 <= limit:
        size = struct.unpack_from(">I", data, cursor)[0]
        box_type = bytes(data[cursor + 4:cursor + 8])
        header = 8
        if size == 1 and cursor + 16 <= limit:
            size = struct.unpack_from(">Q", data, cursor + 8)[0]
            header = 16
        elif size == 0:
            size = limit - cursor
        if size < header or cursor + size > limit:
            break
        yield box_type, cursor, cursor + header, cursor + size
        cursor += size


def _mp4_track_timescales(init_data: bytes) -> dict[int, int]:
    timescales: dict[int, int] = {}
    for box_type, _, payload, box_end in _mp4_boxes(init_data):
        if box_type != b"moov":
            continue
        for child_type, _, child_payload, child_end in _mp4_boxes(init_data, payload, box_end):
            if child_type != b"trak":
                continue
            track_id = None
            timescale = None
            for trak_type, _, trak_payload, trak_end in _mp4_boxes(init_data, child_payload, child_end):
                if trak_type == b"tkhd":
                    version = init_data[trak_payload]
                    offset = trak_payload + (20 if version == 1 else 12)
                    if offset + 4 <= trak_end:
                        track_id = struct.unpack_from(">I", init_data, offset)[0]
                elif trak_type == b"mdia":
                    for mdia_type, _, mdia_payload, mdia_end in _mp4_boxes(init_data, trak_payload, trak_end):
                        if mdia_type != b"mdhd":
                            continue
                        version = init_data[mdia_payload]
                        offset = mdia_payload + (20 if version == 1 else 12)
                        if offset + 4 <= mdia_end:
                            timescale = struct.unpack_from(">I", init_data, offset)[0]
            if track_id and timescale:
                timescales[track_id] = timescale
    return timescales


def _offset_fmp4_timestamps(init_path: Path, media_path: Path, seconds: float) -> None:
    if seconds <= 0:
        return
    timescales = _mp4_track_timescales(init_path.read_bytes())
    if not timescales:
        raise RuntimeError("fragment init has no track timescales")
    data = bytearray(media_path.read_bytes())
    adjusted = 0
    for box_type, _, payload, box_end in _mp4_boxes(data):
        if box_type != b"moof":
            continue
        for child_type, _, child_payload, child_end in _mp4_boxes(data, payload, box_end):
            if child_type != b"traf":
                continue
            track_id = None
            tfdt = None
            for traf_type, _, traf_payload, traf_end in _mp4_boxes(data, child_payload, child_end):
                if traf_type == b"tfhd" and traf_payload + 8 <= traf_end:
                    track_id = struct.unpack_from(">I", data, traf_payload + 4)[0]
                elif traf_type == b"tfdt":
                    tfdt = (traf_payload, traf_end)
            if not track_id or not tfdt or track_id not in timescales:
                continue
            tfdt_payload, tfdt_end = tfdt
            version = data[tfdt_payload]
            value_offset = tfdt_payload + 4
            increment = round(seconds * timescales[track_id])
            if version == 1 and value_offset + 8 <= tfdt_end:
                current = struct.unpack_from(">Q", data, value_offset)[0]
                struct.pack_into(">Q", data, value_offset, current + increment)
                adjusted += 1
            elif version == 0 and value_offset + 4 <= tfdt_end:
                current = struct.unpack_from(">I", data, value_offset)[0]
                next_value = current + increment
                if next_value > 0xFFFFFFFF:
                    raise RuntimeError("fragment timestamp exceeds version 0 tfdt")
                struct.pack_into(">I", data, value_offset, next_value)
                adjusted += 1
    if not adjusted:
        raise RuntimeError("fragment has no adjustable tfdt boxes")
    media_path.write_bytes(data)


def _event_clip_cache_suffix(source_codec: str, backend: str) -> str:
    codec = source_codec or "unknown"
    return f"a3-{backend}-{codec}"


def _event_clip_vaapi_enabled(source_codec: str) -> bool:
    mode = _hardware_acceleration_mode()
    if mode not in {"auto", "vaapi"}:
        return False
    if source_codec not in {"h264", "hevc"}:
        return False
    info = _ffmpeg_vaapi_info()
    has_encoder = "h264_vaapi" in set(info.get("encoders") or [])
    return bool(info.get("available") and has_encoder)


def _event_clip_qsv_enabled(source_codec: str) -> bool:
    mode = _hardware_acceleration_mode()
    if mode == "off":
        return False
    if mode == "auto" and _ffmpeg_vaapi_info().get("available"):
        return False
    if mode not in {"auto", "qsv"}:
        return False
    if source_codec not in {"h264", "hevc"}:
        return False
    info = _ffmpeg_qsv_info()
    decoder = f"{source_codec}_qsv"
    has_decoder = decoder in set(info.get("decoders") or [])
    has_encoder = "h264_qsv" in set(info.get("encoders") or [])
    return bool(info.get("available") and has_decoder and has_encoder)


def _event_clip_cpu_command(concat_path: Path, local_start: float, duration: float, tmp_path: Path) -> list[str]:
    return [
        config.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-ss",
        f"{local_start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-y",
        str(tmp_path),
    ]


def _event_clip_vaapi_command(source_codec: str, concat_path: Path, local_start: float, duration: float, tmp_path: Path) -> list[str]:
    info = _ffmpeg_vaapi_info()
    device = str(info.get("device") or "/dev/dri/renderD128")
    return [
        config.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-vaapi_device",
        device,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-ss",
        f"{local_start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        "format=nv12,hwupload",
        "-c:v",
        "h264_vaapi",
        "-qp",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-y",
        str(tmp_path),
    ]


def _event_clip_qsv_command(source_codec: str, concat_path: Path, local_start: float, duration: float, tmp_path: Path) -> list[str]:
    decoder = "hevc_qsv" if source_codec == "hevc" else "h264_qsv"
    info = _ffmpeg_qsv_info()
    render_devices = info.get("render_devices") or []
    device_args = ["-qsv_device", render_devices[0]] if render_devices else []
    return [
        config.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        *device_args,
        "-hwaccel",
        "qsv",
        "-hwaccel_output_format",
        "qsv",
        "-c:v",
        decoder,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-ss",
        f"{local_start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "h264_qsv",
        "-preset",
        "veryfast",
        "-global_quality",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-y",
        str(tmp_path),
    ]

@app.get("/api/accelerator")
def accelerator() -> dict:
    system = platform.system()
    machine = platform.machine()
    is_macos = system == "Darwin"
    is_apple_silicon = is_macos and machine in {"arm64", "aarch64"}
    openvino_devices: list[str] = []
    openvino_error = ""
    coreml_available = False
    coreml_error = ""
    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core

        openvino_devices = list(Core().available_devices)
    except Exception as exc:
        openvino_error = str(exc) or "OpenVINO device probe failed"

    if is_macos:
        try:
            import coremltools  # noqa: F401

            coreml_available = True
        except Exception as exc:
            coreml_error = str(exc) or "Core ML probe failed"

    recommended_backend = "coreml" if is_apple_silicon and coreml_available else "openvino"
    vaapi_info = _ffmpeg_vaapi_info()
    qsv_info = _ffmpeg_qsv_info()

    return {
        "system": system,
        "machine": machine,
        "is_macos": is_macos,
        "is_apple_silicon": is_apple_silicon,
        "has_nvidia": shutil.which("nvidia-smi") is not None,
        "ffmpeg_path": config.ffmpeg_path,
        "ffprobe_path": _ffprobe_path(),
        "ffplay_path": _ffplay_path(),
        "openvino_devices": openvino_devices,
        "openvino_error": openvino_error,
        "coreml_available": coreml_available,
        "coreml_error": coreml_error,
        "recommended_openvino_device": "GPU" if "GPU" in openvino_devices else "CPU",
        "recommended_detector_backend": recommended_backend,
        "ffmpeg_hardware_acceleration": {
            "configured": _hardware_acceleration_mode(),
            "ffmpeg_path": config.ffmpeg_path,
            "ffprobe_path": _ffprobe_path(),
            "ffplay_path": _ffplay_path(),
            "vaapi": vaapi_info,
            "qsv": qsv_info,
        },
    }


@app.get("/api/detector/status")
def detector_status() -> dict:
    return manager.detector_status()


@app.get("/api/detector/models")
def detector_models() -> dict:
    models: list[dict] = []
    search_roots = sorted({Path("openvino_model"), *Path(".").glob("*_openvino_model")})
    for root in search_roots:
        if not root.exists():
            continue
        for xml_path in sorted(root.rglob("*.xml")):
            bin_path = xml_path.with_suffix(".bin")
            metadata_path = xml_path.parent / "metadata.yaml"
            item = {
                "path": str(xml_path),
                "name": xml_path.stem,
                "bin_path": str(bin_path),
                "bin_present": bin_path.exists(),
                "metadata_path": str(metadata_path) if metadata_path.exists() else "",
                "task": "",
                "classes": [],
                "input_shape": [],
                "output_shapes": [],
                "valid": False,
                "error": "",
            }
            if metadata_path.exists():
                try:
                    import yaml

                    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
                    names = metadata.get("names") or {}
                    if isinstance(names, dict):
                        item["classes"] = [str(value) for _, value in sorted(names.items(), key=lambda entry: int(entry[0]))]
                    elif isinstance(names, list):
                        item["classes"] = [str(value) for value in names]
                    item["task"] = str(metadata.get("task") or "")
                except Exception as exc:
                    item["error"] = f"Metadata: {exc}"
            try:
                try:
                    from openvino import Core
                except ImportError:
                    from openvino.runtime import Core

                model = Core().read_model(model=str(xml_path))
                item["input_shape"] = [int(value) for value in model.input(0).shape]
                item["output_shapes"] = [[int(value) for value in output.shape] for output in model.outputs]
                item["valid"] = bin_path.exists()
            except Exception as exc:
                item["error"] = str(exc)
            models.append(item)
    return {"models": models, "active_path": config.detector.resolved_model_path()}


@app.get("/api/event-clip/settings")
def event_clip_settings() -> dict:
    before, after = _event_clip_window(None, None)
    return {"before_seconds": before, "after_seconds": after}


@app.get("/api/recordings/cache/status")
def recording_cache_status() -> dict:
    root = manager.storage_dir / "playback-cache" / "fmp4"
    files = [path for path in root.glob("*/*") if path.is_file()] if root.exists() else []
    existing_files: list[Path] = []
    total_bytes = 0
    for path in files:
        try:
            total_bytes += path.stat().st_size
            existing_files.append(path)
        except FileNotFoundError:
            continue
    return {
        "path": str(root),
        "entries": len({path.parent for path in existing_files}),
        "bytes": total_bytes,
        "max_bytes": int(float(config.recording_cache_max_gb) * 1024 * 1024 * 1024),
        "max_days": int(config.recording_cache_max_days),
        "prewarm": bool(config.recording_cache_prewarm),
    }


@app.get("/api/system/status")
def system_status() -> dict:
    storage_path = manager.storage_dir
    storage_path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(storage_path)
    cameras = manager.statuses()
    detector = manager.detector_status()
    return {
        "storage": {
            "path": str(storage_path),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "used_percent": round((usage.used / usage.total) * 100, 1) if usage.total else 0,
        },
        "detector": detector,
        "cameras": {
            "total": len(cameras),
            "online": sum(1 for camera in cameras if camera.get("running")),
            "recording": sum(1 for camera in cameras if camera.get("recording")),
        },
        "mqtt": manager.mqtt_status(),
    }


@app.put("/api/config")
def put_config(next_config: AppConfig) -> dict:
    save_config(next_config)
    reload_manager(next_config)
    return {"ok": True, "cameras": len(next_config.cameras)}


@app.put("/api/config/cameras/{camera_id}/zones")
def put_camera_zones(camera_id: str, zones: list[DetectionZone]) -> dict:
    global config
    next_config = config.model_copy(deep=True)
    camera = camera_by_id(next_config, camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    worker = manager.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera worker not found")
    current_camera = camera_by_id(config, camera_id)
    previous_zones = [zone.model_dump(mode="json") for zone in (current_camera.zones if current_camera else [])]
    camera.zones = zones
    save_config(next_config, assign_ids=False)
    config = next_config
    manager.config = next_config
    manager.update_camera_zones(camera_id, camera.zones, previous_zones)
    return {
        "ok": True,
        "camera_id": camera.id,
        "zones": [zone.model_dump(mode="json") for zone in camera.zones],
        "workers_restarted": False,
    }


@app.put("/api/config/cameras/order")
def put_camera_order(camera_ids: list[str]) -> dict:
    global config
    existing_ids = [camera.id for camera in config.cameras]
    if len(camera_ids) != len(existing_ids) or len(set(camera_ids)) != len(camera_ids):
        raise HTTPException(status_code=400, detail="camera order must contain every camera exactly once")
    if set(camera_ids) != set(existing_ids):
        raise HTTPException(status_code=400, detail="camera order does not match configured cameras")
    camera_by_identifier = {camera.id: camera for camera in config.cameras}
    next_config = config.model_copy(deep=True)
    next_config.cameras = [camera_by_identifier[camera_id].model_copy(deep=True) for camera_id in camera_ids]
    save_config(next_config, assign_ids=False)

    config = next_config
    manager.config = next_config
    manager.workers = {camera_id: manager.workers[camera_id] for camera_id in camera_ids}
    return {"ok": True, "camera_ids": camera_ids}


@app.put("/api/config/cameras/{camera_id}")
def put_camera(camera_id: str, camera_settings: CameraConfig) -> dict:
    next_config = config.model_copy(deep=True)
    existing_index = next((index for index, item in enumerate(next_config.cameras) if item.id == camera_id), None)
    existing = next_config.cameras[existing_index] if existing_index is not None else None
    used_ids = {item.id for item in next_config.cameras if item.id != camera_id}
    base_id = slugify_camera_id(camera_settings.name or camera_settings.id)
    next_id = base_id
    suffix = 2
    while next_id in used_ids:
        next_id = f"{base_id}-{suffix}"
        suffix += 1
    camera_settings.id = next_id
    camera_settings.zones = existing.zones if existing is not None else []
    if existing_index is None:
        next_config.cameras.append(camera_settings)
    else:
        next_config.cameras[existing_index] = camera_settings
    save_config(next_config, assign_ids=False)
    reload_manager(next_config)
    return {"ok": True, "camera": camera_settings.model_dump(mode="json")}


@app.delete("/api/config/cameras/{camera_id}")
def delete_camera(camera_id: str) -> dict:
    next_config = config.model_copy(deep=True)
    remaining = [camera for camera in next_config.cameras if camera.id != camera_id]
    if len(remaining) == len(next_config.cameras):
        raise HTTPException(status_code=404, detail="camera not found")
    next_config.cameras = remaining
    save_config(next_config, assign_ids=False)
    reload_manager(next_config)
    return {"ok": True, "camera_id": camera_id}


@app.post("/api/config/probe")
def probe_config(payload: dict) -> dict:
    host = str(payload.get("host") or "").strip()
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "").strip()
    onvif_port = int(payload.get("onvif_port") or 8000)
    baichuan_port = int(payload.get("baichuan_port") or 9000)
    if not host:
        raise HTTPException(status_code=400, detail="host is required")

    result = {
        "host": host,
        "onvif": {
            "port": onvif_port,
            "reachable": _tcp_reachable(host, onvif_port),
            "capabilities": {},
            "error": "",
        },
        "baichuan": {
            "port": baichuan_port,
            "reachable": _tcp_reachable(host, baichuan_port),
        },
        "reolink_likely": False,
    }
    result["reolink_likely"] = bool(result["baichuan"]["reachable"])

    if result["onvif"]["reachable"] and username and password:
        try:
            from onvif import ONVIFCamera
            from zeep import Transport

            transport = Transport(operation_timeout=5)
            camera = ONVIFCamera(host, onvif_port, username, password, transport=transport)
            device = camera.create_devicemgmt_service()
            capabilities = device.GetCapabilities({"Category": "All"})
            result["onvif"]["capabilities"] = {
                "media": bool(getattr(capabilities, "Media", None)),
                "events": bool(getattr(capabilities, "Events", None)),
                "ptz": bool(getattr(capabilities, "PTZ", None)),
                "analytics": bool(getattr(capabilities, "Analytics", None)),
            }
        except Exception as exc:
            error = str(exc) or "ONVIF capability probe failed"
            result["onvif"]["error"] = error[:500]
    return result


@app.get("/api/events")
def events(limit: int = 100) -> list[dict]:
    return [_event_row(row) for row in manager.events.recent(limit)]


@app.get("/api/incidents")
def incidents(limit: int = 200, gap_seconds: int = 45) -> list[dict]:
    rows = [_event_row(row) for row in manager.events.recent(max(limit, 1))]
    return _incidents_with_faces(
        _incident_rows(rows, gap_seconds=max(5, min(gap_seconds, 300)))
    )


@app.get("/api/incidents/search")
def incident_search(
    day: str = "",
    time_zone: str = "America/New_York",
    camera_id: str = "",
    event_type: str = "motion",
    object_label: str = "",
    zone: str = "",
    limit: int = 18,
    offset: int = 0,
    gap_seconds: int = 45,
) -> dict:
    try:
        selected_zone = ZoneInfo(time_zone)
    except ZoneInfoNotFoundError as exc:
        raise HTTPException(status_code=422, detail="unknown timezone") from exc
    if day:
        try:
            selected_date = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="day must use YYYY-MM-DD") from exc
    else:
        selected_date = datetime.now(selected_zone).date()
        day = selected_date.isoformat()
    day_start = datetime.combine(selected_date, datetime.min.time(), selected_zone)
    day_end = day_start + timedelta(days=1)
    bounded_gap = max(5, min(gap_seconds, 300))
    query_start = day_start.astimezone(timezone.utc) - timedelta(seconds=bounded_gap)
    query_end = day_end.astimezone(timezone.utc) + timedelta(seconds=bounded_gap)
    rows = [
        _event_row(row)
        for row in manager.events.between(query_start.isoformat(), query_end.isoformat())
    ]
    day_start_epoch = day_start.timestamp()
    day_end_epoch = day_end.timestamp()
    day_incidents = [
        incident
        for incident in _incident_rows(rows, gap_seconds=bounded_gap)
        if incident["last_epoch"] >= day_start_epoch and incident["start_epoch"] < day_end_epoch
    ]
    facets = {
        "camera_ids": sorted({str(item.get("camera_id") or "") for item in day_incidents if item.get("camera_id")}),
        "labels": sorted({str(label) for item in day_incidents for label in item.get("labels", []) if label}),
        "zones": sorted({str(item_zone) for item in day_incidents for item_zone in item.get("zones", []) if item_zone}),
    }
    filtered = day_incidents
    if event_type == "object":
        filtered = [item for item in filtered if item.get("has_objects")]
    if camera_id:
        filtered = [item for item in filtered if item.get("camera_id") == camera_id]
    if object_label:
        filtered = [item for item in filtered if object_label in item.get("labels", [])]
    if zone:
        filtered = [item for item in filtered if zone in item.get("zones", [])]
    bounded_limit = max(1, min(limit, 100))
    bounded_offset = max(0, offset)
    page_items = filtered[bounded_offset:bounded_offset + bounded_limit]
    return {
        "items": _incidents_with_faces(page_items),
        "total": len(filtered),
        "limit": bounded_limit,
        "offset": bounded_offset,
        "day": day,
        "time_zone": time_zone,
        "start_at": day_start.astimezone(timezone.utc).isoformat(),
        "end_at": day_end.astimezone(timezone.utc).isoformat(),
        "facets": facets,
    }


class FacePersonCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    notes: str = Field(default="", max_length=1000)
    observation_id: int | None = None


class FaceAssignment(BaseModel):
    person_id: int | None = None


def _sync_face_observations(limit: int = 5000) -> int:
    global FACE_OBSERVATIONS_SYNCED
    if FACE_OBSERVATIONS_SYNCED:
        return 0
    inserted = manager.faces.ingest_events(manager.events.recent(max(1, min(limit, 20000))))
    FACE_OBSERVATIONS_SYNCED = True
    return inserted


@app.get("/api/faces/status")
def face_status() -> dict:
    _sync_face_observations()
    stats = manager.faces.stats()
    recognition = manager.faces.recognition_status()
    if recognition.get("ready"):
        recognition_message = (
            f"Recognition ready on {recognition.get('device') or 'OpenVINO'}; "
            f"{recognition.get('embedded', 0)} faces embedded and "
            f"{recognition.get('suggested', 0)} suggestions awaiting review."
        )
    else:
        recognition_message = str(recognition.get("error") or "Configure an OpenVINO face embedding model.")
    return {
        **stats,
        "recognition_ready": bool(recognition.get("ready")),
        "recognition_message": recognition_message,
        "recognition": recognition,
    }


@app.get("/api/faces/people")
def face_people() -> list[dict]:
    _sync_face_observations()
    return manager.faces.people()


@app.post("/api/faces/people")
def create_face_person(payload: FacePersonCreate) -> dict:
    return manager.faces.create_person(payload.name, payload.observation_id, payload.notes)


@app.delete("/api/faces/people/{person_id}")
def delete_face_person(person_id: int) -> dict:
    if not manager.faces.delete_person(person_id):
        raise HTTPException(status_code=404, detail="person not found")
    return {"deleted": True, "person_id": person_id}


@app.get("/api/faces/observations")
def face_observations(
    person_id: int | None = None,
    camera_id: str = "",
    status: str = "all",
    limit: int = 200,
    offset: int = 0,
) -> list[dict]:
    _sync_face_observations()
    return manager.faces.observations(
        person_id=person_id,
        camera_id=camera_id,
        status=status if status in {"all", "known", "unknown", "suggested"} else "all",
        limit=limit,
        offset=offset,
    )


@app.get("/api/faces/observations/count")
def face_observation_count(
    person_id: int | None = None,
    camera_id: str = "",
    status: str = "all",
) -> dict:
    _sync_face_observations()
    return {
        "total": manager.faces.observation_count(
            person_id=person_id,
            camera_id=camera_id,
            status=status if status in {"all", "known", "unknown", "suggested"} else "all",
        )
    }


@app.get("/api/faces/observations/{observation_id}")
def face_observation(observation_id: int) -> dict:
    observation = manager.faces.observation(observation_id)
    if observation is None:
        raise HTTPException(status_code=404, detail="face observation not found")
    return observation


@app.put("/api/faces/observations/{observation_id}")
def assign_face_observation(observation_id: int, payload: FaceAssignment) -> dict:
    try:
        observation = manager.faces.assign(observation_id, payload.person_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if observation is None:
        raise HTTPException(status_code=404, detail="face observation not found")
    return observation


@app.get("/api/faces/observations/{observation_id}/crop.jpg")
def face_crop(observation_id: int, padding: float = 0.2) -> Response:
    result = manager.faces.snapshot_path(observation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="face observation not found")
    snapshot_path, box = result
    frame = cv2.imread(str(snapshot_path))
    if frame is None:
        raise HTTPException(status_code=404, detail="snapshot is unavailable")
    height, width = frame.shape[:2]
    x1, y1 = float(box.get("x1", 0)), float(box.get("y1", 0))
    x2, y2 = float(box.get("x2", 0)), float(box.get("y2", 0))
    pad = max(0.0, min(float(padding), 1.0))
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    left, top = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    right, bottom = min(width, int(x2 + dx)), min(height, int(y2 + dy))
    if right <= left or bottom <= top:
        raise HTTPException(status_code=422, detail="face crop is invalid")
    ok, encoded = cv2.imencode(".jpg", frame[top:bottom, left:right], [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode face crop")
    return Response(encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/events/{event_id}/detect")
def detect_event_snapshot(event_id: int, confidence: float = 0.35) -> dict:
    event = manager.events.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    snapshot_path = Path(str(event.get("snapshot_path") or ""))
    if not snapshot_path.exists() or not snapshot_path.is_file():
        raise HTTPException(status_code=404, detail="snapshot not found")
    try:
        snapshot_path.resolve().relative_to(Path(config.storage_dir).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="snapshot outside storage directory") from None

    safe_confidence = max(0.01, min(0.99, float(confidence)))
    frame = cv2.imread(str(snapshot_path))
    if frame is None:
        raise HTTPException(status_code=422, detail="failed to read snapshot")

    started = time.perf_counter()
    camera = camera_by_id(config, str(event.get("camera_id") or ""))
    effective_confidence = detection_threshold(camera, safe_confidence) if camera else safe_confidence
    objects = manager.detector.detect(frame, confidence_threshold=effective_confidence)
    if camera:
        apply_detection_zones(camera, objects, int(frame.shape[1]), int(frame.shape[0]), safe_confidence)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    for detected_object in objects:
        detected_object["frame_source"] = detected_object.get("frame_source") or "manual_snapshot"
        detected_object["detection_source"] = "manual_openvino"
        detected_object["manual_confidence_threshold"] = safe_confidence
    persisted_event = manager.events.update_objects(event_id, objects_to_json(objects))
    if persisted_event is None:
        raise HTTPException(status_code=404, detail="event not found")
    detected = [
        item for item in objects
        if item.get("label") and item.get("box") and item.get("incident_eligible") is not False
    ]
    if detected:
        manager.publish_event("object", {
            "event_id": event_id,
            "camera_id": str(event.get("camera_id") or ""),
            "timestamp": str(event.get("created_at") or datetime.now(timezone.utc).isoformat()),
            "snapshot_path": str(snapshot_path),
            "recording_path": str(event.get("recording_path") or ""),
            "source": "manual_openvino",
            "objects": detected,
        })
    detector_status = manager.detector_status()
    return {
        "event_id": event_id,
        "camera_id": event.get("camera_id"),
        "snapshot_path": str(snapshot_path),
        "snapshot_width": int(frame.shape[1]),
        "snapshot_height": int(frame.shape[0]),
        "confidence": safe_confidence,
        "elapsed_ms": elapsed_ms,
        "objects": objects,
        "object_count": len(detected),
        "labels": sorted({str(item.get("label")) for item in detected}),
        "event": _event_row(persisted_event),
        "persisted": True,
        "detector": {
            "enabled": detector_status.get("enabled"),
            "loaded_backend": detector_status.get("loaded_backend"),
            "loaded_device": detector_status.get("loaded_device"),
            "configured_device": detector_status.get("configured_device"),
            "input_shape": detector_status.get("input_shape"),
            "output_format": detector_status.get("output_format"),
        },
    }


@app.post("/api/detector/frame")
async def detect_debug_frame(request: Request, confidence: float = 0.35) -> dict:
    content_length = int(request.headers.get("content-length") or 0)
    if content_length > 2 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="debug frame is too large")
    payload = await request.body()
    if not payload or len(payload) > 2 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="invalid debug frame")
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="failed to decode debug frame")
    safe_confidence = max(0.01, min(0.99, float(confidence)))
    started = time.perf_counter()
    objects = manager.detector.detect(frame, confidence_threshold=safe_confidence)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    detected = [item for item in objects if item.get("label") and item.get("box")]
    return {
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "confidence": safe_confidence,
        "elapsed_ms": elapsed_ms,
        "objects": detected,
    }


@app.get("/api/cameras/{camera_id}/snapshot.jpg")
def snapshot(camera_id: str, source: str = "live") -> Response:
    worker = manager.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera not found")
    image = worker.snapshot(source)
    if image is None:
        raise HTTPException(status_code=503, detail="no frame available")
    return Response(image, media_type="image/jpeg")


@app.get("/api/cameras/{camera_id}/zone-snapshot.jpg")
def zone_snapshot(camera_id: str, source: str = "live") -> Response:
    worker = manager.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera not found")
    image = worker.snapshot(source)
    if image is not None:
        return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    storage_root = Path(config.storage_dir).resolve()
    for event in manager.events.recent(1000):
        if event.get("camera_id") != camera_id:
            continue
        snapshot_path = Path(str(event.get("snapshot_path") or ""))
        if not snapshot_path.is_file():
            continue
        try:
            snapshot_path.resolve().relative_to(storage_root)
        except ValueError:
            continue
        return FileResponse(snapshot_path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    raise HTTPException(status_code=503, detail="no camera or event snapshot available")


@app.get("/api/cameras/{camera_id}/stream.mjpg")
async def stream(camera_id: str, request: Request, source: str = "live") -> StreamingResponse:
    worker = manager.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera not found")

    async def frames():
        while not await request.is_disconnected():
            image = await asyncio.to_thread(worker.snapshot, source)
            if image is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + image
                    + b"\r\n"
                )
            await asyncio.sleep(0.25 if image is not None else 0.1)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/api/cameras/{camera_id}/webrtc")
async def webrtc_signaling(websocket: WebSocket, camera_id: str) -> None:
    """Relay go2rtc signaling while its WebRTC media remains direct and shared."""
    try:
        upstream_url = _go2rtc_ws_url(
            camera_id,
            websocket.query_params.get("source", "live"),
            websocket.query_params.get("compat", "native"),
        )
    except (HTTPException, OSError, RuntimeError):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        async with websockets.connect(
            upstream_url,
            open_timeout=5,
            close_timeout=2,
            ping_interval=20,
            ping_timeout=10,
            max_size=2 * 1024 * 1024,
        ) as upstream:
            async def browser_to_go2rtc() -> None:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    if message.get("text") is not None:
                        await upstream.send(message["text"])
                    elif message.get("bytes") is not None:
                        await upstream.send(message["bytes"])

            async def go2rtc_to_browser() -> None:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            tasks = [
                asyncio.create_task(browser_to_go2rtc()),
                asyncio.create_task(go2rtc_to_browser()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        pass
    except Exception as exc:
        logging.getLogger(__name__).warning("WebRTC signaling failed for %s: %s", camera_id, exc)
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.get("/api/cameras/{camera_id}/hls/index.m3u8")
def hls_playlist_default(camera_id: str) -> FileResponse:
    return hls_playlist(camera_id, "live")


@app.get("/api/cameras/{camera_id}/hls/{source}/index.m3u8")
def hls_playlist(camera_id: str, source: str) -> FileResponse:
    source = normalize_source(source)
    camera = next((item for item in config.cameras if item.id == camera_id), None)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    manager.hls.start(camera, source)
    playlist = manager.hls.wait_for_playlist(camera_id, source)
    if playlist is None or not playlist.exists():
        raise HTTPException(status_code=503, detail="HLS stream is starting")
    return FileResponse(
        playlist,
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/cameras/{camera_id}/hls/{filename}")
def hls_file_default(camera_id: str, filename: str) -> FileResponse:
    return hls_file(camera_id, "live", filename)


@app.get("/api/cameras/{camera_id}/hls/{source}/{filename}")
def hls_file(camera_id: str, source: str, filename: str) -> FileResponse:
    source = normalize_source(source)
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="invalid filename")
    path = manager.hls.file_path(camera_id, source, filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="HLS segment not found")
    manager.hls.touch(camera_id, source)
    media_type = "video/mp2t" if path.suffix == ".ts" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-store"})


@app.websocket("/api/cameras/{camera_id}/mse")
async def mse_stream(websocket: WebSocket, camera_id: str) -> None:
    source = normalize_source(websocket.query_params.get("source", "live"))
    stream_key = f"{camera_id}:{source}"
    camera = next((item for item in config.cameras if item.id == camera_id), None)
    if camera is None:
        await websocket.close(code=1008)
        return
    if stream_key in active_mse_streams:
        await websocket.close(code=1013)
        return

    await websocket.accept()
    active_mse_streams.add(stream_key)
    command = [
        config.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        *ffmpeg_input_args(camera, source),
        "-an",
        "-sn",
        "-c:v",
        "copy",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if is_native_baichuan(camera) else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    pipe = start_ffmpeg_pipe(camera, source, process)

    try:
        init_buffer = bytearray()
        sent_init = False
        while process.poll() is None:
            if process.stdout is None:
                break
            chunk = await asyncio.to_thread(process.stdout.read, 64 * 1024)
            if not chunk:
                await asyncio.sleep(0.02)
                continue
            if not sent_init:
                init_buffer.extend(chunk)
                if b"moov" not in init_buffer and len(init_buffer) < 512 * 1024:
                    continue
                await websocket.send_bytes(bytes(init_buffer))
                init_buffer.clear()
                sent_init = True
                continue
            await websocket.send_bytes(chunk)
    except WebSocketDisconnect:
        pass
    finally:
        active_mse_streams.discard(stream_key)
        if pipe is not None:
            pipe.stop()
        try:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)



@app.post("/api/cameras/{camera_id}/camera/start")
def start_camera(camera_id: str) -> dict:
    if not manager.start_camera(camera_id):
        raise HTTPException(status_code=404, detail="camera not found")
    return {"ok": True}


@app.post("/api/cameras/{camera_id}/camera/stop")
def stop_camera(camera_id: str) -> dict:
    if not manager.stop_camera(camera_id):
        raise HTTPException(status_code=404, detail="camera not found")
    return {"ok": True}

@app.post("/api/cameras/{camera_id}/motion-test")
def motion_test(camera_id: str) -> dict:
    worker = manager.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera not found")
    worker.handle_motion_event("manual/test", "manual GUI trigger")
    return {"ok": True}


@app.post("/api/cameras/{camera_id}/recording/start")
def start_recording(camera_id: str, source: str = "main") -> dict:
    camera = next((item for item in config.cameras if item.id == camera_id), None)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    manager.recorder.start(camera, recording_source(source))
    return {"ok": True}


@app.post("/api/cameras/{camera_id}/recording/stop")
def stop_recording(camera_id: str, source: str | None = None) -> dict:
    manager.recorder.stop(camera_id, recording_source(source) if source else None)
    return {"ok": True}


@app.get("/api/cameras/{camera_id}/recordings")
def recordings(camera_id: str, limit: int = 200, source: str = "main") -> list[dict]:
    return _recording_rows(camera_id, limit=max(1, min(limit, RECORDING_LOOKUP_LIMIT)), source=recording_source(source))


@app.get("/api/cameras/{camera_id}/recordings/events")
def recording_events(camera_id: str, limit: int = 1000, source: str = "main") -> list[dict]:
    rows = _recording_rows(camera_id, limit=RECORDING_LOOKUP_LIMIT, source=recording_source(source))
    if not rows:
        return []
    start_epoch = rows[0].get("start_epoch")
    end_epoch = rows[-1].get("end_epoch")
    if start_epoch is None or end_epoch is None:
        return []

    events = manager.events.for_camera_range(
        camera_id,
        datetime.fromtimestamp(float(start_epoch), timezone.utc).isoformat(),
        datetime.fromtimestamp(float(end_epoch), timezone.utc).isoformat(),
        limit=max(1, min(limit, 5000)),
    )
    return [_recording_event_row(event, rows) for event in events]


@app.get("/api/cameras/{camera_id}/recordings/day")
def recording_day(
    camera_id: str,
    start_epoch: float,
    end_epoch: float,
    source: str = "main",
) -> dict:
    if end_epoch <= start_epoch or end_epoch - start_epoch > 90000:
        raise HTTPException(status_code=400, detail="invalid recording day range")
    selected_source = recording_source(source)
    rows = _recording_day_rows(camera_id, start_epoch, end_epoch, selected_source)
    available_sources = [
        candidate for candidate in ("main", "live")
        if _recording_day_rows(camera_id, start_epoch, end_epoch, candidate)
    ]
    events = manager.events.for_camera_range(
        camera_id,
        datetime.fromtimestamp(start_epoch, timezone.utc).isoformat(),
        datetime.fromtimestamp(end_epoch, timezone.utc).isoformat(),
        limit=5000,
    )
    return {
        "camera_id": camera_id,
        "source": selected_source,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "recordings": rows,
        "events": [_event_row(event) for event in events],
        "available_sources": available_sources,
    }


def _recording_day_rows(camera_id: str, start_epoch: float, end_epoch: float, source: str) -> list[dict]:
    selected_source = recording_source(source)
    cache_key = (camera_id, selected_source, int(start_epoch), int(end_epoch))
    now = time.monotonic()
    with RECORDING_DAY_CACHE_LOCK:
        cached = RECORDING_DAY_CACHE.get(cache_key)
        if cached is not None and now - cached[0] < RECORDING_DAY_CACHE_SECONDS:
            return cached[1]
    rows = [
        row for row in manager.recorder.recording_rows_between(
            camera_id,
            start_epoch,
            end_epoch,
            selected_source,
        )
        if int(row.get("size_bytes") or 0) > 1024
    ]
    with RECORDING_DAY_CACHE_LOCK:
        RECORDING_DAY_CACHE[cache_key] = (now, rows)
        expired = [key for key, value in RECORDING_DAY_CACHE.items() if now - value[0] >= RECORDING_DAY_CACHE_SECONDS]
        for key in expired:
            RECORDING_DAY_CACHE.pop(key, None)
    return rows


def _recording_fmp4_files(path: Path, duration: float, media_offset: float) -> tuple[Path, Path]:
    stat = path.stat()
    fingerprint = f"v3:{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{duration:.3f}:{media_offset:.3f}"
    cache_key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
    cache_dir = manager.storage_dir / "playback-cache" / "fmp4" / cache_key
    init_path = cache_dir / "init.mp4"
    media_path = cache_dir / "media.m4s"
    if (
        init_path.exists()
        and init_path.stat().st_size > 0
        and media_path.exists()
        and media_path.stat().st_size > 0
    ):
        return init_path, media_path

    with RECORDING_FMP4_LOCKS_GUARD:
        lock = RECORDING_FMP4_LOCKS.setdefault(cache_key, threading.Lock())
    with lock:
        if (
            init_path.exists()
            and init_path.stat().st_size > 0
            and media_path.exists()
            and media_path.stat().st_size > 0
        ):
            return init_path, media_path
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="fmp4-", dir=cache_dir))
        codec = _probe_video_codec(path)
        command = [
            config.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(path),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            "-output_ts_offset",
            f"{media_offset:.3f}",
        ]
        if codec in {"hevc", "h265"}:
            command.extend(["-tag:v", "hvc1"])
        command.extend([
            "-f",
            "hls",
            "-hls_time",
            "300",
            "-hls_list_size",
            "0",
            "-hls_segment_type",
            "fmp4",
            "-hls_fmp4_init_filename",
            "init.mp4",
            "-hls_segment_filename",
            str(temp_dir / "media_%d.m4s"),
            str(temp_dir / "index.m3u8"),
        ])
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
        generated_init = temp_dir / "init.mp4"
        generated_media = temp_dir / "media_0.m4s"
        if result.returncode != 0 or not generated_init.exists() or not generated_media.exists():
            error = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            shutil.rmtree(temp_dir, ignore_errors=True)
            if time.time() - stat.st_mtime >= float(config.recording_segment_seconds) * 2:
                manager.recorder.mark_unplayable(path, error or "recording fragment failed")
            with RECORDING_DAY_CACHE_LOCK:
                RECORDING_DAY_CACHE.clear()
            raise HTTPException(status_code=500, detail=f"recording fragment failed: {error[-300:]}")
        try:
            _offset_fmp4_timestamps(generated_init, generated_media, media_offset)
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"recording fragment timestamp repair failed: {exc}") from exc
        os.replace(generated_init, init_path)
        os.replace(generated_media, media_path)
        shutil.rmtree(temp_dir, ignore_errors=True)
        _maintain_recording_cache(cache_dir)
        return init_path, media_path


def _maintain_recording_cache(active_dir: Path) -> None:
    global RECORDING_CACHE_LAST_MAINTENANCE
    now_monotonic = time.monotonic()
    if now_monotonic - RECORDING_CACHE_LAST_MAINTENANCE < 600:
        return
    if not RECORDING_CACHE_MAINTENANCE_LOCK.acquire(blocking=False):
        return
    try:
        RECORDING_CACHE_LAST_MAINTENANCE = now_monotonic
        root = manager.storage_dir / "playback-cache" / "fmp4"
        if not root.exists():
            return
        now_epoch = time.time()
        entries: list[tuple[float, int, Path]] = []
        for directory in root.iterdir():
            if not directory.is_dir() or directory == active_dir:
                continue
            files = [item for item in directory.iterdir() if item.is_file()]
            modified_at = max((item.stat().st_mtime for item in files), default=directory.stat().st_mtime)
            size = sum(item.stat().st_size for item in files)
            max_age_seconds = int(config.recording_cache_max_days) * 24 * 60 * 60
            if now_epoch - modified_at > max_age_seconds:
                shutil.rmtree(directory, ignore_errors=True)
            else:
                entries.append((modified_at, size, directory))
        total_size = sum(size for _, size, _ in entries)
        max_bytes = int(float(config.recording_cache_max_gb) * 1024 * 1024 * 1024)
        for _, size, directory in sorted(entries):
            if total_size <= max_bytes:
                break
            shutil.rmtree(directory, ignore_errors=True)
            total_size -= size
    finally:
        RECORDING_CACHE_MAINTENANCE_LOCK.release()


def _start_recording_prewarmer() -> None:
    global RECORDING_PREWARM_THREAD
    RECORDING_PREWARM_STOP.clear()
    if RECORDING_PREWARM_THREAD is not None and RECORDING_PREWARM_THREAD.is_alive():
        return
    RECORDING_PREWARM_THREAD = threading.Thread(
        target=_recording_prewarm_loop,
        name="recording-prewarmer",
        daemon=False,
    )
    RECORDING_PREWARM_THREAD.start()


def _stop_recording_prewarmer() -> None:
    global RECORDING_PREWARM_THREAD
    RECORDING_PREWARM_STOP.set()
    if RECORDING_PREWARM_THREAD is not None:
        RECORDING_PREWARM_THREAD.join(timeout=35)
        if RECORDING_PREWARM_THREAD.is_alive():
            logging.getLogger(__name__).error("recording prewarmer did not stop")
        else:
            RECORDING_PREWARM_THREAD = None


def _recording_prewarm_loop() -> None:
    while not RECORDING_PREWARM_STOP.wait(5):
        if not config.recording_cache_prewarm:
            continue
        for camera in config.cameras:
            sources = []
            if camera.record:
                sources.append("main")
            if camera.record_sub and camera.live_stream_url:
                sources.append("live")
            for source in sources:
                if RECORDING_PREWARM_STOP.is_set():
                    return
                try:
                    row = manager.recorder.latest_indexed_row(camera.id, source)
                    if row is None:
                        continue
                    path = Path(row["path"])
                    if not path.exists() or time.time() - path.stat().st_mtime < float(config.recording_segment_seconds) * 2:
                        continue
                    start_epoch = float(row["start_epoch"])
                    local_day = datetime.fromtimestamp(start_epoch).replace(hour=0, minute=0, second=0, microsecond=0)
                    day_start = local_day.timestamp()
                    day_end = (local_day + timedelta(days=1)).timestamp()
                    rows = manager.recorder.recording_rows_between(camera.id, day_start, day_end, source)
                    index = next((i for i, item in enumerate(rows) if item["path"] == row["path"]), None)
                    if index is None:
                        continue
                    media_offset = sum(float(item["duration_seconds"]) for item in rows[:index])
                    _recording_fmp4_files(path, float(row["duration_seconds"]), media_offset)
                except Exception:
                    logging.getLogger(__name__).exception("Recording prewarm failed for %s/%s", camera.id, source)


@app.get("/api/cameras/{camera_id}/recordings/day.m3u8")
def recording_day_hls_playlist(
    camera_id: str,
    start_epoch: float,
    end_epoch: float,
    source: str = "main",
) -> Response:
    if end_epoch <= start_epoch or end_epoch - start_epoch > 90000:
        raise HTTPException(status_code=400, detail="invalid recording day range")
    selected_source = recording_source(source)
    rows = _recording_day_rows(camera_id, start_epoch, end_epoch, selected_source)
    if not rows:
        raise HTTPException(status_code=404, detail="no recordings found")
    target_duration = max(1, math.ceil(max(float(row["duration_seconds"]) for row in rows)))
    query = f"start_epoch={start_epoch:.3f}&end_epoch={end_epoch:.3f}&source={selected_source}"
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:7",
        f"#EXT-X-TARGETDURATION:{target_duration}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
    ]
    media_offset = 0.0
    for index, row in enumerate(rows):
        row_start = float(row["start_epoch"])
        segment_query = f"{query}&media_offset={media_offset:.3f}"
        lines.extend([
            f"#EXT-X-PROGRAM-DATE-TIME:{datetime.fromtimestamp(row_start, timezone.utc).isoformat()}",
            f"#EXTINF:{float(row['duration_seconds']):.3f},",
            f"day/{index}.m4s?{segment_query}",
        ])
        if index == 0:
            lines.insert(5, f'#EXT-X-MAP:URI="day/0/init.mp4?{segment_query}"')
        media_offset += float(row["duration_seconds"])
    lines.append("#EXT-X-ENDLIST")
    return Response(
        "\n".join(lines) + "\n",
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-store"},
    )


def _recording_day_fmp4_paths(
    camera_id: str,
    segment_index: int,
    start_epoch: float,
    end_epoch: float,
    source: str = "main",
    media_offset: float = 0.0,
) -> tuple[Path, Path]:
    rows = _recording_day_rows(camera_id, start_epoch, end_epoch, source)
    if segment_index < 0 or segment_index >= len(rows):
        raise HTTPException(status_code=404, detail="recording segment not found")
    path = Path(rows[segment_index]["path"])
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="recording file not found")
    segment_duration = max(0.1, min(float(rows[segment_index]["duration_seconds"]), 300.0))
    expected_offset = sum(float(row["duration_seconds"]) for row in rows[:segment_index])
    if abs(media_offset - expected_offset) > 0.1:
        media_offset = expected_offset
    return _recording_fmp4_files(path, segment_duration, media_offset)


@app.get("/api/cameras/{camera_id}/recordings/day/{segment_index}/init.mp4")
def recording_day_hls_init(
    camera_id: str,
    segment_index: int,
    start_epoch: float,
    end_epoch: float,
    source: str = "main",
    media_offset: float = 0.0,
) -> FileResponse:
    init_path, _ = _recording_day_fmp4_paths(camera_id, segment_index, start_epoch, end_epoch, source, media_offset)
    return FileResponse(init_path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/cameras/{camera_id}/recordings/day/{segment_index}.m4s")
def recording_day_hls_segment(
    camera_id: str,
    segment_index: int,
    start_epoch: float,
    end_epoch: float,
    source: str = "main",
    media_offset: float = 0.0,
) -> FileResponse:
    _, media_path = _recording_day_fmp4_paths(camera_id, segment_index, start_epoch, end_epoch, source, media_offset)
    return FileResponse(media_path, media_type="video/iso.segment", headers={"Cache-Control": "private, max-age=86400"})




@app.get("/api/events/{event_id}/clip.mp4")
def event_clip(event_id: int, before: float | None = None, after: float | None = None, source: str = "main") -> FileResponse:
    event = manager.events.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    enriched = _event_row(event)
    before_seconds, after_seconds = _event_clip_window(before, after)
    clip_source = recording_source(source)
    clip_path = _event_clip_path(enriched, before=before_seconds, after=after_seconds, source=clip_source)
    if not clip_path.exists() or clip_path.stat().st_size == 0:
        cache_key = str(clip_path)
        with EVENT_CLIP_LOCKS_GUARD:
            lock = EVENT_CLIP_LOCKS.setdefault(cache_key, threading.Lock())
        with lock:
            if not clip_path.exists() or clip_path.stat().st_size == 0:
                _build_event_clip(enriched, before=before_seconds, after=after_seconds, output_path=clip_path, source=clip_source)
    return FileResponse(
        clip_path,
        media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/events/{event_id}/stream.m3u8")
def event_stream(event_id: int, before: float | None = None, after: float | None = None, source: str = "main") -> Response:
    event = manager.events.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    enriched = _event_row(event)
    camera_id = str(enriched.get("camera_id") or "")
    if not camera_id:
        raise HTTPException(status_code=400, detail="event is missing camera")
    before_seconds, after_seconds = _event_clip_window(before, after)
    event_epoch = _event_epoch(enriched)
    window_start = event_epoch - before_seconds
    window_end = event_epoch + after_seconds
    selected_source = recording_source(source)
    rows = _recording_day_rows(camera_id, window_start, window_end, selected_source)
    if not rows:
        raise HTTPException(status_code=404, detail="no recording window found")

    first_start = float(rows[0]["start_epoch"])
    start_offset = max(0.0, window_start - first_start)
    target_duration = max(1, math.ceil(max(float(row["duration_seconds"]) for row in rows)))
    query = f"start_epoch={window_start:.3f}&end_epoch={window_end:.3f}&source={selected_source}"
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:7",
        f"#EXT-X-TARGETDURATION:{target_duration}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        f"#EXT-X-START:TIME-OFFSET={start_offset:.3f},PRECISE=YES",
    ]
    media_offset = 0.0
    for index, row in enumerate(rows):
        row_start = float(row["start_epoch"])
        segment_query = f"{query}&media_offset={media_offset:.3f}"
        if index == 0:
            lines.append(
                f'#EXT-X-MAP:URI="/api/cameras/{quote(camera_id, safe="")}/recordings/day/0/init.mp4?{segment_query}"'
            )
        lines.extend([
            f"#EXT-X-PROGRAM-DATE-TIME:{datetime.fromtimestamp(row_start, timezone.utc).isoformat()}",
            f"#EXTINF:{float(row['duration_seconds']):.3f},",
            f"/api/cameras/{quote(camera_id, safe='')}/recordings/day/{index}.m4s?{segment_query}",
        ])
        media_offset += float(row["duration_seconds"])
    lines.append("#EXT-X-ENDLIST")
    return Response(
        "\n".join(lines) + "\n",
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "private, max-age=30"},
    )




def _recording_rows(camera_id: str, limit: int, source: str = "main") -> list[dict]:
    return manager.recorder.recording_rows(camera_id, limit=limit, source=recording_source(source))


def _event_row(row: dict) -> dict:
    event = dict(row)
    try:
        objects = json.loads(event.pop("objects_json", "[]") or "[]")
    except json.JSONDecodeError:
        objects = []
    detected_objects = [
        item for item in objects
        if item.get("label")
        and float(item.get("confidence") or 0) > 0
        and item.get("incident_eligible") is not False
    ]
    event["objects"] = objects
    event["has_objects"] = bool(detected_objects)
    event["labels"] = sorted({str(item["label"]) for item in detected_objects})
    event["zones"] = sorted({
        str(zone_name)
        for item in detected_objects
        for zone_name in item.get("zones", [])
        if zone_name
    })
    return event


def _event_epoch(event: dict) -> float:
    return datetime.fromisoformat(event["created_at"]).timestamp()


def _best_incident_event(events: list[dict]) -> dict:
    object_events = [event for event in events if event.get("has_objects")]
    candidates = object_events or events

    def score(event: dict) -> tuple[float, int]:
        objects = event.get("objects") or []
        best_confidence = max((float(item.get("confidence") or 0) for item in objects), default=0.0)
        return (best_confidence, int(event.get("id") or 0))

    return max(candidates, key=score)


def _incident_rows(rows: list[dict], gap_seconds: int = 45) -> list[dict]:
    by_camera: dict[str, list[dict]] = {}
    for event in rows:
        by_camera.setdefault(str(event.get("camera_id") or ""), []).append(event)

    incidents: list[dict] = []
    for camera_id, camera_events in by_camera.items():
        ordered = sorted(camera_events, key=_event_epoch)
        current: list[dict] = []
        current_end = 0.0
        for event in ordered:
            event_epoch = _event_epoch(event)
            if current and event_epoch - current_end > gap_seconds:
                incidents.append(_incident_row(camera_id, current))
                current = []
            current.append(event)
            current_end = event_epoch
        if current:
            incidents.append(_incident_row(camera_id, current))

    incidents.sort(key=lambda item: item["last_epoch"], reverse=True)
    return incidents


def _incidents_with_faces(incidents: list[dict]) -> list[dict]:
    event_ids = [
        int(event["id"])
        for incident in incidents
        for event in incident.get("events", [])
        if str(event.get("id", "")).isdigit()
    ]
    observations_by_event: dict[int, list[dict]] = {}
    for observation in manager.faces.for_event_ids(event_ids):
        observations_by_event.setdefault(int(observation["event_id"]), []).append(observation)

    status_rank = {"confirmed": 0, "possible": 1, "unknown": 2}

    def summarize(observations: list[dict]) -> list[dict]:
        summaries: dict[tuple[str, int], dict] = {}
        for observation in observations:
            person_id = observation.get("person_id")
            candidate_id = observation.get("candidate_person_id")
            if person_id is not None:
                status = "confirmed"
                identity_id = int(person_id)
                name = str(observation.get("person_name") or "Unknown")
                confidence = observation.get("match_confidence")
            elif candidate_id is not None:
                status = "possible"
                identity_id = int(candidate_id)
                name = str(observation.get("candidate_person_name") or "Unknown")
                confidence = observation.get("candidate_confidence")
            else:
                status = "unknown"
                identity_id = 0
                name = "Unknown"
                confidence = observation.get("candidate_confidence")
                if confidence is None:
                    confidence = observation.get("confidence")
            score = max(0.0, min(1.0, float(confidence or 0)))
            key = (status, identity_id)
            current = summaries.get(key)
            if current is None or score > current["confidence"]:
                summaries[key] = {
                    "observation_id": int(observation["observation_id"]),
                    "name": name,
                    "status": status,
                    "confidence": round(score, 4),
                }
        return sorted(
            summaries.values(),
            key=lambda face: (status_rank[face["status"]], -face["confidence"], face["name"].lower()),
        )

    for incident in incidents:
        incident_observations: list[dict] = []
        for event in incident.get("events", []):
            event_observations = observations_by_event.get(int(event.get("id") or 0), [])
            event["faces"] = summarize(event_observations)
            incident_observations.extend(event_observations)
        incident["faces"] = summarize(incident_observations)
    return incidents


def _incident_event_payload(event: dict) -> dict:
    payload = dict(event)
    payload.pop("topic", None)
    payload.pop("message", None)
    return payload


def _incident_row(camera_id: str, events: list[dict]) -> dict:
    ordered = sorted(events, key=_event_epoch)
    first = ordered[0]
    last = ordered[-1]
    representative = _best_incident_event(ordered)
    representative_payload = _incident_event_payload(representative)
    labels = sorted({label for event in ordered for label in event.get("labels", [])})
    zones = sorted({zone for event in ordered for zone in event.get("zones", [])})
    start_epoch = _event_epoch(first)
    last_epoch = _event_epoch(last)
    object_count = sum(1 for event in ordered if event.get("has_objects"))
    incident = {
        **representative_payload,
        "id": f"incident-{camera_id}-{first.get('id')}-{last.get('id')}",
        "incident_id": f"{camera_id}-{first.get('id')}-{last.get('id')}",
        "representative_event_id": representative.get("id"),
        "camera_id": camera_id,
        "kind": "motion",
        "created_at": representative.get("created_at"),
        "start_at": first.get("created_at"),
        "end_at": last.get("created_at"),
        "start_epoch": start_epoch,
        "last_epoch": last_epoch,
        "duration_seconds": max(0.0, last_epoch - start_epoch),
        "event_count": len(ordered),
        "object_event_count": object_count,
        "has_objects": bool(labels),
        "labels": labels,
        "zones": zones,
        "events": [_incident_event_payload(event) for event in reversed(ordered)],
    }
    return incident


def _recording_event_row(event: dict, recordings: list[dict]) -> dict:
    event = _event_row(event)
    created_epoch = _event_epoch(event)
    first_start = float(recordings[0]["start_epoch"])
    event["timeline_offset"] = max(0.0, created_epoch - first_start)
    return event


def _event_clip_window(before: float | None, after: float | None) -> tuple[float, float]:
    configured_before = config.event_clip_before_seconds if before is None else before
    configured_after = config.event_clip_after_seconds if after is None else after
    safe_before = max(0.0, min(float(configured_before or 0.0), 3600.0))
    safe_after = max(0.0, min(float(configured_after or 0.0), 3600.0))
    return safe_before, safe_after


def _event_clip_path(event: dict, before: float, after: float, source: str = "main") -> Path:
    event_id = int(event.get("id") or 0)
    camera_id = str(event.get("camera_id") or "camera").replace("/", "_").replace("\\", "_")
    safe_before = int(max(0.0, min(float(before or 5.0), 3600.0)) * 1000)
    safe_after = int(max(0.0, min(float(after or 5.0), 3600.0)) * 1000)
    clip_source = recording_source(source)
    clip_dir = manager.storage_dir / "event_clips" / camera_id / clip_source
    clip_dir.mkdir(parents=True, exist_ok=True)
    accel_mode = _hardware_acceleration_mode()
    return clip_dir / f"{event_id}-{safe_before}-{safe_after}-a3-{accel_mode}.mp4"


def _build_event_clip(event: dict, before: float, after: float, output_path: Path, source: str = "main") -> None:
    camera_id = str(event.get("camera_id") or "")
    if not camera_id:
        raise HTTPException(status_code=400, detail="event is missing camera")

    event_epoch = _event_epoch(event)
    window_before = max(0.0, min(float(before or 5.0), 3600.0))
    window_after = max(0.0, min(float(after or 5.0), 3600.0))
    window_start = event_epoch - window_before
    window_end = event_epoch + window_after

    rows = [
        row for row in _recording_rows(camera_id, limit=RECORDING_LOOKUP_LIMIT, source=recording_source(source))
        if row.get("start_epoch") is not None
        and row.get("end_epoch") is not None
        and Path(str(row.get("path") or "")).exists()
    ]
    rows.sort(key=lambda row: float(row["start_epoch"]))
    selected = [
        row for row in rows
        if float(row["end_epoch"]) > window_start and float(row["start_epoch"]) < window_end
    ]
    if not selected:
        raise HTTPException(status_code=404, detail="no recording window found")

    concat_start_epoch = float(selected[0]["start_epoch"])
    concat_end_epoch = max(float(row["end_epoch"]) for row in selected)
    local_start = max(0.0, window_start - concat_start_epoch)
    local_end = min(window_end, concat_end_epoch)
    duration = max(0.1, local_end - (concat_start_epoch + local_start))

    concat_path = _write_concat_file(selected)
    tmp_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.mp4")
    source_codec = _probe_video_codec(Path(str(selected[0]["path"])))
    commands: list[tuple[str, list[str]]] = []
    if _event_clip_vaapi_enabled(source_codec):
        commands.append(("vaapi", _event_clip_vaapi_command(source_codec, concat_path, local_start, duration, tmp_path)))
    if _event_clip_qsv_enabled(source_codec):
        commands.append(("qsv", _event_clip_qsv_command(source_codec, concat_path, local_start, duration, tmp_path)))
    commands.append(("cpu", _event_clip_cpu_command(concat_path, local_start, duration, tmp_path)))

    try:
        last_error = "event clip generation failed"
        for backend, command in commands:
            tmp_path.unlink(missing_ok=True)
            clip_timeout = max(60.0, min(600.0, duration * 2.0))
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=clip_timeout)
            if result.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
                tmp_path.replace(output_path)
                logging.getLogger(__name__).info(
                    "built event clip %s using %s acceleration (source codec %s)",
                    output_path.name,
                    backend,
                    source_codec or "unknown",
                )
                return
            last_error = (result.stderr or f"event clip generation failed using {backend}").strip()[-500:]
            if backend in {"vaapi", "qsv"}:
                logging.getLogger(__name__).warning(
                    "%s event clip generation failed for %s, falling back to next backend: %s",
                    backend.upper(),
                    output_path.name,
                    last_error,
                )
        raise HTTPException(status_code=500, detail=last_error or "event clip generation failed")
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="event clip generation timed out") from exc
    finally:
        concat_path.unlink(missing_ok=True)
        tmp_path.unlink(missing_ok=True)


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _write_concat_file(rows: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".ffconcat",
        prefix="survng-recordings-",
        delete=False,
    )
    with handle:
        for row in rows:
            path = Path(row["path"]).resolve()
            escaped = str(path).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
    return Path(handle.name)


def _recording_start_epoch(path: Path) -> float | None:
    return manager.recorder.recording_start_epoch(path)


@app.get("/api/files")
def file(path: str) -> FileResponse:
    requested = Path(path).resolve()
    storage_root = manager.storage_dir.resolve()
    if storage_root not in requested.parents and requested != storage_root:
        raise HTTPException(status_code=403, detail="file outside storage dir")
    if not requested.exists():
        raise HTTPException(status_code=404, detail="file not found")
    media_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
    return FileResponse(requested, media_type=media_type)

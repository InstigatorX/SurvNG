from __future__ import annotations

import json
import logging
import math
import mimetypes
import asyncio
import os
import re
import platform
import shutil
import socket
import subprocess
import tempfile
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .baichuan_native import ffmpeg_input_args, is_native_baichuan, start_ffmpeg_pipe
from .config import AppConfig, load_config, save_config
from .manager import AppManager

config = load_config()
manager = AppManager(config)
active_mse_streams: set[str] = set()
LOG_LINES: deque[dict] = deque(maxlen=1000)
SECRET_URL_RE = re.compile(r"(\b(?:rtsp|rtmp|http|https|reolink)://)([^:/@\s]+):([^@\s]+)@", re.IGNORECASE)


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


def normalize_source(source: str) -> str:
    return "main" if source == "main" else "live"


def reload_manager(next_config: AppConfig) -> None:
    global config, manager
    manager.stop_all()
    config = next_config
    manager = AppManager(config)
    manager.start_all()


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.start_all()
    yield
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

    return {
        "system": system,
        "machine": machine,
        "is_macos": is_macos,
        "is_apple_silicon": is_apple_silicon,
        "has_nvidia": shutil.which("nvidia-smi") is not None,
        "openvino_devices": openvino_devices,
        "openvino_error": openvino_error,
        "coreml_available": coreml_available,
        "coreml_error": coreml_error,
        "recommended_openvino_device": "GPU" if "GPU" in openvino_devices else "CPU",
        "recommended_detector_backend": recommended_backend,
    }


@app.get("/api/detector/status")
def detector_status() -> dict:
    return manager.detector_status()


@app.get("/api/event-clip/settings")
def event_clip_settings() -> dict:
    before, after = _event_clip_window(None, None)
    return {"before_seconds": before, "after_seconds": after}


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
    }


@app.put("/api/config")
def put_config(next_config: AppConfig) -> dict:
    save_config(next_config)
    reload_manager(next_config)
    return {"ok": True, "cameras": len(next_config.cameras)}


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
    return _incident_rows(rows, gap_seconds=max(5, min(gap_seconds, 300)))


@app.get("/api/cameras/{camera_id}/snapshot.jpg")
def snapshot(camera_id: str, source: str = "live") -> Response:
    worker = manager.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera not found")
    image = worker.snapshot(source)
    if image is None:
        raise HTTPException(status_code=503, detail="no frame available")
    return Response(image, media_type="image/jpeg")


@app.get("/api/cameras/{camera_id}/stream.mjpg")
def stream(camera_id: str, source: str = "live") -> StreamingResponse:
    worker = manager.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera not found")
    return StreamingResponse(
        worker.mjpeg_frames(source=source),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


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
def start_recording(camera_id: str) -> dict:
    camera = next((item for item in config.cameras if item.id == camera_id), None)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    manager.recorder.start(camera)
    return {"ok": True}


@app.post("/api/cameras/{camera_id}/recording/stop")
def stop_recording(camera_id: str) -> dict:
    manager.recorder.stop(camera_id)
    return {"ok": True}


@app.get("/api/cameras/{camera_id}/recordings")
def recordings(camera_id: str, limit: int = 200) -> list[dict]:
    return _recording_rows(camera_id, limit=max(1, min(limit, 1000)))


@app.get("/api/cameras/{camera_id}/recordings/events")
def recording_events(camera_id: str, limit: int = 1000) -> list[dict]:
    rows = _recording_rows(camera_id, limit=1000)
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


@app.get("/api/events/{event_id}/clip.mp4")
def event_clip(event_id: int, before: float | None = None, after: float | None = None) -> FileResponse:
    event = manager.events.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    enriched = _event_row(event)
    before_seconds, after_seconds = _event_clip_window(before, after)
    clip_path = _event_clip_path(enriched, before=before_seconds, after=after_seconds)
    if not clip_path.exists() or clip_path.stat().st_size == 0:
        _build_event_clip(enriched, before=before_seconds, after=after_seconds, output_path=clip_path)
    return FileResponse(
        clip_path,
        media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/cameras/{camera_id}/recordings/stream.mp4")
def recording_stream(camera_id: str, offset: float = 0.0, duration: float | None = None) -> StreamingResponse:
    rows = _recording_rows(camera_id, limit=1000)
    if not rows:
        raise HTTPException(status_code=404, detail="no recordings found")

    total_duration = sum(float(row["duration_seconds"]) for row in rows)
    clamped_offset = max(0.0, min(max(0.0, total_duration - 0.01), offset))
    start_index = 0
    local_offset = clamped_offset
    for index, row in enumerate(rows):
        duration = float(row["duration_seconds"])
        if local_offset < duration or index == len(rows) - 1:
            start_index = index
            break
        local_offset -= duration

    concat_path = _write_concat_file(rows[start_index:])
    bounded_duration = None
    if duration is not None and duration > 0:
        bounded_duration = min(float(duration), 3600.0)
    command = [
        config.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-ss",
        f"{local_offset:.3f}",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
    ]
    if bounded_duration is not None:
        command.extend(["-t", f"{bounded_duration:.3f}"])
    command.extend([
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        "pipe:1",
    ])
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    def chunks():
        try:
            if process.stdout is None:
                return
            while True:
                chunk = process.stdout.read(256 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            concat_path.unlink(missing_ok=True)

    return StreamingResponse(
        chunks(),
        media_type="video/mp4",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/cameras/{camera_id}/recordings/clip.m3u8")
def recording_clip_hls_playlist(camera_id: str, offset: float = 0.0, duration: float = 20.0) -> Response:
    bounded_duration = max(0.1, min(float(duration or 20.0), 120.0))
    target_duration = max(1, math.ceil(bounded_duration))
    params = f"offset={max(0.0, offset):.3f}&duration={bounded_duration:.3f}"
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{target_duration}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        f"#EXTINF:{bounded_duration:.3f},",
        f"clip.ts?{params}",
        "#EXT-X-ENDLIST",
    ]
    return Response(
        "\n".join(lines) + "\n",
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/cameras/{camera_id}/recordings/clip.ts")
def recording_clip_hls_segment(camera_id: str, offset: float = 0.0, duration: float = 20.0) -> StreamingResponse:
    rows = _recording_rows(camera_id, limit=1000)
    if not rows:
        raise HTTPException(status_code=404, detail="no recordings found")

    total_duration = sum(float(row["duration_seconds"]) for row in rows)
    clamped_offset = max(0.0, min(max(0.0, total_duration - 0.01), offset))
    start_index = 0
    local_offset = clamped_offset
    for index, row in enumerate(rows):
        row_duration = float(row["duration_seconds"])
        if local_offset < row_duration or index == len(rows) - 1:
            start_index = index
            break
        local_offset -= row_duration

    concat_path = _write_concat_file(rows[start_index:])
    bounded_duration = max(0.1, min(float(duration or 20.0), 120.0))
    command = [
        config.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-ss",
        f"{local_offset:.3f}",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-t",
        f"{bounded_duration:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        "-bsf:v",
        "h264_mp4toannexb",
        "-f",
        "mpegts",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    def chunks():
        try:
            if process.stdout is None:
                return
            while True:
                chunk = process.stdout.read(256 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            concat_path.unlink(missing_ok=True)

    return StreamingResponse(
        chunks(),
        media_type="video/mp2t",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/cameras/{camera_id}/recordings/hls/index.m3u8")
def recording_hls_playlist(camera_id: str) -> Response:
    rows = _recording_rows(camera_id, limit=1000)
    if not rows:
        raise HTTPException(status_code=404, detail="no recordings found")

    target_duration = max(1, math.ceil(max(float(row["duration_seconds"]) for row in rows)))
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{target_duration}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
    ]
    for index, row in enumerate(rows):
        duration = float(row["duration_seconds"])
        lines.extend(
            [
                f"#EXTINF:{duration:.3f},",
                f"{index}.ts",
            ]
        )
    lines.append("#EXT-X-ENDLIST")
    return Response(
        "\n".join(lines) + "\n",
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/cameras/{camera_id}/recordings/hls/{segment_index}.ts")
def recording_hls_segment(camera_id: str, segment_index: int) -> StreamingResponse:
    rows = _recording_rows(camera_id, limit=1000)
    if segment_index < 0 or segment_index >= len(rows):
        raise HTTPException(status_code=404, detail="recording segment not found")

    path = Path(rows[segment_index]["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="recording file not found")

    command = [
        config.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        "-bsf:v",
        "h264_mp4toannexb",
        "-f",
        "mpegts",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    def chunks():
        try:
            if process.stdout is None:
                return
            while True:
                chunk = process.stdout.read(256 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    return StreamingResponse(
        chunks(),
        media_type="video/mp2t",
        headers={"Cache-Control": "no-store"},
    )


def _recording_rows(camera_id: str, limit: int) -> list[dict]:
    return manager.recorder.recording_rows(camera_id, limit=limit)


def _event_row(row: dict) -> dict:
    event = dict(row)
    try:
        objects = json.loads(event.pop("objects_json", "[]") or "[]")
    except json.JSONDecodeError:
        objects = []
    detected_objects = [
        item for item in objects
        if item.get("label") and float(item.get("confidence") or 0) > 0
    ]
    event["objects"] = objects
    event["has_objects"] = bool(detected_objects)
    event["labels"] = sorted({str(item["label"]) for item in detected_objects})
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
    safe_before = max(0.0, min(float(configured_before or 0.0), 30.0))
    safe_after = max(0.0, min(float(configured_after or 0.0), 30.0))
    return safe_before, safe_after


def _event_clip_path(event: dict, before: float, after: float) -> Path:
    event_id = int(event.get("id") or 0)
    camera_id = str(event.get("camera_id") or "camera").replace("/", "_").replace("\\", "_")
    safe_before = int(max(0.0, min(float(before or 5.0), 30.0)) * 1000)
    safe_after = int(max(0.0, min(float(after or 5.0), 30.0)) * 1000)
    clip_dir = manager.storage_dir / "event_clips" / camera_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    return clip_dir / f"{event_id}-{safe_before}-{safe_after}.mp4"


def _build_event_clip(event: dict, before: float, after: float, output_path: Path) -> None:
    camera_id = str(event.get("camera_id") or "")
    if not camera_id:
        raise HTTPException(status_code=400, detail="event is missing camera")

    event_epoch = _event_epoch(event)
    window_before = max(0.0, min(float(before or 5.0), 30.0))
    window_after = max(0.0, min(float(after or 5.0), 30.0))
    window_start = event_epoch - window_before
    window_end = event_epoch + window_after

    rows = [
        row for row in _recording_rows(camera_id, limit=2000)
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
    command = [
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
        "-an",
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        "-y",
        str(tmp_path),
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=60)
        if result.returncode != 0:
            detail = (result.stderr or "event clip generation failed").strip()[-500:]
            raise HTTPException(status_code=500, detail=detail or "event clip generation failed")
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="event clip generation produced no video")
        tmp_path.replace(output_path)
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

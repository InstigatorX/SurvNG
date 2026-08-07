"""System status, capability, catalog, and server-event HTTP boundaries."""

from __future__ import annotations

import asyncio
import json
import platform
import queue
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .motion_pipeline import motion_pipeline_catalog


@dataclass(frozen=True, slots=True)
class SystemRouteDependencies:
    get_manager: Callable[[], Any]
    get_config: Callable[[], Any]
    system_telemetry: Any
    ffprobe_path: Callable[[], str]
    ffplay_path: Callable[[], str]
    ffmpeg_qsv_info: Callable[[], dict]
    ffmpeg_vaapi_info: Callable[[], dict]
    hardware_acceleration_mode: Callable[[], str]
    event_clip_window: Callable[[float | None, float | None], tuple[float, float]]
    recording_cache_status: Callable[[], dict]


@dataclass(frozen=True, slots=True)
class SystemRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]


def _sse_message(event_type: str, payload: object, event_id: str | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event_type}")
    lines.append(f"data: {json.dumps(payload, separators=(',', ':'), default=str)}")
    return "\n".join(lines) + "\n\n"


def create_system_router(deps: SystemRouteDependencies) -> SystemRouteBundle:
    router = APIRouter()

    @router.get("/api/cameras")
    def cameras() -> list[dict]:
        return deps.get_manager().statuses()

    @router.get("/api/events/stream")
    async def application_event_stream(request: Request) -> StreamingResponse:
        active_manager = deps.get_manager()
        subscriber = active_manager.state_events.subscribe()

        async def generate():
            try:
                yield "retry: 3000\n\n"
                last_event_id = request.headers.get("last-event-id", "")
                replay = active_manager.state_events.events_after(last_event_id)
                replayed_ids: set[str] = set()
                snapshot_sequence: int | None = None
                if replay is None:
                    connection_cursor = active_manager.state_events.cursor
                    snapshot_sequence = active_manager.state_events.sequence(connection_cursor)
                    yield _sse_message("cameras_state", await asyncio.to_thread(active_manager.statuses))
                    yield _sse_message(
                        "system_state",
                        await asyncio.to_thread(deps.system_telemetry.system_status, deps.get_manager()),
                    )
                else:
                    for event in replay:
                        replayed_ids.add(event.id)
                        yield _sse_message(event.type, event.data, event.id)
                    connection_cursor = replay[-1].id if replay else last_event_id
                yield _sse_message(
                    "connected",
                    {"instance": active_manager.state_events.instance_id},
                    connection_cursor,
                )
                next_system_update = time.monotonic() + 5.0
                stream_deadline = time.monotonic() + 6.0
                while True:
                    if await request.is_disconnected() or time.monotonic() >= stream_deadline:
                        return
                    try:
                        event = subscriber.get_nowait()
                    except queue.Empty:
                        if time.monotonic() >= next_system_update:
                            yield _sse_message(
                                "system_state",
                                await asyncio.to_thread(
                                    deps.system_telemetry.system_status,
                                    deps.get_manager(),
                                ),
                            )
                            next_system_update = time.monotonic() + 5.0
                        await asyncio.sleep(0.2)
                        continue
                    if event is None:
                        return
                    if event.id in replayed_ids:
                        continue
                    event_sequence = active_manager.state_events.sequence(event.id)
                    if (
                        snapshot_sequence is not None
                        and event_sequence is not None
                        and event_sequence <= snapshot_sequence
                    ):
                        continue
                    yield _sse_message(event.type, event.data, event.id)
            finally:
                active_manager.state_events.unsubscribe(subscriber)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/api/motion/pipeline/catalog")
    def get_motion_pipeline_catalog() -> dict:
        return motion_pipeline_catalog(deps.get_manager().motion_pipeline_registry)

    @router.get("/api/accelerator")
    def accelerator() -> dict:
        active_manager = deps.get_manager()
        active_config = deps.get_config()
        system = platform.system()
        machine = platform.machine()
        is_macos = system == "Darwin"
        is_apple_silicon = is_macos and machine in {"arm64", "aarch64"}
        openvino_probe = active_manager.detector.probe_devices()
        openvino_devices = list(openvino_probe.get("devices") or [])
        openvino_error = str(openvino_probe.get("error") or "")
        coreml_available = False
        coreml_error = ""
        if is_macos:
            try:
                import coremltools  # noqa: F401
                coreml_available = True
            except Exception as exc:
                coreml_error = str(exc) or "Core ML probe failed"
        recommended_backend = "coreml" if is_apple_silicon and coreml_available else "openvino"
        vaapi_info = deps.ffmpeg_vaapi_info()
        qsv_info = deps.ffmpeg_qsv_info()
        return {
            "system": system,
            "machine": machine,
            "is_macos": is_macos,
            "is_apple_silicon": is_apple_silicon,
            "has_nvidia": shutil.which("nvidia-smi") is not None,
            "ffmpeg_path": active_config.ffmpeg_path,
            "ffprobe_path": deps.ffprobe_path(),
            "ffplay_path": deps.ffplay_path(),
            "openvino_devices": openvino_devices,
            "openvino_error": openvino_error,
            "coreml_available": coreml_available,
            "coreml_error": coreml_error,
            "recommended_openvino_device": "GPU" if "GPU" in openvino_devices else "CPU",
            "recommended_detector_backend": recommended_backend,
            "ffmpeg_hardware_acceleration": {
                "configured": deps.hardware_acceleration_mode(),
                "ffmpeg_path": active_config.ffmpeg_path,
                "ffprobe_path": deps.ffprobe_path(),
                "ffplay_path": deps.ffplay_path(),
                "vaapi": vaapi_info,
                "qsv": qsv_info,
            },
        }

    @router.get("/api/detector/status")
    def detector_status() -> dict:
        return deps.get_manager().detector_status()

    @router.get("/api/object-tracking/catalog")
    def object_tracking_catalog() -> dict:
        return {
            "active": "survng_hybrid",
            "implementations": [{
                "id": "survng_hybrid",
                "name": "SurvNG Hybrid",
                "available": True,
                "description": "Lightweight geometry tracking with SurvNG appearance recovery.",
            }],
        }

    @router.get("/api/detector/models")
    def detector_models() -> dict:
        active_manager = deps.get_manager()
        active_config = deps.get_config()
        models: list[dict] = []
        search_roots = sorted({
            Path("models/openvino_model"),
            *Path("models").glob("*_openvino_model"),
            Path("openvino_model"),
            *Path(".").glob("*_openvino_model"),
        })
        for root in search_roots:
            if not root.exists():
                continue
            for xml_path in sorted(root.rglob("*.xml")):
                bin_path = xml_path.with_suffix(".bin")
                metadata_path = xml_path.parent / "metadata.yaml"
                item = {
                    "path": str(xml_path), "name": xml_path.stem,
                    "bin_path": str(bin_path), "bin_present": bin_path.exists(),
                    "metadata_path": str(metadata_path) if metadata_path.exists() else "",
                    "task": "", "classes": [], "input_shape": [],
                    "output_shapes": [], "valid": False, "error": "",
                }
                if metadata_path.exists():
                    try:
                        import yaml
                        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
                        names = metadata.get("names") or {}
                        if isinstance(names, dict):
                            item["classes"] = [
                                str(value)
                                for _, value in sorted(names.items(), key=lambda entry: int(entry[0]))
                            ]
                        elif isinstance(names, list):
                            item["classes"] = [str(value) for value in names]
                        item["task"] = str(metadata.get("task") or "")
                    except Exception as exc:
                        item["error"] = f"Metadata: {exc}"
                inspection = active_manager.detector.inspect_model(str(xml_path))
                item["input_shape"] = list(inspection.get("input_shape") or [])
                item["output_shapes"] = list(inspection.get("output_shapes") or [])
                if inspection.get("error"):
                    item["error"] = str(inspection["error"])
                else:
                    item["valid"] = bin_path.exists()
                models.append(item)
        return {"models": models, "active_path": active_config.detector.resolved_model_path()}

    @router.get("/api/event-clip/settings")
    def event_clip_settings() -> dict:
        before, after = deps.event_clip_window(None, None)
        return {"before_seconds": before, "after_seconds": after}

    @router.get("/api/recordings/cache/status")
    def recording_cache_status() -> dict:
        return deps.recording_cache_status()

    handlers = {
        name: value
        for name, value in locals().items()
        if callable(value) and name in {
            "cameras", "application_event_stream", "get_motion_pipeline_catalog",
            "accelerator", "detector_status", "object_tracking_catalog",
            "detector_models", "event_clip_settings", "recording_cache_status",
        }
    }
    return SystemRouteBundle(router=router, handlers=handlers)

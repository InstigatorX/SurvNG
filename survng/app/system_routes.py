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

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .motion_pipeline import motion_pipeline_catalog
from .detector_labels import openvino_package_classes


SSE_HEARTBEAT_SECONDS = 15.0
SSE_DISCONNECT_POLL_SECONDS = 1.0
DETECTOR_MODEL_SEARCH_BASES = (Path("models"), Path("/models"))


def detector_model_search_roots(
    active_model_path: str = "",
    bases: tuple[Path, ...] | None = None,
) -> list[Path]:
    """Return OpenVINO package directories to scan for Admin model metadata."""
    search_roots: set[Path] = set()
    for base in bases if bases is not None else DETECTOR_MODEL_SEARCH_BASES:
        if not base.exists():
            continue
        search_roots.add(base / "openvino_model")
        search_roots.update(base.glob("*_openvino_model"))
    search_roots.add(Path("openvino_model"))
    search_roots.update(Path(".").glob("*_openvino_model"))
    active_text = str(active_model_path or "").strip()
    if active_text:
        active_path = Path(active_text)
        if active_path.suffix.lower() == ".xml":
            search_roots.add(active_path.parent)
    return sorted(search_roots)


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
    model_evaluation: Any


class ModelEvaluationRequest(BaseModel):
    baseline_path: str = Field(min_length=1, max_length=4096)
    candidate_path: str = Field(min_length=1, max_length=4096)
    sample_count: int = Field(default=200, ge=10, le=500)
    confidence: float = Field(default=0.25, ge=0.01, le=0.99)


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

    @router.get("/api/integrations/home-assistant")
    def home_assistant_metadata() -> dict[str, Any]:
        """Expose credential-safe, read-scoped entity metadata for HA."""
        config = deps.get_config()
        return {
            "schema_version": 1,
            "base_path": config.base_path,
            "mqtt": {
                "enabled": config.mqtt.enabled,
                "topic_prefix": config.mqtt.topic_prefix,
                "discovery_enabled": config.mqtt.discovery_enabled,
                "incident_events_enabled": config.mqtt.incident_events_enabled,
            },
            "cameras": [{
                "id": camera.id,
                "name": camera.name,
                "zones": [
                    {
                        "name": zone.name,
                        "object_classes": list(zone.object_classes),
                    }
                    for zone in camera.zones
                    if zone.enabled
                ],
            } for camera in config.cameras],
        }

    @router.get("/api/events/stream")
    async def application_event_stream(request: Request) -> StreamingResponse:
        async def generate():
            active_manager = deps.get_manager()
            subscriber = active_manager.state_events.subscribe()
            try:
                yield "retry: 3000\n\n"
                query_params = getattr(request, "query_params", {})
                last_event_id = (
                    request.headers.get("last-event-id", "")
                    or query_params.get("last_event_id", "")
                )
                replay = active_manager.state_events.events_after(last_event_id)
                replayed_ids: set[str] = set()
                snapshot_sequence: int | None = None
                if replay is None:
                    connection_cursor = active_manager.state_events.cursor
                    snapshot_sequence = active_manager.state_events.sequence(connection_cursor)
                    yield _sse_message("cameras_state", await asyncio.to_thread(active_manager.statuses))
                    yield _sse_message(
                        "system_state",
                        await asyncio.to_thread(deps.system_telemetry.system_status, active_manager),
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
                next_heartbeat = time.monotonic() + SSE_HEARTBEAT_SECONDS
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        event = subscriber.get_nowait()
                    except queue.Empty:
                        now = time.monotonic()
                        if now >= next_heartbeat:
                            yield ": heartbeat\n\n"
                            next_heartbeat = time.monotonic() + SSE_HEARTBEAT_SECONDS
                            continue
                        await asyncio.sleep(min(
                            SSE_DISCONNECT_POLL_SECONDS,
                            max(0.0, next_heartbeat - now),
                        ))
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
        seen_paths: set[str] = set()
        labels_override = Path(active_config.detector.labels_path) if active_config.detector.labels_path else None
        active_path = active_config.detector.resolved_model_path()
        for root in detector_model_search_roots(active_path):
            if not root.exists():
                continue
            for xml_path in sorted(root.rglob("*.xml")):
                path_text = str(xml_path)
                if path_text in seen_paths:
                    continue
                seen_paths.add(path_text)
                bin_path = xml_path.with_suffix(".bin")
                metadata_path = xml_path.parent / "metadata.yaml"
                classes, task, class_error = openvino_package_classes(xml_path)
                if (
                    not classes
                    and labels_override
                    and path_text == active_path
                    and labels_override.exists()
                ):
                    classes = [
                        line.strip()
                        for line in labels_override.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                item = {
                    "path": path_text, "name": xml_path.stem,
                    "bin_path": str(bin_path), "bin_present": bin_path.exists(),
                    "metadata_path": str(metadata_path) if metadata_path.exists() else "",
                    "task": task, "classes": classes, "input_shape": [],
                    "output_shapes": [], "valid": False, "error": class_error,
                }
                inspection = active_manager.detector.inspect_model(path_text)
                item["input_shape"] = list(inspection.get("input_shape") or [])
                item["output_shapes"] = list(inspection.get("output_shapes") or [])
                if inspection.get("error"):
                    item["error"] = str(inspection["error"])
                else:
                    item["valid"] = bin_path.exists()
                models.append(item)
        return {"models": models, "active_path": active_path}

    @router.get("/api/detector/model-evaluation")
    def model_evaluation_status() -> dict:
        return deps.model_evaluation.status()

    @router.post("/api/detector/model-evaluation", status_code=202)
    def start_model_evaluation(request: ModelEvaluationRequest) -> dict:
        try:
            return deps.model_evaluation.start(
                baseline_path=request.baseline_path,
                candidate_path=request.candidate_path,
                sample_count=request.sample_count,
                confidence=request.confidence,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.delete("/api/detector/model-evaluation", status_code=202)
    def cancel_model_evaluation() -> dict:
        try:
            return deps.model_evaluation.cancel()
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

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
            "cameras", "home_assistant_metadata", "application_event_stream", "get_motion_pipeline_catalog",
            "accelerator", "detector_status", "object_tracking_catalog",
            "detector_models", "event_clip_settings", "recording_cache_status",
        }
    }
    return SystemRouteBundle(router=router, handlers=handlers)

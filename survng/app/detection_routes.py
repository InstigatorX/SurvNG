"""Manual detection HTTP boundary."""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request

from .config import AppConfig, camera_by_id
from .detector import detection_failure, objects_to_json
from .domain_events import ObjectDetected
from .event_store import EventSnapshotChangedError
from .incident_presenter import _event_row
from .incident_utils import event_snapshot_path
from .manager import AppManager
from .manager_access import ManagerAccessCoordinator, guard_manager_generation
from .inference_runtime.types import InferenceWorkload
from .zones import apply_detection_zones, detection_threshold

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DetectionRouteDependencies:
    get_manager: Callable[[], AppManager]
    get_config: Callable[[], AppConfig]
    manager_lock: threading.RLock
    manager_access: ManagerAccessCoordinator | None = None


@dataclass(frozen=True, slots=True)
class DetectionRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]


def create_detection_router(deps: DetectionRouteDependencies) -> DetectionRouteBundle:
    router = APIRouter()

    def generation() -> tuple[AppManager, AppConfig]:
        active_manager = deps.get_manager()
        active_config = getattr(active_manager, "config", None) or deps.get_config()
        return active_manager, active_config.model_copy(deep=True)

    @router.post("/api/events/{event_id}/detect")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def detect_event_snapshot(event_id: int, confidence: float = 0.35) -> dict[str, Any]:
        active_manager, active_config = generation()
        event = active_manager.events.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        try:
            snapshot_path = event_snapshot_path(
                active_manager.storage_dir,
                event,
                getattr(active_manager, "media_storage", None),
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="snapshot not found") from None
        except PermissionError:
            raise HTTPException(
                status_code=403, detail="snapshot outside storage directory"
            ) from None
        if not math.isfinite(confidence):
            raise HTTPException(status_code=422, detail="confidence must be finite")
        safe_confidence = max(0.01, min(0.99, float(confidence)))
        frame = cv2.imread(str(snapshot_path))
        if frame is None:
            raise HTTPException(status_code=422, detail="failed to read snapshot")
        started = time.perf_counter()
        camera = camera_by_id(active_config, str(event.get("camera_id") or ""))
        effective_confidence = (
            detection_threshold(camera, safe_confidence) if camera else safe_confidence
        )
        detect_interactive = getattr(
            active_manager.detector,
            "detect_interactive",
            active_manager.detector.detect,
        )
        objects = detect_interactive(
            frame, confidence_threshold=effective_confidence
        )
        detector_error = detection_failure(objects)
        if detector_error:
            raise HTTPException(status_code=503, detail=detector_error)
        if camera:
            apply_detection_zones(
                camera,
                objects,
                int(frame.shape[1]),
                int(frame.shape[0]),
                safe_confidence,
                bool(active_config.detector.require_incident_zone),
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        for detected_object in objects:
            detected_object["frame_source"] = (
                detected_object.get("frame_source") or "manual_snapshot"
            )
            detected_object["detection_source"] = "manual_openvino"
            detected_object["manual_confidence_threshold"] = safe_confidence
            detected_object["detection_frame_width"] = int(frame.shape[1])
            detected_object["detection_frame_height"] = int(frame.shape[0])
        try:
            persisted = active_manager.events.replace_detected_objects(
                event_id,
                objects_to_json(objects),
                expected_snapshot_path=str(event.get("snapshot_path") or ""),
                **({"expected_revision": int(event["evidence_revision"])}
                   if "evidence_revision" in event else {}),
            )
        except EventSnapshotChangedError:
            raise HTTPException(
                status_code=409,
                detail="event snapshot changed during detection; refresh and try again",
            ) from None
        if persisted is None:
            raise HTTPException(status_code=404, detail="event not found")
        detected = [
            item
            for item in objects
            if item.get("label")
            and item.get("box")
            and item.get("incident_eligible") is not False
        ]
        if detected:
            active_manager.publish_event(
                "object",
                ObjectDetected(
                    event_id=event_id,
                    camera_id=str(event.get("camera_id") or ""),
                    timestamp=str(
                        event.get("created_at")
                        or datetime.now(timezone.utc).isoformat()
                    ),
                    snapshot_path=str(snapshot_path),
                    recording_path=str(event.get("recording_path") or ""),
                    source="manual_openvino",
                    objects=tuple(detected),
                ).to_payload(),
            )
        else:
            active_manager.publish_event(
                "incident_update",
                {
                    "event_id": event_id,
                    "camera_id": str(persisted.get("camera_id") or ""),
                    "updated": True,
                },
            )
        detector_status = active_manager.detector_status()
        return {
            "event_id": event_id,
            "camera_id": event.get("camera_id"),
            "snapshot_path": "available",
            "snapshot_width": int(frame.shape[1]),
            "snapshot_height": int(frame.shape[0]),
            "confidence": safe_confidence,
            "elapsed_ms": elapsed_ms,
            "objects": objects,
            "object_count": len(detected),
            "labels": sorted({str(item.get("label")) for item in detected}),
            "event": _event_row(persisted),
            "persisted": True,
            "detector": {
                key: detector_status.get(key)
                for key in (
                    "enabled",
                    "loaded_backend",
                    "loaded_device",
                    "configured_device",
                    "input_shape",
                    "output_format",
                )
            },
        }


    @router.post("/api/detector/frame")
    async def detect_debug_frame(
        request: Request,
        confidence: float = 0.35,
        depth: bool = False,
        heatmap: bool = False,
    ) -> dict[str, Any]:
        maximum_bytes = 2 * 1024 * 1024
        try:
            content_length = int(request.headers.get("content-length") or 0)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="invalid content-length header"
            ) from None
        if content_length < 0:
            raise HTTPException(
                status_code=400, detail="invalid content-length header"
            )
        if content_length > maximum_bytes:
            raise HTTPException(status_code=413, detail="debug frame is too large")
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > maximum_bytes:
                raise HTTPException(
                    status_code=413, detail="debug frame is too large"
                )
            payload.extend(chunk)
        if not payload:
            raise HTTPException(status_code=422, detail="invalid debug frame")
        # Uploading does not use a manager. Decode and inference share a worker-
        # owned lease that outlives HTTP cancellation and never blocks the loop.
        return await asyncio.to_thread(
            analyze_debug_frame, bytes(payload), confidence, depth, heatmap,
        )

    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def analyze_debug_frame(
        payload: bytes, confidence: float, depth: bool, heatmap: bool,
    ) -> dict[str, Any]:
        frame = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            raise HTTPException(status_code=422, detail="failed to decode debug frame")
        if not math.isfinite(confidence):
            raise HTTPException(status_code=422, detail="confidence must be finite")
        safe_confidence = max(0.01, min(0.99, float(confidence)))
        active_manager = deps.get_manager()
        active_detector = active_manager.detector
        started = time.perf_counter()
        objects = active_detector.detect(frame, confidence_threshold=safe_confidence)
        detector_error = detection_failure(objects)
        if detector_error:
            raise HTTPException(status_code=503, detail=detector_error)
        detect_ms = round((time.perf_counter() - started) * 1000, 1)
        filtered_objects = [
            item for item in objects if item.get("label") and item.get("box")
        ]
        depth_ms: float | None = None
        depth_error = ""
        depth_status: dict[str, Any] | None = None
        heatmap_png_b64 = ""
        heatmap_range_m: dict[str, float] | None = None
        if depth or heatmap:
            depth_status_fn = getattr(active_detector, "depth_status", None)
            depth_status = (
                depth_status_fn() if callable(depth_status_fn) else None
            )
            estimate_depth = getattr(
                active_detector, "estimate_depth_for_objects", None
            )
            if not callable(estimate_depth):
                depth_error = "Depth estimation is not available."
            elif not depth_status or not depth_status.get("enabled"):
                depth_error = "Depth estimation is not configured."
            elif not depth_status.get("ready"):
                depth_error = "Depth model is not ready."
            else:
                depth_started = time.perf_counter()
                depth_kwargs: dict[str, Any] = {
                    "include_heatmap": bool(heatmap),
                    "workload": InferenceWorkload.INTERACTIVE,
                }
                try:
                    enriched, depth_metadata = estimate_depth(
                        frame,
                        filtered_objects,
                        **depth_kwargs,
                    )
                    filtered_objects = [
                        item
                        for item in enriched
                        if item.get("label") and item.get("box")
                    ]
                    depth_ms = round((time.perf_counter() - depth_started) * 1000, 1)
                    if depth_metadata.get("status") == "depth_deferred":
                        depth_error = "Depth estimation deferred for higher-priority inference."
                    elif depth_metadata.get("status") == "depth_error":
                        depth_error = "Depth estimation failed."
                    heatmap_bytes = depth_metadata.get("heatmap_png")
                    if isinstance(heatmap_bytes, (bytes, bytearray)) and heatmap_bytes:
                        heatmap_png_b64 = base64.b64encode(heatmap_bytes).decode("ascii")
                    heatmap_range = depth_metadata.get("heatmap_range_m")
                    if isinstance(heatmap_range, dict):
                        heatmap_range_m = {
                            key: float(heatmap_range[key])
                            for key in ("min_m", "max_m")
                            if key in heatmap_range
                        }
                except Exception as exc:
                    LOGGER.warning(
                        "Interactive depth replay failed (%s)",
                        type(exc).__name__,
                    )
                    depth_error = "Depth estimation failed."
        response: dict[str, Any] = {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "confidence": safe_confidence,
            "detect_ms": detect_ms,
            "elapsed_ms": detect_ms,
            "objects": filtered_objects,
        }
        if depth or heatmap:
            response["depth_requested"] = bool(depth)
            response["heatmap_requested"] = bool(heatmap)
            response["depth_ms"] = depth_ms
            response["depth_error"] = depth_error
            if depth_status:
                response["depth"] = {
                    key: depth_status.get(key)
                    for key in (
                        "enabled",
                        "ready",
                        "min_distance_m",
                        "max_distance_m",
                    )
                }
            if heatmap_png_b64:
                response["heatmap_png_b64"] = heatmap_png_b64
            if heatmap_range_m:
                response["heatmap_range_m"] = heatmap_range_m
            response["elapsed_ms"] = round(
                detect_ms + (depth_ms or 0.0),
                1,
            )
        return response

    handlers: dict[str, Callable[..., Any]] = {
        name: value
        for name, value in locals().copy().items()
        if callable(value) and name not in {"generation"}
    }
    return DetectionRouteBundle(router=router, handlers=handlers)

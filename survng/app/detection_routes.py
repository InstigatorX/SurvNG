"""Manual detection and tracking-comparison HTTP boundary."""

from __future__ import annotations

import asyncio
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
from pydantic import BaseModel, Field

from .config import AppConfig, camera_by_id
from .detector import detection_failure, objects_to_json
from .domain_events import ObjectDetected
from .incident_presenter import _event_row
from .incident_utils import event_epoch, event_snapshot_path
from .manager import AppManager
from .manager_access import ManagerAccessCoordinator, guard_manager_generation
from .tracking_comparison import TRACKING_COMPARISON_IMPLEMENTATIONS
from .zones import apply_detection_zones, detection_threshold

LOGGER = logging.getLogger(__name__)
TRACKING_COMPARISON_MIN_DURATION_SECONDS = 3.0
TRACKING_COMPARISON_DEFAULT_DURATION_SECONDS = 30.0
TRACKING_COMPARISON_MAX_DURATION_SECONDS = 30.0


class TrackingComparisonVerdictRequest(BaseModel):
    verdict: str = Field(
        pattern=r"^(survng_hybrid|ultralytics_botsort|ultralytics_deepocsort|ultralytics_fasttrack|inconclusive)$"
    )


@dataclass(frozen=True, slots=True)
class DetectionRouteDependencies:
    get_manager: Callable[[], AppManager]
    get_config: Callable[[], AppConfig]
    manager_lock: threading.RLock
    get_comparison_limiter: Callable[[], Any]
    ensure_event_clip: Callable[..., Any]
    dependency_status: Callable[[], dict[str, Any]]
    comparison_runner: Callable[..., Any]
    sample_video_frames: Callable[..., Any]
    manager_access: ManagerAccessCoordinator | None = None


@dataclass(frozen=True, slots=True)
class DetectionRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]


def _tracking_comparison_evidence(result: dict[str, Any]) -> dict[str, Any]:
    engines: dict[str, dict[str, Any]] = {}
    for implementation in TRACKING_COMPARISON_IMPLEMENTATIONS:
        engine = result.get("engines", {}).get(implementation, {})
        engines[implementation] = {
            key: engine.get(key)
            for key in (
                "track_count",
                "observations",
                "reid_recoveries",
                "fragmentation_proxy",
                "initialization_ms",
                "processing_ms",
                "average_ms_per_frame",
                "labels",
            )
        }
    return {
        key: result.get(key)
        for key in (
            "sample_fps",
            "frames_processed",
            "duration_seconds",
            "frame_width",
            "frame_height",
            "average_frame_decode_ms",
            "average_detection_ms_per_frame",
            "average_appearance_ms_per_frame",
            "appearance_failures",
            "clip_preparation_ms",
            "elapsed_ms",
        )
    } | {"engines": engines}


def _tracking_comparison_duration(duration_seconds: float | None) -> float:
    requested = (
        float(duration_seconds)
        if duration_seconds is not None
        else TRACKING_COMPARISON_DEFAULT_DURATION_SECONDS
    )
    return max(
        TRACKING_COMPARISON_MIN_DURATION_SECONDS,
        min(TRACKING_COMPARISON_MAX_DURATION_SECONDS, requested),
    )


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
            snapshot_path = event_snapshot_path(active_manager.storage_dir, event)
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
        objects = active_manager.detector.detect(
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
        persisted = active_manager.events.replace_detected_objects(
            event_id, objects_to_json(objects)
        )
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

    @router.get("/api/tracking-comparisons")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def tracking_comparison_history(
        camera_id: str = "", limit: int = 25
    ) -> dict[str, Any]:
        active_manager = deps.get_manager()
        normalized = str(camera_id or "").strip()
        return {
            "items": active_manager.events.tracking_comparison_history(
                camera_id=normalized, limit=limit
            ),
            "summary": active_manager.events.tracking_comparison_summary(
                camera_id=normalized
            ),
        }

    @router.put("/api/tracking-comparisons/{comparison_id}/verdict")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def update_tracking_comparison_verdict(
        comparison_id: int, payload: TrackingComparisonVerdictRequest
    ) -> dict[str, Any]:
        active_manager = deps.get_manager()
        comparison = active_manager.events.set_tracking_comparison_verdict(
            comparison_id, payload.verdict
        )
        if comparison is None:
            raise HTTPException(
                status_code=404, detail="tracking comparison not found"
            )
        return {
            "comparison": comparison,
            "summary": active_manager.events.tracking_comparison_summary(
                camera_id=str(comparison.get("camera_id") or "")
            ),
        }

    @router.post("/api/events/{event_id}/tracking-comparison")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def compare_event_tracking(
        event_id: int, duration_seconds: float | None = None
    ) -> dict[str, Any]:
        dependency = deps.dependency_status()
        if not dependency["available"]:
            raise HTTPException(status_code=503, detail=dependency["reason"])
        if duration_seconds is not None and not math.isfinite(duration_seconds):
            raise HTTPException(
                status_code=422, detail="duration_seconds must be finite"
            )
        active_manager, active_config = generation()
        event = active_manager.events.get(event_id)
        duration = _tracking_comparison_duration(duration_seconds)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        camera = camera_by_id(active_config, str(event.get("camera_id") or ""))
        if camera is None:
            raise HTTPException(
                status_code=404, detail="event camera is not configured"
            )
        limiter = deps.get_comparison_limiter()
        if not limiter.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="another tracking comparison is already running",
                headers={"Retry-After": "3"},
            )
        try:
            request_started = time.perf_counter()
            enriched = _event_row(event)
            clip_started = time.perf_counter()
            comparison_input = deps.ensure_event_clip(
                enriched, before=0.0, after=duration, source="main"
            )
            clip_ms = (time.perf_counter() - clip_started) * 1000.0
            runner = deps.comparison_runner(
                config=active_config.detector.tracking,
                detector=active_manager.detector,
                appearance_encoder=active_manager.person_reidentifier,
            )
            detector_dimensions = [
                int(value)
                for value in (
                    getattr(active_manager.detector, "input_shape", None) or []
                )
                if isinstance(value, (int, float)) and int(value) > 0
            ]
            frames = deps.sample_video_frames(
                comparison_input,
                start_epoch=event_epoch(enriched),
                sample_fps=active_config.detector.tracking.sample_fps,
                duration_seconds=duration,
                ffmpeg_path=active_config.ffmpeg_path,
                maximum_width=max([640, *detector_dimensions]),
            )
            result = runner.run(camera, frames)
            response = {
                "event_id": event_id,
                "camera_id": camera.id,
                "created_at": str(enriched.get("created_at") or ""),
                "requested_duration_seconds": duration,
                "clip_preparation_ms": round(clip_ms, 1),
                "elapsed_ms": round(
                    (time.perf_counter() - request_started) * 1000.0, 1
                ),
                **result,
            }
            comparison = active_manager.events.save_tracking_comparison(
                event_id=event_id,
                camera_id=camera.id,
                event_created_at=str(enriched.get("created_at") or ""),
                result=_tracking_comparison_evidence(response),
            )
            response.update(
                comparison_id=comparison["id"],
                verdict=comparison["verdict"],
                comparison=comparison,
                evidence_summary=active_manager.events.tracking_comparison_summary(
                    camera_id=camera.id
                ),
            )
            return response
        except HTTPException:
            raise
        except Exception:
            LOGGER.exception("tracking comparison failed for event %d", event_id)
            raise HTTPException(
                status_code=422, detail="tracking comparison failed"
            ) from None
        finally:
            limiter.release()

    @router.post("/api/detector/frame")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    async def detect_debug_frame(
        request: Request, confidence: float = 0.35
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
        frame = cv2.imdecode(
            np.frombuffer(bytes(payload), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            raise HTTPException(status_code=422, detail="failed to decode debug frame")
        if not math.isfinite(confidence):
            raise HTTPException(status_code=422, detail="confidence must be finite")
        safe_confidence = max(0.01, min(0.99, float(confidence)))
        active_detector = deps.get_manager().detector
        started = time.perf_counter()
        objects = await asyncio.to_thread(
            active_detector.detect, frame, confidence_threshold=safe_confidence
        )
        detector_error = detection_failure(objects)
        if detector_error:
            raise HTTPException(status_code=503, detail=detector_error)
        return {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "confidence": safe_confidence,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "objects": [
                item for item in objects if item.get("label") and item.get("box")
            ],
        }

    handlers: dict[str, Callable[..., Any]] = {
        name: value
        for name, value in locals().copy().items()
        if callable(value) and name not in {"generation"}
    }
    return DetectionRouteBundle(router=router, handlers=handlers)

"""Read-only, model-generated training sample manifest API."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .config import AppConfig
from .incident_utils import event_snapshot_path, snapshot_media_type
from .manager import AppManager
from .manager_access import ManagerAccessCoordinator, manager_generation_lease


MAX_TRAINING_RANGE = timedelta(days=366)
MAX_SCANNED_EVENTS = 5000
CURSOR_VERSION = 1


class TrainingAnnotation(BaseModel):
    annotation_id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    bbox_xywh: list[float] = Field(min_length=4, max_length=4)
    bbox_normalized_xyxy: list[float] = Field(min_length=4, max_length=4)
    bbox_normalized_cxcywh: list[float] = Field(min_length=4, max_length=4)
    zones: list[str] = Field(default_factory=list)
    incident_eligible: bool
    temporal_consensus: bool
    semantic_tier: str
    source: Literal["survng_object_detection"]
    annotation_state: Literal["model_generated"]


class TrainingImage(BaseModel):
    url: str
    media_type: str
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


class TrainingSample(BaseModel):
    sample_id: str
    revision: str
    event_id: int = Field(gt=0)
    camera_id: str
    event_at: str
    captured_at: str
    image: TrainingImage
    annotations: list[TrainingAnnotation]
    annotation_state: Literal["model_generated"]


class TrainingSamplesResponse(BaseModel):
    schema_version: Literal[1]
    coordinate_conventions: dict[str, str]
    range: dict[str, str]
    filters: dict[str, Any]
    samples: list[TrainingSample]
    count: int = Field(ge=0)
    scanned_events: int = Field(ge=0)
    scan_limited: bool
    next_cursor: str


@dataclass(frozen=True, slots=True)
class TrainingRouteDependencies:
    get_config: Callable[[], AppConfig]
    get_manager: Callable[[], AppManager]
    manager_lock: threading.RLock
    manager_access: ManagerAccessCoordinator


def _utc_datetime(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be an ISO 8601 date and time",
        ) from exc
    if parsed.tzinfo is None:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must include a timezone offset",
        )
    return parsed.astimezone(timezone.utc)


def _csv_values(value: str, *, maximum: int, lower: bool = False) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(
        normalized
        for raw in str(value or "").split(",")
        if (normalized := (raw.strip().lower() if lower else raw.strip()))
    ))
    if len(values) > maximum:
        raise HTTPException(status_code=422, detail=f"filter cannot exceed {maximum} values")
    return values


def _encode_cursor(created_at: str, event_id: int) -> str:
    payload = json.dumps(
        {"v": CURSOR_VERSION, "created_at": created_at, "id": int(event_id)},
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(value: str) -> tuple[str, int] | None:
    if not value:
        return None
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        if payload.get("v") != CURSOR_VERSION:
            raise ValueError("unsupported cursor version")
        created_at = _utc_datetime(str(payload["created_at"]), "cursor").isoformat()
        event_id = int(payload["id"])
        if event_id <= 0:
            raise ValueError("invalid cursor event")
        return created_at, event_id
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        binascii.Error,
        json.JSONDecodeError,
    ) as exc:
        raise HTTPException(status_code=422, detail="cursor is invalid") from exc


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_dimension(value: object) -> int | None:
    number = _finite_number(value)
    if number is None or number <= 0:
        return None
    return int(round(number))


def _object_box(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        coordinates = [value.get(key) for key in ("x1", "y1", "x2", "y2")]
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        coordinates = list(value[:4])
    else:
        return None
    parsed = [_finite_number(coordinate) for coordinate in coordinates]
    if any(coordinate is None for coordinate in parsed):
        return None
    x1, y1, x2, y2 = (float(coordinate) for coordinate in parsed)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _sample_annotations(
    event_id: int,
    objects: object,
    *,
    labels: frozenset[str],
    minimum_confidence: float,
    eligibility: Literal["eligible", "ineligible", "all"],
) -> tuple[list[dict[str, Any]], int | None, int | None, float]:
    if not isinstance(objects, list):
        return [], None, None, 0.0
    annotations: list[dict[str, Any]] = []
    frame_width: int | None = None
    frame_height: int | None = None
    sample_offset = 0.0
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        normalized_label = label.lower()
        confidence = _finite_number(item.get("confidence"))
        box = _object_box(item.get("box"))
        eligible = item.get("incident_eligible") is not False
        width = _positive_dimension(item.get("detection_frame_width"))
        height = _positive_dimension(item.get("detection_frame_height"))
        if (
            not label
            or confidence is None
            or confidence < 0.0
            or confidence > 1.0
            or confidence < minimum_confidence
            or box is None
            or width is None
            or height is None
            or (labels and normalized_label not in labels)
            or (eligibility == "eligible" and not eligible)
            or (eligibility == "ineligible" and eligible)
        ):
            continue
        x1, y1, x2, y2 = box
        x1 = min(float(width), max(0.0, x1))
        y1 = min(float(height), max(0.0, y1))
        x2 = min(float(width), max(0.0, x2))
        y2 = min(float(height), max(0.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        if frame_width is None:
            frame_width, frame_height = width, height
        if width != frame_width or height != frame_height:
            # Objects from another coordinate plane cannot safely annotate the
            # representative snapshot and are intentionally omitted.
            continue
        offset = _finite_number(item.get("temporal_sample_offset_seconds"))
        if offset is not None:
            sample_offset = offset
        bbox_xyxy = [round(value, 4) for value in (x1, y1, x2, y2)]
        bbox_xywh = [
            round(x1, 4),
            round(y1, 4),
            round(x2 - x1, 4),
            round(y2 - y1, 4),
        ]
        normalized_xyxy = [
            round(x1 / width, 8),
            round(y1 / height, 8),
            round(x2 / width, 8),
            round(y2 / height, 8),
        ]
        normalized_cxcywh = [
            round(((x1 + x2) / 2.0) / width, 8),
            round(((y1 + y2) / 2.0) / height, 8),
            round((x2 - x1) / width, 8),
            round((y2 - y1) / height, 8),
        ]
        annotations.append({
            "annotation_id": f"event-{event_id}-object-{index}",
            "label": label,
            "confidence": round(confidence, 6),
            "bbox_xyxy": bbox_xyxy,
            "bbox_xywh": bbox_xywh,
            "bbox_normalized_xyxy": normalized_xyxy,
            "bbox_normalized_cxcywh": normalized_cxcywh,
            "zones": [str(zone) for zone in item.get("zones", []) if str(zone)]
            if isinstance(item.get("zones"), list)
            else [],
            "incident_eligible": eligible,
            "temporal_consensus": item.get("temporal_consensus") is True,
            "semantic_tier": str(item.get("semantic_tier") or ""),
            "source": "survng_object_detection",
            "annotation_state": "model_generated",
        })
    return annotations, frame_width, frame_height, sample_offset


def _parse_objects(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        value = json.loads(str(event.get("objects_json") or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def create_training_router(deps: TrainingRouteDependencies) -> APIRouter:
    router = APIRouter()

    @router.get(
        "/api/training/samples",
        response_model=TrainingSamplesResponse,
        summary="List original incident images and model-generated annotations",
    )
    def training_samples(
        start_at: str,
        end_at: str,
        camera_ids: str = "",
        object_labels: str = "",
        eligibility: Literal["eligible", "ineligible", "all"] = "eligible",
        minimum_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
        include_empty: bool = False,
        limit: int = Query(default=100, ge=1, le=500),
        cursor: str = Query(default="", max_length=2048),
    ) -> dict[str, Any]:
        start = _utc_datetime(start_at, "start_at")
        end = _utc_datetime(end_at, "end_at")
        if end <= start:
            raise HTTPException(status_code=422, detail="end_at must be after start_at")
        if end - start > MAX_TRAINING_RANGE:
            raise HTTPException(
                status_code=422,
                detail="training sample range cannot exceed 366 days",
            )
        selected_cameras = _csv_values(camera_ids, maximum=128)
        selected_labels = frozenset(_csv_values(object_labels, maximum=256, lower=True))
        decoded_cursor = _decode_cursor(cursor)
        before_created_at = decoded_cursor[0] if decoded_cursor else None
        before_id = decoded_cursor[1] if decoded_cursor else None
        start_iso = start.isoformat()
        end_iso = end.isoformat()

        with manager_generation_lease(
            deps.manager_access,
            deps.manager_lock,
            deps.get_manager,
        ) as active_manager:
            base_path = deps.get_config().base_path
            samples: list[dict[str, Any]] = []
            scanned = 0
            exhausted = False
            last_row: dict[str, Any] | None = None
            batch_limit = max(100, min(1000, limit * 4))
            while len(samples) < limit and scanned < MAX_SCANNED_EVENTS:
                rows = active_manager.events.page_between(
                    start_iso,
                    end_iso,
                    limit=min(batch_limit, MAX_SCANNED_EVENTS - scanned),
                    before_created_at=before_created_at,
                    before_id=before_id,
                    camera_ids=selected_cameras,
                    require_snapshot=True,
                )
                if not rows:
                    exhausted = True
                    break
                scanned += len(rows)
                processed_rows = 0
                for event in rows:
                    processed_rows += 1
                    last_row = event
                    before_created_at = str(event.get("created_at") or "")
                    before_id = int(event.get("id") or 0)
                    annotations, width, height, sample_offset = _sample_annotations(
                        before_id,
                        _parse_objects(event),
                        labels=selected_labels,
                        minimum_confidence=minimum_confidence,
                        eligibility=eligibility,
                    )
                    if not annotations and not include_empty:
                        continue
                    try:
                        snapshot_path = event_snapshot_path(
                            active_manager.storage_dir,
                            event,
                        )
                    except (FileNotFoundError, PermissionError):
                        continue
                    event_at = _utc_datetime(str(event.get("created_at") or ""), "event created_at")
                    captured_at = event_at + timedelta(seconds=sample_offset)
                    revision = hashlib.sha256(
                        (
                            f"{before_id}\0{event.get('snapshot_path') or ''}\0"
                            f"{event.get('objects_json') or ''}"
                        ).encode("utf-8")
                    ).hexdigest()[:20]
                    samples.append({
                        "sample_id": f"event-{before_id}",
                        "revision": revision,
                        "event_id": before_id,
                        "camera_id": str(event.get("camera_id") or ""),
                        "event_at": event_at.isoformat(),
                        "captured_at": captured_at.isoformat(),
                        "image": {
                            "url": f"{base_path}/api/events/{before_id}/snapshot.jpg",
                            "media_type": snapshot_media_type(snapshot_path),
                            "width": width,
                            "height": height,
                        },
                        "annotations": annotations,
                        "annotation_state": "model_generated",
                    })
                    if len(samples) >= limit:
                        break
                if len(samples) >= limit:
                    if processed_rows == len(rows) and len(rows) < batch_limit:
                        exhausted = True
                    break
                if len(rows) < batch_limit:
                    exhausted = True
                    break

        next_cursor = ""
        if last_row is not None and not exhausted:
            next_cursor = _encode_cursor(
                str(last_row.get("created_at") or ""),
                int(last_row.get("id") or 0),
            )
        return {
            "schema_version": 1,
            "coordinate_conventions": {
                "bbox_xyxy": "pixel coordinates [left, top, right, bottom]",
                "bbox_xywh": "pixel coordinates [left, top, width, height]",
                "bbox_normalized_xyxy": "0..1 [left, top, right, bottom]",
                "bbox_normalized_cxcywh": "YOLO-style 0..1 [center_x, center_y, width, height]",
            },
            "range": {"start_at": start_iso, "end_at": end_iso},
            "filters": {
                "camera_ids": list(selected_cameras),
                "object_labels": sorted(selected_labels),
                "eligibility": eligibility,
                "minimum_confidence": minimum_confidence,
                "include_empty": include_empty,
            },
            "samples": samples,
            "count": len(samples),
            "scanned_events": scanned,
            "scan_limited": scanned >= MAX_SCANNED_EVENTS and not exhausted,
            "next_cursor": next_cursor,
        }

    return router

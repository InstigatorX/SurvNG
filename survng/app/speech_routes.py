"""Speech detection status and transcript queries.

Transcripts are conversation content, so segment routes require the admin
API scope (see ``security.required_api_scope``); status is integration-safe.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .speech.store import MAX_LIST_LIMIT, STATUSES


@dataclass(frozen=True, slots=True)
class SpeechRouteDependencies:
    get_manager: Callable[[], Any]


def _timestamp(value: str | None, name: str) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{name} must be an ISO 8601 timestamp") from None
    if parsed.tzinfo is None:
        raise HTTPException(status_code=400, detail=f"{name} must include a timezone")
    return parsed.timestamp()


def create_speech_router(deps: SpeechRouteDependencies) -> APIRouter:
    router = APIRouter()

    @router.get("/api/speech/status")
    def speech_status() -> dict[str, Any]:
        return deps.get_manager().speech_status()

    @router.get("/api/speech/segments")
    def speech_segments(
        camera_id: list[str] | None = Query(default=None, max_length=64),
        start: str | None = Query(default=None, max_length=64),
        end: str | None = Query(default=None, max_length=64),
        q: str = Query(default="", max_length=200),
        status: list[str] | None = Query(default=None, max_length=len(STATUSES)),
        before_id: int | None = Query(default=None, ge=1),
        limit: int = Query(default=100, ge=1, le=MAX_LIST_LIMIT),
    ) -> dict[str, Any]:
        if status and any(item not in STATUSES for item in status):
            raise HTTPException(status_code=400, detail="unknown speech segment status")
        start_at = _timestamp(start, "start")
        end_at = _timestamp(end, "end")
        if start_at is not None and end_at is not None and end_at < start_at:
            raise HTTPException(status_code=400, detail="end must not precede start")
        segments = deps.get_manager().speech.segments(
            camera_ids=camera_id or None,
            start=start_at,
            end=end_at,
            query=q,
            statuses=status or None,
            before_id=before_id,
            limit=limit,
        )
        return {
            "segments": segments,
            "next_before_id": segments[-1]["id"] if len(segments) == limit else None,
        }

    @router.get("/api/speech/segments/{segment_id}")
    def speech_segment(segment_id: int) -> dict[str, Any]:
        segment = deps.get_manager().speech.segment(segment_id)
        if segment is None:
            raise HTTPException(status_code=404, detail="speech segment not found")
        return segment

    return router

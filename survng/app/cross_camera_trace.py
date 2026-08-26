"""Shared cross-camera incident investigation timeline builder."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .assistant_investigation import correlate_incident_timeline
from .incident_presenter import _event_row, _incident_rows
from .incident_utils import DEFAULT_INCIDENT_GAP_SECONDS
from .manager import AppManager

LOGGER = logging.getLogger(__name__)

CROSS_CAMERA_TRACE_LIMITATIONS = (
    "Confirmed recognized faces can link incidents across cameras.",
    "Possible face matches remain uncertain.",
    "Shared person, vehicle, or animal labels plus nearby time provide context only.",
    "Appearance similarity uses durable, model-versioned ReID vectors and is stronger "
    "than a shared class label, but it is not proof of identity.",
    "Camera angle, lighting, occlusion, and visually similar subjects can change the score.",
    "Only the strongest 12 candidates and at most 24 hours are returned.",
)

MATCH_EXPORT_KEYS = (
    "event_id",
    "camera_id",
    "start_at",
    "seconds_from_anchor",
    "match_strength",
    "confidence",
    "reasons",
    "appearance_similarity",
)


def parse_trace_datetime(value: str, selected_zone: ZoneInfo) -> datetime | None:
    if not str(value or "").strip():
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=selected_zone)
    return parsed.astimezone(timezone.utc)


def prioritize_trace_candidates(
    candidate_summaries: list[dict[str, Any]],
    appearance_matches: list[dict[str, Any]],
    appearance_event_ids: set[int],
    distance_from_anchor: Callable[[dict[str, Any]], float],
    *,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Keep strongest appearance evidence before filling a bounded temporal scan."""
    candidate_summaries = sorted(candidate_summaries, key=distance_from_anchor)
    appearance_summaries = [
        summary
        for summary in candidate_summaries
        if any(
            int(event.get("id") or 0) in appearance_event_ids
            for event in summary.get("events") or []
        )
    ]
    appearance_score_by_event = {
        int(item.get("event_id") or 0): float(item.get("similarity") or 0.0)
        for item in appearance_matches
        if item.get("visually_similar")
    }
    appearance_summaries.sort(
        key=lambda summary: max(
            (
                appearance_score_by_event.get(int(event.get("id") or 0), 0.0)
                for event in summary.get("events") or []
            ),
            default=0.0,
        ),
        reverse=True,
    )
    retained_ids = {id(summary) for summary in appearance_summaries}
    retained_appearance = appearance_summaries[: min(100, limit)]
    return retained_appearance + [
        summary
        for summary in candidate_summaries
        if id(summary) not in retained_ids
    ][: max(0, limit - len(retained_appearance))]


def _serialize_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: item.get(key) for key in MATCH_EXPORT_KEYS} for item in matches]


def build_cross_camera_trace(
    manager: AppManager,
    *,
    resolve_event: Callable[[AppManager, int], dict[str, Any] | None],
    hydrate: Callable[[AppManager, list[dict[str, Any]]], list[dict[str, Any]]],
    with_faces: Callable[[AppManager, list[dict[str, Any]]], list[dict[str, Any]]],
    event_id: int | None = None,
    object_label: str = "",
    face_name: str = "",
    start_at: str = "",
    end_at: str = "",
    time_zone: str = "America/New_York",
    limit: int = 12,
) -> dict[str, Any]:
    """Build a bounded cross-camera investigation timeline around an anchor incident."""
    wanted_label = str(object_label or "").strip().lower()
    wanted_face = str(face_name or "").strip()
    if not event_id and not wanted_face and not wanted_label:
        raise ValueError("event_id, face_name, or object_label is required")

    anchor = resolve_event(manager, int(event_id)) if event_id else None
    if event_id and anchor is None:
        raise LookupError("incident was not found")

    try:
        selected_zone = ZoneInfo(time_zone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("unknown timezone") from exc

    now = datetime.now(timezone.utc)
    try:
        anchor_at = datetime.fromisoformat(
            str((anchor or {}).get("start_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        anchor_at = now
    if anchor_at.tzinfo is None:
        anchor_at = anchor_at.replace(tzinfo=timezone.utc)

    default_start = anchor_at - timedelta(minutes=15) if anchor else now - timedelta(hours=24)
    default_end = anchor_at + timedelta(minutes=15) if anchor else now
    start = parse_trace_datetime(start_at, selected_zone) or default_start
    end = parse_trace_datetime(end_at, selected_zone) or default_end
    if end <= start:
        start, end = end, start
    start = max(start, end - timedelta(hours=24))

    appearance_matches: list[dict[str, Any]] = []
    appearance_index = getattr(manager, "appearance_index", None)
    if event_id and appearance_index is not None:
        try:
            appearance_matches = appearance_index.matches(
                int(event_id),
                start_at=start.isoformat(),
                end_at=end.isoformat(),
                cross_camera_only=True,
                limit=100,
            )
        except Exception:
            LOGGER.exception(
                "cross-camera appearance lookup failed for event %s", event_id
            )

    appearance_event_ids = {
        int(item.get("event_id") or 0)
        for item in appearance_matches
        if item.get("visually_similar")
    }
    rows = [
        _event_row(row)
        for row in manager.events.between_compact(start.isoformat(), end.isoformat())
    ]
    summaries = _incident_rows(rows, DEFAULT_INCIDENT_GAP_SECONDS)
    anchor_labels = {
        str(label).strip().lower()
        for label in (anchor or {}).get("labels") or []
        if str(label).strip()
    }
    target_labels = {wanted_label} if wanted_label else anchor_labels
    candidate_summaries = [
        summary
        for summary in summaries
        if not target_labels
        or target_labels
        & {str(label).strip().lower() for label in summary.get("labels") or []}
        or wanted_face
        or any(
            int(event.get("id") or 0) in appearance_event_ids
            for event in summary.get("events") or []
        )
    ]

    def distance_from_anchor(summary: dict[str, Any]) -> float:
        parsed = parse_trace_datetime(str(summary.get("start_at") or ""), selected_zone)
        return (
            abs(parsed.timestamp() - anchor_at.timestamp())
            if parsed is not None
            else float("inf")
        )

    candidate_summaries = prioritize_trace_candidates(
        candidate_summaries,
        appearance_matches,
        appearance_event_ids,
        distance_from_anchor,
    )
    candidates = with_faces(manager, hydrate(manager, candidate_summaries))
    correlation_anchor = anchor or {
        "representative_event_id": 0,
        "camera_id": "",
        "start_at": start.isoformat(),
        "labels": [wanted_label] if wanted_label else [],
        "faces": [],
    }
    bounded_limit = max(1, min(int(limit), 12))
    matches = correlate_incident_timeline(
        correlation_anchor,
        candidates,
        object_label=object_label,
        face_name=face_name,
        limit=bounded_limit,
    )

    incident_by_event_id: dict[int, dict[str, Any]] = {}
    for incident in candidates:
        representative_id = int(incident.get("representative_event_id") or 0)
        if representative_id > 0:
            incident_by_event_id[representative_id] = incident
        for event in incident.get("events") or []:
            candidate_event_id = int(event.get("id") or 0)
            if candidate_event_id > 0:
                incident_by_event_id[candidate_event_id] = incident

    matches_by_event_id = {int(item.get("event_id") or 0): item for item in matches}
    for appearance in appearance_matches:
        if not appearance.get("visually_similar"):
            continue
        matched_incident = incident_by_event_id.get(int(appearance["event_id"]))
        if matched_incident is None:
            continue
        representative_id = int(
            matched_incident.get("representative_event_id") or appearance["event_id"]
        )
        similarity = float(appearance.get("similarity") or 0.0)
        model_kind = str(appearance.get("model_kind") or "object").title()
        reason = (
            f"{model_kind} appearance is {round(similarity * 100)}% similar "
            "using the same ReID model"
        )
        existing = matches_by_event_id.get(representative_id)
        if existing is not None:
            existing.setdefault("reasons", []).append(reason)
            existing["appearance_similarity"] = round(similarity, 4)
            if existing.get("match_strength") not in {
                "confirmed_identity",
                "possible_identity",
            }:
                existing["match_strength"] = "appearance_similarity"
                existing["confidence"] = round(similarity, 3)
            continue
        matched_at = str(
            matched_incident.get("start_at") or appearance.get("created_at") or ""
        )
        matched_epoch = parse_trace_datetime(matched_at, selected_zone)
        item = {
            "incident": matched_incident,
            "event_id": representative_id,
            "camera_id": str(
                matched_incident.get("camera_id") or appearance.get("camera_id") or ""
            ),
            "start_at": matched_at,
            "seconds_from_anchor": round(
                matched_epoch.timestamp() - anchor_at.timestamp()
                if matched_epoch is not None
                else 0.0,
                1,
            ),
            "match_strength": "appearance_similarity",
            "confidence": round(similarity, 3),
            "appearance_similarity": round(similarity, 4),
            "reasons": [reason],
        }
        matches.append(item)
        matches_by_event_id[representative_id] = item

    strength_rank = {
        "confirmed_identity": 4,
        "possible_identity": 3,
        "appearance_similarity": 2,
        "context_candidate": 1,
    }
    matches = sorted(
        sorted(
            matches,
            key=lambda item: (
                -strength_rank.get(str(item.get("match_strength") or ""), 0),
                -float(item.get("confidence") or 0.0),
                abs(float(item.get("seconds_from_anchor") or 0.0)),
            ),
        )[:bounded_limit],
        key=lambda item: str(item.get("start_at") or ""),
    )

    counts = {
        "confirmed_identity": sum(
            item["match_strength"] == "confirmed_identity" for item in matches
        ),
        "possible_identity": sum(
            item["match_strength"] == "possible_identity" for item in matches
        ),
        "appearance_similarity": sum(
            item["match_strength"] == "appearance_similarity" for item in matches
        ),
        "context_candidate": sum(
            item["match_strength"] == "context_candidate" for item in matches
        ),
    }
    return {
        "anchor_event_id": int(event_id) if event_id else None,
        "anchor_camera_id": (anchor or {}).get("camera_id"),
        "start_at": start.isoformat(),
        "end_at": end.isoformat(),
        "object_label": object_label,
        "face_name": face_name,
        "matches": _serialize_matches(matches),
        "counts": counts,
        "summary": (
            f"Found {len(matches)} bounded timeline candidate(s): "
            f"{counts['confirmed_identity']} confirmed identity, "
            f"{counts['possible_identity']} possible identity, "
            f"{counts['appearance_similarity']} appearance-similar, and "
            f"{counts['context_candidate']} context-only."
        ),
        "limitations": list(CROSS_CAMERA_TRACE_LIMITATIONS),
    }

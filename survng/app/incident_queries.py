"""Incident and event read/query application boundary."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException

from .audit_ai import motion_audit_interpretation
from .incident_presenter import (
    _event_row,
    _incident_list_payload,
    _incident_row,
    _incident_rows,
)
from .incident_utils import DEFAULT_INCIDENT_GAP_SECONDS, event_snapshot_path
from .identity_projection import apply_incident_identities
from .manager import AppManager
from .manager_access import ManagerAccessCoordinator, manager_generation_lease


def _motion_audit_row(row: dict[str, Any], storage_dir: Path, media_storage=None) -> dict[str, Any]:
    audit = dict(row)
    try:
        features = json.loads(str(audit.pop("features_json", "{}") or "{}"))
    except (json.JSONDecodeError, TypeError):
        features = {}
    audit["features"] = features if isinstance(features, dict) else {}
    snapshot_path = str(audit.pop("snapshot_path", "") or "")
    try:
        event_snapshot_path(storage_dir, {"snapshot_path": snapshot_path}, media_storage)
        audit["has_snapshot"] = True
    except (FileNotFoundError, PermissionError):
        audit["has_snapshot"] = False
    raw_outcome = audit.get("object_detected")
    audit["object_detected"] = None if raw_outcome is None else bool(raw_outcome)
    audit["interpretation"] = motion_audit_interpretation(
        reason=audit.get("reason"),
        event_id=audit.get("event_id"),
        object_detected=audit["object_detected"],
    )
    return audit


def _filter_incidents_by_event_type(
    incidents: list[dict[str, Any]], event_type: str
) -> list[dict[str, Any]]:
    if event_type == "object":
        return [item for item in incidents if item.get("has_objects")]
    if event_type == "motion":
        return [item for item in incidents if not item.get("has_objects")]
    return incidents


def _filter_incident_summaries(
    summaries: list[dict[str, Any]],
    event_type: str,
    camera_id: str = "",
    object_label: str = "",
    zone: str = "",
) -> list[dict[str, Any]]:
    filtered = _filter_incidents_by_event_type(summaries, event_type)
    if camera_id:
        filtered = [item for item in filtered if item.get("camera_id") == camera_id]
    if object_label:
        filtered = [item for item in filtered if object_label in item.get("labels", [])]
    if zone:
        filtered = [item for item in filtered if zone in item.get("zones", [])]
    return filtered


def _filter_incidents_by_person(
    manager: AppManager,
    incidents: list[dict[str, Any]],
    person_id: int,
) -> list[dict[str, Any]]:
    event_ids = [
        int(event["id"])
        for incident in incidents
        for event in incident.get("events", [])
        if str(event.get("id", "")).isdigit()
    ]
    matching_event_ids = {
        int(observation["event_id"])
        for observation in manager.faces.for_event_ids(event_ids)
        if int(observation.get("person_id") or 0) == person_id
    }
    return [
        incident
        for incident in incidents
        if any(
            int(event.get("id") or 0) in matching_event_ids
            for event in incident.get("events", [])
        )
    ]


class IncidentQueryService:
    """Read, group, hydrate, and present incidents for one manager generation."""

    @staticmethod
    def events(manager: AppManager, limit: int = 100) -> list[dict[str, Any]]:
        events = [_event_row(row) for row in manager.events.recent(limit)]
        wrapped = [{"events": [event]} for event in events]
        IncidentQueryService.with_faces(manager, wrapped)
        return [
            incident["events"][0]
            for incident in wrapped
            if incident.get("events")
        ]

    @staticmethod
    def recent_summaries(
        manager: AppManager, limit: int, gap_seconds: int
    ) -> list[dict[str, Any]]:
        batch_size = max(500, min(5000, limit * 8))
        compact_rows: list[dict[str, Any]] = []
        before_created_at: str | None = None
        before_id: int | None = None

        while True:
            batch = manager.events.recent_compact(
                batch_size, before_created_at, before_id
            )
            if not batch:
                return _incident_rows(compact_rows, gap_seconds)[:limit]
            compact_rows.extend(_event_row(row) for row in batch)
            summaries = _incident_rows(compact_rows, gap_seconds)
            if len(summaries) > limit or len(batch) < batch_size:
                return summaries[:limit]
            oldest = batch[-1]
            before_created_at = str(oldest["created_at"])
            before_id = int(oldest["id"])

    @staticmethod
    def recent_filtered_summaries(
        manager: AppManager,
        *,
        limit: int,
        offset: int,
        gap_seconds: int,
        event_type: str,
        camera_id: str = "",
        object_label: str = "",
        zone: str = "",
    ) -> tuple[list[dict[str, Any]], bool, list[dict[str, Any]]]:
        desired = offset + limit + 1
        compact_rows: list[dict[str, Any]] = []
        before_created_at: str | None = None
        before_id: int | None = None
        batch_size = max(500, min(5000, desired * 16))

        while True:
            batch = manager.events.recent_compact(
                batch_size, before_created_at, before_id, camera_id
            )
            if not batch:
                summaries = _incident_rows(compact_rows, gap_seconds)
                filtered = _filter_incident_summaries(
                    summaries, event_type, camera_id, object_label, zone
                )
                return filtered[offset : offset + limit], False, summaries
            compact_rows.extend(_event_row(row) for row in batch)
            summaries = _incident_rows(compact_rows, gap_seconds)
            filtered = _filter_incident_summaries(
                summaries, event_type, camera_id, object_label, zone
            )
            if len(filtered) >= desired:
                return filtered[offset : offset + limit], True, summaries
            if len(batch) < batch_size:
                return filtered[offset : offset + limit], False, summaries
            oldest = batch[-1]
            before_created_at = str(oldest["created_at"])
            before_id = int(oldest["id"])

    @staticmethod
    def hydrate(
        manager: AppManager, summaries: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        event_ids = [
            int(event["id"])
            for summary in summaries
            for event in summary.get("events", [])
            if str(event.get("id", "")).isdigit()
        ]
        full_events = {
            int(event["id"]): _event_row(event)
            for event in manager.events.get_many(event_ids)
        }
        observations_by_event: dict[int, list[dict[str, Any]]] = {}
        for audit in manager.events.motion_audits_for_related_events(event_ids):
            related_event_id = int(audit.get("related_event_id") or 0)
            observations_by_event.setdefault(related_event_id, []).append(
                _motion_audit_row(audit, manager.storage_dir, manager.media_storage)
            )
        for event_id, event in full_events.items():
            event["motion_observations"] = observations_by_event.get(event_id, [])
        hydrated: list[dict[str, Any]] = []
        for summary in summaries:
            events = [
                full_events[int(event["id"])]
                for event in summary.get("events", [])
                if int(event.get("id") or 0) in full_events
            ]
            if events:
                hydrated.append(
                    _incident_row(str(summary.get("camera_id") or ""), events)
                )
        return hydrated

    @staticmethod
    def with_faces(
        manager: AppManager, incidents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        event_ids = [
            int(event["id"])
            for incident in incidents
            for event in incident.get("events", [])
            if str(event.get("id", "")).isdigit()
        ]
        observations_by_event: dict[int, list[dict[str, Any]]] = {}
        for observation in manager.faces.for_event_ids(event_ids):
            observations_by_event.setdefault(
                int(observation["event_id"]), []
            ).append(observation)

        status_rank = {"confirmed": 0, "automatic": 1, "possible": 2, "unknown": 3}

        def summarize(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
            summaries: dict[tuple[str, int], dict[str, Any]] = {}
            for observation in observations:
                person_id = observation.get("person_id")
                candidate_id = observation.get("candidate_person_id")
                if person_id is not None:
                    review_status = str(observation.get("review_status") or "")
                    automatic = bool(
                        observation.get("auto_identified")
                        or review_status == "auto_identified"
                    )
                    status = "automatic" if automatic else "confirmed"
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
                    unknown_cluster_id = int(observation.get("unknown_cluster_id") or 0)
                    identity_id = -unknown_cluster_id if unknown_cluster_id > 0 else 0
                    name = f"Unknown Person {unknown_cluster_id}" if unknown_cluster_id > 0 else "Unknown"
                    confidence = observation.get("candidate_confidence")
                    if confidence is None:
                        confidence = observation.get("confidence")
                try:
                    score = float(confidence or 0)
                except (TypeError, ValueError):
                    score = 0.0
                if not math.isfinite(score):
                    score = 0.0
                score = max(0.0, min(1.0, score))
                unknown_cluster_id = int(observation.get("unknown_cluster_id") or 0)
                key = (
                    status,
                    identity_id
                    if status != "unknown"
                    else (-unknown_cluster_id if unknown_cluster_id > 0 else int(observation["observation_id"])),
                )
                current = summaries.get(key)
                if current is None or score > current["confidence"]:
                    summaries[key] = {
                        "observation_id": int(observation["observation_id"]),
                        "identity_id": identity_id,
                        "unknown_cluster_id": observation.get("unknown_cluster_id"),
                        "name": name,
                        "status": status,
                        "review_status": (
                            review_status
                            if person_id is not None
                            else "suggested" if candidate_id is not None else "unknown"
                        ),
                        "source": (
                            "automatic" if status == "automatic"
                            else "operator" if status == "confirmed"
                            else "recognition" if status == "possible"
                            else "cluster"
                        ),
                        "confidence": round(score, 4),
                        "candidate_count": max(
                            1,
                            int((observation.get("consensus") or {}).get("candidate_count") or 1),
                        ),
                        "agreement_count": max(
                            0,
                            int((observation.get("consensus") or {}).get("agreement_count") or 0),
                        ),
                    }
            return sorted(
                summaries.values(),
                key=lambda face: (
                    status_rank[face["status"]],
                    -face["confidence"],
                    face["name"].lower(),
                ),
            )

        for incident in incidents:
            incident_observations: list[dict[str, Any]] = []
            for event in incident.get("events", []):
                event_observations = observations_by_event.get(
                    int(event.get("id") or 0), []
                )
                event["faces"] = summarize(event_observations)
                incident_observations.extend(event_observations)
            incident["faces"] = summarize(incident_observations)
        return apply_incident_identities(incidents)

    def incidents(
        self, manager: AppManager, limit: int = 200, gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(limit, 200))
        bounded_gap = max(5, min(gap_seconds, 300))
        summaries = self.recent_summaries(manager, bounded_limit, bounded_gap)
        return self.with_faces(manager, self.hydrate(manager, summaries))

    def feed(
        self,
        manager: AppManager,
        *,
        event_type: str = "object",
        camera_id: str = "",
        object_label: str = "",
        zone: str = "",
        limit: int = 18,
        offset: int = 0,
        gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS,
    ) -> dict[str, Any]:
        bounded_limit = max(1, min(limit, 100))
        bounded_offset = max(0, min(offset, 100_000))
        bounded_gap = max(5, min(gap_seconds, 300))
        page, has_more, scanned = self.recent_filtered_summaries(
            manager,
            limit=bounded_limit,
            offset=bounded_offset,
            gap_seconds=bounded_gap,
            event_type=event_type,
            camera_id=camera_id,
            object_label=object_label,
            zone=zone,
        )
        facets = {
            "camera_ids": sorted(
                {
                    str(item.get("camera_id") or "")
                    for item in scanned
                    if item.get("camera_id")
                }
            ),
            "labels": sorted(
                {
                    str(label)
                    for item in scanned
                    for label in item.get("labels", [])
                    if label
                }
            ),
            "zones": sorted(
                {
                    str(item_zone)
                    for item in scanned
                    for item_zone in item.get("zones", [])
                    if item_zone
                }
            ),
        }
        return {
            "items": [_incident_list_payload(item) for item in page],
            "limit": bounded_limit,
            "offset": bounded_offset,
            "has_more": has_more,
            "facets": facets,
        }

    def detail(
        self,
        manager: AppManager,
        event_ids: str,
        gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS,
    ) -> dict[str, Any]:
        try:
            requested_ids = list(
                dict.fromkeys(
                    int(value.strip())
                    for value in event_ids.split(",")
                    if value.strip()
                )
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="event_ids must be comma-separated integers"
            ) from exc
        if (
            not requested_ids
            or len(requested_ids) > 200
            or any(event_id <= 0 for event_id in requested_ids)
        ):
            raise HTTPException(
                status_code=422,
                detail="event_ids must contain 1 to 200 positive integers",
            )

        rows = manager.events.get_many(requested_ids)
        if {int(row["id"]) for row in rows} != set(requested_ids):
            raise HTTPException(status_code=404, detail="incident events were not found")
        bounded_gap = max(5, min(gap_seconds, 300))
        summaries = _incident_rows([_event_row(row) for row in rows], bounded_gap)
        if len(summaries) != 1:
            raise HTTPException(
                status_code=422, detail="event_ids do not identify one incident"
            )
        hydrated = self.with_faces(manager, self.hydrate(manager, summaries))
        if not hydrated:
            raise HTTPException(status_code=404, detail="incident was not found")
        return hydrated[0]

    @staticmethod
    def search(
        manager: AppManager,
        *,
        day: str = "",
        time_zone: str = "America/New_York",
        camera_id: str = "",
        event_type: str = "motion",
        object_label: str = "",
        zone: str = "",
        person_id: int = 0,
        limit: int = 18,
        offset: int = 0,
        gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS,
    ) -> dict[str, Any]:
        try:
            selected_zone = ZoneInfo(time_zone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="unknown timezone") from exc
        if day:
            try:
                selected_date = datetime.strptime(day, "%Y-%m-%d").date()
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail="day must use YYYY-MM-DD"
                ) from exc
        else:
            selected_date = datetime.now(selected_zone).date()
            day = selected_date.isoformat()
        day_start = datetime.combine(selected_date, datetime.min.time(), selected_zone)
        day_end = day_start + timedelta(days=1)
        bounded_gap = max(5, min(gap_seconds, 300))
        query_start = day_start.astimezone(timezone.utc) - timedelta(seconds=bounded_gap)
        query_end = day_end.astimezone(timezone.utc) + timedelta(seconds=bounded_gap)
        compact_rows = [
            _event_row(row)
            for row in manager.events.between_compact(
                query_start.isoformat(), query_end.isoformat(), camera_id
            )
        ]
        # Result rows may be camera-constrained, but facets intentionally retain
        # the full day's choices so changing the camera selector does not make
        # other cameras/labels/zones disappear from the filter UI.
        facet_rows = compact_rows
        if camera_id:
            facet_rows = [
                _event_row(row)
                for row in manager.events.between_compact(
                    query_start.isoformat(), query_end.isoformat()
                )
            ]
        day_start_epoch = day_start.timestamp()
        day_end_epoch = day_end.timestamp()
        day_incidents = [
            incident
            for incident in _incident_rows(compact_rows, gap_seconds=bounded_gap)
            if incident["last_epoch"] >= day_start_epoch
            and incident["start_epoch"] < day_end_epoch
        ]
        facet_incidents = [
            incident
            for incident in _incident_rows(facet_rows, gap_seconds=bounded_gap)
            if incident["last_epoch"] >= day_start_epoch
            and incident["start_epoch"] < day_end_epoch
        ]
        facets = {
            "camera_ids": sorted(
                {
                    str(item.get("camera_id") or "")
                    for item in facet_incidents
                    if item.get("camera_id")
                }
            ),
            "labels": sorted(
                {
                    str(label)
                    for item in facet_incidents
                    for label in item.get("labels", [])
                    if label
                }
            ),
            "zones": sorted(
                {
                    str(item_zone)
                    for item in facet_incidents
                    for item_zone in item.get("zones", [])
                    if item_zone
                }
            ),
        }
        filtered = _filter_incident_summaries(
            day_incidents, event_type, camera_id, object_label, zone
        )
        selected_person_id = max(0, int(person_id))
        if selected_person_id:
            filtered = _filter_incidents_by_person(
                manager, filtered, selected_person_id
            )
        bounded_limit = max(1, min(limit, 100))
        bounded_offset = max(0, offset)
        page_summaries = filtered[bounded_offset : bounded_offset + bounded_limit]
        return {
            "items": [_incident_list_payload(item) for item in page_summaries],
            "total": len(filtered),
            "limit": bounded_limit,
            "offset": bounded_offset,
            "day": day,
            "time_zone": time_zone,
            "start_at": day_start.astimezone(timezone.utc).isoformat(),
            "end_at": day_end.astimezone(timezone.utc).isoformat(),
            "facets": facets,
        }

    def resolve_event(
        self, manager: AppManager, event_id: int
    ) -> dict[str, Any] | None:
        row = manager.events.get(event_id)
        if row is None:
            return None
        try:
            anchor = datetime.fromisoformat(
                str(row["created_at"]).replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError):
            return None
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        rows = [
            _event_row(candidate)
            for candidate in manager.events.for_camera_range(
                str(row.get("camera_id") or ""),
                (anchor - timedelta(minutes=15)).isoformat(),
                (anchor + timedelta(minutes=15)).isoformat(),
                limit=2000,
            )
        ]
        for summary in _incident_rows(rows, DEFAULT_INCIDENT_GAP_SECONDS):
            if any(
                int(event.get("id") or 0) == event_id
                for event in summary.get("events") or []
            ):
                hydrated = self.with_faces(manager, self.hydrate(manager, [summary]))
                return hydrated[0] if hydrated else summary
        return None


@dataclass(frozen=True, slots=True)
class IncidentQueryDependencies:
    get_manager: Callable[[], AppManager]
    manager_lock: threading.RLock
    manager_access: ManagerAccessCoordinator | None = None


@dataclass(frozen=True, slots=True)
class IncidentQueryRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]


def create_incident_query_router(
    dependencies: IncidentQueryDependencies,
    service: IncidentQueryService,
) -> IncidentQueryRouteBundle:
    router = APIRouter()

    def with_manager(operation: Callable[[AppManager], Any]) -> Any:
        with manager_generation_lease(
            dependencies.manager_access,
            dependencies.manager_lock,
            dependencies.get_manager,
        ) as active_manager:
            return operation(active_manager)

    @router.get("/api/events")
    def events(limit: int = 100) -> list[dict[str, Any]]:
        return with_manager(lambda active: service.events(active, limit))

    @router.get("/api/incidents")
    def incidents(
        limit: int = 200,
        gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS,
    ) -> list[dict[str, Any]]:
        return with_manager(
            lambda active: service.incidents(active, limit, gap_seconds)
        )

    @router.get("/api/incidents/feed")
    def incident_feed(
        event_type: str = "object",
        camera_id: str = "",
        object_label: str = "",
        zone: str = "",
        limit: int = 18,
        offset: int = 0,
        gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS,
    ) -> dict[str, Any]:
        return with_manager(
            lambda active: service.feed(
                active,
                event_type=event_type,
                camera_id=camera_id,
                object_label=object_label,
                zone=zone,
                limit=limit,
                offset=offset,
                gap_seconds=gap_seconds,
            )
        )

    @router.get("/api/incidents/detail")
    def incident_detail(
        event_ids: str,
        gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS,
    ) -> dict[str, Any]:
        return with_manager(
            lambda active: service.detail(active, event_ids, gap_seconds)
        )

    @router.get("/api/incidents/search")
    def incident_search(
        day: str = "",
        time_zone: str = "America/New_York",
        camera_id: str = "",
        event_type: str = "motion",
        object_label: str = "",
        zone: str = "",
        person_id: int = 0,
        limit: int = 18,
        offset: int = 0,
        gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS,
    ) -> dict[str, Any]:
        return with_manager(
            lambda active: service.search(
                active,
                day=day,
                time_zone=time_zone,
                camera_id=camera_id,
                event_type=event_type,
                object_label=object_label,
                zone=zone,
                person_id=person_id,
                limit=limit,
                offset=offset,
                gap_seconds=gap_seconds,
            )
        )

    @router.get("/api/incidents/by-event/{event_id}")
    def incident_for_event(event_id: int) -> dict[str, Any]:
        incident = with_manager(lambda active: service.resolve_event(active, event_id))
        if incident is None:
            raise HTTPException(status_code=404, detail="incident was not found")
        return incident

    return IncidentQueryRouteBundle(
        router=router,
        handlers={
            "events": events,
            "incidents": incidents,
            "incident_feed": incident_feed,
            "incident_detail": incident_detail,
            "incident_search": incident_search,
            "incident_for_event": incident_for_event,
        },
    )

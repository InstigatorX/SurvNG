"""Incident and event read/query application boundary."""

from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import FileResponse
from .event_store.scenes import SceneConflict
from .incident_utils import snapshot_media_type

from .audit_ai import motion_audit_interpretation
from .cross_camera_trace import build_cross_camera_trace
from .incident_presenter import (
    _event_row,
    _incident_list_payload,
)
from .incident_utils import (
    DEFAULT_INCIDENT_GAP_SECONDS,
    event_snapshot_path,
)
from .identity_projection import apply_incident_identities
from .manager import AppManager
from .manager_access import ManagerAccessCoordinator, manager_generation_lease


LOGGER = logging.getLogger(__name__)


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
        filtered = [item for item in filtered if camera_id in item.get("camera_ids", [item.get("camera_id")])]
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
    """Read, hydrate, and present canonical incidents for one manager generation."""

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
        return manager.events.list_scene_incidents(limit=limit)

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
        page = manager.events.list_scene_incident_cards(limit=limit+1, offset=offset,
            event_type=event_type, camera_id=camera_id, object_label=object_label, zone=zone)
        # Facets are scalar metadata; unmatched frame histories are never hydrated.
        facets = manager.events.scene_incident_facets()
        return page[:limit], len(page)>limit, [facets]

    @staticmethod
    def hydrate(
        manager: AppManager, summaries: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # Membership and retained observations are already hydrated by the store.
        return summaries

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
                    str(camera)
                    for item in scanned
                    for camera in item.get("camera_ids", [item.get("camera_id")]) if camera
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

        incidents = [manager.events.scene_incident(event_id=event_id) for event_id in requested_ids]
        if any(item is None for item in incidents):
            raise HTTPException(status_code=404, detail="incident events were not found")
        ids = {item["id"] for item in incidents}
        if len(ids) != 1:
            raise HTTPException(status_code=422, detail={"message": "events belong to multiple incidents", "incident_ids": sorted(ids)})
        return self.with_faces(manager, [incidents[0]])[0]

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
        bounds = {"start_epoch":day_start.timestamp(), "end_epoch":day_end.timestamp()}
        facets = manager.events.scene_incident_facets(**bounds)
        filters = {**bounds, "event_type":event_type, "camera_id":camera_id, "object_label":object_label, "zone":zone}
        selected_person_id = max(0, int(person_id))
        bounded_limit = max(1, min(limit, 100))
        bounded_offset = max(0, offset)
        if selected_person_id:
            # Face evidence may be supplied by a separate store. Scan bounded
            # ID-only pages and hydrate only the selected matching incidents.
            selected = []
            total = cursor = 0
            while True:
                batch = manager.events.list_scene_incidents(**filters,limit=100,offset=cursor,summary_only=True)
                matches = _filter_incidents_by_person(manager,batch,selected_person_id)
                for match in matches:
                    if bounded_offset<=total<bounded_offset+bounded_limit:
                        selected.append(match["id"])
                    total += 1
                if len(batch)<100:
                    break
                cursor += len(batch)
            page_summaries = [manager.events.scene_incident(incident_id) for incident_id in selected]
        else:
            total = manager.events.count_scene_incidents(**filters)
            page_summaries = manager.events.list_scene_incident_cards(**filters,limit=bounded_limit,offset=bounded_offset)
        return {
            "items": [_incident_list_payload(item) for item in page_summaries],
            "total": total,
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
        incident = manager.events.scene_incident(event_id=event_id)
        return self.with_faces(manager, [incident])[0] if incident else None

    def notification_detail(self, manager: AppManager, incident_id: str) -> dict[str, Any]:
        detail = manager.events.scene_incident(incident_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="incident was not found")
        detail = self.with_faces(manager, [detail])[0]
        camera = next((camera for camera in manager.config.cameras if camera.id == detail.get("camera_id")), None)
        return {"incident": detail, "notification": manager.incidents.get(incident_id),
                "camera_name": camera.name if camera else detail.get("camera_id"), "incident_id": detail["id"]}

    @staticmethod
    def analysis(manager: AppManager, incident_id: str, *, request: bool = False) -> dict[str, Any]:
        """Read progress or explicitly admit bounded, optional recorded work.

        This is deliberately separate from detail/search: polling and prefetch
        must never start inference. A manager lease fences configuration swaps.
        """
        try:
            status = manager.events.incident_analysis_status(incident_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="incident was not found") from exc
        tracking = manager.config.detector.tracking
        camera_ids = {episode["camera_id"] for episode in status["episodes"]}
        available_ids = set()
        for camera_id in camera_ids:
            worker = manager.workers.get(camera_id)
            if (tracking.enabled and worker is not None
                    and worker.tracking_lifecycle.enabled()
                    and worker.tracking_lifecycle.accepting()):
                available_ids.add(camera_id)
        enabled = bool(available_ids) and (
            tracking.analysis_mode == "on_demand"
            or any(episode.get("admission") in {"deferred", "demand"}
                   and episode["camera_id"] in available_ids for episode in status["episodes"])
        )
        if request and enabled:
            try:
                status = manager.events.request_incident_analysis(incident_id, camera_ids=available_ids)
            except LookupError as exc:
                raise HTTPException(status_code=404, detail="incident was not found") from exc
            # Commit happens before this advisory signal. The existing camera
            # worker remains the only owner that may claim, decode, or wait for
            # capacity; the HTTP request merely removes recovery-poll latency.
            for camera_id in available_ids:
                worker = manager.workers.get(camera_id)
                notify = getattr(
                    getattr(worker, "motion_incidents", None),
                    "notify_scene_analysis_requested",
                    None,
                )
                if callable(notify):
                    try:
                        notify()
                    except Exception:
                        # The request is already durable. An advisory wake must
                        # not turn successful admission into an HTTP failure;
                        # bounded recovery will still discover it.
                        LOGGER.exception(
                            "incident analysis wake failed for camera %s",
                            camera_id,
                        )
        result = {**status, "mode": tracking.analysis_mode, "enabled": enabled}
        if camera_ids - available_ids:
            episodes = [{**episode, "status": "unavailable"}
                        if episode["camera_id"] not in available_ids and episode["status"] not in {"complete", "partial"}
                        else episode for episode in status["episodes"]]
            states = {episode["status"] for episode in episodes}
            aggregate = next((state for state in ("running", "queued", "deferred", "partial") if state in states),
                             "partial" if "complete" in states and "unavailable" in states else
                             "complete" if states == {"complete"} else "unavailable")
            result.update(episodes=episodes, status=aggregate,
                          remaining=sum(episode["status"] == "deferred" for episode in episodes),
                          message="Some detailed analysis is unavailable while camera detection or tracking is off.")
        return result

    def cross_camera_trace(
        self,
        manager: AppManager,
        event_id: int,
        *,
        start_at: str = "",
        end_at: str = "",
        object_label: str = "",
        face_name: str = "",
        time_zone: str = "America/New_York",
        limit: int = 12,
    ) -> dict[str, Any]:
        try:
            return build_cross_camera_trace(
                manager,
                resolve_event=self.resolve_event,
                hydrate=self.hydrate,
                with_faces=self.with_faces,
                event_id=event_id,
                object_label=object_label,
                face_name=face_name,
                start_at=start_at,
                end_at=end_at,
                time_zone=time_zone,
                limit=limit,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="incident was not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


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

    @router.get("/api/events/{event_id}")
    def event_evidence(event_id: int) -> dict[str, Any]:
        def resolve(active):
            event = active.events.get(event_id)
            if event is None:
                raise HTTPException(status_code=404, detail="event was not found")
            wrapped = service.with_faces(active, [{"events": [_event_row(event)]}])
            return wrapped[0]["events"][0]
        return with_manager(resolve)

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
        event_ids: str = "",
        gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS,
        incident_id: str = "",
    ) -> dict[str, Any]:
        if incident_id:
            return canonical_incident(incident_id)
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

    @router.get("/api/incidents/notification/{incident_id}")
    def notification_incident(incident_id: str) -> dict[str, Any]:
        return with_manager(lambda active: service.notification_detail(active, incident_id))

    @router.get("/api/incidents/by-event/{event_id}")
    def incident_for_event(event_id: int) -> dict[str, Any]:
        incident = with_manager(lambda active: service.resolve_event(active, event_id))
        if incident is None:
            raise HTTPException(status_code=404, detail="incident was not found")
        return incident

    @router.get("/api/incidents/by-event/{event_id}/cross-camera-trace")
    def cross_camera_trace(
        event_id: int,
        start_at: str = "",
        end_at: str = "",
        object_label: str = "",
        face_name: str = "",
        time_zone: str = "America/New_York",
        limit: int = 12,
    ) -> dict[str, Any]:
        return with_manager(
            lambda active: service.cross_camera_trace(
                active,
                event_id,
                start_at=start_at,
                end_at=end_at,
                object_label=object_label,
                face_name=face_name,
                time_zone=time_zone,
                limit=limit,
            )
        )

    @router.get("/api/observations")
    def observation_reviews(start_at: str = "", end_at: str = "", camera_id: str = "", status: str = "", limit: int = 25, offset: int = 0):
        if status not in {"", "pending", "not_established", "incomplete", "established"}:
            raise HTTPException(status_code=422, detail="unknown observation status")
        try:
            end = datetime.fromisoformat(end_at.replace("Z", "+00:00")) if end_at else datetime.now(timezone.utc)
            start = datetime.fromisoformat(start_at.replace("Z", "+00:00")) if start_at else end - timedelta(days=1)
            if start.tzinfo is None or end.tzinfo is None or not 0 < (end-start).total_seconds() <= 31*86400:
                raise ValueError("invalid window")
        except ValueError as error:
            raise HTTPException(status_code=422, detail="a timezone-aware observation window of at most 31 days is required") from error
        return with_manager(lambda active: active.events.scene_observation_reviews(
            start_epoch=start.timestamp(), end_epoch=end.timestamp(), camera_id=camera_id,
            status=status, limit=limit, offset=offset))

    @router.get("/api/observations/{sample_id}/snapshot")
    def observation_sample_snapshot(sample_id: str):
        def resolve(active):
            sample = active.events.scene_sample(sample_id)
            if sample is None or not sample.get("snapshot_path"):
                raise HTTPException(status_code=404, detail="supporting image unavailable")
            try:
                path = event_snapshot_path(active.storage_dir, sample, active.media_storage)
            except (FileNotFoundError, PermissionError):
                raise HTTPException(status_code=404, detail="supporting image expired")
            return FileResponse(path, media_type=snapshot_media_type(path), headers={"Cache-Control":"private, no-cache"})
        return with_manager(resolve)

    @router.get("/api/observations/{record_id}")
    def observation_review(record_id: str):
        result = with_manager(lambda active: active.events.scene_observation_review(record_id))
        if result is None:
            raise HTTPException(status_code=404, detail="observation review was not found")
        return result

    @router.get("/api/incidents/observations/{observation_id}/snapshot")
    def observation_snapshot(observation_id: str, width: int = 0, quality: int = 82):
        def resolve(active):
            observation = active.events.scene_observation(observation_id)
            if observation is None:
                observation = active.events.scene_acquired_observation(observation_id)
            if observation is None or not observation.get("snapshot_path"):
                raise HTTPException(status_code=404, detail="supporting image unavailable")
            try:
                path = event_snapshot_path(active.storage_dir, observation, active.media_storage)
            except (FileNotFoundError, PermissionError):
                raise HTTPException(status_code=404, detail="supporting image expired")
            if width:
                import cv2
                from .appearance_routes import _jpeg_thumbnail
                safe_width, safe_quality = max(160,min(2560,width)), max(50,min(95,quality))
                stat = path.stat()
                identity = f"{path}:{stat.st_mtime_ns}:{stat.st_size}:{safe_width}:{safe_quality}"
                def build():
                    frame = cv2.imread(str(path))
                    if frame is None:
                        raise HTTPException(status_code=404, detail="supporting image unavailable")
                    return _jpeg_thumbnail(frame, safe_width, safe_quality)
                cached = active.image_cache.get_or_create("observations", identity, build)
                return FileResponse(cached, media_type="image/jpeg", headers={"Cache-Control":"private, no-cache"})
            return FileResponse(path, media_type=snapshot_media_type(path), headers={"Cache-Control": "private, no-cache"})
        return with_manager(resolve)

    @router.get("/api/incidents/{incident_id}")
    def canonical_incident(incident_id: str):
        def resolve(active):
            incident = active.events.scene_incident(incident_id)
            if incident is None:
                raise HTTPException(status_code=404, detail="incident was not found")
            return service.with_faces(active, [incident])[0]
        return with_manager(resolve)

    @router.post("/api/incidents/{incident_id}/corrections")
    def correct_incident(incident_id: str, correction: dict[str, Any] = Body(...)):
        revision = correction.get("expected_revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise HTTPException(status_code=422, detail="expected_revision is required")
        def correct(active):
            try:
                result = active.events.correct_scene_incident(incident_id, revision, correction)
            except SceneConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except LookupError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except (ValueError, TypeError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            active.state_events.publish("incident", {
                "incident_id": result["id"],
                "revision": result["revision"],
                "updated": True,
                "reason": "operator_correction",
            })
            return service.with_faces(active, [result])[0]
        return with_manager(correct)

    @router.get("/api/incidents/{incident_id}/analysis")
    def incident_analysis(incident_id: str):
        return with_manager(lambda active: service.analysis(active, incident_id))

    @router.post("/api/incidents/{incident_id}/analysis")
    def request_incident_analysis(incident_id: str):
        return with_manager(lambda active: service.analysis(active, incident_id, request=True))

    return IncidentQueryRouteBundle(
        router=router,
        handlers={
            "events": events,
            "event_evidence": event_evidence,
            "incidents": incidents,
            "incident_feed": incident_feed,
            "incident_detail": incident_detail,
            "incident_search": incident_search,
            "incident_for_event": incident_for_event,
            "cross_camera_trace": cross_camera_trace,
            "incident_analysis": incident_analysis,
            "request_incident_analysis": request_incident_analysis,
        },
    )

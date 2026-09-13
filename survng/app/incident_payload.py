"""Transport-independent incident notification payloads."""

from __future__ import annotations

import json
from typing import Any

from .incident_utils import event_epoch, stable_incident_id, stable_incident_key


class IncidentPayloadBuilder:
    @staticmethod
    def _event_objects(event: dict[str, Any]) -> list[dict[str, Any]]:
        raw = event.get("objects")
        if raw is None:
            try:
                raw = json.loads(str(event.get("objects_json") or "[]"))
            except (json.JSONDecodeError, TypeError, ValueError):
                raw = []
        if not isinstance(raw, list):
            return []
        detected: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("label") or item.get("incident_eligible") is False:
                continue
            try:
                confidence = float(item.get("confidence") or 0)
            except (TypeError, ValueError):
                continue
            if confidence > 0:
                detected.append(item)
        return detected

    @staticmethod
    def _event_camera_semantics(event: dict[str, Any]) -> list[dict[str, Any]]:
        raw = event.get("objects")
        if raw is None:
            try:
                raw = json.loads(str(event.get("objects_json") or "[]"))
            except (json.JSONDecodeError, TypeError, ValueError):
                raw = []
        if not isinstance(raw, list):
            return []
        for item in reversed(raw):
            if not isinstance(item, dict) or item.get("status") != "motion_qualification":
                continue
            qualification = item.get("motion_qualification")
            semantics = qualification.get("camera_semantics") if isinstance(qualification, dict) else None
            reports = semantics.get("reports") if isinstance(semantics, dict) else None
            return [dict(report) for report in reports or [] if isinstance(report, dict)]
        return []

    @classmethod
    def _incident_payload(cls, pending: dict[str, Any], state: str) -> dict[str, Any]:
        events = sorted(pending["events"].values(), key=event_epoch)
        first = events[0]
        last = events[-1]

        def representative_score(event: dict[str, Any]) -> tuple[int, float, int, int]:
            objects = cls._event_objects(event)
            return (
                int(bool(objects)),
                max((float(item.get("confidence") or 0) for item in objects), default=0.0),
                int(bool(event.get("snapshot_path"))),
                int(event.get("id") or 0),
            )

        representative = max(events, key=representative_score)
        object_summaries: dict[str, dict[str, Any]] = {}
        for event in events:
            for item in cls._event_objects(event):
                label = str(item.get("label") or "").strip()
                key = label.lower()
                depth_stats = item.get("depth_stats") if isinstance(item.get("depth_stats"), dict) else {}
                try:
                    median_distance_m = float(depth_stats.get("median_m"))
                except (TypeError, ValueError):
                    median_distance_m = None
                current = object_summaries.setdefault(key, {
                    "label": label,
                    "confidence": 0.0,
                    "zones": set(),
                    "count": 0,
                    "min_distance_m": None,
                    "max_distance_m": None,
                    "median_distance_m": None,
                    "_depth_distances_m": [],
                })
                current["confidence"] = max(current["confidence"], float(item.get("confidence") or 0))
                current["zones"].update(str(zone) for zone in item.get("zones", []) if zone)
                current["count"] += 1
                if median_distance_m is not None:
                    if current["min_distance_m"] is None:
                        current["min_distance_m"] = median_distance_m
                    else:
                        current["min_distance_m"] = min(
                            float(current["min_distance_m"]),
                            median_distance_m,
                        )
                    if current["max_distance_m"] is None:
                        current["max_distance_m"] = median_distance_m
                    else:
                        current["max_distance_m"] = max(
                            float(current["max_distance_m"]),
                            median_distance_m,
                        )
                    current["_depth_distances_m"].append(median_distance_m)
        for summary in object_summaries.values():
            distances = sorted(summary.pop("_depth_distances_m"))
            if distances:
                middle = len(distances) // 2
                summary["median_distance_m"] = (
                    distances[middle]
                    if len(distances) % 2
                    else (distances[middle - 1] + distances[middle]) / 2.0
                )
        objects = [
            {
                **summary,
                "confidence": round(float(summary["confidence"]), 4),
                "zones": sorted(summary["zones"]),
                **(
                    {"min_distance_m": round(float(summary["min_distance_m"]), 2)}
                    if summary["min_distance_m"] is not None
                    else {}
                ),
                **(
                    {"max_distance_m": round(float(summary["max_distance_m"]), 2)}
                    if summary["max_distance_m"] is not None
                    else {}
                ),
                **(
                    {"median_distance_m": round(float(summary["median_distance_m"]), 2)}
                    if summary["median_distance_m"] is not None
                    else {}
                ),
            }
            for summary in sorted(object_summaries.values(), key=lambda item: str(item["label"]).lower())
        ]
        first_id = first.get("id")
        incident_id = stable_incident_id(str(pending["camera_id"]), first_id)
        base_path = str(pending.get("base_path") or "").rstrip("/")
        representative_id = int(representative.get("id") or 0)
        identity_by_id: dict[int, dict[str, Any]] = {}
        for event in events:
            for identity in event.get("identities", []) or []:
                if not isinstance(identity, dict):
                    continue
                identity_id = int(identity.get("identity_id") or 0)
                if identity_id <= 0:
                    continue
                previous = identity_by_id.get(identity_id)
                if (
                    previous is None
                    or float(identity.get("confidence") or 0.0)
                    > float(previous.get("confidence") or 0.0)
                ):
                    identity_by_id[identity_id] = dict(identity)
        identities = sorted(
            identity_by_id.values(),
            key=lambda item: (
                -float(item.get("confidence") or 0.0),
                str(item.get("name") or "").lower(),
            ),
        )
        semantic_reports = [
            {
                **report,
                "source_event_id": event.get("id"),
                "source_created_at": event.get("created_at"),
            }
            for event in events
            for report in cls._event_camera_semantics(event)
        ]
        return {
            "schema_version": 1,
            "type": "incident",
            "state": state,
            "incident_id": incident_id,
            "incident_key": stable_incident_key(str(pending["camera_id"]), first_id),
            "camera_id": pending["camera_id"],
            "camera_name": pending["camera_name"],
            "started_at": first.get("created_at"),
            "ended_at": last.get("created_at"),
            "duration_seconds": round(max(0.0, event_epoch(last) - event_epoch(first)), 3),
            "event_count": len(events),
            "event_ids": [int(event.get("id") or 0) for event in events],
            "object_event_count": sum(bool(cls._event_objects(event)) for event in events),
            "has_objects": bool(objects),
            "classes": [item["label"] for item in objects],
            "zones": sorted({zone for item in objects for zone in item["zones"]}),
            "objects": objects,
            "camera_semantics": {"reports": semantic_reports},
            "identities": identities,
            "people": [
                str(item.get("name") or "")
                for item in identities
                if item.get("name")
            ],
            "representative_event_id": representative_id,
            "snapshot_url": f"{base_path}/api/events/{representative_id}/snapshot.jpg",
            "incidents_url": f"{base_path}/incidents",
        }

"""Transport-independent incident notification payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def canonical_incident_payload(incident: dict[str, Any]) -> dict[str, Any]:
    """Transport aliases for one canonical snapshot, without regrouping objects."""
    payload = deepcopy(incident)
    payload["schema_version"] = 3
    payload["type"] = "incident"
    payload["incident_id"] = str(payload.get("incident_id") or payload["id"])
    payload["id"] = payload["incident_id"]
    payload["objects"] = deepcopy(payload.get("scene_objects") or [])
    payload["classes"] = sorted({str(item["label"]) for item in payload["objects"] if item.get("label")})
    payload["has_objects"] = bool(payload["objects"])
    payload.setdefault("alert_decisions", [])
    payload.setdefault("started_at", payload.get("start_at"))
    payload.setdefault("ended_at", payload.get("end_at"))
    payload.setdefault("last_activity_at", payload.get("end_at"))
    payload.setdefault("event_ids", [int(event["id"]) for event in payload.get("events", [])])
    payload.setdefault("event_count", len(payload["event_ids"]))
    payload.setdefault("people", [str(item["name"]) for item in payload.get("identities", []) if item.get("name")])
    payload.setdefault("camera_name", payload.get("camera_id", ""))
    payload.setdefault("image_available", bool(payload.get("snapshot_path")))
    representative_id = payload.get("representative_event_id")
    if payload["image_available"] and representative_id:
        payload.setdefault("snapshot_url", f"/api/events/{int(representative_id)}/snapshot.jpg?v={int(payload.get('evidence_revision') or 0)}")
    else:
        payload["snapshot_url"] = None
    return payload

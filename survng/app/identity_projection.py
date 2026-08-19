"""Operator-facing identity projection for hydrated SurvNG events/incidents."""

from __future__ import annotations

from typing import Any


PERSON_LABELS = {"person", "child"}


def identity_summaries(faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize confirmed face summaries into a compact identity contract."""
    identities: dict[int, dict[str, Any]] = {}
    for face in faces or []:
        if not isinstance(face, dict):
            continue
        status = str(face.get("status") or "")
        if status not in {"confirmed", "unknown"}:
            continue
        unknown_cluster_id = int(face.get("unknown_cluster_id") or 0)
        if status == "unknown":
            if unknown_cluster_id <= 0:
                continue
            identity_id = -unknown_cluster_id
            person_id = None
            name = f"Unknown Person {unknown_cluster_id}"
        else:
            identity_id = int(face.get("identity_id") or face.get("person_id") or 0)
            if identity_id <= 0:
                continue
            person_id = int(face.get("person_id") or identity_id)
            name = str(face.get("name") or f"Person {identity_id}")
        item = {
            "identity_id": identity_id,
            "person_id": person_id,
            "unknown_cluster_id": unknown_cluster_id or None,
            "name": name,
            "status": status,
            "confidence": float(face.get("confidence") or 0.0),
            "observation_id": int(face.get("observation_id") or 0),
        }
        previous = identities.get(identity_id)
        if previous is None or item["confidence"] > previous["confidence"]:
            identities[identity_id] = item
    return sorted(
        identities.values(),
        key=lambda item: (-float(item["confidence"]), str(item["name"]).lower()),
    )


def apply_event_identity(event: dict[str, Any]) -> dict[str, Any]:
    """Attach identity metadata without guessing in ambiguous multi-person scenes."""
    identities = identity_summaries(event.get("faces") or [])
    event["identities"] = identities
    objects = event.get("objects")
    if not isinstance(objects, list):
        objects = []
    people = [
        item
        for item in objects
        if isinstance(item, dict)
        and str(item.get("label") or "").strip().lower() in PERSON_LABELS
        and item.get("incident_eligible") is not False
    ]
    if len(identities) == 1:
        event["primary_identity"] = dict(identities[0])
        if len(people) == 1:
            people[0]["identity"] = dict(identities[0])
    else:
        event.pop("primary_identity", None)
    return event


def apply_incident_identity(incident: dict[str, Any]) -> dict[str, Any]:
    for event in incident.get("events", []) or []:
        if isinstance(event, dict):
            apply_event_identity(event)
    identities = identity_summaries(incident.get("faces") or [])
    incident["identities"] = identities
    if len(identities) == 1:
        incident["primary_identity"] = dict(identities[0])
    else:
        incident.pop("primary_identity", None)
    return incident


def apply_incident_identities(
    incidents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for incident in incidents:
        apply_incident_identity(incident)
    return incidents

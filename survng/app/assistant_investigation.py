from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable


def _epoch(value: object) -> float:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _labels(incident: dict[str, Any]) -> set[str]:
    return {
        str(label).strip().lower()
        for label in incident.get("labels") or []
        if str(label).strip()
    }


def _face_keys(
    incident: dict[str, Any],
    statuses: set[str],
) -> set[tuple[int, str]]:
    result: set[tuple[int, str]] = set()
    for face in incident.get("faces") or []:
        if not isinstance(face, dict) or str(face.get("status")) not in statuses:
            continue
        name = str(face.get("name") or "").strip().lower()
        identity_id = int(face.get("identity_id") or 0)
        if name and name != "unknown":
            result.add((identity_id, name))
    return result


def correlate_incident_timeline(
    anchor: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    *,
    object_label: str = "",
    face_name: str = "",
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Rank bounded timeline candidates while keeping weak context explicitly weak."""
    anchor_event_id = int(anchor.get("representative_event_id") or 0)
    anchor_epoch = _epoch(anchor.get("start_at"))
    wanted_label = object_label.strip().lower()
    target_labels = {wanted_label} if wanted_label else _labels(anchor)
    wanted_face = face_name.strip().lower()
    anchor_confirmed = _face_keys(anchor, {"confirmed"})
    anchor_possible = _face_keys(anchor, {"possible"})
    if wanted_face:
        anchor_confirmed = {(0, wanted_face)}
        anchor_possible = set()

    ranked: list[tuple[float, dict[str, Any]]] = []
    for incident in candidates:
        event_id = int(incident.get("representative_event_id") or 0)
        if event_id <= 0 or event_id == anchor_event_id:
            continue
        confirmed = _face_keys(incident, {"confirmed"})
        possible = _face_keys(incident, {"possible"})
        if wanted_face:
            confirmed = {(0, name) for _identity_id, name in confirmed if name == wanted_face}
            possible = {(0, name) for _identity_id, name in possible if name == wanted_face}
        common_confirmed = anchor_confirmed & confirmed
        common_possible = (
            anchor_confirmed & possible
            or anchor_possible & confirmed
            or anchor_possible & possible
        )
        common_labels = target_labels & _labels(incident)
        delta_seconds = abs(_epoch(incident.get("start_at")) - anchor_epoch)
        if common_confirmed:
            strength = "confirmed_identity"
            score = 1.0
            reasons = [
                f"Confirmed face match: {sorted(common_confirmed)[0][1]}"
            ]
        elif common_possible:
            strength = "possible_identity"
            score = 0.78
            reasons = [
                f"Possible face match: {sorted(common_possible)[0][1]}"
            ]
        elif common_labels:
            strength = "context_candidate"
            # A shared class and nearby time can aid investigation but cannot
            # establish identity, especially for common people and vehicles.
            score = max(0.25, 0.62 - min(delta_seconds, 3600) / 10000)
            reasons = [
                f"Nearby {sorted(common_labels)[0]} detection; identity is not established"
            ]
        else:
            continue
        ranked.append((score, {
            "incident": incident,
            "event_id": event_id,
            "camera_id": str(incident.get("camera_id") or ""),
            "start_at": incident.get("start_at"),
            "seconds_from_anchor": round(
                _epoch(incident.get("start_at")) - anchor_epoch,
                1,
            ),
            "match_strength": strength,
            "confidence": round(score, 3),
            "reasons": reasons,
        }))
    strongest = sorted(
        ranked,
        key=lambda item: (-item[0], abs(item[1]["seconds_from_anchor"])),
    )[:max(1, min(int(limit), 24))]
    return [
        item
        for _score, item in sorted(
            strongest,
            key=lambda ranked_item: _epoch(ranked_item[1]["start_at"]),
        )
    ]

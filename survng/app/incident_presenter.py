"""Pure incident and recording presentation helpers.

These functions translate persisted event rows into stable public incident
payloads. They deliberately own no database, camera, or process lifecycle
state, so API routers and assistant services can reuse them without depending
on the application composition module.
"""

from __future__ import annotations

import json
import math

from .incident_utils import (
    DEFAULT_INCIDENT_GAP_SECONDS,
    event_epoch,
    incident_event_groups,
    stable_incident_id,
    stable_incident_key,
)

def _event_row(row: dict) -> dict:
    event = dict(row)
    event["snapshot_path"] = "available" if event.get("snapshot_path") else ""
    event["recording_path"] = "available" if event.get("recording_path") else ""
    try:
        objects = json.loads(event.pop("objects_json", "[]") or "[]")
    except (json.JSONDecodeError, TypeError):
        objects = []
    if not isinstance(objects, list):
        objects = []
    objects = [item for item in objects if isinstance(item, dict)]
    qualification_entry = next(
        (
            item.get("motion_qualification")
            for item in reversed(objects)
            if item.get("status") == "motion_qualification"
            and isinstance(item.get("motion_qualification"), dict)
        ),
        None,
    )
    raw_trigger_source = str(
        (qualification_entry or {}).get("trigger_source")
        or event.get("topic")
        or "camera"
    ).lower()
    event["trigger_source"] = (
        "ema"
        if raw_trigger_source in {"adaptive", "visual_backup", "adaptive/visual_backup"}
        else "camera"
    )
    tracking_entry = next(
        (
            item.get("object_tracking")
            for item in reversed(objects)
            if item.get("status") == "object_tracking"
            and isinstance(item.get("object_tracking"), dict)
        ),
        None,
    )
    event["object_tracking"] = tracking_entry

    def positive_confidence(item: dict) -> bool:
        try:
            confidence = float(item.get("confidence") or 0)
            return math.isfinite(confidence) and confidence > 0
        except (TypeError, ValueError):
            return False

    detected_objects = [
        item for item in objects
        if item.get("label")
        and positive_confidence(item)
        and item.get("incident_eligible") is not False
    ]
    tracked_objects = (
        [item for item in tracking_entry.get("tracks", []) if isinstance(item, dict)]
        if isinstance(tracking_entry, dict)
        else []
    )
    event["objects"] = objects
    event["has_objects"] = bool(detected_objects)
    event["labels"] = sorted({
        str(item["label"])
        for item in [*detected_objects, *tracked_objects]
        if item.get("label")
    })
    event["zones"] = sorted({
        str(zone_name)
        for item in [*detected_objects, *tracked_objects]
        for zone_name in (
            item.get("zones", [])
            if isinstance(item.get("zones", []), list)
            else []
        )
        if zone_name
    })
    return event


def _best_incident_event(events: list[dict]) -> dict:
    object_events = [event for event in events if event.get("has_objects")]
    candidates = object_events or events

    def score(event: dict) -> tuple[float, int]:
        objects = event.get("objects") or []
        confidences: list[float] = []
        for item in objects:
            if not isinstance(item, dict):
                continue
            try:
                confidence = float(item.get("confidence") or 0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(confidence):
                confidences.append(confidence)
        best_confidence = max(confidences, default=0.0)
        return (best_confidence, int(event.get("id") or 0))

    return max(candidates, key=score)


def _incident_rows(rows: list[dict], gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS) -> list[dict]:
    return [
        _incident_row(camera_id, events)
        for camera_id, events in incident_event_groups(rows, gap_seconds)
    ]


def _incident_event_payload(event: dict) -> dict:
    payload = dict(event)
    payload.pop("topic", None)
    payload.pop("message", None)
    payload["objects"] = [
        {
            key: item[key]
            for key in (
                "label",
                "confidence",
                "box",
                "zones",
                "mask_polygon",
                "detection_frame_width",
                "detection_frame_height",
                "incident_eligible",
                "temporal_consensus",
                "temporal_sample_offset_seconds",
                "temporal_observations",
                "temporal_track_observations",
                "temporal_incident_observations",
                "temporal_required_observations",
                "temporal_samples",
                "temporal_peak_confidence",
                "temporal_label_votes",
                "temporal_center_displacement_ratio",
                "temporal_center_path_ratio",
                "temporal_first_observation_offset_seconds",
                "temporal_last_observation_offset_seconds",
                "temporal_newly_appeared",
                "motion_correlated",
                "motion_correlation",
                "motion_correlation_threshold",
                "motion_temporal_evidence_available",
                "track_id",
                "track_state",
                "track_observations",
            )
            if key in item
        }
        for item in payload.get("objects", [])
        if isinstance(item, dict) and item.get("label")
    ]
    payload["object_tracking"] = event.get("object_tracking")
    return payload


def _incident_list_payload(incident: dict) -> dict:
    """Return the media-card data without expensive investigation details."""
    payload = dict(incident)
    representative_id = int(payload.get("representative_event_id") or 0)
    representative_tracking = payload.get("object_tracking")
    payload.pop("object_tracking", None)
    payload.pop("motion_observations", None)
    payload.pop("faces", None)

    def compact_objects(objects: object) -> list[dict]:
        if not isinstance(objects, list):
            return []
        return [
            {
                key: item[key]
                for key in (
                    "label",
                    "confidence",
                    "box",
                    "zones",
                    "mask_polygon",
                    "detection_frame_width",
                    "detection_frame_height",
                    "incident_eligible",
                    "track_id",
                    "track_state",
                )
                if key in item
            }
            for item in objects
            if isinstance(item, dict) and item.get("label")
        ]

    payload["objects"] = compact_objects(payload.get("objects"))

    def compact_tracking_dimensions(tracking: object) -> dict[str, int] | None:
        if not isinstance(tracking, dict):
            return None
        try:
            width = int(tracking.get("frame_width") or 0)
            height = int(tracking.get("frame_height") or 0)
        except (TypeError, ValueError, OverflowError):
            return None
        if width <= 0 or height <= 0:
            return None
        return {"frame_width": width, "frame_height": height}

    tracking_dimensions = compact_tracking_dimensions(representative_tracking)
    if tracking_dimensions:
        # The list view needs only the coordinate plane, not tracks or samples.
        # Supplying it with the first payload prevents annotations from briefly
        # scaling against the progressively resized thumbnail dimensions.
        payload["object_tracking"] = tracking_dimensions
    compact_events: list[dict] = []
    for event in payload.get("events", []):
        event_id = int(event.get("id") or 0)
        compact_event = {
            key: event[key]
            for key in (
                "id",
                "camera_id",
                "kind",
                "created_at",
                "has_objects",
                "labels",
                "zones",
                "trigger_source",
            )
            if key in event
        }
        compact_event["objects"] = (
            compact_objects(event.get("objects")) if event_id == representative_id else []
        )
        event_tracking_dimensions = compact_tracking_dimensions(event.get("object_tracking"))
        if event_tracking_dimensions:
            compact_event["object_tracking"] = event_tracking_dimensions
        compact_events.append(compact_event)
    payload["events"] = compact_events
    return payload


def _recording_grid_incident_payload(incident: dict) -> dict:
    """Return only fields used by Recording History's timeline and thumbnail rail."""
    return {
        key: incident[key]
        for key in (
            "id",
            "representative_event_id",
            "camera_id",
            "snapshot_path",
            "start_epoch",
            "last_epoch",
            "has_objects",
            "labels",
        )
        if key in incident
    }


def _incident_row(camera_id: str, events: list[dict]) -> dict:
    ordered = sorted(events, key=event_epoch)
    first = ordered[0]
    last = ordered[-1]
    representative = _best_incident_event(ordered)
    representative_payload = _incident_event_payload(representative)
    labels = sorted({label for event in ordered for label in event.get("labels", [])})
    zones = sorted({zone for event in ordered for zone in event.get("zones", [])})
    start_epoch = event_epoch(first)
    motion_observations = sorted(
        [
            observation
            for event in ordered
            for observation in event.get("motion_observations", [])
            if isinstance(observation, dict)
        ],
        key=event_epoch,
    )
    tracking_updates = [
        {"created_at": str(tracking.get("updated_at") or "")}
        for event in ordered
        if isinstance((tracking := event.get("object_tracking")), dict)
        and tracking.get("updated_at")
        and tracking.get("tracks")
    ]
    final_item = max([last, *motion_observations, *tracking_updates], key=event_epoch)
    last_epoch = event_epoch(final_item)
    object_count = sum(1 for event in ordered if event.get("has_objects"))
    incident = {
        **representative_payload,
        "id": stable_incident_id(camera_id, first.get("id")),
        "incident_id": stable_incident_key(camera_id, first.get("id")),
        "representative_event_id": representative.get("id"),
        "camera_id": camera_id,
        "kind": "motion",
        "created_at": representative.get("created_at"),
        "start_at": first.get("created_at"),
        "end_at": final_item.get("created_at"),
        "start_epoch": start_epoch,
        "last_epoch": last_epoch,
        "duration_seconds": max(0.0, last_epoch - start_epoch),
        "event_count": len(ordered),
        "motion_observation_count": len(motion_observations),
        "object_event_count": object_count,
        # An incident's trigger is the source that opened it, even when later
        # events from another source are grouped into the same incident.
        "trigger_source": first.get("trigger_source", "camera"),
        "has_objects": bool(labels),
        "labels": labels,
        "zones": zones,
        "events": [_incident_event_payload(event) for event in reversed(ordered)],
        "motion_observations": list(reversed(motion_observations)),
        # Top-level incident media and objects come from the representative
        # event, so its tracking metadata must use the same temporal context.
        "object_tracking": representative.get("object_tracking"),
    }
    return incident


def _recording_event_row(event: dict, recordings: list[dict]) -> dict:
    event = _event_row(event)
    created_epoch = event_epoch(event)
    first_start = float(recordings[0]["start_epoch"])
    event["timeline_offset"] = max(0.0, created_epoch - first_start)
    return event


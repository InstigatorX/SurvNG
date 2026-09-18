"""Pure projection rules between native incident truth and cover presentation."""
from __future__ import annotations

from copy import deepcopy


PRESENTATION_FIELDS = frozenset({
    "box",
    "mask_polygon",
    "confidence",
    "detection_frame_width",
    "detection_frame_height",
    "snapshot_visible",
    "snapshot_source",
    "snapshot_captured_at",
    "snapshot_detection_confidence",
    "frame_source",
    "frame_captured_at_epoch",
    "native_alignment",
    "native_cover_verified",
    "box_provenance",
    "verification",
    "native_cover_score",
    "snapshot_quality_score",
    "snapshot_subject_area_ratio",
    "snapshot_edge_clearance_ratio",
    "snapshot_primary_subject",
    "temporal_sample_offset_seconds",
})


def _identity(item):
    if not isinstance(item, dict) or not item.get("label"):
        return None
    if item.get("track_id") is not None:
        return ("track", item.get("track_id"))
    if item.get("native_identity"):
        return ("native", str(item["native_identity"]))
    if item.get("native_track_id") is not None:
        return (
            "native_track",
            str(item.get("label") or ""),
            item.get("native_track_id"),
        )
    return None


def _split(objects, *, drop_tracking=False):
    labels = []
    metadata = []
    for item in objects or []:
        if not isinstance(item, dict):
            continue
        if item.get("label"):
            labels.append(item)
            continue
        if drop_tracking and item.get("status") == "object_tracking":
            continue
        metadata.append(item)
    return labels, metadata


def _index(labels):
    indexed = {}
    for item in labels:
        key = _identity(item)
        if key is not None and key not in indexed:
            indexed[key] = item
    return indexed


def merge_inventory_objects(existing_objects, inventory_objects):
    """Update durable incident contents without moving current cover geometry."""
    existing_labels, metadata = _split(existing_objects, drop_tracking=True)
    existing_by_id = _index(existing_labels)
    matched = set()
    merged = []

    for incoming in inventory_objects or []:
        if not isinstance(incoming, dict) or not incoming.get("label"):
            continue
        item = deepcopy(incoming)
        existing = existing_by_id.get(_identity(incoming))
        if existing is not None:
            matched.add(id(existing))
            if existing.get("snapshot_visible") is not False:
                for field in PRESENTATION_FIELDS:
                    if field in existing:
                        item[field] = deepcopy(existing[field])
        else:
            item["snapshot_visible"] = False
        merged.append(item)

    merged.extend(
        dict(deepcopy(item), snapshot_visible=False)
        for item in existing_labels
        if id(item) not in matched
    )
    return [*merged, *deepcopy(metadata)]


def merge_cover_objects(existing_objects, cover_objects):
    """Apply one cover's raster-specific annotations to durable incident truth."""
    existing_labels, metadata = _split(existing_objects)
    existing_by_id = _index(existing_labels)
    cover_by_id = _index(cover_objects or [])
    used_cover = set()
    merged = []

    for existing in existing_labels:
        item = deepcopy(existing)
        cover = cover_by_id.get(_identity(existing))
        if cover is None:
            item["snapshot_visible"] = False
        else:
            used_cover.add(id(cover))
            for field in PRESENTATION_FIELDS:
                if field in cover:
                    item[field] = deepcopy(cover[field])
            item["snapshot_visible"] = cover.get("snapshot_visible") is not False
        merged.append(item)

    for cover in cover_objects or []:
        if not isinstance(cover, dict) or not cover.get("label"):
            continue
        if id(cover) in used_cover:
            continue
        # Defensive compatibility for old incidents that predate inventory IDs.
        merged.append(deepcopy(cover))

    return [*merged, *deepcopy(metadata)]

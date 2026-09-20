"""Episode identity from concurrent evidence and motion-continuity stitching.

Track ID cardinality is a poor census in fragmented scenes. Peak concurrency
estimates how many instances of a label were present together. Motion-continuity
stitching merges sequential fragment tracks that never co-occur into stable
episode identities for cover/semantic presentation.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def _history_samples(track: dict[str, Any]) -> list[tuple[float, float, float, float, float]]:
    samples = []
    for point in track.get("box_history") or ():
        if not point or len(point) < 5:
            continue
        try:
            epoch, x1, y1, x2, y2 = (float(point[0]), float(point[1]), float(point[2]),
                                     float(point[3]), float(point[4]))
        except (TypeError, ValueError, IndexError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        samples.append((epoch, x1, y1, x2, y2))
    return samples


def _track_span(samples: list[tuple[float, float, float, float, float]]) -> tuple[float, float] | None:
    if not samples:
        return None
    return samples[0][0], samples[-1][0]


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5


def _center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    acx, acy = _center(a)
    bcx, bcy = _center(b)
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def peak_concurrent_count(
    tracks: list[dict[str, Any]],
    *,
    label: str | None = None,
    bin_seconds: float = 1.0,
) -> int:
    """Maximum number of same-label tracks active in any time bin."""
    if bin_seconds <= 0:
        bin_seconds = 1.0
    bins: dict[int, set[Any]] = {}
    for track in tracks or ():
        if not isinstance(track, dict):
            continue
        if label is not None and track.get("label") != label:
            continue
        track_id = track.get("track_id")
        if track_id is None:
            continue
        for sample in _history_samples(track):
            key = int(sample[0] / bin_seconds)
            bins.setdefault(key, set()).add(track_id)
    return max((len(ids) for ids in bins.values()), default=0)


def episode_label_counts(
    tracks: list[dict[str, Any]],
    *,
    bin_seconds: float = 1.0,
) -> dict[str, int]:
    labels = sorted({
        str(track.get("label"))
        for track in tracks or ()
        if isinstance(track, dict) and track.get("label")
    })
    return {
        name: peak_concurrent_count(tracks, label=name, bin_seconds=bin_seconds)
        for name in labels
    }


def _co_occur(
    left: list[tuple[float, float, float, float, float]],
    right: list[tuple[float, float, float, float, float]],
    *,
    bin_seconds: float,
) -> bool:
    left_bins = {int(sample[0] / bin_seconds) for sample in left}
    right_bins = {int(sample[0] / bin_seconds) for sample in right}
    return bool(left_bins & right_bins)


def _continuity_score(
    earlier: list[tuple[float, float, float, float, float]],
    later: list[tuple[float, float, float, float, float]],
    *,
    frame_width: float,
    frame_height: float,
) -> float | None:
    if not earlier or not later:
        return None
    a = earlier[-1][1:]
    b = later[0][1:]
    iou = _iou(a, b)
    diagonal = max(1.0, (frame_width ** 2 + frame_height ** 2) ** 0.5)
    distance = _center_distance(a, b) / diagonal
    # Higher is better: prefer overlap, then near centers.
    return iou * 2.0 + max(0.0, 1.0 - distance)


def stitch_motion_identities(
    tracks: list[dict[str, Any]],
    *,
    max_gap_seconds: float = 4.0,
    bin_seconds: float = 1.0,
    min_iou: float = 0.05,
    max_center_distance_ratio: float = 0.35,
    frame_width: float | None = None,
    frame_height: float | None = None,
) -> list[dict[str, Any]]:
    """Merge sequential non-co-occurring fragment tracks into episode identities.

    Returns one record per identity:
    ``{episode_identity, label, track_ids, peak_confidence, representative_track_id}``.
    Tracks that co-occur in the same time bin never share an identity.
    """
    prepared = []
    for track in tracks or ():
        if not isinstance(track, dict) or track.get("track_id") is None or not track.get("label"):
            continue
        samples = _history_samples(track)
        span = _track_span(samples)
        if span is None:
            # No history: identity stays singleton.
            prepared.append({
                "track": track,
                "track_id": track["track_id"],
                "label": track["label"],
                "samples": [],
                "first": None,
                "last": None,
            })
            continue
        prepared.append({
            "track": track,
            "track_id": track["track_id"],
            "label": track["label"],
            "samples": samples,
            "first": span[0],
            "last": span[1],
        })
    if not prepared:
        return []

    width = float(frame_width or next(
        (track["track"].get("detection_frame_width") for track in prepared
         if track["track"].get("detection_frame_width")),
        1,
    ) or 1)
    height = float(frame_height or next(
        (track["track"].get("detection_frame_height") for track in prepared
         if track["track"].get("detection_frame_height")),
        1,
    ) or 1)
    diagonal = max(1.0, (width ** 2 + height ** 2) ** 0.5)

    parent = {item["track_id"]: item["track_id"] for item in prepared}

    def find(track_id):
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(left_id, right_id):
        left, right = find(left_id), find(right_id)
        if left != right:
            parent[right] = left

    by_id = {item["track_id"]: item for item in prepared}
    candidates = []
    for index, left in enumerate(prepared):
        if left["first"] is None:
            continue
        for right in prepared[index + 1:]:
            if right["first"] is None:
                continue
            if left["label"] != right["label"]:
                continue
            if _co_occur(left["samples"], right["samples"], bin_seconds=bin_seconds):
                continue
            if left["last"] <= right["first"]:
                earlier, later = left, right
            elif right["last"] <= left["first"]:
                earlier, later = right, left
            else:
                continue
            gap = later["first"] - earlier["last"]
            if gap > max_gap_seconds:
                continue
            a = earlier["samples"][-1][1:]
            b = later["samples"][0][1:]
            iou = _iou(a, b)
            distance_ratio = _center_distance(a, b) / diagonal
            if iou < min_iou and distance_ratio > max_center_distance_ratio:
                continue
            score = _continuity_score(
                earlier["samples"], later["samples"],
                frame_width=width, frame_height=height,
            )
            if score is None:
                continue
            candidates.append((score, earlier["track_id"], later["track_id"]))

    for _score, left_id, right_id in sorted(candidates, key=lambda item: item[0], reverse=True):
        # Refuse merges that would put co-occurring tracks under one identity.
        left_members = [by_id[tid] for tid in by_id if find(tid) == find(left_id)]
        right_members = [by_id[tid] for tid in by_id if find(tid) == find(right_id)]
        conflict = False
        for left_member in left_members:
            for right_member in right_members:
                if left_member["samples"] and right_member["samples"] and _co_occur(
                    left_member["samples"], right_member["samples"], bin_seconds=bin_seconds,
                ):
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        union(left_id, right_id)

    groups: dict[Any, list[dict[str, Any]]] = {}
    for item in prepared:
        groups.setdefault(find(item["track_id"]), []).append(item)

    identities = []
    for root, members in groups.items():
        label = members[0]["label"]
        track_ids = sorted(member["track_id"] for member in members)
        best = max(
            members,
            key=lambda member: float(
                member["track"].get("max_confidence")
                or member["track"].get("confidence")
                or 0.0
            ),
        )
        identities.append({
            "episode_identity": f"{label}:{root}",
            "label": label,
            "track_ids": track_ids,
            "representative_track_id": best["track_id"],
            "peak_confidence": float(
                best["track"].get("max_confidence")
                or best["track"].get("confidence")
                or 0.0
            ),
        })
    identities.sort(key=lambda item: (item["label"], item["episode_identity"]))
    return identities


def annotate_tracks_with_episode_identities(
    tracks: list[dict[str, Any]],
    *,
    frame_width: float | None = None,
    frame_height: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Return tracks stamped with episode_identity plus identity/count summaries."""
    identities = stitch_motion_identities(
        tracks,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    by_track = {
        track_id: identity
        for identity in identities
        for track_id in identity["track_ids"]
    }
    annotated = []
    for track in tracks or ():
        item = deepcopy(track) if isinstance(track, dict) else track
        if not isinstance(item, dict):
            continue
        identity = by_track.get(item.get("track_id"))
        if identity is not None:
            item["episode_identity"] = identity["episode_identity"]
            item["episode_identity_representative"] = (
                item.get("track_id") == identity["representative_track_id"]
            )
        elif item.get("track_id") is not None and item.get("label"):
            item["episode_identity"] = f"{item['label']}:{item['track_id']}"
            item["episode_identity_representative"] = True
        annotated.append(item)
    counts = episode_label_counts(annotated)
    return annotated, identities, counts


def annotate_objects_with_episode_identities(
    objects: list[dict[str, Any]],
    tracks: list[dict[str, Any]] | None = None,
    *,
    frame_width: float | None = None,
    frame_height: float | None = None,
) -> list[dict[str, Any]]:
    """Stamp labeled objects with episode_identity derived from track stitching."""
    source_tracks = tracks
    if source_tracks is None:
        source_tracks = [
            item for item in objects or ()
            if isinstance(item, dict) and item.get("label") and item.get("track_id") is not None
        ]
    annotated_tracks, _identities, _counts = annotate_tracks_with_episode_identities(
        source_tracks,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    by_track = {
        item.get("track_id"): item
        for item in annotated_tracks
        if isinstance(item, dict) and item.get("track_id") is not None
    }
    result = []
    for item in objects or ():
        obj = deepcopy(item) if isinstance(item, dict) else item
        if not isinstance(obj, dict):
            continue
        track = by_track.get(obj.get("track_id"))
        if track is not None and track.get("episode_identity"):
            obj["episode_identity"] = track["episode_identity"]
            obj["episode_identity_representative"] = track.get(
                "episode_identity_representative"
            )
        result.append(obj)
    return result


def best_objects_by_episode_identity(
    objects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the highest-confidence object per episode identity (or track/label)."""
    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in objects or ():
        if not isinstance(item, dict) or not item.get("label"):
            continue
        key = str(
            item.get("episode_identity")
            or (
                f"track:{item.get('track_id')}"
                if item.get("track_id") is not None
                else f"row:{id(item)}"
            )
        )
        try:
            confidence = float(item.get("confidence") or item.get("max_confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        visible_bonus = 1.0 if item.get("snapshot_visible") is True else 0.0
        score = confidence + visible_bonus
        previous = best.get(key)
        if previous is None:
            best[key] = item
            order.append(key)
            continue
        try:
            previous_confidence = float(
                previous.get("confidence") or previous.get("max_confidence") or 0.0
            )
        except (TypeError, ValueError):
            previous_confidence = 0.0
        previous_score = previous_confidence + (
            1.0 if previous.get("snapshot_visible") is True else 0.0
        )
        if score > previous_score:
            best[key] = item
    return [best[key] for key in order]

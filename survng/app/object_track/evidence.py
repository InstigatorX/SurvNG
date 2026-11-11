"""Evidence gallery images nominated by object tracking.

The detection pass fills the gallery from the few frames it happened to
sample. Tracking follows the subject through the whole incident, so it can
nominate later, different views. Only metadata is kept while tracking runs;
full-detail frames are decoded once, after the session completes.

These images are evidence only. They never change the cover, admission, or
alert eligibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..evidence_gallery import MAX_GALLERY_IMAGES, TRACKING_SOURCE, _box as _normalized_box, gallery_image
from ..video_frames import VideoFrameReference
from ..visual_quality import image_quality
from .geometry import Box, _box, _confidence

# One nomination per subject per window keeps memory bounded on long sessions.
VIEW_WINDOW_SECONDS = 2.0
MAX_NOMINATED_VIEWS = 256
# Two views of one subject closer than this in time show the same moment.
MINIMUM_VIEW_GAP_SECONDS = 1.0
# Same place and similar size is the same picture, however far apart in time.
SAME_VIEW_CENTER_DISTANCE = 0.05
SAME_VIEW_AREA_RATIO = (0.6, 1.6)
MINIMUM_EDGE_CLEARANCE_RATIO = 0.01
_COPIED_OBJECT_KEYS = (
    "label", "confidence", "track_id", "zones", "zone_matches", "zone_eligible",
    "zone_admission_reason", "incident_eligible", "confidence_eligible",
)


@dataclass(frozen=True, slots=True)
class TrackedEvidenceView:
    """One nominated moment, measured on the tracking frame."""

    captured_at: float
    track_id: int
    box: Box
    frame_width: int
    frame_height: int
    area_ratio: float
    edge_clearance_ratio: float
    quality_score: float
    confidence: float
    incident_eligible: bool
    frame_reference: VideoFrameReference
    objects: tuple[dict[str, Any], ...]

    @property
    def rank(self) -> tuple[int, int, float, float]:
        return (
            int(self.incident_eligible),
            int(self.edge_clearance_ratio >= MINIMUM_EDGE_CLEARANCE_RATIO),
            # Resolved detail, as in recorded cover selection.
            math.sqrt(self.area_ratio) * self.quality_score,
            self.confidence,
        )

    def metadata(self, camera_id: str, width: int, height: int) -> dict[str, Any]:
        """Gallery payload with boxes scaled to the retained image."""
        scale_x = width / self.frame_width
        scale_y = height / self.frame_height
        objects = []
        for item in self.objects:
            x1, y1, x2, y2 = item["box"]
            objects.append({
                **{key: item[key] for key in _COPIED_OBJECT_KEYS if key in item},
                "box": {"x1": x1 * scale_x, "y1": y1 * scale_y, "x2": x2 * scale_x, "y2": y2 * scale_y},
                "detection_frame_width": int(width),
                "detection_frame_height": int(height),
                "frame_source": "recorded_main",
                "captured_at_epoch": self.captured_at,
                "evidence_source": TRACKING_SOURCE,
            })
        return {
            "camera_id": camera_id, "captured_epoch": self.captured_at, "objects": objects,
            "width": int(width), "height": int(height), "frame_source": "recorded_main",
            "frame_timestamp_exact": True,
            "recording_path": str(self.frame_reference.source_path), "role": "gallery",
            "source": TRACKING_SOURCE,
        }

    def comparison(self, camera_id: str) -> dict[str, Any]:
        return self.metadata(camera_id, self.frame_width, self.frame_height)


def _evidence_subject(item: dict[str, Any]) -> bool:
    # Mirrors the detection gallery: evidence selection, not an admission gate.
    return bool(
        item.get("label")
        and str(item.get("label")).strip().lower() != "face"
        and item.get("track_state") == "confirmed"
        and item.get("track_id") is not None
        and item.get("activity_role") != "scene_context"
        and item.get("zone_admission_reason") != "ignored_zone"
        and "ignored_zone" not in (item.get("alert_reasons") or [])
    )


class TrackedEvidenceCollector:
    """Keep the best view of each tracked subject per time window."""

    def __init__(self) -> None:
        self._views: dict[tuple[int, int], TrackedEvidenceView] = {}

    def reset(self) -> None:
        self._views.clear()

    def views(self) -> list[TrackedEvidenceView]:
        return list(self._views.values())

    def consider(
        self,
        frame: np.ndarray,
        captured_at: float,
        tracked_objects: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
        frame_reference: VideoFrameReference | None,
    ) -> None:
        # Without an exact reference the full-detail frame cannot be re-read.
        if (
            frame_reference is None
            or not frame_reference.exact
            or frame is None
            or not frame.size
            or frame_width <= 0
            or frame_height <= 0
        ):
            return
        subjects: list[tuple[dict[str, Any], Box]] = []
        for item in tracked_objects:
            box = _box(item.get("box"))
            if box is None or not _evidence_subject(item):
                continue
            try:
                int(item["track_id"])
            except (TypeError, ValueError):
                continue
            subjects.append((item, box))
        if not subjects:
            return
        objects = tuple(
            {**item, "box": box, "track_id": int(item["track_id"])} for item, box in subjects
        )
        source_height, source_width = frame.shape[:2]
        scale_x = source_width / frame_width
        scale_y = source_height / frame_height
        frame_area = float(frame_width * frame_height)
        for item, box in subjects:
            x1, y1, x2, y2 = box
            crop = frame[
                max(0, min(source_height, int(np.floor(y1 * scale_y)))):
                max(0, min(source_height, int(np.ceil(y2 * scale_y)))),
                max(0, min(source_width, int(np.floor(x1 * scale_x)))):
                max(0, min(source_width, int(np.ceil(x2 * scale_x)))),
            ]
            view = TrackedEvidenceView(
                captured_at=float(captured_at),
                track_id=int(item["track_id"]),
                box=box,
                frame_width=int(frame_width),
                frame_height=int(frame_height),
                area_ratio=max(0.0, (x2 - x1) * (y2 - y1) / frame_area),
                edge_clearance_ratio=max(0.0, min(
                    x1 / frame_width, y1 / frame_height,
                    (frame_width - x2) / frame_width, (frame_height - y2) / frame_height,
                )),
                quality_score=float(image_quality(crop).score) if crop.size else 0.0,
                confidence=_confidence(item),
                incident_eligible=item.get("incident_eligible") is not False,
                frame_reference=frame_reference,
                objects=objects,
            )
            key = (view.track_id, int(view.captured_at // VIEW_WINDOW_SECONDS))
            current = self._views.get(key)
            if current is None or view.rank > current.rank:
                self._views[key] = view
        while len(self._views) > MAX_NOMINATED_VIEWS:
            del self._views[min(self._views, key=lambda item: self._views[item].rank)]


def _same_view(view: TrackedEvidenceView, camera_id: str, frame: dict[str, Any]) -> bool:
    if frame.get("camera_id") not in (None, camera_id):
        return False
    try:
        gap = abs(float(frame["captured_epoch"]) - view.captured_at)
    except (KeyError, TypeError, ValueError):
        return False
    # The same image, whichever subject nominated it.
    if gap < 0.1:
        return True
    subject = next(o for o in view.comparison(camera_id)["objects"] if o.get("track_id") == view.track_id)
    box = _normalized_box(subject)
    if box is None:
        return False
    area = (box[2] - box[0]) * (box[3] - box[1])
    for other in frame.get("objects") or []:
        other_box = _normalized_box(other) if isinstance(other, dict) else None
        if other_box is None:
            continue
        other_track = other.get("track_id")
        if other_track is not None and other_track != view.track_id:
            # A different tracked subject standing in the same place is new evidence.
            continue
        if other_track == view.track_id and gap < MINIMUM_VIEW_GAP_SECONDS:
            return True
        distance = math.hypot(
            (box[0] + box[2] - other_box[0] - other_box[2]) / 2,
            (box[1] + box[3] - other_box[1] - other_box[3]) / 2,
        )
        other_area = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])
        low, high = SAME_VIEW_AREA_RATIO
        if distance < SAME_VIEW_CENTER_DISTANCE and low < area / other_area < high:
            return True
    return False


def select_evidence_views(
    views: list[TrackedEvidenceView],
    camera_id: str,
    previous: list[dict[str, Any]],
    limit: int,
) -> list[TrackedEvidenceView]:
    """Best distinct views, in the order they happened."""
    retained = list(previous)
    chosen: list[TrackedEvidenceView] = []
    for view in sorted(views, key=lambda item: (item.rank, -item.captured_at), reverse=True):
        if len(chosen) >= limit:
            break
        if any(_same_view(view, camera_id, frame) for frame in retained):
            continue
        chosen.append(view)
        retained.append(view.comparison(camera_id))
    return sorted(chosen, key=lambda item: item.captured_at)


def save_tracked_evidence(
    views: list[TrackedEvidenceView],
    *,
    event_id: int,
    camera_id: str,
    capacity: Callable[[int], dict[str, Any]],
    writer: Callable[..., int],
    frame_provider: Callable[[float, int, VideoFrameReference | None], np.ndarray | None],
    snapshot_writer: Callable[..., str],
    known_frames: list[dict[str, Any]],
    stopped: Callable[[], bool],
) -> dict[str, Any]:
    """Decode, encode and retain one image at a time; report what happened."""
    def outcome(result: str, saved: int = 0, nominated: int = 0) -> dict[str, Any]:
        return {
            "evidence_images_result": result,
            "evidence_images_saved": saved,
            "evidence_images_nominated": nominated,
        }

    if not views:
        return outcome("no_tracked_views")
    request = capacity(event_id) or {}
    remaining = min(MAX_GALLERY_IMAGES, max(0, int(request.get("remaining", 0))))
    if remaining <= 0:
        return outcome("gallery_full")
    previous = [frame for frame in request.get("frames") or [] if isinstance(frame, dict)]
    chosen = select_evidence_views(views, camera_id, [*previous, *known_frames], remaining)
    if not chosen:
        return outcome("no_distinct_view")
    saved = 0
    for view in chosen:
        if stopped():
            return outcome("stopped", saved, len(chosen))
        frame = frame_provider(view.captured_at, view.frame_width, view.frame_reference)
        if frame is None or not frame.size:
            continue
        height, width = frame.shape[:2]
        image = gallery_image(frame, view.metadata(camera_id, width, height))
        del frame
        if image is None:
            continue
        saved += int(writer(event_id, [image], snapshot_writer) or 0)
    return outcome("saved" if saved else "not_retained", saved, len(chosen))

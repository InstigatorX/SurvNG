"""Native DL Streamer metadata normalization and provenance reconciliation."""
from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
import json
import threading
import time
from typing import Any

def _gva_object_label(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    detection = item.get("detection") if isinstance(item.get("detection"), dict) else item
    return str(detection.get("label") or item.get("label") or "").strip()


def _normalize_gva_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    from survng.native_spatial import ROI_LABEL

    objects: list[dict[str, Any]] = []
    for item in payload.get("objects") or ():
        if not isinstance(item, dict):
            continue
        detection = item.get("detection") if isinstance(item.get("detection"), dict) else item
        label = _gva_object_label(item)
        if label == ROI_LABEL:
            continue
        try:
            confidence = float(detection.get("confidence", item.get("confidence", 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        x = item.get("x", item.get("x_min"))
        y = item.get("y", item.get("y_min"))
        width = item.get("w", item.get("width"))
        height = item.get("h", item.get("height"))
        if None in (x, y, width, height):
            box = item.get("box")
            if isinstance(box, dict):
                x1, y1, x2, y2 = box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")
            else:
                continue
        else:
            try:
                x1 = float(x)
                y1 = float(y)
                x2 = x1 + float(width)
                y2 = y1 + float(height)
            except (TypeError, ValueError):
                continue
        try:
            normalized = {
                "label": label or "object",
                "confidence": confidence,
                "box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
            }
            # Native IDs are authoritative only within a capture session.
            native_track_id = item.get("id", item.get("object_id"))
            if native_track_id is None:
                native_track_id = detection.get("object_id")
            if native_track_id is not None:
                native_track_id = int(native_track_id)
                if native_track_id >= 0:
                    normalized["native_track_id"] = native_track_id
            if "zone_violations" in item:
                ids = item["zone_violations"]
                if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
                    raise ValueError("invalid native zone membership")
                normalized["native_zone_ids"] = list(ids)
            objects.append(normalized)
        except (TypeError, ValueError):
            continue
    return objects


def _packed_gray(pixels: bytes, width: int, height: int) -> bytes:
    expected = width * height
    if len(pixels) == expected:
        return pixels
    if height <= 0 or len(pixels) < expected:
        raise RuntimeError("GStreamer grayscale frame was truncated")
    stride = len(pixels) // height
    if stride < width:
        raise RuntimeError("GStreamer grayscale frame was truncated")
    return b"".join(pixels[row * stride : row * stride + width] for row in range(height))


class _NativeInferenceEvidence:
    """Bounded pre-tracker evidence, before the leaky output queue.

    gvadetect's no-block=false contract runs the first buffer and
    every inference-interval buffer thereafter, emitting buffers in order.
    ROI mode supplies exactly one region on every input, including sweeps.
    Count here, never at appsink (which drops buffers). Capture ROIs here too:
    gvatrack can append predictions even on frames with fresh detections.
    See DL Streamer inference_impl.cpp::TransformFrameIp and tracker.cpp::track.
    """

    def __init__(self, interval: int, *, tracking: bool = False) -> None:
        self.interval = interval
        self.tracking = tracking
        self.sequence = 0
        self.results = OrderedDict()
        self.lock = threading.Lock()
        self.invalid = 0
        self.last_pts = None
        self.identity_valid = True
        self.starts = OrderedDict()
        self.latencies_ms = deque(maxlen=100)

    def begin(self, pts):
        with self.lock:
            self.starts[pts] = time.monotonic()
            while len(self.starts) > 128:
                self.starts.popitem(last=False)

    def timing_status(self):
        with self.lock:
            values = sorted(self.latencies_ms)
        return {
            "native_detector_average_ms": round(sum(values) / len(values), 2) if values else None,
            "native_detector_p95_ms": round(values[min(len(values) - 1, int(len(values) * .95))], 2) if values else None,
            "native_detector_timing_samples": len(values),
        }

    def observe(self, buffer, caps, video_frame_type) -> None:
        self.sequence += 1
        fresh = (self.sequence - 1) % self.interval == 0
        with self.lock:
            started = self.starts.pop(buffer.pts, None)
            if fresh and started is not None:
                self.latencies_ms.append((time.monotonic() - started) * 1000)
        objects = []
        if self.last_pts is not None and buffer.pts <= self.last_pts:
            # PTS reuse can collide with metadata still queued after this probe.
            # Fail closed until pipeline recreation establishes a new identity.
            self.identity_valid = False
            self.invalid += 1
            with self.lock:
                self.results.clear()
        self.last_pts = buffer.pts
        try:
            if not self.identity_valid:
                result = ("unknown", [])
            elif fresh:
                from survng.native_spatial import ROI_LABEL
                for region in video_frame_type(buffer, caps=caps).regions():
                    if region.label() == ROI_LABEL:
                        continue
                    rect = region.rect()
                    objects.append({
                        "label": region.label(), "confidence": region.confidence(),
                        "box": {"x1": rect.x, "y1": rect.y,
                                "x2": rect.x + rect.w, "y2": rect.y + rect.h},
                    })
            if self.identity_valid:
                result = ("native_fresh_detection" if fresh else (
                    "native_tracked_prediction" if self.tracking else "unknown"
                ), objects)
        except Exception:
            # Do not turn an adapter failure into authoritative empty evidence.
            # Report the bounded counter in status; this frame cannot admit activity.
            self.invalid += 1
            result = ("unknown", [])
        with self.lock:
            self.results[buffer.pts] = result
            while len(self.results) > 128:
                self.results.popitem(last=False)

    def pop(self, pts):
        with self.lock:
            return self.results.pop(pts, ("unknown", []))


class _NativeReidEvidence:
    """Verify that gvainference attached tracker-compatible MARS embeddings."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.person_regions = 0
        self.valid = 0
        self.missing = 0
        self.last_missing_tensors: list[dict[str, object]] = []

    @staticmethod
    def _tracker_compatible(tensor) -> tuple[bool, dict[str, object]]:
        name = str(tensor.name() or "")
        layer_name = str(tensor.layer_name() or "")
        try:
            data = tensor.data()
            size = int(data.size) if data is not None else 0
        except Exception:
            size = 0
        compatible_name = (
            ("output" in layer_name and "inference_layer_name:output" in name)
            or ("features" in layer_name and "inference_layer_name:features" in name)
        )
        return size == 128 and compatible_name, {
            "name": name, "layer_name": layer_name, "size": size,
        }

    def observe(self, buffer, caps, video_frame_type) -> None:
        person_regions = valid = missing = 0
        last_missing: list[dict[str, object]] = []
        for region in video_frame_type(buffer, caps=caps).regions():
            if region.label().strip().lower() != "person":
                continue
            person_regions += 1
            tensors = []
            found = False
            for tensor in region.tensors():
                compatible, descriptor = self._tracker_compatible(tensor)
                tensors.append(descriptor)
                found = found or compatible
            if found:
                valid += 1
            else:
                missing += 1
                last_missing = tensors[:8]
        with self.lock:
            self.person_regions += person_regions
            self.valid += valid
            self.missing += missing
            if last_missing:
                self.last_missing_tensors = last_missing

    def status(self) -> dict[str, object]:
        with self.lock:
            person_regions = self.person_regions
            valid = self.valid
            missing = self.missing
            tensors = list(self.last_missing_tensors)
        if person_regions == 0:
            health = "waiting_for_person"
        elif missing:
            health = "missing_features"
        else:
            health = "healthy"
        return {
            "native_reid_feature_health": health,
            "native_reid_person_regions": person_regions,
            "native_reid_features_valid": valid,
            "native_reid_features_missing": missing,
            "native_reid_last_missing_tensors": tensors,
        }


def _filter_tracking_regions(buffer, caps, video_frame_type, allowed_classes):
    """Remove excluded ROI metadata before tracking, without mapping pixel memory."""
    from survng.native_spatial import ROI_LABEL
    from gi.repository import GstAnalytics, GLib
    frame = video_frame_type(buffer, caps=caps)
    regions = list(frame.regions())
    selected = [region for region in regions if region.label() != ROI_LABEL
                and (allowed_classes is None or region.label().strip().lower() in allowed_classes)]
    if len(selected) == len(regions):
        return
    kept = [(region.rect(), region.label(), region.confidence(), region.label_id()) for region in selected]
    for region in regions:
        frame.remove_region(region)
    # DL Streamer 2026 also stores detections in analytics relation metadata;
    # its public API has no individual-record removal. Rebuild the selected
    # bounding-box detections in both representations, retaining class IDs.
    meta = buffer.get_meta(GstAnalytics.relation_meta_api_get_type())
    if meta is not None and not buffer.remove_meta(meta):
        raise RuntimeError("could not replace tracking analytics metadata")
    for rect, label, confidence, label_id in kept:
        roi = frame.add_region(rect.x, rect.y, rect.w, rect.h, label, confidence)
        relation = GstAnalytics.buffer_get_analytics_relation_meta(buffer)
        quarks = [0] * (label_id + 1)
        scores = [0.0] * (label_id + 1)
        quarks[label_id], scores[label_id] = GLib.quark_from_string(label), confidence
        success, classification = relation.add_cls_mtd(scores, quarks)
        if not success or not relation.set_relation(GstAnalytics.RelTypes.RELATE_TO, roi.meta().id, classification.id):
            raise RuntimeError("could not retain tracking class identity")


def _detection_metadata(sample, video_frame_type, *, inference_sequence: int, gst_second: int,
                        clock_time_none: int, native_result=None, spatial_plan=None,
                        tracking_classes=None):
    """Read GstGVAJSONMeta; mapping a video buffer yields pixels, not JSON."""
    from survng.app.live_detections import DetectionSnapshot
    buffer = sample.get_buffer()
    if buffer.pts == clock_time_none:
        raise ValueError("inferred frame has no source PTS")
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    messages = video_frame_type(buffer, caps=caps).messages()
    if len(messages) != 1:
        raise ValueError("expected one authoritative inference message")
    payload = json.loads(messages[0])
    if not isinstance(payload, dict):
        raise ValueError("invalid inference metadata")
    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("invalid inference objects")
    from survng.native_spatial import ROI_LABEL
    expected_objects = [item for item in objects if _gva_object_label(item) != ROI_LABEL]
    normalized = _normalize_gva_objects(payload)
    if len(normalized) != len(expected_objects):
        raise ValueError("incomplete inference metadata")
    if spatial_plan is not None:
        for obj in normalized:
            obj.setdefault("native_zone_ids", [])
            obj["native_zone_revision"] = spatial_plan["revision"]
    provenance, fresh_objects = native_result or ("unknown", [])
    # gvatrack preserves detector ROIs and appends unassociated predictions.
    # Transfer IDs only on an exact, unambiguous detector ROI match. Never
    # promote an appended prediction (which may carry confidence=1) to evidence.
    remaining = list(fresh_objects) if provenance == "native_fresh_detection" else []
    for obj in normalized:
        matches = [item for item in remaining
                   if item["label"] == obj["label"] and item["box"] == obj["box"]
                   and abs(item["confidence"] - obj["confidence"]) < 1e-5]
        obj["detection_provenance"] = "native_tracked_prediction"
        if len(matches) == 1:
            obj["detection_provenance"] = "native_fresh_detection"
            remaining.remove(matches[0])
    if remaining:
        selected = (
            None if tracking_classes is None
            else {str(label).strip().lower() for label in tracking_classes}
        )
        # Missing downstream metadata is fatal for a class that was supposed
        # to traverse gvatrack. Detector classes intentionally excluded from
        # tracking are still authoritative fresh context and must reach the
        # object registry without inventing a tracker ID.
        lost_tracked = [
            item for item in remaining
            if selected is None
            or str(item.get("label") or "").strip().lower() in selected
        ]
        if lost_tracked:
            raise ValueError("tracker metadata lost authoritative detector ROIs")
        for item in remaining:
            context = deepcopy(item)
            context.pop("native_track_id", None)
            context["detection_provenance"] = "native_fresh_detection"
            normalized.append(context)
    snapshot = {
        "schema_version": 1,
        "provenance": provenance,
        "source_pts": float(buffer.pts) / gst_second,
        "inference_sequence": inference_sequence,
        "width": int(structure.get_value("width") or 0),
        "height": int(structure.get_value("height") or 0),
        "objects": normalized,
    }
    if spatial_plan is not None:
        snapshot["zone_revision"] = spatial_plan["revision"]
    DetectionSnapshot.parse(snapshot)
    return snapshot



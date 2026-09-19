"""Shared native object identity and incident inventory state.

The registry owns temporal identity/confirmation/history once. Activity policy
and incident inventory consume that state independently.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import math
from statistics import median


NativeObjectKey = tuple[str, int]


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def compact_history(samples, limit=1024):
    """Bound memory while preserving the episode start and recent detail."""
    if len(samples) <= limit:
        return samples
    split = len(samples) // 2
    return samples[:split:2] + samples[split:]


_LABEL_FAMILIES = (
    frozenset({"car", "truck", "bus", "van", "motorcycle", "bicycle", "robot_lawnmower"}),
    frozenset({"person", "child"}),
    frozenset({"dog", "cat", "horse", "deer", "bird", "animal"}),
)


def _compatible_labels(left: str, right: str) -> bool:
    left, right = left.strip().lower(), right.strip().lower()
    if not left or not right or left == right:
        return True
    return any(left in family and right in family for family in _LABEL_FAMILIES)


def _box(item):
    raw = item.get("box") if isinstance(item, dict) else None
    if not isinstance(raw, dict):
        return None
    try:
        values = tuple(float(raw[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _association_score(previous, current):
    before, after = _box(previous), _box(current)
    if before is None or after is None:
        return None
    ax1, ay1, ax2, ay2 = before
    bx1, by1, bx2, by2 = after
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - overlap
    iou = overlap / union if union > 0 else 0.0
    if iou >= .15:
        return 2.0 + iou
    acx, acy = (ax1+ax2)/2, (ay1+ay2)/2
    bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
    scale = max(
        1.0,
        math.hypot(ax2-ax1, ay2-ay1),
        math.hypot(bx2-bx1, by2-by1),
    )
    distance = math.hypot(acx-bcx, acy-bcy) / scale
    return 1.0 - distance/.65 if distance <= .65 else None


class NativeObjectRegistry:
    """Single owner for native object identity, confirmation, and history."""

    def __init__(self, camera_id, config):
        self.camera_id = camera_id
        self.config = config
        self.tracks: dict[NativeObjectKey, dict] = {}
        self.counts = Counter()
        self._next_track_id = 1
        self._next_fallback_id = 1

    def reset(self):
        self.tracks.clear()
        self._next_track_id = 1
        self._next_fallback_id = 1

    def _expire(self, now, fresh_fps):
        timeout = max(
            self.config.native.activity_timeout_seconds,
            (self.config.native.batch_size + .5) / max(.001, fresh_fps),
        )
        for key, track in list(self.tracks.items()):
            if now - track["last_monotonic"] >= timeout:
                del self.tracks[key]

    def _key(self, obj, used):
        native_id = obj.get("native_track_id")
        if type(native_id) is int and native_id >= 0:
            return ("native", native_id)
        label = str(obj.get("label") or "").strip()
        scored = []
        for key, track in self.tracks.items():
            if key in used or key[0] != "fallback":
                continue
            if not _compatible_labels(str(track.get("label") or ""), label):
                continue
            score = _association_score(track, obj)
            if score is not None:
                scored.append((score, key))
        if scored:
            return max(scored)[1]
        key = ("fallback", self._next_fallback_id)
        self._next_fallback_id += 1
        return key

    @staticmethod
    def _winning_label(track):
        votes = track["_label_votes"]
        confidences = track["_label_confidences"]

        def score(label):
            values = confidences.get(label, [])
            return (
                votes[label],
                median(values) if values else 0.0,
                max(values, default=0.0),
                label,
            )

        return max(votes, key=score, default=str(track.get("label") or ""))

    def observe(
        self,
        objects,
        *,
        session,
        identity_epoch,
        epoch,
        now,
        dimensions,
        fresh_fps,
    ):
        self._expire(now, fresh_fps)
        seen: set[NativeObjectKey] = set()
        for obj in objects:
            if obj.get("detection_provenance") != "native_fresh_detection":
                continue
            label = str(obj.get("label") or "").strip()
            box = _box(obj)
            if not label or box is None:
                continue
            try:
                confidence = float(obj.get("confidence") or 0.0)
                standard_threshold = float(
                    obj.get("confidence_threshold")
                    or self.config.confidence_threshold
                )
                candidate_threshold = min(
                    standard_threshold,
                    float(self.config.event_candidate_confidence_threshold),
                )
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                not math.isfinite(confidence)
                or not math.isfinite(standard_threshold)
                or not math.isfinite(candidate_threshold)
                or confidence < candidate_threshold
            ):
                continue
            confirming = confidence >= standard_threshold
            key = self._key(obj, seen)
            if key in seen:
                self.counts["duplicate_objects"] += 1
                continue
            seen.add(key)
            track = self.tracks.get(key)
            if track is None:
                if len(self.tracks) >= self.config.native.maximum_tracks:
                    self.counts["capacity_drops"] += 1
                    continue
                native_id = obj.get("native_track_id")
                identity = (
                    f"{self.camera_id}/{session}/{identity_epoch}/{native_id}"
                    if type(native_id) is int and native_id >= 0
                    else f"{self.camera_id}/{session}/{identity_epoch}/fallback-{key[1]}"
                )
                track = {
                    "track_id": self._next_track_id,
                    "native_track_id": native_id if type(native_id) is int and native_id >= 0 else None,
                    "native_identity": identity,
                    "label": label,
                    "first_seen": iso(epoch),
                    "observations": 0,
                    "consecutive": 0,
                    "last_monotonic": now,
                    "box_history": [],
                    "trajectory": [],
                    "_label_votes": Counter(),
                    "_label_confidences": {},
                    "_label_confirmations": Counter(),
                }
                self.tracks[key] = track
                self._next_track_id += 1
                if key[0] == "fallback":
                    self.counts["fallback_identities"] += 1
            if now - track["last_monotonic"] > max(
                self.config.native.maximum_observation_age_seconds,
                3 / max(.001, fresh_fps),
            ):
                track["consecutive"] = 0
                track["_label_confirmations"].clear()
            track["_label_votes"][label] += 1
            track["_label_confidences"].setdefault(label, []).append(confidence)
            if len(track["_label_confidences"][label]) > 32:
                del track["_label_confidences"][label][:-32]
            if confirming:
                track["_label_confirmations"][label] += 1
            winning = self._winning_label(track)
            winning_confidences = track["_label_confidences"].get(winning, [])
            aggregate_confidence = (
                float(median(winning_confidences))
                if winning_confidences
                else confidence
            )
            track.update(
                label=winning,
                last_monotonic=now,
                last_seen=iso(epoch),
                box=dict(zip(("x1", "y1", "x2", "y2"), box)),
                confidence=aggregate_confidence,
                zones=deepcopy(obj.get("zones", [])),
                incident_eligible=bool(obj.get("incident_eligible")),
                zone_eligible=bool(obj.get("zone_eligible")),
                confidence_eligible=obj.get("confidence_eligible"),
                zone_admission_reason=obj.get("zone_admission_reason"),
                spatial_zones=deepcopy(obj.get("spatial_zones", [])),
                detection_frame_width=int(dimensions[0]),
                detection_frame_height=int(dimensions[1]),
            )
            track["observations"] += 1
            track["consecutive"] += 1
            required = self.config.event_class_confirmation_frames.get(
                winning, self.config.event_confirmation_frames
            )
            winning_confirmations = track["_label_confirmations"][winning]
            track["state"] = (
                "confirmed"
                if winning_confirmations >= required
                else "tentative"
            )
            track["confirmed"] = (
                track.get("confirmed", False)
                or track["state"] == "confirmed"
            )
            track["confirming_observations"] = winning_confirmations
            track["required_observations"] = required
            track["candidate_threshold"] = candidate_threshold
            track["confidence_threshold"] = standard_threshold
            track["max_confidence"] = max(winning_confidences, default=0.0)
            x1, y1, x2, y2 = box
            track["box_history"] = compact_history(
                [*track["box_history"], [epoch, x1, y1, x2, y2]]
            )
            track["trajectory"] = compact_history(
                [*track["trajectory"], [epoch, (x1+x2)/2, (y1+y2)/2]]
            )
        for key, track in self.tracks.items():
            if key not in seen:
                track["consecutive"] = 0
                track["_label_confirmations"].clear()
                track["confirming_observations"] = 0
        return seen

    def get(self, key):
        return self.tracks.get(key)

    def confirmed(self, keys):
        return [
            key for key in keys
            if key in self.tracks and self.tracks[key].get("state") == "confirmed"
        ]

    def export(self, key, *, include_history=True):
        track = self.tracks.get(key)
        if track is None:
            return None
        excluded = {
            "last_monotonic",
            "consecutive",
            "_label_votes",
            "_label_confidences",
            "_label_confirmations",
        }
        if not include_history:
            excluded |= {"box_history", "trajectory"}
        result = {k: deepcopy(v) for k, v in track.items() if k not in excluded}
        winning = str(track.get("label") or "")
        votes = track.get("_label_votes") or {}
        confidences = track.get("_label_confidences") or {}
        confirmations = track.get("_label_confirmations") or {}
        winning_confidences = list(confidences.get(winning, []))
        result["track_state"] = track["state"]
        result["track_observations"] = track["observations"]
        result["detection_provenance"] = "native_fresh_detection"
        result["temporal_consensus"] = track.get("state") == "confirmed"
        result["temporal_observations"] = int(votes.get(winning, 0))
        result["temporal_track_observations"] = int(track.get("observations") or 0)
        result["temporal_incident_observations"] = int(
            confirmations.get(winning, 0)
        )
        result["temporal_required_observations"] = int(
            track.get("required_observations") or 0
        )
        result["temporal_peak_confidence"] = max(
            winning_confidences,
            default=float(track.get("confidence") or 0.0),
        )
        result["temporal_label_votes"] = dict(votes)
        return result


class NativeIncidentInventory:
    """Episode-local inventory built from confirmed registry objects."""

    def __init__(self, maximum_tracks):
        self.maximum_tracks = maximum_tracks
        self.started_epoch = None
        self._tracks: dict[int, dict] = {}

    def clear(self):
        self.started_epoch = None
        self._tracks.clear()

    def begin(self, trigger_epoch, pre_roll_seconds):
        if self.started_epoch is None:
            self.started_epoch = float(trigger_epoch) - max(0.0, float(pre_roll_seconds))

    def _clip_history(self, track):
        if self.started_epoch is None:
            return track
        result = deepcopy(track)
        history = [
            point for point in result.get("box_history", [])
            if point and point[0] >= self.started_epoch
        ]
        trajectory = [
            point for point in result.get("trajectory", [])
            if point and point[0] >= self.started_epoch
        ]
        result["box_history"] = history
        result["trajectory"] = trajectory
        if history:
            result["first_seen"] = iso(history[0][0])
        return result

    def record(self, track, activity=None):
        if not isinstance(track, dict) or track.get("track_id") is None:
            return False
        track_id = int(track["track_id"])
        if track_id not in self._tracks and len(self._tracks) >= self.maximum_tracks:
            return False
        stored = self._clip_history(track)
        activity = activity or {}
        stored.update(
            motion_state=activity.get("motion_state", stored.get("motion_state", "context")),
            motion_extent=activity.get("motion_extent", stored.get("motion_extent")),
            activity_eligible=bool(activity.get("activity_eligible", stored.get("activity_eligible", False))),
        )
        previous = self._tracks.get(track_id)
        if previous is not None:
            stored["first_seen"] = previous.get("first_seen", stored.get("first_seen"))
            stored["max_confidence"] = max(
                float(previous.get("max_confidence") or 0.0),
                float(stored.get("max_confidence") or stored.get("confidence") or 0.0),
            )
            stored["observations"] = int(previous.get("observations") or 0) + int(
                stored.get("last_seen") != previous.get("last_seen")
            )
        else:
            stored["observations"] = max(
                1,
                len(stored.get("box_history") or []),
            )
        self._tracks[track_id] = stored
        return True

    def capture(self, registry, keys, activity_states):
        for key in registry.confirmed(keys):
            track = registry.export(key)
            if track is not None:
                self.record(track, activity_states.get(key))

    def tracking_tracks(self):
        result = []
        for track in self._tracks.values():
            item = deepcopy(track)
            try:
                first = datetime.fromisoformat(item["first_seen"]).timestamp()
                last = datetime.fromisoformat(item["last_seen"]).timestamp()
                item["duration_seconds"] = max(0.0, last-first)
            except (KeyError, TypeError, ValueError):
                item["duration_seconds"] = 0.0
            result.append(item)
        return result

    @staticmethod
    def object_view(track):
        excluded = {"box_history", "trajectory", "duration_seconds"}
        result = {k: deepcopy(v) for k, v in track.items() if k not in excluded and not k.startswith("_")}
        result["track_state"] = track.get("state", track.get("track_state"))
        result["track_observations"] = track.get("observations", track.get("track_observations"))
        result["detection_provenance"] = "native_fresh_detection"
        return result

    def objects(self, *, visible_track_ids=None, cover=None):
        objects = [self.object_view(track) for track in self._tracks.values()]
        if visible_track_ids is not None:
            visible = set(visible_track_ids)
            for item in objects:
                item["snapshot_visible"] = item.get("track_id") in visible
        if cover is not None:
            cover = deepcopy(cover)
            match = next((item for item in objects if item.get("track_id") == cover.get("track_id")), None)
            if match is None:
                match = next((
                    item for item in objects
                    if item.get("native_identity") == cover.get("native_identity")
                    and item.get("native_identity")
                ), None)
            if match is None:
                match = next((
                    item for item in objects
                    if item.get("label") == cover.get("label")
                    and item.get("native_track_id") == cover.get("native_track_id")
                    and item.get("native_track_id") is not None
                ), None)
            for item in objects:
                item["snapshot_visible"] = False
            if match is not None:
                match.update(cover)
                match["snapshot_visible"] = True
                match["verification"] = {"status": "confirmed", "source": "main_crop"}
            else:
                cover["snapshot_visible"] = True
                cover["verification"] = {"status": "confirmed", "source": "main_crop"}
                objects.insert(0, cover)
        return objects

    def first_seen_epoch(self):
        values = []
        for track in self._tracks.values():
            try:
                values.append(datetime.fromisoformat(track["first_seen"]).timestamp())
            except (KeyError, TypeError, ValueError):
                continue
        return min(values) if values else None

    def __len__(self):
        return len(self._tracks)

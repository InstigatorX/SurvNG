from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import logging
import threading
import time
from typing import Any, Callable, Protocol

import numpy as np

from .config import CameraConfig, ObjectTrackingConfig
from .detector import detection_failure
from .security import redact_secret_text
from .zones import apply_detection_zones


LOGGER = logging.getLogger(__name__)
TRACKING_STOP_TIMEOUT_SECONDS = 18.0
Box = tuple[float, float, float, float]
FrameSample = tuple[np.ndarray, float, float]
FrameProvider = Callable[[], FrameSample | None]
TrackingUpdate = Callable[[int, dict[str, Any], list[dict[str, Any]] | None], object | None]
TrackingPublisher = Callable[[str, dict[str, Any]], None]


class ObjectDetectorBackend(Protocol):
    config: Any

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        ...


class AppearanceEncoder(Protocol):
    @property
    def enabled(self) -> bool:
        ...

    def embed(self, person: np.ndarray) -> np.ndarray:
        ...

    def supports_label(self, label: str) -> bool:
        ...

    def embed_for_label(self, label: str, crop: np.ndarray) -> np.ndarray:
        ...


def _encoder_supports_label(
    encoder: AppearanceEncoder,
    config: ObjectTrackingConfig,
    label: str,
) -> bool:
    if not config.reid_enabled_for_label(label):
        return False
    supports = getattr(encoder, "supports_label", None)
    return bool(supports(label)) if callable(supports) else label == "person"


def _encode_appearance(
    encoder: AppearanceEncoder,
    label: str,
    crop: np.ndarray,
) -> np.ndarray:
    embed_for_label = getattr(encoder, "embed_for_label", None)
    if callable(embed_for_label):
        return np.asarray(embed_for_label(label, crop), dtype=np.float32)
    return np.asarray(encoder.embed(crop), dtype=np.float32)


class ObjectTrackerBackend(Protocol):
    def update(
        self,
        detections: list[dict[str, Any]],
        captured_at: float,
        *,
        confirm_new: bool = False,
    ) -> list[dict[str, Any]]:
        ...

    def has_live_tracks(self, captured_at: float) -> bool:
        ...

    def summaries(self, captured_at: float) -> list[dict[str, Any]]:
        ...

    def diagnostics(self) -> dict[str, Any]:
        ...


ObjectTrackerBuilder = Callable[[ObjectTrackingConfig, float], ObjectTrackerBackend]
SUPPORTED_ULTRALYTICS_TRACKING_VERSION = "8.4.108"


def ultralytics_botsort_dependency_status() -> dict[str, Any]:
    try:
        installed_version = version("ultralytics")
    except PackageNotFoundError:
        installed_version = ""
    lap_present = importlib.util.find_spec("lap") is not None
    available = (
        importlib.util.find_spec("ultralytics") is not None
        and installed_version == SUPPORTED_ULTRALYTICS_TRACKING_VERSION
        and lap_present
    )
    if not installed_version:
        reason = "Ultralytics is not installed."
    elif installed_version != SUPPORTED_ULTRALYTICS_TRACKING_VERSION:
        reason = (
            f"Ultralytics {installed_version} is installed; "
            f"SurvNG requires {SUPPORTED_ULTRALYTICS_TRACKING_VERSION}."
        )
    elif not lap_present:
        reason = "The LAP assignment dependency is not installed."
    else:
        reason = ""
    return {
        "available": available,
        "installed_version": installed_version,
        "required_version": SUPPORTED_ULTRALYTICS_TRACKING_VERSION,
        "reason": reason,
    }


def ultralytics_botsort_available() -> bool:
    return bool(ultralytics_botsort_dependency_status()["available"])


def _build_ultralytics_botsort(
    config: ObjectTrackingConfig,
    high_confidence_threshold: float,
) -> ObjectTrackerBackend:
    from .ultralytics_tracking import UltralyticsBotSortObjectTracker

    return UltralyticsBotSortObjectTracker(config, high_confidence_threshold)


class ObjectTrackerRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, ObjectTrackerBuilder] = {}

    def register(self, implementation: str, builder: ObjectTrackerBuilder) -> None:
        name = str(implementation or "").strip().lower()
        if not name:
            raise ValueError("object tracker implementation cannot be empty")
        if name in self._builders:
            raise ValueError(f"object tracker implementation already registered: {name}")
        self._builders[name] = builder

    def create(
        self,
        implementation: str,
        config: ObjectTrackingConfig,
        high_confidence_threshold: float,
    ) -> ObjectTrackerBackend:
        name = str(implementation or "").strip().lower()
        builder = self._builders.get(name)
        if builder is None:
            available = ", ".join(sorted(self._builders)) or "none"
            raise ValueError(
                f"unknown object tracker implementation {name!r}; available: {available}"
            )
        return builder(config, high_confidence_threshold)

    def require(self, implementation: str) -> None:
        name = str(implementation or "").strip().lower()
        if name not in self._builders:
            available = ", ".join(sorted(self._builders)) or "none"
            raise ValueError(
                f"unknown object tracker implementation {name!r}; available: {available}"
            )


def _box(value: object) -> Box | None:
    if not isinstance(value, dict):
        return None
    try:
        x1 = float(value["x1"])
        y1 = float(value["y1"])
        x2 = float(value["x2"])
        y2 = float(value["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(item) for item in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _confidence(detection: dict[str, Any]) -> float:
    try:
        value = float(detection.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _iou(left: Box, right: Box) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1.0, left_area + right_area - intersection)


def _appearance(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or not np.all(np.isfinite(vector)) or norm <= 1e-9:
        return None
    return vector / norm


def _ensure_detection_appearance(
    detection: dict[str, Any],
    reason: str = "unspecified",
) -> np.ndarray | None:
    """Resolve a detection's appearance at most once, only when association needs it."""
    embedding = _appearance(detection.get("_tracking_embedding"))
    if embedding is not None:
        return embedding
    provider = detection.pop("_tracking_embedding_provider", None)
    if not callable(provider):
        return None
    detection["_tracking_embedding_reason"] = reason
    try:
        embedding = _appearance(provider())
    except Exception:
        # Providers normally report sanitized details through session telemetry.
        # Keep this defensive boundary free of exception text in case a future
        # provider includes a credential-bearing model or camera path.
        LOGGER.warning("lazy appearance provider failed")
        return None
    if embedding is not None:
        detection["_tracking_embedding"] = embedding
    return embedding


@dataclass(slots=True)
class ObjectTrack:
    track_id: int
    label: str
    box: Box
    first_seen: float
    last_seen: float
    confidence: float
    max_confidence: float
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    velocity: Box = (0.0, 0.0, 0.0, 0.0)
    zones: set[str] = field(default_factory=set)
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)
    box_history: list[tuple[float, float, float, float, float]] = field(default_factory=list)
    appearance: np.ndarray | None = field(default=None, repr=False)
    reid_matches: int = 0
    reid_recovery_history: list[dict[str, Any]] = field(default_factory=list)

    def predicted_box(self, captured_at: float) -> Box:
        elapsed = max(0.0, min(captured_at - self.last_seen, 2.0))
        return tuple(
            coordinate + speed * elapsed
            for coordinate, speed in zip(self.box, self.velocity, strict=True)
        )  # type: ignore[return-value]

    def observe(self, detection: dict[str, Any], captured_at: float, box: Box) -> None:
        # Capture timestamps are wall-clock values and can briefly move backwards
        # after a clock correction. Keep track chronology monotonic even though
        # frame freshness is independently enforced with a monotonic token.
        elapsed = captured_at - self.last_seen
        captured_at = max(captured_at, self.last_seen)
        if elapsed > 1e-3:
            instantaneous = tuple(
                (coordinate - previous) / elapsed
                for coordinate, previous in zip(box, self.box, strict=True)
            )
            self.velocity = tuple(
                prior * 0.65 + current * 0.35
                for prior, current in zip(self.velocity, instantaneous, strict=True)
            )  # type: ignore[assignment]
        self.box = box
        self.last_seen = captured_at
        self.confidence = _confidence(detection)
        self.max_confidence = max(self.max_confidence, self.confidence)
        self.hits += 1
        self.missed = 0
        self.zones.update(str(zone) for zone in detection.get("zones", []) if zone)
        next_appearance = _appearance(detection.get("_tracking_embedding"))
        if next_appearance is not None:
            if self.appearance is not None and self.appearance.shape == next_appearance.shape:
                next_appearance = _appearance(self.appearance * 0.8 + next_appearance * 0.2)
            self.appearance = next_appearance
        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[1] + box[3]) / 2.0
        self.trajectory.append((round(captured_at, 3), round(center_x, 1), round(center_y, 1)))
        self.box_history.append((
            round(captured_at, 3),
            round(box[0], 1),
            round(box[1], 1),
            round(box[2], 1),
            round(box[3], 1),
        ))
        if len(self.trajectory) > 60:
            del self.trajectory[:-60]
        if len(self.box_history) > 60:
            del self.box_history[:-60]

    def summary(self, *, active: bool) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "state": "confirmed" if self.confirmed and active else "lost" if self.confirmed else "tentative",
            "first_seen": datetime.fromtimestamp(self.first_seen, timezone.utc).isoformat(),
            "last_seen": datetime.fromtimestamp(self.last_seen, timezone.utc).isoformat(),
            "duration_seconds": round(max(0.0, self.last_seen - self.first_seen), 3),
            "observations": self.hits,
            "max_confidence": round(self.max_confidence, 4),
            "box": {
                "x1": round(self.box[0], 1),
                "y1": round(self.box[1], 1),
                "x2": round(self.box[2], 1),
                "y2": round(self.box[3], 1),
            },
            "zones": sorted(self.zones),
            "trajectory": [list(point) for point in self.trajectory],
            "box_history": [list(sample) for sample in self.box_history],
            "reid_matches": self.reid_matches,
            "reid_recovery_history": [dict(item) for item in self.reid_recovery_history],
        }


class ByteTrackObjectTracker:
    """Small tracking-by-detection engine using ByteTrack's two-pass association."""

    def __init__(self, config: ObjectTrackingConfig, high_confidence_threshold: float) -> None:
        self.config = config
        self.high_confidence_threshold = max(
            config.low_confidence_threshold,
            float(high_confidence_threshold),
        )
        self._tracks: dict[int, ObjectTrack] = {}
        self._completed: dict[int, ObjectTrack] = {}
        self._next_track_id = 1
        self._association_counts = {
            "new_track": 0,
            "geometry": 0,
            "appearance_recovery": 0,
        }
        self._reid_avoided_geometry_matches = 0
        self._reid_avoided_by_label: dict[str, int] = {}

    def update(
        self,
        detections: list[dict[str, Any]],
        captured_at: float,
        *,
        confirm_new: bool = False,
    ) -> list[dict[str, Any]]:
        usable = [
            (index, detection, parsed)
            for index, detection in enumerate(detections)
            if detection.get("label") and (parsed := _box(detection.get("box"))) is not None
        ]
        high = [
            item
            for item in usable
            if confirm_new
            or _confidence(item[1]) >= self.high_confidence_threshold
        ]
        low = [
            item
            for item in usable
            if not confirm_new
            and self.config.low_confidence_threshold
            <= _confidence(item[1])
            < self.high_confidence_threshold
        ]
        unmatched_tracks = set(self._tracks)
        assignments: dict[int, int] = {}
        self._associate(high, captured_at, unmatched_tracks, assignments)
        self._associate(low, captured_at, unmatched_tracks, assignments)

        for index, detection, box in high:
            if index in assignments or detection.get("incident_eligible") is False:
                continue
            if len(self._tracks) + len(self._completed) >= self.config.max_tracks_per_session:
                break
            confidence = _confidence(detection)
            try:
                candidate_first_seen = float(detection.get("_tracking_first_seen_at"))
                first_seen = (
                    min(captured_at, candidate_first_seen)
                    if np.isfinite(candidate_first_seen) and candidate_first_seen >= 0.0
                    else captured_at
                )
            except (TypeError, ValueError):
                first_seen = captured_at
            track = ObjectTrack(
                track_id=self._next_track_id,
                label=str(detection["label"]),
                box=box,
                first_seen=first_seen,
                last_seen=captured_at,
                confidence=confidence,
                max_confidence=confidence,
                confirmed=confirm_new or self.config.min_confirmations <= 1,
                zones={str(zone) for zone in detection.get("zones", []) if zone},
                trajectory=[(
                    round(captured_at, 3),
                    round((box[0] + box[2]) / 2.0, 1),
                    round((box[1] + box[3]) / 2.0, 1),
                )],
                box_history=[(
                    round(captured_at, 3),
                    round(box[0], 1),
                    round(box[1], 1),
                    round(box[2], 1),
                    round(box[3], 1),
                )],
                appearance=_ensure_detection_appearance(detection, "track_seed"),
            )
            self._tracks[track.track_id] = track
            assignments[index] = track.track_id
            self._next_track_id += 1
            self._association_counts["new_track"] += 1

        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.missed += 1
        self._expire(captured_at)

        tracked: list[dict[str, Any]] = []
        for index, detection, _box_value in usable:
            track_id = assignments.get(index)
            if track_id is None:
                continue
            track = self._tracks.get(track_id)
            if track is None:
                continue
            tracked.append({
                **{
                    key: value
                    for key, value in detection.items()
                    if not key.startswith("_tracking_")
                },
                "track_id": track_id,
                "track_state": "confirmed" if track.confirmed else "tentative",
                "track_observations": track.hits,
            })
        return tracked

    def _associate(
        self,
        detections: list[tuple[int, dict[str, Any], Box]],
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        candidates: list[tuple[float, int, int, dict[str, Any], Box]] = []
        for index, detection, box in detections:
            for track_id in unmatched_tracks:
                track = self._tracks[track_id]
                if track.label != str(detection.get("label")):
                    continue
                if captured_at - track.last_seen > self.config.lost_timeout_seconds:
                    continue
                elapsed = max(0.0, captured_at - track.last_seen)
                startup_distance_multiplier = 1.0
                if track.confirmed and track.hits == 1:
                    expected_interval = 1.0 / max(0.1, self.config.sample_fps)
                    startup_distance_multiplier = 1.0 + min(
                        0.25,
                        max(0.0, elapsed - expected_interval)
                        / max(expected_interval, self.config.lost_timeout_seconds),
                    )
                score = self._geometry_score(
                    track.predicted_box(captured_at),
                    box,
                    center_distance_multiplier=startup_distance_multiplier,
                )
                if score is not None:
                    candidates.append((score, track_id, index, detection, box))
        used_detections: set[int] = set()
        for _score, track_id, index, detection, box in sorted(
            candidates,
            key=lambda item: item[0],
            reverse=True,
        ):
            if track_id not in unmatched_tracks or index in used_detections:
                continue
            track = self._tracks[track_id]
            supports_reid = (
                self.config.appearance_reid_enabled
                and self.config.reid_enabled_for_label(track.label)
            )
            refresh_due = (
                track.appearance is None
                or track.hits % self.config.reid_refresh_interval_frames == 0
            )
            if supports_reid and refresh_due:
                _ensure_detection_appearance(
                    detection,
                    "track_seed" if track.appearance is None else "periodic_refresh",
                )
            elif supports_reid and callable(detection.get("_tracking_embedding_provider")):
                self._reid_avoided_geometry_matches += 1
                label = track.label.lower()
                self._reid_avoided_by_label[label] = (
                    self._reid_avoided_by_label.get(label, 0) + 1
                )
            track.observe(detection, captured_at, box)
            track.confirmed = track.confirmed or track.hits >= self.config.min_confirmations
            unmatched_tracks.remove(track_id)
            used_detections.add(index)
            assignments[index] = track_id
            self._association_counts["geometry"] += 1

        self._associate_appearance(detections, captured_at, unmatched_tracks, assignments)

    def _geometry_score(
        self,
        left: Box,
        right: Box,
        *,
        center_distance_multiplier: float = 1.0,
    ) -> float | None:
        overlap = _iou(left, right)
        if overlap >= self.config.match_iou_threshold:
            return 1.0 + overlap
        left_width = left[2] - left[0]
        left_height = left[3] - left[1]
        right_width = right[2] - right[0]
        right_height = right[3] - right[1]
        left_area = left_width * left_height
        right_area = right_width * right_height
        intersection = (
            max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
            * max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        )
        containment = intersection / max(1.0, min(left_area, right_area))
        area_ratio = min(left_area, right_area) / max(1.0, max(left_area, right_area))
        if containment >= 0.55 and area_ratio >= 0.10:
            return 0.9 + containment
        left_center = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
        right_center = ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0)
        distance = float(np.hypot(
            left_center[0] - right_center[0],
            left_center[1] - right_center[1],
        ))
        scale = max(1.0, float(np.hypot(
            max(left_width, right_width),
            max(left_height, right_height),
        )))
        distance_ratio = distance / scale
        center_limit = (
            self.config.match_center_distance_ratio
            * max(1.0, min(1.25, center_distance_multiplier))
        )
        if area_ratio >= 0.20 and distance_ratio <= center_limit:
            return 0.5 + (1.0 - distance_ratio / center_limit)
        return None

    def _associate_appearance(
        self,
        detections: list[tuple[int, dict[str, Any], Box]],
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        if not self.config.appearance_reid_enabled:
            return
        candidates: list[tuple[float, int, bool, int, dict[str, Any], Box]] = []
        for index, detection, box in detections:
            label = str(detection.get("label") or "").lower()
            if index in assignments or not self.config.reid_enabled_for_label(label):
                continue
            active_candidates = [
                (track_id, track)
                for track_id in unmatched_tracks
                if (
                    (track := self._tracks[track_id]).label
                    == str(detection.get("label"))
                    and track.appearance is not None
                    and captured_at - track.last_seen <= self.config.reid_max_age_seconds
                )
            ]
            completed_candidates = [
                (track_id, track)
                for track_id, track in self._completed.items()
                if (
                    track.label == str(detection.get("label"))
                    and track.appearance is not None
                    and captured_at - track.last_seen <= self.config.reid_max_age_seconds
                )
            ]
            if not active_candidates and not completed_candidates:
                continue
            embedding = _ensure_detection_appearance(detection, "geometry_recovery")
            if embedding is None:
                continue
            for track_id, track in active_candidates:
                if track.appearance.shape == embedding.shape:
                    candidates.append((
                        float(np.dot(track.appearance, embedding)),
                        track_id,
                        False,
                        index,
                        detection,
                        box,
                    ))
            for track_id, track in completed_candidates:
                if track.appearance.shape != embedding.shape:
                    continue
                candidates.append((
                    float(np.dot(track.appearance, embedding)),
                    track_id,
                    True,
                    index,
                    detection,
                    box,
                ))
        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        for score, track_id, completed, index, detection, box in sorted(
            candidates,
            key=lambda item: item[0],
            reverse=True,
        ):
            if (
                score < self.config.reid_threshold_for_label(
                    str(detection.get("label") or "")
                )
                or index in used_detections
                or track_id in used_tracks
            ):
                continue
            if completed:
                track = self._completed.pop(track_id, None)
                if track is None:
                    continue
                self._tracks[track_id] = track
            else:
                if track_id not in unmatched_tracks:
                    continue
                track = self._tracks[track_id]
                unmatched_tracks.remove(track_id)
            track.observe(detection, captured_at, box)
            track.reid_matches += 1
            track.reid_recovery_history.append({
                "captured_at": round(captured_at, 3),
                "similarity": round(score, 4),
                "resumed_completed_track": completed,
                "box": [round(value, 1) for value in box],
            })
            if len(track.reid_recovery_history) > 60:
                del track.reid_recovery_history[:-60]
            track.confirmed = track.confirmed or track.hits >= self.config.min_confirmations
            assignments[index] = track_id
            used_detections.add(index)
            used_tracks.add(track_id)
            self._association_counts["appearance_recovery"] += 1

    def _expire(self, captured_at: float) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if captured_at - track.last_seen > self.config.lost_timeout_seconds
        ]
        for track_id in expired:
            self._completed[track_id] = self._tracks.pop(track_id)

    def has_live_tracks(self, captured_at: float) -> bool:
        return any(
            captured_at - track.last_seen <= self.config.lost_timeout_seconds
            for track in self._tracks.values()
        )

    def summaries(self, captured_at: float) -> list[dict[str, Any]]:
        completed = [track.summary(active=False) for track in self._completed.values() if track.confirmed]
        active = [
            track.summary(
                active=captured_at - track.last_seen <= self.config.lost_timeout_seconds,
            )
            for track in self._tracks.values()
            if track.confirmed
        ]
        return sorted([*completed, *active], key=lambda item: int(item["track_id"]))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "association_counts": dict(self._association_counts),
            "reid_avoided_geometry_matches": self._reid_avoided_geometry_matches,
            "reid_avoided_by_label": dict(self._reid_avoided_by_label),
        }


def build_builtin_object_tracker_registry() -> ObjectTrackerRegistry:
    registry = ObjectTrackerRegistry()
    registry.register("survng_hybrid", ByteTrackObjectTracker)
    # Compatibility alias for configurations created before the tracker gained
    # SurvNG-specific geometry and appearance association.
    registry.register("bytetrack", ByteTrackObjectTracker)
    registry.register("ultralytics_botsort", _build_ultralytics_botsort)
    return registry


class ObjectTrackingSession:
    def __init__(
        self,
        camera: CameraConfig,
        config: ObjectTrackingConfig,
        detector: ObjectDetectorBackend,
        frame_provider: FrameProvider,
        update_event: TrackingUpdate,
        publisher: TrackingPublisher | None,
        limiter: threading.BoundedSemaphore,
        tracker_registry: ObjectTrackerRegistry | None = None,
        appearance_encoder: AppearanceEncoder | None = None,
    ) -> None:
        self.camera = camera
        self.config = config
        self.detector = detector
        self.frame_provider = frame_provider
        self.update_event = update_event
        self.publisher = publisher
        self.limiter = limiter
        self.tracker_registry = tracker_registry or build_builtin_object_tracker_registry()
        self.appearance_encoder = appearance_encoder
        self._lock = threading.RLock()
        self._transition_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._event_id: int | None = None
        self._deadline = 0.0
        self._accepting = False
        self._status: dict[str, Any] = self._idle_status()
        self._status["enabled"] = config.enabled
        self._reid_recovery_base = 0
        self._reid_recovery_base_by_label: dict[str, int] = {}
        self._reid_avoided_base = 0
        self._reid_avoided_base_by_label: dict[str, int] = {}
        self._reid_attempt_base = 0
        self._reid_attempt_base_by_reason: dict[str, int] = {}

    @staticmethod
    def _idle_status() -> dict[str, Any]:
        return {
            "enabled": False,
            "active": False,
            "event_id": None,
            "track_count": 0,
            "confirmed_tracks": 0,
            "frames_processed": 0,
            "last_error": "",
            "reid_failures": 0,
            "reid_attempts": 0,
            "reid_successes": 0,
            "reid_inference_ms": 0.0,
            "reid_average_ms": 0.0,
            "reid_attempts_by_label": {},
            "reid_attempts_by_reason": {},
            "reid_recoveries": 0,
            "reid_recoveries_by_label": {},
            "reid_avoided_geometry_matches": 0,
            "reid_avoided_by_label": {},
            "last_reid_error": "",
        }

    def start(
        self,
        event_id: int,
        event_at: datetime,
        initial_objects: list[dict[str, Any]],
        initial_frame: np.ndarray | None = None,
    ) -> bool:
        with self._transition_lock:
            return self._start_session(event_id, event_at, initial_objects, initial_frame)

    def _start_session(
        self,
        event_id: int,
        event_at: datetime,
        initial_objects: list[dict[str, Any]],
        initial_frame: np.ndarray | None,
    ) -> bool:
        if not self.config.enabled or not any(
            item.get("label") and item.get("incident_eligible") is not False
            for item in initial_objects
        ):
            return False
        previous_thread: threading.Thread | None = None
        with self._lock:
            if not self._accepting:
                return False
            if self._thread is not None and self._thread.is_alive():
                if self._event_id == event_id:
                    self._deadline = max(self._deadline, time.monotonic() + self.config.max_session_seconds)
                    return True
                self._stop.set()
                previous_thread = self._thread
        if previous_thread is not None:
            previous_thread.join(timeout=TRACKING_STOP_TIMEOUT_SECONDS)
            if previous_thread.is_alive():
                LOGGER.warning(
                    "object tracking for %s is still stopping; skipped event %d",
                    self.camera.id,
                    event_id,
                )
                return False
        with self._lock:
            if not self._accepting:
                return False
            self._stop = threading.Event()
            self._event_id = event_id
            self._deadline = time.monotonic() + self.config.max_session_seconds
            self._reid_recovery_base = int(self._status.get("reid_recoveries") or 0)
            self._reid_recovery_base_by_label = dict(
                self._status.get("reid_recoveries_by_label") or {}
            )
            self._reid_avoided_base = int(
                self._status.get("reid_avoided_geometry_matches") or 0
            )
            self._reid_avoided_base_by_label = dict(
                self._status.get("reid_avoided_by_label") or {}
            )
            self._reid_attempt_base = int(self._status.get("reid_attempts") or 0)
            self._reid_attempt_base_by_reason = dict(
                self._status.get("reid_attempts_by_reason") or {}
            )
            thread = threading.Thread(
                target=self._run,
                args=(
                    event_id,
                    event_at,
                    [dict(item) for item in initial_objects],
                    initial_frame.copy() if initial_frame is not None else None,
                    self._stop,
                ),
                name=f"object-tracking-{self.camera.id}",
                daemon=False,
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                self._event_id = None
                raise
            return True

    def set_accepting(self, accepting: bool) -> None:
        with self._lock:
            self._accepting = bool(accepting and self.config.enabled)
        if not accepting:
            self.stop()

    def stop(self) -> bool:
        with self._lock:
            self._accepting = False
            stop = self._stop
            thread = self._thread
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=TRACKING_STOP_TIMEOUT_SECONDS)
            if thread.is_alive():
                LOGGER.error("object tracking worker did not stop for %s", self.camera.id)
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
                self._event_id = None
            return self._thread is None or not self._thread.is_alive()

    def running(self) -> bool:
        with self._lock:
            return bool(self._thread is not None and self._thread.is_alive())

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._status,
                "accepting": self._accepting,
                "worker_running": bool(self._thread is not None and self._thread.is_alive()),
            }

    def _run(
        self,
        event_id: int,
        event_at: datetime,
        initial_objects: list[dict[str, Any]],
        initial_frame: np.ndarray | None,
        stop: threading.Event,
    ) -> None:
        acquired = self.limiter.acquire(blocking=False)
        if not acquired:
            skipped = {
                "implementation": self.config.implementation,
                "state": "skipped_capacity",
                "sample_fps": self.config.sample_fps,
                "frames_processed": 0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "tracks": [],
            }
            try:
                self.update_event(event_id, skipped, None)
            except Exception as exc:
                self._set_status(last_error=str(exc))
                LOGGER.exception(
                    "failed to persist capacity status for %s event %d",
                    self.camera.id,
                    event_id,
                )
            self._set_status(enabled=True, active=False, event_id=event_id, last_error="tracking capacity is busy")
            self._publish_safely(event_id, skipped)
            LOGGER.info("object tracking skipped for %s event %d: capacity busy", self.camera.id, event_id)
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
                    self._event_id = None
            return
        tracker: ObjectTrackerBackend | None = None
        frames_processed = 0
        try:
            tracker = self.tracker_registry.create(
                self.config.implementation,
                self.config,
                float(self.detector.config.confidence_threshold),
            )
            if initial_frame is not None:
                self._annotate_appearances(
                    initial_frame,
                    initial_objects,
                    lazy=self.config.implementation == "survng_hybrid",
                )
            captured_at = time.time()
            for detected in initial_objects:
                detected["_tracking_first_seen_at"] = event_at.timestamp()
            initial_tracked = tracker.update(initial_objects, captured_at, confirm_new=True)
            self._set_status(enabled=True, active=True, event_id=event_id, last_error="")
            self._persist(event_id, tracker, captured_at, initial_tracked, frames_processed, "active")
            interval = 1.0 / self.config.sample_fps
            next_sample = time.monotonic()
            frame_acquisition_deadline = min(
                self._deadline,
                time.monotonic() + 5.0,
            )
            consecutive_failures = 0
            last_frame_token: float | None = None
            while not stop.is_set():
                with self._lock:
                    deadline = self._deadline
                if time.monotonic() >= deadline:
                    break
                wait_seconds = max(0.0, next_sample - time.monotonic())
                if stop.wait(wait_seconds):
                    break
                sample = self.frame_provider()
                now_epoch = time.time()
                if sample is None:
                    if time.monotonic() >= frame_acquisition_deadline and not tracker.has_live_tracks(now_epoch):
                        break
                    next_sample = time.monotonic() + interval
                    continue
                frame, sample_epoch, frame_token = sample
                if last_frame_token is not None and frame_token <= last_frame_token:
                    if not tracker.has_live_tracks(now_epoch):
                        break
                    next_sample = time.monotonic() + interval
                    continue
                last_frame_token = frame_token
                objects = self.detector.detect(
                    frame,
                    confidence_threshold=self.config.low_confidence_threshold,
                )
                failure = detection_failure(objects)
                if failure:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        raise RuntimeError(f"tracking detector failed repeatedly: {failure}")
                    next_sample = time.monotonic() + interval
                    continue
                consecutive_failures = 0
                apply_detection_zones(
                    self.camera,
                    objects,
                    int(frame.shape[1]),
                    int(frame.shape[0]),
                    float(self.detector.config.confidence_threshold),
                    bool(getattr(self.detector.config, "require_incident_zone", True)),
                )
                self._annotate_appearances(
                    frame,
                    objects,
                    lazy=self.config.implementation == "survng_hybrid",
                )
                tracked = tracker.update(objects, sample_epoch)
                frames_processed += 1
                self._persist(event_id, tracker, sample_epoch, tracked, frames_processed, "active")
                if not tracker.has_live_tracks(sample_epoch):
                    break
                next_sample = max(next_sample + interval, time.monotonic())
            final_epoch = time.time()
            self._persist(event_id, tracker, final_epoch, None, frames_processed, "complete")
            self._set_status(
                enabled=True,
                active=False,
                event_id=event_id,
                track_count=len(tracker.summaries(final_epoch)),
                confirmed_tracks=len(tracker.summaries(final_epoch)),
                frames_processed=frames_processed,
            )
        except Exception as exc:
            self._persist_failure(event_id, tracker, frames_processed, exc)
            self._set_status(
                enabled=True,
                active=False,
                event_id=event_id,
                last_error=redact_secret_text(exc)[:240],
            )
            LOGGER.exception("object tracking failed for %s event %d", self.camera.id, event_id)
        finally:
            self.limiter.release()
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
                    self._event_id = None

    def _persist(
        self,
        event_id: int,
        tracker: ObjectTrackerBackend,
        captured_at: float,
        tracked_objects: list[dict[str, Any]] | None,
        frames_processed: int,
        state: str,
    ) -> None:
        tracks = tracker.summaries(captured_at)
        diagnostics_method = getattr(tracker, "diagnostics", None)
        tracker_diagnostics = (
            diagnostics_method() if callable(diagnostics_method) else {}
        )
        avoided = int(tracker_diagnostics.get("reid_avoided_geometry_matches") or 0)
        avoided_by_label = dict(tracker_diagnostics.get("reid_avoided_by_label") or {})
        recoveries_by_label: dict[str, int] = {}
        for track in tracks:
            recoveries = int(track.get("reid_matches") or 0)
            if recoveries:
                label = str(track.get("label") or "unknown")
                recoveries_by_label[label] = recoveries_by_label.get(label, 0) + recoveries
        payload = {
            "implementation": self.config.implementation,
            "state": state,
            "sample_fps": self.config.sample_fps,
            "frames_processed": frames_processed,
            "updated_at": datetime.fromtimestamp(captured_at, timezone.utc).isoformat(),
            "tracks": tracks,
            "reid_diagnostics": {
                **tracker_diagnostics,
                "inference_attempts": max(
                    0,
                    int(self.status().get("reid_attempts") or 0)
                    - self._reid_attempt_base,
                ),
                "inference_attempts_by_reason": {
                    reason: max(
                        0,
                        int(count)
                        - int(self._reid_attempt_base_by_reason.get(reason) or 0),
                    )
                    for reason, count in dict(
                        self.status().get("reid_attempts_by_reason") or {}
                    ).items()
                    if int(count)
                    - int(self._reid_attempt_base_by_reason.get(reason) or 0)
                    > 0
                },
            },
        }
        if self.update_event(event_id, payload, tracked_objects) is None:
            raise RuntimeError(f"tracking event {event_id} no longer exists")
        self._set_status(
            enabled=True,
            active=state == "active",
            event_id=event_id,
            track_count=len(tracks),
            confirmed_tracks=len(tracks),
            frames_processed=frames_processed,
            reid_recoveries=(
                self._reid_recovery_base + sum(recoveries_by_label.values())
            ),
            reid_recoveries_by_label={
                label: int(self._reid_recovery_base_by_label.get(label) or 0)
                + int(recoveries_by_label.get(label) or 0)
                for label in (
                    self._reid_recovery_base_by_label.keys()
                    | recoveries_by_label.keys()
                )
            },
            reid_avoided_geometry_matches=self._reid_avoided_base + avoided,
            reid_avoided_by_label={
                label: int(self._reid_avoided_base_by_label.get(label) or 0)
                + int(avoided_by_label.get(label) or 0)
                for label in (
                    self._reid_avoided_base_by_label.keys()
                    | avoided_by_label.keys()
                )
            },
        )
        if state == "complete":
            self._publish_safely(event_id, payload)

    def _persist_failure(
        self,
        event_id: int,
        tracker: ObjectTrackerBackend | None,
        frames_processed: int,
        error: Exception,
    ) -> None:
        captured_at = time.time()
        payload = {
            "implementation": self.config.implementation,
            "state": "failed",
            "sample_fps": self.config.sample_fps,
            "frames_processed": frames_processed,
            "updated_at": datetime.fromtimestamp(captured_at, timezone.utc).isoformat(),
            "error": redact_secret_text(error)[:240],
            "tracks": tracker.summaries(captured_at) if tracker is not None else [],
        }
        try:
            self.update_event(event_id, payload, None)
        except Exception:
            LOGGER.exception(
                "failed to persist tracking failure for %s event %d",
                self.camera.id,
                event_id,
            )
        self._publish_safely(event_id, payload)

    def _publish_safely(self, event_id: int, payload: dict[str, Any]) -> None:
        if self.publisher is None:
            return
        try:
            self.publisher("object_tracking", {
                "event_id": event_id,
                "camera_id": self.camera.id,
                **payload,
            })
        except Exception:
            LOGGER.exception(
                "object tracking notification failed for %s event %d",
                self.camera.id,
                event_id,
            )

    def _set_status(self, **values: Any) -> None:
        with self._lock:
            self._status = {**self._status, **values}

    def _annotate_appearances(
        self,
        frame: np.ndarray,
        objects: list[dict[str, Any]],
        *,
        lazy: bool = False,
    ) -> None:
        encoder = self.appearance_encoder
        if encoder is None or not encoder.enabled or not self.config.appearance_reid_enabled:
            return
        height, width = frame.shape[:2]
        candidates = sorted(
            (
                detected
                for detected in objects
                if _encoder_supports_label(
                    encoder,
                    self.config,
                    str(detected.get("label") or "").lower(),
                )
            ),
            key=_confidence,
            reverse=True,
        )[:self.config.reid_max_embeddings_per_frame]
        frame_state = {"failed": False}
        for detected in candidates:
            box = _box(detected.get("box"))
            if box is None:
                continue
            x1 = max(0, min(width - 1, int(box[0])))
            y1 = max(0, min(height - 1, int(box[1])))
            x2 = max(x1 + 1, min(width, int(box[2])))
            y2 = max(y1 + 1, min(height, int(box[3])))
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 16 or crop.shape[1] < 8:
                continue
            label = str(detected.get("label") or "").lower()
            if lazy:
                # The provider is consumed synchronously by tracker.update(), so
                # retaining a view avoids copying large vehicle crops.
                owned_crop = crop
                detected["_tracking_embedding_provider"] = (
                    lambda label=label, crop=owned_crop, detected=detected: self._encode_with_telemetry(
                        encoder,
                        label,
                        crop,
                        frame_state,
                        reason=str(
                            detected.get("_tracking_embedding_reason") or "unspecified"
                        ),
                    )
                )
                continue
            embedding = self._encode_with_telemetry(
                encoder,
                label,
                crop,
                frame_state,
                reason="eager",
            )
            if embedding is not None:
                detected["_tracking_embedding"] = embedding
            if frame_state["failed"]:
                break

    def _encode_with_telemetry(
        self,
        encoder: AppearanceEncoder,
        label: str,
        crop: np.ndarray,
        frame_state: dict[str, bool],
        *,
        reason: str,
    ) -> np.ndarray | None:
        if frame_state["failed"]:
            return None
        started_at = time.perf_counter()
        embedding: np.ndarray | None = None
        error = ""
        try:
            embedding = _encode_appearance(encoder, label, crop)
        except Exception as exc:
            error = redact_secret_text(exc)[:240]
            frame_state["failed"] = True
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        with self._lock:
            attempts = int(self._status.get("reid_attempts") or 0) + 1
            successes = int(self._status.get("reid_successes") or 0)
            failures = int(self._status.get("reid_failures") or 0)
            total_ms = float(self._status.get("reid_inference_ms") or 0.0) + elapsed_ms
            by_label = dict(self._status.get("reid_attempts_by_label") or {})
            by_label[label] = int(by_label.get(label) or 0) + 1
            by_reason = dict(self._status.get("reid_attempts_by_reason") or {})
            by_reason[reason] = int(by_reason.get(reason) or 0) + 1
            if embedding is not None:
                successes += 1
            if error:
                failures += 1
            self._status = {
                **self._status,
                "reid_attempts": attempts,
                "reid_successes": successes,
                "reid_failures": failures,
                "reid_inference_ms": round(total_ms, 3),
                "reid_average_ms": round(total_ms / attempts, 3),
                "reid_attempts_by_label": by_label,
                "reid_attempts_by_reason": by_reason,
                "last_reid_error": error or self._status.get("last_reid_error", ""),
            }
        if error:
            LOGGER.debug(
                "appearance ReID unavailable for %s: %s",
                self.camera.id,
                error,
            )
        return embedding


class ObjectTrackingSessionFactory:
    def __init__(
        self,
        config: ObjectTrackingConfig,
        detector: ObjectDetectorBackend,
        update_event: TrackingUpdate,
        publisher: TrackingPublisher | None,
        limiter: threading.BoundedSemaphore,
        tracker_registry: ObjectTrackerRegistry | None = None,
        appearance_encoder: AppearanceEncoder | None = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.update_event = update_event
        self.publisher = publisher
        self.limiter = limiter
        self.tracker_registry = tracker_registry or build_builtin_object_tracker_registry()
        self.appearance_encoder = appearance_encoder
        # Fail configuration loading before any event tries to start a session.
        self.tracker_registry.require(config.implementation)
        if (
            config.implementation == "ultralytics_botsort"
            and not ultralytics_botsort_available()
        ):
            raise ValueError(
                "ultralytics_botsort requires the optional Ultralytics tracking dependencies"
            )

    def create(self, camera: CameraConfig, frame_provider: FrameProvider) -> ObjectTrackingSession:
        return ObjectTrackingSession(
            camera=camera,
            config=self.config,
            detector=self.detector,
            frame_provider=frame_provider,
            update_event=self.update_event,
            publisher=self.publisher,
            limiter=self.limiter,
            tracker_registry=self.tracker_registry,
            appearance_encoder=self.appearance_encoder,
        )

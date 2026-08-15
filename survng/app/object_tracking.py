from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import logging
import threading
import time
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from .config import CameraConfig, ObjectTrackingConfig
from .detector import detection_failure
from .security import redact_secret_text
from .domain_events import TrackingCompleted
from .visual_quality import image_quality
from .video_frames import DecodedVideoFrame, VideoFrameReference
from .zones import apply_detection_zones


LOGGER = logging.getLogger(__name__)
TRACKING_STOP_TIMEOUT_SECONDS = 18.0
TRACKING_CATCHUP_SETTLE_SECONDS = 5.0
TRACKING_CATCHUP_RETRY_SECONDS = 0.25
Box = tuple[float, float, float, float]
FrameSample = tuple[np.ndarray, float, float]
FrameProvider = Callable[[], FrameSample | None]
CatchupFrameProvider = Callable[
    [float, float, float, int],
    Iterable[tuple[float, np.ndarray] | DecodedVideoFrame],
]
TrackingUpdate = Callable[[int, dict[str, Any], list[dict[str, Any]] | None], object | None]
TrackingPublisher = Callable[[str, dict[str, Any]], None]
AppearanceIndexWriter = Callable[[int, str, Iterable[dict[str, Any]]], int]
TrackingCoverFrameProvider = Callable[
    [float, int, VideoFrameReference | None],
    np.ndarray | None,
]
TrackingSnapshotWriter = Callable[[np.ndarray, datetime], str]
TrackingCoverPromoter = Callable[..., dict[str, Any] | None]


@dataclass(frozen=True, slots=True)
class _TrackingCoverCandidate:
    captured_at: float
    tracked_objects: tuple[dict[str, Any], ...]
    primary_track_id: int
    subject_area_ratio: float
    edge_clearance_ratio: float
    quality_score: float
    detection_confidence: float
    fully_framed: bool
    frame_reference: VideoFrameReference | None = None

    @property
    def score(self) -> tuple[int, float, float, float, float]:
        return (
            int(self.fully_framed),
            self.subject_area_ratio,
            self.quality_score,
            self.detection_confidence,
            self.captured_at,
        )


class AdaptiveTrackingLimiter:
    """Semaphore-compatible limiter with a guarded, observable burst slot."""

    def __init__(
        self,
        baseline: int,
        burst_limit: int,
        *,
        burst_enabled: bool = True,
        burst_guard: Callable[[], bool] | None = None,
    ) -> None:
        self.baseline = max(1, int(baseline))
        self.burst_limit = max(self.baseline, int(burst_limit))
        self.burst_enabled = bool(burst_enabled)
        self.burst_guard = burst_guard
        self._condition = threading.Condition()
        self._active = 0
        self._burst_admissions = 0
        self._burst_denials = 0

    def _limit(self) -> int:
        if not self.burst_enabled or self.burst_limit <= self.baseline:
            return self.baseline
        healthy = False
        try:
            healthy = self.burst_guard is None or bool(self.burst_guard())
        except Exception:
            LOGGER.exception("tracking burst capacity guard failed")
        if not healthy:
            self._burst_denials += 1
            return self.baseline
        return self.burst_limit

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                limit = self._limit()
                if self._active < limit:
                    self._active += 1
                    if self._active > self.baseline:
                        self._burst_admissions += 1
                    return True
                if not blocking:
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise ValueError("tracking limiter released too many times")
            self._active -= 1
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "active": self._active,
                "baseline": self.baseline,
                "burst_limit": self.burst_limit,
                "burst_enabled": self.burst_enabled,
                "burst_admissions": self._burst_admissions,
                "burst_denials": self._burst_denials,
            }


def _rescale_detection_boxes(
    objects: list[dict[str, Any]],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> None:
    if (
        source_width <= 0
        or source_height <= 0
        or target_width <= 0
        or target_height <= 0
        or (source_width == target_width and source_height == target_height)
    ):
        return
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    for detected in objects:
        box = detected.get("box")
        if not isinstance(box, dict):
            continue
        try:
            box["x1"] = float(box["x1"]) * scale_x
            box["y1"] = float(box["y1"]) * scale_y
            box["x2"] = float(box["x2"]) * scale_x
            box["y2"] = float(box["y2"]) * scale_y
        except (KeyError, TypeError, ValueError):
            continue


class ObjectDetectorBackend(Protocol):
    config: Any

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        ...


def _detect_tracking_objects(
    detector: ObjectDetectorBackend,
    frame: np.ndarray,
    confidence_threshold: float,
    *,
    enrichment: bool = False,
) -> list[dict[str, Any]]:
    method_name = "detect_enrichment" if enrichment else "detect_tracking"
    method = getattr(detector, method_name, detector.detect)
    return list(method(frame, confidence_threshold=confidence_threshold))


def _inference_deferred(objects: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(item, dict) and item.get("status") == "inference_deferred"
        for item in objects
    )


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

    def model_identity_for_label(self, label: str) -> dict[str, Any] | None:
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
TESTED_ULTRALYTICS_TRACKING_VERSION = "8.4.115"
MINIMUM_ULTRALYTICS_TRACKING_VERSION = (8, 4, 108)
MAXIMUM_ULTRALYTICS_TRACKING_VERSION = (8, 5, 0)


def _ultralytics_tracking_version_supported(installed_version: str) -> bool:
    """Accept compatible patches without importing the heavyweight runtime."""
    try:
        release = tuple(
            int(part)
            for part in installed_version.partition("+")[0].partition("-")[0].split(".")[:3]
        )
        release = (*release, *(0 for _ in range(3 - len(release))))
    except (TypeError, ValueError):
        return False
    return MINIMUM_ULTRALYTICS_TRACKING_VERSION <= release < MAXIMUM_ULTRALYTICS_TRACKING_VERSION


def _ultralytics_tracking_dependency_status(
    *,
    tracker_name: str,
    module_name: str,
) -> dict[str, Any]:
    try:
        installed_version = version("ultralytics")
    except PackageNotFoundError:
        installed_version = ""
    package_present = importlib.util.find_spec("ultralytics") is not None
    lap_present = importlib.util.find_spec("lap") is not None
    try:
        tracker_present = importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        tracker_present = False
    version_supported = _ultralytics_tracking_version_supported(installed_version)
    available = bool(
        installed_version
        and package_present
        and lap_present
        and tracker_present
        and version_supported
    )
    if not installed_version:
        reason = "Ultralytics is not installed."
    elif not lap_present:
        reason = "The LAP assignment dependency is not installed."
    elif not tracker_present:
        reason = f"The installed Ultralytics build does not include {tracker_name}."
    elif not version_supported:
        reason = (
            f"Ultralytics {installed_version} is outside SurvNG's supported "
            f"{tracker_name} API range (8.4.108 through the latest 8.4.x release)."
        )
    else:
        reason = ""
    return {
        "available": available,
        "installed_version": installed_version,
        # Keep required_version for API compatibility with older frontends.
        # Availability is capability-based; this is the reproducible version
        # pinned by requirements-ultralytics-tracking.txt and exercised by CI.
        "required_version": TESTED_ULTRALYTICS_TRACKING_VERSION,
        "tested_version": TESTED_ULTRALYTICS_TRACKING_VERSION,
        "is_tested_version": installed_version == TESTED_ULTRALYTICS_TRACKING_VERSION,
        "supported_version_range": ">=8.4.108,<8.5",
        "reason": reason,
    }


def ultralytics_deepocsort_dependency_status() -> dict[str, Any]:
    return _ultralytics_tracking_dependency_status(
        tracker_name="Deep OC-SORT",
        module_name="ultralytics.trackers.deep_oc_sort",
    )


def ultralytics_fasttrack_dependency_status() -> dict[str, Any]:
    return _ultralytics_tracking_dependency_status(
        tracker_name="FastTrack",
        module_name="ultralytics.trackers.fast_tracker",
    )


def _build_ultralytics_deepocsort(
    config: ObjectTrackingConfig,
    high_confidence_threshold: float,
) -> ObjectTrackerBackend:
    from .ultralytics_tracking import UltralyticsDeepOCSortObjectTracker

    return UltralyticsDeepOCSortObjectTracker(config, high_confidence_threshold)


def _build_ultralytics_fasttrack(
    config: ObjectTrackingConfig,
    high_confidence_threshold: float,
) -> ObjectTrackerBackend:
    from .ultralytics_tracking import UltralyticsFastTrackObjectTracker

    return UltralyticsFastTrackObjectTracker(config, high_confidence_threshold)


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
    seeded: bool = False

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
        self.seeded = False
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
            if self.config.tracks_label(detection.get("label"))
            and (parsed := _box(detection.get("box"))) is not None
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
                seeded=confirm_new,
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
                if captured_at - track.last_seen > self._association_stale_limit(track):
                    continue
                score = self._geometry_score(
                    track.predicted_box(captured_at),
                    box,
                    allow_scale_jump=track.seeded,
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
            self._observe_geometry(track, detection, captured_at, box)
            unmatched_tracks.remove(track_id)
            used_detections.add(index)
            assignments[index] = track_id
            self._association_counts["geometry"] += 1

        self._associate_unambiguous(
            detections,
            captured_at,
            unmatched_tracks,
            assignments,
        )
        self._associate_appearance(detections, captured_at, unmatched_tracks, assignments)

    def _associate_unambiguous(
        self,
        detections: list[tuple[int, dict[str, Any], Box]],
        captured_at: float,
        unmatched_tracks: set[int],
        assignments: dict[int, int],
    ) -> None:
        """Bridge a short geometry jump when one identity is the only candidate.

        Perspective changes and detector box jitter can move a nearby person's
        center by more than the normal association threshold.  When exactly one
        confirmed track and one detection remain for a label, a bounded relaxed
        geometry check is safer than manufacturing a second identity.
        """
        labels = {
            str(detection.get("label") or "")
            for index, detection, _box_value in detections
            if index not in assignments
        }
        maximum_gap = min(
            self.config.lost_timeout_seconds,
            max(1.0, 2.0 / max(0.1, self.config.sample_fps)),
        )
        for label in labels:
            remaining_detections = [
                item
                for item in detections
                if item[0] not in assignments
                and str(item[1].get("label") or "") == label
            ]
            remaining_tracks = [
                track_id
                for track_id in unmatched_tracks
                if self._tracks[track_id].label == label
                and self._tracks[track_id].confirmed
                and captured_at - self._tracks[track_id].last_seen <= (
                    self._association_stale_limit(self._tracks[track_id])
                    if self._tracks[track_id].seeded
                    else maximum_gap
                )
            ]
            if len(remaining_detections) != 1 or len(remaining_tracks) != 1:
                continue
            index, detection, box = remaining_detections[0]
            track_id = remaining_tracks[0]
            track = self._tracks[track_id]
            score = self._geometry_score(
                track.predicted_box(captured_at),
                box,
                center_distance_multiplier=2.0 if track.seeded else 1.75,
                allow_scale_jump=track.seeded,
            )
            if score is None:
                continue
            self._observe_geometry(track, detection, captured_at, box)
            unmatched_tracks.remove(track_id)
            assignments[index] = track_id
            self._association_counts["geometry"] += 1

    def _observe_geometry(
        self,
        track: ObjectTrack,
        detection: dict[str, Any],
        captured_at: float,
        box: Box,
    ) -> None:
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

    def _geometry_score(
        self,
        left: Box,
        right: Box,
        *,
        center_distance_multiplier: float = 1.0,
        allow_scale_jump: bool = False,
    ) -> float | None:
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
        # A detector can occasionally emit a scene-sized box around an object
        # for one frame.  Its overlap is deceptively high because it contains
        # the prior box, but accepting it produces a physically impossible
        # track jump.  Permit major scale changes only while the center remains
        # close; normal perspective growth and edge entry still satisfy this.
        if (
            not allow_scale_jump
            and right_area > left_area * 2.0
            and distance_ratio > self.config.match_center_distance_ratio * 0.55
        ):
            return None
        overlap = _iou(left, right)
        if overlap >= self.config.match_iou_threshold:
            return 1.0 + overlap
        if containment >= 0.55 and area_ratio >= 0.10:
            return 0.9 + containment
        center_limit = (
            self.config.match_center_distance_ratio
            * max(1.0, min(2.0, center_distance_multiplier))
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

    def appearance_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for track in [*self._completed.values(), *self._tracks.values()]:
            if not track.confirmed or track.appearance is None:
                continue
            records.append({
                "track_id": track.track_id,
                "label": track.label,
                "embedding": track.appearance.copy(),
                "observation_count": track.hits,
                "quality": track.max_confidence,
                "first_seen": datetime.fromtimestamp(
                    track.first_seen,
                    timezone.utc,
                ).isoformat(),
                "last_seen": datetime.fromtimestamp(
                    track.last_seen,
                    timezone.utc,
                ).isoformat(),
            })
        return records

    def _association_stale_limit(self, track: ObjectTrack) -> float:
        if track.seeded:
            return max(
                self.config.lost_timeout_seconds,
                min(8.0, self.config.max_session_seconds),
            )
        return self.config.lost_timeout_seconds

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
    # Ultralytics alternatives are registered for bounded offline comparisons.
    # User configuration normalizes optional upstream trackers back to Hybrid,
    # so production sessions cannot select this implementation.
    registry.register("ultralytics_deepocsort", _build_ultralytics_deepocsort)
    registry.register("ultralytics_fasttrack", _build_ultralytics_fasttrack)
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
        appearance_indexer: AppearanceIndexWriter | None = None,
        catchup_frame_provider: CatchupFrameProvider | None = None,
        cover_frame_provider: TrackingCoverFrameProvider | None = None,
        snapshot_writer: TrackingSnapshotWriter | None = None,
        cover_promoter: TrackingCoverPromoter | None = None,
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
        self.appearance_indexer = appearance_indexer
        self.catchup_frame_provider = catchup_frame_provider
        self.cover_frame_provider = cover_frame_provider
        self.snapshot_writer = snapshot_writer
        self.cover_promoter = cover_promoter
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
        self._frame_width = 0
        self._frame_height = 0
        self._catchup_frames_processed = 0
        self._coverage_gap_count = 0
        self._maximum_coverage_gap_seconds = 0.0
        self._completion_reason = ""
        self._cover_baseline: _TrackingCoverCandidate | None = None
        self._cover_candidate: _TrackingCoverCandidate | None = None
        self._cover_promotion: dict[str, Any] | None = None

    @staticmethod
    def _idle_status() -> dict[str, Any]:
        return {
            "enabled": False,
            "active": False,
            "event_id": None,
            "track_count": 0,
            "confirmed_tracks": 0,
            "frames_processed": 0,
            "catchup_frames_processed": 0,
            "coverage_gap_count": 0,
            "maximum_coverage_gap_seconds": 0.0,
            "coverage_incomplete": False,
            "completion_reason": "",
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
            "appearance_vectors_indexed": 0,
            "appearance_index_error": "",
            "capacity_requests": 0,
            "capacity_waits": 0,
            "capacity_timeouts": 0,
            "capacity_wait_seconds_total": 0.0,
            "capacity_wait_seconds_max": 0.0,
            "capacity_wait_seconds_last": 0.0,
            "cover_promoted": False,
            "cover_promotion_attempted": False,
            "cover_promotion_result": "",
            "cover_verification_inference_ms": 0.0,
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
        initial_objects = [
            item
            for item in initial_objects
            if self.config.tracks_label(item.get("label"))
        ]
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
        self.request_stop()
        return self.wait_stopped(TRACKING_STOP_TIMEOUT_SECONDS)

    def request_stop(self) -> None:
        """Stop accepting work and signal the session without joining it."""
        with self._lock:
            self._accepting = False
            self._stop.set()

    def wait_stopped(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for the signalled session to exit."""
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
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

    def _consider_cover_candidate(
        self,
        frame: np.ndarray,
        captured_at: float,
        tracked_objects: list[dict[str, Any]],
        primary_track_ids: set[int],
        frame_reference: VideoFrameReference | None = None,
    ) -> None:
        """Remember metadata for the best later view without retaining its frame."""
        if (
            frame is None
            or not frame.size
            or self._frame_width <= 0
            or self._frame_height <= 0
        ):
            return
        primary: list[tuple[dict[str, Any], Box, float]] = []
        frame_area = float(self._frame_width * self._frame_height)
        for item in tracked_objects:
            try:
                track_id = int(item.get("track_id"))
            except (TypeError, ValueError):
                continue
            box = _box(item.get("box"))
            if track_id not in primary_track_ids or box is None:
                continue
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / frame_area
            primary.append((item, box, area))
        if not primary:
            return
        item, box, area = max(primary, key=lambda value: value[2])
        clearance = max(
            0.0,
            min(
                box[0] / self._frame_width,
                box[1] / self._frame_height,
                (self._frame_width - box[2]) / self._frame_width,
                (self._frame_height - box[3]) / self._frame_height,
            ),
        )
        source_height, source_width = frame.shape[:2]
        scale_x = source_width / self._frame_width
        scale_y = source_height / self._frame_height
        x1 = max(0, min(source_width, int(np.floor(box[0] * scale_x))))
        y1 = max(0, min(source_height, int(np.floor(box[1] * scale_y))))
        x2 = max(0, min(source_width, int(np.ceil(box[2] * scale_x))))
        y2 = max(0, min(source_height, int(np.ceil(box[3] * scale_y))))
        crop = frame[y1:y2, x1:x2]
        quality = image_quality(crop).score if crop.size else 0.0
        candidate = _TrackingCoverCandidate(
            captured_at=float(captured_at),
            tracked_objects=tuple(dict(value) for value in tracked_objects),
            primary_track_id=int(item["track_id"]),
            subject_area_ratio=float(area),
            edge_clearance_ratio=float(clearance),
            quality_score=float(quality),
            detection_confidence=_confidence(item),
            fully_framed=clearance >= 0.01,
            frame_reference=frame_reference,
        )
        if self._cover_baseline is None:
            self._cover_baseline = candidate
        elif frame_reference is None or not frame_reference.exact:
            return
        if self._cover_candidate is None or candidate.score > self._cover_candidate.score:
            self._cover_candidate = candidate

    def _promote_cover_candidate(self, event_id: int) -> None:
        """Verify and persist one materially better tracked cover."""
        baseline = self._cover_baseline
        candidate = self._cover_candidate
        if (
            baseline is None
            or candidate is None
            or candidate.frame_reference is None
            or not candidate.frame_reference.exact
            or self.cover_frame_provider is None
            or self.snapshot_writer is None
            or self.cover_promoter is None
            or candidate.captured_at < baseline.captured_at + 0.25
            or not candidate.fully_framed
            or candidate.subject_area_ratio
            < max(baseline.subject_area_ratio * 1.5, baseline.subject_area_ratio + 0.0025)
            or candidate.quality_score < max(0.12, baseline.quality_score * 0.55)
        ):
            return
        self._cover_promotion = {
            "cover_promoted": False,
            "cover_promotion_attempted": True,
            "cover_promotion_result": "decode_unavailable",
            "cover_source": "object_tracking",
        }
        frame = self.cover_frame_provider(
            candidate.captured_at,
            self._frame_width,
            candidate.frame_reference,
        )
        if frame is None or not frame.size:
            return
        frame_height, frame_width = frame.shape[:2]
        expected_objects = [dict(item) for item in candidate.tracked_objects]
        for item in expected_objects:
            box = item.get("box")
            if isinstance(box, dict):
                item["box"] = dict(box)
        _rescale_detection_boxes(
            expected_objects,
            self._frame_width,
            self._frame_height,
            frame_width,
            frame_height,
        )
        inference_started = time.monotonic()
        try:
            detected_objects = _detect_tracking_objects(
                self.detector,
                frame,
                self.config.low_confidence_threshold,
                enrichment=True,
            )
        except Exception as error:
            self._cover_promotion.update({
                "cover_promotion_result": "detector_error",
                "cover_verification_error": redact_secret_text(error)[:160],
                "cover_verification_inference_ms": round(
                    (time.monotonic() - inference_started) * 1000.0,
                    3,
                ),
            })
            LOGGER.warning(
                "tracked cover verification failed for %s event %d: %s",
                self.camera.id,
                event_id,
                redact_secret_text(error),
            )
            return
        inference_ms = round((time.monotonic() - inference_started) * 1000.0, 3)
        if _inference_deferred(detected_objects):
            self._cover_promotion.update({
                "cover_promotion_result": "inference_deferred",
                "cover_verification_inference_ms": inference_ms,
            })
            return
        failure = detection_failure(detected_objects)
        if failure:
            self._cover_promotion.update({
                "cover_promotion_result": "detector_unavailable",
                "cover_verification_error": failure[:160],
                "cover_verification_inference_ms": inference_ms,
            })
            return
        tracked_objects = self._associate_cover_detections(
            expected_objects,
            detected_objects,
            candidate.primary_track_id,
            frame_width,
            frame_height,
        )
        primary = next(
            (
                item
                for item in tracked_objects
                if int(item.get("track_id") or 0) == candidate.primary_track_id
            ),
            None,
        )
        primary_box = _box(primary.get("box")) if primary is not None else None
        if primary_box is None:
            self._cover_promotion.update({
                "cover_promotion_result": "primary_subject_not_verified",
                "cover_verification_inference_ms": inference_ms,
                "cover_verification_detection_count": len(detected_objects),
            })
            return
        final_area = (
            (primary_box[2] - primary_box[0])
            * (primary_box[3] - primary_box[1])
            / float(frame_width * frame_height)
        )
        final_clearance = max(
            0.0,
            min(
                primary_box[0] / frame_width,
                primary_box[1] / frame_height,
                (frame_width - primary_box[2]) / frame_width,
                (frame_height - primary_box[3]) / frame_height,
            ),
        )
        crop_x1 = max(0, min(frame_width, int(np.floor(primary_box[0]))))
        crop_y1 = max(0, min(frame_height, int(np.floor(primary_box[1]))))
        crop_x2 = max(0, min(frame_width, int(np.ceil(primary_box[2]))))
        crop_y2 = max(0, min(frame_height, int(np.ceil(primary_box[3]))))
        verified_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        final_quality = image_quality(verified_crop).score if verified_crop.size else 0.0
        if (
            final_clearance < 0.01
            or final_area
            < max(baseline.subject_area_ratio * 1.5, baseline.subject_area_ratio + 0.0025)
            or final_quality < max(0.12, baseline.quality_score * 0.55)
        ):
            self._cover_promotion.update({
                "cover_promotion_result": "verified_frame_not_better",
                "cover_verification_inference_ms": inference_ms,
                "cover_verified_subject_area_ratio": round(final_area, 6),
                "cover_verified_edge_clearance_ratio": round(final_clearance, 6),
                "cover_verified_quality_score": round(final_quality, 6),
            })
            return
        snapshot_path = self.snapshot_writer(
            frame,
            datetime.fromtimestamp(candidate.captured_at, timezone.utc),
        )
        promoted = self.cover_promoter(
            event_id,
            snapshot_path=snapshot_path,
            captured_at=candidate.captured_at,
            frame_width=frame_width,
            frame_height=frame_height,
            tracked_objects=tracked_objects,
            cover_metrics={
                "snapshot_primary_subject": True,
                "snapshot_subject_area_ratio": round(final_area, 6),
                "snapshot_edge_clearance_ratio": round(final_clearance, 6),
                "snapshot_quality_score": round(final_quality, 6),
            },
        )
        if promoted is None:
            self._cover_promotion.update({
                "cover_promotion_result": "event_unavailable",
                "cover_verification_inference_ms": inference_ms,
            })
            return
        self._cover_promotion = {
            "cover_promoted": True,
            "cover_promotion_attempted": True,
            "cover_promotion_result": "promoted",
            "cover_source": "object_tracking",
            "cover_captured_at": datetime.fromtimestamp(
                candidate.captured_at,
                timezone.utc,
            ).isoformat(),
            "cover_primary_track_id": candidate.primary_track_id,
            "cover_subject_area_ratio": round(final_area, 6),
            "cover_previous_subject_area_ratio": round(
                baseline.subject_area_ratio,
                6,
            ),
            "cover_verification_inference_ms": inference_ms,
            "cover_verification_detection_count": len(detected_objects),
            "cover_verified_quality_score": round(final_quality, 6),
            "cover_frame_timestamp_source": "source_pts",
            "cover_frame_pts": candidate.frame_reference.pts,
            "cover_frame_time_base": (
                f"{candidate.frame_reference.time_base_num}/"
                f"{candidate.frame_reference.time_base_den}"
            ),
        }

    @staticmethod
    def _associate_cover_detections(
        expected_objects: list[dict[str, Any]],
        detected_objects: list[dict[str, Any]],
        primary_track_id: int,
        frame_width: int,
        frame_height: int,
    ) -> list[dict[str, Any]]:
        """Associate detections from the saved pixels with nominated tracks."""
        usable = [
            item
            for item in detected_objects
            if isinstance(item, dict)
            and item.get("label")
            and _box(item.get("box")) is not None
        ]
        used: set[int] = set()
        verified: list[dict[str, Any]] = []
        diagonal = max(1.0, float(np.hypot(frame_width, frame_height)))
        ordered = sorted(
            expected_objects,
            key=lambda item: int(int(item.get("track_id") or 0) == primary_track_id),
            reverse=True,
        )
        for expected in ordered:
            expected_box = _box(expected.get("box"))
            if expected_box is None:
                continue
            expected_center = (
                (expected_box[0] + expected_box[2]) / 2.0,
                (expected_box[1] + expected_box[3]) / 2.0,
            )
            expected_diagonal_ratio = float(np.hypot(
                expected_box[2] - expected_box[0],
                expected_box[3] - expected_box[1],
            )) / diagonal
            distance_limit = max(0.04, min(0.12, expected_diagonal_ratio * 0.75))
            matches: list[tuple[float, float, int, dict[str, Any]]] = []
            for index, detected in enumerate(usable):
                if index in used or str(detected.get("label")) != str(expected.get("label")):
                    continue
                detected_box = _box(detected.get("box"))
                if detected_box is None:
                    continue
                overlap = _iou(expected_box, detected_box)
                detected_center = (
                    (detected_box[0] + detected_box[2]) / 2.0,
                    (detected_box[1] + detected_box[3]) / 2.0,
                )
                distance = float(np.hypot(
                    expected_center[0] - detected_center[0],
                    expected_center[1] - detected_center[1],
                )) / diagonal
                if overlap < 0.05 and distance > distance_limit:
                    continue
                score = (
                    2.0 * overlap
                    + max(0.0, 1.0 - distance / distance_limit)
                    + 0.1 * _confidence(detected)
                )
                matches.append((score, overlap, index, detected))
            matches.sort(key=lambda item: item[0], reverse=True)
            if not matches:
                continue
            if (
                len(matches) > 1
                and matches[0][1] < 0.25
                and matches[0][0] - matches[1][0] < 0.15
            ):
                continue
            _score, _overlap, index, detected = matches[0]
            used.add(index)
            verified.append({
                **expected,
                "box": dict(detected["box"]),
                "confidence": float(detected.get("confidence") or 0.0),
                "cover_verification_confidence": float(
                    detected.get("confidence") or 0.0
                ),
            })
        return verified

    def _run(
        self,
        event_id: int,
        event_at: datetime,
        initial_objects: list[dict[str, Any]],
        initial_frame: np.ndarray | None,
        stop: threading.Event,
    ) -> None:
        wait_started = time.monotonic()
        acquired = self.limiter.acquire(blocking=False)
        if not acquired and self.config.capacity_wait_seconds > 0:
            wait_deadline = wait_started + self.config.capacity_wait_seconds
            while not acquired and not stop.is_set():
                remaining = wait_deadline - time.monotonic()
                if remaining <= 0:
                    break
                acquired = self.limiter.acquire(timeout=min(0.1, remaining))
        capacity_wait_seconds = max(0.0, time.monotonic() - wait_started)
        with self._lock:
            requests = int(self._status.get("capacity_requests") or 0) + 1
            waits = int(self._status.get("capacity_waits") or 0)
            timeouts = int(self._status.get("capacity_timeouts") or 0)
            wait_total = float(self._status.get("capacity_wait_seconds_total") or 0.0)
            wait_max = float(self._status.get("capacity_wait_seconds_max") or 0.0)
            if capacity_wait_seconds >= 0.01:
                waits += 1
                wait_total += capacity_wait_seconds
                wait_max = max(wait_max, capacity_wait_seconds)
            if not acquired and not stop.is_set():
                timeouts += 1
            self._status = {
                **self._status,
                "capacity_requests": requests,
                "capacity_waits": waits,
                "capacity_timeouts": timeouts,
                "capacity_wait_seconds_total": round(wait_total, 3),
                "capacity_wait_seconds_max": round(wait_max, 3),
                "capacity_wait_seconds_last": round(capacity_wait_seconds, 3),
            }
        if stop.is_set() and not acquired:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
                    self._event_id = None
            return
        if not acquired:
            skipped = {
                "implementation": self.config.implementation,
                "state": "skipped_capacity",
                "sample_fps": self.config.sample_fps,
                "lost_timeout_seconds": self.config.lost_timeout_seconds,
                "frames_processed": 0,
                "catchup_frames_processed": 0,
                "capacity_wait_seconds": round(capacity_wait_seconds, 3),
                "capacity_wait_limit_seconds": self.config.capacity_wait_seconds,
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
            LOGGER.info(
                "object tracking skipped for %s event %d: capacity busy after %.3fs",
                self.camera.id,
                event_id,
                capacity_wait_seconds,
            )
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
                    self._event_id = None
            return
        tracker: ObjectTrackerBackend | None = None
        frames_processed = 0
        try:
            with self._lock:
                # Capacity waiting must not consume the useful tracking window.
                self._deadline = max(
                    self._deadline,
                    time.monotonic() + self.config.max_session_seconds,
                )
            self._frame_width = 0
            self._frame_height = 0
            self._catchup_frames_processed = 0
            self._coverage_gap_count = 0
            self._maximum_coverage_gap_seconds = 0.0
            self._completion_reason = ""
            self._cover_baseline = None
            self._cover_candidate = None
            self._cover_promotion = None
            self._set_status(
                cover_promoted=False,
                cover_promotion_attempted=False,
                cover_promotion_result="",
                cover_verification_inference_ms=0.0,
            )
            tracker = self.tracker_registry.create(
                self.config.implementation,
                self.config,
                float(self.detector.config.confidence_threshold),
            )
            if initial_frame is not None:
                self._frame_height = int(initial_frame.shape[0])
                self._frame_width = int(initial_frame.shape[1])
                self._annotate_appearances(
                    initial_frame,
                    initial_objects,
                    lazy=self.config.implementation == "survng_hybrid",
                )
            # Preserve the actual selected recording sample time. Consensus may
            # choose a frame up to one second on either side of the event.
            seed_offsets = []
            for detected in initial_objects:
                try:
                    offset = float(detected.get("temporal_sample_offset_seconds"))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(offset):
                    seed_offsets.append(offset)
            seed_epoch = event_at.timestamp() + (
                float(np.median(seed_offsets)) if seed_offsets else 0.0
            )
            captured_at = seed_epoch
            for detected in initial_objects:
                detected["_tracking_first_seen_at"] = seed_epoch
            initial_tracked = tracker.update(initial_objects, captured_at, confirm_new=True)
            primary_track_ids = {
                int(item["track_id"])
                for item in initial_tracked
                if item.get("track_id") is not None
                and item.get("snapshot_primary_subject") is True
            }
            if not primary_track_ids:
                primary_track_ids = {
                    int(item["track_id"])
                    for item in initial_tracked
                    if item.get("track_id") is not None
                }
            if initial_frame is not None:
                self._consider_cover_candidate(
                    initial_frame,
                    captured_at,
                    initial_tracked,
                    primary_track_ids,
                )
            self._set_status(enabled=True, active=True, event_id=event_id, last_error="")
            self._persist(
                event_id,
                tracker,
                captured_at,
                initial_tracked,
                frames_processed,
                "active",
                capacity_wait_seconds=capacity_wait_seconds,
            )
            interval = 1.0 / self.config.sample_fps
            consecutive_failures = 0

            def process_frame(
                frame: np.ndarray,
                sample_epoch: float,
                *,
                catchup: bool,
                frame_reference: VideoFrameReference | None = None,
            ) -> bool:
                nonlocal consecutive_failures, frames_processed
                source_height = int(frame.shape[0])
                source_width = int(frame.shape[1])
                if self._frame_width <= 0 or self._frame_height <= 0:
                    self._frame_width = source_width
                    self._frame_height = source_height
                objects = _detect_tracking_objects(
                    self.detector,
                    frame,
                    self.config.low_confidence_threshold,
                )
                if _inference_deferred(objects):
                    return False
                failure = detection_failure(objects)
                if failure:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        raise RuntimeError(f"tracking detector failed repeatedly: {failure}")
                    return False
                consecutive_failures = 0
                objects = [
                    item
                    for item in objects
                    if self.config.tracks_label(item.get("label"))
                ]
                apply_detection_zones(
                    self.camera,
                    objects,
                    source_width,
                    source_height,
                    float(self.detector.config.confidence_threshold),
                    bool(getattr(self.detector.config, "require_incident_zone", True)),
                )
                self._annotate_appearances(
                    frame,
                    objects,
                    lazy=self.config.implementation == "survng_hybrid",
                )
                _rescale_detection_boxes(
                    objects,
                    source_width,
                    source_height,
                    self._frame_width,
                    self._frame_height,
                )
                tracked = tracker.update(objects, sample_epoch)
                self._consider_cover_candidate(
                    frame,
                    sample_epoch,
                    tracked,
                    primary_track_ids,
                    frame_reference,
                )
                frames_processed += 1
                if catchup:
                    self._catchup_frames_processed += 1
                self._persist(event_id, tracker, sample_epoch, tracked, frames_processed, "active")
                return True

            def process_catchup_until(target_epoch: float) -> bool:
                """Consume newly finalized recording frames up to ``target_epoch``.

                The recorder index intentionally exposes only finalized segments. A
                delayed tracking session can therefore catch up to the end of the
                previous segment while the next segment is still being written. This
                helper is safe to call repeatedly as segments become available.
                """
                nonlocal captured_at
                if self.catchup_frame_provider is None or initial_frame is None:
                    return False
                catchup_start = captured_at + interval
                if target_epoch <= catchup_start:
                    return False
                advanced = False
                catchup_frames = iter(
                    self.catchup_frame_provider(
                        catchup_start,
                        target_epoch,
                        self.config.sample_fps,
                        min(1280, int(initial_frame.shape[1])),
                    )
                )
                try:
                    for sample in catchup_frames:
                        sample_epoch, frame = sample
                        frame_reference = getattr(sample, "reference", None)
                        if stop.is_set() or time.monotonic() >= self._deadline:
                            break
                        if sample_epoch <= captured_at or sample_epoch > target_epoch:
                            continue
                        process_frame(
                            frame,
                            sample_epoch,
                            catchup=True,
                            frame_reference=frame_reference,
                        )
                        captured_at = sample_epoch
                        advanced = True
                finally:
                    close_catchup = getattr(catchup_frames, "close", None)
                    if callable(close_catchup):
                        close_catchup()
                return advanced

            catchup_until = time.time()
            if (
                self.catchup_frame_provider is not None
                and initial_frame is not None
                and catchup_until - captured_at > interval * 1.5
            ):
                # Start immediately after the actual selected sample. The
                # provider and loop still reject non-increasing timestamps.
                try:
                    process_catchup_until(catchup_until)
                except Exception:
                    LOGGER.exception(
                        "recorded tracking catch-up failed for %s event %d; continuing live",
                        self.camera.id,
                        event_id,
                    )

            next_sample = time.monotonic()
            frame_acquisition_deadline = min(
                self._deadline,
                time.monotonic() + 5.0,
            )
            last_frame_token: float | None = None
            # If recorded evidence already followed every confirmed track until
            # it naturally expired, the event has complete useful coverage.
            # Bridging to a much newer live frame would manufacture a timestamp
            # gap after the object left and incorrectly suppress cover promotion.
            live_bridge_required = tracker.has_live_tracks(captured_at)
            if not live_bridge_required:
                self._completion_reason = "object_exited_recorded_window"
            while live_bridge_required and not stop.is_set():
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
                if self._catchup_frames_processed and sample_epoch <= captured_at:
                    next_sample = time.monotonic() + interval
                    continue
                coverage_gap = sample_epoch - captured_at
                gap_backfilled = False
                if (
                    self.catchup_frame_provider is not None
                    and initial_frame is not None
                    and coverage_gap > self.config.lost_timeout_seconds
                ):
                    # A live frame far ahead of the last recorded sample would age
                    # every track out immediately. Give the recorder's currently-open
                    # segment a bounded opportunity to finalize, then replay the gap
                    # in timestamp order before returning to live frames.
                    settle_deadline = min(
                        deadline,
                        time.monotonic() + TRACKING_CATCHUP_SETTLE_SECONDS,
                    )
                    while (
                        not stop.is_set()
                        and sample_epoch - captured_at > interval * 1.5
                        and time.monotonic() < settle_deadline
                    ):
                        try:
                            gap_backfilled = bool(
                                process_catchup_until(sample_epoch)
                                or gap_backfilled
                            )
                        except Exception:
                            LOGGER.exception(
                                "recorded tracking gap backfill failed for %s event %d",
                                self.camera.id,
                                event_id,
                            )
                            break
                        if sample_epoch - captured_at <= interval * 1.5:
                            break
                        wait_for = min(
                            TRACKING_CATCHUP_RETRY_SECONDS,
                            max(0.0, settle_deadline - time.monotonic()),
                        )
                        if wait_for <= 0.0 or stop.wait(wait_for):
                            break
                    coverage_gap = sample_epoch - captured_at
                    if (
                        coverage_gap > self.config.lost_timeout_seconds
                        and tracker.has_live_tracks(captured_at)
                    ):
                        self._coverage_gap_count += 1
                        self._maximum_coverage_gap_seconds = max(
                            self._maximum_coverage_gap_seconds,
                            coverage_gap,
                        )
                        LOGGER.warning(
                            "object tracking coverage gap for %s event %d: %.3fs",
                            self.camera.id,
                            event_id,
                            coverage_gap,
                        )
                        self._completion_reason = "missing_media_while_object_active"
                    elif not tracker.has_live_tracks(captured_at):
                        self._completion_reason = "object_exited_during_catchup"
                        break
                if gap_backfilled and sample_epoch <= captured_at + interval * 0.5:
                    last_frame_token = frame_token
                    next_sample = time.monotonic() + interval
                    continue
                if last_frame_token is not None and frame_token <= last_frame_token:
                    if not tracker.has_live_tracks(now_epoch):
                        break
                    next_sample = time.monotonic() + interval
                    continue
                last_frame_token = frame_token
                if not process_frame(frame, sample_epoch, catchup=False):
                    next_sample = time.monotonic() + interval
                    continue
                captured_at = sample_epoch
                if not tracker.has_live_tracks(sample_epoch):
                    self._completion_reason = "object_exited_live_window"
                    break
                next_sample = max(next_sample + interval, time.monotonic())
            final_epoch = time.time()
            final_state = "interrupted" if self._coverage_gap_count else "complete"
            if not self._completion_reason:
                self._completion_reason = (
                    "session_stopped" if stop.is_set() else "tracking_window_complete"
                )
            if final_state == "complete" and not stop.is_set():
                try:
                    self._promote_cover_candidate(event_id)
                except Exception:
                    LOGGER.exception(
                        "tracked cover promotion failed for %s event %d",
                        self.camera.id,
                        event_id,
                    )
            self._persist(event_id, tracker, final_epoch, None, frames_processed, final_state)
            self._set_status(
                enabled=True,
                active=False,
                event_id=event_id,
                track_count=len(tracker.summaries(final_epoch)),
                confirmed_tracks=len(tracker.summaries(final_epoch)),
                frames_processed=frames_processed,
                catchup_frames_processed=self._catchup_frames_processed,
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
        capacity_wait_seconds: float | None = None,
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
            "capacity_wait_seconds": round(
                float(
                    capacity_wait_seconds
                    if capacity_wait_seconds is not None
                    else self._status.get("capacity_wait_seconds_last") or 0.0
                ),
                3,
            ),
            "capacity_wait_limit_seconds": self.config.capacity_wait_seconds,
            "lost_timeout_seconds": self.config.lost_timeout_seconds,
            "frames_processed": frames_processed,
            "catchup_frames_processed": self._catchup_frames_processed,
            "coverage_gap_count": self._coverage_gap_count,
            "maximum_coverage_gap_seconds": round(
                self._maximum_coverage_gap_seconds,
                3,
            ),
            "coverage_incomplete": self._coverage_gap_count > 0,
            "completion_reason": self._completion_reason,
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
        if self._cover_promotion:
            payload.update(self._cover_promotion)
        if self._frame_width > 0 and self._frame_height > 0:
            payload["frame_width"] = self._frame_width
            payload["frame_height"] = self._frame_height
        if self.update_event(event_id, payload, tracked_objects) is None:
            raise RuntimeError(f"tracking event {event_id} no longer exists")
        if state in {"complete", "interrupted"}:
            self._index_track_appearances(event_id, tracker)
        self._set_status(
            enabled=True,
            active=state == "active",
            event_id=event_id,
            track_count=len(tracks),
            confirmed_tracks=len(tracks),
            frames_processed=frames_processed,
            catchup_frames_processed=self._catchup_frames_processed,
            coverage_gap_count=self._coverage_gap_count,
            maximum_coverage_gap_seconds=round(
                self._maximum_coverage_gap_seconds,
                3,
            ),
            coverage_incomplete=self._coverage_gap_count > 0,
            completion_reason=self._completion_reason,
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
            **(self._cover_promotion or {}),
        )
        if state in {"complete", "interrupted"}:
            self._publish_safely(event_id, payload)

    def _index_track_appearances(
        self,
        event_id: int,
        tracker: ObjectTrackerBackend,
    ) -> None:
        if self.appearance_indexer is None or self.appearance_encoder is None:
            return
        records_method = getattr(tracker, "appearance_records", None)
        identity_method = getattr(
            self.appearance_encoder,
            "model_identity_for_label",
            None,
        )
        if not callable(records_method) or not callable(identity_method):
            return
        prepared: list[dict[str, Any]] = []
        for record in records_method():
            identity = identity_method(str(record.get("label") or ""))
            if not isinstance(identity, dict) or not identity.get("model_fingerprint"):
                continue
            prepared.append({
                **record,
                "model_kind": identity.get("model_kind"),
                "model_fingerprint": identity.get("model_fingerprint"),
                "match_threshold": identity.get("match_threshold"),
                "created_at": record.get("last_seen"),
            })
        if not prepared:
            return
        try:
            indexed = self.appearance_indexer(
                event_id,
                self.camera.id,
                prepared,
            )
        except Exception as exc:
            error = redact_secret_text(exc)[:240]
            self._set_status(appearance_index_error=error)
            LOGGER.exception(
                "failed to index object appearances for %s event %d",
                self.camera.id,
                event_id,
            )
            return
        self._set_status(
            appearance_vectors_indexed=int(indexed),
            appearance_index_error="",
        )

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
            "capacity_wait_seconds": round(
                float(self._status.get("capacity_wait_seconds_last") or 0.0),
                3,
            ),
            "capacity_wait_limit_seconds": self.config.capacity_wait_seconds,
            "lost_timeout_seconds": self.config.lost_timeout_seconds,
            "frames_processed": frames_processed,
            "catchup_frames_processed": self._catchup_frames_processed,
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
            self.publisher(
                "object_tracking",
                TrackingCompleted(
                    event_id=event_id,
                    camera_id=self.camera.id,
                    state=str(payload.get("state") or "complete"),
                    details=payload,
                ).to_payload(),
            )
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
        appearance_indexer: AppearanceIndexWriter | None = None,
        cover_promoter: TrackingCoverPromoter | None = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.update_event = update_event
        self.publisher = publisher
        self.limiter = limiter
        self.tracker_registry = tracker_registry or build_builtin_object_tracker_registry()
        self.appearance_encoder = appearance_encoder
        self.appearance_indexer = appearance_indexer
        self.cover_promoter = cover_promoter
        # Fail configuration loading before any event tries to start a session.
        self.tracker_registry.require(config.implementation)

    def create(
        self,
        camera: CameraConfig,
        frame_provider: FrameProvider,
        catchup_frame_provider: CatchupFrameProvider | None = None,
        cover_frame_provider: TrackingCoverFrameProvider | None = None,
        snapshot_writer: TrackingSnapshotWriter | None = None,
    ) -> ObjectTrackingSession:
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
            appearance_indexer=self.appearance_indexer,
            catchup_frame_provider=catchup_frame_provider,
            cover_frame_provider=cover_frame_provider,
            snapshot_writer=snapshot_writer,
            cover_promoter=self.cover_promoter,
        )

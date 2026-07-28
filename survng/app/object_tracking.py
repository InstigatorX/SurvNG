from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
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


ObjectTrackerBuilder = Callable[[ObjectTrackingConfig, float], ObjectTrackerBackend]


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


def _iou(left: Box, right: Box) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1.0, left_area + right_area - intersection)


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
        captured_at = max(captured_at, self.last_seen)
        elapsed = max(1e-3, captured_at - self.last_seen)
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
        self.confidence = float(detection.get("confidence") or 0.0)
        self.max_confidence = max(self.max_confidence, self.confidence)
        self.hits += 1
        self.missed = 0
        self.zones.update(str(zone) for zone in detection.get("zones", []) if zone)
        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[1] + box[3]) / 2.0
        self.trajectory.append((round(captured_at, 3), round(center_x, 1), round(center_y, 1)))
        if len(self.trajectory) > 60:
            del self.trajectory[:-60]

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
            or float(item[1].get("confidence") or 0.0) >= self.high_confidence_threshold
        ]
        low = [
            item
            for item in usable
            if not confirm_new
            and self.config.low_confidence_threshold
            <= float(item[1].get("confidence") or 0.0)
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
            confidence = float(detection.get("confidence") or 0.0)
            track = ObjectTrack(
                track_id=self._next_track_id,
                label=str(detection["label"]),
                box=box,
                first_seen=captured_at,
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
            )
            self._tracks[track.track_id] = track
            assignments[index] = track.track_id
            self._next_track_id += 1

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
                **detection,
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
                score = _iou(track.predicted_box(captured_at), box)
                if score >= self.config.match_iou_threshold:
                    candidates.append((score, track_id, index, detection, box))
        used_detections: set[int] = set()
        for _score, track_id, index, detection, box in sorted(candidates, reverse=True):
            if track_id not in unmatched_tracks or index in used_detections:
                continue
            track = self._tracks[track_id]
            track.observe(detection, captured_at, box)
            track.confirmed = track.confirmed or track.hits >= self.config.min_confirmations
            unmatched_tracks.remove(track_id)
            used_detections.add(index)
            assignments[index] = track_id

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


def build_builtin_object_tracker_registry() -> ObjectTrackerRegistry:
    registry = ObjectTrackerRegistry()
    registry.register("bytetrack", ByteTrackObjectTracker)
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
    ) -> None:
        self.camera = camera
        self.config = config
        self.detector = detector
        self.frame_provider = frame_provider
        self.update_event = update_event
        self.publisher = publisher
        self.limiter = limiter
        self.tracker_registry = tracker_registry or build_builtin_object_tracker_registry()
        self._lock = threading.RLock()
        self._transition_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._event_id: int | None = None
        self._deadline = 0.0
        self._accepting = False
        self._status: dict[str, Any] = self._idle_status()
        self._status["enabled"] = config.enabled

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
        }

    def start(
        self,
        event_id: int,
        event_at: datetime,
        initial_objects: list[dict[str, Any]],
    ) -> bool:
        with self._transition_lock:
            return self._start_session(event_id, event_at, initial_objects)

    def _start_session(
        self,
        event_id: int,
        event_at: datetime,
        initial_objects: list[dict[str, Any]],
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
            thread = threading.Thread(
                target=self._run,
                args=(event_id, event_at, [dict(item) for item in initial_objects], self._stop),
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
            captured_at = event_at.timestamp()
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
        payload = {
            "implementation": self.config.implementation,
            "state": state,
            "sample_fps": self.config.sample_fps,
            "frames_processed": frames_processed,
            "updated_at": datetime.fromtimestamp(captured_at, timezone.utc).isoformat(),
            "tracks": tracks,
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


class ObjectTrackingSessionFactory:
    def __init__(
        self,
        config: ObjectTrackingConfig,
        detector: ObjectDetectorBackend,
        update_event: TrackingUpdate,
        publisher: TrackingPublisher | None,
        limiter: threading.BoundedSemaphore,
        tracker_registry: ObjectTrackerRegistry | None = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.update_event = update_event
        self.publisher = publisher
        self.limiter = limiter
        self.tracker_registry = tracker_registry or build_builtin_object_tracker_registry()
        # Fail configuration loading before any event tries to start a session.
        self.tracker_registry.require(config.implementation)

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
        )

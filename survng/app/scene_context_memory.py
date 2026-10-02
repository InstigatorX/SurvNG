"""Per-camera memory of subjects repeatedly seen at the same place.

Live admission and recorded analysis share one memory. An in-memory memory
keeps today's process-local behavior. A durable memory hydrates that working
set from the event store and writes through only when the set of sightings
changes, a subject is forgotten, or an expired row is pruned.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol


def normalized_box(
    observation: Mapping[str, Any],
) -> tuple[float, float, float, float] | None:
    """Normalize a detection box by the frame size stored on the observation."""
    box = observation.get("box")
    try:
        width = float(observation.get("detection_frame_width") or 0.0)
        height = float(observation.get("detection_frame_height") or 0.0)
        if not isinstance(box, Mapping) or width <= 0 or height <= 0:
            return None
        normalized = (
            float(box["x1"]) / width,
            float(box["y1"]) / height,
            float(box["x2"]) / width,
            float(box["y2"]) / height,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        return None
    if not all(math.isfinite(value) for value in normalized):
        return None
    return normalized


def prior_sightings(event_keys: list[str], event_key: str) -> int:
    """Sightings from other events. The current event is not a prior."""
    if not event_key:
        return len(event_keys)
    return sum(1 for key in event_keys if key != event_key)


def subject_visible(
    subject: SceneContextSubject,
    event_key: str,
    observed_at_epoch: float,
) -> bool:
    """A subject still applies at this instant, including later frames of its event."""
    if subject.last_seen_epoch <= observed_at_epoch:
        return True
    return bool(event_key) and event_key in subject.stable_event_keys


def merge_subjects(
    subjects: list[SceneContextSubject],
    *,
    event_key: str,
    box: tuple[float, float, float, float],
    observed_at_epoch: float,
    max_sightings: int,
) -> tuple[SceneContextSubject, list[SceneContextSubject]]:
    """Collapse overlapping rows for one car into the earliest subject."""
    survivor = min(subjects, key=lambda subject: (subject.first_seen_epoch, subject.row_id or 0))
    keys: list[str] = []
    for subject in sorted(subjects, key=lambda subject: subject.first_seen_epoch):
        for key in subject.stable_event_keys:
            if key not in keys:
                keys.append(key)
    if event_key and event_key not in keys:
        keys.append(event_key)
    survivor.stable_event_keys = keys[-max_sightings:]
    survivor.box = box
    survivor.first_seen_epoch = min(subject.first_seen_epoch for subject in subjects)
    survivor.last_seen_epoch = observed_at_epoch
    moved = [subject.last_moved_epoch for subject in subjects if subject.last_moved_epoch is not None]
    if moved:
        survivor.last_moved_epoch = max(moved)
    return survivor, [subject for subject in subjects if subject is not survivor]


class SceneContextSnapshot:
    """Read-only subjects already loaded by the caller's open transaction.

    Establishment runs while that transaction holds the event-store lock, so
    this match does not touch the store or prune rows.
    """

    def __init__(self, subjects: list[SceneContextSubject]) -> None:
        self._subjects = list(subjects)

    def match(
        self,
        *,
        camera_id: str,
        label: str,
        box: tuple[float, float, float, float],
        event_key: str,
        observed_at_epoch: float,
        min_iou: float,
        ttl_seconds: float,
    ) -> tuple[bool, int, float | None]:
        del camera_id
        cutoff = observed_at_epoch - ttl_seconds
        matches = [
            subject
            for subject in self._subjects
            if subject.label == label
            and subject.last_seen_epoch >= cutoff
            and subject_visible(subject, event_key, observed_at_epoch)
            and box_iou(subject.box, box) >= min_iou
        ]
        if not matches:
            return False, 0, None
        found = max(
            matches,
            key=lambda subject: (
                prior_sightings(subject.stable_event_keys, event_key),
                box_iou(subject.box, box),
            ),
        )
        return (
            True,
            prior_sightings(found.stable_event_keys, event_key),
            max(0.0, observed_at_epoch - found.last_seen_epoch),
        )


def box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(slots=True)
class SceneContextSubject:
    label: str
    box: tuple[float, float, float, float]
    last_seen_epoch: float
    first_seen_epoch: float
    stable_event_keys: list[str] = field(default_factory=list)
    row_id: int | None = None
    last_moved_epoch: float | None = None

    @property
    def stable_sightings(self) -> int:
        return len(self.stable_event_keys)


class SceneContextMemory(Protocol):
    writes: int

    def hold(self) -> Iterator[None]: ...

    def match(
        self,
        *,
        camera_id: str,
        label: str,
        box: tuple[float, float, float, float],
        event_key: str,
        observed_at_epoch: float,
        min_iou: float,
        ttl_seconds: float,
    ) -> tuple[bool, int, float | None]: ...

    def remember(
        self,
        *,
        camera_id: str,
        label: str,
        box: tuple[float, float, float, float],
        event_key: str,
        observed_at_epoch: float,
        min_iou: float,
        ttl_seconds: float,
        max_entries: int,
        max_sightings: int,
    ) -> None: ...

    def forget(
        self,
        *,
        camera_id: str,
        label: str,
        box: tuple[float, float, float, float],
        event_key: str,
        invalidate_location: bool,
        min_iou: float,
    ) -> None: ...

    def clear(self, camera_id: str = "") -> None: ...

    def count(self, camera_id: str = "") -> int: ...


class InMemorySceneContextMemory:
    """Process-local subjects. ``camera_id`` keeps cameras from sharing a box."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subjects: list[tuple[str, SceneContextSubject]] = []
        self.writes = 0

    @contextmanager
    def hold(self) -> Iterator[None]:
        with self._lock:
            yield

    def match(
        self,
        *,
        camera_id: str,
        label: str,
        box: tuple[float, float, float, float],
        event_key: str,
        observed_at_epoch: float,
        min_iou: float,
        ttl_seconds: float,
    ) -> tuple[bool, int, float | None]:
        with self._lock:
            self._prune(camera_id, observed_at_epoch, ttl_seconds)
            matches = [
                subject
                for stored_camera, subject in self._subjects
                if stored_camera == camera_id
                and subject.label == label
                and subject_visible(subject, event_key, observed_at_epoch)
                and box_iou(subject.box, box) >= min_iou
            ]
            if not matches:
                return False, 0, None
            match = max(
                matches,
                key=lambda subject: (
                    prior_sightings(subject.stable_event_keys, event_key),
                    box_iou(subject.box, box),
                ),
            )
            return (
                True,
                prior_sightings(match.stable_event_keys, event_key),
                max(0.0, observed_at_epoch - match.last_seen_epoch),
            )

    def remember(
        self,
        *,
        camera_id: str,
        label: str,
        box: tuple[float, float, float, float],
        event_key: str,
        observed_at_epoch: float,
        min_iou: float,
        ttl_seconds: float,
        max_entries: int,
        max_sightings: int,
    ) -> None:
        with self._lock:
            self._prune(camera_id, observed_at_epoch, ttl_seconds)
            overlapping = [
                subject
                for stored_camera, subject in self._subjects
                if stored_camera == camera_id
                and subject.label == label
                and subject_visible(subject, event_key, observed_at_epoch)
                and box_iou(subject.box, box) >= min_iou
            ]
            if overlapping:
                _survivor, extras = merge_subjects(
                    overlapping,
                    event_key=event_key,
                    box=box,
                    observed_at_epoch=observed_at_epoch,
                    max_sightings=max_sightings,
                )
                drop = {id(subject) for subject in extras}
                self._subjects = [
                    item for item in self._subjects if id(item[1]) not in drop
                ]
            else:
                self._subjects.append((
                    camera_id,
                    SceneContextSubject(
                        label, box, observed_at_epoch, observed_at_epoch, [event_key],
                    ),
                ))
            camera_subjects = [
                (index, subject)
                for index, (stored_camera, subject) in enumerate(self._subjects)
                if stored_camera == camera_id
            ]
            if len(camera_subjects) > max_entries:
                camera_subjects.sort(key=lambda item: item[1].last_seen_epoch)
                drop = {index for index, _ in camera_subjects[:-max_entries]}
                self._subjects = [
                    item for index, item in enumerate(self._subjects) if index not in drop
                ]

    def forget(
        self,
        *,
        camera_id: str,
        label: str,
        box: tuple[float, float, float, float],
        event_key: str,
        invalidate_location: bool,
        min_iou: float,
    ) -> None:
        with self._lock:
            retained: list[tuple[str, SceneContextSubject]] = []
            for stored_camera, subject in self._subjects:
                matches = bool(
                    stored_camera == camera_id
                    and subject.label == label
                    and box_iou(subject.box, box) >= min_iou
                )
                if matches and invalidate_location:
                    continue
                if matches and event_key in subject.stable_event_keys:
                    subject.stable_event_keys.remove(event_key)
                if subject.stable_event_keys:
                    retained.append((stored_camera, subject))
            self._subjects[:] = retained

    def clear(self, camera_id: str = "") -> None:
        with self._lock:
            if camera_id:
                self._subjects = [
                    item for item in self._subjects if item[0] != camera_id
                ]
            else:
                self._subjects.clear()

    def count(self, camera_id: str = "") -> int:
        with self._lock:
            if not camera_id:
                return len(self._subjects)
            return sum(1 for stored, _ in self._subjects if stored == camera_id)

    def _prune(self, camera_id: str, observed_at_epoch: float, ttl_seconds: float) -> None:
        cutoff = observed_at_epoch - ttl_seconds
        self._subjects = [
            item
            for item in self._subjects
            if item[0] != camera_id or item[1].last_seen_epoch >= cutoff
        ]

class DurableSceneContextMemory:
    """Working set for one camera, persisted in ``scene_context_subjects``."""

    def __init__(self, store: Any, camera_id: str) -> None:
        self._store = store
        self.camera_id = camera_id
        self._lock = threading.RLock()
        self._subjects: list[SceneContextSubject] = []
        self.writes = 0
        for row in store.load_scene_context(camera_id):
            self._subjects.append(SceneContextSubject(
                label=str(row["label"]),
                box=(float(row["box_x1"]), float(row["box_y1"]), float(row["box_x2"]), float(row["box_y2"])),
                last_seen_epoch=float(row["last_seen_epoch"]),
                first_seen_epoch=float(row["first_seen_epoch"]),
                stable_event_keys=list(row["sightings"]),
                row_id=int(row["id"]) if row.get("id") is not None else None,
                last_moved_epoch=row.get("last_moved_epoch"),
            ))

    @contextmanager
    def hold(self) -> Iterator[None]:
        with self._lock:
            yield

    def match(
        self,
        *,
        camera_id: str,
        label: str,
        box: tuple[float, float, float, float],
        event_key: str,
        observed_at_epoch: float,
        min_iou: float,
        ttl_seconds: float,
    ) -> tuple[bool, int, float | None]:
        if camera_id != self.camera_id:
            return False, 0, None
        with self._lock:
            self._prune(observed_at_epoch, ttl_seconds)
            matches = [
                subject
                for subject in self._subjects
                if subject.label == label
                and subject_visible(subject, event_key, observed_at_epoch)
                and box_iou(subject.box, box) >= min_iou
            ]
            if not matches:
                return False, 0, None
            match = max(
                matches,
                key=lambda subject: (
                    prior_sightings(subject.stable_event_keys, event_key),
                    box_iou(subject.box, box),
                ),
            )
            return (
                True,
                prior_sightings(match.stable_event_keys, event_key),
                max(0.0, observed_at_epoch - match.last_seen_epoch),
            )

    def remember(
        self,
        *,
        camera_id: str,
        label: str,
        box: tuple[float, float, float, float],
        event_key: str,
        observed_at_epoch: float,
        min_iou: float,
        ttl_seconds: float,
        max_entries: int,
        max_sightings: int,
    ) -> None:
        if camera_id != self.camera_id:
            return
        with self._lock:
            self._prune(observed_at_epoch, ttl_seconds)
            overlapping = [
                subject
                for subject in self._subjects
                if subject.label == label
                and subject_visible(subject, event_key, observed_at_epoch)
                and box_iou(subject.box, box) >= min_iou
            ]
            if overlapping:
                had_key = any(event_key in subject.stable_event_keys for subject in overlapping)
                survivor, extras = merge_subjects(
                    overlapping,
                    event_key=event_key,
                    box=box,
                    observed_at_epoch=observed_at_epoch,
                    max_sightings=max_sightings,
                )
                for extra in extras:
                    self._delete(extra)
                drop = {id(subject) for subject in extras}
                self._subjects = [subject for subject in self._subjects if id(subject) not in drop]
                if not had_key or extras:
                    self._persist(survivor)
            else:
                subject = SceneContextSubject(
                    label, box, observed_at_epoch, observed_at_epoch, [event_key],
                )
                self._subjects.append(subject)
                self._persist(subject)
            if len(self._subjects) > max_entries:
                self._subjects.sort(key=lambda subject: subject.last_seen_epoch)
                dropped = self._subjects[:-max_entries]
                self._subjects = self._subjects[-max_entries:]
                for subject in dropped:
                    self._delete(subject)

    def forget(
        self,
        *,
        camera_id: str,
        label: str,
        box: tuple[float, float, float, float],
        event_key: str,
        invalidate_location: bool,
        min_iou: float,
    ) -> None:
        if camera_id != self.camera_id:
            return
        with self._lock:
            retained: list[SceneContextSubject] = []
            for subject in self._subjects:
                matches = bool(
                    subject.label == label and box_iou(subject.box, box) >= min_iou
                )
                if matches and invalidate_location:
                    subject.last_moved_epoch = subject.last_seen_epoch
                    self._delete(subject)
                    continue
                if matches and event_key in subject.stable_event_keys:
                    subject.stable_event_keys.remove(event_key)
                    if subject.stable_event_keys:
                        self._persist(subject)
                if subject.stable_event_keys:
                    retained.append(subject)
                elif matches:
                    self._delete(subject)
            self._subjects = retained

    def clear(self, camera_id: str = "") -> None:
        if camera_id and camera_id != self.camera_id:
            return
        with self._lock:
            self._subjects.clear()
            self.writes += 1
            self._store.clear_scene_context(self.camera_id)

    def count(self, camera_id: str = "") -> int:
        if camera_id and camera_id != self.camera_id:
            return 0
        with self._lock:
            return len(self._subjects)

    def _prune(self, observed_at_epoch: float, ttl_seconds: float) -> None:
        cutoff = observed_at_epoch - ttl_seconds
        retained: list[SceneContextSubject] = []
        for subject in self._subjects:
            if subject.last_seen_epoch >= cutoff:
                retained.append(subject)
            else:
                self._delete(subject)
        self._subjects = retained

    def _persist(self, subject: SceneContextSubject) -> None:
        self.writes += 1
        subject.row_id = self._store.save_scene_context(
            self.camera_id,
            {
                "id": subject.row_id,
                "label": subject.label,
                "box": subject.box,
                "first_seen_epoch": subject.first_seen_epoch,
                "last_seen_epoch": subject.last_seen_epoch,
                "sightings": list(subject.stable_event_keys),
                "last_moved_epoch": subject.last_moved_epoch,
            },
        )

    def _delete(self, subject: SceneContextSubject) -> None:
        if subject.row_id is None:
            return
        self.writes += 1
        self._store.delete_scene_context(self.camera_id, subject.row_id)
        subject.row_id = None

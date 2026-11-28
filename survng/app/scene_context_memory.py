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


# Detectors routinely flip one vehicle between these classes from one incident
# to the next. They match one another so that a parked vehicle is learned once;
# every other label must match exactly. The stored label keeps the first class
# seen.
INTERCHANGEABLE_SUBJECT_LABELS = frozenset({"car", "truck", "bus"})
# Each new sighting moves a subject's reference box a fifth of the way toward
# its own box. A single detection's box error of 10-15% then shifts the
# reference by only 2-3%, so sightings that err in opposite directions still
# match it, while a subject that has genuinely settled into a slightly
# different box is followed within a handful of sightings.
REFERENCE_BOX_WEIGHT = 0.2


def same_subject_label(stored: str, observed: str) -> bool:
    """Whether two detector labels may describe one remembered subject."""
    return stored == observed or (
        stored in INTERCHANGEABLE_SUBJECT_LABELS
        and observed in INTERCHANGEABLE_SUBJECT_LABELS
    )


def blend_reference_box(
    reference: tuple[float, float, float, float],
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (
        old * (1.0 - REFERENCE_BOX_WEIGHT) + new * REFERENCE_BOX_WEIGHT
        for old, new in zip(reference, box)
    )
    return x1, y1, x2, y2


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


def sighting_epoch(subject: SceneContextSubject, event_key: str) -> float:
    """When a sighting was recorded.

    A ledger written before sightings carried times has none; such a sighting
    dates from the subject's last sighting, which every in-order event follows.
    """
    return subject.sighting_epochs.get(event_key, subject.last_seen_epoch)


def prior_sightings(
    subject: SceneContextSubject,
    event_key: str,
    observed_at_epoch: float,
) -> int:
    """Sightings from other events at or before this one. The current event is not a prior."""
    return sum(
        1
        for key in subject.stable_event_keys
        if key != event_key and sighting_epoch(subject, key) <= observed_at_epoch
    )


def subject_visible(
    subject: SceneContextSubject,
    event_key: str,
    observed_at_epoch: float,
) -> bool:
    """A subject applies to an event it was already known before.

    Recorded analysis of an event often finishes after a later live event
    has refreshed the subject, so what counts is when the subject was first
    seen, not when it was last seen. A subject first seen after the event,
    such as a car that arrived in it and then parked, does not apply to it.
    Later frames of an event the subject already records still apply.

    An out-of-order event counts only the subject's sightings at or before
    it (see prior_sightings). A later sighting is not evidence that the
    subject was already there: a car that left unseen and another that parked in the same
    bay are one subject to the memory. The count can still exceed what
    in-order analysis would have given when a later event merged two
    overlapping subjects, but every sighting it counts came before the event.
    The event's own TTL cutoff is older
    than the newest event's, so it never prunes a subject that the newest
    sighting keeps alive.
    """
    if subject.first_seen_epoch <= observed_at_epoch:
        return True
    return bool(event_key) and event_key in subject.stable_event_keys


def revoke_sighting(subject: SceneContextSubject, event_key: str) -> bool:
    """Remove one event's sighting; True when the subject recorded it.

    The subject is then dated from its first remaining sighting, so an event
    between the revoked sighting and that one is no longer judged against it.
    """
    if event_key not in subject.stable_event_keys:
        return False
    subject.stable_event_keys.remove(event_key)
    subject.sighting_epochs.pop(event_key, None)
    if subject.stable_event_keys:
        subject.first_seen_epoch = max(
            subject.first_seen_epoch,
            min(sighting_epoch(subject, key) for key in subject.stable_event_keys),
        )
    return True


def merge_subjects(
    subjects: list[SceneContextSubject],
    *,
    event_key: str,
    box: tuple[float, float, float, float],
    observed_at_epoch: float,
    max_sightings: int,
) -> tuple[SceneContextSubject, list[SceneContextSubject], bool]:
    """Collapse overlapping rows for one car into the earliest subject.

    Returns the survivor, the merged-away subjects, and whether this event
    was added as a new sighting.

    The survivor keeps a stable reference box. Only a new sighting, an event
    not already recorded, blends its box in; later frames of a recorded event
    leave it unchanged, so the reference changes only with the sighting set.
    The last sighting time follows the same rule and never moves backwards:
    an older event remembered late keeps the newest time, which drives the
    TTL and eviction, and a durable subject reloads exactly as it is held.
    A full set takes a new sighting only from an event strictly newer than
    its oldest kept sighting, so an event of the same time cannot swap with
    it on every frame. An event whose sighting was trimmed is not taken
    back: a later frame of it would otherwise be dated as the newest sighting
    and trim another in turn.
    """
    survivor = min(subjects, key=lambda subject: (subject.first_seen_epoch, subject.row_id or 0))
    keys: list[str] = []
    epochs: dict[str, float] = {}
    dated: dict[str, float] = {}
    for subject in sorted(subjects, key=lambda subject: subject.first_seen_epoch):
        for key in subject.stable_event_keys:
            if key not in keys:
                keys.append(key)
            when = sighting_epoch(subject, key)
            dated[key] = min(dated.get(key, when), when)
            if key in subject.sighting_epochs:
                epochs[key] = min(epochs.get(key, when), when)
    last_seen = max(subject.last_seen_epoch for subject in subjects)
    evicted = list(dict.fromkeys(
        key for subject in subjects for key in subject.evicted_event_keys
        if key not in keys
    ))

    def latest(candidates: list[str]) -> list[str]:
        # Keep the latest sightings by time; a late, older event goes first.
        if len(candidates) <= max_sightings:
            return candidates
        kept = set(sorted(candidates, key=lambda key: dated[key])[-max_sightings:])
        evicted.extend(key for key in candidates if key not in kept)
        return [key for key in candidates if key in kept]

    keys = latest(keys)
    added = False
    if (
        event_key
        and event_key not in keys
        and event_key not in evicted
        and (
            len(keys) < max_sightings
            or observed_at_epoch > min(dated[key] for key in keys)
        )
    ):
        added = True
        keys.append(event_key)
        dated[event_key] = observed_at_epoch
        epochs[event_key] = observed_at_epoch
        survivor.box = blend_reference_box(survivor.box, box)
        last_seen = max(last_seen, observed_at_epoch)
        keys = latest(keys)
    survivor.evicted_event_keys = evicted[-max_sightings:]
    survivor.stable_event_keys = keys
    survivor.sighting_epochs = {key: epochs[key] for key in keys if key in epochs}
    survivor.first_seen_epoch = min(subject.first_seen_epoch for subject in subjects)
    survivor.last_seen_epoch = last_seen
    moved = [subject.last_moved_epoch for subject in subjects if subject.last_moved_epoch is not None]
    if moved:
        survivor.last_moved_epoch = max(moved)
    return survivor, [subject for subject in subjects if subject is not survivor], added


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
            if same_subject_label(subject.label, label)
            and subject.last_seen_epoch >= cutoff
            and subject_visible(subject, event_key, observed_at_epoch)
            and box_iou(subject.box, box) >= min_iou
        ]
        if not matches:
            return False, 0, None
        found = max(
            matches,
            key=lambda subject: (
                prior_sightings(subject, event_key, observed_at_epoch),
                box_iou(subject.box, box),
            ),
        )
        return (
            True,
            prior_sightings(found, event_key, observed_at_epoch),
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
    sighting_epochs: dict[str, float] = field(default_factory=dict)
    # Events whose sighting a full set trimmed, newest last and bounded by the
    # sighting cap. Held in the working set only: after a restart an evicted
    # event can be taken back once, then it is recorded again.
    evicted_event_keys: list[str] = field(default_factory=list)


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
                and same_subject_label(subject.label, label)
                and subject_visible(subject, event_key, observed_at_epoch)
                and box_iou(subject.box, box) >= min_iou
            ]
            if not matches:
                return False, 0, None
            match = max(
                matches,
                key=lambda subject: (
                    prior_sightings(subject, event_key, observed_at_epoch),
                    box_iou(subject.box, box),
                ),
            )
            return (
                True,
                prior_sightings(match, event_key, observed_at_epoch),
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
                and same_subject_label(subject.label, label)
                and subject_visible(subject, event_key, observed_at_epoch)
                and box_iou(subject.box, box) >= min_iou
            ]
            if overlapping:
                _survivor, extras, _added = merge_subjects(
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
                        sighting_epochs={event_key: observed_at_epoch},
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
                    and same_subject_label(subject.label, label)
                    and box_iou(subject.box, box) >= min_iou
                )
                if matches and invalidate_location:
                    continue
                if matches:
                    revoke_sighting(subject, event_key)
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
                sighting_epochs=dict(row.get("sighting_epochs") or {}),
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
                if same_subject_label(subject.label, label)
                and subject_visible(subject, event_key, observed_at_epoch)
                and box_iou(subject.box, box) >= min_iou
            ]
            if not matches:
                return False, 0, None
            match = max(
                matches,
                key=lambda subject: (
                    prior_sightings(subject, event_key, observed_at_epoch),
                    box_iou(subject.box, box),
                ),
            )
            return (
                True,
                prior_sightings(match, event_key, observed_at_epoch),
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
                if same_subject_label(subject.label, label)
                and subject_visible(subject, event_key, observed_at_epoch)
                and box_iou(subject.box, box) >= min_iou
            ]
            if overlapping:
                survivor, extras, added = merge_subjects(
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
                if added or extras:
                    self._persist(survivor)
            else:
                subject = SceneContextSubject(
                    label, box, observed_at_epoch, observed_at_epoch, [event_key],
                    sighting_epochs={event_key: observed_at_epoch},
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
                    same_subject_label(subject.label, label)
                    and box_iou(subject.box, box) >= min_iou
                )
                if matches and invalidate_location:
                    subject.last_moved_epoch = subject.last_seen_epoch
                    self._delete(subject)
                    continue
                if matches and revoke_sighting(subject, event_key):
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
                "sighting_epochs": dict(subject.sighting_epochs),
                "last_moved_epoch": subject.last_moved_epoch,
            },
        )

    def _delete(self, subject: SceneContextSubject) -> None:
        if subject.row_id is None:
            return
        self.writes += 1
        self._store.delete_scene_context(self.camera_id, subject.row_id)
        subject.row_id = None

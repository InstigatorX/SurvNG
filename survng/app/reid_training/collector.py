"""In-session retention and flush of representative person ReID training crops."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..config import ObjectTrackingConfig
from ..image_storage import DurableImageWriter
from ..incident_utils import portable_media_path
from ..object_track.geometry import _appearance, _box, _confidence
from ..visual_quality import image_quality
from .store import ReidTrainingStore

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _Candidate:
    track_id: int
    captured_at: float
    box: tuple[float, float, float, float]
    confidence: float
    quality: float
    area: float
    crop: np.ndarray
    embedding: np.ndarray | None
    model_kind: str
    model_fingerprint: str


@dataclass
class ReidTrainingBuffer:
    """Per-tracking-session candidate crops awaiting selection/flush."""

    by_track: dict[int, list[_Candidate]] = field(default_factory=dict)

    def clear(self) -> None:
        self.by_track.clear()


class ReidTrainingCollector:
    """Select representative person crops for environment-adaptation training."""

    def __init__(
        self,
        store: ReidTrainingStore,
        image_writer: DurableImageWriter,
        config: ObjectTrackingConfig,
    ) -> None:
        self.store = store
        self.image_writer = image_writer
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.reid_training_collector_enabled)

    def retain_from_frame(
        self,
        buffer: ReidTrainingBuffer,
        frame: np.ndarray,
        tracked: list[dict[str, Any]],
        detections: list[dict[str, Any]],
        captured_at: float,
    ) -> None:
        if not self.enabled or frame is None or not getattr(frame, "size", 0):
            return
        height, width = frame.shape[:2]
        detection_by_box = {
            _box(item.get("box")): item
            for item in detections
            if _box(item.get("box")) is not None
        }
        for item in tracked:
            label = str(item.get("label") or "").strip().lower()
            if label != "person":
                continue
            try:
                track_id = int(item.get("track_id"))
            except (TypeError, ValueError):
                continue
            box = _box(item.get("box"))
            if box is None:
                continue
            confidence = _confidence(item)
            if confidence < self.config.reid_training_min_confidence:
                continue
            x1 = max(0, min(width - 1, int(box[0])))
            y1 = max(0, min(height - 1, int(box[1])))
            x2 = max(x1 + 1, min(width, int(box[2])))
            y2 = max(y1 + 1, min(height, int(box[3])))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            area = float(crop.shape[0] * crop.shape[1])
            if area < float(self.config.reid_training_min_crop_pixels):
                continue
            if crop.shape[0] < 16 or crop.shape[1] < 8:
                continue
            quality = float(image_quality(crop).score)
            if quality < self.config.reid_training_min_quality:
                continue
            source = detection_by_box.get(box) or {}
            embedding = _appearance(source.get("_tracking_embedding"))
            model_kind = ""
            model_fingerprint = ""
            # Keep at most a small overflow pool per track for later selection.
            pool = buffer.by_track.setdefault(track_id, [])
            pool.append(
                _Candidate(
                    track_id=track_id,
                    captured_at=float(captured_at),
                    box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=confidence,
                    quality=quality,
                    area=area,
                    crop=np.ascontiguousarray(crop.copy()),
                    embedding=None if embedding is None else embedding.copy(),
                    model_kind=model_kind,
                    model_fingerprint=model_fingerprint,
                )
            )
            overflow = self.config.reid_training_max_samples_per_track * 4
            if len(pool) > overflow:
                # Drop the weakest mid-track samples first.
                ranked = sorted(
                    enumerate(pool),
                    key=lambda pair: (
                        pair[1].confidence * 0.55
                        + pair[1].quality * 0.25
                        + min(1.0, pair[1].area / 20000.0) * 0.20,
                        -abs(pair[0] - len(pool) / 2.0),
                    ),
                )
                drop_index = ranked[0][0]
                del pool[drop_index]

    def flush(
        self,
        buffer: ReidTrainingBuffer,
        *,
        event_id: int,
        camera_id: str,
        tracks: list[dict[str, Any]],
        model_identity_for_label: Any | None = None,
    ) -> int:
        if not self.enabled:
            buffer.clear()
            return 0
        confirmed_ids = {
            int(track["track_id"])
            for track in tracks
            if track.get("track_id") is not None
            and str(track.get("state") or "") in {"confirmed", "lost"}
            and str(track.get("label") or "").strip().lower() == "person"
        }
        selected: list[tuple[_Candidate, str]] = []
        for track_id, candidates in buffer.by_track.items():
            if track_id not in confirmed_ids or not candidates:
                continue
            selected.extend(
                (candidate, reason)
                for candidate, reason in self._select_for_track(candidates)
            )
        if not selected:
            buffer.clear()
            return 0
        # Global event budget: keep highest utility samples.
        if len(selected) > self.config.reid_training_max_samples_per_event:
            selected = sorted(
                selected,
                key=lambda item: (
                    item[0].confidence * 0.5
                    + item[0].quality * 0.3
                    + min(1.0, item[0].area / 20000.0) * 0.2
                ),
                reverse=True,
            )[: self.config.reid_training_max_samples_per_event]

        model_kind = ""
        model_fingerprint = ""
        if callable(model_identity_for_label):
            identity = model_identity_for_label("person")
            if isinstance(identity, dict):
                model_kind = str(identity.get("model_kind") or "")
                model_fingerprint = str(identity.get("model_fingerprint") or "")

        stored = 0
        identity_by_track: dict[int, int] = {}
        event_dir = self.store.crops_root / str(camera_id) / str(int(event_id))
        for candidate, reason in selected:
            person_id = identity_by_track.get(candidate.track_id)
            if person_id is None:
                person_id = self.store.create_identity()
                identity_by_track[candidate.track_id] = person_id
            stamp = datetime.fromtimestamp(
                candidate.captured_at,
                timezone.utc,
            ).strftime("%Y%m%dT%H%M%S%f")
            stem = f"track{candidate.track_id}_{stamp}_{reason}"
            path = self.image_writer.write(event_dir, stem, candidate.crop)
            if path is None:
                LOGGER.warning(
                    "failed to write ReID training crop for event %s track %s",
                    event_id,
                    candidate.track_id,
                )
                continue
            sample_id = (
                f"e{int(event_id)}-t{candidate.track_id}-"
                f"{stamp}-{reason}"
            )
            inserted = self.store.insert_sample({
                "sample_id": sample_id,
                "event_id": int(event_id),
                "camera_id": str(camera_id),
                "track_id": int(candidate.track_id),
                "captured_at": datetime.fromtimestamp(
                    candidate.captured_at,
                    timezone.utc,
                ).isoformat(),
                "bounding_box": {
                    "x1": round(candidate.box[0], 1),
                    "y1": round(candidate.box[1], 1),
                    "x2": round(candidate.box[2], 1),
                    "y2": round(candidate.box[3], 1),
                },
                "detection_confidence": round(candidate.confidence, 4),
                "crop_path": portable_media_path(self.store.storage_dir, path),
                "embedding": candidate.embedding,
                "model_kind": model_kind or candidate.model_kind,
                "model_fingerprint": model_fingerprint or candidate.model_fingerprint,
                "assigned_person_id": person_id,
                "assignment_source": "track",
                "assignment_confidence": 1.0,
                "review_status": "auto",
                "selection_reason": reason,
                "quality_score": round(candidate.quality, 4),
            })
            if inserted is not None:
                stored += 1
        buffer.clear()
        return stored

    def _select_for_track(
        self,
        candidates: list[_Candidate],
    ) -> list[tuple[_Candidate, str]]:
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda item: item.captured_at)
        max_count = self.config.reid_training_max_samples_per_track
        min_count = min(self.config.reid_training_min_samples_per_track, max_count)
        picks: dict[int, tuple[_Candidate, str]] = {}

        def take(index: int, reason: str) -> None:
            if index < 0 or index >= len(ordered) or len(picks) >= max_count:
                return
            if index in picks:
                return
            picks[index] = (ordered[index], reason)

        take(0, "start")
        take(len(ordered) - 1, "end")
        take(len(ordered) // 2, "middle")
        take(max(range(len(ordered)), key=lambda i: ordered[i].area), "largest")
        take(
            max(range(len(ordered)), key=lambda i: ordered[i].confidence),
            "highest_confidence",
        )

        # Embedding-diverse: greedily add the crop farthest from selected set.
        embedded = [
            (index, candidate)
            for index, candidate in enumerate(ordered)
            if candidate.embedding is not None and index not in picks
        ]
        while embedded and len(picks) < max_count:
            selected_vectors = [
                ordered[index].embedding
                for index in picks
                if ordered[index].embedding is not None
            ]
            if not selected_vectors:
                index, _candidate = embedded.pop(0)
                take(index, "embedding_diverse")
                embedded = [(i, c) for i, c in embedded if i not in picks]
                continue

            def min_similarity(item: tuple[int, _Candidate]) -> float:
                vector = item[1].embedding
                assert vector is not None
                return float(
                    min(np.dot(vector, other) for other in selected_vectors)
                )

            index, _candidate = min(embedded, key=min_similarity)
            take(index, "embedding_diverse")
            embedded = [(i, c) for i, c in embedded if i not in picks]

        # Fill remaining slots with highest quality unused samples.
        remaining = [
            (index, candidate)
            for index, candidate in enumerate(ordered)
            if index not in picks
        ]
        remaining.sort(
            key=lambda item: item[1].confidence * 0.6 + item[1].quality * 0.4,
            reverse=True,
        )
        for index, _candidate in remaining:
            if len(picks) >= max_count:
                break
            take(index, "quality")

        chosen = list(picks.values())
        if len(chosen) < min_count:
            # Prefer sparse useful tracks over empty ones when the track was short.
            return chosen
        return chosen

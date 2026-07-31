from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from types import SimpleNamespace
from typing import Any

import numpy as np

from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
from ultralytics.trackers.utils import matching
from ultralytics.trackers.utils.stracks import parse_bboxes

from .config import ObjectTrackingConfig
from .object_tracking import Box, ObjectTrack, _appearance, _box, _confidence


@dataclass(slots=True)
class _TrackerDetections:
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    @property
    def xywh(self) -> np.ndarray:
        if not len(self):
            return np.empty((0, 4), dtype=np.float32)
        result = self.xyxy.copy()
        result[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        result[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
        result[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        result[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return result

    def __getitem__(self, index: Any) -> "_TrackerDetections":
        return _TrackerDetections(
            xyxy=np.asarray(self.xyxy[index], dtype=np.float32).reshape(-1, 4),
            conf=np.asarray(self.conf[index], dtype=np.float32).reshape(-1),
            cls=np.asarray(self.cls[index], dtype=np.float32).reshape(-1),
        )


class _ClassAwareBOTSORT(BOTSORT):
    """Prevent Ultralytics' class-agnostic association from crossing labels."""

    def __init__(self, args: Any) -> None:
        super().__init__(args)

        # Ultralytics' BaseTrack counter is process-global and every tracker
        # constructor resets it. SurvNG can run multiple camera sessions at
        # once, so each instance needs its own track class and counter.
        class SessionBOTrack(BOTrack):
            _session_count = 0

            @classmethod
            def next_id(cls) -> int:
                cls._session_count += 1
                return cls._session_count

            @classmethod
            def reset_id(cls) -> None:
                cls._session_count = 0

        self._session_track_type = SessionBOTrack

    def init_track(self, results: Any, img: np.ndarray | None = None) -> list[BOTrack]:
        if len(results) == 0:
            return []
        boxes = parse_bboxes(results)
        if self.args.with_reid and self.encoder is not None and img is not None:
            features = self.encoder(img, boxes)
            return [
                self._session_track_type(xywh, score, cls, feature)
                for xywh, score, cls, feature in zip(
                    boxes,
                    results.conf,
                    results.cls,
                    features,
                    strict=True,
                )
            ]
        return [
            self._session_track_type(xywh, score, cls)
            for xywh, score, cls in zip(
                boxes,
                results.conf,
                results.cls,
                strict=True,
            )
        ]

    @staticmethod
    def _block_cross_class(
        distances: np.ndarray,
        tracks: list[Any],
        detections: list[Any],
    ) -> np.ndarray:
        if distances.size:
            track_classes = np.asarray([int(track.cls) for track in tracks])[:, None]
            detection_classes = np.asarray(
                [int(detection.cls) for detection in detections]
            )[None, :]
            distances[track_classes != detection_classes] = 1.0
        return distances

    def get_dists(self, tracks: list[Any], detections: list[Any]) -> np.ndarray:
        return self._block_cross_class(
            super().get_dists(tracks, detections),
            tracks,
            detections,
        )

    def _second_association(
        self,
        strack_pool: list[Any],
        u_track: list[int],
        detections_second: list[Any],
        activated: list[Any],
        refind: list[Any],
        lost: list[Any],
    ) -> None:
        remaining = [
            strack_pool[index]
            for index in u_track
            if strack_pool[index].state == TrackState.Tracked
        ]
        if remaining and detections_second:
            distances = self._block_cross_class(
                matching.iou_distance(remaining, detections_second),
                remaining,
                detections_second,
            )
            matches, unmatched, _ = matching.linear_assignment(distances, thresh=0.5)
            self._apply_matches(matches, remaining, detections_second, activated, refind)
        else:
            unmatched = list(range(len(remaining)))
        for index in unmatched:
            track = remaining[index]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)


class UltralyticsBotSortObjectTracker:
    """Adapter around Ultralytics BoT-SORT using SurvNG detections and ReID features."""

    _CLASS_COORDINATE_STRIDE = 100_000.0

    def __init__(
        self,
        config: ObjectTrackingConfig,
        high_confidence_threshold: float,
    ) -> None:
        self.config = config
        self.high_confidence_threshold = max(
            config.low_confidence_threshold,
            float(high_confidence_threshold),
        )
        track_buffer = ceil(
            config.sample_fps
            * (
                max(config.lost_timeout_seconds, config.reid_max_age_seconds)
                if config.appearance_reid_enabled
                else config.lost_timeout_seconds
            )
        )
        appearance_thresholds = []
        if config.reid_enabled:
            appearance_thresholds.append(config.reid_match_threshold)
        if config.vehicle_reid_enabled:
            appearance_thresholds.append(config.vehicle_reid_match_threshold)
        appearance_threshold = min(appearance_thresholds, default=1.0)
        self._tracker = _ClassAwareBOTSORT(SimpleNamespace(
            track_high_thresh=self.high_confidence_threshold,
            track_low_thresh=config.low_confidence_threshold,
            new_track_thresh=self.high_confidence_threshold,
            track_buffer=max(1, track_buffer),
            match_thresh=config.botsort_match_threshold,
            fuse_score=config.botsort_fuse_score,
            gmc_method="none",
            proximity_thresh=config.botsort_proximity_threshold,
            # Ultralytics thresholds its cosine distance after dividing by two,
            # so convert SurvNG's direct cosine-similarity threshold to the
            # equivalent [0, 1] similarity scale used by BoT-SORT.
            appearance_thresh=(appearance_threshold + 1.0) / 2.0,
            with_reid=config.appearance_reid_enabled,
            model="auto",
            device="cpu",
        ))
        self._records: dict[int, ObjectTrack] = {}
        self._class_ids: dict[str, int] = {}

    def update(
        self,
        detections: list[dict[str, Any]],
        captured_at: float,
        *,
        confirm_new: bool = False,
    ) -> list[dict[str, Any]]:
        self._prune_expired_reid_tracks(captured_at)
        usable = [
            (detection, parsed)
            for detection in detections
            if self.config.tracks_label(detection.get("label"))
            and detection.get("incident_eligible") is not False
            and (parsed := _box(detection.get("box"))) is not None
        ]
        usable = sorted(usable, key=lambda item: _confidence(item[0]), reverse=True)[
            : self.config.max_tracks_per_session
        ]
        tracker_input = self._results(usable, confirm_new=confirm_new)
        features = self._features(usable) if self.config.appearance_reid_enabled else None
        output = self._tracker.update(tracker_input, img=None, feats=features)
        tracked: list[dict[str, Any]] = []
        for row in np.asarray(output, dtype=np.float32).reshape(-1, 8):
            track_id = int(row[4])
            detection_index = int(row[7])
            if detection_index < 0 or detection_index >= len(usable):
                continue
            detection, box = usable[detection_index]
            record = self._records.get(track_id)
            if record is not None and record.label != str(detection["label"]):
                raise RuntimeError(
                    f"BoT-SORT associated track {track_id} across object classes"
                )
            if record is None:
                if len(self._records) >= self.config.max_tracks_per_session:
                    continue
                first_seen = self._first_seen(detection, captured_at)
                record = ObjectTrack(
                    track_id=track_id,
                    label=str(detection["label"]),
                    box=box,
                    first_seen=first_seen,
                    last_seen=captured_at,
                    confidence=_confidence(detection),
                    max_confidence=_confidence(detection),
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
                    appearance=_appearance(detection.get("_tracking_embedding")),
                )
                native = next(
                    (
                        item
                        for item in self._tracker.tracked_stracks
                        if int(item.track_id) == track_id
                    ),
                    None,
                )
                if native is not None:
                    record.hits = max(1, int(native.tracklet_len) + 1)
                self._records[track_id] = record
            else:
                record.observe(detection, captured_at, box)
            record.confirmed = record.confirmed or record.hits >= self.config.min_confirmations
            if not record.confirmed:
                continue
            tracked.append({
                **{
                    key: value
                    for key, value in detection.items()
                    if not key.startswith("_tracking_")
                },
                "track_id": track_id,
                "track_state": "confirmed",
                "track_observations": record.hits,
            })
        return tracked

    def _prune_expired_reid_tracks(self, captured_at: float) -> None:
        if not self.config.appearance_reid_enabled:
            return
        retained: list[Any] = []
        for native in self._tracker.lost_stracks:
            record = self._records.get(int(native.track_id))
            if (
                record is not None
                and captured_at - record.last_seen > self.config.reid_max_age_seconds
            ):
                native.mark_removed()
                self._tracker.removed_stracks.append(native)
            else:
                retained.append(native)
        self._tracker.lost_stracks = retained

    def has_live_tracks(self, captured_at: float) -> bool:
        return any(
            record.confirmed
            and captured_at - record.last_seen <= self.config.lost_timeout_seconds
            for record in self._records.values()
        )

    def summaries(self, captured_at: float) -> list[dict[str, Any]]:
        return [
            record.summary(
                active=captured_at - record.last_seen <= self.config.lost_timeout_seconds,
            )
            for record in sorted(self._records.values(), key=lambda item: item.track_id)
            if record.confirmed
        ]

    def diagnostics(self) -> dict[str, Any]:
        # BoT-SORT owns its association internals. Keep the shared persistence
        # contract explicit without claiming SurvNG Hybrid diagnostics.
        return {}

    def _results(
        self,
        usable: list[tuple[dict[str, Any], Box]],
        *,
        confirm_new: bool,
    ) -> _TrackerDetections:
        boxes: list[Box] = []
        confidences: list[float] = []
        classes: list[int] = []
        for detection, box in usable:
            label = str(detection["label"])
            class_id = self._class_ids.setdefault(label, len(self._class_ids))
            confidence = _confidence(detection)
            if confirm_new:
                confidence = max(confidence, self.high_confidence_threshold)
            # Ultralytics also de-duplicates its tracked/lost pools by geometry
            # after association without checking class. Separate class coordinate
            # planes make every internal geometry operation class-safe; persisted
            # and returned boxes always come from the original detection.
            offset = class_id * self._CLASS_COORDINATE_STRIDE
            boxes.append((box[0] + offset, box[1], box[2] + offset, box[3]))
            confidences.append(confidence)
            classes.append(class_id)
        return _TrackerDetections(
            xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            conf=np.asarray(confidences, dtype=np.float32),
            cls=np.asarray(classes, dtype=np.float32),
        )

    @staticmethod
    def _features(
        usable: list[tuple[dict[str, Any], Box]],
    ) -> np.ndarray | None:
        vectors = [
            _appearance(detection.get("_tracking_embedding"))
            for detection, _box_value in usable
        ]
        dimension = max((vector.size for vector in vectors if vector is not None), default=0)
        if dimension <= 0:
            return None
        return np.stack([
            np.pad(vector, (0, dimension - vector.size))
            if vector is not None and vector.size <= dimension
            else np.zeros(dimension, dtype=np.float32)
            for vector in vectors
        ])

    @staticmethod
    def _first_seen(detection: dict[str, Any], captured_at: float) -> float:
        try:
            value = float(detection.get("_tracking_first_seen_at"))
        except (TypeError, ValueError):
            return captured_at
        return min(captured_at, value) if np.isfinite(value) and value >= 0.0 else captured_at

"""Detection-only face crops. Names and gallery rows never enter this store."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
import numpy as np

from .face_detection_geometry import (
    box_tuple,
    scale_box_to_frame,
    upper_body_window,
    window_dict,
)


SPLIT_SEED = "survng-face-detection-v1"
ANNOTATION_KINDS = frozenset({"face", "no_face", "not_a_face"})
BOX_KINDS = frozenset({"face", "not_a_face"})
EXPORT_FORBIDDEN_KEYS = frozenset({
    "person",
    "person_id",
    "name",
    "embedding",
    "identity",
    "identity_id",
})


class FaceDetectionDatasetError(ValueError):
    """A detection-label request cannot be stored or exported."""


def calendar_day(captured_epoch: float, timezone_name: str) -> str:
    try:
        zone = ZoneInfo(str(timezone_name or "UTC"))
    except ZoneInfoNotFoundError as exc:
        raise FaceDetectionDatasetError(f"Unknown timezone: {timezone_name}") from exc
    try:
        return datetime.fromtimestamp(float(captured_epoch), zone).date().isoformat()
    except (OverflowError, OSError, ValueError) as exc:
        raise FaceDetectionDatasetError("Observation time is outside the supported range") from exc


def pixel_sha256(image: np.ndarray) -> str:
    pixels = np.ascontiguousarray(image)
    payload = str(pixels.shape).encode("utf-8") + pixels.tobytes()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def propose_day_splits(days: list[str]) -> dict[str, str]:
    """Assign whole days. One or two days cannot fill train, calibration, and held-out."""
    unique = sorted({str(day) for day in days if str(day)})
    if len(unique) < 3:
        raise FaceDetectionDatasetError(
            "Label crops on at least three calendar days before freezing splits"
        )
    ordered = sorted(unique, key=lambda day: hashlib.sha256(f"{SPLIT_SEED}:{day}".encode()).hexdigest())
    held_count = max(1, round(len(ordered) * 0.15))
    calibration_count = max(1, round(len(ordered) * 0.15))
    if held_count + calibration_count >= len(ordered):
        held_count = 1
        calibration_count = 1
    train_count = len(ordered) - held_count - calibration_count
    assignment: dict[str, str] = {}
    for day in ordered[:train_count]:
        assignment[day] = "train"
    for day in ordered[train_count:train_count + calibration_count]:
        assignment[day] = "calibration"
    for day in ordered[train_count + calibration_count:]:
        assignment[day] = "held_out"
    return assignment


def hash_leakage(samples: list[dict[str, Any]]) -> list[str]:
    """Return pixel hashes whose copies would sit in more than one split."""
    splits_for_hash: dict[str, set[str]] = {}
    for sample in samples:
        digest = str(sample.get("pixel_sha256") or "")
        split = str(sample.get("split") or "")
        if not digest or not split or sample.get("disposition") == "drop":
            continue
        splits_for_hash.setdefault(digest, set()).add(split)
    return sorted(digest for digest, splits in splits_for_hash.items() if len(splits) > 1)


def assert_no_hash_leakage(samples: list[dict[str, Any]]) -> None:
    leaked = hash_leakage(samples)
    if leaked:
        raise FaceDetectionDatasetError(
            "Identical crop pixels would cross splits: " + ", ".join(leaked)
        )


def load_person_observations(database_path: Path, *, limit: int = 400) -> list[dict[str, Any]]:
    """Read person scene observations that have a recording and an exact frame time."""
    path = Path(database_path)
    if not path.is_file():
        return []
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        try:
            rows = connection.execute(
                """
                select id, event_id, camera_id, captured_epoch, recording_path, payload_json
                from scene_observations
                where recording_path != ''
                  and json_extract(payload_json, '$.label') = 'person'
                order by captured_epoch desc
                limit ?
                """,
                (max(1, min(int(limit), 2000)),),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    finally:
        connection.close()
    observations = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        exact = payload.get("frame_timestamp_exact")
        if exact is not True and exact != 1:
            continue
        box = payload.get("box")
        if box_tuple(box if isinstance(box, dict) else None) is None:
            continue
        observations.append(
            {
                "source_observation_id": str(row["id"]),
                "event_id": int(row["event_id"]),
                "camera_id": str(row["camera_id"]),
                "captured_epoch": float(row["captured_epoch"]),
                "recording_path": str(row["recording_path"]),
                "person_box": {
                    "x1": float(box["x1"]),
                    "y1": float(box["y1"]),
                    "x2": float(box["x2"]),
                    "y2": float(box["y2"]),
                },
                "detection_frame_width": int(payload.get("detection_frame_width") or 0),
                "detection_frame_height": int(payload.get("detection_frame_height") or 0),
                "frame_timestamp_exact": True,
            }
        )
    return observations


def _walk_forbidden(value: Any, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in EXPORT_FORBIDDEN_KEYS:
                found.add(key)
            _walk_forbidden(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk_forbidden(item, found)


def assert_export_has_no_identity(payload: dict[str, Any]) -> None:
    found: set[str] = set()
    _walk_forbidden(payload, found)
    if found:
        raise FaceDetectionDatasetError(
            "Detection export contains identity fields: " + ", ".join(sorted(found))
        )


class FaceDetectionDataset:
    def __init__(self, database_path: Path, crop_dir: Path) -> None:
        self.database_path = Path(database_path)
        self.crop_dir = Path(crop_dir)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(
                """
                create table if not exists face_detection_samples (
                    id integer primary key,
                    source_observation_id text not null unique,
                    event_id integer,
                    camera_id text not null,
                    captured_epoch real not null,
                    calendar_day text not null,
                    source_frame_width integer not null,
                    source_frame_height integer not null,
                    person_box_json text not null,
                    upper_body_crop_box_json text not null,
                    crop_path text not null,
                    crop_sha256 text not null,
                    pixel_sha256 text not null,
                    crop_width integer not null,
                    crop_height integer not null,
                    baseline_model_sha256 text not null default '',
                    baseline_predictions_json text not null default '[]',
                    baseline_miss integer not null default 1,
                    baseline_ran integer not null default 0,
                    split text not null default '',
                    disposition text not null default 'unreviewed',
                    drop_reason text not null default '',
                    reviewed_at text not null default ''
                );
                create table if not exists face_detection_annotations (
                    id integer primary key,
                    sample_id integer not null references face_detection_samples(id) on delete cascade,
                    kind text not null,
                    bbox_json text not null default '',
                    created_at text not null
                );
                create index if not exists face_detection_sample_day
                    on face_detection_samples(calendar_day, disposition);
                create index if not exists face_detection_annotation_sample
                    on face_detection_annotations(sample_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma foreign_keys = on")
        return connection

    def materialize(
        self,
        observations: list[dict[str, Any]],
        *,
        frame_reader: Callable[[str, float], np.ndarray | None],
        detector: Callable[[np.ndarray], list[dict[str, Any]]] | None = None,
        timezone_name: str = "UTC",
        limit: int = 50,
        baseline_model_sha256: str = "",
        scan_multiplier: int = 4,
    ) -> dict[str, int]:
        limit = max(1, min(int(limit), 200))
        scan_cap = max(limit, min(limit * max(1, scan_multiplier), 800))
        decoded: list[dict[str, Any]] = []
        stats = {
            "stored": 0,
            "already": 0,
            "undecoded": 0,
            "tiny": 0,
            "skipped": 0,
            "scanned": 0,
        }
        with self._lock:
            known = self._known_sources()
            for observation in observations:
                if len(decoded) >= scan_cap:
                    break
                stats["scanned"] += 1
                source_id = str(observation.get("source_observation_id") or "")
                if not source_id:
                    stats["skipped"] += 1
                    continue
                if source_id in known:
                    stats["already"] += 1
                    continue
                prepared = self._prepare_crop(
                    observation,
                    frame_reader=frame_reader,
                    detector=detector,
                    timezone_name=timezone_name,
                    baseline_model_sha256=baseline_model_sha256,
                )
                if prepared is None:
                    reason = observation.get("_skip_reason")
                    if reason in stats:
                        stats[reason] += 1
                    else:
                        stats["skipped"] += 1
                    continue
                decoded.append(prepared)
            selected = _select_materialized(decoded, limit)
            written = []
            for item in selected:
                if self._write_crop(item):
                    written.append(item)
            with self._connect() as connection:
                for item in written:
                    connection.execute(
                        """
                        insert into face_detection_samples (
                            source_observation_id, event_id, camera_id, captured_epoch, calendar_day,
                            source_frame_width, source_frame_height, person_box_json,
                            upper_body_crop_box_json, crop_path, crop_sha256, pixel_sha256,
                            crop_width, crop_height, baseline_model_sha256, baseline_predictions_json,
                            baseline_miss, baseline_ran
                        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item["source_observation_id"],
                            item["event_id"],
                            item["camera_id"],
                            item["captured_epoch"],
                            item["calendar_day"],
                            item["source_frame_width"],
                            item["source_frame_height"],
                            json.dumps(item["person_box"]),
                            json.dumps(item["upper_body_crop_box"]),
                            item["crop_path"],
                            item["crop_sha256"],
                            item["pixel_sha256"],
                            item["crop_width"],
                            item["crop_height"],
                            item["baseline_model_sha256"],
                            json.dumps(item["baseline_predictions"]),
                            int(item["baseline_miss"]),
                            int(item["baseline_ran"]),
                        ),
                    )
                connection.commit()
        stats["stored"] = len(written)
        return stats

    def _known_sources(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "select source_observation_id from face_detection_samples"
            ).fetchall()
        return {str(row["source_observation_id"]) for row in rows}

    def _prepare_crop(
        self,
        observation: dict[str, Any],
        *,
        frame_reader: Callable[[str, float], np.ndarray | None],
        detector: Callable[[np.ndarray], list[dict[str, Any]]] | None,
        timezone_name: str,
        baseline_model_sha256: str,
    ) -> dict[str, Any] | None:
        if observation.get("frame_timestamp_exact") is False:
            observation["_skip_reason"] = "skipped"
            return None
        person = box_tuple(observation.get("person_box"))
        recording_path = str(observation.get("recording_path") or "")
        seek = observation.get("seek_seconds")
        if person is None or not recording_path or seek is None:
            observation["_skip_reason"] = "skipped"
            return None
        try:
            seek_seconds = float(seek)
        except (TypeError, ValueError):
            observation["_skip_reason"] = "skipped"
            return None
        frame = frame_reader(recording_path, seek_seconds)
        if frame is None or getattr(frame, "size", 0) == 0:
            observation["_skip_reason"] = "undecoded"
            return None
        frame_height, frame_width = frame.shape[:2]
        source_width = int(observation.get("detection_frame_width") or 0)
        source_height = int(observation.get("detection_frame_height") or 0)
        scaled = scale_box_to_frame(
            person,
            source_width or frame_width,
            source_height or frame_height,
            frame_width,
            frame_height,
        )
        window = upper_body_window(scaled, frame_width, frame_height)
        if window is None:
            observation["_skip_reason"] = "tiny"
            return None
        left, top, right, bottom = window
        crop = np.ascontiguousarray(frame[top:bottom, left:right])
        if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
            observation["_skip_reason"] = "tiny"
            return None
        source_id = str(observation["source_observation_id"])
        predictions: list[dict[str, Any]] = []
        ran = detector is not None
        if detector is not None:
            predictions = _normalize_predictions(detector(crop))
        day = calendar_day(float(observation["captured_epoch"]), timezone_name)
        return {
            "source_observation_id": source_id,
            "event_id": int(observation.get("event_id") or 0),
            "camera_id": str(observation.get("camera_id") or ""),
            "captured_epoch": float(observation["captured_epoch"]),
            "calendar_day": day,
            "source_frame_width": int(frame_width),
            "source_frame_height": int(frame_height),
            "person_box": {
                "x1": person[0], "y1": person[1], "x2": person[2], "y2": person[3],
            },
            "upper_body_crop_box": window_dict(window),
            "image": crop,
            "pixel_sha256": pixel_sha256(crop),
            "crop_width": int(crop.shape[1]),
            "crop_height": int(crop.shape[0]),
            "baseline_model_sha256": baseline_model_sha256 if ran else "",
            "baseline_predictions": predictions,
            "baseline_miss": not predictions,
            "baseline_ran": ran,
            "source_stem": source_id,
        }

    def _write_crop(self, item: dict[str, Any]) -> bool:
        crop = item.pop("image", None)
        source_id = str(item.pop("source_stem", item["source_observation_id"]))
        if crop is None:
            return False
        crop_path = self.crop_dir / f"{_safe_stem(source_id)}.png"
        if not cv2.imwrite(str(crop_path), crop):
            return False
        item["crop_path"] = str(crop_path)
        item["crop_sha256"] = file_sha256(crop_path)
        return True

    def queue(self, *, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        with self._lock, self._connect() as connection:
            misses = connection.execute(
                """
                select * from face_detection_samples
                where disposition = 'unreviewed' and (baseline_miss = 1 or baseline_ran = 0)
                order by captured_epoch desc
                limit ?
                """,
                (limit,),
            ).fetchall()
            hit_slots = max(1, round(limit * 0.25)) if misses else limit
            if len(misses) > limit - hit_slots:
                misses = misses[: limit - hit_slots]
            hits = connection.execute(
                """
                select * from face_detection_samples
                where disposition = 'unreviewed' and baseline_ran = 1 and baseline_miss = 0
                order by captured_epoch desc
                limit ?
                """,
                (hit_slots,),
            ).fetchall()
            rows = list(misses) + list(hits)
            return [self._public(connection, row) for row in rows[:limit]]

    def get(self, sample_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "select * from face_detection_samples where id = ?",
                (int(sample_id),),
            ).fetchone()
            if row is None:
                return None
            return self._public(connection, row)

    def crop_file(self, sample_id: int) -> Path | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "select crop_path from face_detection_samples where id = ?",
                (int(sample_id),),
            ).fetchone()
        if row is None:
            return None
        path = Path(str(row["crop_path"]))
        if not path.is_file():
            return None
        return path

    def label(
        self,
        sample_id: int,
        annotations: list[dict[str, Any]],
        *,
        drop_reason: str = "",
    ) -> dict[str, Any]:
        sample = self.get(sample_id)
        if sample is None:
            raise FaceDetectionDatasetError("Detection sample was not found")
        stored = _validate_annotations(
            annotations,
            drop_reason=drop_reason,
            crop_width=int(sample["crop_width"]),
            crop_height=int(sample["crop_height"]),
        )
        disposition = "drop" if stored["drop"] else "reviewed"
        reviewed_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                update face_detection_samples
                set disposition = ?, drop_reason = ?, reviewed_at = ?
                where id = ?
                """,
                (disposition, stored["drop_reason"], reviewed_at, int(sample_id)),
            )
            connection.execute(
                "delete from face_detection_annotations where sample_id = ?",
                (int(sample_id),),
            )
            for annotation in stored["annotations"]:
                connection.execute(
                    """
                    insert into face_detection_annotations (sample_id, kind, bbox_json, created_at)
                    values (?, ?, ?, ?)
                    """,
                    (
                        int(sample_id),
                        annotation["kind"],
                        json.dumps(annotation["box"]) if annotation.get("box") else "",
                        reviewed_at,
                    ),
                )
            connection.commit()
            row = connection.execute(
                "select * from face_detection_samples where id = ?",
                (int(sample_id),),
            ).fetchone()
            return self._public(connection, row)

    def assign_splits(self) -> dict[str, str]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                select calendar_day, split, pixel_sha256, disposition
                from face_detection_samples
                where disposition = 'reviewed'
                """
            ).fetchall()
            frozen = {
                str(row["calendar_day"]): str(row["split"])
                for row in rows
                if str(row["split"] or "")
            }
            pending = sorted({
                str(row["calendar_day"])
                for row in rows
                if str(row["calendar_day"]) not in frozen
            })
            if not pending and not frozen:
                raise FaceDetectionDatasetError("Review face crops before freezing splits")
            proposed = dict(frozen)
            if pending:
                proposed.update(propose_day_splits_for_new_days(pending, frozen))
            preview = [
                {
                    "pixel_sha256": str(row["pixel_sha256"]),
                    "split": proposed.get(str(row["calendar_day"]), ""),
                    "disposition": str(row["disposition"]),
                }
                for row in rows
            ]
            assert_no_hash_leakage(preview)
            for day, split in frozen.items():
                connection.execute(
                    """
                    update face_detection_samples
                    set split = ?
                    where calendar_day = ? and disposition = 'reviewed' and split = ''
                    """,
                    (split, day),
                )
            for day, split in proposed.items():
                if day in frozen:
                    continue
                connection.execute(
                    """
                    update face_detection_samples
                    set split = ?
                    where calendar_day = ? and disposition = 'reviewed' and split = ''
                    """,
                    (split, day),
                )
            connection.commit()
        return proposed

    def reviewed_samples(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                select * from face_detection_samples
                where disposition = 'reviewed' and split != ''
                order by calendar_day, id
                """
            ).fetchall()
            return [self._public(connection, row) for row in rows]

    def export(self, destination: Path) -> dict[str, Any]:
        destination = Path(destination)
        with self._lock, self._connect() as connection:
            pending = connection.execute(
                """
                select count(*) from face_detection_samples
                where disposition = 'reviewed' and split = ''
                """
            ).fetchone()[0]
        if pending:
            raise FaceDetectionDatasetError(
                "Freeze splits on every reviewed crop before exporting"
            )
        samples = self.reviewed_samples()
        if not samples:
            raise FaceDetectionDatasetError("Freeze splits on reviewed crops before exporting")
        missing = [sample["id"] for sample in samples if not sample.get("split")]
        if missing:
            raise FaceDetectionDatasetError("Every reviewed crop needs a frozen split before export")
        assert_no_hash_leakage(samples)
        if destination.exists():
            raise FaceDetectionDatasetError("Export directory already exists")
        crop_dir = destination / "crops"
        crop_dir.mkdir(parents=True)
        manifest_samples = []
        for sample in samples:
            source = Path(str(self.crop_file(int(sample["id"])) or ""))
            if not source.is_file():
                raise FaceDetectionDatasetError(f"Crop for sample {sample['id']} is missing")
            name = f"{int(sample['id'])}.png"
            target = crop_dir / name
            shutil.copyfile(source, target)
            digest = file_sha256(target)
            if digest != sample["crop_sha256"]:
                raise FaceDetectionDatasetError(
                    f"Crop bytes for sample {sample['id']} no longer match the stored hash"
                )
            manifest_samples.append(
                {
                    "id": str(sample["id"]),
                    "path": f"crops/{name}",
                    "sha256": digest,
                    "pixel_sha256": sample["pixel_sha256"],
                    "split": sample["split"],
                    "camera_id": sample["camera_id"],
                    "day": sample["calendar_day"],
                    "width": sample["crop_width"],
                    "height": sample["crop_height"],
                    "annotations": [
                        {
                            "kind": annotation["kind"],
                            **(
                                {"bbox_xyxy": [
                                    annotation["box"]["x1"],
                                    annotation["box"]["y1"],
                                    annotation["box"]["x2"],
                                    annotation["box"]["y2"],
                                ]}
                                if annotation.get("box")
                                else {}
                            ),
                        }
                        for annotation in sample["annotations"]
                    ],
                    "baseline_predictions": sample["baseline_predictions"],
                }
            )
        manifest = {
            "version": 1,
            "task": "face_detection",
            "contract": {
                "input": "BGR NCHW float32 0-255",
                "resize": "stretch cv2.INTER_AREA to the static IR input",
                "letterbox": False,
                "output": "[1,1,N,7] SSD rows; confidence at index 2; normalized xyxy at 3:6",
                "baked_preprocess": "RGB conversion, normalization, decode, and NMS stay inside the IR",
            },
            "samples": manifest_samples,
        }
        assert_export_has_no_identity(manifest)
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    def _public(self, connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        annotations = []
        for item in connection.execute(
            """
            select kind, bbox_json from face_detection_annotations
            where sample_id = ? order by id
            """,
            (int(row["id"]),),
        ):
            box = json.loads(item["bbox_json"]) if item["bbox_json"] else None
            annotations.append({"kind": item["kind"], "box": box})
        return {
            "id": int(row["id"]),
            "source_observation_id": str(row["source_observation_id"]),
            "event_id": int(row["event_id"] or 0),
            "camera_id": str(row["camera_id"]),
            "captured_epoch": float(row["captured_epoch"]),
            "calendar_day": str(row["calendar_day"]),
            "crop_width": int(row["crop_width"]),
            "crop_height": int(row["crop_height"]),
            "crop_sha256": str(row["crop_sha256"]),
            "pixel_sha256": str(row["pixel_sha256"]),
            "baseline_miss": bool(row["baseline_miss"]),
            "baseline_ran": bool(row["baseline_ran"]),
            "baseline_predictions": json.loads(row["baseline_predictions_json"] or "[]"),
            "baseline_model_sha256": str(row["baseline_model_sha256"] or ""),
            "split": str(row["split"] or ""),
            "disposition": str(row["disposition"]),
            "drop_reason": str(row["drop_reason"] or ""),
            "reviewed_at": str(row["reviewed_at"] or ""),
            "annotations": annotations,
        }


def propose_day_splits_for_new_days(
    pending: list[str],
    frozen: dict[str, str],
) -> dict[str, str]:
    """Assign only new days. Days that already have a split stay where they are."""
    if not pending:
        return {}
    if not frozen:
        return propose_day_splits(pending)
    if len(set(frozen) | set(pending)) < 3:
        raise FaceDetectionDatasetError(
            "Label crops on at least three calendar days before freezing splits"
        )
    combined = propose_day_splits([*frozen.keys(), *pending])
    return {day: combined[day] for day in pending}


def _select_materialized(decoded: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    misses = [item for item in decoded if item["baseline_miss"] or not item["baseline_ran"]]
    hits = [item for item in decoded if item["baseline_ran"] and not item["baseline_miss"]]
    if not misses:
        return hits[:limit]
    if not hits:
        return misses[:limit]
    hit_reserve = max(1, round(limit * 0.25)) if limit > 1 else 0
    miss_keep = min(len(misses), max(0, limit - hit_reserve))
    hit_keep = min(len(hits), limit - miss_keep)
    return misses[:miss_keep] + hits[:hit_keep]


def _normalize_predictions(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for detection in detections or []:
        box = box_tuple(detection.get("box") if isinstance(detection, dict) else None)
        if box is None:
            continue
        try:
            confidence = float(detection.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        normalized.append(
            {
                "confidence": confidence,
                "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
            }
        )
    return normalized


def _validate_annotations(
    annotations: list[dict[str, Any]],
    *,
    drop_reason: str,
    crop_width: int,
    crop_height: int,
) -> dict[str, Any]:
    if not isinstance(annotations, list) or not annotations:
        raise FaceDetectionDatasetError("A detection label needs at least one annotation")
    drop = any(str(item.get("kind") or "") == "drop" for item in annotations if isinstance(item, dict))
    reason = str(drop_reason or "").strip()
    if drop:
        if not reason:
            raise FaceDetectionDatasetError("Drop needs a reason")
        if len(reason) > 200:
            raise FaceDetectionDatasetError("Drop reason is too long")
        return {"drop": True, "drop_reason": reason, "annotations": []}
    parsed = []
    for item in annotations:
        if not isinstance(item, dict):
            raise FaceDetectionDatasetError("Annotation must be an object")
        kind = str(item.get("kind") or "")
        if kind not in ANNOTATION_KINDS:
            raise FaceDetectionDatasetError(f"Unsupported annotation kind: {kind}")
        box = None
        if kind in BOX_KINDS:
            raw_box = item.get("box")
            parsed_box = box_tuple(raw_box if isinstance(raw_box, dict) else None)
            if parsed_box is None:
                raise FaceDetectionDatasetError(f"{kind} needs a box")
            x1, y1, x2, y2 = parsed_box
            if x1 < 0 or y1 < 0 or x2 > crop_width or y2 > crop_height:
                raise FaceDetectionDatasetError("Box sits outside the upper-body crop")
            box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        elif item.get("box"):
            raise FaceDetectionDatasetError("no_face does not take a box")
        parsed.append({"kind": kind, "box": box})
    kinds = {item["kind"] for item in parsed}
    if "no_face" in kinds and "face" in kinds:
        raise FaceDetectionDatasetError("no_face cannot be combined with a face box")
    if "no_face" in kinds and len(parsed) != 1:
        raise FaceDetectionDatasetError("no_face applies to the whole crop")
    return {"drop": False, "drop_reason": "", "annotations": parsed}


def _safe_stem(source_id: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in source_id)
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:80]}_{digest}"

"""HTTP boundary for face-detection labels, export, and the local IR gate."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .camera import _decode_recording_still
from .config import AppConfig, DetectorConfig
from .face_detection import OpenVinoFaceDetector
from .face_detection_dataset import (
    FaceDetectionDataset,
    FaceDetectionDatasetError,
    file_sha256,
    load_person_observations,
)
from .face_detection_evaluation import evaluate_samples, validate_ir_contract
from .manager import AppManager


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionAnnotation(BaseModel):
    kind: str = Field(pattern=r"^(face|no_face|not_a_face|drop)$")
    box: DetectionBox | None = None


class DetectionLabelRequest(BaseModel):
    annotations: list[DetectionAnnotation] = Field(min_length=1, max_length=32)
    drop_reason: str = Field(default="", max_length=200)


class DetectionMaterializeRequest(BaseModel):
    timezone: str = Field(default="UTC", min_length=1, max_length=64)
    limit: int = Field(default=50, ge=1, le=200)


class DetectionScoreRequest(BaseModel):
    candidate_model_path: str = Field(min_length=1, max_length=4096)
    baseline_model_path: str = Field(min_length=1, max_length=4096)
    threshold: float | None = Field(default=None, ge=0.01, le=0.99)


@dataclass(frozen=True, slots=True)
class FaceDetectionRouteDependencies:
    get_manager: Callable[[], AppManager]
    get_config: Callable[[], AppConfig]
    frame_reader: Callable[[str, float], np.ndarray | None] | None = None
    detector_factory: Callable[[str], Callable[[np.ndarray, float], list[dict[str, Any]]]] | None = None
    fingerprint_reader: Callable[[], str] | None = None


def create_face_detection_router(dependencies: FaceDetectionRouteDependencies) -> APIRouter:
    router = APIRouter()

    def dataset() -> FaceDetectionDataset:
        manager = dependencies.get_manager()
        root = Path(manager.database_dir)
        return FaceDetectionDataset(
            root / "face-detection-dataset.sqlite3",
            root / "face-detection-crops",
        )

    def frame_reader() -> Callable[[str, float], np.ndarray | None]:
        if dependencies.frame_reader is not None:
            return dependencies.frame_reader
        ffmpeg_path = dependencies.get_config().ffmpeg_path

        def read(path: str, seek_seconds: float) -> np.ndarray | None:
            return _decode_recording_still(
                Path(path),
                ffmpeg_path=ffmpeg_path,
                seek_seconds=seek_seconds,
            )

        return read

    def make_detector(model_path: str) -> Callable[[np.ndarray, float], list[dict[str, Any]]]:
        if dependencies.detector_factory is not None:
            return dependencies.detector_factory(model_path)
        return _openvino_detector(dependencies.get_config().detector, model_path)

    def fingerprint() -> str:
        if dependencies.fingerprint_reader is not None:
            return dependencies.fingerprint_reader()
        faces = getattr(dependencies.get_manager(), "faces", None)
        status = faces.recognition_status() if faces is not None and hasattr(faces, "recognition_status") else {}
        return str(status.get("model_fingerprint") or "")

    @router.get("/api/face-detection/samples")
    def face_detection_queue(limit: int = 20) -> dict[str, Any]:
        samples = dataset().queue(limit=limit)
        for sample in samples:
            sample["crop_url"] = f"/api/face-detection/samples/{sample['id']}/crop.png"
        return {"samples": samples, "count": len(samples)}

    @router.get("/api/face-detection/samples/{sample_id}/crop.png")
    def face_detection_crop(sample_id: int) -> FileResponse:
        path = dataset().crop_file(sample_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Detection crop was not found")
        return FileResponse(path, media_type="image/png")

    @router.post("/api/face-detection/samples/materialize")
    def face_detection_materialize(request: DetectionMaterializeRequest) -> dict[str, int]:
        manager = dependencies.get_manager()
        observations = load_person_observations(manager.events.db_path, limit=max(request.limit * 4, request.limit))
        resolved, skipped = _attach_recordings(manager, observations)
        detector = None
        model_path = str(dependencies.get_config().detector.face_detection_model_path or "")
        model_hash = ""
        if model_path and Path(model_path).is_file():
            model_hash = file_sha256(Path(model_path))
            runner = make_detector(model_path)

            def detector(image: np.ndarray, _runner: Callable = runner) -> list[dict[str, Any]]:
                return _runner(image, 0.60)

        try:
            stats = dataset().materialize(
                resolved,
                frame_reader=frame_reader(),
                detector=detector,
                timezone_name=request.timezone,
                limit=request.limit,
                baseline_model_sha256=model_hash,
            )
        except FaceDetectionDatasetError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        stats["skipped"] += skipped
        return stats

    @router.put("/api/face-detection/samples/{sample_id}")
    def face_detection_label(sample_id: int, request: DetectionLabelRequest) -> dict[str, Any]:
        try:
            sample = dataset().label(
                sample_id,
                [item.model_dump() for item in request.annotations],
                drop_reason=request.drop_reason,
            )
        except FaceDetectionDatasetError as exc:
            status = 404 if "not found" in str(exc) else 422
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        sample["crop_url"] = f"/api/face-detection/samples/{sample['id']}/crop.png"
        return sample

    @router.post("/api/face-detection/splits")
    def face_detection_splits() -> dict[str, Any]:
        try:
            assignment = dataset().assign_splits()
        except FaceDetectionDatasetError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"splits": assignment}

    @router.post("/api/face-detection/export")
    def face_detection_export() -> dict[str, Any]:
        root = Path(dependencies.get_manager().database_dir) / "face-detection-exports"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = root / stamp
        suffix = 1
        while destination.exists():
            destination = root / f"{stamp}-{suffix}"
            suffix += 1
        try:
            manifest = dataset().export(destination)
        except FaceDetectionDatasetError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "directory": str(destination),
            "manifest": str(destination / "manifest.json"),
            "samples": len(manifest["samples"]),
            "version": manifest["version"],
        }

    @router.post("/api/face-detection/score")
    def face_detection_score(request: DetectionScoreRequest) -> dict[str, Any]:
        store = dataset()
        samples = store.reviewed_samples()
        if not samples:
            raise HTTPException(status_code=422, detail="Freeze splits on reviewed crops before scoring")
        candidate_path = Path(request.candidate_model_path).expanduser()
        baseline_path = Path(request.baseline_model_path).expanduser()
        try:
            candidate_contract = validate_ir_contract(candidate_path)
            baseline_contract = validate_ir_contract(baseline_path)
            report = evaluate_samples(
                samples,
                make_detector(str(candidate_path)),
                make_detector(str(baseline_path)),
                image_reader=lambda sample: _read_crop(store, int(sample["id"])),
                threshold=request.threshold,
            )
        except (FaceDetectionDatasetError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        report["candidate_contract"] = candidate_contract
        report["baseline_contract"] = baseline_contract
        report["embedding_fingerprint"] = fingerprint()
        report["activation"] = (
            "Copy the passing XML and BIN into the model directory and set "
            "detector.face_detection_model_path to that XML. Compare embedding_fingerprint "
            "after the face worker reloads. It must be unchanged. Leave the previous "
            "detector on disk so the path can be switched back."
        )
        return report

    return router


def _attach_recordings(
    manager: AppManager,
    observations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    recorder = getattr(manager, "recorder", None)
    resolved = []
    skipped = 0
    for observation in observations:
        recording = None
        if recorder is not None and hasattr(recorder, "recording_at"):
            recording = recorder.recording_at(observation["camera_id"], observation["captured_epoch"])
        if not recording or recording.get("start_epoch") is None:
            skipped += 1
            continue
        seek = float(observation["captured_epoch"]) - float(recording["start_epoch"])
        if seek < 0:
            skipped += 1
            continue
        path = str(recording.get("path") or observation.get("recording_path") or "")
        if not path:
            skipped += 1
            continue
        resolved.append({**observation, "recording_path": path, "seek_seconds": seek})
    return resolved, skipped


def _openvino_detector(
    config: DetectorConfig,
    model_path: str,
) -> Callable[[np.ndarray, float], list[dict[str, Any]]]:
    updated = config.model_copy(
        update={
            "face_recognition_enabled": True,
            "face_detection_model_path": model_path,
        }
    )
    detector = OpenVinoFaceDetector(updated)
    if not detector.ready:
        raise FaceDetectionDatasetError(detector.error or "Face detector failed to load")

    def run(image: np.ndarray, threshold: float) -> list[dict[str, Any]]:
        return detector.detect(image, threshold=threshold)

    return run


def _read_crop(store: FaceDetectionDataset, sample_id: int) -> np.ndarray:
    path = store.crop_file(sample_id)
    if path is None:
        raise FaceDetectionDatasetError(f"Crop for sample {sample_id} is missing")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FaceDetectionDatasetError(f"Crop for sample {sample_id} is unreadable")
    return image


def embedding_fingerprint_unchanged(before: str, after: str) -> bool:
    """The face-detector path swap must not change the embedding model identity."""
    return bool(before) and before == after

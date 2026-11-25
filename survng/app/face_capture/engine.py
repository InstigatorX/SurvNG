"""Face capture models, owned by the isolated face worker.

YuNet finds faces (with five landmarks) in an upper-body crop; each face is
aligned on its own landmarks and AdaFace returns its unit embedding and its
feature norm, the quality signal used to drop non-faces and unrecognisable
faces.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# YuNet's cost grows with input size; larger crops are downscaled first.
YUNET_MAX_SIDE = 640
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 50
ALIGNED_SIZE = (112, 112)
# The ArcFace five-point template for a 112x112 aligned face (left eye,
# right eye, nose, left and right mouth corner, in image order).
ALIGNMENT_TEMPLATE = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def resolve_model_path(value: str) -> Path:
    """A configured model path, or the same path under ./models for ``/models/...``."""
    path = Path(value).expanduser()
    if path.is_file():
        return path
    if path.is_absolute() and path.parts[1:2] == ("models",):
        local = Path("models", *path.parts[2:])
        if local.is_file():
            return local
    raise FileNotFoundError(f"model not found: {value}")


def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
    """The 112x112 face aligned on five landmarks (image coordinates)."""
    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    if not np.all(np.isfinite(points)):
        return None
    matrix, _ = cv2.estimateAffinePartial2D(points, ALIGNMENT_TEMPLATE, method=cv2.LMEDS)
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None
    return cv2.warpAffine(image, matrix, ALIGNED_SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class FaceCaptureEngine:
    """Detect, align and score faces in one crop."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.ready = False
        self.error = ""
        self.model_load_ms: float | None = None
        self._lock = threading.Lock()
        self._detector: Any = None
        self._embedder: Any = None
        self.embedding_model = ""
        try:
            self._load()
        except Exception as exc:
            self.error = str(exc) or "Face capture models failed to load."
            LOGGER.exception("face capture models failed to load")

    def _load(self) -> None:
        started = time.perf_counter()
        detector_path = resolve_model_path(self.config.face_capture_detector_model_path)
        embedder_path = resolve_model_path(self.config.face_capture_embedding_model_path)
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320),
            float(self.config.face_capture_detection_threshold), YUNET_NMS_THRESHOLD, YUNET_TOP_K,
        )
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core
        from ..inference_runtime.worker_topology import openvino_latency_compile_config

        core = Core()
        model = core.read_model(str(embedder_path))
        if [list(output.shape) for output in model.outputs] != [[1, 512], [1, 1]]:
            raise ValueError("face capture embedding model must output a 512-d embedding and a norm")
        compile_config = openvino_latency_compile_config("CPU")
        # The AdaFace export failed parity at the GPU's default f16; f32 on CPU.
        compile_config["INFERENCE_PRECISION_HINT"] = "f32"
        compiled = core.compile_model(model, "CPU", compile_config)
        self._embedder = compiled
        self.embedding_model = embedder_path.stem
        self.model_load_ms = round((time.perf_counter() - started) * 1000, 1)
        self.ready = True

    def _embed(self, aligned: np.ndarray) -> tuple[np.ndarray, float]:
        tensor = ((aligned.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None].copy()
        result = self._embedder([tensor])
        embedding = np.asarray(result[self._embedder.output(0)], dtype=np.float32).reshape(-1)
        norm = float(np.asarray(result[self._embedder.output(1)]).reshape(-1)[0])
        return embedding, norm

    def capture(self, crop: np.ndarray) -> list[dict[str, Any]]:
        """Faces in ``crop``: box and landmarks in crop pixels, confidence, norm, embedding."""
        if not self.ready:
            raise RuntimeError(self.error or "Face capture is unavailable.")
        height, width = crop.shape[:2]
        if width < 8 or height < 8:
            return []
        scale = min(1.0, YUNET_MAX_SIDE / max(height, width))
        image = crop if scale == 1.0 else cv2.resize(
            crop, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA,
        )
        faces: list[dict[str, Any]] = []
        with self._lock:
            self._detector.setInputSize((image.shape[1], image.shape[0]))
            _, rows = self._detector.detect(image)
            for row in () if rows is None else rows:
                values = np.asarray(row[:14], dtype=np.float32) / scale
                x, y, box_width, box_height = (float(value) for value in values[:4])
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
                if x2 <= x1 or y2 <= y1:
                    continue
                landmarks = values[4:14].reshape(5, 2)
                aligned = align_face(crop, landmarks)
                if aligned is None:
                    continue
                embedding, norm = self._embed(aligned)
                faces.append({
                    "box": [x1, y1, x2, y2],
                    "landmarks": landmarks.round(2).tolist(),
                    "confidence": round(float(row[14]), 4),
                    "norm": round(norm, 3),
                    "embedding": embedding.tolist(),
                })
        return faces

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "error": self.error,
            "detector_model_path": self.config.face_capture_detector_model_path,
            "embedding_model_path": self.config.face_capture_embedding_model_path,
            "embedding_model": self.embedding_model,
            "model_load_ms": self.model_load_ms,
        }

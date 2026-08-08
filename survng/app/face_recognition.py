from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np

from .config import DetectorConfig


LOGGER = logging.getLogger(__name__)


class OpenVinoFaceRecognizer:
    """Generate normalized face embeddings without sharing detector inference state."""

    _ARCFACE_TEMPLATE = np.asarray(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.enabled = bool(config.face_recognition_enabled)
        self.ready = False
        self.error = ""
        self.loaded_device = ""
        self.model_fingerprint = ""
        self.input_shape = (128, 128)
        self.input_layout = "NCHW"
        self.input_color_order = "BGR"
        self.embedding_size = 0
        self.landmark_input_shape = (48, 48)
        self.landmark_input_layout = "NCHW"
        self.alignment_enabled = False
        self.model_load_ms: float | None = None
        self._compiled_model: Any = None
        self._infer_request: Any = None
        self._input: Any = None
        self._output: Any = None
        self._landmark_request: Any = None
        self._landmark_input: Any = None
        self._landmark_output: Any = None
        self._lock = threading.Lock()
        if self.enabled:
            self._load()
        else:
            self.error = "Face recognition is disabled."

    def _load(self) -> None:
        model_path = Path(self.config.face_embedding_model_path).expanduser()
        landmark_path = Path(self.config.face_landmark_model_path).expanduser()
        if not model_path.is_file():
            self.error = "Configure a valid OpenVINO face embedding model path."
            return
        if not landmark_path.is_file():
            self.error = "Configure a valid OpenVINO face landmark model path."
            return
        started = time.perf_counter()
        try:
            try:
                from openvino import Core
            except ImportError:
                from openvino.runtime import Core
            core = Core()
            core.set_property({"ENABLE_MMAP": True})
            if self.config.cache_enabled:
                cache_dir = Path(self.config.cache_dir or ".cache/openvino").expanduser().resolve()
                cache_dir.mkdir(parents=True, exist_ok=True)
                core.set_property({"CACHE_DIR": str(cache_dir)})
            model = core.read_model(model=model_path)
            self.input_layout, self.input_shape = self._image_input(model.input(0).shape)
            if "arcface" in model_path.name.lower():
                self.input_color_order = "RGB"
            landmark_model = core.read_model(model=landmark_path)
            self.landmark_input_layout, self.landmark_input_shape = self._image_input(
                landmark_model.input(0).shape
            )
            device = self.config.face_recognition_device or "AUTO"
            compile_config = {"PERFORMANCE_HINT": "LATENCY"}
            if device.upper() != "AUTO":
                compile_config["NUM_STREAMS"] = "1"
            try:
                self._compiled_model = core.compile_model(model, device, compile_config)
                self.loaded_device = device
            except Exception:
                if device.upper() == "CPU":
                    raise
                LOGGER.warning("Face embedding model failed on %s; retrying on CPU", device)
                self._compiled_model = core.compile_model(
                    model, "CPU", {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
                )
                self.loaded_device = "CPU"
            self._infer_request = self._compiled_model.create_infer_request()
            self._input = self._compiled_model.input(0)
            self._output = self._compiled_model.output(0)
            try:
                compiled_landmarks = core.compile_model(landmark_model, self.loaded_device, compile_config)
            except Exception:
                if self.loaded_device.upper() == "CPU":
                    raise
                LOGGER.warning("Face landmark model failed on %s; retrying on CPU", self.loaded_device)
                compiled_landmarks = core.compile_model(
                    landmark_model, "CPU", {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
                )
            self._landmark_request = compiled_landmarks.create_infer_request()
            self._landmark_input = compiled_landmarks.input(0)
            self._landmark_output = compiled_landmarks.output(0)
            output_shape = [int(value) for value in self._output.shape]
            self.embedding_size = int(np.prod(output_shape[1:] or output_shape))
            self.model_fingerprint = self._fingerprint(model_path, landmark_path)
            self.model_load_ms = round((time.perf_counter() - started) * 1000, 1)
            self._infer_request.infer(
                {self._input: self._image_tensor(np.zeros((self.input_shape[1], self.input_shape[0], 3), dtype=np.uint8), self.input_shape, self.input_layout)}
            )
            self._landmark_request.infer(
                {self._landmark_input: self._image_tensor(np.zeros((48, 48, 3), dtype=np.uint8), self.landmark_input_shape, self.landmark_input_layout)}
            )
            self.alignment_enabled = True
            self.ready = True
            LOGGER.info(
                "OpenVINO face recognition ready on %s in %.1f ms (%d dimensions)",
                self.loaded_device,
                self.model_load_ms,
                self.embedding_size,
            )
        except Exception as exc:
            self.error = str(exc) or "Face embedding model failed to load."
            LOGGER.exception("OpenVINO face embedding model failed to load %s", model_path)

    @staticmethod
    def _image_input(raw_shape: Any) -> tuple[str, tuple[int, int]]:
        shape = [int(value) for value in raw_shape]
        if len(shape) != 4:
            raise ValueError(f"expected a four-dimensional image input, got {shape}")
        if shape[1] == 3:
            return "NCHW", (shape[3], shape[2])
        if shape[3] == 3:
            return "NHWC", (shape[2], shape[1])
        raise ValueError(f"expected a three-channel image input, got {shape}")

    @staticmethod
    def _image_tensor(image: np.ndarray, shape: tuple[int, int], layout: str) -> np.ndarray:
        height, width = image.shape[:2]
        interpolation = (
            cv2.INTER_AREA
            if width >= shape[0] and height >= shape[1]
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(image, shape, interpolation=interpolation)
        tensor = resized.astype(np.float32)
        if layout == "NCHW":
            tensor = np.transpose(tensor, (2, 0, 1))
        return np.expand_dims(tensor, axis=0)

    @staticmethod
    def _fingerprint(*model_paths: Path) -> str:
        digest = hashlib.sha256()
        for model_path in model_paths:
            for path in (model_path, model_path.with_suffix(".bin")):
                if not path.is_file():
                    continue
                digest.update(path.name.encode("utf-8"))
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()[:24]

    def _align(self, face: np.ndarray) -> np.ndarray:
        if self._landmark_request is None:
            raise RuntimeError("Face landmark alignment is unavailable.")
        height, width = face.shape[:2]
        if width < 2 or height < 2:
            raise ValueError("Face crop is too small for alignment.")
        tensor = self._image_tensor(face, self.landmark_input_shape, self.landmark_input_layout)
        result = self._landmark_request.infer({self._landmark_input: tensor})[
            self._landmark_output
        ]
        values = np.asarray(result, dtype=np.float32).reshape(-1)
        if values.size < 10 or not np.all(np.isfinite(values[:10])):
            raise ValueError("Face landmark output was invalid.")
        source = values[:10].reshape(5, 2)
        source[:, 0] *= width
        source[:, 1] *= height
        target = self._ARCFACE_TEMPLATE.copy()
        target[:, 0] *= self.input_shape[0] / 112.0
        target[:, 1] *= self.input_shape[1] / 112.0
        matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
        if matrix is None or not np.all(np.isfinite(matrix)):
            raise ValueError("Could not align face landmarks.")
        return cv2.warpAffine(
            face,
            matrix,
            self.input_shape,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def embed(self, face: np.ndarray) -> np.ndarray:
        if self._infer_request is None:
            raise RuntimeError(self.error or "Face recognition is unavailable.")
        with self._lock:
            aligned = self._align(face)
            if self.input_color_order == "RGB":
                aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            tensor = self._image_tensor(aligned, self.input_shape, self.input_layout)
            result = self._infer_request.infer({self._input: tensor})[self._output]
        vector = np.asarray(result, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-9:
            raise ValueError("Face embedding was empty or invalid.")
        return vector / norm

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "error": self.error,
            "device": self.loaded_device or self.config.face_recognition_device,
            "model_path": self.config.face_embedding_model_path,
            "landmark_model_path": self.config.face_landmark_model_path,
            "alignment_enabled": self.alignment_enabled,
            "landmark_input_shape": list(self.landmark_input_shape),
            "model_fingerprint": self.model_fingerprint,
            "input_shape": list(self.input_shape),
            "input_color_order": self.input_color_order,
            "embedding_size": self.embedding_size,
            "model_load_ms": self.model_load_ms,
            "match_threshold": self.config.face_match_threshold,
        }

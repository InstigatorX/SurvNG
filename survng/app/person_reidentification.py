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


class OpenVinoPersonReidentifier:
    """Generate normalized whole-person appearance embeddings."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        tracking = config.tracking
        self.enabled = bool(tracking.reid_enabled)
        self.ready = False
        self.error = ""
        self.loaded_device = ""
        self.model_fingerprint = ""
        self.input_shape = (128, 256)
        self.input_layout = "NCHW"
        self.embedding_size = 0
        self.model_load_ms: float | None = None
        self._infer_request: Any = None
        self._input: Any = None
        self._output: Any = None
        self._lock = threading.Lock()
        if self.enabled:
            self._load()
        else:
            self.error = "Person ReID is disabled."

    def _load(self) -> None:
        model_path = Path(self.config.tracking.reid_model_path).expanduser()
        if not model_path.is_file():
            self.error = "Configure a valid OpenVINO person ReID model path."
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
            device = self.config.tracking.reid_device or "AUTO"
            compile_config = {"PERFORMANCE_HINT": "LATENCY"}
            if device.upper() != "AUTO":
                compile_config["NUM_STREAMS"] = "1"
            try:
                compiled = core.compile_model(model, device, compile_config)
                self.loaded_device = device
            except Exception:
                if device.upper() == "CPU":
                    raise
                LOGGER.warning("Person ReID model failed on %s; retrying on CPU", device)
                compiled = core.compile_model(
                    model,
                    "CPU",
                    {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"},
                )
                self.loaded_device = "CPU"
            self._infer_request = compiled.create_infer_request()
            self._input = compiled.input(0)
            self._output = compiled.output(0)
            output_shape = [int(value) for value in self._output.shape]
            self.embedding_size = int(np.prod(output_shape[1:] or output_shape))
            self.model_fingerprint = self._fingerprint(model_path)
            self._infer_request.infer({
                self._input: self._image_tensor(
                    np.zeros((self.input_shape[1], self.input_shape[0], 3), dtype=np.uint8)
                )
            })
            self.model_load_ms = round((time.perf_counter() - started) * 1000, 1)
            self.ready = True
            LOGGER.info(
                "OpenVINO person ReID ready on %s in %.1f ms (%d dimensions)",
                self.loaded_device,
                self.model_load_ms,
                self.embedding_size,
            )
        except Exception as exc:
            self.error = str(exc) or "Person ReID model failed to load."
            LOGGER.exception("OpenVINO person ReID model failed to load %s", model_path)

    @staticmethod
    def _image_input(raw_shape: Any) -> tuple[str, tuple[int, int]]:
        shape = [int(value) for value in raw_shape]
        if len(shape) != 4:
            raise ValueError(f"expected a four-dimensional image input, got {shape}")
        if shape[1] in (1, 3, 4):
            return "NCHW", (shape[3], shape[2])
        if shape[3] in (1, 3, 4):
            return "NHWC", (shape[2], shape[1])
        raise ValueError(f"could not determine input layout from {shape}")

    def _image_tensor(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32)
        if self.input_layout == "NCHW":
            tensor = np.transpose(tensor, (2, 0, 1))
        return np.expand_dims(tensor, axis=0)

    @staticmethod
    def _fingerprint(model_path: Path) -> str:
        digest = hashlib.sha256()
        for path in (model_path, model_path.with_suffix(".bin")):
            if not path.is_file():
                continue
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()[:24]

    def embed(self, person: np.ndarray) -> np.ndarray:
        if self._infer_request is None:
            raise RuntimeError(self.error or "Person ReID is unavailable.")
        if person.ndim != 3 or person.shape[2] != 3 or min(person.shape[:2]) < 8:
            raise ValueError("Person crop is too small for ReID.")
        with self._lock:
            result = self._infer_request.infer({
                self._input: self._image_tensor(person)
            })[self._output]
        vector = np.asarray(result, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-9:
            raise ValueError("Person ReID embedding was empty or invalid.")
        return vector / norm

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "error": self.error,
            "device": self.loaded_device or self.config.tracking.reid_device,
            "model_path": self.config.tracking.reid_model_path,
            "model_fingerprint": self.model_fingerprint,
            "input_shape": list(self.input_shape),
            "embedding_size": self.embedding_size,
            "model_load_ms": self.model_load_ms,
            "match_threshold": self.config.tracking.reid_match_threshold,
        }

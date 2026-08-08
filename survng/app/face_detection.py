from __future__ import annotations

import logging
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np

from .config import DetectorConfig


LOGGER = logging.getLogger(__name__)


class OpenVinoFaceDetector:
    """Small dedicated face detector owned by the isolated face worker."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.enabled = bool(
            config.face_recognition_enabled and config.face_detection_model_path
        )
        self.ready = False
        self.error = ""
        self.loaded_device = ""
        self.input_shape = (300, 300)
        self.model_load_ms: float | None = None
        self._request: Any = None
        self._input: Any = None
        self._output: Any = None
        self._lock = threading.Lock()
        if self.enabled:
            self._load()
        else:
            self.error = "Dedicated face detection is not configured."

    def _load(self) -> None:
        model_path = Path(self.config.face_detection_model_path).expanduser()
        if not model_path.is_file():
            self.error = "Configure a valid OpenVINO face detector model path."
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
            shape = [int(value) for value in model.input(0).shape]
            if len(shape) != 4 or shape[1] != 3:
                raise ValueError(f"expected NCHW face detector input, got {shape}")
            self.input_shape = (shape[3], shape[2])
            device = self.config.face_recognition_device or "AUTO"
            compile_config = {"PERFORMANCE_HINT": "LATENCY"}
            if device.upper() != "AUTO":
                compile_config["NUM_STREAMS"] = "1"
            try:
                compiled = core.compile_model(model, device, compile_config)
                self.loaded_device = device
            except Exception:
                if device.upper() == "CPU":
                    raise
                LOGGER.warning("Face detector failed on %s; retrying on CPU", device)
                compiled = core.compile_model(
                    model,
                    "CPU",
                    {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"},
                )
                self.loaded_device = "CPU"
            self._request = compiled.create_infer_request()
            self._input = compiled.input(0)
            self._output = compiled.output(0)
            self.model_load_ms = round((time.perf_counter() - started) * 1000, 1)
            self.detect(np.zeros((300, 300, 3), dtype=np.uint8), threshold=1.0)
            self.ready = True
        except Exception as exc:
            self.error = str(exc) or "Face detector failed to load."
            LOGGER.exception("OpenVINO face detector failed to load %s", model_path)

    def detect(
        self,
        frame: np.ndarray,
        *,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if self._request is None:
            return []
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            return []
        resized = cv2.resize(frame, self.input_shape, interpolation=cv2.INTER_AREA)
        tensor = np.expand_dims(np.transpose(resized.astype(np.float32), (2, 0, 1)), axis=0)
        with self._lock:
            raw = self._request.infer({self._input: tensor})[self._output]
        minimum = max(
            0.01,
            min(
                0.99,
                float(
                    self.config.face_detection_threshold
                    if threshold is None
                    else threshold
                ),
            ),
        )
        detections: list[dict[str, Any]] = []
        for row in np.asarray(raw, dtype=np.float32).reshape(-1, 7):
            confidence = float(row[2])
            if not np.isfinite(confidence) or confidence < minimum:
                continue
            x1 = max(0.0, min(float(width), float(row[3]) * width))
            y1 = max(0.0, min(float(height), float(row[4]) * height))
            x2 = max(0.0, min(float(width), float(row[5]) * width))
            y2 = max(0.0, min(float(height), float(row[6]) * height))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "label": "face",
                    "confidence": round(confidence, 4),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "detection_source": "dedicated_face",
                }
            )
        return detections

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "error": self.error,
            "device": self.loaded_device or self.config.face_recognition_device,
            "model_path": self.config.face_detection_model_path,
            "input_shape": list(self.input_shape),
            "model_load_ms": self.model_load_ms,
            "threshold": self.config.face_detection_threshold,
        }

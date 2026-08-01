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

    def __init__(
        self,
        config: DetectorConfig,
        *,
        kind: str = "Person",
        enabled: bool | None = None,
        model_path: str | None = None,
        device: str | None = None,
        match_threshold: float | None = None,
        default_input_shape: tuple[int, int] = (128, 256),
        input_color_order: str = "BGR",
    ) -> None:
        self.config = config
        tracking = config.tracking
        self.kind = kind
        self.enabled = bool(tracking.reid_enabled if enabled is None else enabled)
        self.configured_model_path = (
            tracking.reid_model_path if model_path is None else model_path
        )
        self.configured_device = tracking.reid_device if device is None else device
        self.match_threshold = (
            tracking.reid_match_threshold
            if match_threshold is None
            else match_threshold
        )
        self.input_color_order = input_color_order.upper()
        self.ready = False
        self.error = ""
        self.loaded_device = ""
        self.model_fingerprint = ""
        self.input_shape = default_input_shape
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
            self.error = f"{self.kind} ReID is disabled."

    def _load(self) -> None:
        model_path = Path(self.configured_model_path).expanduser()
        if not model_path.is_file():
            self.error = f"Configure a valid OpenVINO {self.kind.lower()} ReID model path."
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
            if model.input(0).partial_shape.is_dynamic:
                width, height = self.input_shape
                model.reshape({model.input(0).any_name: [1, 3, height, width]})
            self.input_layout, self.input_shape = self._image_input(model.input(0).shape)
            device = self.configured_device or "AUTO"
            compile_config = {"PERFORMANCE_HINT": "LATENCY"}
            if device.upper() != "AUTO":
                compile_config["NUM_STREAMS"] = "1"
            try:
                compiled = core.compile_model(model, device, compile_config)
                self.loaded_device = device
            except Exception:
                if device.upper() == "CPU":
                    raise
                LOGGER.warning("%s ReID model failed on %s; retrying on CPU", self.kind, device)
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
                "OpenVINO %s ReID ready on %s in %.1f ms (%d dimensions)",
                self.kind.lower(),
                self.loaded_device,
                self.model_load_ms,
                self.embedding_size,
            )
        except Exception as exc:
            self.error = str(exc) or f"{self.kind} ReID model failed to load."
            LOGGER.exception("OpenVINO %s ReID model failed to load %s", self.kind.lower(), model_path)

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
        if self.input_color_order == "RGB":
            tensor = tensor[:, :, ::-1]
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
            raise RuntimeError(self.error or f"{self.kind} ReID is unavailable.")
        if person.ndim != 3 or person.shape[2] != 3 or min(person.shape[:2]) < 8:
            raise ValueError(f"{self.kind} crop is too small for ReID.")
        with self._lock:
            result = self._infer_request.infer({
                self._input: self._image_tensor(person)
            })[self._output]
        vector = np.asarray(result, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-9:
            raise ValueError(f"{self.kind} ReID embedding was empty or invalid.")
        return vector / norm

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "error": self.error,
            "kind": self.kind.lower(),
            "device": self.loaded_device or self.configured_device,
            "model_path": self.configured_model_path,
            "model_fingerprint": self.model_fingerprint,
            "input_shape": list(self.input_shape),
            "input_color_order": self.input_color_order,
            "embedding_size": self.embedding_size,
            "model_load_ms": self.model_load_ms,
            "match_threshold": self.match_threshold,
        }


class OpenVinoAppearanceReidentifier:
    """Route appearance crops to label-specific OpenVINO ReID models."""

    def __init__(self, config: DetectorConfig) -> None:
        tracking = config.tracking
        self.config = tracking
        self.person = OpenVinoPersonReidentifier(config)
        self.vehicle = OpenVinoPersonReidentifier(
            config,
            kind="Vehicle",
            enabled=tracking.vehicle_reid_enabled,
            model_path=tracking.vehicle_reid_model_path,
            device=tracking.vehicle_reid_device,
            match_threshold=tracking.vehicle_reid_match_threshold,
            default_input_shape=(208, 208),
            input_color_order="RGB",
        )
        self.enabled = tracking.appearance_reid_enabled

    @property
    def ready(self) -> bool:
        enabled_engines = [
            engine for engine in (self.person, self.vehicle) if engine.enabled
        ]
        return bool(enabled_engines) and all(engine.ready for engine in enabled_engines)

    def supports_label(self, label: str) -> bool:
        normalized = str(label or "").strip().lower()
        if normalized == "person":
            return bool(self.person.enabled and self.person.ready)
        return bool(
            normalized in self.config.vehicle_reid_labels
            and self.vehicle.enabled
            and self.vehicle.ready
        )

    def embed(self, person: np.ndarray) -> np.ndarray:
        return self.person.embed(person)

    def embed_for_label(self, label: str, crop: np.ndarray) -> np.ndarray:
        normalized = str(label or "").strip().lower()
        if normalized == "person" and self.person.enabled:
            return self.person.embed(crop)
        if normalized in self.config.vehicle_reid_labels and self.vehicle.enabled:
            return self.vehicle.embed(crop)
        raise ValueError(f"ReID is not configured for label {normalized!r}")

    def model_identity_for_label(self, label: str) -> dict[str, Any] | None:
        normalized = str(label or "").strip().lower()
        if normalized == "person":
            engine = self.person
            model_kind = "person"
        elif normalized in self.config.vehicle_reid_labels:
            engine = self.vehicle
            model_kind = "vehicle"
        else:
            return None
        if not engine.ready or not engine.model_fingerprint:
            return None
        return {
            "model_kind": model_kind,
            "model_fingerprint": engine.model_fingerprint,
            "embedding_size": engine.embedding_size,
            "match_threshold": engine.match_threshold,
        }

    def status(self) -> dict[str, Any]:
        person = self.person.status()
        vehicle = self.vehicle.status()
        failures = [
            status["error"]
            for status in (person, vehicle)
            if status["enabled"] and not status["ready"] and status["error"]
        ]
        return {
            **person,
            "enabled": self.enabled,
            "ready": self.ready,
            "error": "; ".join(failures),
            "person": person,
            "vehicle": {
                **vehicle,
                "labels": list(self.config.vehicle_reid_labels),
            },
        }

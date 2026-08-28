from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DepthConfig, DetectorConfig

LOGGER = logging.getLogger(__name__)


def _box_coords(box: object) -> tuple[int, int, int, int] | None:
    if isinstance(box, dict):
        try:
            x1 = int(round(float(box["x1"])))
            y1 = int(round(float(box["y1"])))
            x2 = int(round(float(box["x2"])))
            y2 = int(round(float(box["y2"])))
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(box, (list, tuple)) and len(box) >= 4:
        try:
            x1 = int(round(float(box[0])))
            y1 = int(round(float(box[1])))
            x2 = int(round(float(box[2])))
            y2 = int(round(float(box[3])))
        except (TypeError, ValueError):
            return None
    else:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def sample_bbox_depth_stats(
    depth_map: np.ndarray,
    box: object,
    *,
    min_m: float,
    max_m: float,
) -> dict[str, Any] | None:
    """Summarize depth inside a detection box."""
    coords = _box_coords(box)
    if coords is None or depth_map.ndim != 2:
        return None
    height, width = depth_map.shape
    x1, y1, x2, y2 = coords
    left = max(0, min(width, x1))
    top = max(0, min(height, y1))
    right = max(left + 1, min(width, x2))
    bottom = max(top + 1, min(height, y2))
    region = depth_map[top:bottom, left:right]
    if region.size == 0:
        return None
    valid = region[np.isfinite(region) & (region >= min_m) & (region <= max_m)]
    if valid.size == 0:
        return None
    return {
        "median_m": round(float(np.median(valid)), 2),
        "min_m": round(float(np.min(valid)), 2),
        "max_m": round(float(np.max(valid)), 2),
        "p25_m": round(float(np.percentile(valid, 25)), 2),
        "p75_m": round(float(np.percentile(valid, 75)), 2),
        "valid_fraction": round(float(valid.size / region.size), 3),
    }


def scale_depth_map(
    depth_map: np.ndarray,
    target_shape: tuple[int, int],
    metadata: dict[str, float],
) -> np.ndarray:
    """Map a letterboxed depth tensor back to the original image size."""
    target_height, target_width = target_shape
    scale = float(metadata.get("scale") or 1.0)
    pad_x = float(metadata.get("pad_x") or 0.0)
    pad_y = float(metadata.get("pad_y") or 0.0)
    input_width = int(metadata.get("input_width") or depth_map.shape[1])
    input_height = int(metadata.get("input_height") or depth_map.shape[0])
    image_width = int(metadata.get("image_width") or target_width)
    image_height = int(metadata.get("image_height") or target_height)

    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    left = int(round(pad_x))
    top = int(round(pad_y))
    right = min(input_width, left + resized_width)
    bottom = min(input_height, top + resized_height)
    cropped = depth_map[top:bottom, left:right]
    if cropped.size == 0:
        return np.zeros((target_height, target_width), dtype=np.float32)
    restored = cv2.resize(
        cropped.astype(np.float32),
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return restored


def encode_depth_heatmap(depth_map: np.ndarray, *, max_width: int = 192) -> bytes:
    """Encode a depth map as a compact false-color PNG."""
    finite = np.isfinite(depth_map)
    if not finite.any():
        return b""
    values = depth_map[finite]
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    if high <= low:
        high = low + 1.0
    normalized = np.clip((depth_map - low) / (high - low), 0.0, 1.0)
    normalized = np.where(finite, normalized, 0.0)
    height, width = normalized.shape
    if width > max_width:
        target_height = max(1, int(round(height * (max_width / width))))
        normalized = cv2.resize(
            normalized.astype(np.float32),
            (max_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
    colored = cv2.applyColorMap(
        (normalized * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    success, encoded = cv2.imencode(".png", colored)
    return encoded.tobytes() if success else b""


def depth_motion_evidence_values(
    objects: list[dict[str, Any]],
    *,
    captured_at: float,
    frame_offset_s: float,
) -> dict[str, Any]:
    """Build motion-evidence payload from per-object depth stats."""
    depths = [
        float(item["depth_stats"]["median_m"])
        for item in objects
        if isinstance(item, dict)
        and isinstance(item.get("depth_stats"), dict)
        and item["depth_stats"].get("median_m") is not None
    ]
    if not depths:
        return {}
    nearest = min(depths)
    farthest = max(depths)
    foreground_score = round(max(0.0, min(1.0, 1.0 - (nearest / 30.0))), 3)
    return {
        "captured_at": captured_at,
        "frame_offset_s": frame_offset_s,
        "object_count": len(depths),
        "nearest_m": round(nearest, 2),
        "farthest_m": round(farthest, 2),
        "median_m": round(float(np.median(depths)), 2),
        "foreground_score": foreground_score,
        "score": foreground_score,
        "warmed": 1.0,
    }


class OpenVinoDepthEstimator:
    """Monocular depth estimator backed by an OpenVINO-exported YOLO26-depth model."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._apply_depth_config(config.depth)
        self.ready = False
        self.error = ""
        self.loaded_device = ""
        self.model_load_ms: float | None = None
        self._last_inference_ms: float | None = None
        self._request: Any = None
        self._input: Any = None
        self._output: Any = None
        self._lock = threading.Lock()
        self._resize_buffers: dict[tuple[int, int], np.ndarray] = {}
        self._preprocess_canvas: np.ndarray | None = None
        self._embedded_preprocess = False
        if self.enabled:
            self._load()
        else:
            self.error = "Depth estimation is not configured."

    def _apply_depth_config(self, depth: DepthConfig) -> None:
        self.depth_config = depth
        self.enabled = bool(depth.enabled and depth.resolved_model_path())
        self.input_shape = (depth.input_size, depth.input_size)

    def _load(self) -> None:
        model_path = Path(self.depth_config.resolved_model_path()).expanduser()
        if not model_path.is_file():
            self.error = "Configure a valid OpenVINO depth model path."
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
            if len(shape) == 4:
                if shape[1] == 3:
                    self.input_shape = (shape[3], shape[2])
                elif shape[-1] == 3:
                    self.input_shape = (shape[2], shape[1])
            device = self.depth_config.device or "AUTO"
            compile_config = {"PERFORMANCE_HINT": "LATENCY"}
            if device.upper() != "AUTO":
                compile_config["NUM_STREAMS"] = "1"
            try:
                compiled = core.compile_model(model, device, compile_config)
                self.loaded_device = device
            except Exception:
                if device.upper() == "CPU":
                    raise
                LOGGER.warning("Depth estimator failed on %s; retrying on CPU", device)
                compiled = core.compile_model(
                    model,
                    "CPU",
                    {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"},
                )
                self.loaded_device = "CPU"
            self._request = compiled.create_infer_request()
            self._input = compiled.input(0)
            self._output = compiled.output(0)
            input_dtype = str(self._input.get_element_type()).lower()
            self._embedded_preprocess = "u8" in input_dtype or "uint8" in input_dtype
            self.model_load_ms = round((time.perf_counter() - started) * 1000, 1)
            self.estimate_depth_map(
                np.zeros((self.input_shape[1], self.input_shape[0], 3), dtype=np.uint8)
            )
            self.ready = True
        except Exception as exc:
            # Keep native OpenVINO messages, paths, and device details out of
            # runtime status and support-facing logs. The exception class is
            # enough to distinguish dependency/runtime failure categories.
            self.error = "Depth estimator failed to load."
            LOGGER.error(
                "OpenVINO depth estimator failed to load (%s)",
                type(exc).__name__,
            )

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        input_width, input_height = self.input_shape
        image_height, image_width = frame.shape[:2]
        scale = min(input_width / image_width, input_height / image_height)
        resized_width = max(1, int(round(image_width * scale)))
        resized_height = max(1, int(round(image_height * scale)))
        pad_x = (input_width - resized_width) / 2
        pad_y = (input_height - resized_height) / 2

        resize_key = (resized_width, resized_height)
        resized = self._resize_buffers.get(resize_key)
        if resized is None:
            resized = np.empty((resized_height, resized_width, 3), dtype=np.uint8)
            if len(self._resize_buffers) >= 8:
                self._resize_buffers.clear()
            self._resize_buffers[resize_key] = resized
        cv2.resize(frame, (resized_width, resized_height), dst=resized)
        if self._preprocess_canvas is None or self._preprocess_canvas.shape != (input_height, input_width, 3):
            self._preprocess_canvas = np.empty((input_height, input_width, 3), dtype=np.uint8)
        canvas = self._preprocess_canvas
        canvas.fill(114)
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))
        canvas[top : top + resized_height, left : left + resized_width] = resized

        if self._embedded_preprocess:
            tensor = canvas[np.newaxis, :]
        else:
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            tensor = rgb.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32) / 255.0
        metadata = {
            "scale": scale,
            "pad_x": float(left),
            "pad_y": float(top),
            "input_width": float(input_width),
            "input_height": float(input_height),
            "image_width": float(image_width),
            "image_height": float(image_height),
        }
        return tensor, metadata

    def _parse_depth_output(self, raw: Any) -> np.ndarray:
        array = np.asarray(raw, dtype=np.float32)
        squeezed = np.squeeze(array)
        if squeezed.ndim == 3 and squeezed.shape[0] == 1:
            squeezed = squeezed[0]
        if squeezed.ndim != 2:
            raise ValueError(f"expected 2D depth map, got shape {array.shape}")
        return squeezed

    def estimate_depth_map(self, frame: np.ndarray) -> np.ndarray:
        if self._request is None:
            raise RuntimeError(self.error or "depth estimator unavailable")
        tensor, metadata = self._preprocess(frame)
        started = time.perf_counter()
        with self._lock:
            raw = self._request.infer({self._input: tensor})[self._output]
        self._last_inference_ms = round((time.perf_counter() - started) * 1000, 1)
        depth_map = self._parse_depth_output(raw)
        return scale_depth_map(
            depth_map,
            (frame.shape[0], frame.shape[1]),
            metadata,
        )

    def estimate_object_depth_stats(
        self,
        frame: np.ndarray,
        objects: list[dict[str, Any]],
        *,
        frame_offset_s: float | None = None,
    ) -> list[dict[str, Any] | None]:
        depth_map = self.estimate_depth_map(frame)
        min_m = float(self.depth_config.min_distance_m)
        max_m = float(self.depth_config.max_distance_m)
        stats: list[dict[str, Any] | None] = []
        for item in objects:
            if not isinstance(item, dict):
                stats.append(None)
                continue
            sampled = sample_bbox_depth_stats(
                depth_map,
                item.get("box"),
                min_m=min_m,
                max_m=max_m,
            )
            if sampled is not None and frame_offset_s is not None:
                sampled["source_frame_offset_s"] = round(float(frame_offset_s), 3)
            stats.append(sampled)
        return stats

    def enrich_objects(
        self,
        frame: np.ndarray,
        objects: list[dict[str, Any]],
        *,
        frame_offset_s: float | None = None,
        include_heatmap: bool | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not objects:
            return objects, {}
        depth_map = self.estimate_depth_map(frame)
        min_m = float(self.depth_config.min_distance_m)
        max_m = float(self.depth_config.max_distance_m)
        max_incident_distance = self.depth_config.max_incident_distance_m
        enriched: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {
            "inference_ms": self._last_inference_ms,
        }
        for item in objects:
            if not isinstance(item, dict):
                continue
            next_item = dict(item)
            sampled = sample_bbox_depth_stats(
                depth_map,
                next_item.get("box"),
                min_m=min_m,
                max_m=max_m,
            )
            if sampled is not None:
                if frame_offset_s is not None:
                    sampled["source_frame_offset_s"] = round(float(frame_offset_s), 3)
                next_item["depth_stats"] = sampled
                median_m = float(sampled["median_m"])
                if (
                    max_incident_distance is not None
                    and median_m > float(max_incident_distance)
                    and next_item.get("incident_eligible") is not False
                ):
                    next_item["incident_eligible"] = False
                    next_item["depth_filtered"] = True
            enriched.append(next_item)
        should_store_heatmap = (
            self.depth_config.store_heatmap
            if include_heatmap is None
            else bool(include_heatmap)
        )
        if should_store_heatmap:
            heatmap = encode_depth_heatmap(
                depth_map,
                max_width=int(self.depth_config.heatmap_max_width),
            )
            if heatmap:
                metadata["heatmap_png"] = heatmap
                metadata["heatmap_range_m"] = {
                    "min_m": round(min_m, 2),
                    "max_m": round(max_m, 2),
                }
        return enriched, metadata

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "error": self.error,
            "configured_device": self.depth_config.device,
            "loaded_device": self.loaded_device,
            "model_path": self.depth_config.resolved_model_path(),
            "input_shape": list(self.input_shape),
            "model_load_ms": self.model_load_ms,
            "last_inference_ms": self._last_inference_ms,
            "min_distance_m": self.depth_config.min_distance_m,
            "max_distance_m": self.depth_config.max_distance_m,
            "max_incident_distance_m": self.depth_config.max_incident_distance_m,
            "store_heatmap": self.depth_config.store_heatmap,
        }

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np

from .config import DetectorConfig

LOGGER = logging.getLogger(__name__)


class OpenVinoDetector:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.labels = self._load_labels(config)
        self.compiled_model: Any = None
        self.cv_net: Any = None
        self.coreml_model: Any = None
        self.coreml_input_name = ""
        self.coreml_image_input = False
        self.input_layer: Any = None
        self.output_layer: Any = None
        self.output_layers: list[Any] = []
        self.input_shape: tuple[int, int] = (300, 300)
        self.output_format = "unknown"
        self.backend = ""
        self.loaded_device = ""
        self._stats_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._pending_requests = 0
        self._active_inferences = 0
        self._durations_ms: deque[float] = deque(maxlen=100)
        self._completion_times: deque[float] = deque(maxlen=240)
        self._last_inference_ms: float | None = None
        self._last_inference_at = ""
        self._last_detection_at = ""
        self._last_detection_epoch: float | None = None
        self._last_detection_labels: list[str] = []
        self._total_inferences = 0
        self._failed_inferences = 0
        self._object_hit_inferences = 0

        openvino_model_path_text = config.resolved_model_path()
        coreml_model_path_text = config.resolved_coreml_model_path()
        backend_preference = config.backend.lower()
        self.enabled = config.enabled and bool(coreml_model_path_text or openvino_model_path_text)
        if not self.enabled:
            return

        if backend_preference == "coreml":
            if self._load_coreml(coreml_model_path_text):
                return
            LOGGER.warning("Core ML detector unavailable, falling back to OpenVINO/OpenCV")

        if not self._load_openvino(openvino_model_path_text):
            self.enabled = False
            return

    def _load_openvino(self, model_path_text: str) -> bool:
        if not model_path_text:
            return False

        model_path = Path(model_path_text)
        if not model_path.exists():
            return False

        try:
            try:
                from openvino import Core
            except ImportError:
                from openvino.runtime import Core

            core = Core()
            model = core.read_model(model=model_path)
            try:
                self.compiled_model = core.compile_model(model=model, device_name=self.config.device)
            except Exception:
                if self.config.device.upper() == "CPU":
                    raise
                LOGGER.warning("OpenVINO failed on %s, retrying on CPU", self.config.device)
                self.compiled_model = core.compile_model(model=model, device_name="CPU")
                self.loaded_device = "CPU"
            if not self.loaded_device:
                self.loaded_device = self.config.device
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            self.output_layers = list(self.compiled_model.outputs)
            _, _, height, width = self.input_layer.shape
            self.input_shape = (int(width), int(height))
            self.output_format = self._detect_output_format()
            self.backend = "openvino"
            return True
        except Exception as exc:
            if model_path.suffix.lower() != ".onnx":
                LOGGER.exception("OpenVINO detector failed to load %s", model_path)
                return False
            LOGGER.warning("OpenVINO failed to load %s, falling back to OpenCV DNN: %s", model_path, exc)
            self.cv_net = cv2.dnn.readNetFromONNX(str(model_path))
            self.input_shape = (640, 640)
            self.output_format = "yolo"
            self.backend = "opencv-dnn"
            self.loaded_device = "CPU"
            return True

    def _load_coreml(self, model_path_text: str) -> bool:
        if not model_path_text:
            return False

        model_path = Path(model_path_text)
        if not model_path.exists():
            LOGGER.warning("Core ML model path does not exist: %s", model_path)
            return False

        try:
            import coremltools as ct

            self.coreml_model = ct.models.MLModel(
                str(model_path),
                compute_units=ct.ComputeUnit.CPU_ONLY,
            )
            spec = self.coreml_model.get_spec()
            if not spec.description.input:
                LOGGER.warning("Core ML model has no input description: %s", model_path)
                self.coreml_model = None
                return False

            input_description = spec.description.input[0]
            self.coreml_input_name = input_description.name
            input_type = input_description.type.WhichOneof("Type")
            self.coreml_image_input = input_type == "imageType"
            if self.coreml_image_input:
                width = int(input_description.type.imageType.width or 640)
                height = int(input_description.type.imageType.height or 640)
                self.input_shape = (width, height)
            elif input_type == "multiArrayType":
                shape = [int(value) for value in input_description.type.multiArrayType.shape]
                if len(shape) >= 4:
                    self.input_shape = (shape[-1], shape[-2])
                else:
                    self.input_shape = (640, 640)
            self.output_format = "coreml"
            self.backend = "coreml"
            return True
        except Exception:
            LOGGER.exception("Core ML detector failed to load %s", model_path)
            self.coreml_model = None
            return False

    def detect(self, frame: np.ndarray, confidence_threshold: float | None = None) -> list[dict[str, Any]]:
        with self._inference_lock:
            return self._detect_locked(frame, confidence_threshold=confidence_threshold)

    def _detect_locked(self, frame: np.ndarray, confidence_threshold: float | None = None) -> list[dict[str, Any]]:
        if not self.enabled or (
            self.compiled_model is None and self.cv_net is None and self.coreml_model is None
        ):
            return [{"status": "detector_unavailable"}]

        original_threshold = self.config.confidence_threshold
        if confidence_threshold is not None:
            self.config.confidence_threshold = max(0.01, min(0.99, float(confidence_threshold)))

        self._begin_inference()
        started = time.perf_counter()
        objects: list[dict[str, Any]] = []
        try:
            if self.coreml_model is not None:
                objects = self._detect_coreml(frame)
                return objects

            tensor, metadata = self._preprocess(frame)
            if self.compiled_model is not None:
                inference = self.compiled_model([tensor])
                if self.output_format == "yolo-seg":
                    outputs = [np.asarray(inference[layer]) for layer in self.output_layers]
                    objects = self._parse_yolo_seg_outputs(outputs, metadata)
                    return objects
                output = inference[self.output_layer]
            else:
                self.cv_net.setInput(tensor)
                output = self.cv_net.forward()
            if self.output_format == "yolo":
                objects = self._parse_yolo_output(output, metadata)
            else:
                objects = self._parse_ssd_output(output, frame.shape[1], frame.shape[0])
            return objects
        finally:
            self.config.confidence_threshold = original_threshold
            self._finish_inference((time.perf_counter() - started) * 1000, objects)

    def _begin_inference(self) -> None:
        with self._stats_lock:
            self._pending_requests += 1
            self._active_inferences += 1

    def _finish_inference(self, duration_ms: float, objects: list[dict[str, Any]]) -> None:
        now_monotonic = time.monotonic()
        now_epoch = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        labels = sorted({str(item.get("label")) for item in objects if item.get("label")})
        failed = any(item.get("status") == "detector_unavailable" for item in objects)
        with self._stats_lock:
            self._pending_requests = max(0, self._pending_requests - 1)
            self._active_inferences = max(0, self._active_inferences - 1)
            self._total_inferences += 1
            self._last_inference_ms = duration_ms
            self._last_inference_at = now_iso
            self._durations_ms.append(duration_ms)
            self._completion_times.append(now_monotonic)
            while self._completion_times and now_monotonic - self._completion_times[0] > 60:
                self._completion_times.popleft()
            if failed:
                self._failed_inferences += 1
            if labels:
                self._object_hit_inferences += 1
                self._last_detection_at = now_iso
                self._last_detection_epoch = now_epoch
                self._last_detection_labels = labels

    def _runtime_stats(self) -> dict[str, Any]:
        now_monotonic = time.monotonic()
        now_epoch = time.time()
        with self._stats_lock:
            while self._completion_times and now_monotonic - self._completion_times[0] > 60:
                self._completion_times.popleft()
            average_ms = (sum(self._durations_ms) / len(self._durations_ms)) if self._durations_ms else None
            if len(self._completion_times) >= 2:
                span = max(1.0, self._completion_times[-1] - self._completion_times[0])
                detections_per_second = len(self._completion_times) / span
            else:
                detections_per_second = float(len(self._completion_times)) if self._completion_times else 0.0
            last_detection_age = (now_epoch - self._last_detection_epoch) if self._last_detection_epoch else None
            return {
                "last_inference_ms": round(self._last_inference_ms, 1) if self._last_inference_ms is not None else None,
                "average_inference_ms": round(average_ms, 1) if average_ms is not None else None,
                "detection_fps": round(detections_per_second, 2),
                "queue_depth": self._pending_requests,
                "pending_frames": self._pending_requests,
                "active_inferences": self._active_inferences,
                "last_inference_at": self._last_inference_at,
                "last_detection_at": self._last_detection_at,
                "last_detection_age_seconds": round(last_detection_age, 1) if last_detection_age is not None else None,
                "last_detection_labels": list(self._last_detection_labels),
                "total_inferences": self._total_inferences,
                "failed_inferences": self._failed_inferences,
                "object_hit_inferences": self._object_hit_inferences,
            }

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "configured_backend": self.config.backend,
            "loaded_backend": self.backend or "",
            "configured_device": self.config.device,
            "loaded_device": self.loaded_device,
            "input_shape": list(self.input_shape),
            "output_format": self.output_format,
            "labels": len(self.labels),
            "coreml_loaded": self.coreml_model is not None,
            "coreml_input_name": self.coreml_input_name,
            "coreml_image_input": self.coreml_image_input,
            "openvino_loaded": self.compiled_model is not None,
            "opencv_loaded": self.cv_net is not None,
            "runtime": self._runtime_stats(),
        }

    def _load_labels(self, config: DetectorConfig) -> list[str]:
        labels = list(config.labels)
        if config.labels_path:
            labels_path = Path(config.labels_path)
            if labels_path.exists():
                labels = [
                    line.strip()
                    for line in labels_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        if not labels:
            model_path_text = config.resolved_model_path()
            model_path = Path(model_path_text) if model_path_text else None
            metadata_path = model_path.parent / "metadata.yaml" if model_path else None
            if metadata_path and metadata_path.exists():
                try:
                    import yaml

                    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
                    names = metadata.get("names") or {}
                    if isinstance(names, dict):
                        labels = [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
                    elif isinstance(names, list):
                        labels = [str(value) for value in names]
                except Exception:
                    LOGGER.exception("Failed to load detector labels from %s", metadata_path)
        return labels

    def _detect_coreml(self, frame: np.ndarray) -> list[dict[str, Any]]:
        try:
            if self.coreml_image_input:
                image, metadata = self._preprocess_coreml_image(frame)
                outputs = self.coreml_model.predict({self.coreml_input_name: image})
                return self._parse_coreml_outputs(outputs, frame.shape[1], frame.shape[0], metadata)

            tensor, metadata = self._preprocess(frame)
            outputs = self.coreml_model.predict({self.coreml_input_name: tensor})
            return self._parse_coreml_outputs(outputs, frame.shape[1], frame.shape[0], metadata)
        except Exception:
            LOGGER.exception("Core ML detection failed")
            return [{"status": "detector_unavailable"}]

    def _parse_coreml_outputs(
        self,
        outputs: dict[str, Any],
        image_width: int,
        image_height: int,
        metadata: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        lower_keys = {key.lower(): key for key in outputs}
        coordinates_key = lower_keys.get("coordinates")
        confidence_key = lower_keys.get("confidence")
        if coordinates_key and confidence_key:
            return self._parse_coreml_coordinates(
                np.asarray(outputs[coordinates_key]),
                np.asarray(outputs[confidence_key]),
                image_width,
                image_height,
            )

        for value in outputs.values():
            array = np.asarray(value)
            squeezed = np.squeeze(array)
            if squeezed.ndim == 2 and min(squeezed.shape) >= 6:
                if metadata is not None and (
                    squeezed.shape[0] == len(self.labels) + 4
                    or squeezed.shape[1] == len(self.labels) + 4
                    or max(squeezed.shape) > 100
                ):
                    return self._parse_yolo_output(array, metadata)
                return self._parse_ssd_output(array, image_width, image_height)
            if squeezed.ndim == 3 and min(squeezed.shape[-2:]) >= 6 and metadata is not None:
                return self._parse_yolo_output(array, metadata)

        return []

    def _preprocess_coreml_image(self, frame: np.ndarray) -> tuple[Any, dict[str, float]]:
        from PIL import Image

        input_width, input_height = self.input_shape
        image_height, image_width = frame.shape[:2]
        scale = min(input_width / image_width, input_height / image_height)
        resized_width = int(round(image_width * scale))
        resized_height = int(round(image_height * scale))
        pad_x = (input_width - resized_width) / 2
        pad_y = (input_height - resized_height) / 2

        resized = cv2.resize(frame, (resized_width, resized_height))
        canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))
        canvas[top : top + resized_height, left : left + resized_width] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb), {
            "scale": scale,
            "pad_x": float(left),
            "pad_y": float(top),
            "image_width": float(image_width),
            "image_height": float(image_height),
        }

    def _parse_coreml_coordinates(
        self,
        coordinates: np.ndarray,
        confidence: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> list[dict[str, Any]]:
        coordinates = np.squeeze(coordinates)
        confidence = np.squeeze(confidence)
        if coordinates.ndim == 1:
            coordinates = np.array([coordinates])
        if confidence.ndim == 1:
            confidence = np.array([confidence])

        objects: list[dict[str, Any]] = []
        for index, box_values in enumerate(coordinates):
            if len(box_values) < 4 or index >= len(confidence):
                continue
            scores = np.asarray(confidence[index])
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])
            if score < self.config.confidence_threshold:
                continue

            x, y, width, height = [float(value) for value in box_values[:4]]
            if max(abs(x), abs(y), abs(width), abs(height)) <= 1.5:
                x *= image_width
                width *= image_width
                y *= image_height
                height *= image_height

            x1 = max(0, min(image_width, x - width / 2))
            y1 = max(0, min(image_height, y - height / 2))
            x2 = max(0, min(image_width, x + width / 2))
            y2 = max(0, min(image_height, y + height / 2))
            label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
            objects.append(
                {
                    "label": label,
                    "confidence": round(score, 4),
                    "box": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    },
                }
            )
        return objects

    def _detect_output_format(self) -> str:
        shapes = [[int(dim) for dim in layer.shape if int(dim) > 0] for layer in self.output_layers]
        if any(len(shape) == 4 and shape[1] == 32 for shape in shapes) and any(
            len(shape) == 3 and shape[-1] >= 38 for shape in shapes
        ):
            return "yolo-seg"
        shape = [int(dim) for dim in self.output_layer.shape if int(dim) > 0]
        if len(shape) == 3:
            channels = min(shape[1], shape[2])
            if channels >= 6 and channels == len(self.labels) + 4:
                return "yolo"
        return "ssd"

    def _parse_yolo_seg_outputs(
        self,
        outputs: list[np.ndarray],
        metadata: dict[str, float],
    ) -> list[dict[str, Any]]:
        detections_output = next((value for value in outputs if value.ndim == 3), None)
        prototypes_output = next((value for value in outputs if value.ndim == 4 and value.shape[1] == 32), None)
        if detections_output is None:
            return []
        detections = np.squeeze(detections_output, axis=0)
        prototypes = np.squeeze(prototypes_output, axis=0) if prototypes_output is not None else None
        image_width = int(metadata["image_width"])
        image_height = int(metadata["image_height"])
        scale = metadata["scale"]
        pad_x = metadata["pad_x"]
        pad_y = metadata["pad_y"]
        objects: list[dict[str, Any]] = []

        for detection in detections:
            if len(detection) < 6:
                continue
            confidence = float(detection[4])
            if confidence < self.config.confidence_threshold:
                continue
            class_id = int(detection[5])
            input_x1, input_y1, input_x2, input_y2 = [float(value) for value in detection[:4]]
            x1 = max(0, min(image_width, (input_x1 - pad_x) / scale))
            y1 = max(0, min(image_height, (input_y1 - pad_y) / scale))
            x2 = max(0, min(image_width, (input_x2 - pad_x) / scale))
            y2 = max(0, min(image_height, (input_y2 - pad_y) / scale))
            if x2 <= x1 or y2 <= y1:
                continue
            item: dict[str, Any] = {
                "label": self.labels[class_id] if class_id < len(self.labels) else str(class_id),
                "confidence": round(confidence, 4),
                "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            }
            if prototypes is not None and len(detection) >= 6 + prototypes.shape[0]:
                polygon = self._segmentation_polygon(
                    detection[6 : 6 + prototypes.shape[0]],
                    prototypes,
                    metadata,
                    (input_x1, input_y1, input_x2, input_y2),
                )
                if polygon:
                    item["mask_polygon"] = polygon
            objects.append(item)
        return objects

    def _segmentation_polygon(
        self,
        coefficients: np.ndarray,
        prototypes: np.ndarray,
        metadata: dict[str, float],
        input_box: tuple[float, float, float, float],
    ) -> list[list[int]]:
        mask_height, mask_width = prototypes.shape[1:]
        logits = np.asarray(coefficients, dtype=np.float32) @ prototypes.reshape(prototypes.shape[0], -1)
        mask = (1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))).reshape(mask_height, mask_width)
        input_width, input_height = self.input_shape
        x1, y1, x2, y2 = input_box
        mx1 = max(0, min(mask_width, int(x1 * mask_width / input_width)))
        my1 = max(0, min(mask_height, int(y1 * mask_height / input_height)))
        mx2 = max(0, min(mask_width, int(np.ceil(x2 * mask_width / input_width))))
        my2 = max(0, min(mask_height, int(np.ceil(y2 * mask_height / input_height))))
        cropped = np.zeros_like(mask, dtype=np.uint8)
        if mx2 <= mx1 or my2 <= my1:
            return []
        cropped[my1:my2, mx1:mx2] = (mask[my1:my2, mx1:mx2] >= 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 2:
            return []
        contour = cv2.approxPolyDP(contour, max(1.0, 0.01 * cv2.arcLength(contour, True)), True)
        scale = metadata["scale"]
        pad_x = metadata["pad_x"]
        pad_y = metadata["pad_y"]
        image_width = int(metadata["image_width"])
        image_height = int(metadata["image_height"])
        polygon: list[list[int]] = []
        for point in contour.reshape(-1, 2):
            input_x = float(point[0]) * input_width / mask_width
            input_y = float(point[1]) * input_height / mask_height
            image_x = int(max(0, min(image_width, (input_x - pad_x) / scale)))
            image_y = int(max(0, min(image_height, (input_y - pad_y) / scale)))
            polygon.append([image_x, image_y])
        return polygon if len(polygon) >= 3 else []

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        input_width, input_height = self.input_shape
        image_height, image_width = frame.shape[:2]
        scale = min(input_width / image_width, input_height / image_height)
        resized_width = int(round(image_width * scale))
        resized_height = int(round(image_height * scale))
        pad_x = (input_width - resized_width) / 2
        pad_y = (input_height - resized_height) / 2

        resized = cv2.resize(frame, (resized_width, resized_height))
        canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))
        canvas[top : top + resized_height, left : left + resized_width] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32) / 255.0
        return tensor, {
            "scale": scale,
            "pad_x": float(left),
            "pad_y": float(top),
            "image_width": float(image_width),
            "image_height": float(image_height),
        }

    def _parse_ssd_output(
        self,
        output: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> list[dict[str, Any]]:
        detections = np.squeeze(output)
        objects: list[dict[str, Any]] = []
        if detections.ndim == 1:
            detections = np.array([detections])

        for detection in detections:
            if len(detection) < 7:
                continue
            confidence = float(detection[2])
            if confidence < self.config.confidence_threshold:
                continue
            label_id = int(detection[1])
            label = (
                self.labels[label_id]
                if label_id < len(self.labels)
                else str(label_id)
            )
            objects.append(
                {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "box": {
                        "x1": int(float(detection[3]) * image_width),
                        "y1": int(float(detection[4]) * image_height),
                        "x2": int(float(detection[5]) * image_width),
                        "y2": int(float(detection[6]) * image_height),
                    },
                }
            )
        return objects

    def _parse_yolo_output(
        self,
        output: np.ndarray,
        metadata: dict[str, float],
    ) -> list[dict[str, Any]]:
        detections = np.squeeze(output)
        if detections.ndim != 2:
            return []
        if detections.shape[0] == len(self.labels) + 4:
            detections = detections.T

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        image_width = metadata["image_width"]
        image_height = metadata["image_height"]
        scale = metadata["scale"]
        pad_x = metadata["pad_x"]
        pad_y = metadata["pad_y"]

        for detection in detections:
            if len(detection) < 6:
                continue
            class_scores = detection[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < self.config.confidence_threshold:
                continue

            center_x, center_y, width, height = [float(value) for value in detection[:4]]
            x1 = (center_x - width / 2 - pad_x) / scale
            y1 = (center_y - height / 2 - pad_y) / scale
            x2 = (center_x + width / 2 - pad_x) / scale
            y2 = (center_y + height / 2 - pad_y) / scale
            x1 = max(0, min(image_width, x1))
            y1 = max(0, min(image_height, y1))
            x2 = max(0, min(image_width, x2))
            y2 = max(0, min(image_height, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        indexes = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.config.confidence_threshold,
            self.config.nms_threshold,
        )
        selected = np.array(indexes).flatten().tolist() if len(indexes) else []
        objects: list[dict[str, Any]] = []
        for index in selected:
            x, y, width, height = boxes[index]
            class_id = class_ids[index]
            label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
            objects.append(
                {
                    "label": label,
                    "confidence": round(scores[index], 4),
                    "box": {
                        "x1": x,
                        "y1": y,
                        "x2": x + width,
                        "y2": y + height,
                    },
                }
            )
        return objects


def objects_to_json(objects: list[dict[str, Any]]) -> str:
    return json.dumps(objects, separators=(",", ":"))

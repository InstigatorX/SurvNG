"""Application services around native per-camera inference.

No Python inference supervisor, tracking factory, ReID, depth, or backfill
worker is constructed. Historical identity/search records remain readable.
"""
from __future__ import annotations

from .face_store import FaceStore
from .inference_runtime.process import load_detector_labels
from .inference_runtime.types import InferenceUnavailable
from .semantic_search import DisabledSemanticSearch
from survng.native_deepsort import resolve_native_tracking


class UnavailableEnrichment:
    enabled = False
    ready = False
    model_fingerprint = ""

    def __init__(self, config):
        self.config = config

    def status(self):
        return {"enabled": False, "ready": False, "state": "removed", "reason": "native-first runtime"}

    def supports_label(self, label):
        return False

    def enqueue(self, *args, **kwargs):
        return False

    def embed(self, image):
        raise InferenceUnavailable("appearance inference is not part of the native runtime")


class NativeDetectorStatus:
    """Configuration/status endpoint; deliberately has no detect method."""
    def __init__(self, config, workers):
        self.config = config
        self.labels = load_detector_labels(config)
        self.enabled = config.enabled
        self._workers = workers

    def inspect_model(self, path):
        """Resolve effective native model settings without claiming full validation."""
        import json
        import xml.etree.ElementTree as ET
        from pathlib import Path

        from .detector_labels import openvino_package_classes
        from .detector_model_settings import read_model_metadata, resolve_output_format
        from .dlstreamer_capture import adjacent_model_proc

        model_path = Path(path)
        result = {
            "path": str(model_path),
            "validated": False,
            "warnings": [],
            "error": "",
            "input_shape": [],
            "output_shapes": [],
            "labels": [],
            "labels_source": "",
            "model_proc_path": "",
            "model_proc_match": "",
            "nms_threshold": self.config.nms_threshold,
            "nms_source": "detector.nms_threshold",
            "output_format": "",
            "output_format_source": "",
            "metadata": {},
        }
        try:
            root = ET.parse(model_path).getroot()
        except (OSError, ET.ParseError, ValueError) as error:
            result["error"] = type(error).__name__
            return result

        inputs, outputs = [], []
        for layer in root.findall("./layers/layer"):
            kind = layer.get("type", "").lower()
            if kind in {"input", "parameter"}:
                inputs.append([int(dim.text) for dim in layer.findall("./output/port/dim")])
            elif kind == "result":
                outputs.append([int(dim.text) for dim in layer.findall("./input/port/dim")])
        result["input_shape"] = inputs[0] if inputs else []
        result["output_shapes"] = outputs

        metadata, metadata_warnings = read_model_metadata(model_path)
        result["metadata"] = {
            key: metadata[key]
            for key in ("model_type", "task", "nms", "end2end")
            if key in metadata
        }
        result["warnings"].extend(metadata_warnings)

        classes, task, class_error = openvino_package_classes(model_path)
        if class_error:
            result["warnings"].append(class_error)
        if task:
            result["metadata"].setdefault("task", task)
        configured = list(self.labels or ())
        if configured:
            result["labels"] = configured
            result["labels_source"] = "detector.labels"
        elif classes:
            result["labels"] = list(classes)
            result["labels_source"] = "package_metadata"
        else:
            result["labels_source"] = "unavailable"
            result["warnings"].append("No labels resolved from config or package metadata.")

        model_proc = adjacent_model_proc(str(model_path))
        result["model_proc_path"] = model_proc
        if model_proc:
            proc_name = Path(model_proc).name
            stem = model_path.stem
            if proc_name in {f"{stem}.json", f"{stem}_proc.json"}:
                result["model_proc_match"] = "basename"
            elif proc_name == "model-proc.json":
                result["model_proc_match"] = "directory_default"
                result["warnings"].append(
                    "Using directory model-proc.json; confirm it matches this IR export."
                )
            else:
                result["model_proc_match"] = "adjacent"
            try:
                payload = json.loads(Path(model_proc).read_text(encoding="utf-8"))
                outputs = payload.get("output_postproc")
                if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
                    converter = str(outputs[0].get("converter") or "")
                    if converter:
                        result["output_format"] = converter
                        result["output_format_source"] = "model-proc"
                    if "iou_threshold" in outputs[0]:
                        result["nms_threshold"] = outputs[0].get("iou_threshold")
                        result["nms_source"] = "model-proc.iou_threshold"
            except (OSError, ValueError, TypeError) as error:
                result["warnings"].append(f"model-proc unreadable ({type(error).__name__})")
        else:
            result["model_proc_match"] = "none"
            result["warnings"].append(
                "No adjacent model-proc found; DL Streamer will use IR model_info when present."
            )

        if not result["output_format"]:
            try:
                output_format, source, format_warnings = resolve_output_format(
                    outputs,
                    "auto",
                    metadata,
                    has_nms=False,
                )
                result["output_format"] = output_format
                result["output_format_source"] = source
                result["warnings"].extend(format_warnings)
            except Exception as error:
                result["warnings"].append(
                    f"output format unresolved ({type(error).__name__})"
                )

        result["validated"] = False
        if not result["error"] and result["input_shape"] and result["output_shapes"]:
            # Shapes and discovered sidecars are reported; DL Streamer still owns
            # runtime preprocessing/decoding. Do not claim full validation.
            result["inspection_complete"] = True
        else:
            result["inspection_complete"] = False
        return result

    def status(self):
        cameras = {key: worker.status().get("native_activity", {}) for key, worker in self._workers().items()}
        expected = [key for key, worker in self._workers().items()
                    if worker.runtime_state.enabled and worker.runtime_state.detection_enabled]
        healthy = sum(cameras[key].get("health") == "healthy" for key in expected)
        model_path = self.config.resolved_model_path()
        model_inspection = self.inspect_model(model_path) if model_path else {
            "error": "no_model_path",
            "validated": False,
            "inspection_complete": False,
        }
        return {"enabled": self.enabled, "ready": bool(healthy), "implementation": "dlstreamer",
                "configured_device": self.config.device, "device": self.config.device,
                "labels": self.labels, "native": True, "sample_fps": self.config.live_sample_fps,
                "batch_size": self.config.native.batch_size,
                "tracking_classes": self.config.native.tracking_classes,
                "inference_interval": self.config.native.inference_interval,
                "effective_inference_fps": self.config.live_sample_fps / self.config.native.inference_interval, "tracking": resolve_native_tracking(self.config).mode,
                "native_cameras": cameras, "healthy_cameras": healthy, "active_cameras": len(expected),
                "degraded": bool(expected and healthy < len(expected)),
                "python_inference_workers": 0, "model_path": model_path,
                "model_inspection": model_inspection,
                "model_validated": bool(model_inspection.get("validated")),
                }

    def probe_devices(self):
        import json
        import subprocess
        from .dlstreamer_capture import live_python_executable
        try:
            result = subprocess.run([live_python_executable(), "-c",
                "import json,openvino; print(json.dumps(openvino.Core().available_devices))"],
                capture_output=True, text=True, timeout=10, check=True)
            return {"devices": json.loads(result.stdout), "error": ""}
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return {"devices": [], "error": type(error).__name__}

    def cached_object_status(self):
        return self.status()


class NativeRuntime:
    def __init__(self, *, config, semantic_config, storage_dir, events, appearance_index,
                 semantic_index, event_publisher, database_dir,
                 media_storage=None, database_write_lock=None):
        self._workers = {}
        self._started = False
        self._closed = False
        self.detector = NativeDetectorStatus(config, lambda: self._workers)
        self.face_recognizer = UnavailableEnrichment(config)
        self.person_reidentifier = UnavailableEnrichment(config.native.tracking)
        self.appearance_backfill = UnavailableEnrichment(config.native.tracking)
        self.faces = FaceStore(storage_dir, config.face_max_observations, self.face_recognizer,
                               start_recognition=False, database_dir=database_dir,
                               media_storage=media_storage, database_write_lock=database_write_lock)
        self.semantic_search = DisabledSemanticSearch(semantic_config, semantic_index)
        self.tracking_limiter = self

    def bind_workers(self, workers):
        self._workers = dict(workers)

    def replace_worker(self, camera_id, worker):
        self._workers[camera_id] = worker

    def start_core(self):
        if self._closed:
            raise RuntimeError("native runtime is closed")
        self._started = True

    def start_auxiliary(self):
        return None

    def maintain(self):
        return None

    def close(self):
        self.faces.close()
        self.semantic_search.close()
        self._closed = True
        self._started = False

    def status(self):
        return {"implementation": "native", "core_started": self._started,
                "closed": self._closed, "bound_cameras": len(self._workers),
                "active": sum(worker.runtime_state.enabled and worker.runtime_state.detection_enabled and worker.config.enabled for worker in self._workers.values()),
                "capacity": len(self._workers), "python_inference_workers": 0}

    def reconfigure_policy(self, config):
        self.detector.config = config
        for worker in self._workers.values():
            worker.reconfigure_policy(config)

    def reconfigure_tracking(self, config):
        raise ValueError("gvatrack is the sole tracker; legacy tracking settings are unavailable")

    def reconfigure_roles(self, config, roles, **kwargs):
        raise ValueError("native model/device changes require a capture reload")

    def reconfigure_semantic_search(self, config):
        # Semantic inference remains disabled, but configuration can be stored
        # without manufacturing a retired inference-role restart.
        self.semantic_search.config = config.model_copy(deep=True)

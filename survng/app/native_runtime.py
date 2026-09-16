"""Application services around native per-camera inference.

No Python inference supervisor, tracking factory, ReID, depth, or backfill
worker is constructed. Historical identity/search records remain readable.
"""
from __future__ import annotations

from .face_store import FaceStore
from .inference_runtime.process import load_detector_labels
from .inference_runtime.types import InferenceUnavailable
from .semantic_search import DisabledSemanticSearch


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

    def status(self):
        cameras = {key: worker.status().get("native_activity", {}) for key, worker in self._workers().items()}
        expected = [key for key, worker in self._workers().items()
                    if worker.runtime_state.enabled and worker.runtime_state.detection_enabled]
        healthy = sum(cameras[key].get("health") == "healthy" for key in expected)
        return {"enabled": self.enabled, "ready": bool(healthy), "implementation": "dlstreamer",
                "configured_device": self.config.device, "device": self.config.device,
                "labels": self.labels, "native": True, "sample_fps": self.config.live_sample_fps,
                "inference_interval": self.config.native.inference_interval,
                "effective_inference_fps": self.config.live_sample_fps / self.config.native.inference_interval, "tracking": "short-term-imageless",
                "native_cameras": cameras, "healthy_cameras": healthy, "active_cameras": len(expected),
                "degraded": bool(expected and healthy < len(expected)),
                "python_inference_workers": 0, "model_path": self.config.resolved_model_path()}

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

    def inspect_model(self, path):
        import xml.etree.ElementTree as ET
        try:
            root = ET.parse(path).getroot()
            inputs, outputs = [], []
            for layer in root.findall("./layers/layer"):
                kind = layer.get("type", "").lower()
                if kind in {"input", "parameter"}:
                    inputs.append([int(dim.text) for dim in layer.findall("./output/port/dim")])
                elif kind == "result":
                    outputs.append([int(dim.text) for dim in layer.findall("./input/port/dim")])
            return {"input_shape": inputs[0] if inputs else [], "output_shapes": outputs, "error": ""}
        except (OSError, ET.ParseError, ValueError) as error:
            return {"error": type(error).__name__}

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
        self.person_reidentifier = UnavailableEnrichment(config.tracking)
        self.appearance_backfill = UnavailableEnrichment(config.tracking)
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
        raise ValueError("semantic inference is unavailable in the native-first runtime")

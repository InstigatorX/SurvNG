from __future__ import annotations

import math
import multiprocessing
import os
from pathlib import Path
import resource
import threading
from typing import Any

import numpy as np

from ..config import DetectorConfig
from .types import LOGGER, RESOURCE_TRACKER_STOP_TIMEOUT_SECONDS


def stop_multiprocessing_resource_tracker(
    timeout: float = RESOURCE_TRACKER_STOP_TIMEOUT_SECONDS,
) -> bool:
    """Reap Python's tracker after all SurvNG multiprocessing users stop."""
    try:
        from multiprocessing import resource_tracker

        tracker = resource_tracker._resource_tracker
        pid = getattr(tracker, "_pid", None)
        if pid is None:
            return True
        failure: list[BaseException] = []

        def stop_tracker() -> None:
            try:
                tracker._stop()
            except BaseException as exc:
                failure.append(exc)

        thread = threading.Thread(
            target=stop_tracker,
            name="stop-multiprocessing-resource-tracker",
            daemon=True,
        )
        thread.start()
        thread.join(timeout=max(0.0, float(timeout)))
        if thread.is_alive():
            LOGGER.warning(
                "multiprocessing resource tracker pid=%s did not stop within %.1fs; systemd will clean it up",
                pid,
                timeout,
            )
            return False
        if failure:
            LOGGER.warning(
                "multiprocessing resource tracker pid=%s cleanup failed: %s",
                pid,
                failure[0],
            )
            return False
        LOGGER.info("multiprocessing resource tracker stopped pid=%s", pid)
        return True
    except (AttributeError, ImportError, RuntimeError) as exc:
        LOGGER.warning("multiprocessing resource tracker cleanup unavailable: %s", exc)
        return False

def load_detector_labels(config: DetectorConfig) -> list[str]:
    labels = list(config.labels)
    if config.labels_path:
        labels_path = Path(config.labels_path)
        if labels_path.exists():
            labels = [
                line.strip()
                for line in labels_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    if labels:
        return labels

    model_path_text = config.resolved_model_path()
    model_path = Path(model_path_text) if model_path_text else None
    metadata_path = model_path.parent / "metadata.yaml" if model_path else None
    if metadata_path is None or not metadata_path.exists():
        return []
    try:
        import yaml

        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        names = metadata.get("names") or {}
        if isinstance(names, dict):
            return [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
        if isinstance(names, list):
            return [str(value) for value in names]
    except Exception:
        LOGGER.exception("Failed to load detector labels from %s", metadata_path)
    return []

def _disable_worker_core_dumps() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        pass

def _set_worker_process_name(role: str) -> str:
    name = f"survng-{role}"[:15]
    multiprocessing.current_process().name = name
    try:
        Path("/proc/self/comm").write_text(name, encoding="utf-8")
    except OSError:
        pass
    return name

def _openvino_devices() -> dict[str, Any]:
    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core

        return {"devices": list(Core().available_devices), "error": ""}
    except Exception as exc:
        return {"devices": [], "error": str(exc) or "OpenVINO device probe failed"}

def _inspect_openvino_model(path_text: str) -> dict[str, Any]:
    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core

        model = Core().read_model(model=path_text)
        return {
            "input_shape": [int(value) for value in model.input(0).shape],
            "output_shapes": [[int(value) for value in output.shape] for output in model.outputs],
            "error": "",
        }
    except Exception as exc:
        return {"input_shape": [], "output_shapes": [], "error": str(exc)}

def _inference_worker_main(
    connection,
    frame_buffer,
    config_payload: dict[str, Any],
    role: str,
) -> None:
    _set_worker_process_name(role)
    _disable_worker_core_dumps()
    config = DetectorConfig.model_validate(config_payload)
    try:
        if role == "object":
            from ..detector import OpenVinoDetector

            engine = OpenVinoDetector(config)
        elif role == "face":
            from ..face_detection import OpenVinoFaceDetector
            from ..face_recognition import OpenVinoFaceRecognizer

            engine = OpenVinoFaceRecognizer(config)
            face_detector = OpenVinoFaceDetector(config)
        elif role == "reid":
            from ..person_reidentification import OpenVinoAppearanceReidentifier

            engine = OpenVinoAppearanceReidentifier(config)
        else:
            raise ValueError(f"unknown inference worker role: {role}")
        engine_status = engine.status()
        if role == "face":
            engine_status["detector"] = face_detector.status()
        connection.send({
            "type": "ready",
            "pid": os.getpid(),
            "role": role,
            "status": engine_status,
        })
    except BaseException as exc:
        try:
            connection.send({"type": "fatal", "error": str(exc) or type(exc).__name__})
        finally:
            connection.close()
        return

    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            request_id = request.get("id")
            operation = request.get("op")
            if operation == "shutdown":
                connection.send({"id": request_id, "ok": True, "result": {"stopped": True}})
                return
            try:
                if operation in {"detect", "detect_faces", "embed", "embed_person", "embed_reid"}:
                    shape = tuple(int(value) for value in request.get("shape") or ())
                    dtype_name = str(request.get("dtype") or "")
                    byte_count = int(request.get("byte_count") or 0)
                    if (
                        dtype_name != "uint8"
                        or len(shape) != 3
                        or shape[2] != 3
                        or any(value <= 0 for value in shape)
                        or byte_count != math.prod(shape)
                        or byte_count > len(frame_buffer)
                    ):
                        raise ValueError("invalid shared inference frame")
                    view = np.frombuffer(frame_buffer, dtype=np.uint8, count=byte_count)
                    frame = view.reshape(shape)
                    if operation == "detect" and role == "object":
                        result = engine.detect(
                            frame,
                            confidence_threshold=request.get("confidence_threshold"),
                        )
                    elif operation == "embed" and role == "face":
                        result = engine.embed(frame).astype(np.float32).tolist()
                    elif operation == "detect_faces" and role == "face":
                        result = face_detector.detect(
                            frame,
                            threshold=request.get("confidence_threshold"),
                        )
                    elif operation == "embed_person" and role == "reid":
                        result = engine.embed(frame).astype(np.float32).tolist()
                    elif operation == "embed_reid" and role == "reid":
                        result = engine.embed_for_label(
                            str(request.get("label") or ""),
                            frame,
                        ).astype(np.float32).tolist()
                    else:
                        raise ValueError(f"{operation} is unavailable in the {role} worker")
                elif operation == "status":
                    result = engine.status()
                    if role == "face":
                        result["detector"] = face_detector.status()
                elif operation == "probe_devices" and role == "object":
                    result = _openvino_devices()
                elif operation == "inspect_model" and role == "object":
                    result = _inspect_openvino_model(str(request.get("path") or ""))
                else:
                    raise ValueError(f"unknown inference operation: {operation}")
                connection.send({"id": request_id, "ok": True, "result": result})
            except BaseException as exc:
                connection.send({
                    "id": request_id,
                    "ok": False,
                    "error": str(exc) or type(exc).__name__,
                })
    finally:
        connection.close()

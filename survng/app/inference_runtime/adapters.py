from __future__ import annotations

from typing import Any

import numpy as np

from ..config import DetectorConfig
from .supervisor import InferenceSupervisor
from .types import InferenceUnavailable


class IsolatedFaceRecognizer:
    def __init__(self, supervisor: InferenceSupervisor) -> None:
        self.supervisor = supervisor

    @property
    def config(self) -> DetectorConfig:
        return self.supervisor.config

    @property
    def enabled(self) -> bool:
        return bool(self.config.face_recognition_enabled)

    @property
    def ready(self) -> bool:
        return bool(self.status().get("ready"))

    @property
    def model_fingerprint(self) -> str:
        return str(self.status().get("model_fingerprint") or "")

    def embed(self, face: np.ndarray) -> np.ndarray:
        return self.supervisor.embed(face)

    def status(self) -> dict[str, Any]:
        return self.supervisor.face_status()

class IsolatedPersonReidentifier:
    def __init__(self, supervisor: InferenceSupervisor) -> None:
        self.supervisor = supervisor
        self.config = supervisor.config.tracking

    @property
    def enabled(self) -> bool:
        return bool(self.config.appearance_reid_enabled)

    @property
    def ready(self) -> bool:
        return bool(self.status().get("ready"))

    def embed(self, person: np.ndarray) -> np.ndarray:
        return self.supervisor.embed_person(person)

    def supports_label(self, label: str) -> bool:
        normalized = str(label or "").strip().lower()
        if not self.config.reid_enabled_for_label(normalized):
            return False
        status = self.supervisor.cached_reid_status()
        if normalized == "person":
            engine = status.get("person", status)
        else:
            engine = status.get("vehicle", {})
        return bool(isinstance(engine, dict) and engine.get("ready"))

    def embed_for_label(self, label: str, crop: np.ndarray) -> np.ndarray:
        return self.supervisor.embed_reid(label, crop)

    def model_identity_for_label(self, label: str) -> dict[str, Any] | None:
        normalized = str(label or "").strip().lower()
        status = self.supervisor.cached_reid_status()
        if normalized == "person":
            engine = status.get("person", status)
            model_kind = "person"
        elif normalized in self.config.vehicle_reid_labels:
            engine = status.get("vehicle", {})
            model_kind = "vehicle"
        else:
            return None
        if not isinstance(engine, dict) or not engine.get("ready"):
            return None
        fingerprint = str(engine.get("model_fingerprint") or "")
        if not fingerprint:
            return None
        return {
            "model_kind": model_kind,
            "model_fingerprint": fingerprint,
            "embedding_size": int(engine.get("embedding_size") or 0),
            "match_threshold": self.config.reid_threshold_for_label(normalized),
        }

    def status(self) -> dict[str, Any]:
        return self.supervisor.reid_status()

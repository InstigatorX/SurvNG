"""Detector package class metadata shared by discovery and inference."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import DetectorConfig

LOGGER = logging.getLogger(__name__)


def openvino_package_classes(xml_path: Path) -> tuple[list[str], str, str]:
    """Load class names from metadata.yaml, then classes.txt beside the IR."""
    metadata_path = xml_path.parent / "metadata.yaml"
    classes: list[str] = []
    task = ""
    error = ""
    if metadata_path.exists():
        try:
            import yaml

            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            names = metadata.get("names") or {}
            if isinstance(names, dict):
                classes = [
                    str(value)
                    for _, value in sorted(names.items(), key=lambda entry: int(entry[0]))
                ]
            elif isinstance(names, list):
                classes = [str(value) for value in names]
            task = str(metadata.get("task") or "")
        except Exception as exc:
            error = f"Metadata: {exc}"
    if not classes:
        labels_path = xml_path.parent / "classes.txt"
        if labels_path.exists():
            try:
                classes = [
                    line.strip()
                    for line in labels_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except Exception as exc:
                error = error or f"Labels: {exc}"
    return classes, task, error


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
    if not model_path_text:
        return []
    classes, _task, error = openvino_package_classes(Path(model_path_text))
    if error:
        LOGGER.warning("Failed to load detector package labels for %s: %s", model_path_text, error)
    return classes

"""Inspect detector contracts without inferring precision from filenames."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


class ModelSettingsError(ValueError):
    """The model cannot be safely decoded with the selected settings."""


def metadata_value(value: Any) -> Any:
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str) and len(value) < 65536:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def read_model_metadata(path: Path, model: Any = None) -> tuple[dict, list[str]]:
    metadata: dict = {}
    warnings: list[str] = []
    sidecar = path.parent / "metadata.yaml"
    if sidecar.exists():
        try:
            import yaml

            value = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected a mapping")
            metadata.update(value)
        except Exception:
            warnings.append("Could not read metadata.yaml; using model graph and shapes.")
    if model is not None and hasattr(model, "has_rt_info"):
        for key in ("model_type", "task", "labels", "names", "args", "nms", "end2end"):
            if model.has_rt_info(["model_info", key]):
                metadata[key] = metadata_value(model.get_rt_info(["model_info", key]))
                if key == "labels" and isinstance(metadata[key], str):
                    metadata[key] = metadata[key].split()
    return metadata, warnings


def final_detection_hint(metadata: dict, has_nms: bool) -> tuple[bool | None, str]:
    args = metadata_value(metadata.get("args", {}))
    args = args if isinstance(args, dict) else {}
    def flag(name: str) -> bool | None:
        value = metadata_value(metadata.get(name, args.get(name)))
        if isinstance(value, str):
            value = {"true": True, "false": False, "yes": True, "no": False}.get(value.lower())
        return value if isinstance(value, bool) else None
    if has_nms:
        return True, "graph NMS"
    if flag("nms") is True:
        return True, "embedded NMS metadata"
    if flag("end2end") is True:
        return True, "end-to-end metadata"
    # nms=False alone also describes NMS-free end-to-end exports.
    if flag("nms") is False and flag("end2end") is False:
        return False, "raw output metadata"
    return None, "output shapes"


def resolve_output_format(shapes: list[list[int]], override: str, metadata: dict,
                          has_nms: bool = False) -> tuple[str, str, list[str]]:
    original_rank = len(shapes[0]) if shapes else 0
    shapes = [[1, *shape] if len(shape) == 2 else shape for shape in shapes]
    final, source = final_detection_hint(metadata, has_nms)
    warnings: list[str] = []
    if not shapes or any(not shape for shape in shapes):
        raise ModelSettingsError("Model has no detection outputs.")
    if override == "auto" and metadata.get("task") not in (None, "", "detect", "segment"):
        raise ModelSettingsError("Only detection and raw segmentation models are supported.")
    shape = shapes[0]
    six_columns = len(shape) == 3 and (shape[-1] == 6 or shape[1] == 6)
    segmentation = (len(shapes) == 2 and any(len(s) == 4 and s[1] == 32 for s in shapes)
                    and any(len(s) == 3 and max(s[1:]) >= 37 for s in shapes))
    if override != "auto":
        selected, source = override, "manual override"
    elif segmentation:
        detection_shape = next((s for s in shapes if len(s) == 3), [])
        final_shape = 38 in detection_shape[1:]
        if final is True or (final is None and final_shape and max(detection_shape[1:]) <= 1000):
            selected = "yolo-seg-e2e"
            if final is None:
                warnings.append("Segmentation output inferred as final detections; use the output-format override for a raw two-class model.")
        else:
            selected = "yolo-seg"
    # SSD commonly declares [1, 1, N, 7] or [N, 7]. Preserve that
    # distinction before normalizing batchless outputs: [1, N, 7] is also
    # raw three-class YOLO (xywh plus three scores), not evidence of SSD.
    elif len(shapes) == 1 and shape[-1] == 7 and original_rank in (2, 4) and final is not False:
        selected = "ssd"
    elif len(shapes) == 1 and len(shape) == 3:
        channels, anchors = min(shape[1:]), max(shape[1:])
        if final is True:
            if not six_columns:
                raise ModelSettingsError("Final detections require six columns (xyxy, score, class); select a compatible export.")
            selected = "yolo-e2e"
        elif final is False:
            selected = "yolo"
        elif channels == 6 and 6 < anchors <= 1000:
            selected = "yolo-e2e"
            warnings.append("Six-column output inferred as final detections. Use the output-format override if this is a raw two-class model.")
        elif channels >= 5 and anchors > channels:
            selected = "yolo"
        elif channels == -1 and anchors > 6:
            selected = "yolo"
        else:
            raise ModelSettingsError("Ambiguous detection output; select the model output format override.")
    else:
        raise ModelSettingsError("Unsupported detection outputs; select a compatible model or output format override.")
    if selected == "yolo-e2e" and not (len(shapes) == 1 and six_columns):
        raise ModelSettingsError("Final detections require a six-column output.")
    if selected == "yolo" and not (len(shapes) == 1 and len(shape) == 3 and
                                   (min(shape[1:]) >= 5 or (-1 in shape[1:] and max(shape[1:]) >= 5))):
        raise ModelSettingsError("Raw YOLO requires one rank-three detection output.")
    if selected in {"yolo-seg", "yolo-seg-e2e"} and not segmentation:
        raise ModelSettingsError("YOLO segmentation requires detections and 32-channel mask prototypes.")
    if selected == "yolo-seg-e2e" and not any(len(s) == 3 and 38 in s[1:] for s in shapes):
        raise ModelSettingsError("Final segmentation detections require 38 columns, including mask coefficients.")
    if selected == "ssd" and not (len(shapes) == 1 and shape[-1] == 7):
        raise ModelSettingsError("SSD requires a seven-column detection output.")
    return selected, source, warnings

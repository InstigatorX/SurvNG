"""Process-local native postprocessing policy; installed models stay read-only."""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


def _final_yolo_converter(name: str) -> bool:
    return name in {"yolo_v10", "yolo_v26"} or name.startswith("yolo_v26_")


def _final_yolo_ir(tree: ET.ElementTree) -> bool:
    entry = tree.find("rt_info/model_info/model_type")
    kind = entry.get("value", "").lower() if entry is not None else ""
    if _final_yolo_converter(kind):
        return True
    if kind != "yolo":
        return False
    # Ultralytics uses generic YOLO metadata. A final detection head exports
    # [batch, proposals, xyxy+score+class], unlike the raw [batch, channels, N].
    for port in tree.findall("layers/layer[@type='Result']/input/port"):
        dims = [int(dim.text) for dim in port.findall("dim")]
        if len(dims) == 3 and dims[-1] == 6 and dims[-2] > 6:
            return True
    return False


@contextmanager
def configured_model(model: Path | None, model_proc: str, nms_threshold: float | None):
    """Override Intel's iou_threshold without adding a second NMS pass.

    DL Streamer 2026.2 reads this from model-proc or IR model_info, not an
    element property. Intel forces NMS in its final-output YOLO converters;
    IoU=1 disables suppression there to match recorded-main final detections.
    Preserve converter/preprocessing and never add a SurvNG NMS pass. The XML
    is small; weights are referenced by symlink, never copied or modified.
    """
    if nms_threshold is None or model is None:
        yield model, model_proc
        return
    if not math.isfinite(nms_threshold) or not 0 < nms_threshold < 1:
        raise ValueError("NMS threshold must be finite and between 0 and 1")
    with tempfile.TemporaryDirectory(prefix="survng-dls-model-") as directory:
        root = Path(directory)
        if model_proc:
            payload = json.loads(Path(model_proc).read_text(encoding="utf-8"))
            outputs = payload.get("output_postproc")
            if not isinstance(outputs, list) or not outputs or not all(isinstance(item, dict) for item in outputs):
                raise ValueError("model-proc must define output_postproc to configure NMS")
            for output in outputs:
                output["iou_threshold"] = (
                    1.0 if _final_yolo_converter(str(output.get("converter", ""))) else nms_threshold
                )
            proc = root / "model-proc.json"
            proc.write_text(json.dumps(payload), encoding="utf-8")
            yield model, str(proc)
        else:
            if model.suffix.lower() != ".xml":
                raise ValueError("native NMS configuration requires OpenVINO IR or an explicit model-proc")
            tree = ET.parse(model)
            info = tree.getroot().find("rt_info")
            if info is None:
                info = ET.SubElement(tree.getroot(), "rt_info")
            model_info = info.find("model_info")
            if model_info is None:
                model_info = ET.SubElement(info, "model_info")
            threshold = model_info.find("iou_threshold")
            if threshold is None:
                threshold = ET.SubElement(model_info, "iou_threshold")
            threshold.set("value", str(1.0 if _final_yolo_ir(tree) else nms_threshold))
            configured = root / model.name
            tree.write(configured, encoding="utf-8", xml_declaration=True)
            weights = model.with_suffix(".bin")
            if weights.is_file():
                configured.with_suffix(".bin").symlink_to(weights.resolve())
            # Intel resolves YOLO's generic model_type from metadata.yaml;
            # other supported exporters use the neighboring JSON files.
            for name in ("metadata.yaml", "config.json", "preprocessor_config.json"):
                sidecar = model.parent / name
                if sidecar.is_file():
                    (root / name).symlink_to(sidecar.resolve())
            yield configured, ""

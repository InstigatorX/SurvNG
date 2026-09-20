"""Native model inspection reports effective settings without claiming validation."""
from pathlib import Path

from survng.app.config import AppConfig
from survng.app.native_runtime import NativeDetectorStatus


def _minimal_ir(path: Path, *, dims=((1, 3, 640, 640),), outputs=((1, 84, 8400),)):
    input_dims = "".join(f"<dim>{value}</dim>" for value in dims[0])
    output_layers = []
    for index, shape in enumerate(outputs):
        dims_xml = "".join(f"<dim>{value}</dim>" for value in shape)
        output_layers.append(
            f'<layer id="{index + 2}" name="out{index}" type="Result">'
            f"<input><port id=\"0\">{dims_xml}</port></input></layer>"
        )
    path.write_text(
        "<net>"
        "<layers>"
        f'<layer id="0" name="input" type="Parameter">'
        f"<output><port id=\"0\">{input_dims}</port></output></layer>"
        + "".join(output_layers)
        + "</layers></net>",
        encoding="utf-8",
    )
    return path


def test_inspect_model_reports_shapes_proc_and_unvalidated(tmp_path):
    model = _minimal_ir(tmp_path / "best.xml")
    proc = tmp_path / "best.json"
    proc.write_text(
        '{"output_postproc":[{"converter":"yolo_v8","iou_threshold":0.5}]}',
        encoding="utf-8",
    )
    config = AppConfig()
    config.detector.labels = ["person", "car"]
    detector = NativeDetectorStatus(config.detector, lambda: {})
    inspection = detector.inspect_model(str(model))
    assert inspection["error"] == ""
    assert inspection["input_shape"] == [1, 3, 640, 640]
    assert inspection["output_shapes"] == [[1, 84, 8400]]
    assert inspection["labels"] == ["person", "car"]
    assert inspection["labels_source"] == "detector.labels"
    assert inspection["model_proc_path"] == str(proc)
    assert inspection["model_proc_match"] == "basename"
    assert inspection["output_format"] == "yolo_v8"
    assert inspection["validated"] is False
    assert inspection["inspection_complete"] is True

    status = detector.status()
    assert "model_inspection" in status
    assert status["model_validated"] is False

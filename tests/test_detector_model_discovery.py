from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from survng.app.config import AppConfig, DetectorConfig
from survng.app.system_routes import (
    create_system_router,
    detector_model_search_roots,
    openvino_package_classes,
)


def _router_endpoint(config: AppConfig | None = None):
    deps = SimpleNamespace(
        get_manager=lambda: SimpleNamespace(
            detector=SimpleNamespace(
                inspect_model=lambda _path: {
                    "input_shape": [1, 3, 640, 640],
                    "output_shapes": [[1, 84, 8400]],
                    "error": "",
                }
            )
        ),
        get_config=lambda: config or AppConfig(),
        model_evaluation=SimpleNamespace(),
        event_clip_window=lambda *_args: (2.0, 4.0),
        recording_cache_status=lambda: {},
    )
    router = create_system_router(deps).router
    return next(
        route
        for route in router.routes
        if getattr(route, "path", None) == "/api/detector/models"
    ).endpoint


class DetectorModelDiscoveryTest(unittest.TestCase):
    def test_docker_models_mount_is_scanned_for_openvino_packages(self) -> None:
        package = Path("/models/yolo26s_openvino_model")
        xml = package / "yolo26s.xml"
        bin_path = package / "yolo26s.bin"
        metadata = package / "metadata.yaml"
        endpoint = _router_endpoint()

        def exists(self: Path) -> bool:
            return str(self) in {
                "/models",
                str(package),
                str(xml),
                str(bin_path),
                str(metadata),
            }

        def is_file(self: Path) -> bool:
            return str(self) in {str(xml), str(bin_path), str(metadata)}

        def glob(self: Path, pattern: str):
            if str(self) == "/models" and pattern == "*_openvino_model":
                return [package]
            return []

        def rglob(self: Path, pattern: str):
            if str(self) == str(package) and pattern == "*.xml":
                return [xml]
            return []

        def read_text(self: Path, encoding: str = "utf-8") -> str:
            del encoding
            if self == metadata:
                return "task: detect\nnames:\n  0: person\n  1: car\n"
            raise FileNotFoundError(self)

        with (
            patch.object(Path, "exists", exists),
            patch.object(Path, "is_file", is_file),
            patch.object(Path, "glob", glob),
            patch.object(Path, "rglob", rglob),
            patch.object(Path, "read_text", read_text),
        ):
            payload = endpoint()

        self.assertEqual(payload["models"][0]["path"], str(xml))
        self.assertTrue(payload["models"][0]["valid"])
        self.assertEqual(payload["models"][0]["classes"], ["person", "car"])

    def test_package_classes_fall_back_to_classes_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "yolo26s_openvino_model"
            package.mkdir()
            xml = package / "yolo26s.xml"
            xml.write_text("<net name='det'></net>\n", encoding="utf-8")
            (package / "classes.txt").write_text("person\ntruck\n", encoding="utf-8")

            classes, task, error = openvino_package_classes(xml)

        self.assertEqual(classes, ["person", "truck"])
        self.assertEqual(task, "")
        self.assertEqual(error, "")

    def test_active_model_directory_is_scanned_even_without_openvino_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "custom-detector"
            package.mkdir()
            xml = package / "model.xml"
            xml.write_text("<net name='det'></net>\n", encoding="utf-8")
            (package / "model.bin").write_bytes(b"weights")
            (package / "classes.txt").write_text("person\ncar\n", encoding="utf-8")
            roots = detector_model_search_roots(str(xml), bases=())
            self.assertIn(package, roots)

            endpoint = _router_endpoint(
                AppConfig(
                    detector=DetectorConfig(
                        model_path=str(xml),
                        labels_path=str(package / "classes.txt"),
                    )
                )
            )
            payload = endpoint()

        match = next(item for item in payload["models"] if item["path"] == str(xml))
        self.assertEqual(match["classes"], ["person", "car"])
        self.assertTrue(match["valid"])


if __name__ == "__main__":
    unittest.main()

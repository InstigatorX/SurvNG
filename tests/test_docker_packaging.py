from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from survng.app.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class DockerPackagingTest(unittest.TestCase):
    def test_docker_config_is_valid_camera_free_and_secret_free(self) -> None:
        path = ROOT / "docker" / "config.example.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = load_config(path)

        self.assertEqual(config.cameras, [])
        self.assertFalse(config.detector.enabled)
        self.assertFalse(config.mqtt.enabled)
        self.assertFalse(config.audit_ai.enabled)
        self.assertEqual(payload["mqtt"]["password"], "")
        self.assertEqual(payload["audit_ai"]["api_key"], "")

    def test_build_context_excludes_confidential_runtime_material(self) -> None:
        patterns = set(
            line.strip()
            for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

        for expected in (
            "config.json",
            ".env",
            ".env.*",
            "docker-data/",
            "runtime/",
            "*.sqlite3*",
            "*.pem",
            "*.key",
            "*.crt",
            "*.onnx",
            "*.xml",
            "*.bin",
        ):
            with self.subTest(pattern=expected):
                self.assertIn(expected, patterns)

    def test_compose_mounts_state_outside_the_image(self) -> None:
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")

        self.assertIn("SURVNG_CONFIG_PATH: /config/config.json", compose)
        self.assertIn("SURVNG_GIT_SHA: ${SURVNG_GIT_SHA:-}", compose)
        self.assertIn(":/config", compose)
        self.assertIn(":/data", compose)
        self.assertIn(":/media", compose)
        self.assertIn(":/models:ro", compose)
        self.assertNotIn("password:", compose.lower())
        self.assertNotIn("api_key:", compose.lower())

    def test_dockerfile_records_optional_git_sha(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG SURVNG_GIT_SHA=", dockerfile)
        self.assertIn("ENV SURVNG_GIT_SHA=$SURVNG_GIT_SHA", dockerfile)
        self.assertIn("/app/SURVNG_GIT_SHA", dockerfile)

    def test_update_from_git_script_is_executable(self) -> None:
        script = ROOT / "scripts" / "update-from-git.sh"
        self.assertTrue(script.is_file())
        self.assertTrue(os.access(script, os.X_OK))
        text = script.read_text(encoding="utf-8")
        self.assertIn("git pull --ff-only", text)
        self.assertIn("docker compose", text)

    def test_lxc_override_is_explicit_and_not_part_of_default_compose(self) -> None:
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        lxc_override = (ROOT / "compose.lxc.yaml").read_text(encoding="utf-8")

        self.assertNotIn("apparmor=unconfined", compose)
        self.assertNotIn("privileged: true", compose)
        self.assertIn("apparmor=unconfined", lxc_override)

    def test_intel_image_uses_modern_pinned_ubuntu_gpu_userspace(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("FROM ubuntu:24.04 AS runtime-base", dockerfile)
        self.assertIn("INTEL_COMPUTE_VERSION=26.27.39122.14-1~24.04~ppa1", dockerfile)
        self.assertIn("INTEL_IGC_VERSION=2.38.5-1~24.04", dockerfile)
        self.assertIn("INTEL_LEVEL_ZERO_VERSION=1.32.0-1~24.04~ppa1", dockerfile)
        self.assertIn("INTEL_MEDIA_VERSION=26.2.2-1~24.04~ppa1", dockerfile)
        self.assertIn("ppa:kobuk-team/intel-graphics", dockerfile)
        self.assertIn('"libze-intel-gpu1=${INTEL_COMPUTE_VERSION}"', dockerfile)
        self.assertIn('"intel-media-va-driver-non-free=${INTEL_MEDIA_VERSION}"', dockerfile)
        self.assertNotIn("intel-media-va-driver \\", dockerfile)
        # Legacy docker builders reject COPY --chmod (BuildKit-only).
        self.assertNotIn("COPY --chmod=", dockerfile)
        self.assertIn(
            "COPY docker/entrypoint.sh /usr/local/bin/survng-entrypoint",
            dockerfile,
        )
        self.assertIn("RUN chmod 755 /usr/local/bin/survng-entrypoint", dockerfile)

    def test_lxc_builder_is_persistent_and_scope_limited(self) -> None:
        script = (ROOT / "scripts" / "docker-build-lxc.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("--security-opt apparmor=unconfined", script)
        self.assertIn("--privileged", script)
        self.assertIn('docker-container://${BUILDER_CONTAINER}', script)
        self.assertIn("--restart unless-stopped", script)
        self.assertIn('survng-buildkit-state', script)
        self.assertNotIn("trap cleanup EXIT INT TERM", script)
        self.assertIn('runtime|runtime-intel', script)

    def test_docker_readme_covers_safe_migration_and_rollback(self) -> None:
        readme = (ROOT / "docker" / "README.md").read_text(encoding="utf-8")

        self.assertIn("systemctl stop survng.service", readme)
        self.assertIn("up -d --no-build", readme)
        self.assertIn("systemctl start survng.service", readme)
        self.assertIn("SURVNG_CONFIG_DIR=", readme)
        self.assertIn("SURVNG_MEDIA_DIR=", readme)
        self.assertIn("OpenVINO should report both `CPU` and `GPU`", readme)
        self.assertNotIn("privileged: true", readme)
        self.assertIn("scripts/install-docker-models.sh", readme)
        self.assertIn("yolo26s_openvino_model", readme)

    def test_install_docker_models_script_is_executable(self) -> None:
        script = ROOT / "scripts" / "install-docker-models.sh"
        self.assertTrue(script.is_file())
        self.assertTrue(os.access(script, os.X_OK))
        text = script.read_text(encoding="utf-8")
        self.assertIn(
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-reidentification-retail-0286/FP16",
            text,
        )
        self.assertIn(
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx",
            text,
        )
        self.assertIn(
            "4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f",
            text,
        )
        self.assertIn("yolo26s_openvino_model", text)
        self.assertIn("/models/", text)
        self.assertIn("mobileclip2-b-openvino-fp16", text)
        self.assertIn("AGPL-3.0", text)
        self.assertIn("Apple", text)
        self.assertIn("No SurvNG Git checkout is required", text)
        self.assertIn("raw.githubusercontent.com/InstigatorX/SurvNG", text)
        self.assertIn("--native", text)
        self.assertIn("survng-model-installer", text)
        self.assertIn("THIRD_PARTY_MODELS.md", text)
        self.assertIn("quantize=16", text)
        self.assertNotIn("resolve_survng_root", text)

    def test_install_docker_models_runs_without_repo_checkout(self) -> None:
        script = ROOT / "scripts" / "install-docker-models.sh"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            standalone = tmp_path / "install-docker-models.sh"
            shutil.copy2(script, standalone)
            standalone.chmod(0o755)
            models = tmp_path / "models"
            config_path = tmp_path / "config" / "config.json"
            models.mkdir()
            (models / "yolo26s_openvino_model").mkdir()
            (models / "yolo26s_openvino_model" / "yolo26s.xml").write_text(
                "<net name='det'></net>\n", encoding="utf-8"
            )
            (models / "yolo26s_openvino_model" / "yolo26s.bin").write_bytes(b"det")
            (models / "yolo26s_openvino_model" / "classes.txt").write_text(
                "person\n", encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    str(standalone),
                    "--native",
                    "--models-dir",
                    str(models),
                    "--config",
                    str(config_path),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--skip-detector",
                    "--skip-reid",
                    "--skip-semantic",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(tmp_path),
            )
            self.assertIn("SurvNG Docker model installer", completed.stdout)
            self.assertNotIn("Could not locate SurvNG repository root", completed.stdout)
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["detector"]["model_path"],
                "/models/yolo26s_openvino_model/yolo26s.xml",
            )

    def test_install_docker_models_patches_config_without_wiping_cameras(self) -> None:
        script = ROOT / "scripts" / "install-docker-models.sh"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            models = tmp_path / "models"
            config_path = tmp_path / "config" / "config.json"
            detector = models / "yolo26s_openvino_model"
            person = models / "person_reid_model"
            vehicle = models / "vehicle_reid_model"
            semantic = models / "mobileclip2-b-openvino-fp16"
            detector.mkdir(parents=True)
            person.mkdir()
            vehicle.mkdir()
            semantic.mkdir()
            (detector / "yolo26s.xml").write_text("<net name='det'></net>\n", encoding="utf-8")
            (detector / "yolo26s.bin").write_bytes(b"det")
            (detector / "classes.txt").write_text("person\ncar\n", encoding="utf-8")
            (person / "person-reidentification-retail-0286.xml").write_text(
                "<net name='reid'></net>\n", encoding="utf-8"
            )
            (person / "person-reidentification-retail-0286.bin").write_bytes(b"reid")
            (vehicle / "vehicle-reid-0001.onnx").write_bytes(b"onnx")
            (semantic / "semantic_model.json").write_text("{}\n", encoding="utf-8")
            (semantic / "image_encoder.xml").write_text("<net></net>\n", encoding="utf-8")

            payload = json.loads(
                (ROOT / "docker" / "config.example.json").read_text(encoding="utf-8")
            )
            payload["cameras"] = [{"name": "front-gate", "enabled": False}]
            payload["mqtt"]["host"] = "keep-this-broker.example"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    str(script),
                    "--native",
                    "--models-dir",
                    str(models),
                    "--config",
                    str(config_path),
                    "--device",
                    "GPU",
                    "--cache-dir",
                    str(tmp_path / "cache"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Updated", completed.stdout)

            updated = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["cameras"], payload["cameras"])
            self.assertEqual(updated["mqtt"]["host"], "keep-this-broker.example")
            self.assertTrue(updated["detector"]["enabled"])
            self.assertEqual(
                updated["detector"]["model_path"],
                "/models/yolo26s_openvino_model/yolo26s.xml",
            )
            self.assertEqual(
                updated["detector"]["labels_path"],
                "/models/yolo26s_openvino_model/classes.txt",
            )
            self.assertEqual(updated["detector"]["device"], "GPU")
            self.assertEqual(updated["detector"]["cache_dir"], "/data/openvino-cache")
            tracking = updated["detector"]["tracking"]
            self.assertTrue(tracking["reid_enabled"])
            self.assertEqual(
                tracking["reid_model_path"],
                "/models/person_reid_model/person-reidentification-retail-0286.xml",
            )
            self.assertTrue(tracking["vehicle_reid_enabled"])
            self.assertEqual(
                tracking["vehicle_reid_model_path"],
                "/models/vehicle_reid_model/vehicle-reid-0001.onnx",
            )
            self.assertTrue(updated["semantic_search"]["enabled"])
            self.assertEqual(
                updated["semantic_search"]["model_dir"],
                "/models/mobileclip2-b-openvino-fp16",
            )
            self.assertEqual(updated["semantic_search"]["device"], "GPU")
            mode = stat.S_IMODE(config_path.stat().st_mode)
            self.assertEqual(mode, 0o600)

    def test_install_docker_models_patches_config_without_semantic(self) -> None:
        script = ROOT / "scripts" / "install-docker-models.sh"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            models = tmp_path / "models"
            config_path = tmp_path / "config" / "config.json"
            detector = models / "yolo26s_openvino_model"
            person = models / "person_reid_model"
            vehicle = models / "vehicle_reid_model"
            detector.mkdir(parents=True)
            person.mkdir()
            vehicle.mkdir()
            (detector / "yolo26s.xml").write_text("<net name='det'></net>\n", encoding="utf-8")
            (detector / "yolo26s.bin").write_bytes(b"det")
            (detector / "classes.txt").write_text("person\ncar\n", encoding="utf-8")
            (person / "person-reidentification-retail-0286.xml").write_text(
                "<net name='reid'></net>\n", encoding="utf-8"
            )
            (person / "person-reidentification-retail-0286.bin").write_bytes(b"reid")
            (vehicle / "vehicle-reid-0001.onnx").write_bytes(b"onnx")

            completed = subprocess.run(
                [
                    str(script),
                    "--native",
                    "--models-dir",
                    str(models),
                    "--config",
                    str(config_path),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--skip-detector",
                    "--skip-reid",
                    "--skip-semantic",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Updated", completed.stdout)
            updated = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertTrue(updated["detector"]["enabled"])
            self.assertEqual(
                updated["detector"]["model_path"],
                "/models/yolo26s_openvino_model/yolo26s.xml",
            )
            tracking = updated["detector"]["tracking"]
            self.assertTrue(tracking["reid_enabled"])
            self.assertTrue(tracking["vehicle_reid_enabled"])
            self.assertNotIn("semantic_search", updated)

    def test_model_installer_dockerfile_and_notices_exist(self) -> None:
        dockerfile = ROOT / "Dockerfile.model-installer"
        self.assertTrue(dockerfile.is_file())
        text = dockerfile.read_text(encoding="utf-8")
        self.assertIn("model-installer", text)
        self.assertIn("THIRD_PARTY_MODELS.md", text)
        self.assertIn("install-docker-models.sh", text)
        notice = ROOT / "docker" / "model-installer" / "NOTICE"
        models_doc = ROOT / "docker" / "model-installer" / "THIRD_PARTY_MODELS.md"
        self.assertTrue(notice.is_file())
        self.assertTrue(models_doc.is_file())
        self.assertIn("vehicle-reid-0001", models_doc.read_text(encoding="utf-8"))
        self.assertIn("ONNX", models_doc.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

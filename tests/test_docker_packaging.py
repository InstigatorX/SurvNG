from __future__ import annotations

import json
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
        self.assertIn(":/config", compose)
        self.assertIn(":/data", compose)
        self.assertIn(":/media", compose)
        self.assertIn(":/models:ro", compose)
        self.assertNotIn("password:", compose.lower())
        self.assertNotIn("api_key:", compose.lower())

    def test_lxc_override_is_explicit_and_not_part_of_default_compose(self) -> None:
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        lxc_override = (ROOT / "compose.lxc.yaml").read_text(encoding="utf-8")

        self.assertNotIn("apparmor=unconfined", compose)
        self.assertNotIn("privileged: true", compose)
        self.assertIn("apparmor=unconfined", lxc_override)

    def test_intel_image_uses_modern_pinned_ubuntu_gpu_userspace(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("FROM ubuntu:24.04 AS runtime-base", dockerfile)
        self.assertIn("INTEL_COMPUTE_VERSION=26.22.38646.6-1~24.04~ppa1", dockerfile)
        self.assertIn("INTEL_IGC_VERSION=2.36.3-2~24.04", dockerfile)
        self.assertIn("INTEL_MEDIA_VERSION=26.2.2-1~24.04~ppa1", dockerfile)
        self.assertIn("ppa:kobuk-team/intel-graphics", dockerfile)
        self.assertIn('"libze-intel-gpu1=${INTEL_COMPUTE_VERSION}"', dockerfile)
        self.assertIn('"intel-media-va-driver-non-free=${INTEL_MEDIA_VERSION}"', dockerfile)
        self.assertNotIn("intel-media-va-driver \\", dockerfile)

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


if __name__ == "__main__":
    unittest.main()

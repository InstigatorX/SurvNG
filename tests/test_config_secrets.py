from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from survng.app.config import (
    ApiAuthConfig,
    ApiTokenConfig,
    AppConfig,
    AuditAiConfig,
    CameraConfig,
    MqttConfig,
    OnvifConfig,
    WebAuthConfig,
    WebUserConfig,
    load_config,
    save_config,
)
from survng.app.config_secrets import (
    SECRET_PLACEHOLDER,
    apply_secrets_to_raw_config,
    extract_secrets,
    public_config_dict,
    secrets_path,
)
from survng.app.security import hash_api_token, hash_password


def _secret_config() -> AppConfig:
    return AppConfig(
        mqtt=MqttConfig(password="mqtt-secret"),
        audit_ai=AuditAiConfig(api_key="ai-secret"),
        api_auth=ApiAuthConfig(
            enabled=True,
            tokens=[
                ApiTokenConfig(
                    id="ha",
                    name="Home Assistant",
                    token_hash=hash_api_token("token-secret"),
                    scopes=["read"],
                )
            ],
        ),
        web_auth=WebAuthConfig(
            enabled=True,
            session_key="a" * 64,
            users=[
                WebUserConfig(
                    id="alex",
                    username="alex",
                    display_name="Alex",
                    role="admin",
                    password_hash=hash_password("correct-horse"),
                )
            ],
        ),
        cameras=[
            CameraConfig(
                id="gate",
                name="Gate",
                stream_url="rtsp://viewer:cam-secret@camera.local/main",
                live_stream_url="rtsp://viewer:live-secret@camera.local/sub",
                onvif=OnvifConfig(
                    enabled=True,
                    host="camera.local",
                    username="viewer",
                    password="onvif-secret",
                ),
            )
        ],
    )


class ConfigSecretsTest(unittest.TestCase):
    def test_legacy_monolithic_config_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config = _secret_config()
            path.write_text(
                json.dumps(config.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
            loaded = load_config(str(path))
            self.assertEqual(loaded.mqtt.password, "mqtt-secret")
            self.assertEqual(loaded.cameras[0].onvif.password, "onvif-secret")
            self.assertFalse((Path(tmpdir) / "secrets.json").exists())

    def test_save_splits_secrets_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            sidecar = Path(tmpdir) / "secrets.json"
            config = _secret_config()
            save_config(config, str(path), assign_ids=False)

            public = json.loads(path.read_text(encoding="utf-8"))
            secrets = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(sidecar.stat().st_mode & 0o777, 0o600)
            self.assertEqual(public["mqtt"]["password"], SECRET_PLACEHOLDER)
            self.assertEqual(public["audit_ai"]["api_key"], SECRET_PLACEHOLDER)
            self.assertEqual(public["api_auth"]["tokens"][0]["token_hash"], SECRET_PLACEHOLDER)
            self.assertEqual(public["web_auth"]["session_key"], SECRET_PLACEHOLDER)
            self.assertEqual(public["web_auth"]["users"][0]["password_hash"], SECRET_PLACEHOLDER)
            self.assertIn(SECRET_PLACEHOLDER, public["cameras"][0]["stream_url"])
            self.assertNotIn("cam-secret", path.read_text(encoding="utf-8"))
            self.assertNotIn("mqtt-secret", path.read_text(encoding="utf-8"))
            self.assertEqual(secrets["mqtt"]["password"], "mqtt-secret")
            self.assertEqual(secrets["cameras"]["gate"]["onvif"]["password"], "onvif-secret")

            loaded = load_config(str(path))
            self.assertEqual(loaded.mqtt.password, config.mqtt.password)
            self.assertEqual(loaded.audit_ai.api_key, config.audit_ai.api_key)
            self.assertEqual(
                loaded.api_auth.tokens[0].token_hash,
                config.api_auth.tokens[0].token_hash,
            )
            self.assertEqual(loaded.web_auth.session_key, config.web_auth.session_key)
            self.assertEqual(
                loaded.web_auth.users[0].password_hash,
                config.web_auth.users[0].password_hash,
            )
            self.assertEqual(loaded.cameras[0].stream_url, config.cameras[0].stream_url)
            self.assertEqual(loaded.cameras[0].live_stream_url, config.cameras[0].live_stream_url)
            self.assertEqual(loaded.cameras[0].onvif.password, config.cameras[0].onvif.password)

    def test_secrets_path_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            sidecar = Path(tmpdir) / "private" / "creds.json"
            with patch.dict(os.environ, {"SURVNG_SECRETS_PATH": str(sidecar)}):
                self.assertEqual(secrets_path(config_file), sidecar)
                save_config(_secret_config(), str(config_file), assign_ids=False)
                self.assertTrue(sidecar.is_file())
                self.assertFalse((Path(tmpdir) / "secrets.json").exists())

    def test_merge_prefers_sidecar_secret_fields(self) -> None:
        raw = {
            "mqtt": {"password": SECRET_PLACEHOLDER},
            "cameras": [
                {
                    "id": "gate",
                    "name": "Gate",
                    "stream_url": f"rtsp://viewer:{SECRET_PLACEHOLDER}@camera.local/main",
                    "onvif": {"password": SECRET_PLACEHOLDER},
                }
            ],
        }
        secrets = {
            "version": 1,
            "mqtt": {"password": "mqtt-secret"},
            "cameras": {
                "gate": {
                    "stream_url": "rtsp://viewer:cam-secret@camera.local/main",
                    "onvif": {"password": "onvif-secret"},
                }
            },
        }
        merged = apply_secrets_to_raw_config(raw, secrets)
        self.assertEqual(merged["mqtt"]["password"], "mqtt-secret")
        self.assertEqual(
            merged["cameras"][0]["stream_url"],
            "rtsp://viewer:cam-secret@camera.local/main",
        )
        self.assertEqual(merged["cameras"][0]["onvif"]["password"], "onvif-secret")

    def test_public_config_keeps_revoked_sessions(self) -> None:
        config = AppConfig(
            web_auth=WebAuthConfig(
                enabled=True,
                session_key="a" * 64,
                users=[
                    WebUserConfig(
                        id="alex",
                        username="alex",
                        role="admin",
                        password_hash=hash_password("correct-horse"),
                    )
                ],
                revoked_sessions={"a" * 64: 2_000_000_000},
            )
        )
        public = public_config_dict(config)
        self.assertEqual(public["web_auth"]["revoked_sessions"], {"a" * 64: 2_000_000_000})
        self.assertEqual(public["web_auth"]["session_key"], SECRET_PLACEHOLDER)

    def test_extract_omits_empty_secret_blocks(self) -> None:
        secrets = extract_secrets(AppConfig())
        self.assertEqual(secrets, {"version": 1})


if __name__ == "__main__":
    unittest.main()

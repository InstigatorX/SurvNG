"""Persist credential material beside config.json in an encrypted sidecar."""

from __future__ import annotations

import base64
import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import AppConfig, CONFIG_PATH_ENV

SECRET_PLACEHOLDER = "__SURVNG_SECRET_SET__"
SECRETS_PATH_ENV = "SURVNG_SECRETS_PATH"
SECRETS_KEY_PATH_ENV = "SURVNG_SECRETS_KEY_PATH"
SECRETS_SCHEMA_VERSION = 1
SECRETS_ENVELOPE_VERSION = 2
SECRETS_ALG = "AESGCM"
PRIVATE_FILE_MODE = 0o600
_KEY_BYTES = 32
_NONCE_BYTES = 12

_URL_SECRET_KEYS = frozenset({
    "password", "passwd", "pass", "pwd", "token", "accesstoken", "refreshtoken",
    "idtoken", "apikey", "key", "secret", "clientsecret", "authorization", "auth",
})


def _url_secret_key(key: str) -> bool:
    return unquote_plus(key).casefold().replace("_", "").replace("-", "") in _URL_SECRET_KEYS


def mask_url_password(value: str | None) -> str | None:
    if not value:
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    userinfo, separator, host = parsed.netloc.rpartition("@")
    if separator and ":" in userinfo:
        username, _password = userinfo.split(":", 1)
        parsed = parsed._replace(netloc=f"{username}:{SECRET_PLACEHOLDER}@{host}")
    parts = []
    for part in parsed.query.split("&"):
        key, separator, item = part.partition("=")
        parts.append(
            f"{key}={SECRET_PLACEHOLDER}"
            if separator and item and _url_secret_key(key)
            else part
        )
    return urlunsplit(parsed._replace(query="&".join(parts)))


def secrets_path(config_file: str | Path | None = None) -> Path:
    configured = os.environ.get(SECRETS_PATH_ENV, "").strip()
    if configured:
        return Path(configured)
    if config_file is not None:
        return Path(config_file).expanduser().resolve().parent / "secrets.json"
    configured_config = os.environ.get(CONFIG_PATH_ENV, "").strip()
    if configured_config:
        return Path(configured_config).expanduser().resolve().parent / "secrets.json"
    return Path("secrets.json")


def secrets_key_path(secrets_file: str | Path | None = None) -> Path:
    configured = os.environ.get(SECRETS_KEY_PATH_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    if secrets_file is not None:
        return Path(secrets_file).expanduser().resolve().parent / "secrets.key"
    return secrets_path().parent / "secrets.key"


def extract_secrets(config: AppConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {"version": SECRETS_SCHEMA_VERSION}
    if config.mqtt.password:
        payload["mqtt"] = {"password": config.mqtt.password}
    if config.audit_ai.api_key:
        payload["audit_ai"] = {"api_key": config.audit_ai.api_key}
    tokens = {
        token.id: token.token_hash
        for token in config.api_auth.tokens
        if token.token_hash and token.token_hash != SECRET_PLACEHOLDER
    }
    if tokens:
        payload["api_auth"] = {"tokens": tokens}
    web_auth: dict[str, Any] = {}
    if config.web_auth.session_key and config.web_auth.session_key != SECRET_PLACEHOLDER:
        web_auth["session_key"] = config.web_auth.session_key
    users = {
        user.id: user.password_hash
        for user in config.web_auth.users
        if user.password_hash and user.password_hash != SECRET_PLACEHOLDER
    }
    if users:
        web_auth["users"] = users
    if web_auth:
        payload["web_auth"] = web_auth
    cameras: dict[str, Any] = {}
    for camera in config.cameras:
        camera_secrets: dict[str, Any] = {}
        if camera.stream_url:
            camera_secrets["stream_url"] = camera.stream_url
        if camera.live_stream_url:
            camera_secrets["live_stream_url"] = camera.live_stream_url
        if camera.onvif.password:
            camera_secrets["onvif"] = {"password": camera.onvif.password}
        if camera_secrets:
            cameras[camera.id] = camera_secrets
    if cameras:
        payload["cameras"] = cameras
    return payload


def public_config_dict(config: AppConfig) -> dict[str, Any]:
    """Serialize AppConfig for config.json without credential material."""
    payload = config.model_dump(mode="json")
    payload["mqtt"]["password"] = SECRET_PLACEHOLDER if config.mqtt.password else ""
    payload["audit_ai"]["api_key"] = SECRET_PLACEHOLDER if config.audit_ai.api_key else ""
    for token in payload["api_auth"]["tokens"]:
        if token.get("token_hash"):
            token["token_hash"] = SECRET_PLACEHOLDER
    if payload["web_auth"].get("session_key"):
        payload["web_auth"]["session_key"] = SECRET_PLACEHOLDER
    for user in payload["web_auth"]["users"]:
        if user.get("password_hash"):
            user["password_hash"] = SECRET_PLACEHOLDER
    for camera in payload["cameras"]:
        camera["stream_url"] = mask_url_password(camera.get("stream_url"))
        camera["live_stream_url"] = mask_url_password(camera.get("live_stream_url"))
        onvif = camera.get("onvif") or {}
        onvif["password"] = SECRET_PLACEHOLDER if onvif.get("password") else ""
        camera["onvif"] = onvif
    return payload


def apply_secrets_to_raw_config(
    raw: dict[str, Any],
    secrets_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if not secrets_payload:
        return raw
    merged = json.loads(json.dumps(raw))
    mqtt = secrets_payload.get("mqtt") or {}
    if mqtt.get("password"):
        merged.setdefault("mqtt", {})["password"] = mqtt["password"]
    audit_ai = secrets_payload.get("audit_ai") or {}
    if audit_ai.get("api_key"):
        merged.setdefault("audit_ai", {})["api_key"] = audit_ai["api_key"]
    token_secrets = (secrets_payload.get("api_auth") or {}).get("tokens") or {}
    if token_secrets:
        for token in merged.setdefault("api_auth", {}).setdefault("tokens", []):
            token_id = str(token.get("id") or "")
            if token_id in token_secrets:
                token["token_hash"] = token_secrets[token_id]
    web_secrets = secrets_payload.get("web_auth") or {}
    if web_secrets.get("session_key"):
        merged.setdefault("web_auth", {})["session_key"] = web_secrets["session_key"]
    user_secrets = web_secrets.get("users") or {}
    if user_secrets:
        for user in merged.setdefault("web_auth", {}).setdefault("users", []):
            user_id = str(user.get("id") or "")
            if user_id in user_secrets:
                user["password_hash"] = user_secrets[user_id]
    camera_secrets = secrets_payload.get("cameras") or {}
    if camera_secrets:
        for camera in merged.setdefault("cameras", []):
            camera_id = str(camera.get("id") or "")
            secret = camera_secrets.get(camera_id)
            if not isinstance(secret, dict):
                continue
            if secret.get("stream_url"):
                camera["stream_url"] = secret["stream_url"]
            if "live_stream_url" in secret:
                camera["live_stream_url"] = secret.get("live_stream_url")
            onvif_secret = secret.get("onvif") or {}
            if onvif_secret.get("password"):
                camera.setdefault("onvif", {})["password"] = onvif_secret["password"]
    return merged


def _load_or_create_key(path: Path) -> bytes:
    if path.is_file():
        raw = path.read_bytes().strip()
        if len(raw) == _KEY_BYTES:
            return raw
        try:
            decoded = base64.urlsafe_b64decode(raw)
        except Exception as exc:  # noqa: BLE001 — surface as ValueError below
            raise ValueError(f"secrets key file is not valid base64: {path}") from exc
        if len(decoded) != _KEY_BYTES:
            raise ValueError(
                f"secrets key must be {_KEY_BYTES} bytes (raw or urlsafe base64): {path}"
            )
        return decoded
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(_KEY_BYTES)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            try:
                os.chmod(temporary_path, PRIVATE_FILE_MODE)
            except OSError:
                pass
            handle.write(base64.urlsafe_b64encode(key))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        try:
            os.chmod(path, PRIVATE_FILE_MODE)
        except OSError:
            pass
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return key


def _is_encrypted_envelope(payload: dict[str, Any]) -> bool:
    return (
        int(payload.get("version") or 0) == SECRETS_ENVELOPE_VERSION
        and payload.get("alg") == SECRETS_ALG
        and isinstance(payload.get("nonce"), str)
        and isinstance(payload.get("ciphertext"), str)
    )


def _encrypt_secrets(plaintext: dict[str, Any], key: bytes) -> dict[str, Any]:
    nonce = secrets.token_bytes(_NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(
        nonce,
        json.dumps(plaintext, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        None,
    )
    return {
        "version": SECRETS_ENVELOPE_VERSION,
        "alg": SECRETS_ALG,
        "nonce": base64.urlsafe_b64encode(nonce).decode("ascii"),
        "ciphertext": base64.urlsafe_b64encode(ciphertext).decode("ascii"),
    }


def _decrypt_secrets(envelope: dict[str, Any], key: bytes) -> dict[str, Any]:
    try:
        nonce = base64.urlsafe_b64decode(envelope["nonce"])
        ciphertext = base64.urlsafe_b64decode(envelope["ciphertext"])
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
        payload = json.loads(plaintext.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — wrap for callers
        raise ValueError("failed to decrypt secrets.json (wrong key or corrupt file)") from exc
    if not isinstance(payload, dict):
        raise ValueError("decrypted secrets payload must be a JSON object")
    return payload


def load_secrets_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"secrets file must be a JSON object: {path}")
    if _is_encrypted_envelope(payload):
        key = _load_or_create_key(secrets_key_path(path))
        return _decrypt_secrets(payload, key)
    # Legacy plaintext sidecar (schema v1) remains readable until the next save.
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            try:
                os.chmod(temporary_path, PRIVATE_FILE_MODE)
            except OSError:
                pass
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        try:
            os.chmod(path, PRIVATE_FILE_MODE)
        except OSError:
            pass
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def save_secrets_file(secrets_payload: dict[str, Any], path: Path) -> None:
    key = _load_or_create_key(secrets_key_path(path))
    _atomic_write_json(path, _encrypt_secrets(secrets_payload, key))

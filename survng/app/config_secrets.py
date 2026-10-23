"""Persist credential material beside config.json in a private sidecar file."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from .config import AppConfig, CONFIG_PATH_ENV

SECRET_PLACEHOLDER = "__SURVNG_SECRET_SET__"
SECRETS_PATH_ENV = "SURVNG_SECRETS_PATH"
SECRETS_SCHEMA_VERSION = 1
PRIVATE_FILE_MODE = 0o600

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
    secrets: dict[str, Any] | None,
) -> dict[str, Any]:
    if not secrets:
        return raw
    merged = json.loads(json.dumps(raw))
    mqtt = secrets.get("mqtt") or {}
    if mqtt.get("password"):
        merged.setdefault("mqtt", {})["password"] = mqtt["password"]
    audit_ai = secrets.get("audit_ai") or {}
    if audit_ai.get("api_key"):
        merged.setdefault("audit_ai", {})["api_key"] = audit_ai["api_key"]
    token_secrets = (secrets.get("api_auth") or {}).get("tokens") or {}
    if token_secrets:
        for token in merged.setdefault("api_auth", {}).setdefault("tokens", []):
            token_id = str(token.get("id") or "")
            if token_id in token_secrets:
                token["token_hash"] = token_secrets[token_id]
    web_secrets = secrets.get("web_auth") or {}
    if web_secrets.get("session_key"):
        merged.setdefault("web_auth", {})["session_key"] = web_secrets["session_key"]
    user_secrets = web_secrets.get("users") or {}
    if user_secrets:
        for user in merged.setdefault("web_auth", {}).setdefault("users", []):
            user_id = str(user.get("id") or "")
            if user_id in user_secrets:
                user["password_hash"] = user_secrets[user_id]
    camera_secrets = secrets.get("cameras") or {}
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


def load_secrets_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"secrets file must be a JSON object: {path}")
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


def save_secrets_file(secrets: dict[str, Any], path: Path) -> None:
    _atomic_write_json(path, secrets)

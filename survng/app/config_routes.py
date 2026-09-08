"""Admin configuration HTTP boundary."""

from __future__ import annotations

import logging
import secrets
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .config import ApiScope, ApiTokenConfig, AppConfig, CameraConfig, DetectionZone, camera_by_id, slugify_camera_id
from .security import hash_api_token, redact_secret_text

SECRET_PLACEHOLDER = "__SURVNG_SECRET_SET__"
LOGGER = logging.getLogger(__name__)
_URL_SECRET_KEYS = frozenset({
    "password", "passwd", "pass", "pwd", "token", "accesstoken", "refreshtoken",
    "idtoken", "apikey", "key", "secret", "clientsecret", "authorization", "auth",
})


def _url_secret_key(key: str) -> bool:
    return unquote_plus(key).casefold().replace("_", "").replace("-", "") in _URL_SECRET_KEYS


def _query_secrets(query: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for part in query.split("&"):
        key, separator, value = part.partition("=")
        if separator and _url_secret_key(key):
            values.setdefault(unquote_plus(key).casefold(), []).append(value)
    return values



class ConfigProbeRequest(BaseModel):
    camera_id: str = Field(default="", max_length=128)
    host: str = Field(min_length=1, max_length=255, pattern=r"^[^\s/@?#]+$")
    username: str = Field(default="", max_length=256)
    password: str = Field(default="", max_length=1024)
    onvif_port: int = Field(default=8000, ge=1, le=65535)


class ApiTokenCreateRequest(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    name: str = Field(min_length=1, max_length=128)
    scopes: list[ApiScope] = Field(default_factory=lambda: ["read"], min_length=1)


class ConfigRuntime(Protocol):
    """Narrow runtime surface required by configuration HTTP handlers."""

    config: AppConfig
    workers: dict[str, Any]

    def update_camera_zones(
        self,
        camera_id: str,
        zones: list[DetectionZone],
        rollback_zones: list[dict],
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class ConfigRouteDependencies:
    get_config: Callable[[], AppConfig]
    get_manager: Callable[[], ConfigRuntime]
    publish_config: Callable[[AppConfig], None]
    apply_config: Callable[..., tuple[AppConfig, dict[str, object]]]
    reload_manager: Callable[[AppConfig], AppConfig]
    save_config: Callable[..., None]
    validate_config: Callable[[AppConfig], None]
    lock: threading.RLock
    probe_limiter: threading.BoundedSemaphore


def _mask_url_password(value: str | None) -> str | None:
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
    # Keep non-secret values and their encoding byte-for-byte (some cameras sign URLs).
    parts = []
    for part in parsed.query.split("&"):
        key, separator, value = part.partition("=")
        parts.append(f"{key}={SECRET_PLACEHOLDER}" if separator and value and _url_secret_key(key) else part)
    return urlunsplit(parsed._replace(query="&".join(parts)))


def _encoded_url_password(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    userinfo, separator, _host = parsed.netloc.rpartition("@")
    return userinfo.split(":", 1)[1] if separator and ":" in userinfo else None


def _restore_url_password(masked: str | None, current: str | None, field: str) -> str | None:
    if not masked:
        return masked
    parsed = urlsplit(masked)
    if _encoded_url_password(masked) == SECRET_PLACEHOLDER:
        current_password = _encoded_url_password(current)
        if current_password is None:
            raise ValueError(f"{field} contains a masked secret without an existing value")
        userinfo, _separator, host = parsed.netloc.rpartition("@")
        username, _masked = userinfo.split(":", 1)
        parsed = parsed._replace(netloc=f"{username}:{current_password}@{host}")
    existing = _query_secrets(urlsplit(current or "").query)
    occurrences: dict[str, int] = {}
    parts = []
    for part in parsed.query.split("&"):
        key, separator, value = part.partition("=")
        if separator and _url_secret_key(key):
            normalized = unquote_plus(key).casefold()
            index = occurrences.get(normalized, 0)
            occurrences[normalized] = index + 1
            if unquote_plus(value) == SECRET_PLACEHOLDER:
                values = existing.get(normalized, [])
                if index >= len(values) or not values[index]:
                    raise ValueError(f"{field} contains a masked secret without an existing value")
                part = f"{key}={values[index]}"
        parts.append(part)
    return urlunsplit(parsed._replace(query="&".join(parts)))


def _url_uses_masked_secret(value: str | None) -> bool:
    if not value:
        return False
    return _encoded_url_password(value) == SECRET_PLACEHOLDER or any(
        unquote_plus(secret) == SECRET_PLACEHOLDER
        for values in _query_secrets(urlsplit(value).query).values()
        for secret in values
    )


def _restore_secret(masked: str, current: str, field: str) -> str:
    if masked != SECRET_PLACEHOLDER:
        return masked
    if not current:
        raise ValueError(f"{field} contains a masked secret without an existing value")
    return current


def _camera_uses_masked_secret(camera: CameraConfig) -> bool:
    return any(
        (
            _url_uses_masked_secret(camera.stream_url),
            _url_uses_masked_secret(camera.live_stream_url),
            camera.onvif.password == SECRET_PLACEHOLDER,
        )
    )


def _restore_camera_secrets(incoming: CameraConfig, current: CameraConfig | None) -> CameraConfig:
    restored = incoming.model_copy(deep=True)
    if not _camera_uses_masked_secret(restored):
        return restored
    if current is None:
        raise ValueError("new cameras must provide their own credentials")
    restored.stream_url = str(
        _restore_url_password(restored.stream_url, current.stream_url, "stream_url")
        or ""
    )
    restored.live_stream_url = _restore_url_password(
        restored.live_stream_url,
        current.live_stream_url,
        "live_stream_url",
    )
    restored.onvif.password = _restore_secret(
        restored.onvif.password,
        current.onvif.password,
        "onvif.password",
    )
    return restored


def _camera_secret_identity_matches(incoming: CameraConfig, current: CameraConfig) -> bool:
    if _mask_url_password(incoming.stream_url) == _mask_url_password(current.stream_url):
        return True
    if incoming.live_stream_url and _mask_url_password(incoming.live_stream_url) == _mask_url_password(current.live_stream_url):
        return True
    return bool(
        incoming.onvif.host
        and incoming.onvif.host == current.onvif.host
        and incoming.onvif.username == current.onvif.username
    )


def restore_config_secrets(incoming: AppConfig, current: AppConfig) -> AppConfig:
    restored = incoming.model_copy(deep=True)
    restored.mqtt.password = _restore_secret(
        restored.mqtt.password,
        current.mqtt.password,
        "mqtt.password",
    )
    restored.audit_ai.api_key = _restore_secret(
        restored.audit_ai.api_key,
        current.audit_ai.api_key,
        "audit_ai.api_key",
    )
    current_api_tokens = {token.id: token for token in current.api_auth.tokens}
    for token in restored.api_auth.tokens:
        if token.token_hash != SECRET_PLACEHOLDER:
            continue
        existing = current_api_tokens.get(token.id)
        if existing is None:
            raise ValueError("new API tokens must provide their own token hash")
        token.token_hash = existing.token_hash
    if restored.web_auth.session_key == SECRET_PLACEHOLDER:
        restored.web_auth.session_key = current.web_auth.session_key
    # Browser configuration snapshots cannot undo logout/revocation state.
    restored.web_auth.revoked_sessions = dict(current.web_auth.revoked_sessions)
    current_users = {user.id: user for user in current.web_auth.users}
    for user in restored.web_auth.users:
        existing_user = current_users.get(user.id)
        if user.password_hash == SECRET_PLACEHOLDER:
            if existing_user is None:
                raise ValueError("new web users must be created through Access")
            user.password_hash = existing_user.password_hash
        if existing_user is not None:
            user.session_epoch = existing_user.session_epoch
    current_by_id = {camera.id: camera for camera in current.cameras}
    same_shape = len(restored.cameras) == len(current.cameras)
    restored.cameras = [
        _restore_camera_secrets(
            camera,
            current_by_id.get(camera.id)
            or (
                current.cameras[index]
                if same_shape
                and _camera_secret_identity_matches(camera, current.cameras[index])
                else None
            ),
        )
        for index, camera in enumerate(restored.cameras)
    ]
    restored = AppConfig.model_validate(restored.model_dump(mode="json"))
    return ensure_web_auth_session_key(restored)


def ensure_web_auth_session_key(config: AppConfig) -> AppConfig:
    if not config.web_auth.enabled:
        return config
    if config.web_auth.session_key and config.web_auth.session_key != SECRET_PLACEHOLDER:
        return config
    next_config = config.model_copy(deep=True)
    next_config.web_auth.session_key = secrets.token_hex(32)
    return next_config


def redacted_camera_payload(camera: CameraConfig) -> dict:
    payload = camera.model_dump(mode="json")
    payload["stream_url"] = _mask_url_password(camera.stream_url)
    payload["live_stream_url"] = _mask_url_password(camera.live_stream_url)
    payload["onvif"]["password"] = SECRET_PLACEHOLDER if camera.onvif.password else ""
    return payload


def redacted_config_payload(config: AppConfig) -> dict:
    payload = config.model_dump(mode="json")
    payload["mqtt"]["password"] = SECRET_PLACEHOLDER if config.mqtt.password else ""
    payload["audit_ai"]["api_key"] = SECRET_PLACEHOLDER if config.audit_ai.api_key else ""
    for token in payload["api_auth"]["tokens"]:
        token["token_hash"] = SECRET_PLACEHOLDER
    if payload["web_auth"].get("session_key"):
        payload["web_auth"]["session_key"] = SECRET_PLACEHOLDER
    payload["web_auth"].pop("revoked_sessions", None)
    for user in payload["web_auth"]["users"]:
        user["password_hash"] = SECRET_PLACEHOLDER
    payload["cameras"] = [redacted_camera_payload(camera) for camera in config.cameras]
    return payload


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _masked_probe_credentials_match(
    camera: CameraConfig,
    *,
    host: str,
    username: str,
    onvif_port: int,
) -> bool:
    """Only reuse a stored camera password for that camera's own endpoint.

    The probe is intentionally able to reach a camera that is not yet in the
    configuration, but the secret placeholder must never turn it into a way to
    forward an already-stored credential to an arbitrary host.
    """
    return (
        bool(camera.onvif.password)
        and host.casefold() == camera.onvif.host.strip().casefold()
        and username == camera.onvif.username
        and onvif_port == camera.onvif.port
    )


def create_config_router(deps: ConfigRouteDependencies) -> APIRouter:
    router = APIRouter()

    @router.get("/api/config")
    def get_config() -> dict:
        return redacted_config_payload(deps.get_config())

    @router.put("/api/config")
    def put_config(next_config: AppConfig) -> dict:
        # Secret restoration and application must observe the same active
        # generation. The application callback uses this same re-entrant lock.
        with deps.lock:
            try:
                next_config = restore_config_secrets(next_config, deps.get_config())
                deps.validate_config(next_config)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            try:
                effective, result = deps.apply_config(next_config, assign_ids=False)
            except (OSError, ValueError) as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        return {"ok": True, "cameras": len(effective.cameras), **result}

    @router.get("/api/config/api-tokens")
    def list_api_tokens() -> dict[str, Any]:
        auth = deps.get_config().api_auth
        return {
            "enabled": auth.enabled,
            "tokens": [
                {"id": token.id, "name": token.name, "scopes": token.scopes}
                for token in auth.tokens
            ],
        }

    @router.post("/api/config/api-tokens", status_code=201)
    def create_api_token(request: ApiTokenCreateRequest) -> dict[str, Any]:
        with deps.lock:
            current = deps.get_config()
            if any(token.id == request.id for token in current.api_auth.tokens):
                raise HTTPException(status_code=409, detail="API token id already exists")
            raw_token = f"survng_{secrets.token_urlsafe(32)}"
            next_config = current.model_copy(deep=True)
            next_config.api_auth.tokens.append(ApiTokenConfig(
                id=request.id,
                name=request.name,
                token_hash=hash_api_token(raw_token),
                scopes=request.scopes,
            ))
            effective, result = deps.apply_config(next_config, assign_ids=False)
        return {
            "token": raw_token,
            "credential": {
                "id": request.id,
                "name": request.name,
                "scopes": request.scopes,
            },
            "enabled": effective.api_auth.enabled,
            **result,
        }

    @router.delete("/api/config/api-tokens/{token_id}")
    def delete_api_token(token_id: str) -> dict[str, Any]:
        with deps.lock:
            current = deps.get_config()
            retained = [token for token in current.api_auth.tokens if token.id != token_id]
            if len(retained) == len(current.api_auth.tokens):
                raise HTTPException(status_code=404, detail="API token not found")
            next_config = current.model_copy(deep=True)
            next_config.api_auth.tokens = retained
            if not retained:
                next_config.api_auth.enabled = False
            effective, result = deps.apply_config(next_config, assign_ids=False)
        return {
            "ok": True,
            "id": token_id,
            "enabled": effective.api_auth.enabled,
            **result,
        }

    @router.put("/api/config/cameras/{camera_id}/zones")
    def put_camera_zones(camera_id: str, zones: list[DetectionZone]) -> dict:
        with deps.lock:
            current = deps.get_config()
            runtime = deps.get_manager()
            next_config = current.model_copy(deep=True)
            camera = camera_by_id(next_config, camera_id)
            if camera is None:
                raise HTTPException(status_code=404, detail="camera not found")
            if runtime.workers.get(camera_id) is None:
                raise HTTPException(status_code=404, detail="camera worker not found")
            previous = camera_by_id(current, camera_id)
            previous_zones = [zone.model_dump(mode="json") for zone in (previous.zones if previous else [])]
            camera.zones = zones
            next_payload = [zone.model_dump(mode="json") for zone in zones]
            try:
                runtime.update_camera_zones(camera_id, zones, previous_zones)
                deps.save_config(next_config, assign_ids=False)
            except Exception:
                try:
                    runtime.update_camera_zones(camera_id, [DetectionZone.model_validate(zone) for zone in previous_zones], next_payload)
                except Exception:
                    LOGGER.exception("failed to roll back camera zones for %s", camera_id)
                raise
            deps.publish_config(next_config)
        return {"ok": True, "camera_id": camera.id, "zones": next_payload, "workers_restarted": False}

    @router.put("/api/config/cameras/order")
    def put_camera_order(camera_ids: list[str]) -> dict:
        with deps.lock:
            current = deps.get_config()
            existing_ids = [camera.id for camera in current.cameras]
            if len(camera_ids) != len(existing_ids) or len(set(camera_ids)) != len(camera_ids):
                raise HTTPException(status_code=400, detail="camera order must contain every camera exactly once")
            if set(camera_ids) != set(existing_ids):
                raise HTTPException(status_code=400, detail="camera order does not match configured cameras")
            by_id = {camera.id: camera for camera in current.cameras}
            next_config = current.model_copy(deep=True)
            next_config.cameras = [by_id[camera_id].model_copy(deep=True) for camera_id in camera_ids]
            runtime = deps.get_manager()
            try:
                reordered_workers = {
                    camera_id: runtime.workers[camera_id]
                    for camera_id in camera_ids
                }
            except KeyError as error:
                raise HTTPException(
                    status_code=409,
                    detail=f"camera worker is unavailable: {error.args[0]}",
                ) from None
            deps.save_config(next_config, assign_ids=False)
            runtime.workers = reordered_workers
            deps.publish_config(next_config)
        return {"ok": True, "camera_ids": camera_ids}

    @router.put("/api/config/cameras/{camera_id}")
    def put_camera(camera_id: str, camera_settings: CameraConfig) -> dict:
        with deps.lock:
            next_config = deps.get_config().model_copy(deep=True)
            index = next((i for i, item in enumerate(next_config.cameras) if item.id == camera_id), None)
            existing = next_config.cameras[index] if index is not None else None
            try:
                camera_settings = _restore_camera_secrets(camera_settings, existing)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            used = {item.id for item in next_config.cameras if item.id != camera_id}
            # Existing IDs own recordings, incidents, zones and route references.
            # A display-name edit must not create a new persistent identity.
            base_id = existing.id if existing is not None else slugify_camera_id(camera_settings.name or camera_settings.id)
            next_id, suffix = base_id, 2
            while next_id in used:
                next_id, suffix = f"{base_id}-{suffix}", suffix + 1
            camera_settings.id = next_id
            camera_settings.zones = existing.zones if existing is not None else []
            if index is None:
                next_config.cameras.append(camera_settings)
            else:
                next_config.cameras[index] = camera_settings
            try:
                deps.validate_config(next_config)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            _effective, result = deps.apply_config(next_config)
        return {"ok": True, "camera": redacted_camera_payload(camera_settings), **result}

    @router.delete("/api/config/cameras/{camera_id}")
    def delete_camera(camera_id: str) -> dict:
        with deps.lock:
            next_config = deps.get_config().model_copy(deep=True)
            remaining = [camera for camera in next_config.cameras if camera.id != camera_id]
            if len(remaining) == len(next_config.cameras):
                raise HTTPException(status_code=404, detail="camera not found")
            next_config.cameras = remaining
            deps.reload_manager(next_config)
        return {"ok": True, "camera_id": camera_id}

    @router.post("/api/config/probe")
    def probe_config(payload: ConfigProbeRequest) -> dict:
        if not deps.probe_limiter.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="too many camera probes are already running",
                headers={"Retry-After": "2"},
            )
        try:
            host, username, password = payload.host.strip(), payload.username.strip(), payload.password
            if password == SECRET_PLACEHOLDER:
                existing = camera_by_id(deps.get_config(), payload.camera_id)
                if existing is None:
                    raise HTTPException(status_code=422, detail="masked probe credentials require an existing camera")
                if not _masked_probe_credentials_match(
                    existing,
                    host=host,
                    username=username,
                    onvif_port=payload.onvif_port,
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="masked probe credentials may only be used with the configured camera endpoint",
                    )
                password = existing.onvif.password
            result: dict[str, Any] = {
                "host": host,
                "onvif": {
                    "port": payload.onvif_port,
                    "reachable": _tcp_reachable(host, payload.onvif_port),
                    "capabilities": {},
                    "error": "",
                },
            }
            if result["onvif"]["reachable"] and username and password:
                try:
                    from onvif import ONVIFCamera
                    from zeep import Transport
                    camera = ONVIFCamera(
                        host,
                        payload.onvif_port,
                        username,
                        password,
                        transport=Transport(operation_timeout=5),
                    )
                    capabilities = camera.create_devicemgmt_service().GetCapabilities(
                        {"Category": "All"}
                    )
                    result["onvif"]["capabilities"] = {
                        name.lower(): bool(getattr(capabilities, name, None))
                        for name in ("Media", "Events", "PTZ", "Analytics")
                    }
                except Exception as exc:
                    result["onvif"]["error"] = redact_secret_text(str(exc) or "ONVIF capability probe failed")[:500]
            return result
        finally:
            deps.probe_limiter.release()

    return router

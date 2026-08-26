"""Browser sign-in and local user administration."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import AppConfig, WebRole, WebUserConfig
from .security import (
    SESSION_COOKIE_NAME,
    authenticate_password,
    encode_session,
    hash_password,
    public_user_payload,
    session_ttl_seconds,
)

USERNAME_PATTERN = r"^[A-Za-z][A-Za-z0-9._-]{2,63}$"
_LOGIN_WINDOW_SECONDS = 15 * 60
_LOGIN_MAX_FAILURES = 8
_LOGIN_FAILURES: dict[str, list[float]] = {}
_LOGIN_FAILURE_LOCK = threading.Lock()


class AuthCredentials(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=USERNAME_PATTERN)
    password: str = Field(min_length=8, max_length=1024)
    display_name: str = Field(default="", max_length=128)


class UserCreateRequest(AuthCredentials):
    role: WebRole = "viewer"


class UserUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=128)
    role: WebRole | None = None


class PasswordChangeRequest(BaseModel):
    password: str = Field(min_length=8, max_length=1024)


class WebAuthSettingsRequest(BaseModel):
    enabled: bool


@dataclass(frozen=True, slots=True)
class AuthRouteDependencies:
    get_config: Callable[[], AppConfig]
    apply_config: Callable[..., tuple[AppConfig, dict[str, object]]]
    lock: threading.RLock


def _client_key(request: Request, username: str) -> str:
    host = request.client.host if request.client else "unknown"
    return f"{host}:{username.strip().casefold()}"


def _login_blocked(key: str) -> bool:
    now = time.monotonic()
    with _LOGIN_FAILURE_LOCK:
        stamps = [stamp for stamp in _LOGIN_FAILURES.get(key, []) if now - stamp < _LOGIN_WINDOW_SECONDS]
        _LOGIN_FAILURES[key] = stamps
        return len(stamps) >= _LOGIN_MAX_FAILURES


def _record_login_failure(key: str) -> None:
    now = time.monotonic()
    with _LOGIN_FAILURE_LOCK:
        stamps = [stamp for stamp in _LOGIN_FAILURES.get(key, []) if now - stamp < _LOGIN_WINDOW_SECONDS]
        stamps.append(now)
        _LOGIN_FAILURES[key] = stamps


def _clear_login_failures(key: str) -> None:
    with _LOGIN_FAILURE_LOCK:
        _LOGIN_FAILURES.pop(key, None)


def _request_is_secure(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return request.url.scheme == "https" or forwarded == "https"


def _attach_session_cookie(response: Response, request: Request, token: str, *, ttl_seconds: int) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=ttl_seconds,
        httponly=True,
        samesite="lax",
        secure=_request_is_secure(request),
        path="/",
    )


def _clear_session_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=_request_is_secure(request),
    )


def _ensure_session_key(config: AppConfig) -> AppConfig:
    if config.web_auth.session_key and config.web_auth.session_key != "__SURVNG_SECRET_SET__":
        return config
    next_config = config.model_copy(deep=True)
    next_config.web_auth.session_key = secrets.token_hex(32)
    return next_config


def _new_user_id(existing: list[WebUserConfig], username: str) -> str:
    base = username.strip().casefold()
    used = {user.id for user in existing}
    if base not in used:
        return base
    index = 2
    while f"{base}-{index}" in used:
        index += 1
    return f"{base}-{index}"


def _admin_count(users: list[WebUserConfig], *, excluding: str = "") -> int:
    return sum(1 for user in users if user.role == "admin" and user.id != excluding)


def session_payload(config: AppConfig, user: WebUserConfig | None) -> dict[str, Any]:
    return {
        "enabled": config.web_auth.enabled,
        "bootstrap_required": bool(config.web_auth.enabled) and not config.web_auth.users,
        "user": public_user_payload(user) if user is not None else None,
    }


def create_auth_router(deps: AuthRouteDependencies) -> APIRouter:
    router = APIRouter()

    def current_user(request: Request) -> WebUserConfig | None:
        principal = request.scope.get("survng_principal")
        if principal is None or getattr(principal, "kind", None) != "user":
            return None
        return next(
            (user for user in deps.get_config().web_auth.users if user.id == principal.token_id),
            None,
        )

    def require_admin(request: Request) -> WebUserConfig | None:
        config = deps.get_config()
        principal = request.scope.get("survng_principal")
        if not config.web_auth.enabled and not config.api_auth.enabled:
            if principal is not None and getattr(principal, "kind", None) == "user":
                return next(
                    (user for user in config.web_auth.users if user.id == principal.token_id),
                    None,
                )
            return None
        if principal is None:
            raise HTTPException(status_code=401, detail="sign-in required")
        if not principal.permits("admin"):
            raise HTTPException(status_code=403, detail="administrator access required")
        if getattr(principal, "kind", None) == "user":
            return next(
                (user for user in config.web_auth.users if user.id == principal.token_id),
                None,
            )
        return None

    @router.get("/api/auth/session")
    def get_session(request: Request) -> dict[str, Any]:
        return session_payload(deps.get_config(), current_user(request))

    @router.post("/api/auth/login")
    def login(request: Request, body: AuthCredentials) -> JSONResponse:
        config = deps.get_config()
        if not config.web_auth.enabled:
            raise HTTPException(status_code=404, detail="sign-in is not enabled")
        key = _client_key(request, body.username)
        if _login_blocked(key):
            raise HTTPException(status_code=429, detail="too many failed sign-in attempts; try again later")
        user = authenticate_password(body.username, body.password, config.web_auth)
        if user is None:
            _record_login_failure(key)
            raise HTTPException(status_code=401, detail="invalid username or password")
        _clear_login_failures(key)
        ttl = session_ttl_seconds(config.web_auth.session_days)
        token = encode_session(user.id, config.web_auth.session_key, ttl_seconds=ttl)
        response = JSONResponse(session_payload(config, user))
        _attach_session_cookie(response, request, token, ttl_seconds=ttl)
        return response

    @router.post("/api/auth/logout")
    def logout(request: Request) -> JSONResponse:
        response = JSONResponse({"ok": True})
        _clear_session_cookie(response, request)
        return response

    @router.post("/api/auth/bootstrap", status_code=201)
    def bootstrap(request: Request, body: AuthCredentials) -> JSONResponse:
        with deps.lock:
            current = deps.get_config()
            if current.web_auth.users:
                raise HTTPException(status_code=409, detail="an administrator already exists")
            next_config = _ensure_session_key(current.model_copy(deep=True))
            next_config.web_auth.users.append(WebUserConfig(
                id=_new_user_id([], body.username),
                username=body.username,
                display_name=body.display_name.strip() or body.username,
                role="admin",
                password_hash=hash_password(body.password),
            ))
            next_config.web_auth.enabled = True
            effective, result = deps.apply_config(next_config, assign_ids=False)
        user = effective.web_auth.users[0]
        ttl = session_ttl_seconds(effective.web_auth.session_days)
        token = encode_session(user.id, effective.web_auth.session_key, ttl_seconds=ttl)
        payload = session_payload(effective, user)
        payload.update(result)
        response = JSONResponse(payload, status_code=201)
        _attach_session_cookie(response, request, token, ttl_seconds=ttl)
        return response

    @router.get("/api/auth/users")
    def list_users(request: Request) -> dict[str, Any]:
        require_admin(request)
        auth = deps.get_config().web_auth
        return {
            "enabled": auth.enabled,
            "users": [public_user_payload(user) for user in auth.users],
        }

    @router.put("/api/auth/settings")
    def put_settings(request: Request, body: WebAuthSettingsRequest) -> dict[str, Any]:
        require_admin(request)
        with deps.lock:
            current = deps.get_config()
            next_config = _ensure_session_key(current.model_copy(deep=True))
            next_config.web_auth.enabled = body.enabled
            try:
                effective, result = deps.apply_config(next_config, assign_ids=False)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "ok": True,
            "enabled": effective.web_auth.enabled,
            "users": [public_user_payload(user) for user in effective.web_auth.users],
            **result,
        }

    @router.post("/api/auth/users", status_code=201)
    def create_user(request: Request, body: UserCreateRequest) -> dict[str, Any]:
        require_admin(request)
        with deps.lock:
            current = deps.get_config()
            if find_username(current, body.username) is not None:
                raise HTTPException(status_code=409, detail="username already exists")
            next_config = _ensure_session_key(current.model_copy(deep=True))
            user = WebUserConfig(
                id=_new_user_id(next_config.web_auth.users, body.username),
                username=body.username,
                display_name=body.display_name.strip() or body.username,
                role=body.role,
                password_hash=hash_password(body.password),
            )
            next_config.web_auth.users.append(user)
            effective, result = deps.apply_config(next_config, assign_ids=False)
        created = next(item for item in effective.web_auth.users if item.id == user.id)
        return {"user": public_user_payload(created), **result}

    @router.patch("/api/auth/users/{user_id}")
    def update_user(request: Request, user_id: str, body: UserUpdateRequest) -> dict[str, Any]:
        require_admin(request)
        with deps.lock:
            current = deps.get_config()
            next_config = current.model_copy(deep=True)
            user = next((item for item in next_config.web_auth.users if item.id == user_id), None)
            if user is None:
                raise HTTPException(status_code=404, detail="user not found")
            last_admin = (
                body.role == "viewer"
                and user.role == "admin"
                and _admin_count(next_config.web_auth.users, excluding=user.id) < 1
            )
            if last_admin and next_config.web_auth.enabled:
                raise HTTPException(
                    status_code=409,
                    detail="turn off sign-in before demoting the last administrator",
                )
            if body.display_name is not None:
                user.display_name = body.display_name.strip() or user.username
            if body.role is not None:
                user.role = body.role
            effective, result = deps.apply_config(next_config, assign_ids=False)
        updated = next(item for item in effective.web_auth.users if item.id == user_id)
        return {"user": public_user_payload(updated), **result}

    @router.put("/api/auth/users/{user_id}/password")
    def change_password(request: Request, user_id: str, body: PasswordChangeRequest) -> dict[str, Any]:
        require_admin(request)
        with deps.lock:
            current = deps.get_config()
            next_config = current.model_copy(deep=True)
            user = next((item for item in next_config.web_auth.users if item.id == user_id), None)
            if user is None:
                raise HTTPException(status_code=404, detail="user not found")
            user.password_hash = hash_password(body.password)
            effective, result = deps.apply_config(next_config, assign_ids=False)
        return {"ok": True, "id": user_id, **result}

    @router.delete("/api/auth/users/{user_id}")
    def delete_user(request: Request, user_id: str) -> JSONResponse:
        actor = require_admin(request)
        with deps.lock:
            current = deps.get_config()
            user = next((item for item in current.web_auth.users if item.id == user_id), None)
            if user is None:
                raise HTTPException(status_code=404, detail="user not found")
            remaining_admins = _admin_count(current.web_auth.users, excluding=user.id)
            deleting_self = actor is not None and actor.id == user_id
            if deleting_self and remaining_admins >= 1 and current.web_auth.enabled:
                raise HTTPException(status_code=409, detail="you cannot delete your own account")
            next_config = current.model_copy(deep=True)
            next_config.web_auth.users = [item for item in next_config.web_auth.users if item.id != user_id]
            if remaining_admins < 1:
                next_config.web_auth.enabled = False
            effective, result = deps.apply_config(next_config, assign_ids=False)
        payload = {
            "ok": True,
            "id": user_id,
            "enabled": effective.web_auth.enabled,
            "users": [public_user_payload(item) for item in effective.web_auth.users],
            **result,
        }
        response = JSONResponse(payload)
        if deleting_self:
            _clear_session_cookie(response, request)
        return response

    return router


def find_username(config: AppConfig, username: str) -> WebUserConfig | None:
    wanted = username.strip().casefold()
    return next((user for user in config.web_auth.users if user.username.casefold() == wanted), None)

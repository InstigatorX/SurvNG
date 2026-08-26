"""HTTPS certificate management for administrators."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import AppConfig
from .tls import (
    generate_self_signed_certificate,
    parse_certificate_pem,
    parse_private_key_pem,
    tls_files_present,
    tls_status,
    write_tls_material,
)


class TlsSettingsRequest(BaseModel):
    enabled: bool
    hostname: str = Field(default="", max_length=255)
    port: int = Field(default=0, ge=0, le=65535)


class SelfSignedRequest(BaseModel):
    hostname: str = Field(default="", max_length=255)


class TlsPemUploadRequest(BaseModel):
    certificate_pem: str = Field(min_length=32, max_length=256_000)
    private_key_pem: str = Field(min_length=32, max_length=256_000)


@dataclass(frozen=True, slots=True)
class TlsRouteDependencies:
    get_config: Callable[[], AppConfig]
    apply_config: Callable[..., tuple[AppConfig, dict[str, object]]]
    request_server_restart: Callable[[], dict[str, object]]
    lock: threading.RLock


def create_tls_router(deps: TlsRouteDependencies) -> APIRouter:
    router = APIRouter()

    @router.get("/api/tls")
    def get_tls() -> dict[str, Any]:
        return tls_status(deps.get_config())

    @router.put("/api/tls")
    def put_tls(body: TlsSettingsRequest) -> dict[str, Any]:
        with deps.lock:
            current = deps.get_config()
            next_config = current.model_copy(deep=True)
            next_config.tls.enabled = body.enabled
            next_config.tls.hostname = body.hostname.strip()
            next_config.tls.port = body.port
            if next_config.tls.enabled and not tls_files_present(next_config):
                try:
                    generate_self_signed_certificate(next_config, next_config.tls.hostname)
                except Exception as error:
                    raise HTTPException(status_code=422, detail=str(error)) from error
            try:
                effective, result = deps.apply_config(next_config, assign_ids=False)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        return {"ok": True, "restart_required": True, **tls_status(effective), **result}

    @router.post("/api/tls/self-signed")
    def create_self_signed(body: SelfSignedRequest) -> dict[str, Any]:
        with deps.lock:
            current = deps.get_config()
            try:
                status = generate_self_signed_certificate(current, body.hostname)
            except Exception as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            if body.hostname.strip() and body.hostname.strip() != current.tls.hostname:
                next_config = current.model_copy(deep=True)
                next_config.tls.hostname = body.hostname.strip()
                effective, result = deps.apply_config(next_config, assign_ids=False)
                status = tls_status(effective)
                status.update(result)
        return {"ok": True, "restart_required": current.tls.enabled, **status}

    def install_pem(certificate_pem: str, private_key_pem: str) -> dict[str, Any]:
        try:
            cert_pem = parse_certificate_pem(certificate_pem)
            key_pem = parse_private_key_pem(private_key_pem)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        with deps.lock:
            current = deps.get_config()
            try:
                write_tls_material(current, cert_pem, key_pem)
            except Exception as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        return {"ok": True, "restart_required": current.tls.enabled, **tls_status(current)}

    @router.post("/api/tls/upload")
    async def upload_certificate(
        certificate: UploadFile = File(...),
        private_key: UploadFile = File(...),
    ) -> dict[str, Any]:
        try:
            cert_text = (await certificate.read()).decode("utf-8")
            key_text = (await private_key.read()).decode("utf-8")
        except UnicodeDecodeError as error:
            raise HTTPException(status_code=422, detail="certificate files must be UTF-8 PEM text") from error
        return install_pem(cert_text, key_text)

    @router.post("/api/tls/certificate")
    def upload_certificate_pem(body: TlsPemUploadRequest) -> dict[str, Any]:
        return install_pem(body.certificate_pem, body.private_key_pem)

    @router.post("/api/tls/apply")
    def apply_tls() -> dict[str, Any]:
        try:
            restart = deps.request_server_restart()
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"ok": True, "restarting": True, **restart}

    return router

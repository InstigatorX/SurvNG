"""Trusted reverse-proxy headers for HTTPS cookies, client IP, and HSTS."""

from __future__ import annotations

import ipaddress
from typing import Any

from pydantic import BaseModel, Field, field_validator

DEFAULT_TRUSTED_PROXIES = ("127.0.0.1", "::1")


class ProxyConfig(BaseModel):
    trusted_proxies: list[str] = Field(default_factory=lambda: list(DEFAULT_TRUSTED_PROXIES))

    @field_validator("trusted_proxies")
    @classmethod
    def normalize_trusted_proxies(cls, value: list[str]) -> list[str]:
        networks: list[str] = []
        for raw in value:
            item = str(raw).strip()
            if not item:
                continue
            if item == "*":
                networks.append("*")
                continue
            try:
                ipaddress.ip_network(item, strict=False)
            except ValueError as error:
                raise ValueError(f"trusted proxy must be an IP, CIDR, or *: {item}") from error
            networks.append(item)
        return networks


def parse_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if "%" in text:
        text = text.split("%", 1)[0]
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        return None


def ip_is_trusted(ip: str, trusted_proxies: list[str]) -> bool:
    address = parse_ip(ip)
    if address is None:
        return False
    for item in trusted_proxies:
        if item.strip() == "*":
            return True
        try:
            network = ipaddress.ip_network(item.strip(), strict=False)
        except ValueError:
            continue
        if address in network:
            return True
    return False


def ip_is_local(ip: str) -> bool:
    text = str(ip or "").strip().lower()
    if text in {"", "testclient", "localhost"}:
        return True
    address = parse_ip(text)
    if address is None:
        return False
    return bool(
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
    )


def _scope_header(scope: dict[str, Any], name: bytes) -> str:
    for header_name, value in scope.get("headers", []):
        if header_name.lower() == name:
            return value.decode("latin-1").strip()
    return ""


def _peer_ip(scope: dict[str, Any]) -> str:
    client = scope.get("client")
    if isinstance(client, (tuple, list)) and client:
        return str(client[0] or "")
    return ""


def apply_trusted_proxy_headers(scope: dict[str, Any], trusted_proxies: list[str]) -> dict[str, Any]:
    """Honor X-Forwarded-* only when the immediate peer is a configured proxy."""
    peer = _peer_ip(scope)
    if not ip_is_trusted(peer, trusted_proxies):
        return scope
    next_scope = dict(scope)
    proto = _scope_header(scope, b"x-forwarded-proto").split(",")[0].strip().lower()
    if proto in {"http", "https"}:
        next_scope["scheme"] = proto
    forwarded_for = _scope_header(scope, b"x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
        if parse_ip(client_ip) is not None:
            port = 0
            client = scope.get("client")
            if isinstance(client, (tuple, list)) and len(client) > 1:
                try:
                    port = int(client[1])
                except (TypeError, ValueError):
                    port = 0
            next_scope["client"] = (client_ip, port)
    return next_scope


def request_is_secure(scope_or_scheme: dict[str, Any] | str) -> bool:
    if isinstance(scope_or_scheme, str):
        scheme = scope_or_scheme
    else:
        scheme = str(scope_or_scheme.get("scheme") or "")
    return scheme.lower() in {"https", "wss"}

"""Network bind policy: refuse open binds when authentication is off."""

from __future__ import annotations

import os
from typing import Any

OPEN_BIND_ENVIRONMENT_VARIABLE = "SURVNG_ALLOW_OPEN_BIND"


def host_is_loopback_only(host: str) -> bool:
    """Return True when the listen address cannot accept remote clients."""
    text = str(host or "").strip().lower()
    if text in {"127.0.0.1", "::1", "localhost"}:
        return True
    # Uvicorn treats these as "all interfaces"; they are never loopback-only.
    if text in {"0.0.0.0", "::", "[::]", "*"}:
        return False
    try:
        import ipaddress

        address = ipaddress.ip_address(text[1:-1] if text.startswith("[") and text.endswith("]") else text)
    except ValueError:
        # Hostnames are treated as potentially remote.
        return False
    return bool(address.is_loopback)


def auth_protects_network(config: Any) -> bool:
    api_auth = getattr(config, "api_auth", None)
    web_auth = getattr(config, "web_auth", None)
    return bool(getattr(api_auth, "enabled", False) or getattr(web_auth, "enabled", False))


def open_bind_override_enabled(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    value = str(env.get(OPEN_BIND_ENVIRONMENT_VARIABLE, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def assert_network_bind_allowed(host: str, config: Any, *, environ: dict[str, str] | None = None) -> None:
    """Raise SystemExit when an unauthenticated process would listen remotely."""
    if host_is_loopback_only(host):
        return
    if auth_protects_network(config):
        return
    if open_bind_override_enabled(environ):
        return
    raise SystemExit(
        "Refusing to bind "
        f"{host!r} while browser sign-in and API authentication are both disabled. "
        "Enable web_auth or api_auth in config, bind with --host 127.0.0.1, "
        f"or set {OPEN_BIND_ENVIRONMENT_VARIABLE}=1 for a trusted lab network."
    )

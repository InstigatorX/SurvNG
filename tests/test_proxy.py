from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

from survng.app.main import app
from survng.app.proxy import (
    apply_trusted_proxy_headers,
    ip_is_local,
    ip_is_trusted,
)


def test_ip_is_local_covers_loopback_and_private() -> None:
    assert ip_is_local("127.0.0.1")
    assert ip_is_local("::1")
    assert ip_is_local("10.0.0.5")
    assert ip_is_local("192.168.1.9")
    assert ip_is_local("testclient")
    assert not ip_is_local("8.8.8.8")
    assert not ip_is_local("1.1.1.1")


def test_trusted_proxy_match() -> None:
    assert ip_is_trusted("127.0.0.1", ["127.0.0.1", "::1"])
    assert ip_is_trusted("10.0.0.9", ["10.0.0.0/8"])
    assert ip_is_trusted("1.2.3.4", ["*"])
    assert not ip_is_trusted("1.2.3.4", ["127.0.0.1"])
    assert not ip_is_trusted("1.2.3.4", [])


def test_untrusted_peer_cannot_set_https_via_header() -> None:
    scope = {
        "type": "http",
        "scheme": "http",
        "client": ("203.0.113.9", 40000),
        "headers": [(b"x-forwarded-proto", b"https")],
    }
    apply_trusted_proxy_headers(scope, ["127.0.0.1", "::1"])
    assert scope["scheme"] == "http"


def test_trusted_peer_can_set_https_via_header() -> None:
    scope = {
        "type": "http",
        "scheme": "http",
        "client": ("127.0.0.1", 40000),
        "headers": [(b"x-forwarded-proto", b"https"), (b"x-forwarded-for", b"203.0.113.9")],
    }
    rewritten = apply_trusted_proxy_headers(scope, ["127.0.0.1"])
    assert rewritten["scheme"] == "https"
    assert rewritten["client"][0] == "203.0.113.9"


def test_forwarded_for_uses_first_untrusted_hop_from_right() -> None:
    scope = {
        "type": "http",
        "scheme": "http",
        "client": ("10.0.0.2", 40000),
        "headers": [
            # The leftmost value is attacker-controlled and must not win.
            (b"x-forwarded-for", b"192.168.1.9, 198.51.100.23, 10.0.0.1"),
        ],
    }
    rewritten = apply_trusted_proxy_headers(scope, ["10.0.0.0/8"])
    assert rewritten["client"][0] == "198.51.100.23"


def test_invalid_trusted_proxy_is_rejected() -> None:
    from pydantic import ValidationError

    from survng.app.proxy import ProxyConfig

    with pytest.raises(ValidationError):
        ProxyConfig(trusted_proxies=["not-an-ip"])


def test_server_disables_uvicorn_proxy_header_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    import survng.app.__main__ as server_main

    captured: dict[str, object] = {}
    monkeypatch.setattr(sys, "argv", ["survng", "--port", "8088"])
    monkeypatch.setattr(server_main, "load_config", lambda: object())
    monkeypatch.setattr(
        server_main,
        "uvicorn_tls_kwargs",
        lambda config, port: {"port": port},
    )
    monkeypatch.setattr(server_main.uvicorn, "run", lambda **kwargs: captured.update(kwargs))

    server_main.main()

    assert captured["proxy_headers"] is False


def test_hsts_only_when_scheme_is_https() -> None:
    client = TestClient(app)
    response = client.get("/api/health")
    assert "strict-transport-security" not in {
        k.lower() for k in response.headers.keys()
    }

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from survng.app.bind_policy import (
    OPEN_BIND_ENVIRONMENT_VARIABLE,
    assert_network_bind_allowed,
    auth_protects_network,
    host_is_loopback_only,
    open_bind_override_enabled,
)


def test_host_is_loopback_only() -> None:
    assert host_is_loopback_only("127.0.0.1")
    assert host_is_loopback_only("::1")
    assert host_is_loopback_only("localhost")
    assert not host_is_loopback_only("0.0.0.0")
    assert not host_is_loopback_only("::")
    assert not host_is_loopback_only("192.168.1.10")


def test_auth_protects_network() -> None:
    assert not auth_protects_network(SimpleNamespace(api_auth=SimpleNamespace(enabled=False), web_auth=SimpleNamespace(enabled=False)))
    assert auth_protects_network(SimpleNamespace(api_auth=SimpleNamespace(enabled=True), web_auth=SimpleNamespace(enabled=False)))
    assert auth_protects_network(SimpleNamespace(api_auth=SimpleNamespace(enabled=False), web_auth=SimpleNamespace(enabled=True)))


def test_open_bind_override_enabled() -> None:
    assert open_bind_override_enabled({OPEN_BIND_ENVIRONMENT_VARIABLE: "1"})
    assert open_bind_override_enabled({OPEN_BIND_ENVIRONMENT_VARIABLE: "true"})
    assert not open_bind_override_enabled({})
    assert not open_bind_override_enabled({OPEN_BIND_ENVIRONMENT_VARIABLE: "0"})


def test_assert_network_bind_allowed_blocks_open_unauthenticated_bind() -> None:
    config = SimpleNamespace(
        api_auth=SimpleNamespace(enabled=False),
        web_auth=SimpleNamespace(enabled=False),
    )
    with pytest.raises(SystemExit, match="Refusing to bind"):
        assert_network_bind_allowed("0.0.0.0", config, environ={})


def test_assert_network_bind_allowed_permits_loopback_auth_or_override() -> None:
    open_config = SimpleNamespace(
        api_auth=SimpleNamespace(enabled=False),
        web_auth=SimpleNamespace(enabled=False),
    )
    assert_network_bind_allowed("127.0.0.1", open_config, environ={})
    assert_network_bind_allowed(
        "0.0.0.0",
        SimpleNamespace(api_auth=SimpleNamespace(enabled=True), web_auth=SimpleNamespace(enabled=False)),
        environ={},
    )
    assert_network_bind_allowed(
        "0.0.0.0",
        open_config,
        environ={OPEN_BIND_ENVIRONMENT_VARIABLE: "1"},
    )


def test_server_main_enforces_bind_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    import survng.app.__main__ as server_main

    captured: dict[str, object] = {}
    monkeypatch.setattr(sys, "argv", ["survng", "--host", "0.0.0.0", "--port", "8088"])
    monkeypatch.setattr(
        server_main,
        "load_config",
        lambda: SimpleNamespace(
            api_auth=SimpleNamespace(enabled=False),
            web_auth=SimpleNamespace(enabled=False),
        ),
    )
    monkeypatch.setattr(
        server_main,
        "uvicorn_tls_kwargs",
        lambda config, port: {"port": port},
    )
    monkeypatch.setattr(server_main.uvicorn, "run", lambda **kwargs: captured.update(kwargs))

    with pytest.raises(SystemExit, match="Refusing to bind"):
        server_main.main()
    assert captured == {}

    monkeypatch.setenv(OPEN_BIND_ENVIRONMENT_VARIABLE, "1")
    server_main.main()
    assert captured["proxy_headers"] is False
    assert captured["host"] == "0.0.0.0"

"""Telemetry identity must not run git status on the request path."""

from __future__ import annotations

from unittest.mock import patch

from survng.app.system_telemetry import SystemTelemetryService


def test_product_version_does_not_call_git_status() -> None:
    with patch("survng.app.product_update.ProductUpdateService.status") as status:
        identity = SystemTelemetryService.product_version()
    status.assert_not_called()
    assert "deployment_mode" in identity

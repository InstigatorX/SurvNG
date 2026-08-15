from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from survng.app.system_telemetry import (
    DiagnosticSessionRequest,
    SystemTelemetryDependencies,
    SystemTelemetryService,
    create_system_telemetry_router,
)


class SystemTelemetryRouterTest(unittest.TestCase):
    @staticmethod
    def _event_store(marker: str) -> SimpleNamespace:
        return SimpleNamespace(
            tracking_capacity_activity=Mock(return_value=[]),
            lifecycle_events=Mock(return_value=[]),
        )

    @staticmethod
    def _telemetry_store(marker: str) -> SimpleNamespace:
        return SimpleNamespace(
            operational_history=Mock(return_value=[{"marker": marker}]),
            memory_history=Mock(return_value=[]),
            sample_times=Mock(return_value=[]),
        )

    def test_persisted_history_cache_is_invalidated_for_new_event_store(self) -> None:
        service = SystemTelemetryService()
        first_store = self._event_store("first")
        second_store = self._event_store("second")
        first_telemetry = self._telemetry_store("first")
        second_telemetry = self._telemetry_store("second")

        first = service.persisted_history(SimpleNamespace(events=first_store, telemetry=first_telemetry), "gate")
        second = service.persisted_history(SimpleNamespace(events=second_store, telemetry=second_telemetry), "gate")

        self.assertEqual(first["runtime"]["short"], [{"marker": "first"}])
        self.assertEqual(second["runtime"]["short"], [{"marker": "second"}])
        self.assertEqual(first_telemetry.operational_history.call_count, 2)
        self.assertEqual(second_telemetry.operational_history.call_count, 2)

    def test_telemetry_request_uses_one_manager_generation_and_its_config(self) -> None:
        first_config = object()
        second_config = object()
        managers = iter(
            (
                SimpleNamespace(config=first_config),
                SimpleNamespace(config=second_config),
            )
        )
        service = Mock()
        service.telemetry.side_effect = ({"generation": 1}, {"generation": 2})
        fallback_config = Mock()
        router = create_system_telemetry_router(
            SystemTelemetryDependencies(
                get_manager=lambda: next(managers),
                get_config=fallback_config,
            ),
            service,
        )
        endpoint = next(
            route.endpoint for route in router.routes if route.path == "/api/telemetry"
        )

        self.assertEqual(endpoint(hours=2, camera_id="gate"), {"generation": 1})
        self.assertEqual(endpoint(hours=24, camera_id=""), {"generation": 2})

        first_call, second_call = service.telemetry.call_args_list
        self.assertIs(first_call.args[1], first_config)
        self.assertEqual(first_call.kwargs, {"hours": 2, "camera_id": "gate"})
        self.assertIs(second_call.args[1], second_config)
        fallback_config.assert_not_called()

    def test_system_status_request_resolves_manager_once(self) -> None:
        active_manager = object()
        get_manager = Mock(return_value=active_manager)
        service = Mock()
        service.system_status.return_value = {"instance_id": "test"}
        router = create_system_telemetry_router(
            SystemTelemetryDependencies(
                get_manager=get_manager,
                get_config=Mock(),
            ),
            service,
        )
        endpoint = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/system/status"
        )

        self.assertEqual(endpoint(), {"instance_id": "test"})
        get_manager.assert_called_once_with()
        service.system_status.assert_called_once_with(active_manager)

    def test_diagnostic_routes_delegate_to_runtime_monitor(self) -> None:
        runtime_monitor = Mock()
        runtime_monitor.start_diagnostics.return_value = {"id": "session-1"}
        runtime_monitor.stop_diagnostics.return_value = True
        manager = SimpleNamespace(runtime_monitor=runtime_monitor)
        router = create_system_telemetry_router(
            SystemTelemetryDependencies(get_manager=lambda: manager, get_config=Mock()),
            Mock(),
        )
        routes = {(route.path, tuple(route.methods)): route.endpoint for route in router.routes}

        created = routes[("/api/telemetry/diagnostics", ("POST",))](
            DiagnosticSessionRequest(
                scope="camera", camera_id="gate", duration_seconds=900
            )
        )
        assert created == {"id": "session-1"}
        assert routes[("/api/telemetry/diagnostics/{session_id}", ("DELETE",))]("session-1") == {"stopped": True}

    def test_system_status_includes_lightweight_cpu_and_application_memory(self) -> None:
        manager = SimpleNamespace(
            storage_dir=Path("/tmp"),
            recorder=SimpleNamespace(
                retention_status=Mock(return_value={
                    "last_plan_at": "2026-08-12T22:00:00+00:00",
                    "plan": {"storage": {
                        "total_bytes": 100,
                        "used_bytes": 60,
                        "free_bytes": 40,
                    }},
                }),
            ),
            statuses=Mock(return_value=[]),
            detector_status=Mock(return_value={}),
            mqtt_status=Mock(return_value={}),
            go2rtc_status=Mock(return_value={}),
            camera_startup_status=Mock(return_value={}),
        )
        service = SystemTelemetryService()

        with (
            patch.object(
                SystemTelemetryService,
                "cgroup_memory_status",
                return_value={"application_bytes": 3_221_225_472},
            ),
            patch("survng.app.system_telemetry.os.cpu_count", return_value=8),
            patch("survng.app.system_telemetry.os.getloadavg", return_value=(2.0, 1.0, 0.5)),
            patch(
                "survng.app.system_telemetry.shutil.disk_usage",
                side_effect=AssertionError("system status must not touch storage"),
            ),
        ):
            status = service.system_status(manager)

        self.assertEqual(status["resources"]["application_memory_bytes"], 3_221_225_472)
        self.assertEqual(status["resources"]["cpu_load_percent"], 25.0)
        self.assertEqual(status["lifecycle"], "running")
        self.assertGreaterEqual(status["uptime_seconds"], 0.0)
        self.assertEqual(status["storage"]["free_bytes"], 40)
        self.assertTrue(status["storage"]["available"])


if __name__ == "__main__":
    unittest.main()

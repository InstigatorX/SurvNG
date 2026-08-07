from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from survng.app.system_telemetry import (
    SystemTelemetryDependencies,
    SystemTelemetryService,
    create_system_telemetry_router,
)


class SystemTelemetryRouterTest(unittest.TestCase):
    @staticmethod
    def _event_store(marker: str) -> SimpleNamespace:
        return SimpleNamespace(
            runtime_telemetry_history=Mock(return_value=[{"marker": marker}]),
            tracking_capacity_activity=Mock(return_value=[]),
        )

    def test_persisted_history_cache_is_invalidated_for_new_event_store(self) -> None:
        service = SystemTelemetryService()
        first_store = self._event_store("first")
        second_store = self._event_store("second")

        first = service.persisted_history(SimpleNamespace(events=first_store), "gate")
        second = service.persisted_history(SimpleNamespace(events=second_store), "gate")

        self.assertEqual(first["runtime"]["short"], [{"marker": "first"}])
        self.assertEqual(second["runtime"]["short"], [{"marker": "second"}])
        self.assertEqual(first_store.runtime_telemetry_history.call_count, 2)
        self.assertEqual(second_store.runtime_telemetry_history.call_count, 2)

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


if __name__ == "__main__":
    unittest.main()

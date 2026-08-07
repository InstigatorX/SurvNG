from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import HTTPException

from survng.app.incident_queries import (
    IncidentQueryDependencies,
    IncidentQueryService,
    create_incident_query_router,
)


class IncidentQueryRouterTest(unittest.TestCase):
    def test_feed_resolves_manager_while_generation_lock_is_held(self) -> None:
        class GenerationLock:
            held = False

            def __enter__(self) -> None:
                self.held = True

            def __exit__(self, *_args: object) -> None:
                self.held = False

        lock = GenerationLock()
        active_manager = object()

        def get_manager() -> object:
            self.assertTrue(lock.held)
            return active_manager

        service = Mock(spec=IncidentQueryService)
        service.feed.return_value = {"items": []}
        bundle = create_incident_query_router(
            IncidentQueryDependencies(get_manager=get_manager, manager_lock=lock),
            service,
        )

        response = bundle.handlers["incident_feed"](
            event_type="object",
            camera_id="gate",
            limit=12,
        )

        self.assertEqual(response, {"items": []})
        service.feed.assert_called_once_with(
            active_manager,
            event_type="object",
            camera_id="gate",
            object_label="",
            zone="",
            limit=12,
            offset=0,
            gap_seconds=45,
        )

    def test_by_event_not_found_is_decided_inside_query_boundary(self) -> None:
        service = Mock(spec=IncidentQueryService)
        service.resolve_event.return_value = None
        bundle = create_incident_query_router(
            IncidentQueryDependencies(
                get_manager=lambda: SimpleNamespace(),
                manager_lock=threading.RLock(),
            ),
            service,
        )

        with self.assertRaises(HTTPException) as raised:
            bundle.handlers["incident_for_event"](99)
        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()

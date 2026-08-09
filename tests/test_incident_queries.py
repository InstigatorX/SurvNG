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
    def test_face_enrichment_keeps_distinct_unknown_tracks(self) -> None:
        manager = SimpleNamespace(
            faces=SimpleNamespace(
                for_event_ids=lambda _ids: [
                    {
                        "observation_id": 11,
                        "event_id": 7,
                        "person_id": None,
                        "candidate_person_id": None,
                        "confidence": 0.8,
                        "consensus": {"candidate_count": 3},
                    },
                    {
                        "observation_id": 12,
                        "event_id": 7,
                        "person_id": None,
                        "candidate_person_id": None,
                        "confidence": 0.7,
                        "consensus": {"candidate_count": 2},
                    },
                ]
            )
        )
        incidents = [{"events": [{"id": 7}]}]

        result = IncidentQueryService.with_faces(manager, incidents)

        self.assertEqual(len(result[0]["faces"]), 2)
        self.assertEqual(
            {face["candidate_count"] for face in result[0]["faces"]},
            {2, 3},
        )

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

        def feed(*_args: object, **_kwargs: object) -> dict:
            self.assertFalse(lock.held)
            return {"items": []}

        service.feed.side_effect = feed
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

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
from survng.app.identity_projection import identity_summaries
from survng.app.manager_access import ManagerAccessCoordinator


class IncidentQueryRouterTest(unittest.TestCase):
    def test_search_keeps_full_day_facets_when_results_are_camera_filtered(self) -> None:
        rows = [
            {
                "id": 1,
                "camera_id": "gate",
                "kind": "object",
                "created_at": "2026-01-01T12:00:00+00:00",
                "snapshot_path": "gate.jpg",
                "recording_path": "",
                "objects_json": '[{"label":"person","confidence":0.9,"zones":["front"]}]',
            },
            {
                "id": 2,
                "camera_id": "garage",
                "kind": "object",
                "created_at": "2026-01-01T12:01:00+00:00",
                "snapshot_path": "garage.jpg",
                "recording_path": "",
                "objects_json": '[{"label":"vehicle","confidence":0.9,"zones":["drive"]}]',
            },
        ]
        calls: list[str] = []

        def between_compact(_start: str, _end: str, camera_id: str = ""):
            calls.append(camera_id)
            return [row for row in rows if not camera_id or row["camera_id"] == camera_id]

        manager = SimpleNamespace(
            events=SimpleNamespace(between_compact=between_compact),
            faces=SimpleNamespace(for_event_ids=lambda _ids: []),
        )

        result = IncidentQueryService.search(
            manager,
            day="2026-01-01",
            time_zone="UTC",
            camera_id="gate",
            event_type="all",
        )

        self.assertEqual([item["camera_id"] for item in result["items"]], ["gate"])
        self.assertEqual(result["facets"]["camera_ids"], ["garage", "gate"])
        self.assertEqual(result["facets"]["labels"], ["person", "vehicle"])
        self.assertEqual(result["facets"]["zones"], ["drive", "front"])
        self.assertEqual(calls, ["gate", ""])

    def test_confirmed_identity_wins_over_automatic_duplicate(self) -> None:
        identities = identity_summaries([
            {
                "identity_id": 3,
                "person_id": 3,
                "name": "Steve",
                "status": "automatic",
                "confidence": 0.99,
                "observation_id": 10,
            },
            {
                "identity_id": 3,
                "person_id": 3,
                "name": "Steve",
                "status": "confirmed",
                "confidence": 0.75,
                "observation_id": 11,
            },
        ])

        self.assertEqual(len(identities), 1)
        self.assertEqual(identities[0]["status"], "confirmed")

    def test_face_enrichment_preserves_automatic_identity_provenance(self) -> None:
        manager = SimpleNamespace(
            faces=SimpleNamespace(
                for_event_ids=lambda _ids: [
                    {
                        "observation_id": 10,
                        "event_id": 7,
                        "person_id": 3,
                        "person_name": "Steve",
                        "candidate_person_id": None,
                        "match_confidence": 0.88,
                        "review_status": "auto_identified",
                        "auto_identified": 1,
                        "consensus": {},
                    }
                ]
            )
        )

        result = IncidentQueryService.with_faces(
            manager,
            [{"events": [{"id": 7, "objects": []}]}],
        )

        identity = result[0]["primary_identity"]
        self.assertEqual(identity["name"], "Steve")
        self.assertEqual(identity["status"], "automatic")
        self.assertEqual(identity["review_status"], "auto_identified")
        self.assertEqual(identity["source"], "automatic")

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
            IncidentQueryDependencies(
                get_manager=get_manager,
                manager_lock=lock,
                manager_access=ManagerAccessCoordinator(),
            ),
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

from __future__ import annotations

import threading
import tempfile
import json
from pathlib import Path
from survng.app.events import EventStore
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
    def test_event_evidence_without_scene_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            event = store.add_event("garage", "motion", objects_json='[{"label":"person","confidence":0.9}]')
            with store._connect() as conn:
                conn.execute("delete from scene_event_membership where event_id=?", (event["id"],))
            self.assertIsNone(store.scene_incident(event_id=event["id"]))
            manager = SimpleNamespace(events=store, faces=SimpleNamespace(for_event_ids=lambda _ids: []))
            bundle = create_incident_query_router(IncidentQueryDependencies(
                get_manager=lambda: manager, manager_lock=threading.RLock(),
            ), IncidentQueryService())
            result = bundle.handlers["event_evidence"](event["id"])
            self.assertEqual(result["id"], event["id"])
            self.assertEqual(result["camera_id"], "garage")
            self.assertEqual(result["objects"][0]["label"], "person")
            with self.assertRaises(HTTPException) as raised:
                bundle.handlers["event_evidence"](event["id"] + 100)
            self.assertEqual(raised.exception.status_code, 404)
            self.assertIsNone(store.scene_incident(event_id=event["id"]))

    def test_recent_feed_pages_canonical_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            first = store.add_event("older", "motion", created_at="2026-07-30T00:00:00+00:00")
            newer = store.add_event("newer", "motion", created_at="2026-07-30T00:00:10+00:00")
            store.add_event("older", "motion", created_at="2026-07-30T00:00:20+00:00")
            manager = SimpleNamespace(events=store)
            page, more, scanned = IncidentQueryService.recent_filtered_summaries(manager, limit=1, offset=0, gap_seconds=5, event_type="all")
            self.assertEqual(page[0]["event_ids"], [newer["id"]])
            self.assertTrue(more)
            self.assertEqual(scanned[0]["camera_ids"], ["newer", "older"])
            self.assertEqual(store.scene_incident(event_id=first["id"])["event_count"], 2)

    def test_search_keeps_full_day_facets_when_results_are_camera_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            for camera, label, zone in (("gate", "person", "front"), ("garage", "vehicle", "drive")):
                store.add_event(camera, "object", created_at="2026-01-01T12:00:00+00:00", objects_json=json.dumps([{"label":label,"confidence":.9,"zones":[zone]}]))
            manager = SimpleNamespace(events=store, faces=SimpleNamespace(for_event_ids=lambda _ids: []))
            result = IncidentQueryService.search(manager, day="2026-01-01", time_zone="UTC", camera_id="gate", event_type="all")
            self.assertEqual([item["camera_id"] for item in result["items"]], ["gate"])
            self.assertEqual(result["facets"]["camera_ids"], ["garage", "gate"])
            self.assertEqual(result["facets"]["labels"], ["person", "vehicle"])
            self.assertEqual(result["facets"]["zones"], ["drive", "front"])

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


class NotificationIncidentTest(unittest.TestCase):
    def test_historical_link_resolves_alias_and_rejects_unknown_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            event = store.add_event("front-door", "motion")
            service = IncidentQueryService()
            manager = SimpleNamespace(events=store, faces=SimpleNamespace(for_event_ids=lambda _ids: []),
                incidents=SimpleNamespace(get=lambda _key: None),
                config=SimpleNamespace(cameras=[SimpleNamespace(id="front-door", name="Front Door")]))
            result = service.notification_detail(manager, f"incident-front-door-{event['id']}")
            self.assertEqual(result["camera_name"], "Front Door")
            self.assertEqual(result["incident_id"], store.scene_incident(event_id=event["id"])["id"])
            for key in ("invalid", f"incident-garage-{event['id']}", "incident-front-door-0"):
                with self.assertRaises(HTTPException) as error:
                    service.notification_detail(manager, key)
                self.assertEqual(error.exception.status_code, 404)

    def test_membership_survives_partial_event_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(Path(tmp))
            first = store.add_event("gate", "motion", created_at="2026-09-13T01:00:00+00:00")
            second = store.add_event("gate", "motion", created_at="2026-09-13T01:00:10+00:00")
            original = store.scene_incident(event_id=first["id"])["id"]
            with store._connect() as conn:
                conn.execute("delete from events where id=?", (first["id"],))
            result = store.scene_incident(f"incident-gate-{first['id']}")
            self.assertEqual(result["id"], original)
            self.assertEqual(result["event_ids"], [second["id"]])

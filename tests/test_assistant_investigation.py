from __future__ import annotations

import unittest

from survng.app.assistant_investigation import correlate_incident_timeline


def incident(
    event_id: int,
    camera_id: str,
    start_at: str,
    *,
    labels: tuple[str, ...] = ("person",),
    faces: tuple[dict, ...] = (),
) -> dict:
    return {
        "representative_event_id": event_id,
        "camera_id": camera_id,
        "start_at": start_at,
        "labels": list(labels),
        "faces": list(faces),
    }


class AssistantInvestigationTest(unittest.TestCase):
    def test_confirmed_identity_outranks_nearby_shared_class(self) -> None:
        anchor = incident(
            1,
            "gate",
            "2026-08-01T12:00:00+00:00",
            faces=({
                "identity_id": 7,
                "name": "Steve",
                "status": "confirmed",
            },),
        )
        candidates = [
            incident(2, "front-door", "2026-08-01T12:02:00+00:00"),
            incident(
                3,
                "foyer",
                "2026-08-01T12:05:00+00:00",
                faces=({
                    "identity_id": 7,
                    "name": "Steve",
                    "status": "confirmed",
                },),
            ),
        ]

        matches = correlate_incident_timeline(anchor, candidates, limit=2)

        by_event = {item["event_id"]: item for item in matches}
        self.assertEqual(by_event[3]["match_strength"], "confirmed_identity")
        self.assertEqual(by_event[3]["confidence"], 1.0)
        self.assertEqual(by_event[2]["match_strength"], "context_candidate")
        self.assertIn("identity is not established", by_event[2]["reasons"][0])

    def test_explicit_face_name_correlates_across_internal_identity_ids(self) -> None:
        anchor = incident(1, "gate", "2026-08-01T12:00:00+00:00", labels=())
        candidate = incident(
            4,
            "foyer",
            "2026-08-01T12:03:00+00:00",
            faces=({
                "identity_id": 22,
                "name": "Steve",
                "status": "confirmed",
            },),
        )

        matches = correlate_incident_timeline(
            anchor,
            [candidate],
            face_name="Steve",
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["match_strength"], "confirmed_identity")

    def test_automatic_identity_remains_distinct_from_confirmation(self) -> None:
        anchor = incident(
            1,
            "gate",
            "2026-08-01T12:00:00+00:00",
            faces=({"identity_id": 7, "name": "Steve", "status": "confirmed"},),
        )
        candidate = incident(
            4,
            "foyer",
            "2026-08-01T12:03:00+00:00",
            faces=({"identity_id": 7, "name": "Steve", "status": "automatic"},),
        )

        matches = correlate_incident_timeline(anchor, [candidate])

        self.assertEqual(matches[0]["match_strength"], "automatic_identity")
        self.assertIn("Automatic face match", matches[0]["reasons"][0])

    def test_unrelated_classes_are_not_added_to_timeline(self) -> None:
        anchor = incident(
            1,
            "gate",
            "2026-08-01T12:00:00+00:00",
            labels=("car",),
        )
        unrelated = incident(
            5,
            "foyer",
            "2026-08-01T12:01:00+00:00",
            labels=("dog",),
        )

        self.assertEqual(
            correlate_incident_timeline(anchor, [unrelated]),
            [],
        )


if __name__ == "__main__":
    unittest.main()

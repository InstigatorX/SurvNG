from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from survng.app.cross_camera_trace import build_cross_camera_trace


def _incident(
    event_id: int,
    camera_id: str,
    started: str,
    *,
    label: str = "person",
    faces: list | None = None,
) -> dict:
    return {
        "id": f"incident-{camera_id}-{event_id}",
        "representative_event_id": event_id,
        "camera_id": camera_id,
        "start_at": started,
        "end_at": started,
        "duration_seconds": 1,
        "event_count": 1,
        "trigger_source": "camera",
        "labels": [label],
        "zones": [],
        "motion_observations": [],
        "faces": faces or [],
        "events": [{"id": event_id, "kind": "motion", "objects": [], "faces": []}],
    }


class CrossCameraTraceTests(unittest.TestCase):
    def test_confirmed_face_match_is_ranked_first(self) -> None:
        anchor = _incident(
            42,
            "gate",
            "2026-08-01T12:00:00+00:00",
            faces=[{
                "identity_id": 7,
                "name": "Steve",
                "status": "confirmed",
                "confidence": 0.95,
            }],
        )
        match = _incident(43, "front-door", "2026-08-01T12:03:00+00:00")
        manager = SimpleNamespace(
            events=SimpleNamespace(
                between_compact=lambda *_args: [{"id": 43}],
            ),
            appearance_index=None,
        )

        def resolve_event(_manager, event_id: int):
            return anchor if event_id == 42 else match

        with patch("survng.app.cross_camera_trace._incident_rows", return_value=[match]):
            payload = build_cross_camera_trace(
                manager,
                resolve_event=resolve_event,
                hydrate=lambda _manager, summaries: summaries,
                with_faces=lambda _manager, summaries: summaries,
                event_id=42,
                start_at="2026-08-01T11:45:00+00:00",
                end_at="2026-08-01T12:15:00+00:00",
            )

        self.assertEqual(payload["matches"][0]["match_strength"], "confirmed_identity")
        self.assertEqual(payload["matches"][0]["event_id"], 43)
        self.assertIn("confirmed identity", payload["summary"])

    def test_vehicle_appearance_similarity_is_included(self) -> None:
        anchor = _incident(42, "gate", "2026-08-01T12:00:00+00:00", label="car")
        match = _incident(43, "upper-garage", "2026-08-01T12:04:00+00:00", label="truck")
        manager = SimpleNamespace(
            events=SimpleNamespace(between_compact=lambda *_args: [{"id": 43}]),
            appearance_index=SimpleNamespace(
                matches=lambda *_args, **_kwargs: [{
                    "event_id": 43,
                    "camera_id": "upper-garage",
                    "created_at": "2026-08-01T12:04:00+00:00",
                    "model_kind": "vehicle",
                    "similarity": 0.91,
                    "threshold": 0.8,
                    "visually_similar": True,
                }]
            ),
        )

        with patch("survng.app.cross_camera_trace._incident_rows", return_value=[match]):
            payload = build_cross_camera_trace(
                manager,
                resolve_event=lambda _manager, event_id: anchor if event_id == 42 else match,
                hydrate=lambda _manager, summaries: summaries,
                with_faces=lambda _manager, summaries: summaries,
                event_id=42,
                start_at="2026-08-01T11:45:00+00:00",
                end_at="2026-08-01T12:15:00+00:00",
            )

        self.assertEqual(payload["matches"][0]["match_strength"], "appearance_similarity")
        self.assertAlmostEqual(payload["matches"][0]["appearance_similarity"], 0.91)

    def test_missing_anchor_raises_lookup_error(self) -> None:
        with self.assertRaises(LookupError):
            build_cross_camera_trace(
                SimpleNamespace(events=SimpleNamespace(), appearance_index=None),
                resolve_event=lambda *_args: None,
                hydrate=lambda *_args: [],
                with_faces=lambda _manager, summaries: summaries,
                event_id=99,
            )


if __name__ == "__main__":
    unittest.main()

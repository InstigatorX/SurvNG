from __future__ import annotations

import json
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from survng.app.assistant import (
    AssistantAnswer,
    AssistantChatRequest,
    AssistantEvidence,
    AssistantPlan,
    AssistantProvider,
    AssistantToolCall,
    IncidentVisualAdvice,
    IncidentVisualReviewer,
    INCIDENT_VISUAL_SCHEMA,
    sanitize_assistant_data,
)
from survng.app.audit_ai import AuditAiAdvisor, AuditAiChange, AuditAiError
from survng.app.config import AppConfig, AuditAiConfig, CameraConfig


class AssistantModelsTest(unittest.TestCase):
    def test_request_normalizes_context_and_enforces_total_size(self) -> None:
        request = AssistantChatRequest.model_validate({
            "message": "  is gate healthy?  ",
            "context": {
                "camera_id": " gate ",
                "filters": {str(index): "x" * 500 for index in range(30)},
            },
        })

        self.assertEqual(request.message, "is gate healthy?")
        self.assertEqual(request.context.camera_id, "gate")
        self.assertEqual(len(request.context.filters), 16)
        self.assertTrue(all(len(value) == 256 for value in request.context.filters.values()))

        with self.assertRaisesRegex(ValidationError, "safe size limit"):
            AssistantChatRequest(
                message="x" * 8_000,
                history=[
                    {"role": "user", "content": "x" * 8_000}
                    for _ in range(20)
                ],
            )

    def test_evidence_sanitizer_removes_secrets_paths_and_urls(self) -> None:
        sanitized = sanitize_assistant_data({
            "api_key": "secret",
            "nested": {
                "model_path": "/config/model.xml",
                "stream_url": "rtsp://user:pass@camera/live",
                "documentation": "https://internal.example/help",
                "healthy": True,
            },
            "loose_path": "/mnt/recordings/file.mp4",
            "file_count": 12,
            "temporal_center_path_ratio": 0.42,
            "score": float("nan"),
        })

        encoded = json.dumps(sanitized)
        self.assertNotIn("secret", encoded)
        self.assertNotIn("model.xml", encoded)
        self.assertNotIn("user:pass", encoded)
        self.assertNotIn("internal.example", encoded)
        self.assertTrue(sanitized["nested"]["healthy"])
        self.assertEqual(sanitized["file_count"], 12)
        self.assertEqual(sanitized["temporal_center_path_ratio"], 0.42)
        self.assertIsNone(sanitized["score"])

    def test_client_visual_details_receive_the_same_redaction(self) -> None:
        evidence = AssistantEvidence(
            "E1",
            "incident_visual_review",
            "Visual review",
            "Review complete",
            {},
            client_data={"api_key": "secret", "image_path": "/private/image.jpg", "ok": True},
        )

        details = evidence.client_payload()["details"]

        self.assertEqual(details, {"ok": True})

    def test_client_evidence_allows_only_internal_api_images(self) -> None:
        internal = AssistantEvidence(
            "E1", "incident", "Gate", "Evidence", {},
            image_url="/api/events/42/thumbnail.jpg?width=960",
        )
        external = AssistantEvidence(
            "E2", "incident", "Gate", "Evidence", {},
            image_url="https://provider.example/private.jpg",
        )

        self.assertEqual(
            internal.client_payload()["image_url"],
            "/api/events/42/thumbnail.jpg?width=960",
        )
        self.assertNotIn("image_url", external.client_payload())


class AssistantProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AuditAiConfig(
            enabled=True,
            provider="gemini",
            api_key="test-key",
            model="analysis-model",
            assistant_reasoning_model="deep-model",
        )
        self.provider = AssistantProvider(self.config)
        self.request = AssistantChatRequest(message="Why was this incident missed?")

    def test_hybrid_model_roles_have_independent_fallbacks(self) -> None:
        self.assertEqual(self.provider.model_for_tier("fast"), "analysis-model")
        self.assertEqual(self.provider.model_for_tier("deep"), "deep-model")

        fallback = AssistantProvider(AuditAiConfig(provider="gemini", model="shared-model"))
        self.assertEqual(fallback.model_for_tier("fast"), "shared-model")
        self.assertEqual(fallback.model_for_tier("deep"), "shared-model")

    def test_planner_always_uses_fast_model(self) -> None:
        response = json.dumps({
            "reasoning_tier": "deep",
            "tool_calls": [{
                "name": "inspect_incident",
                "camera_id": "",
                "event_id": 42,
                "start_at": "",
                "end_at": "",
                "event_type": "all",
                "object_label": "",
                "zone": "",
                "minimum_confidence": None,
                "face_name": "",
                "limit": 12,
            }],
        })
        with patch.object(self.provider, "_complete_json", return_value=response) as complete:
            plan = self.provider.plan(self.request, {"cameras": []}, "2026-07-31T10:00:00-04:00")

        self.assertEqual(plan.reasoning_tier, "deep")
        self.assertEqual(plan.tool_calls[0].event_id, 42)
        self.assertEqual(complete.call_args.kwargs["model_override"], "analysis-model")

    def test_answer_uses_selected_tier_and_discards_unknown_citations(self) -> None:
        response = json.dumps({
            "answer": "The detector was delayed [E1].",
            "citations": ["E1", "E-unknown", "E1"],
            "suggestions": [],
        })
        evidence = [AssistantEvidence("E1", "incident", "Gate", "One event", {})]
        with patch.object(self.provider, "_complete_json", return_value=response) as complete:
            answer = self.provider.answer(self.request, evidence, "deep")

        self.assertEqual(answer.citations, ["E1"])
        self.assertEqual(complete.call_args.kwargs["model_override"], "deep-model")

    def test_invalid_provider_json_is_reported_as_provider_error(self) -> None:
        with patch.object(self.provider, "_complete_json", return_value="not json"):
            with self.assertRaisesRegex(AuditAiError, "invalid assistant plan"):
                self.provider.plan(self.request, {"cameras": []}, "2026-07-31T10:00:00-04:00")


class IncidentVisualReviewerTest(unittest.TestCase):
    def test_visual_schema_allows_only_camera_scoped_changes(self) -> None:
        change_schema = INCIDENT_VISUAL_SCHEMA["properties"]["changes"]["items"]
        self.assertEqual(change_schema["properties"]["scope"]["enum"], ["camera"])

    def test_visual_review_reuses_transport_with_deep_model(self) -> None:
        config = AuditAiConfig(
            enabled=True,
            provider="gemini",
            api_key="test-key",
            model="fast-model",
            assistant_reasoning_model="deep-model",
        )
        expected = IncidentVisualAdvice(
            verdict="detection_consistent",
            confidence=0.9,
            visible_subjects=["person"],
            summary="The person label agrees with the image.",
            observations=["One person is visible."],
            detector_assessment="consistent",
            tracking_assessment="consistent",
            changes=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "incident.jpg"
            image.write_bytes(b"jpeg")
            with patch.object(AuditAiAdvisor, "analyze_structured", return_value=expected) as analyze:
                result = IncidentVisualReviewer(config).review(
                    image,
                    {"stream_url": "rtsp://secret/live", "incident": {"id": 7}},
                )

        self.assertEqual(result, expected)
        self.assertEqual(analyze.call_args.kwargs["model_override"], "deep-model")
        self.assertNotIn("rtsp://secret", analyze.call_args.args[1])


class AssistantApiTest(unittest.TestCase):
    def test_assistant_catalog_includes_only_safe_face_identity_fields(self) -> None:
        from survng.app import main

        catalog = main._assistant_catalog(
            AppConfig(),
            SimpleNamespace(
                detector=SimpleNamespace(labels=["person"]),
                faces=SimpleNamespace(people=lambda: [{
                    "id": 7,
                    "name": "Steve",
                    "notes": "private note",
                }]),
            ),
        )

        self.assertEqual(
            catalog["recognized_faces"],
            [{"id": 7, "name": "Steve"}],
        )
        self.assertNotIn("private note", json.dumps(catalog))

    def test_visual_incident_evidence_builds_server_owned_change_preview(self) -> None:
        from survng.app import main

        active_config = AppConfig(
            audit_ai=AuditAiConfig(
                enabled=True,
                allow_apply_recommendations=True,
            ),
            cameras=[CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")],
        )
        incident = {
            "id": "gate-incident",
            "representative_event_id": 42,
            "camera_id": "gate",
            "start_at": "2026-07-31T20:00:00+00:00",
            "events": [{"id": 42, "kind": "motion", "objects": []}],
            "labels": ["person"],
            "zones": [],
            "motion_observations": [],
        }
        active_manager = SimpleNamespace(
            events=SimpleNamespace(get=lambda _event_id: {"id": 42, "camera_id": "gate"}),
            storage_dir=Path("/tmp"),
        )
        advice = IncidentVisualAdvice(
            verdict="probable_missed_detection",
            confidence=0.8,
            visible_subjects=["person"],
            summary="A person is visible but was not labeled.",
            observations=["One person is visible."],
            detector_assessment="missed",
            tracking_assessment="unavailable",
            changes=[AuditAiChange(
                scope="camera",
                setting="sensitivity",
                value="high",
                reason="Repeated motion-trigger misses would justify this adjustment.",
            )],
        )
        with (
            patch.object(main, "_assistant_incident_for_event", return_value=incident),
            patch.object(main, "event_snapshot_path", return_value=Path("/tmp/incident.jpg")),
            patch.object(main, "_assistant_camera_evidence", return_value=[]),
            patch.object(IncidentVisualReviewer, "review", return_value=advice),
        ):
            evidence = main._assistant_visual_incident_evidence(
                42, active_config, active_manager
            )

        self.assertIsNotNone(evidence)
        details = evidence.client_payload()["details"]
        self.assertTrue(details["can_apply"])
        self.assertEqual(details["proposals"][0]["current"], "balanced")
        self.assertEqual(details["proposals"][0]["proposed"], "high")
        self.assertEqual(len(details["configuration_fingerprint"]), 64)
        self.assertEqual(
            evidence.client_payload()["image_url"],
            "/api/events/42/thumbnail.jpg?width=960&quality=82",
        )

    def test_motion_change_preview_resolves_camera_inheritance_and_deduplicates(self) -> None:
        from survng.app import main

        active_config = AppConfig(
            cameras=[CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")],
        )
        camera = active_config.cameras[0]
        changes = [
            AuditAiChange(
                scope="camera",
                setting="sensitivity",
                value="high",
                reason="Repeated misses need more sensitivity.",
            ),
            AuditAiChange(
                scope="camera",
                setting="sensitivity",
                value="low",
                reason="Conflicting duplicate should be discarded.",
            ),
        ]

        normalized, previews = main._assistant_motion_change_previews(
            active_config, camera, changes
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(previews[0]["current"], "balanced")
        self.assertEqual(previews[0]["proposed"], "high")

    def test_incident_apply_requires_explicit_confirmation_before_state_access(self) -> None:
        from fastapi import HTTPException
        from survng.app import main

        with self.assertRaises(HTTPException) as raised:
            main.incident_ai_apply(
                42,
                main.IncidentAiApplyRequest(changes=[], confirmed=False),
            )

        self.assertEqual(raised.exception.status_code, 400)

    def test_incident_apply_rejects_global_changes_before_state_access(self) -> None:
        from fastapi import HTTPException
        from survng.app import main

        request = main.IncidentAiApplyRequest(
            confirmed=True,
            changes=[AuditAiChange(
                scope="global",
                setting="sensitivity",
                value="high",
                reason="A single incident must not tune every camera.",
            )],
        )
        with self.assertRaises(HTTPException) as raised:
            main.incident_ai_apply(42, request)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("camera-scoped", raised.exception.detail)

    def test_incident_apply_rejects_stale_configuration_fingerprint(self) -> None:
        from fastapi import HTTPException
        from survng.app import main

        active_config = AppConfig(
            audit_ai=AuditAiConfig(allow_apply_recommendations=True),
            cameras=[CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")],
        )
        active_manager = SimpleNamespace(
            events=SimpleNamespace(get=lambda _event_id: {"camera_id": "gate"}),
        )
        request = main.IncidentAiApplyRequest(
            confirmed=True,
            configuration_fingerprint="stale",
            changes=[AuditAiChange(
                scope="camera",
                setting="sensitivity",
                value="high",
                reason="Repeated misses need more sensitivity.",
            )],
        )
        with (
            patch.object(main, "config", active_config),
            patch.object(main, "manager", active_manager),
            self.assertRaises(HTTPException) as raised,
        ):
            main.incident_ai_apply(42, request)

        self.assertEqual(raised.exception.status_code, 409)

    def test_visual_tool_uses_current_incident_context(self) -> None:
        from survng.app import main

        request = AssistantChatRequest.model_validate({
            "message": "Look at this incident",
            "context": {"incident_event_id": 42},
        })
        expected = AssistantEvidence("E-visual-42", "incident_visual_review", "Review", "Done", {})
        with patch.object(main, "_assistant_visual_incident_evidence", return_value=expected) as review:
            evidence = main._assistant_execute_tool(
                AssistantToolCall(name="analyze_incident_visual"),
                request,
                AppConfig(),
                SimpleNamespace(),
            )

        self.assertEqual(evidence, [expected])
        self.assertEqual(review.call_args.args[0], 42)

    def test_cross_camera_tool_uses_current_incident_context(self) -> None:
        from survng.app import main

        request = AssistantChatRequest.model_validate({
            "message": "Where did this person go next?",
            "context": {"incident_event_id": 42},
        })
        expected = AssistantEvidence(
            "E-trace-42",
            "cross_camera_timeline",
            "Timeline",
            "One confirmed match",
            {},
        )
        with patch.object(
            main,
            "_assistant_trace_across_cameras",
            return_value=[expected],
        ) as trace:
            evidence = main._assistant_execute_tool(
                AssistantToolCall(name="trace_across_cameras"),
                request,
                AppConfig(),
                SimpleNamespace(),
            )

        self.assertEqual(evidence, [expected])
        self.assertEqual(trace.call_args.args[0].name, "trace_across_cameras")
        self.assertEqual(trace.call_args.args[1].context.incident_event_id, 42)

    def test_cross_camera_trace_returns_ranked_timeline_and_incident_images(self) -> None:
        from survng.app import main

        def incident(event_id: int, camera_id: str, started: str) -> dict:
            return {
                "id": f"incident-{camera_id}-{event_id}",
                "representative_event_id": event_id,
                "camera_id": camera_id,
                "start_at": started,
                "end_at": started,
                "duration_seconds": 1,
                "event_count": 1,
                "trigger_source": "camera",
                "labels": ["person"],
                "zones": [],
                "motion_observations": [],
                "faces": [{
                    "identity_id": 7,
                    "name": "Steve",
                    "status": "confirmed",
                    "confidence": 0.95,
                }],
                "events": [{
                    "id": event_id,
                    "kind": "motion",
                    "objects": [{"label": "person", "confidence": 0.9}],
                    "faces": [],
                }],
            }

        anchor = incident(42, "gate", "2026-08-01T12:00:00+00:00")
        match = incident(43, "front-door", "2026-08-01T12:03:00+00:00")
        request = AssistantChatRequest.model_validate({
            "message": "Trace this person",
            "context": {"incident_event_id": 42},
        })
        manager = SimpleNamespace(
            events=SimpleNamespace(between_compact=lambda *_args: [{"id": 43}]),
        )
        with (
            patch.object(main, "_assistant_incident_for_event", return_value=anchor),
            patch.object(main, "_incident_rows", return_value=[match]),
            patch.object(main, "_hydrate_incidents", return_value=[match]),
            patch.object(main, "_incidents_with_faces", return_value=[match]),
        ):
            evidence = main._assistant_trace_across_cameras(
                AssistantToolCall(
                    name="trace_across_cameras",
                    event_id=42,
                    start_at="2026-08-01T11:45:00+00:00",
                    end_at="2026-08-01T12:15:00+00:00",
                ),
                request,
                manager,
            )

        self.assertEqual(evidence[0].kind, "cross_camera_timeline")
        timeline = evidence[0].client_payload()["details"]["timeline"]
        self.assertEqual(timeline["matches"][0]["match_strength"], "confirmed_identity")
        self.assertEqual(evidence[1].client_payload()["image_url"], "/api/events/42/thumbnail.jpg?width=960&quality=82")
        self.assertEqual(evidence[2].client_payload()["image_url"], "/api/events/43/thumbnail.jpg?width=960&quality=82")

    def test_cross_camera_trace_uses_durable_vehicle_appearance_across_labels(self) -> None:
        from survng.app import main

        def incident(event_id: int, camera_id: str, started: str, label: str) -> dict:
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
                "faces": [],
                "events": [{"id": event_id, "kind": "motion", "objects": []}],
            }

        anchor = incident(42, "gate", "2026-08-01T12:00:00+00:00", "car")
        match = incident(43, "upper-garage", "2026-08-01T12:04:00+00:00", "truck")
        manager = SimpleNamespace(
            events=SimpleNamespace(between_compact=lambda *_args: [{"id": 43}]),
            appearance_index=SimpleNamespace(matches=lambda *_args, **_kwargs: [{
                "event_id": 43,
                "camera_id": "upper-garage",
                "created_at": "2026-08-01T12:04:00+00:00",
                "model_kind": "vehicle",
                "similarity": 0.91,
                "threshold": 0.8,
                "visually_similar": True,
            }]),
        )
        request = AssistantChatRequest.model_validate({
            "message": "Is this the same vehicle?",
            "context": {"incident_event_id": 42},
        })
        with (
            patch.object(main, "_assistant_incident_for_event", return_value=anchor),
            patch.object(main, "_incident_rows", return_value=[match]),
            patch.object(main, "_hydrate_incidents", return_value=[match]),
            patch.object(main, "_incidents_with_faces", return_value=[match]),
        ):
            evidence = main._assistant_trace_across_cameras(
                AssistantToolCall(
                    name="trace_across_cameras",
                    event_id=42,
                    start_at="2026-08-01T11:45:00+00:00",
                    end_at="2026-08-01T12:15:00+00:00",
                ),
                request,
                manager,
            )

        timeline = evidence[0].client_payload()["details"]["timeline"]
        self.assertEqual(timeline["matches"][0]["match_strength"], "appearance_similarity")
        self.assertEqual(timeline["matches"][0]["appearance_similarity"], 0.91)
        self.assertIn("appearance-similar", evidence[0].summary)

    def test_configuration_evidence_includes_model_roles_without_provider_secrets(self) -> None:
        from survng.app import main

        active_config = AppConfig(audit_ai=AuditAiConfig(
            enabled=True,
            provider="openai",
            api_key="private-key",
            base_url="https://private-provider.example/v1",
            model="fast-model",
            assistant_reasoning_model="deep-model",
        ))

        payload = main._assistant_configuration_evidence(active_config).prompt_payload()
        encoded = json.dumps(payload)

        self.assertEqual(payload["data"]["ai"]["analysis_and_fast_model"], "fast-model")
        self.assertEqual(payload["data"]["ai"]["deep_reasoning_model"], "deep-model")
        self.assertTrue(payload["data"]["ai"]["deep_reasoning_uses_separate_model"])
        self.assertNotIn("private-key", encoded)
        self.assertNotIn("private-provider", encoded)

    def test_chat_routes_with_fast_planner_and_deep_answer_model(self) -> None:
        from survng.app import main

        active_config = AppConfig(audit_ai=AuditAiConfig(
            enabled=True,
            assistant_enabled=True,
            provider="gemini",
            api_key="test-key",
            model="fast-model",
            assistant_reasoning_model="deep-model",
        ))
        active_manager = SimpleNamespace(detector=SimpleNamespace(labels=["person", "car"]))
        request = AssistantChatRequest(message="Why was this incident missed?")
        plan = AssistantPlan(
            reasoning_tier="deep",
            tool_calls=[AssistantToolCall(name="get_system_health")],
        )
        answer = AssistantAnswer(
            answer="The camera was healthy [E-system].",
            citations=["E-system"],
            suggestions=["Inspect the incident"],
        )
        evidence = AssistantEvidence(
            "E-system", "system_health", "System", "All cameras healthy", {}, "/config"
        )

        with (
            patch.object(main, "config", active_config),
            patch.object(main, "manager", active_manager),
            patch.object(AssistantProvider, "plan", return_value=plan) as planner,
            patch.object(AssistantProvider, "answer", return_value=answer) as responder,
            patch.object(main, "_assistant_execute_tool", return_value=[evidence]),
            patch.object(main.asyncio, "to_thread", new=AsyncMock(side_effect=lambda function: function())),
        ):
            response = asyncio.run(main.assistant_chat(request))

        self.assertEqual(response["reasoning_tier"], "deep")
        self.assertEqual(response["model"], "deep-model")
        self.assertEqual(response["evidence"][0]["id"], "E-system")
        planner.assert_called_once()
        self.assertEqual(responder.call_args.args[2], "deep")


if __name__ == "__main__":
    unittest.main()

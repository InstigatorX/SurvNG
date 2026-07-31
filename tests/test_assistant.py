from __future__ import annotations

import json
import asyncio
import unittest
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
    sanitize_assistant_data,
)
from survng.app.audit_ai import AuditAiError
from survng.app.config import AppConfig, AuditAiConfig


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
            "score": float("nan"),
        })

        encoded = json.dumps(sanitized)
        self.assertNotIn("secret", encoded)
        self.assertNotIn("model.xml", encoded)
        self.assertNotIn("user:pass", encoded)
        self.assertNotIn("internal.example", encoded)
        self.assertTrue(sanitized["nested"]["healthy"])
        self.assertIsNone(sanitized["score"])


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


class AssistantApiTest(unittest.TestCase):
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

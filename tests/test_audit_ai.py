from __future__ import annotations

import unittest
import json
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from pydantic import ValidationError

from survng.app.audit_ai import (
    MAX_AUDIT_IMAGE_BYTES,
    AuditAiAdvice,
    AuditAiAdvisor,
    AuditAiChange,
    AuditAiError,
    SYSTEM_PROMPT,
    audit_analysis_prompt,
    motion_audit_interpretation,
    motion_paradigm_context,
    validate_tuning_value,
)
from survng.app.config import AuditAiConfig


class AuditAiTest(unittest.TestCase):
    def test_active_and_cooldown_audits_are_duplicate_control_not_detector_misses(self) -> None:
        active = motion_audit_interpretation(
            reason="event_state_active",
            event_id=None,
            object_detected=None,
        )
        cooldown = motion_audit_interpretation(
            reason="event_state_cooldown",
            event_id=None,
            object_detected=None,
        )

        self.assertEqual(active["category"], "duplicate_active_event")
        self.assertEqual(cooldown["category"], "duplicate_event_cooldown")
        self.assertFalse(active["object_detection_miss"])
        self.assertFalse(cooldown["object_detection_miss"])

    def test_uncorrelated_backup_object_is_not_treated_as_detector_miss(self) -> None:
        interpretation = motion_audit_interpretation(
            reason="object_not_motion_correlated",
            event_id=None,
            object_detected=False,
        )

        self.assertEqual(interpretation["category"], "object_not_motion_correlated")
        self.assertFalse(interpretation["object_detection_miss"])

    def test_startup_readiness_hold_is_not_treated_as_detector_miss(self) -> None:
        interpretation = motion_audit_interpretation(
            reason="startup_not_ready",
            event_id=None,
            object_detected=None,
        )

        self.assertEqual(interpretation["category"], "visual_backup_scene_learning")
        self.assertFalse(interpretation["object_detection_miss"])

    def test_prompt_describes_current_trigger_validator_paradigm(self) -> None:
        self.assertIn("camera_triggered", SYSTEM_PROMPT)
        self.assertIn("visual_triggered", SYSTEM_PROMPT)
        self.assertIn("usually means detection did not run", SYSTEM_PROMPT)
        self.assertIn("detection was attempted but did not complete", SYSTEM_PROMPT)
        self.assertIn("operator-owned safety settings", SYSTEM_PROMPT)
        self.assertIn("starts only after the initial detector decision", SYSTEM_PROMPT)
        self.assertIn("audit category of visual_backup", SYSTEM_PROMPT)

    def test_audit_prompt_is_neutral_about_decision_outcome(self) -> None:
        prompt = audit_analysis_prompt('{"decision_outcome":{"object_detection_ran":true}}')

        self.assertIn("accepted, filtered, rescued", prompt)
        self.assertIn("incident-eligibility", prompt)
        self.assertNotIn("rejected motion audit", prompt)

    def test_camera_triggered_paradigm_identifies_optional_validators(self) -> None:
        context = motion_paradigm_context(
            mode="camera",
            onvif_enabled=True,
            has_live_substream=True,
            fusion={
                "policy": "all",
                "sources": ["mog2"],
                "include_primary": True,
                "fail_open": True,
            },
            mog2_available=True,
        )

        self.assertEqual(context["paradigm"], "camera_triggered")
        self.assertEqual(context["automatic_trigger"]["source"], "onvif_camera_notice")
        self.assertEqual(context["adaptive_visual"]["role"], "validator")
        self.assertEqual(context["mog2"]["role"], "validator")
        self.assertEqual(context["onvif"]["role"], "automatic_trigger")
        self.assertTrue(context["validator_decision"]["fail_open"])
        self.assertEqual(context["schema_version"], 4)
        self.assertEqual(context["incident_eligibility"]["policy"], "zones_only")

    def test_camera_visual_backup_paradigm_explains_conservative_rescue(self) -> None:
        context = motion_paradigm_context(
            mode="camera_rescue",
            onvif_enabled=True,
            has_live_substream=True,
            fusion={"policy": "audit", "sources": [], "include_primary": True},
            mog2_available=False,
        )

        self.assertEqual(context["paradigm"], "camera_triggered_with_visual_backup")
        self.assertEqual(context["onvif"]["role"], "primary_automatic_trigger")
        self.assertEqual(context["adaptive_visual"]["role"], "validator_and_backup_trigger")
        self.assertTrue(context["adaptive_visual"]["backup_trigger_enabled"])

    def test_visual_triggered_paradigm_makes_onvif_diagnostic_only(self) -> None:
        context = motion_paradigm_context(
            mode="adaptive",
            onvif_enabled=True,
            has_live_substream=False,
            fusion={"policy": "audit", "sources": [], "include_primary": True},
            mog2_available=True,
        )

        self.assertEqual(context["paradigm"], "visual_triggered")
        self.assertEqual(context["adaptive_visual"]["role"], "required_trigger")
        self.assertEqual(context["adaptive_visual"]["analysis_feed"], "main_stream_fallback")
        self.assertEqual(context["onvif"]["role"], "diagnostic_only")
        self.assertEqual(context["mog2"]["role"], "disabled")

    def _advice_json(self) -> str:
        return json.dumps({
            "verdict": "real_motion",
            "confidence": 0.9,
            "visible_subjects": ["dog"],
            "summary": "A dog is visible.",
            "explanation": ["The subject persists across the scene."],
            "changes": [],
        })

    def test_openai_response_text_is_parsed(self) -> None:
        advisor = AuditAiAdvisor(AuditAiConfig(enabled=True, provider="openai", api_key="secret"))
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "audit.jpg"
            image.write_bytes(b"jpeg")
            with patch.object(advisor, "_request", return_value={
                "output": [{"content": [{"type": "output_text", "text": self._advice_json()}]}],
            }):
                advice = advisor.analyze(image, {"audit": {"score": 0.47}})
        self.assertEqual(advice.verdict, "real_motion")

    def test_gemini_candidate_text_is_parsed(self) -> None:
        advisor = AuditAiAdvisor(AuditAiConfig(enabled=True, provider="gemini", api_key="secret"))
        request_payload = {}

        def request(_url, payload, _headers):
            request_payload.update(payload)
            return {"candidates": [{"content": {"parts": [{"text": self._advice_json()}]}}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "audit.jpg"
            image.write_bytes(b"jpeg")
            with patch.object(advisor, "_request", side_effect=request):
                advice = advisor.analyze(image, {"audit": {"score": 0.47}})
        self.assertEqual(advice.visible_subjects, ["dog"])
        generation_config = request_payload["generationConfig"]
        self.assertNotIn("responseSchema", generation_config)
        schema = generation_config["responseJsonSchema"]
        variants = schema["properties"]["changes"]["items"]["anyOf"]
        self.assertEqual(len(variants), 2)
        for variant in variants:
            value_schema = variant["properties"]["value"]
            self.assertEqual(
                value_schema["anyOf"],
                [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                ],
            )
        camera_variant = next(
            variant for variant in variants
            if variant["properties"]["scope"]["enum"] == ["camera"]
        )
        camera_settings = camera_variant["properties"]["setting"]["enum"]
        self.assertNotIn("sample_fps", camera_settings)
        self.assertNotIn("window_seconds", camera_settings)

    def test_advice_accepts_bounded_camera_change(self) -> None:
        advice = AuditAiAdvice.model_validate({
            "verdict": "real_motion",
            "confidence": 0.92,
            "visible_subjects": ["dog"],
            "summary": "A dog entered near the image edge.",
            "explanation": ["Motion was persistent but changed apparent size."],
            "changes": [{
                "scope": "camera",
                "setting": "borderline_margin",
                "value": 0.03,
                "reason": "Inspect narrowly missed real motion.",
            }],
        })
        self.assertEqual(advice.changes[0].value, 0.03)

    def test_camera_change_rejects_global_only_setting(self) -> None:
        with self.assertRaises(ValidationError):
            AuditAiChange(
                scope="camera",
                setting="sample_fps",
                value=6,
                reason="Not supported as a camera override.",
            )

    def test_tuning_bounds_reject_unsafe_margin(self) -> None:
        with self.assertRaises(ValueError):
            validate_tuning_value("borderline_margin", 0.25)
        with self.assertRaisesRegex(ValueError, "numeric"):
            validate_tuning_value("borderline_margin", False)

    def test_mog2_tuning_is_bounded_but_validator_topology_is_operator_owned(self) -> None:
        self.assertEqual(validate_tuning_value("mog2_history_seconds", 45), 45.0)
        with self.assertRaises(ValidationError):
            AuditAiChange(
                scope="camera",
                setting="mog2_audit_enabled",
                value=False,
                reason="Do not let AI change validator topology.",
            )
        with self.assertRaises(ValueError):
            validate_tuning_value("mog2_history_seconds", 301)

    def test_pipeline_recommendations_are_high_level_and_bounded(self) -> None:
        self.assertEqual(validate_tuning_value("analysis_preset", "MODULAR"), "modular")
        self.assertEqual(validate_tuning_value("analysis_preset", "ADAPTIVE"), "adaptive")
        self.assertEqual(validate_tuning_value("visual_backup_min_consecutive", 4), 4)
        self.assertEqual(validate_tuning_value("visual_backup_min_score", 0.8), 0.8)
        with self.assertRaises(ValueError):
            validate_tuning_value("visual_backup_cooldown_seconds", 2)
        with self.assertRaises(ValueError):
            validate_tuning_value("fusion_policy", "weighted")
        with self.assertRaises(ValidationError):
            AuditAiChange(
                scope="camera",
                setting="fusion_sources",
                value=["onvif"],
                reason="Do not let AI change validator topology.",
            )

    def test_oversized_audit_image_is_rejected_before_provider_request(self) -> None:
        advisor = AuditAiAdvisor(AuditAiConfig(enabled=True, provider="openai", api_key="secret"))
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "audit.jpg"
            with image.open("wb") as handle:
                handle.truncate(MAX_AUDIT_IMAGE_BYTES + 1)
            with (
                patch.object(advisor, "_request") as request,
                self.assertRaisesRegex(AuditAiError, "too large"),
            ):
                advisor.analyze(image, {})

        request.assert_not_called()

    def test_non_finite_audit_context_is_rejected_before_provider_request(self) -> None:
        advisor = AuditAiAdvisor(AuditAiConfig(enabled=True, provider="openai", api_key="secret"))
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "audit.jpg"
            image.write_bytes(b"jpeg")
            with (
                patch.object(advisor, "_request") as request,
                self.assertRaisesRegex(AuditAiError, "invalid numeric"),
            ):
                advisor.analyze(image, {"score": float("nan")})

        request.assert_not_called()

    def test_malformed_provider_shapes_raise_bounded_audit_errors(self) -> None:
        advisor = AuditAiAdvisor(AuditAiConfig(enabled=True, provider="openai", api_key="secret"))
        with self.assertRaisesRegex(AuditAiError, "no output text"):
            with patch.object(advisor, "_request", return_value={"output": [None, "bad"]}):
                advisor._openai_responses("model", "prompt", b"image", "image/jpeg")

        compatible = AuditAiAdvisor(AuditAiConfig(
            enabled=True,
            provider="openai_compatible",
            api_key="secret",
            base_url="http://localhost/v1",
        ))
        with self.assertRaisesRegex(AuditAiError, "no message content"):
            with patch.object(compatible, "_request", return_value={
                "choices": [{"message": {"content": [None, "bad"]}}],
            }):
                compatible._openai_compatible("model", "prompt", b"image", "image/jpeg")

        gemini = AuditAiAdvisor(AuditAiConfig(enabled=True, provider="gemini", api_key="secret"))
        with self.assertRaisesRegex(AuditAiError, "no candidate text"):
            with patch.object(gemini, "_request", return_value={
                "candidates": [{"content": {"parts": "bad"}}],
            }):
                gemini._gemini("model", "prompt", b"image", "image/jpeg")

    def test_http_error_body_is_not_reflected(self) -> None:
        advisor = AuditAiAdvisor(AuditAiConfig(enabled=True, provider="openai", api_key="secret"))
        error = HTTPError(
            "https://example.invalid",
            401,
            "Unauthorized",
            {},
            BytesIO(b'{"error":"api_key=secret"}'),
        )

        with (
            patch("survng.app.audit_ai.urlopen", side_effect=error),
            self.assertRaises(AuditAiError) as raised,
        ):
            advisor._request("https://example.invalid", {}, {})

        self.assertEqual(str(raised.exception), "AI provider returned HTTP 401")

    def test_invalid_recommendation_does_not_reflect_provider_values(self) -> None:
        advisor = AuditAiAdvisor(AuditAiConfig(enabled=True, provider="openai", api_key="secret"))
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "audit.jpg"
            image.write_bytes(b"jpeg")
            with (
                patch.object(advisor, "_openai_responses", return_value=json.dumps({
                    "verdict": "provider-secret-value",
                })),
                self.assertRaises(AuditAiError) as raised,
            ):
                advisor.analyze(image, {})

        self.assertEqual(str(raised.exception), "AI provider returned invalid recommendation JSON")
        self.assertNotIn("provider-secret-value", str(raised.exception))


if __name__ == "__main__":
    unittest.main()

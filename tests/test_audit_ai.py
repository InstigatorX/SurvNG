from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from survng.app.audit_ai import AuditAiAdvice, AuditAiAdvisor, AuditAiChange, validate_tuning_value
from survng.app.config import AuditAiConfig


class AuditAiTest(unittest.TestCase):
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
            self.assertNotIn("type", value_schema)
            self.assertEqual(
                value_schema["anyOf"],
                [{"type": "string"}, {"type": "number"}, {"type": "boolean"}],
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


if __name__ == "__main__":
    unittest.main()

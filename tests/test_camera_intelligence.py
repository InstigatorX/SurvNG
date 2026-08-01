from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException

from survng.app import main
from survng.app.camera_intelligence import (
    aggregate_camera_intelligence,
    compare_camera_intelligence_results,
    select_balanced_samples,
)
from survng.app.config import AppConfig


class CameraIntelligenceTest(unittest.TestCase):
    @staticmethod
    def configured_app() -> AppConfig:
        return AppConfig.model_validate({
            "audit_ai": {
                "enabled": True,
                "api_key": "secret",
                "allow_apply_recommendations": True,
            },
            "cameras": [{
                "id": "gate",
                "name": "Gate",
                "stream_url": "rtsp://camera.invalid/main",
            }],
        })

    def test_balanced_selection_preserves_rare_outcomes(self) -> None:
        candidates = [
            {"record_id": index, "category": "recognized_incident"}
            for index in range(20)
        ] + [
            {"record_id": 101, "category": "possible_miss"},
            {"record_id": 102, "category": "visual_backup"},
            {"record_id": 103, "category": "motion_filtered"},
        ]

        selected = select_balanced_samples(candidates, 6)

        self.assertEqual(len(selected), 6)
        self.assertEqual(
            {item["category"] for item in selected},
            {"possible_miss", "visual_backup", "motion_filtered", "recognized_incident"},
        )

    def test_aggregation_requires_repeated_camera_evidence(self) -> None:
        analyses = [
            {
                "kind": "incident",
                "record_id": index,
                "category": "possible_miss",
                "verdict": "likely_miss",
                "confidence": 0.8,
                "summary": "A person appears to have been missed.",
                "visible_subjects": ["person"],
                "detector_assessment": "missed",
                "tracking_assessment": "unavailable",
                "image_url": f"/api/events/{index}/thumbnail.jpg",
                "changes": ([{
                    "scope": "camera",
                    "setting": "frame_width",
                    "value": 480,
                    "reason": "Repeated distant subjects need more detail.",
                }] if index < 2 else [{
                    "scope": "camera",
                    "setting": "sensitivity",
                    "value": "high",
                    "reason": "Single-image suggestion.",
                }] if index == 2 else []),
            }
            for index in range(4)
        ]

        report = aggregate_camera_intelligence(
            analyses,
            records_considered=100,
            selected_images=4,
            failed=0,
            hours=24,
        )

        self.assertEqual(report["review_type"], "camera_intelligence")
        self.assertEqual(report["verdict_counts"], {"likely_miss": 4})
        self.assertEqual(report["visible_subject_counts"], {"person": 4})
        self.assertEqual(
            report["category_verdict_counts"],
            {"possible_miss": {"likely_miss": 4}},
        )
        self.assertEqual(len(report["samples"]), 4)
        self.assertEqual(len(report["recommendations"]), 1)
        self.assertEqual(report["recommendations"][0]["setting"], "frame_width")
        self.assertEqual(report["recommendations"][0]["support_count"], 2)

    def test_conflicting_equally_supported_values_are_not_recommended(self) -> None:
        analyses = []
        for index, value in enumerate((320, 320, 480, 480)):
            analyses.append({
                "kind": "motion_decision",
                "record_id": index,
                "category": "motion_filtered",
                "verdict": "likely_false_alarm",
                "confidence": 0.9,
                "changes": [{
                    "scope": "camera",
                    "setting": "frame_width",
                    "value": value,
                    "reason": "Conflicting evidence.",
                }],
            })

        report = aggregate_camera_intelligence(
            analyses,
            records_considered=4,
            selected_images=4,
            failed=0,
            hours=24,
        )

        self.assertEqual(report["recommendations"], [])

    def test_effectiveness_comparison_uses_rates_and_marks_improvement(self) -> None:
        comparison = compare_camera_intelligence_results(
            {
                "analyzed": 10,
                "verdict_counts": {
                    "likely_miss": 3,
                    "likely_false_alarm": 2,
                    "consistent": 5,
                },
                "category_verdict_counts": {
                    "recognized_incident": {
                        "likely_miss": 3,
                        "likely_false_alarm": 2,
                        "consistent": 5,
                    },
                },
            },
            {
                "analyzed": 8,
                "verdict_counts": {
                    "likely_miss": 1,
                    "likely_false_alarm": 1,
                    "consistent": 6,
                },
                "category_verdict_counts": {
                    "recognized_incident": {
                        "likely_miss": 1,
                        "likely_false_alarm": 1,
                        "consistent": 6,
                    },
                },
            },
        )

        self.assertEqual(comparison["outcome"], "improved")
        self.assertEqual(comparison["before_issue_rate"], 0.5)
        self.assertEqual(comparison["after_issue_rate"], 0.25)
        self.assertEqual(comparison["issue_rate_change_points"], -25.0)
        self.assertEqual(comparison["matched_sample_support"], 8)

    def test_effectiveness_is_inconclusive_without_matched_categories(self) -> None:
        comparison = compare_camera_intelligence_results(
            {
                "analyzed": 8,
                "verdict_counts": {"likely_miss": 4, "consistent": 4},
                "category_verdict_counts": {
                    "motion_filtered": {"likely_miss": 4, "consistent": 4},
                },
            },
            {
                "analyzed": 8,
                "verdict_counts": {"consistent": 8},
                "category_verdict_counts": {
                    "recognized_incident": {"consistent": 8},
                },
            },
        )

        self.assertEqual(comparison["outcome"], "inconclusive")
        self.assertEqual(comparison["matched_sample_support"], 0)

    def test_apply_accepts_only_persisted_review_recommendations(self) -> None:
        active_config = self.configured_app()
        camera = active_config.cameras[0]
        fingerprint = main._assistant_motion_config_fingerprint(active_config, camera)
        review = {
            "id": 7,
            "camera_id": "gate",
            "status": "completed",
            "result": {
                "review_type": "camera_intelligence",
                "recommendations": [{
                    "setting": "frame_width",
                    "value": 480,
                    "proposed": 480,
                    "reasons": ["Repeated distant subjects need more detail."],
                }],
            },
        }
        create_evaluation = Mock(return_value={"id": 9, "status": "collecting"})
        fake_manager = SimpleNamespace(events=SimpleNamespace(
            get_motion_ai_review=lambda _review_id: review,
            create_camera_intelligence_evaluation=create_evaluation,
        ))
        request = main.CameraIntelligenceApplyRequest.model_validate({
            "confirmed": True,
            "configuration_fingerprint": fingerprint,
            "evaluation_hours": 24,
            "changes": [{
                "scope": "camera",
                "setting": "frame_width",
                "value": 480,
                "reason": "Submitted client reason is not trusted.",
            }],
        })

        with (
            patch.object(main, "config", active_config),
            patch.object(main, "manager", fake_manager),
            patch.object(main, "apply_config_update", return_value=(active_config, {
                "camera_workers_restarted": True,
                "apply_mode": "camera_reload",
            })) as apply_update,
        ):
            response = main.camera_intelligence_apply(7, request)

        self.assertTrue(response["ok"])
        self.assertEqual(response["camera_id"], "gate")
        self.assertEqual(response["applied"][0]["proposed"], 480)
        self.assertEqual(response["effectiveness_evaluation"]["id"], 9)
        self.assertEqual(
            create_evaluation.call_args.kwargs["baseline_review_id"],
            7,
        )
        applied_config = apply_update.call_args.args[0]
        self.assertEqual(applied_config.cameras[0].motion_qualification.frame_width, 480)

    def test_apply_rejects_change_not_in_persisted_review(self) -> None:
        active_config = self.configured_app()
        camera = active_config.cameras[0]
        fingerprint = main._assistant_motion_config_fingerprint(active_config, camera)
        fake_manager = SimpleNamespace(events=SimpleNamespace(
            get_motion_ai_review=lambda _review_id: {
                "id": 7,
                "camera_id": "gate",
                "status": "completed",
                "result": {
                    "review_type": "camera_intelligence",
                    "recommendations": [{"setting": "frame_width", "value": 480}],
                },
            },
        ))
        request = main.CameraIntelligenceApplyRequest.model_validate({
            "confirmed": True,
            "configuration_fingerprint": fingerprint,
            "changes": [{
                "scope": "camera",
                "setting": "sensitivity",
                "value": "high",
                "reason": "Not recommended by the review.",
            }],
        })

        with patch.object(main, "config", active_config), patch.object(main, "manager", fake_manager):
            with self.assertRaises(HTTPException) as raised:
                main.camera_intelligence_apply(7, request)

        self.assertEqual(raised.exception.status_code, 400)

    def test_ready_effectiveness_check_starts_bounded_followup(self) -> None:
        active_config = self.configured_app()
        ready = {
            "id": 9,
            "camera_id": "gate",
            "status": "ready",
            "applied_at": "2026-07-30T12:00:00+00:00",
            "baseline_result": {"analyzed": 8},
        }
        reviewing = {**ready, "status": "reviewing", "followup_review_id": 22}
        events = SimpleNamespace(
            get_camera_intelligence_evaluation=Mock(
                side_effect=[ready, reviewing]
            ),
            create_motion_ai_review=Mock(return_value={"id": 22}),
            start_camera_intelligence_followup=Mock(return_value=reviewing),
        )
        fake_manager = SimpleNamespace(events=events)
        limiter = SimpleNamespace(acquire=Mock(return_value=True), release=Mock())
        thread = SimpleNamespace(start=Mock())
        samples = [{"kind": "incident", "event_id": 5, "camera_id": "gate"}]

        with (
            patch.object(main, "config", active_config),
            patch.object(main, "manager", fake_manager),
            patch.object(main, "AUDIT_AI_LIMITER", limiter),
            patch.object(main, "_begin_ai_operation"),
            patch.object(
                main,
                "_camera_intelligence_candidates",
                return_value=(samples, 10),
            ) as candidates,
            patch.object(main.threading, "Thread", return_value=thread) as thread_factory,
        ):
            response = main.start_camera_intelligence_followup(
                9,
                main.CameraIntelligenceFollowupRequest(image_limit=8),
            )

        self.assertEqual(response["status"], "reviewing")
        limiter.acquire.assert_called_once_with(blocking=False)
        candidates.assert_called_once()
        self.assertEqual(candidates.call_args.kwargs["image_limit"], 8)
        events.start_camera_intelligence_followup.assert_called_once_with(9, 22)
        self.assertEqual(
            thread_factory.call_args.kwargs["name"],
            "camera-effectiveness-gate",
        )
        thread.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

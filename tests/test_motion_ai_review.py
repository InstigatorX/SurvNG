from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from survng.app.audit_ai import AuditAiAdvice
from survng.app.events import EventStore
from survng.app.motion_ai_review import aggregate_motion_ai_review


class MotionAiReviewTest(unittest.TestCase):
    @staticmethod
    def advice(
        verdict: str,
        confidence: float,
        *,
        setting: str = "",
        value: object = None,
        scope: str = "camera",
    ) -> AuditAiAdvice:
        changes = [] if not setting else [{
            "scope": scope,
            "setting": setting,
            "value": value,
            "reason": "Repeated distant subject was too small at the current resolution.",
        }]
        return AuditAiAdvice.model_validate({
            "verdict": verdict,
            "confidence": confidence,
            "visible_subjects": ["person"] if verdict == "real_motion" else [],
            "summary": "Structured review result.",
            "explanation": [],
            "changes": changes,
        })

    def test_aggregation_requires_repeated_camera_scoped_support(self) -> None:
        analyses = [
            (
                {"id": index, "reason": "low_score", "created_at": f"2026-07-27T12:00:0{index}+00:00"},
                self.advice(
                    "real_motion" if index < 3 else "noise",
                    0.8,
                    setting="frame_width" if index < 2 else ("sensitivity" if index == 2 else ""),
                    value=480 if index < 2 else ("high" if index == 2 else None),
                ),
            )
            for index in range(5)
        ]

        report = aggregate_motion_ai_review(
            analyses,
            audits_considered=100,
            images_available=5,
            failed=0,
            current_settings={"frame_width": 320},
        )

        self.assertEqual(report["analyzed"], 5)
        self.assertEqual(report["verdict_counts"], {"real_motion": 3, "noise": 2})
        self.assertEqual(report["visible_subject_counts"], {"person": 3})
        self.assertEqual(len(report["recommendations"]), 1)
        recommendation = report["recommendations"][0]
        self.assertEqual(recommendation["setting"], "frame_width")
        self.assertEqual(recommendation["value"], 480)
        self.assertEqual(recommendation["current_value"], 320)
        self.assertEqual(recommendation["support_count"], 2)
        self.assertEqual(recommendation["evidence_audit_ids"], [0, 1])

    def test_aggregation_ignores_global_changes_for_per_camera_report(self) -> None:
        report = aggregate_motion_ai_review(
            [
                ({"id": 1, "reason": "noise"}, self.advice("noise", 0.9, setting="sample_fps", value=3, scope="global")),
                ({"id": 2, "reason": "noise"}, self.advice("noise", 0.9, setting="sample_fps", value=3, scope="global")),
            ],
            audits_considered=2,
            images_available=2,
            failed=0,
        )

        self.assertEqual(report["recommendations"], [])

    def test_review_progress_and_result_are_durable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            review = store.create_motion_ai_review("gate", 100)
            store.update_motion_ai_review(
                int(review["id"]),
                status="running",
                images_available=8,
                analyzed=3,
                failed=1,
            )
            completed = store.update_motion_ai_review(
                int(review["id"]),
                status="completed",
                images_available=8,
                analyzed=7,
                failed=1,
                result={"summary": "done", "recommendations": []},
            )

            reloaded = EventStore(Path(tmpdir)).latest_motion_ai_review("gate")

        self.assertEqual(completed["status"], "completed")
        self.assertEqual(reloaded["analyzed"], 7)
        self.assertEqual(reloaded["result"]["summary"], "done")

    def test_store_marks_incomplete_review_interrupted_on_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = EventStore(Path(tmpdir))
            review = first.create_motion_ai_review("gate", 20)
            first.update_motion_ai_review(int(review["id"]), status="running", analyzed=2)

            second = EventStore(Path(tmpdir))
            interrupted = second.get_motion_ai_review(int(review["id"]))
            ignored_late_update = first.update_motion_ai_review(
                int(review["id"]),
                status="completed",
                analyzed=20,
                result={"summary": "stale"},
            )

        self.assertEqual(interrupted["status"], "interrupted")
        self.assertEqual(ignored_late_update["status"], "interrupted")
        self.assertNotEqual(ignored_late_update["result"].get("summary"), "stale")


if __name__ == "__main__":
    unittest.main()

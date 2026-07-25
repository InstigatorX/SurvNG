from __future__ import annotations

import unittest
import time
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from survng.app.motion import (
    _legacy_score_motion_masks,
    difference_motion_frames,
    morphology_motion_masks,
    preprocess_motion_frames,
    qualify_motion,
    threshold_motion_differences,
)
from survng.app.motion_pipeline import (
    MotionContext,
    MotionPipelineFactory,
    MotionStageConfig,
    MotionStageDependencies,
    MotionStageRegistration,
    MotionStageRegistry,
    build_builtin_motion_registry,
    build_legacy_motion_pipeline,
    default_motion_stage_configs,
    motion_pipeline_catalog,
)


@dataclass
class RecordingStage:
    stage_id: str
    marker: str

    def process(self, context: MotionContext) -> MotionContext:
        context.debug.values.setdefault("order", []).append(self.marker)
        context.runtime.state_for(self.stage_id, list).append(context.captured_at)
        return context


@dataclass
class DelayedEvidenceStage:
    stage_id: str
    source: str
    delay: float

    def process(self, context: MotionContext) -> MotionContext:
        time.sleep(self.delay)
        context.source_evidence[self.source] = {"score": 0.5}
        return context


def build_recording_stage(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> RecordingStage:
    del dependencies
    return RecordingStage(stage_id, str(options.get("marker") or stage_id))


def build_delayed_evidence_stage(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> DelayedEvidenceStage:
    del dependencies
    return DelayedEvidenceStage(
        stage_id,
        str(options.get("source") or stage_id),
        float(options.get("delay") or 0.0),
    )


class MotionPipelineTest(unittest.TestCase):
    def test_parallel_group_runs_isolated_stages_concurrently_and_merges_evidence(self) -> None:
        registry = MotionStageRegistry()
        registry.register(MotionStageRegistration(
            implementation="delayed_evidence",
            builder=build_delayed_evidence_stage,
            provides=frozenset({"source_evidence"}),
        ))
        pipeline = MotionPipelineFactory(registry).create(
            "gate",
            [
                MotionStageConfig("first", "delayed_evidence", {"source": "one", "delay": 0.08}, "sources"),
                MotionStageConfig("second", "delayed_evidence", {"source": "two", "delay": 0.08}, "sources"),
            ],
        )
        context = MotionContext(
            camera_id="gate",
            captured_at=10.0,
            original_frame=None,
            configuration={},
            runtime=pipeline.runtime,
        )

        started = time.perf_counter()
        result = pipeline.process(context)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.14)
        self.assertEqual(set(result.source_evidence), {"one", "two"})
        self.assertEqual(set(result.timings), {"first", "second"})
        self.assertEqual(pipeline.status()["execution_groups"][0]["mode"], "parallel")
        pipeline.close()

    def test_parallel_group_rejects_conflicting_non_mergeable_outputs(self) -> None:
        registry = MotionStageRegistry()
        registry.register(MotionStageRegistration(
            implementation="score_a",
            builder=build_recording_stage,
            provides=frozenset({"scoring"}),
        ))

        with self.assertRaisesRegex(ValueError, "conflicting outputs: scoring"):
            MotionPipelineFactory(registry).create(
                "gate",
                [
                    MotionStageConfig("first", "score_a", parallel_group="scores"),
                    MotionStageConfig("second", "score_a", parallel_group="scores"),
                ],
            )

    def test_builtin_catalog_exposes_stages_options_and_available_presets(self) -> None:
        registry = build_builtin_motion_registry()
        catalog = motion_pipeline_catalog(registry)
        stages = {
            stage["implementation"]: stage
            for stage in catalog["stages"]
        }

        self.assertEqual(catalog["schema_version"], 1)
        self.assertEqual(stages["dominant_centroid"]["name"], "Centroid motion tracking")
        self.assertEqual(stages["dominant_centroid"]["graph"], "qualification")
        self.assertIn(
            "minimum_active_area_ratio",
            {option["key"] for option in stages["dominant_centroid"]["options"]},
        )
        self.assertTrue(all(preset["available"] for preset in catalog["presets"]))
        self.assertEqual(
            next(preset for preset in catalog["presets"] if preset["recommended"])["id"],
            "modular",
        )

    def test_legacy_pipeline_helper_builds_reference_stage(self) -> None:
        pipeline = build_legacy_motion_pipeline("gate")

        self.assertEqual([stage.stage_id for stage in pipeline.stages], ["qualification"])

    def test_factory_executes_registered_stages_in_order_and_records_timing(self) -> None:
        registry = MotionStageRegistry()
        registry.register(
            MotionStageRegistration(
                implementation="record",
                builder=build_recording_stage,
                provides=frozenset({"debug"}),
            )
        )
        pipeline = MotionPipelineFactory(registry).create(
            "gate",
            [
                MotionStageConfig("first", "record", {"marker": "one"}),
                MotionStageConfig("second", "record", {"marker": "two"}),
            ],
        )
        context = MotionContext(
            camera_id="gate",
            captured_at=10.0,
            original_frame=np.zeros((4, 4), dtype=np.uint8),
            configuration={},
            runtime=pipeline.runtime,
        )

        result = pipeline.process(context)

        self.assertIs(result, context)
        self.assertEqual(result.debug.values["order"], ["one", "two"])
        self.assertEqual(set(result.timings), {"first", "second"})
        self.assertEqual(pipeline.status()["stages"]["first"]["calls"], 1)
        self.assertEqual(pipeline.runtime.state_for("first", list), [10.0])

        snapshot = pipeline.audit_snapshot(result.timings)
        self.assertEqual(set(snapshot["invocation_timings"]), {"first", "second"})
        self.assertTrue(snapshot["invocation_timings"]["first"]["succeeded"])
        self.assertEqual(snapshot["stage_metrics"]["first"]["calls"], 1)

    def test_audit_snapshot_redacts_sensitive_stage_options(self) -> None:
        registry = MotionStageRegistry()
        registry.register(MotionStageRegistration(
            implementation="record",
            builder=build_recording_stage,
            provides=frozenset({"debug"}),
        ))
        pipeline = MotionPipelineFactory(registry).create(
            "gate",
            [MotionStageConfig("first", "record", {"marker": "one", "api_key": "secret"})],
        )

        configuration = pipeline.audit_snapshot()["configuration"]

        self.assertEqual(configuration[0]["options"]["marker"], "one")
        self.assertEqual(configuration[0]["options"]["api_key"], "[redacted]")

    def test_pipeline_rejects_context_from_another_camera_or_runtime(self) -> None:
        factory = MotionPipelineFactory(build_builtin_motion_registry())
        gate = factory.create("gate", [MotionStageConfig("qualification", "legacy_qualifier")])
        foyer = factory.create("foyer", [MotionStageConfig("qualification", "legacy_qualifier")])
        context = MotionContext(
            camera_id="gate",
            captured_at=10.0,
            original_frame=None,
            configuration={},
            runtime=foyer.runtime,
        )

        with self.assertRaisesRegex(ValueError, "per-camera runtime"):
            gate.process(context)

    def test_factory_validates_stage_artifact_dependencies(self) -> None:
        registry = MotionStageRegistry()
        registry.register(
            MotionStageRegistration(
                implementation="needs_mask",
                builder=build_recording_stage,
                requires=frozenset({"binary_motion_mask"}),
            )
        )

        with self.assertRaisesRegex(ValueError, "binary_motion_mask"):
            MotionPipelineFactory(registry).create(
                "gate",
                [MotionStageConfig("consumer", "needs_mask")],
            )

    def test_legacy_stage_matches_existing_qualifier(self) -> None:
        frames = []
        for index in range(9):
            frame = np.zeros((180, 320), dtype=np.uint8)
            cv2.rectangle(frame, (70 + index * 8, 60), (105 + index * 8, 125), 255, -1)
            frames.append(frame)
        prepared = preprocess_motion_frames(frames)
        differences = difference_motion_frames(prepared)
        masks = morphology_motion_masks(threshold_motion_differences(differences))
        expected = _legacy_score_motion_masks(prepared, masks, "balanced")
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "gate",
            [MotionStageConfig("qualification", "legacy_qualifier")],
        )
        context = MotionContext(
            camera_id="gate",
            captured_at=10.0,
            original_frame=frames[-1],
            frame_history=tuple(frames),
            configuration={"sensitivity": "balanced"},
            runtime=pipeline.runtime,
        )

        result = pipeline.process(context)

        self.assertEqual(result.scoring.accepted, expected.accepted)
        self.assertEqual(result.scoring.score, expected.score)
        self.assertEqual(result.scoring.threshold, expected.threshold)
        self.assertEqual(result.scoring.reason, expected.reason)
        self.assertEqual(result.scoring.features, expected.features)
        self.assertEqual(result.decision.run_object_detection, expected.accepted)

    def test_modular_image_stages_match_legacy_qualifier_and_expose_artifacts(self) -> None:
        frames = []
        for index in range(9):
            frame = np.zeros((180, 320), dtype=np.uint8)
            cv2.rectangle(frame, (70 + index * 8, 60), (105 + index * 8, 125), 255, -1)
            frames.append(frame)
        expected = qualify_motion(frames, "balanced")
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "gate",
            default_motion_stage_configs(),
        )
        context = MotionContext(
            camera_id="gate",
            captured_at=10.0,
            original_frame=frames[-1],
            frame_history=tuple(frames),
            configuration={"sensitivity": "balanced"},
            runtime=pipeline.runtime,
        )

        result = pipeline.process(context)

        self.assertEqual(result.scoring.accepted, expected.accepted)
        self.assertEqual(result.scoring.score, expected.score)
        self.assertEqual(result.scoring.threshold, expected.threshold)
        self.assertEqual(result.scoring.reason, expected.reason)
        self.assertEqual(result.scoring.features, expected.features)
        self.assertEqual(len(result.processed_frame_history), len(frames))
        self.assertEqual(len(result.difference_history), len(frames) - 1)
        self.assertEqual(len(result.threshold_mask_history), len(frames) - 1)
        self.assertEqual(len(result.motion_mask_history), len(frames) - 1)
        self.assertEqual(len(result.raw_blob_history), len(frames) - 1)
        self.assertEqual(len(result.filtered_blob_history), len(frames) - 1)
        self.assertTrue(result.blobs)
        self.assertIsNotNone(result.dominant_track)
        self.assertTrue(result.tracked_objects)
        self.assertEqual(result.dominant_track.score, expected.score)
        self.assertIs(result.processed_frame, result.processed_frame_history[-1])
        self.assertIs(result.difference_image, result.difference_history[-1])
        self.assertIs(result.binary_motion_mask, result.motion_mask_history[-1])
        self.assertEqual(
            list(result.timings),
            [
                "preprocess",
                "difference",
                "threshold",
                "morphology",
                "blob_extract",
                "blob_filter",
                "tracking",
                "scoring",
            ],
        )
        self.assertIsNone(result.decision)


if __name__ == "__main__":
    unittest.main()

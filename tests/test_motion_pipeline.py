from __future__ import annotations

import unittest
import threading
import time
from dataclasses import dataclass, field
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
    MotionRuntimeState,
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
class SnapshotOnlyState:
    value: int
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> "SnapshotOnlyState":
        return SnapshotOnlyState(self.value)


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


@dataclass
class ObservationStage:
    stage_id: str
    observation_kinds: frozenset[str]

    def process(self, context: MotionContext) -> MotionContext:
        context.debug.values.setdefault("observed_by", []).append(self.stage_id)
        return context


@dataclass
class FailingStage:
    stage_id: str

    def process(self, context: MotionContext) -> MotionContext:
        raise RuntimeError("parallel stage failed")


@dataclass
class CompletingStage:
    stage_id: str
    completed: threading.Event

    def process(self, context: MotionContext) -> MotionContext:
        time.sleep(0.05)
        self.completed.set()
        return context


@dataclass
class LifecycleStage:
    stage_id: str
    closed: bool = False

    def process(self, context: MotionContext) -> MotionContext:
        return context

    def close(self) -> None:
        self.closed = True


@dataclass
class LifecycleState:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


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


def build_frame_observer(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> ObservationStage:
    del options, dependencies
    return ObservationStage(stage_id, frozenset({"frame"}))


def build_event_observer(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> ObservationStage:
    del options, dependencies
    return ObservationStage(stage_id, frozenset({"motion_event"}))


def build_lifecycle_stage(
    stage_id: str,
    options: Mapping[str, Any],
    dependencies: MotionStageDependencies,
) -> LifecycleStage:
    del options, dependencies
    return LifecycleStage(stage_id)


class MotionPipelineTest(unittest.TestCase):
    def test_pipeline_close_releases_isolated_stage_and_runtime_resources(self) -> None:
        registry = MotionStageRegistry()
        registry.register(MotionStageRegistration(
            implementation="lifecycle",
            builder=build_lifecycle_stage,
        ))
        pipeline = MotionPipelineFactory(registry).create(
            "gate",
            [MotionStageConfig("resource", "lifecycle")],
        )
        replay = pipeline.isolated_copy()
        replay_stage = replay.stages[0]
        production_stage = pipeline.stages[0]
        runtime_state = replay.runtime.state_for("native", LifecycleState)

        replay.close()

        self.assertTrue(replay_stage.closed)
        self.assertTrue(runtime_state.closed)
        self.assertFalse(production_stage.closed)
        with self.assertRaisesRegex(RuntimeError, "pipeline is closed"):
            replay.process(MotionContext(
                camera_id="gate",
                captured_at=10.0,
                original_frame=None,
                configuration={},
                runtime=replay.runtime,
            ))
        pipeline.close()
        self.assertTrue(production_stage.closed)

    def test_observation_context_runs_only_stages_registered_for_its_kind(self) -> None:
        registry = MotionStageRegistry()
        registry.register(MotionStageRegistration(
            implementation="frame_observer",
            builder=build_frame_observer,
            provides=frozenset({"debug"}),
        ))
        registry.register(MotionStageRegistration(
            implementation="event_observer",
            builder=build_event_observer,
            provides=frozenset({"debug"}),
        ))
        pipeline = MotionPipelineFactory(registry).create(
            "gate",
            [
                MotionStageConfig("frame", "frame_observer"),
                MotionStageConfig("event", "event_observer"),
            ],
        )
        try:
            result = pipeline.process(MotionContext(
                camera_id="gate",
                captured_at=10.0,
                original_frame=None,
                configuration={"observation_kind": "motion_event"},
                runtime=pipeline.runtime,
            ))

            self.assertEqual(result.debug.values["observed_by"], ["event"])
            self.assertEqual(pipeline.status()["stages"]["frame"]["calls"], 0)
            self.assertEqual(pipeline.status()["stages"]["event"]["calls"], 1)
        finally:
            pipeline.close()

    def test_parallel_group_waits_for_siblings_before_raising_failure(self) -> None:
        completed = threading.Event()

        def build_failure(stage_id, options, dependencies):
            del options, dependencies
            return FailingStage(stage_id)

        def build_completion(stage_id, options, dependencies):
            del options, dependencies
            return CompletingStage(stage_id, completed)

        registry = MotionStageRegistry()
        registry.register(MotionStageRegistration(
            implementation="failure",
            builder=build_failure,
            provides=frozenset({"source_evidence"}),
        ))
        registry.register(MotionStageRegistration(
            implementation="completion",
            builder=build_completion,
            provides=frozenset({"source_evidence"}),
        ))
        pipeline = MotionPipelineFactory(registry).create(
            "gate",
            [
                MotionStageConfig("failure", "failure", parallel_group="sources"),
                MotionStageConfig("completion", "completion", parallel_group="sources"),
            ],
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "parallel stage failed"):
                pipeline.process(MotionContext(
                    camera_id="gate",
                    captured_at=10.0,
                    original_frame=None,
                    configuration={},
                    runtime=pipeline.runtime,
                ))

            self.assertTrue(completed.is_set())
            self.assertEqual(pipeline.status()["stages"]["completion"]["calls"], 1)
        finally:
            pipeline.close()

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
        self.assertTrue(stages["adaptive_ema_background"]["continuous_analysis"])
        self.assertEqual(stages["adaptive_ema_background"]["motion_source"], "adaptive_background")
        self.assertEqual(
            next(preset for preset in catalog["presets"] if preset["recommended"])["id"],
            "adaptive",
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

    def test_isolated_pipeline_clones_runtime_without_mutating_production_state(self) -> None:
        registry = MotionStageRegistry()
        registry.register(MotionStageRegistration(
            implementation="record",
            builder=build_recording_stage,
            provides=frozenset({"debug"}),
        ))
        pipeline = MotionPipelineFactory(registry).create(
            "gate",
            [MotionStageConfig("record", "record")],
        )
        pipeline.runtime.state_for("record", list).append(1.0)
        replay = pipeline.isolated_copy(clone_runtime=True)
        try:
            self.assertIsNot(replay.stages[0], pipeline.stages[0])
            result = replay.process(MotionContext(
                camera_id="gate",
                captured_at=2.0,
                original_frame=None,
                configuration={},
                runtime=replay.runtime,
            ))
            self.assertEqual(pipeline.runtime.state_for("record", list), [1.0])
            self.assertEqual(replay.runtime.state_for("record", list), [1.0, 2.0])
            replay_snapshot = replay.audit_snapshot(result.timings)
            self.assertEqual(replay_snapshot["stage_metrics"]["record"]["calls"], 1)
            self.assertTrue(replay_snapshot["invocation_timings"]["record"]["succeeded"])
        finally:
            replay.close()
            pipeline.close()

    def test_runtime_snapshot_contract_supports_native_noncopyable_state(self) -> None:
        runtime = MotionRuntimeState("gate")
        original = runtime.state_for("native", lambda: SnapshotOnlyState(7))

        cloned = runtime.clone()
        snapshot = cloned.state_for("native", lambda: SnapshotOnlyState(0))

        self.assertEqual(snapshot.value, 7)
        self.assertIsNot(snapshot, original)
        self.assertIsNot(snapshot.lock, original.lock)

    def test_pipeline_capabilities_are_derived_from_stage_registration(self) -> None:
        registry = MotionStageRegistry()
        registry.register(MotionStageRegistration(
            implementation="continuous_custom",
            builder=build_recording_stage,
            provides=frozenset({"debug"}),
            continuous_analysis=True,
            motion_source="custom_flow",
        ))
        pipeline = MotionPipelineFactory(registry).create(
            "gate",
            [MotionStageConfig("custom", "continuous_custom")],
        )
        try:
            self.assertTrue(pipeline.continuous_analysis)
            self.assertEqual(pipeline.primary_motion_source, "custom_flow")
        finally:
            pipeline.close()

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

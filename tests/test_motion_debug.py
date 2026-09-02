from __future__ import annotations

import unittest

import cv2
import numpy as np

from survng.app.motion_pipeline import (
    MotionContext,
    MotionDebugSnapshotStore,
    MotionPipelineFactory,
    build_builtin_motion_registry,
    default_motion_stage_configs,
    adaptive_motion_stage_configs,
)


class MotionDebugSnapshotTest(unittest.TestCase):
    def test_capture_encodes_bounded_layers_and_metadata(self) -> None:
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "gate",
            default_motion_stage_configs(),
        )
        frames = []
        for offset in range(6):
            frame = np.zeros((90, 160), dtype=np.uint8)
            cv2.rectangle(frame, (12 + offset * 9, 30), (40 + offset * 9, 62), 255, -1)
            frames.append(frame)
        context = pipeline.process(MotionContext(
            camera_id="gate",
            captured_at=100.0,
            original_frame=frames[-1],
            frame_history=tuple(frames),
            configuration={"sensitivity": "balanced"},
            runtime=pipeline.runtime,
        ))
        store = MotionDebugSnapshotStore()
        store.set_enabled(True)

        snapshot = store.capture(context)

        self.assertIsNotNone(snapshot)
        status = store.status()
        self.assertTrue(status["enabled"])
        self.assertEqual(status["snapshot"]["captured_at"], 100.0)
        layers = {layer["id"] for layer in status["snapshot"]["layers"]}
        self.assertTrue({"overlay", "original", "difference", "motion_mask"} <= layers)
        self.assertTrue(store.image("overlay").startswith(b"\xff\xd8"))
        self.assertLess(sum(len(image) for image in snapshot.images.values()), 500_000)
        pipeline.close()

    def test_disabling_clears_snapshot_and_images(self) -> None:
        store = MotionDebugSnapshotStore()
        store.set_enabled(True)
        store.set_enabled(False)

        self.assertFalse(store.status()["enabled"])
        self.assertIsNone(store.status()["snapshot"])
        self.assertIsNone(store.image("overlay"))

    def test_adaptive_pipeline_exposes_background_and_event_state(self) -> None:
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "gate",
            adaptive_motion_stage_configs(),
        )
        frames = []
        for offset in range(6):
            frame = np.zeros((90, 160), dtype=np.uint8)
            cv2.rectangle(frame, (20 + offset * 5, 25), (45 + offset * 5, 70), 180, -1)
            frames.append(frame)
        context = pipeline.process(MotionContext(
            camera_id="gate",
            captured_at=100.0,
            original_frame=frames[-1],
            frame_history=tuple(frames),
            configuration={
                "sensitivity": "balanced",
                "sample_fps": 5.0,
                "motion_zones": [{
                    "name": "Trees",
                    "exclude_from_ema": True,
                    "points": [
                        {"x": 0.0, "y": 0.0}, {"x": 0.25, "y": 0.0},
                        {"x": 0.25, "y": 1.0}, {"x": 0.0, "y": 1.0},
                    ],
                }],
            },
            runtime=pipeline.runtime,
        ))
        store = MotionDebugSnapshotStore()
        store.set_enabled(True)
        store.capture(context)

        status = store.status()["snapshot"]
        layers = {layer["id"] for layer in status["layers"]}
        self.assertIn("background", layers)
        self.assertIn("ema_exclusion", layers)
        self.assertEqual(status["event_state"], "idle")
        pipeline.close()

    def test_write_protected_gray_and_float_masks_encode(self) -> None:
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "gate",
            default_motion_stage_configs(),
        )
        frames = []
        for offset in range(4):
            frame = np.zeros((180, 320), dtype=np.uint8)
            cv2.rectangle(frame, (20 + offset * 8, 40), (60 + offset * 8, 90), 200, -1)
            frame.setflags(write=False)
            frames.append(frame)
        context = pipeline.process(MotionContext(
            camera_id="gate",
            captured_at=100.0,
            original_frame=frames[-1],
            frame_history=tuple(frames),
            configuration={"sensitivity": "balanced"},
            runtime=pipeline.runtime,
        ))
        context.original_frame.setflags(write=False)
        if context.processed_frame is not None:
            context.processed_frame.setflags(write=False)
        context.difference_image = np.full((180, 320), 0.4, dtype=np.float32)
        context.binary_motion_mask = np.zeros((180, 320), dtype=bool)
        context.binary_motion_mask[40:90, 20:80] = True
        store = MotionDebugSnapshotStore()
        store.set_enabled(True)

        snapshot = store.capture(context)

        self.assertIsNotNone(snapshot)
        self.assertIsNone(store.status()["last_error"])
        self.assertFalse(store.capture_due())
        layers = {layer["id"] for layer in store.status()["snapshot"]["layers"]}
        self.assertTrue({"overlay", "original", "difference", "motion_mask"} <= layers)
        self.assertTrue(store.image("original").startswith(b"\xff\xd8"))
        self.assertTrue(store.image("difference").startswith(b"\xff\xd8"))
        pipeline.close()

    def test_failed_layer_does_not_abort_snapshot(self) -> None:
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "gate",
            default_motion_stage_configs(),
        )
        frame = np.full((90, 160), 32, dtype=np.uint8)
        context = MotionContext(
            camera_id="gate",
            captured_at=100.0,
            original_frame=frame,
            processed_frame=frame,
            difference_image=np.zeros((0, 0), dtype=np.uint8),
            configuration={"sensitivity": "balanced"},
            runtime=pipeline.runtime,
        )
        store = MotionDebugSnapshotStore()
        store.set_enabled(True)

        snapshot = store.capture(context)

        self.assertIsNotNone(snapshot)
        layers = {layer["id"] for layer in store.status()["snapshot"]["layers"]}
        self.assertIn("original", layers)
        self.assertNotIn("difference", layers)
        pipeline.close()

    def test_status_exposes_last_error_when_no_layers_encode(self) -> None:
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "gate",
            default_motion_stage_configs(),
        )
        store = MotionDebugSnapshotStore()
        store.set_enabled(True)
        context = MotionContext(
            camera_id="gate",
            captured_at=100.0,
            original_frame=None,
            configuration={},
            runtime=pipeline.runtime,
        )

        self.assertIsNone(store.capture(context))
        self.assertIsNotNone(store.status()["last_error"])
        self.assertIsNone(store.status()["snapshot"])
        pipeline.close()


if __name__ == "__main__":
    unittest.main()

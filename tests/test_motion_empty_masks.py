from __future__ import annotations

import unittest
from dataclasses import fields, is_dataclass
from unittest.mock import patch

import cv2
import numpy as np

from survng.app.motion import morphology_motion_masks
from survng.app.motion_pipeline import (
    ConnectedComponentBlobStage,
    MotionContext,
    MotionPipelineFactory,
    MotionRuntimeState,
    adaptive_motion_stage_configs,
    build_builtin_motion_registry,
)
from survng.app.motion_types import MotionBlob, MotionFrameBlobs


def native_morphology(masks, kernel_size=3, close_iterations=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return [cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel),
        cv2.MORPH_CLOSE, kernel, iterations=close_iterations,
    ) for mask in masks]


def native_components(stage, context):
    """Unconditional native labeling oracle, including zero-mask observations."""
    height, width = context.processed_frame_history[0].shape[:2]
    area = max(1, width * height)
    history = []
    for index, mask in enumerate(context.motion_mask_history):
        if context.motion_inclusion_mask is not None:
            mask = cv2.bitwise_and(mask, context.motion_inclusion_mask)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        intensity = context.difference_history[index] if index < len(context.difference_history) else mask
        blobs = []
        for label in range(1, count):
            x, y, w, h, pixels = stats[label]
            cx, cy = centroids[label]
            edge = min(cx / width, cy / height, (width - cx) / width, (height - cy) / height)
            blobs.append(MotionBlob(
                box=(x / width, y / height, (x + w) / width, (y + h) / height),
                centroid=(float(cx) / width, float(cy) / height),
                area_pixels=float(pixels), area_ratio=float(pixels) / area,
                touches_edge=edge <= stage.edge_margin_ratio,
                fill_ratio=float(pixels) / (int(w) * int(h)),
                aspect_ratio=float(w) / float(h),
                average_motion_intensity=float(np.mean(intensity[labels == label])),
                edge_distance=float(edge),
                zone_overlap=0.0, ignored_zone_overlap=0.0, zone_names=(),
            ))
        changed = int(cv2.countNonZero(mask))
        history.append(MotionFrameBlobs(area, changed, changed / area, tuple(blobs)))
    context.raw_blob_history = tuple(history)
    return context


class EmptyMotionMasksTest(unittest.TestCase):
    def context(self, masks, inclusion=None):
        return MotionContext(
            camera_id="test", captured_at=100.0, original_frame=None,
            configuration={}, runtime=MotionRuntimeState("test"),
            processed_frame_history=(np.zeros((48, 64), np.uint8),),
            motion_mask_history=tuple(masks), motion_inclusion_mask=inclusion,
        )

    def assert_equivalent(self, actual, expected):
        if isinstance(expected, np.ndarray):
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_array_equal(actual, expected)
        elif is_dataclass(expected):
            for item in fields(expected):
                if item.name not in {"lock", "_lock", "timings"}:
                    self.assert_equivalent(getattr(actual, item.name), getattr(expected, item.name))
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_equivalent(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_equivalent(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_morphology_matches_native_and_owns_each_output(self):
        empty = np.zeros((48, 64), np.uint8)
        empty.flags.writeable = False
        sparse = empty.copy()
        sparse[20:23, 20:23] = 255  # Below 0.5%; must not be treated as empty.
        noise = np.random.default_rng(42).integers(0, 2, empty.shape, dtype=np.uint8) * 255
        masks = [empty, empty, empty[:, ::2], sparse, noise, np.full_like(empty, 255)]
        for kernel in (1, 3, 5):
            for iterations in (0, 1, 2):
                with self.subTest(kernel=kernel, iterations=iterations):
                    actual = morphology_motion_masks(masks, kernel, iterations)
                    self.assert_equivalent(actual, native_morphology(masks, kernel, iterations))
                    for index, result in enumerate(actual):
                        self.assertTrue(result.flags.writeable)
                        self.assertFalse(any(np.shares_memory(result, mask) for mask in masks))
                        self.assertFalse(any(np.shares_memory(result, other) for other in actual[:index]))
                    actual[0][0, 0] = 255
                    self.assertEqual(actual[1][0, 0], 0)
                    self.assertEqual(empty[0, 0], 0)

    def test_morphology_only_skips_exact_zero_binary_masks(self):
        empty = np.zeros((48, 64), np.uint8)
        single = empty.copy()
        single[20, 20] = 1
        with patch("survng.app.motion.cv2.morphologyEx", wraps=cv2.morphologyEx) as native:
            morphology_motion_masks([empty, single])
        self.assertEqual(native.call_count, 2)
        self.assertEqual(morphology_motion_masks([]), [])

    def test_morphology_retains_native_format_and_invalid_input_behavior(self):
        for mask in (
            np.zeros((0, 64), np.uint8), np.zeros((48, 64), np.int32),
            np.zeros((48, 64), bool), np.zeros((48, 64), np.float32),
            np.zeros((48, 64, 3), np.uint8),
        ):
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                try:
                    expected = native_morphology([mask])
                except cv2.error:
                    with self.assertRaises(cv2.error):
                        morphology_motion_masks([mask])
                else:
                    self.assert_equivalent(morphology_motion_masks([mask]), expected)

    def test_morphology_retains_native_iteration_parameter_behavior(self):
        empty = np.zeros((48, 64), np.uint8)
        for iterations in (1.5, True, 2**31, -1, np.int32(2), None):
            with self.subTest(iterations=iterations):
                try:
                    expected = native_morphology([empty], close_iterations=iterations)
                except (cv2.error, TypeError, OverflowError) as error:
                    with self.assertRaises(type(error)):
                        morphology_motion_masks([empty], close_iterations=iterations)
                else:
                    self.assert_equivalent(
                        morphology_motion_masks([empty], close_iterations=iterations), expected,
                    )

    def test_components_match_native_with_empty_and_excluded_observations(self):
        empty = np.zeros((48, 64), np.uint8)
        sparse = empty.copy()
        sparse[20:23, 20:23] = 255
        sparse[0, 0] = 1
        sparse.flags.writeable = False
        stage = ConnectedComponentBlobStage("components")
        for inclusion in (None, np.zeros_like(empty), np.full_like(empty, 255)):
            masks = [empty, sparse, empty]
            actual = stage.process(self.context(masks, inclusion))
            expected = native_components(stage, self.context(masks, inclusion))
            self.assert_equivalent(actual.raw_blob_history, expected.raw_blob_history)
            self.assertEqual(len(actual.raw_blob_history), 3)
            with patch("survng.app.motion_pipeline.adaptive_stages.cv2.connectedComponentsWithStats",
                       wraps=cv2.connectedComponentsWithStats) as native:
                stage.process(self.context(masks, inclusion))
            self.assertEqual(native.call_count, 0 if inclusion is not None and not inclusion.any() else 1)
        self.assertEqual(stage.process(self.context([])).raw_blob_history, ())

    def test_components_do_not_accept_unsupported_zero_masks(self):
        stage = ConnectedComponentBlobStage("components")
        for mask in (np.zeros((48, 64), np.float32), np.zeros((48, 64, 3), np.uint8)):
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with self.assertRaises(cv2.error):
                    native_components(stage, self.context([mask]))
                with self.assertRaises(cv2.error):
                    stage.process(self.context([mask]))

    def test_adaptive_sequences_match_native_outputs_and_all_runtime_state(self):
        quiet = np.full((90, 160), 30, np.uint8)

        def subject(x, size=12):
            frame = quiet.copy()
            frame[35:35 + size, x:x + size] = 180
            return frame

        sequences = {
            "quiet": [quiet] * 12,
            "small_motion": [quiet] * 3 + [subject(40 + i * 3, 7) for i in range(12)],
            "stationary": [quiet] * 3 + [subject(60)] * 30,
            "quiet_active_quiet": [quiet] * 3 + [subject(35 + i * 4) for i in range(12)] + [quiet] * 30,
        }
        for name, frames in sequences.items():
            with self.subTest(sequence=name):
                factory = MotionPipelineFactory(build_builtin_motion_registry())
                actual_pipeline = factory.create("test", adaptive_motion_stage_configs())
                native_pipeline = factory.create("test", adaptive_motion_stage_configs())
                saw_empty = saw_motion = saw_track = False
                try:
                    for index in range(1, len(frames)):
                        start = max(0, index - 3)

                        def context(pipeline):
                            return MotionContext(
                                camera_id="test", captured_at=100.0 + index * 0.2,
                                original_frame=frames[index],
                                frame_history=tuple(frames[start:index + 1]),
                                frame_timestamps=tuple(100.0 + i * 0.2 for i in range(start, index + 1)),
                                configuration={"sensitivity": "balanced", "sample_fps": 5.0},
                                runtime=pipeline.runtime,
                            )

                        actual = actual_pipeline.process(context(actual_pipeline))
                        with patch("survng.app.motion_pipeline.image_stages.morphology_motion_masks", native_morphology), \
                             patch.object(ConnectedComponentBlobStage, "process", native_components):
                            expected = native_pipeline.process(context(native_pipeline))
                        self.assert_equivalent(actual, expected)
                        saw_empty |= any(not item.blobs for item in actual.raw_blob_history)
                        saw_motion |= bool(actual.blobs)
                        saw_track |= bool(actual.tracked_objects)
                    self.assertTrue(saw_empty)
                    if name != "quiet":
                        self.assertTrue(saw_motion)
                        self.assertTrue(saw_track)
                    if name == "quiet_active_quiet":
                        self.assertEqual(actual.raw_blob_history[-1].blobs, ())
                finally:
                    actual_pipeline.close()
                    native_pipeline.close()


if __name__ == "__main__":
    unittest.main()

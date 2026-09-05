from __future__ import annotations

import unittest
import warnings
from unittest.mock import patch

import numpy as np

from survng.app.motion_pipeline.adaptive_stages import (
    AdaptiveStatisticalThresholdStage,
    _difference_statistics,
)
from survng.app.motion_pipeline.context import MotionContext
from survng.app.motion_pipeline.runtime import MotionRuntimeState


def numpy_statistics(difference: np.ndarray) -> tuple[float, float, float]:
    flat = difference.reshape(-1).astype(np.float32, copy=False)
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    return median, mad, float(np.percentile(flat, 80))


class MotionStatisticsTest(unittest.TestCase):
    def test_byte_statistics_exactly_match_numpy(self) -> None:
        rng = np.random.default_rng(1702)
        cases = [
            np.array([0, 255], dtype=np.uint8),
            np.array([0, 1, 2, 255], dtype=np.uint8),
            np.array([0, 0, 1, 2, 3, 255], dtype=np.uint8),
            np.full((360, 640), 37, dtype=np.uint8),
            rng.integers(0, 12, (360, 640), dtype=np.uint8),
            rng.integers(0, 256, (360, 640), dtype=np.uint8),
        ]
        # Exercise odd/even medians, each percentile interpolation position,
        # and wide jumps across the two percentile ranks.
        for size in [*range(1, 22), 101, 102, 1001, 1002, 230401]:
            cases.append(rng.integers(0, 256, size, dtype=np.uint8))
            boundary = int((size - 1) * 0.8) + 1
            cases.append(np.where(np.arange(size) < boundary, 0, 255).astype(np.uint8))
        sparse = np.zeros((360, 640), dtype=np.uint8)
        sparse[30:90, 80:120] = 255
        cases.extend([sparse, cases[-1][::-2], sparse.T, sparse[::3, ::2]])

        for index, difference in enumerate(cases):
            with self.subTest(index=index, shape=difference.shape):
                before = difference.copy()
                difference.flags.writeable = False
                self.assertEqual(_difference_statistics(difference), numpy_statistics(difference))
                np.testing.assert_array_equal(difference, before)

    def test_other_dtypes_and_nonfinite_values_preserve_numpy_behavior(self) -> None:
        cases = [
            np.array([-1.25, 0.1, 1.75, 256.5], dtype=np.float32),
            np.array([0.123456789, 2.987654321, 300.25], dtype=np.float64),
            np.array([-50, 0, 20, 1000], dtype=np.int16),
            np.array([0, 257, 65535], dtype=np.uint16),
            np.array([1, np.nan, 3], dtype=np.float32),
            np.array([-np.inf, 0, 3, np.inf], dtype=np.float64),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for difference in cases:
                with self.subTest(dtype=difference.dtype, values=difference):
                    np.testing.assert_array_equal(
                        _difference_statistics(difference), numpy_statistics(difference),
                    )

    def test_empty_input_preserves_numpy_error(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for dtype in [np.uint8, np.float32]:
                difference = np.empty((0, 3), dtype=dtype)
                try:
                    numpy_statistics(difference)
                except Exception as expected:
                    with self.assertRaises(type(expected)) as actual:
                        _difference_statistics(difference)
                    self.assertEqual(str(actual.exception), str(expected))
                else:
                    self.fail("The original statistics unexpectedly accepted an empty array")

    def test_threshold_masks_state_and_cache_match_numpy_across_overlap(self) -> None:
        rng = np.random.default_rng(91)
        differences = [rng.integers(0, 32, (18, 27), dtype=np.uint8) for _ in range(24)]
        actual_runtime = MotionRuntimeState("actual")
        expected_runtime = MotionRuntimeState("expected")
        actual_stage = AdaptiveStatisticalThresholdStage("threshold")
        expected_stage = AdaptiveStatisticalThresholdStage("threshold")
        windows = [(0, 5), (0, 5), (2, 8), (6, 14), (12, 24), (0, 5)]
        for start, end in windows:
            for timed in [True, False]:
                with self.subTest(start=start, end=end, timed=timed):
                    timestamps = tuple(float(index) for index in range(start, end + 1)) if timed else ()

                    def context(runtime: MotionRuntimeState) -> MotionContext:
                        return MotionContext(
                            camera_id=runtime.camera_id,
                            captured_at=float(end),
                            original_frame=None,
                            configuration={},
                            runtime=runtime,
                            difference_history=tuple(differences[start:end]),
                            frame_timestamps=timestamps,
                        )

                    actual = actual_stage.process(context(actual_runtime))
                    with patch(
                        "survng.app.motion_pipeline.adaptive_stages._difference_statistics",
                        side_effect=numpy_statistics,
                    ):
                        expected = expected_stage.process(context(expected_runtime))
                    for actual_mask, expected_mask in zip(
                        actual.threshold_mask_history, expected.threshold_mask_history, strict=True,
                    ):
                        np.testing.assert_array_equal(actual_mask, expected_mask)
                    self.assertEqual(actual.debug.values, expected.debug.values)
                    actual_state = actual_runtime.stage_state["threshold"]
                    expected_state = expected_runtime.stage_state["threshold"]
                    for attribute in ["threshold_ema", "noise_ema", "last_processed_at", "statistics"]:
                        self.assertEqual(getattr(actual_state, attribute), getattr(expected_state, attribute))
                    self.assertLessEqual(len(actual_state.statistics), 16)

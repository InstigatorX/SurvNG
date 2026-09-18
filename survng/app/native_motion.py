"""Bounded motion evidence from native boxes and stream time, without pixels."""
from collections import deque
from statistics import median


class NativeMotion:
    def __init__(self):
        self.samples = deque(maxlen=256)
        self.anchor = None
        self.quiet_since = None
        self.state = "uncertain"
        self.extent = 0.0

    @staticmethod
    def _metrics(coords):
        x1, y1, x2, y2 = coords
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            width,
            height,
        )

    def update(self, box, pts, policy, maximum_gap):
        if self.samples and pts - self.samples[-1][0] > maximum_gap:
            self.__init__()
        coords = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
        self.samples.append((pts, coords))
        # Retain five observations even when a low-rate stream cannot fill
        # the time window. This also permits trimming a single box outlier.
        while len(self.samples) > 5 and pts - self.samples[0][0] > policy.window_seconds:
            self.samples.popleft()
        if len(self.samples) < 5 or pts - self.samples[0][0] < 0.4:
            return self.state

        # Object detectors can move individual box edges by a pixel or two on a
        # completely static subject. Measure physical translation from the box
        # center instead of treating the noisiest edge as object motion. Scale
        # change remains evidence, but only after exceeding the policy's normal
        # stationary jitter allowance.
        columns = [
            sorted(self._metrics(sample[1])[axis] for sample in self.samples)
            for axis in range(4)
        ]
        current = tuple(median(column) for column in columns)
        if self.anchor is None:
            self.anchor = current

        width = max(1.0, (current[2] + self.anchor[2]) / 2)
        height = max(1.0, (current[3] + self.anchor[3]) / 2)
        scale = max(1.0, (width * height) ** 0.5)

        # Trim isolated extremes. Use the whole window's spread, not merely
        # first-to-last displacement, which misses movement away and back.
        trim = max(0, len(self.samples) // 5)
        center_spread = max(
            columns[0][-1 - trim] - columns[0][trim],
            columns[1][-1 - trim] - columns[1][trim],
        ) / scale
        center_drift = max(
            abs(current[0] - self.anchor[0]),
            abs(current[1] - self.anchor[1]),
        ) / scale

        # Size evidence uses the same characteristic object scale as center
        # translation. This avoids amplifying one-pixel detector jitter on the
        # narrow dimension of a small, tall box (for example a distant person).
        scale_spread = max(
            columns[2][-1 - trim] - columns[2][trim],
            columns[3][-1 - trim] - columns[3][trim],
        ) / scale
        scale_drift = max(
            abs(current[2] - self.anchor[2]),
            abs(current[3] - self.anchor[3]),
        ) / scale
        meaningful_scale_spread = max(
            0.0, scale_spread - policy.stationary_threshold
        )
        meaningful_scale_drift = max(
            0.0, scale_drift - policy.stationary_threshold
        )

        quiet_extent = max(center_spread, meaningful_scale_spread)
        self.extent = max(
            quiet_extent,
            center_drift,
            meaningful_scale_drift,
        )
        if self.extent >= policy.moving_threshold:
            self.state = "moving"
            self.quiet_since = None
            self.anchor = current
        elif quiet_extent <= policy.stationary_threshold:
            if self.quiet_since is None:
                self.quiet_since = pts
            if pts - self.quiet_since >= policy.stationary_seconds:
                if self.state != "stationary":
                    self.anchor = current
                self.state = "stationary"
        else:
            self.quiet_since = None
        return self.state

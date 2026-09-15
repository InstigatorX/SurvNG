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
        columns = [sorted(sample[1][axis] for sample in self.samples) for axis in range(4)]
        center = tuple(median(column) for column in columns)
        if self.anchor is None:
            self.anchor = center
        width = max(1.0, (center[2] - center[0] + self.anchor[2] - self.anchor[0]) / 2)
        height = max(1.0, (center[3] - center[1] + self.anchor[3] - self.anchor[1]) / 2)
        # Trim isolated extremes. Use the whole window's spread, not merely
        # first-to-last displacement, which misses movement away and back.
        trim = max(0, len(self.samples) // 5)
        spread = max((column[-1 - trim] - column[trim]) / (width if axis % 2 == 0 else height)
                     for axis, column in enumerate(columns))
        drift = max(abs(value - self.anchor[axis]) / (width if axis % 2 == 0 else height)
                    for axis, value in enumerate(center))
        self.extent = max(spread, drift)
        if self.extent >= policy.moving_threshold:
            self.state = "moving"
            self.quiet_since = None
            self.anchor = center
        elif spread <= policy.stationary_threshold:
            if self.quiet_since is None:
                self.quiet_since = pts
            if pts - self.quiet_since >= policy.stationary_seconds:
                if self.state != "stationary":
                    self.anchor = center
                self.state = "stationary"
        else:
            self.quiet_since = None
        return self.state

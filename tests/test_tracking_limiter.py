from __future__ import annotations

import unittest

from survng.app.object_tracking import AdaptiveTrackingLimiter


class AdaptiveTrackingLimiterTest(unittest.TestCase):
    def test_burst_slot_requires_healthy_guard(self) -> None:
        healthy = False
        limiter = AdaptiveTrackingLimiter(2, 3, burst_guard=lambda: healthy)
        self.assertTrue(limiter.acquire(blocking=False))
        self.assertTrue(limiter.acquire(blocking=False))
        self.assertFalse(limiter.acquire(blocking=False))
        healthy = True
        self.assertTrue(limiter.acquire(blocking=False))
        self.assertEqual(limiter.status()["burst_admissions"], 1)
        limiter.release()
        limiter.release()
        limiter.release()


if __name__ == "__main__":
    unittest.main()

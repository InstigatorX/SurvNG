from __future__ import annotations

import threading
import time
import unittest

from survng.app.motion_analysis import FairMotionAnalysisLimiter


class FairMotionAnalysisLimiterTest(unittest.TestCase):
    def test_waiting_camera_is_granted_before_repeat_camera(self) -> None:
        limiter = FairMotionAnalysisLimiter(1)
        order: list[str] = []
        started = threading.Event()
        release = threading.Event()

        def first() -> None:
            with limiter.acquire("gate"):
                order.append("gate-1")
                started.set()
                release.wait(timeout=1)
            with limiter.acquire("gate"):
                order.append("gate-2")

        def second() -> None:
            started.wait(timeout=1)
            with limiter.acquire("foyer"):
                order.append("foyer")

        gate = threading.Thread(target=first)
        foyer = threading.Thread(target=second)
        gate.start()
        foyer.start()
        started.wait(timeout=1)
        time.sleep(0.02)
        release.set()
        gate.join(timeout=1)
        foyer.join(timeout=1)

        self.assertEqual(order, ["gate-1", "foyer", "gate-2"])
        self.assertEqual(limiter.status(), {"capacity": 1, "active": 0, "pending": 0})


if __name__ == "__main__":
    unittest.main()

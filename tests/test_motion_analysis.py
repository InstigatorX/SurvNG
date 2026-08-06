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

    def test_waiting_camera_can_cancel_without_waiting_for_active_analysis(self) -> None:
        limiter = FairMotionAnalysisLimiter(1)
        active = threading.Event()
        release = threading.Event()
        cancelled = threading.Event()
        result: list[float | None] = []

        def holder() -> None:
            with limiter.acquire("gate"):
                active.set()
                release.wait(timeout=1)

        def waiter() -> None:
            active.wait(timeout=1)
            with limiter.acquire("foyer", cancel_event=cancelled) as wait_seconds:
                result.append(wait_seconds)

        holding_thread = threading.Thread(target=holder)
        waiting_thread = threading.Thread(target=waiter)
        holding_thread.start()
        waiting_thread.start()
        active.wait(timeout=1)
        deadline = time.monotonic() + 1
        while limiter.status()["pending"] != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        cancelled.set()
        waiting_thread.join(timeout=0.5)

        self.assertFalse(waiting_thread.is_alive())
        self.assertEqual(result, [None])
        self.assertEqual(limiter.status(), {"capacity": 1, "active": 1, "pending": 0})
        release.set()
        holding_thread.join(timeout=1)

    def test_newer_request_supersedes_same_camera_waiter_without_stranding_it(self) -> None:
        limiter = FairMotionAnalysisLimiter(1)
        holder_active = threading.Event()
        release_holder = threading.Event()
        first_done = threading.Event()
        results: list[tuple[str, float | None]] = []

        def holder() -> None:
            with limiter.acquire("boiler"):
                holder_active.set()
                release_holder.wait(timeout=1)

        def waiter(name: str, done: threading.Event | None = None) -> None:
            with limiter.acquire("gate") as waited:
                results.append((name, waited))
            if done is not None:
                done.set()

        holding = threading.Thread(target=holder)
        first = threading.Thread(target=waiter, args=("first", first_done))
        second = threading.Thread(target=waiter, args=("second",))
        holding.start()
        holder_active.wait(timeout=1)
        first.start()
        deadline = time.monotonic() + 1
        while limiter.status()["pending"] != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        second.start()

        self.assertTrue(first_done.wait(timeout=0.5))
        release_holder.set()
        holding.join(timeout=1)
        first.join(timeout=1)
        second.join(timeout=1)

        self.assertFalse(second.is_alive())
        self.assertIn(("first", None), results)
        self.assertTrue(any(name == "second" and waited is not None for name, waited in results))


if __name__ == "__main__":
    unittest.main()

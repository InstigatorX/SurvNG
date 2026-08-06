from __future__ import annotations

import threading
import time
import unittest

from survng.app.camera_startup import CameraStartupCoordinator, CameraStartupTask


def startup_task(
    camera_id: str,
    *,
    start_camera=lambda: None,
    capture_ready=lambda: True,
    start_recorders=lambda: None,
    publish_state=lambda: None,
    is_enabled=lambda: True,
) -> CameraStartupTask:
    return CameraStartupTask(
        camera_id=camera_id,
        is_enabled=is_enabled,
        start_camera=start_camera,
        capture_ready=capture_ready,
        start_recorders=start_recorders,
        publish_state=publish_state,
    )


class CameraStartupCoordinatorTest(unittest.TestCase):
    def test_camera_admission_never_exceeds_configured_concurrency(self) -> None:
        lock = threading.Lock()
        active = 0
        maximum = 0

        def task(camera_id: str) -> CameraStartupTask:
            def start() -> None:
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)

            def publish() -> None:
                nonlocal active
                time.sleep(0.02)
                with lock:
                    active -= 1

            return startup_task(camera_id, start_camera=start, publish_state=publish)

        coordinator = CameraStartupCoordinator(
            max_concurrency=2,
            recorder_settle_seconds=0.0,
        )
        coordinator.start([task(f"camera-{index}") for index in range(6)])

        self.assertTrue(coordinator.wait(timeout=1))
        self.assertEqual(maximum, 2)
        self.assertEqual(coordinator.status()["counts"], {"ready": 6})

    def test_missing_frame_is_degraded_but_still_starts_recorders(self) -> None:
        recorders_started = threading.Event()
        coordinator = CameraStartupCoordinator(
            readiness_timeout_seconds=0.02,
            recorder_settle_seconds=0.0,
            poll_interval_seconds=0.005,
        )
        coordinator.start([
            startup_task(
                "gate",
                capture_ready=lambda: False,
                start_recorders=recorders_started.set,
            )
        ])

        self.assertTrue(coordinator.wait(timeout=1))
        self.assertTrue(recorders_started.is_set())
        state = coordinator.status()["cameras"]["gate"]
        self.assertEqual(state["phase"], "degraded")
        self.assertGreaterEqual(state["wait_seconds"], 0.02)

    def test_one_camera_failure_does_not_block_later_camera(self) -> None:
        second_started = threading.Event()
        coordinator = CameraStartupCoordinator(
            max_concurrency=1,
            recorder_settle_seconds=0.0,
        )

        def fail() -> None:
            raise RuntimeError("camera refused connection")

        coordinator.start([
            startup_task("broken", start_camera=fail),
            startup_task("healthy", start_camera=second_started.set),
        ])

        self.assertTrue(coordinator.wait(timeout=1))
        self.assertTrue(second_started.is_set())
        status = coordinator.status()
        self.assertEqual(status["cameras"]["broken"]["phase"], "failed")
        self.assertEqual(status["cameras"]["healthy"]["phase"], "ready")

    def test_failure_status_redacts_stream_credentials(self) -> None:
        coordinator = CameraStartupCoordinator(recorder_settle_seconds=0.0)

        def fail() -> None:
            raise RuntimeError("failed rtsp://admin:secret@camera.local/main")

        coordinator.start([startup_task("gate", start_camera=fail)])

        self.assertTrue(coordinator.wait(timeout=1))
        error = coordinator.status()["cameras"]["gate"]["error"]
        self.assertNotIn("secret", error)
        self.assertIn("***", error)

    def test_state_publish_failure_does_not_change_camera_readiness(self) -> None:
        coordinator = CameraStartupCoordinator(recorder_settle_seconds=0.0)

        def fail_publish() -> None:
            raise RuntimeError("broker unavailable")

        coordinator.start([
            startup_task("gate", publish_state=fail_publish),
        ])

        self.assertTrue(coordinator.wait(timeout=1))
        self.assertEqual(coordinator.status()["cameras"]["gate"]["phase"], "ready")

    def test_disabled_camera_is_skipped_without_opening_capture(self) -> None:
        capture_started = threading.Event()
        state_published = threading.Event()
        coordinator = CameraStartupCoordinator(recorder_settle_seconds=0.0)
        coordinator.start([
            startup_task(
                "disabled",
                is_enabled=lambda: False,
                start_camera=capture_started.set,
                publish_state=state_published.set,
            )
        ])

        self.assertTrue(coordinator.wait(timeout=1))
        self.assertFalse(capture_started.is_set())
        self.assertTrue(state_published.is_set())
        self.assertEqual(
            coordinator.status()["cameras"]["disabled"]["phase"],
            "skipped",
        )

    def test_completion_callback_observes_complete_status(self) -> None:
        observed: list[bool] = []
        coordinator = CameraStartupCoordinator(recorder_settle_seconds=0.0)
        coordinator.start(
            [startup_task("gate")],
            on_complete=lambda: observed.append(coordinator.status()["complete"]),
        )

        self.assertTrue(coordinator.wait(timeout=1))
        deadline = time.monotonic() + 1
        while not observed and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(observed, [True])
        self.assertFalse(coordinator.status()["active"])

    def test_cancel_stops_active_wait_and_marks_queued_cameras(self) -> None:
        first_started = threading.Event()
        coordinator = CameraStartupCoordinator(
            max_concurrency=1,
            readiness_timeout_seconds=10.0,
            recorder_settle_seconds=0.0,
        )
        coordinator.start([
            startup_task(
                "first",
                start_camera=first_started.set,
                capture_ready=lambda: False,
            ),
            startup_task("second"),
        ])
        self.assertTrue(first_started.wait(timeout=1))

        self.assertTrue(coordinator.cancel(timeout=1))

        status = coordinator.status()
        self.assertEqual(status["cameras"]["first"]["phase"], "cancelled")
        self.assertEqual(status["cameras"]["second"]["phase"], "cancelled")


if __name__ == "__main__":
    unittest.main()

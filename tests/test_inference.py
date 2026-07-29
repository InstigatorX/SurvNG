from __future__ import annotations

import os
from multiprocessing import resource_tracker
from pathlib import Path
import signal
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pydantic import ValidationError

from survng.app.config import DetectorConfig
from survng.app.inference import (
    InferenceSupervisor,
    InferenceUnavailable,
    IsolatedFaceRecognizer,
    IsolatedPersonReidentifier,
    PERSON_REID_REQUEST_TIMEOUT_SECONDS,
    _InferenceWorker,
)


class InferenceSupervisorTest(unittest.TestCase):
    def test_detector_threshold_configuration_is_bounded(self) -> None:
        with self.assertRaises(ValidationError):
            DetectorConfig(confidence_threshold=0.0)
        with self.assertRaises(ValidationError):
            DetectorConfig(nms_threshold=1.0)
        with self.assertRaises(ValidationError):
            DetectorConfig(backend="unknown")

    def setUp(self) -> None:
        self.supervisor = InferenceSupervisor(DetectorConfig(enabled=False))

    def tearDown(self) -> None:
        self.supervisor.stop()

    def test_disabled_detector_uses_isolated_worker_contract(self) -> None:
        self.assertTrue(self.supervisor.start())
        status = self.supervisor.status()

        self.assertTrue(status["isolation"]["worker_alive"])
        self.assertIsNotNone(status["isolation"]["worker_pid"])
        comm_path = Path(f"/proc/{status['isolation']['worker_pid']}/comm")
        if comm_path.exists():
            self.assertEqual(comm_path.read_text(encoding="utf-8"), "survng-object\n")
        self.assertEqual(status["workers"]["object"]["role"], "object")
        self.assertFalse(status["workers"]["face"]["enabled"])
        self.assertFalse(status["workers"]["reid"]["enabled"])
        self.assertFalse(status["enabled"])

        result = self.supervisor.detect(np.zeros((24, 32, 3), dtype=np.uint8))
        self.assertEqual(result, [{"status": "detector_unavailable"}])

    def test_worker_restarts_after_native_style_process_death(self) -> None:
        self.assertTrue(self.supervisor.start())
        first_pid = self.supervisor.isolation_status()["worker_pid"]
        self.assertIsNotNone(first_pid)

        os.kill(int(first_pid), signal.SIGKILL)
        process = self.supervisor._object._process
        self.assertIsNotNone(process)
        process.join(timeout=3.0)

        failed_status = self.supervisor.status()["isolation"]
        self.assertFalse(failed_status["worker_alive"])
        self.assertEqual(failed_status["last_exit_code"], -signal.SIGKILL)
        self.assertEqual(failed_status["crash_count"], 1)

        time.sleep(1.1)
        recovered_status = self.supervisor.status()["isolation"]
        self.assertTrue(recovered_status["worker_alive"])
        self.assertNotEqual(recovered_status["worker_pid"], first_pid)
        self.assertEqual(recovered_status["restart_count"], 1)

    def test_face_proxy_reports_worker_status(self) -> None:
        self.assertTrue(self.supervisor.start())
        proxy = IsolatedFaceRecognizer(self.supervisor)

        self.assertFalse(proxy.ready)
        self.assertFalse(proxy.enabled)
        self.assertEqual(proxy.status()["device"], "AUTO")
        self.assertFalse(proxy.status()["isolation"]["enabled"])

    def test_reid_proxy_only_supports_labels_with_a_ready_model(self) -> None:
        config = DetectorConfig.model_validate({
            "tracking": {
                "reid_enabled": True,
                "reid_model_path": "person.xml",
                "vehicle_reid_enabled": True,
                "vehicle_reid_model_path": "vehicle.xml",
            },
        })
        supervisor = Mock(config=config)
        supervisor.cached_reid_status.return_value = {
            "ready": False,
            "person": {"ready": False},
            "vehicle": {"ready": True},
        }
        proxy = IsolatedPersonReidentifier(supervisor)

        self.assertFalse(proxy.supports_label("person"))
        self.assertTrue(proxy.supports_label("car"))
        self.assertFalse(proxy.supports_label("dog"))

    def test_cached_worker_status_does_not_report_stale_readiness_after_exit(self) -> None:
        worker = _InferenceWorker(
            DetectorConfig(enabled=False),
            "reid",
            {
                "ready": True,
                "person": {"ready": True},
                "vehicle": {"ready": True},
            },
            start_enabled=False,
        )
        worker._process = Mock(is_alive=Mock(return_value=False))

        status = worker.cached_status()

        self.assertFalse(status["ready"])
        self.assertFalse(status["person"]["ready"])
        self.assertFalse(status["vehicle"]["ready"])

    def test_person_reid_proxy_reports_disabled_worker_status(self) -> None:
        self.assertTrue(self.supervisor.start())
        proxy = IsolatedPersonReidentifier(self.supervisor)

        self.assertFalse(proxy.ready)
        self.assertFalse(proxy.enabled)
        self.assertEqual(proxy.status()["device"], "AUTO")
        self.assertFalse(proxy.status()["isolation"]["enabled"])
        self.assertEqual(
            proxy.status()["isolation"]["request_timeout_seconds"],
            PERSON_REID_REQUEST_TIMEOUT_SECONDS,
        )

    def test_person_reid_requests_use_short_timeout(self) -> None:
        with patch.object(
            self.supervisor._reid,
            "request",
            return_value=[1.0, 0.0],
        ) as request:
            result = self.supervisor.embed_person(np.zeros((32, 16, 3), dtype=np.uint8))

        self.assertEqual(result.tolist(), [1.0, 0.0])
        request.assert_called_once()
        self.assertEqual(request.call_args.kwargs["timeout"], PERSON_REID_REQUEST_TIMEOUT_SECONDS)

    def test_face_worker_crash_does_not_restart_object_worker(self) -> None:
        supervisor = InferenceSupervisor(
            DetectorConfig(enabled=False, face_recognition_enabled=True)
        )
        self.addCleanup(supervisor.stop)
        self.assertTrue(supervisor.start())
        workers = supervisor.worker_status()
        object_pid = workers["object"]["worker_pid"]
        face_pid = workers["face"]["worker_pid"]
        self.assertIsNotNone(object_pid)
        self.assertIsNotNone(face_pid)
        self.assertNotEqual(object_pid, face_pid)
        face_comm_path = Path(f"/proc/{face_pid}/comm")
        if face_comm_path.exists():
            self.assertEqual(face_comm_path.read_text(encoding="utf-8"), "survng-face\n")

        os.kill(int(face_pid), signal.SIGKILL)
        face_process = supervisor._face._process
        self.assertIsNotNone(face_process)
        face_process.join(timeout=3.0)

        failed_face = supervisor.face_status()["isolation"]
        self.assertFalse(failed_face["worker_alive"])
        self.assertEqual(failed_face["last_exit_code"], -signal.SIGKILL)
        self.assertEqual(supervisor.worker_status()["object"]["worker_pid"], object_pid)

        time.sleep(1.1)
        recovered_face = supervisor.face_status()["isolation"]
        self.assertTrue(recovered_face["worker_alive"])
        self.assertNotEqual(recovered_face["worker_pid"], face_pid)
        self.assertEqual(recovered_face["restart_count"], 1)
        self.assertEqual(supervisor.worker_status()["object"]["worker_pid"], object_pid)

    def test_stop_reaps_both_workers(self) -> None:
        supervisor = InferenceSupervisor(
            DetectorConfig(enabled=False, face_recognition_enabled=True)
        )
        self.assertTrue(supervisor.start())
        object_process = supervisor._object._process
        face_process = supervisor._face._process
        self.assertIsNotNone(object_process)
        self.assertIsNotNone(face_process)

        supervisor.stop()

        self.assertFalse(object_process.is_alive())
        self.assertFalse(face_process.is_alive())
        self.assertIsNone(supervisor.worker_status()["object"]["worker_pid"])
        self.assertIsNone(supervisor.worker_status()["face"]["worker_pid"])

    def test_supervisor_attempts_every_worker_stop_after_failure(self) -> None:
        supervisor = InferenceSupervisor(DetectorConfig(enabled=False))
        supervisor._reid.stop = Mock(side_effect=RuntimeError("reid stuck"))
        supervisor._face.stop = Mock()
        supervisor._object.stop = Mock()

        with self.assertRaisesRegex(RuntimeError, "reid"):
            supervisor.stop()

        supervisor._face.stop.assert_called_once_with()
        supervisor._object.stop.assert_called_once_with()

    def test_supervisor_preserves_base_exception_after_stopping_every_worker(self) -> None:
        supervisor = InferenceSupervisor(DetectorConfig(enabled=False))
        supervisor._reid.stop = Mock(side_effect=KeyboardInterrupt())
        supervisor._face.stop = Mock()
        supervisor._object.stop = Mock()

        with self.assertRaises(KeyboardInterrupt):
            supervisor.stop()

        supervisor._face.stop.assert_called_once_with()
        supervisor._object.stop.assert_called_once_with()

    def test_final_shutdown_reaps_multiprocessing_resource_tracker(self) -> None:
        self.assertTrue(self.supervisor.start())
        tracker_pid = resource_tracker._resource_tracker._pid
        self.assertIsNotNone(tracker_pid)

        self.supervisor.stop()
        stopped = self.supervisor.stop_resource_tracker()

        self.assertTrue(stopped)
        self.assertIsNone(resource_tracker._resource_tracker._pid)
        self.assertFalse(Path(f"/proc/{tracker_pid}").exists())

    def test_resource_tracker_is_preserved_while_worker_is_alive(self) -> None:
        self.assertTrue(self.supervisor.start())

        with patch(
            "survng.app.inference.stop_multiprocessing_resource_tracker"
        ) as stop_tracker:
            stopped = self.supervisor.stop_resource_tracker()

        self.assertFalse(stopped)
        stop_tracker.assert_not_called()

    def test_request_timeout_includes_time_waiting_for_worker_lock(self) -> None:
        worker = _InferenceWorker(
            DetectorConfig(enabled=False),
            "object",
            {},
            start_enabled=False,
        )
        acquired = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            with worker._lock:
                acquired.set()
                release.wait(timeout=2.0)

        thread = threading.Thread(target=hold_lock)
        thread.start()
        self.assertTrue(acquired.wait(timeout=1.0))
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(InferenceUnavailable, "timed out waiting"):
                worker.request("status", timeout=0.05)
        finally:
            release.set()
            thread.join(timeout=1.0)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(worker.isolation_status()["pending_requests"], 0)

    def test_shared_frame_contract_rejects_non_bgr_and_non_uint8_arrays(self) -> None:
        worker = _InferenceWorker(DetectorConfig(enabled=False), "object", {})
        with self.assertRaisesRegex(InferenceUnavailable, "uint8 BGR"):
            worker._write_frame_locked(np.zeros((10, 10), dtype=np.uint8))
        with self.assertRaisesRegex(InferenceUnavailable, "uint8 BGR"):
            worker._write_frame_locked(np.zeros((10, 10, 3), dtype=np.float32))

    def test_stop_retains_and_reports_worker_that_survives_forced_termination(self) -> None:
        class StubbornProcess:
            exitcode = None

            @staticmethod
            def is_alive() -> bool:
                return True

            @staticmethod
            def join(timeout=None) -> None:
                return None

            @staticmethod
            def terminate() -> None:
                return None

            @staticmethod
            def kill() -> None:
                return None

        worker = _InferenceWorker(
            DetectorConfig(enabled=False),
            "object",
            {},
            start_enabled=False,
        )
        process = StubbornProcess()
        frame_buffer = object()
        worker._process = process
        worker._frame_buffer = frame_buffer

        with self.assertRaisesRegex(RuntimeError, "did not stop"):
            worker.stop()

        self.assertIs(worker._process, process)
        self.assertIs(worker._frame_buffer, frame_buffer)

    def test_stop_clears_state_when_connection_close_fails(self) -> None:
        worker = _InferenceWorker(
            DetectorConfig(enabled=False),
            "object",
            {},
            start_enabled=False,
        )
        connection = Mock()
        connection.close.side_effect = OSError("already closed")
        worker._connection = connection
        worker._frame_buffer = object()

        with self.assertLogs("uvicorn.error", level="ERROR"):
            worker.stop()

        self.assertIsNone(worker._connection)
        self.assertIsNone(worker._frame_buffer)


if __name__ == "__main__":
    unittest.main()

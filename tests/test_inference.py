from __future__ import annotations

import os
from pathlib import Path
import signal
import time
import unittest

import numpy as np

from survng.app.config import DetectorConfig
from survng.app.inference import InferenceSupervisor, IsolatedFaceRecognizer


class InferenceSupervisorTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

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

from survng.app.config import DepthConfig, DetectorConfig, ObjectTrackingConfig
from survng.app.inference import (
    InferenceRollbackIncomplete,
    InferenceWorkload,
    InferenceSupervisor,
    InferenceUnavailable,
    IsolatedFaceRecognizer,
    IsolatedPersonReidentifier,
    PERSON_REID_REQUEST_TIMEOUT_SECONDS,
    _InferenceWorker,
)


def _single_object_detector(**overrides) -> DetectorConfig:
    """Disabled detector with one object worker and tracking off.

    Tracking would otherwise raise the OpenVINO worker floor to 2, which
    resizes the pool during tests that only change device, cache, or NMS.
    """
    values: dict = {
        "enabled": False,
        "object_worker_count": 1,
        "tracking": {"enabled": False},
    }
    values.update(overrides)
    return DetectorConfig(**values)


class InferenceSupervisorTest(unittest.TestCase):
    def test_worker_rollback_failure_raises_fencing_exception(self) -> None:
        worker = _InferenceWorker(
            DetectorConfig(enabled=False),
            "object",
            {},
        )
        worker._process = Mock()
        worker._process.is_alive.return_value = True

        with (
            patch.object(
                worker,
                "stop",
                side_effect=[None, RuntimeError("rollback stop failed")],
            ),
            patch.object(worker, "start", side_effect=[False, False]),
            self.assertRaises(InferenceRollbackIncomplete),
        ):
            worker.reconfigure(
                DetectorConfig(enabled=False),
                {},
                start_enabled=True,
            )

    def test_auxiliary_workers_treat_auto_as_cpu(self) -> None:
        config = DetectorConfig(
            enabled=False,
            face_recognition_device="AUTO",
            depth=DepthConfig(device="AUTO"),
            tracking=ObjectTrackingConfig(
                reid_enabled=True,
                reid_model_path="/tmp/person-reid.xml",
                reid_device="AUTO",
                vehicle_reid_enabled=True,
                vehicle_reid_model_path="/tmp/vehicle-reid.xml",
                vehicle_reid_device="GPU",
            ),
        )
        face = _InferenceWorker(config, "face", {}, start_enabled=False)
        depth = _InferenceWorker(config, "depth", {}, start_enabled=False)
        reid = _InferenceWorker(config, "reid", {}, start_enabled=False)

        self.assertEqual(face.configured_device, "CPU")
        self.assertEqual(depth.configured_device, "CPU")
        self.assertEqual(reid.configured_device, "CPU")
        payload = face._active_config_payload()
        self.assertEqual(payload["face_recognition_device"], "CPU")
        self.assertEqual(payload["depth"]["device"], "CPU")
        self.assertEqual(payload["tracking"]["reid_device"], "CPU")
        self.assertEqual(payload["tracking"]["vehicle_reid_device"], "GPU")

    def test_detector_threshold_configuration_is_bounded(self) -> None:
        with self.assertRaises(ValidationError):
            DetectorConfig(confidence_threshold=0.0)
        with self.assertRaises(ValidationError):
            DetectorConfig(nms_threshold=1.0)
        with self.assertRaises(ValidationError):
            DetectorConfig(backend="unknown")
        with self.assertRaises(ValidationError):
            DetectorConfig(object_worker_count=0)
        with self.assertRaises(ValidationError):
            DetectorConfig(object_worker_count=5)
        with self.assertRaises(ValidationError):
            DetectorConfig(object_activity_attribution="unsupported")

    def setUp(self) -> None:
        self.supervisor = InferenceSupervisor(_single_object_detector())

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

    def test_multiple_object_workers_start_report_and_stop_independently(self) -> None:
        supervisor = InferenceSupervisor(
            DetectorConfig(enabled=False, object_worker_count=2)
        )
        self.addCleanup(supervisor.stop)

        self.assertTrue(supervisor.start())
        status = supervisor.status()
        isolation = status["isolation"]

        self.assertEqual(isolation["configured_workers"], 2)
        self.assertEqual(isolation["alive_workers"], 2)
        self.assertTrue(isolation["all_workers_alive"])
        self.assertEqual(len(set(isolation["worker_pids"])), 2)
        self.assertEqual(len(status["runtime"]["workers"]), 2)

        processes = [worker._process for worker in supervisor._object_workers]
        supervisor.stop()
        self.assertTrue(all(process is not None and not process.is_alive() for process in processes))

    def test_multi_worker_status_polling_is_concurrent(self) -> None:
        supervisor = InferenceSupervisor(
            DetectorConfig(enabled=False, object_worker_count=2)
        )
        self.addCleanup(supervisor.stop)
        barrier = threading.Barrier(2)

        def synchronized_status() -> dict[str, object]:
            barrier.wait(timeout=1.0)
            return supervisor._base_detector_status()

        for worker in supervisor._object_workers:
            worker.status = Mock(side_effect=synchronized_status)
        supervisor._reid.status = Mock(return_value=supervisor._base_reid_status())

        status = supervisor.status()

        self.assertEqual(status["object_worker_count"], 2)
        self.assertEqual(barrier.n_waiting, 0)

    def test_object_requests_rotate_across_equally_idle_workers(self) -> None:
        supervisor = InferenceSupervisor(
            DetectorConfig(enabled=False, object_worker_count=2)
        )
        self.addCleanup(supervisor.stop)
        workers = supervisor._object_workers
        for index, worker in enumerate(workers):
            worker.pending_requests = Mock(return_value=0)
            worker.request = Mock(return_value=[{"worker": index + 1}])

        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        first = supervisor.detect(frame)
        second = supervisor.detect(frame)

        self.assertEqual(first, [{"worker": 1}])
        self.assertEqual(second, [{"worker": 2}])
        workers[0].request.assert_called_once()
        workers[1].request.assert_called_once()

    def test_object_request_fails_over_to_another_pool_worker(self) -> None:
        supervisor = InferenceSupervisor(
            DetectorConfig(enabled=False, object_worker_count=2)
        )
        self.addCleanup(supervisor.stop)
        first, second = supervisor._object_workers
        first.pending_requests = Mock(return_value=0)
        second.pending_requests = Mock(return_value=0)
        first.request = Mock(side_effect=InferenceUnavailable("worker one unavailable"))
        second.request = Mock(return_value=[{"label": "person"}])

        result = supervisor.detect(np.zeros((24, 32, 3), dtype=np.uint8))

        self.assertEqual(result, [{"label": "person"}])
        first.request.assert_called_once()
        second.request.assert_called_once()

    def test_initial_detection_uses_bounded_per_worker_failover(self) -> None:
        supervisor = InferenceSupervisor(
            DetectorConfig(enabled=False, object_worker_count=2)
        )
        self.addCleanup(supervisor.stop)
        first, second = supervisor._object_workers
        first.pending_requests = Mock(return_value=0)
        second.pending_requests = Mock(return_value=0)
        first.request = Mock(side_effect=InferenceUnavailable("worker one wedged"))
        second.request = Mock(return_value=[{"label": "person"}])

        result = supervisor.detect_initial(
            np.zeros((24, 32, 3), dtype=np.uint8)
        )

        self.assertEqual(result, [{"label": "person"}])
        self.assertEqual(first.request.call_args.kwargs["timeout"], 3.0)
        self.assertEqual(first.request.call_args.kwargs["admission_timeout"], 0.75)
        self.assertEqual(
            second.request.call_args.kwargs["workload"],
            InferenceWorkload.INCIDENT_INITIAL,
        )

    def test_tracking_uses_only_non_reserved_object_worker(self) -> None:
        supervisor = InferenceSupervisor(
            DetectorConfig(enabled=False, object_worker_count=2)
        )
        self.addCleanup(supervisor.stop)
        reserved, background = supervisor._object_workers
        reserved.pending_requests = Mock(return_value=0)
        background.pending_requests = Mock(return_value=0)
        reserved.request = Mock(return_value=[{"worker": 1}])
        background.request = Mock(return_value=[{"worker": 2}])

        result = supervisor.detect_tracking(
            np.zeros((24, 32, 3), dtype=np.uint8)
        )

        self.assertEqual(result, [{"worker": 2}])
        reserved.request.assert_not_called()
        self.assertEqual(
            background.request.call_args.kwargs["workload"],
            InferenceWorkload.TRACKING,
        )

    def test_security_workload_quiesces_and_sheds_optional_gpu_work(self) -> None:
        self.assertTrue(
            self.supervisor._enter_device_workload(InferenceWorkload.TRACKING)
        )
        incident_admitted = threading.Event()

        def admit_incident() -> None:
            if self.supervisor._enter_device_workload(
                InferenceWorkload.INCIDENT_INITIAL
            ):
                incident_admitted.set()

        thread = threading.Thread(target=admit_incident)
        thread.start()
        with self.supervisor._device_condition:
            self.assertTrue(self.supervisor._device_condition.wait_for(
                lambda: self.supervisor._security_waiting == 1,
                timeout=1.0,
            ))
        self.assertFalse(
            self.supervisor._enter_device_workload(InferenceWorkload.ENRICHMENT)
        )
        self.supervisor._leave_device_workload(InferenceWorkload.TRACKING)
        self.assertTrue(incident_admitted.wait(1.0))
        self.supervisor._leave_device_workload(InferenceWorkload.INCIDENT_INITIAL)
        thread.join(timeout=1.0)

        status = self.supervisor.workload_status()
        self.assertEqual(status["classes"]["enrichment"]["shed"], 1)
        self.assertEqual(status["classes"]["incident_initial"]["admitted"], 1)

    def test_interactive_work_is_single_flight_and_yields_to_security(self) -> None:
        self.assertTrue(
            self.supervisor._enter_device_workload(
                InferenceWorkload.INTERACTIVE,
                shed_optional=True,
            )
        )
        self.assertFalse(
            self.supervisor._enter_device_workload(
                InferenceWorkload.INTERACTIVE,
                shed_optional=True,
            )
        )
        incident_admitted = threading.Event()

        def admit_incident() -> None:
            if self.supervisor._enter_device_workload(
                InferenceWorkload.INCIDENT_INITIAL,
            ):
                incident_admitted.set()

        thread = threading.Thread(target=admit_incident)
        thread.start()
        self.assertFalse(incident_admitted.wait(0.05))
        self.supervisor._leave_device_workload(InferenceWorkload.INTERACTIVE)
        self.assertTrue(incident_admitted.wait(1.0))
        self.supervisor._leave_device_workload(InferenceWorkload.INCIDENT_INITIAL)
        thread.join(timeout=1.0)

        status = self.supervisor.workload_status()
        self.assertEqual(status["interactive_active"], 0)
        self.assertEqual(status["classes"]["interactive"]["shed"], 1)

    def test_optional_depth_requests_are_single_flight(self) -> None:
        supervisor = InferenceSupervisor(DetectorConfig.model_validate({
            "depth": {"enabled": True, "model_path": "depth.xml"},
        }))
        self.addCleanup(supervisor.stop)
        self.assertTrue(supervisor._depth_optional_slot.acquire(blocking=False))

        objects = [{"label": "person", "box": [0, 0, 10, 10]}]
        enriched, metadata = supervisor.estimate_depth_for_objects(
            np.zeros((12, 12, 3), dtype=np.uint8),
            objects,
            workload=InferenceWorkload.INTERACTIVE,
        )

        self.assertEqual(enriched, objects)
        self.assertEqual(metadata, {"status": "depth_deferred"})
        supervisor._depth_optional_slot.release()

    def test_depth_worker_failure_does_not_expose_native_error(self) -> None:
        supervisor = InferenceSupervisor(DetectorConfig.model_validate({
            "depth": {"enabled": True, "model_path": "depth.xml"},
        }))
        self.addCleanup(supervisor.stop)
        supervisor._depth.request = Mock(
            side_effect=RuntimeError("/private/models/depth.xml failed on GPU.0")
        )

        _objects, metadata = supervisor.estimate_depth_for_objects(
            np.zeros((12, 12, 3), dtype=np.uint8),
            [{"label": "person", "box": [0, 0, 10, 10]}],
        )

        self.assertEqual(metadata["status"], "depth_error")
        self.assertEqual(
            metadata["error"],
            "Depth estimation failed in the isolated worker.",
        )
        self.assertNotIn("/private", metadata["error"])

    def test_offline_lease_is_exclusive_with_optional_device_work(self) -> None:
        self.assertTrue(
            self.supervisor._enter_device_workload(InferenceWorkload.TRACKING)
        )
        offline_admitted = threading.Event()

        def admit_offline() -> None:
            if self.supervisor._enter_device_workload(
                InferenceWorkload.OFFLINE,
                shed_optional=False,
                timeout=1.0,
            ):
                offline_admitted.set()

        thread = threading.Thread(target=admit_offline)
        thread.start()
        self.assertFalse(offline_admitted.wait(0.05))
        self.supervisor._leave_device_workload(InferenceWorkload.TRACKING)
        self.assertTrue(offline_admitted.wait(1.0))
        self.assertFalse(
            self.supervisor._enter_device_workload(InferenceWorkload.ENRICHMENT)
        )
        self.supervisor._leave_device_workload(InferenceWorkload.OFFLINE)
        thread.join(timeout=1.0)
        self.assertEqual(self.supervisor.workload_status()["offline_active"], 0)

    def test_waiting_initial_runs_before_another_refinement(self) -> None:
        self.assertTrue(self.supervisor._enter_device_workload(
            InferenceWorkload.INCIDENT_REFINEMENT
        ))
        order: list[str] = []
        initial_entered = threading.Event()
        refinement_entered = threading.Event()

        def enter_initial() -> None:
            assert self.supervisor._enter_device_workload(
                InferenceWorkload.INCIDENT_INITIAL
            )
            order.append("initial")
            initial_entered.set()

        def enter_refinement() -> None:
            assert self.supervisor._enter_device_workload(
                InferenceWorkload.INCIDENT_REFINEMENT
            )
            order.append("refinement")
            refinement_entered.set()

        initial_thread = threading.Thread(target=enter_initial)
        initial_thread.start()
        self.assertTrue(initial_entered.wait(1.0))
        refinement_thread = threading.Thread(target=enter_refinement)
        refinement_thread.start()
        self.assertFalse(refinement_entered.wait(0.05))
        self.supervisor._leave_device_workload(
            InferenceWorkload.INCIDENT_REFINEMENT
        )
        self.assertFalse(refinement_entered.is_set())
        self.supervisor._leave_device_workload(InferenceWorkload.INCIDENT_INITIAL)
        self.assertTrue(refinement_entered.wait(1.0))
        self.supervisor._leave_device_workload(
            InferenceWorkload.INCIDENT_REFINEMENT
        )
        initial_thread.join(timeout=1.0)
        refinement_thread.join(timeout=1.0)
        self.assertEqual(order, ["initial", "refinement"])

    def test_initial_admits_while_refinement_active(self) -> None:
        self.assertTrue(
            self.supervisor._enter_device_workload(
                InferenceWorkload.INCIDENT_REFINEMENT
            )
        )
        self.assertTrue(
            self.supervisor._enter_device_workload(
                InferenceWorkload.INCIDENT_INITIAL,
                timeout=0.2,
            )
        )
        status = self.supervisor.workload_status()
        self.assertGreaterEqual(status["refinement_active"], 1)
        self.assertEqual(status["initial_active"], 1)
        self.supervisor._leave_device_workload(InferenceWorkload.INCIDENT_INITIAL)
        self.supervisor._leave_device_workload(InferenceWorkload.INCIDENT_REFINEMENT)

    def test_refinement_burst_does_not_timeout_initial(self) -> None:
        held = 3
        for _ in range(held):
            self.assertTrue(
                self.supervisor._enter_device_workload(
                    InferenceWorkload.INCIDENT_REFINEMENT
                )
            )
        admitted = threading.Event()

        def enter_initial() -> None:
            if self.supervisor._enter_device_workload(
                InferenceWorkload.INCIDENT_INITIAL,
                timeout=0.5,
            ):
                admitted.set()

        thread = threading.Thread(target=enter_initial)
        thread.start()
        self.assertTrue(admitted.wait(1.0))
        status = self.supervisor.workload_status()
        self.assertEqual(status["initial_active"], 1)
        self.assertEqual(status["refinement_active"], held)
        self.assertEqual(status["classes"]["incident_initial"]["timed_out"], 0)
        self.supervisor._leave_device_workload(InferenceWorkload.INCIDENT_INITIAL)
        for _ in range(held):
            self.supervisor._leave_device_workload(
                InferenceWorkload.INCIDENT_REFINEMENT
            )
        thread.join(timeout=1.0)

    def test_optional_still_sheds_behind_concurrent_initial_and_refinement(self) -> None:
        self.assertTrue(
            self.supervisor._enter_device_workload(
                InferenceWorkload.INCIDENT_REFINEMENT
            )
        )
        self.assertTrue(
            self.supervisor._enter_device_workload(
                InferenceWorkload.INCIDENT_INITIAL
            )
        )
        self.assertFalse(
            self.supervisor._enter_device_workload(InferenceWorkload.ENRICHMENT)
        )
        self.assertFalse(
            self.supervisor._enter_device_workload(InferenceWorkload.TRACKING)
        )
        status = self.supervisor.workload_status()
        self.assertEqual(status["classes"]["enrichment"]["shed"], 1)
        self.assertEqual(status["classes"]["tracking"]["shed"], 1)
        self.supervisor._leave_device_workload(InferenceWorkload.INCIDENT_INITIAL)
        self.supervisor._leave_device_workload(InferenceWorkload.INCIDENT_REFINEMENT)

    def test_refinement_admission_bounded_under_burst(self) -> None:
        supervisor = InferenceSupervisor(
            DetectorConfig(enabled=False, max_concurrent_refinements=2)
        )
        self.addCleanup(supervisor.stop)
        self.assertTrue(
            supervisor._enter_device_workload(InferenceWorkload.INCIDENT_REFINEMENT)
        )
        self.assertTrue(
            supervisor._enter_device_workload(InferenceWorkload.INCIDENT_REFINEMENT)
        )
        self.assertFalse(
            supervisor._enter_device_workload(
                InferenceWorkload.INCIDENT_REFINEMENT,
                timeout=0.05,
            )
        )
        self.assertTrue(
            supervisor._enter_device_workload(
                InferenceWorkload.INCIDENT_INITIAL,
                timeout=0.2,
            )
        )
        status = supervisor.workload_status()
        self.assertEqual(status["refinement_active"], 2)
        self.assertEqual(status["max_concurrent_refinements"], 2)
        self.assertEqual(status["initial_active"], 1)
        self.assertEqual(status["classes"]["incident_refinement"]["timed_out"], 1)
        supervisor._leave_device_workload(InferenceWorkload.INCIDENT_INITIAL)
        supervisor._leave_device_workload(InferenceWorkload.INCIDENT_REFINEMENT)
        supervisor._leave_device_workload(InferenceWorkload.INCIDENT_REFINEMENT)

    def test_runtime_config_update_reaches_supervisor_and_future_worker_respawns(self) -> None:
        updated = DetectorConfig(
            enabled=False,
            confidence_threshold=0.67,
            event_confirmation_frames=3,
            face_match_threshold=0.52,
        )

        self.supervisor.update_runtime_config(updated)

        self.assertEqual(self.supervisor.config.confidence_threshold, 0.67)
        self.assertEqual(self.supervisor.config.event_confirmation_frames, 3)
        self.assertEqual(self.supervisor.config.face_match_threshold, 0.52)
        self.assertIs(self.supervisor._object.config, self.supervisor.config)
        self.assertIs(self.supervisor._face.config, self.supervisor.config)
        self.assertIs(self.supervisor._reid.config, self.supervisor.config)

    def test_failed_runtime_metadata_validation_does_not_mutate_config_references(self) -> None:
        previous = self.supervisor.config
        updated = DetectorConfig(enabled=False, labels_path="/unreadable/labels.txt")

        with patch(
            "survng.app.inference_runtime.supervisor.load_detector_labels",
            side_effect=OSError("labels unreadable"),
        ):
            with self.assertRaisesRegex(OSError, "labels unreadable"):
                self.supervisor.update_runtime_config(updated)

        self.assertIs(self.supervisor.config, previous)
        self.assertIs(self.supervisor._object.config, previous)
        self.assertIs(self.supervisor._face.config, previous)
        self.assertIs(self.supervisor._reid.config, previous)

    def test_failed_role_metadata_validation_leaves_live_worker_unchanged(self) -> None:
        self.assertTrue(self.supervisor.start())
        previous = self.supervisor.config
        previous_pid = self.supervisor.isolation_status()["worker_pid"]
        updated = DetectorConfig(enabled=False, labels_path="/unreadable/labels.txt")

        with patch(
            "survng.app.inference_runtime.supervisor.load_detector_labels",
            side_effect=OSError("labels unreadable"),
        ):
            with self.assertRaisesRegex(OSError, "labels unreadable"):
                self.supervisor.reconfigure_roles(updated, {"object"})

        self.assertIs(self.supervisor.config, previous)
        self.assertEqual(
            self.supervisor.isolation_status()["worker_pid"],
            previous_pid,
        )

    def test_incomplete_reconfiguration_rollback_keeps_device_fenced(self) -> None:
        with patch.object(
            self.supervisor,
            "_reconfigure_roles_unfenced",
            side_effect=InferenceRollbackIncomplete("rollback incomplete"),
        ):
            with self.assertRaises(InferenceRollbackIncomplete):
                self.supervisor.reconfigure_roles(
                    DetectorConfig(enabled=False),
                    {"object"},
                )

        self.assertFalse(self.supervisor.workload_status()["accepting"])

    def test_role_reconfiguration_restarts_only_selected_worker(self) -> None:
        updated = _single_object_detector(device="GPU")

        with (
            patch.object(self.supervisor._object, "reconfigure") as object_reconfigure,
            patch.object(self.supervisor._face, "reconfigure") as face_reconfigure,
            patch.object(self.supervisor._reid, "reconfigure") as reid_reconfigure,
        ):
            self.supervisor.reconfigure_roles(updated, {"object"})

        object_reconfigure.assert_called_once()
        face_reconfigure.assert_not_called()
        reid_reconfigure.assert_not_called()
        self.assertEqual(self.supervisor.config.device, "GPU")
        self.assertIs(self.supervisor._face.config, self.supervisor.config)
        self.assertIs(self.supervisor._reid.config, self.supervisor.config)

    def test_object_role_reconfiguration_replaces_live_worker_process(self) -> None:
        self.assertTrue(self.supervisor.start())
        previous_pid = self.supervisor.isolation_status()["worker_pid"]
        updated = _single_object_detector(nms_threshold=0.51)

        self.supervisor.reconfigure_roles(updated, {"object"})

        status = self.supervisor.isolation_status()
        self.assertTrue(status["worker_alive"])
        self.assertNotEqual(status["worker_pid"], previous_pid)
        self.assertEqual(self.supervisor.config.nms_threshold, 0.51)

    def test_object_role_reconfiguration_resizes_worker_pool(self) -> None:
        self.assertTrue(self.supervisor.start())
        previous_process = self.supervisor._object._process
        self.assertIsNotNone(previous_process)

        self.supervisor.reconfigure_roles(
            DetectorConfig(enabled=False, object_worker_count=2),
            {"object"},
        )

        isolation = self.supervisor.isolation_status()
        self.assertEqual(isolation["configured_workers"], 2)
        self.assertEqual(isolation["alive_workers"], 2)
        self.assertFalse(previous_process.is_alive())
        self.assertEqual(self.supervisor.config.object_worker_count, 2)

    def test_multi_role_reconfiguration_rolls_back_completed_roles(self) -> None:
        previous = self.supervisor.config.model_copy(deep=True)
        updated = _single_object_detector(cache_dir="/tmp/new-cache")

        with (
            patch.object(self.supervisor._object, "reconfigure") as object_reconfigure,
            patch.object(
                self.supervisor._face,
                "reconfigure",
                side_effect=RuntimeError("face restart failed"),
            ) as face_reconfigure,
            patch.object(self.supervisor._reid, "reconfigure") as reid_reconfigure,
        ):
            with self.assertRaisesRegex(RuntimeError, "face restart failed"):
                self.supervisor.reconfigure_roles(
                    updated,
                    {"object", "face", "reid"},
                )

        self.assertEqual(object_reconfigure.call_count, 2)
        self.assertEqual(object_reconfigure.call_args_list[-1].args[0], previous)
        face_reconfigure.assert_called_once()
        reid_reconfigure.assert_not_called()
        self.assertEqual(self.supervisor.config, previous)

    def test_failed_worker_stop_keeps_prior_worker_recoverable(self) -> None:
        worker = _InferenceWorker(
            DetectorConfig(enabled=False),
            "object",
            {},
        )

        def fail_stop() -> None:
            worker._stopping = True
            raise RuntimeError("worker remained alive")

        with patch.object(worker, "stop", side_effect=fail_stop):
            with self.assertRaisesRegex(RuntimeError, "remained alive"):
                worker.reconfigure(
                    DetectorConfig(enabled=False, device="GPU"),
                    {},
                    start_enabled=True,
                )

        self.assertFalse(worker._stopping)
        self.assertEqual(worker.config.device, "CPU")

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
        self.assertEqual(proxy.status()["device"], "CPU")
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

    def test_hot_reid_threshold_is_used_for_status_and_model_identity(self) -> None:
        config = DetectorConfig.model_validate({
            "tracking": {
                "reid_enabled": True,
                "reid_model_path": "person.xml",
                "reid_match_threshold": 0.83,
            },
        })
        supervisor = InferenceSupervisor(config)
        self.addCleanup(supervisor.stop)
        supervisor._reid._status = {
            "ready": True,
            "match_threshold": 0.70,
            "person": {
                "ready": True,
                "model_fingerprint": "person-model",
                "embedding_size": 256,
                "match_threshold": 0.70,
            },
            "vehicle": {"ready": False},
        }
        supervisor._reid.start_enabled = False
        supervisor._reid._process = Mock(is_alive=Mock(return_value=True))
        self.addCleanup(setattr, supervisor._reid, "_process", None)
        proxy = IsolatedPersonReidentifier(supervisor)

        status = supervisor.cached_reid_status()
        identity = proxy.model_identity_for_label("person")

        self.assertEqual(status["match_threshold"], 0.83)
        self.assertEqual(status["person"]["match_threshold"], 0.83)
        self.assertIsNotNone(identity)
        self.assertEqual(identity["match_threshold"], 0.83)

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
        self.assertEqual(proxy.status()["device"], "CPU")
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
            "survng.app.inference_runtime.supervisor.stop_multiprocessing_resource_tracker"
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

    def test_stop_wakes_queued_priority_admissions(self) -> None:
        worker = _InferenceWorker(
            DetectorConfig(enabled=False),
            "object",
            {},
            start_enabled=False,
        )

        class Connection:
            request_id = 0

            def send(self, request):
                self.request_id = int(request["id"])

            @staticmethod
            def poll(_timeout):
                return True

            def recv(self):
                return {"id": self.request_id, "ok": True, "result": []}

            @staticmethod
            def close():
                return None

        worker._connection = Connection()
        worker._ensure_worker_locked = Mock(return_value=True)
        lock_held = threading.Event()
        release_lock = threading.Event()
        queued_done = threading.Event()
        errors: list[str] = []

        def hold_worker_lock() -> None:
            with worker._lock:
                lock_held.set()
                release_lock.wait(timeout=2.0)

        def request(operation: str, done: threading.Event | None = None) -> None:
            try:
                worker.request(operation)
            except InferenceUnavailable as error:
                errors.append(str(error))
            finally:
                if done is not None:
                    done.set()

        holder = threading.Thread(target=hold_worker_lock)
        holder.start()
        self.assertTrue(lock_held.wait(1.0))
        active = threading.Thread(target=request, args=("active",))
        active.start()
        with worker._admission:
            self.assertTrue(worker._admission.wait_for(
                lambda: worker._admission_active,
                timeout=1.0,
            ))
        queued = threading.Thread(target=request, args=("queued", queued_done))
        queued.start()
        with worker._admission:
            self.assertTrue(worker._admission.wait_for(
                lambda: len(worker._admission_waiters) == 1,
                timeout=1.0,
            ))
        stopper = threading.Thread(target=worker.stop)
        stopper.start()
        self.assertTrue(queued_done.wait(1.0))
        self.assertIn("stopping", errors[0])
        release_lock.set()
        for thread in (holder, active, queued, stopper):
            thread.join(timeout=1.0)

    def test_incident_request_runs_before_queued_tracking_request(self) -> None:
        worker = _InferenceWorker(
            DetectorConfig(enabled=False),
            "object",
            {},
            start_enabled=False,
        )
        sent: list[str] = []

        class Connection:
            request_id = 0

            def send(self, request):
                self.request_id = int(request["id"])
                sent.append(str(request["op"]))

            @staticmethod
            def poll(_timeout):
                return True

            def recv(self):
                return {"id": self.request_id, "ok": True, "result": []}

        worker._connection = Connection()
        worker._ensure_worker_locked = Mock(return_value=True)
        lock_held = threading.Event()
        release_lock = threading.Event()

        def hold_worker_lock() -> None:
            with worker._lock:
                lock_held.set()
                release_lock.wait(timeout=2.0)

        holder = threading.Thread(target=hold_worker_lock)
        holder.start()
        self.assertTrue(lock_held.wait(timeout=1.0))

        threads = [
            threading.Thread(
                target=worker.request,
                args=("tracking-active",),
                kwargs={"workload": InferenceWorkload.TRACKING},
            )
        ]
        threads[0].start()
        with worker._admission:
            self.assertTrue(
                worker._admission.wait_for(
                    lambda: worker._admission_active,
                    timeout=1.0,
                )
            )
        threads.extend([
            threading.Thread(
                target=worker.request,
                args=("tracking-queued",),
                kwargs={"workload": InferenceWorkload.TRACKING},
            ),
            threading.Thread(
                target=worker.request,
                args=("refinement",),
                kwargs={"workload": InferenceWorkload.INCIDENT_REFINEMENT},
            ),
            threading.Thread(
                target=worker.request,
                args=("incident-initial",),
                kwargs={"workload": InferenceWorkload.INCIDENT_INITIAL},
            ),
        ])
        for thread in threads[1:]:
            thread.start()
        with worker._admission:
            self.assertTrue(worker._admission.wait_for(
                lambda: {
                    priority
                    for priority, _sequence, _token, _queued_at
                    in worker._admission_waiters
                } == {
                    int(InferenceWorkload.INCIDENT_INITIAL),
                    int(InferenceWorkload.INCIDENT_REFINEMENT),
                    int(InferenceWorkload.TRACKING),
                },
                timeout=1.0,
            ))
        release_lock.set()
        holder.join(timeout=1.0)
        for thread in threads:
            thread.join(timeout=1.0)

        self.assertEqual(
            sent,
            ["tracking-active", "incident-initial", "refinement", "tracking-queued"],
        )
        self.assertEqual(worker.pending_requests(), 0)

    def test_shared_frame_contract_rejects_non_bgr_and_non_uint8_arrays(self) -> None:
        worker = _InferenceWorker(DetectorConfig(enabled=False), "object", {})
        with self.assertRaisesRegex(InferenceUnavailable, "uint8 BGR"):
            worker._write_frame_locked(np.zeros((10, 10), dtype=np.uint8))
        with self.assertRaisesRegex(InferenceUnavailable, "uint8 BGR"):
            worker._write_frame_locked(np.zeros((10, 10, 3), dtype=np.float32))

    def test_object_worker_downscales_ipc_frame_and_restores_boxes(self) -> None:
        worker = _InferenceWorker(
            DetectorConfig(enabled=False),
            "object",
            {"input_shape": [640, 640]},
            start_enabled=False,
        )
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        ipc_frame, scale = worker._prepare_object_frame_locked(frame)
        result = worker._restore_object_boxes(
            [{"box": {"x1": 100, "y1": 50, "x2": 500, "y2": 300}}],
            scale,
        )

        self.assertEqual(ipc_frame.shape, (720, 1280, 3))
        self.assertEqual(scale, (3.0, 3.0))
        self.assertEqual(
            result[0]["box"],
            {"x1": 300.0, "y1": 150.0, "x2": 1500.0, "y2": 900.0},
        )

        list_result = worker._restore_object_boxes(
            [{"box": [100, 50, 500, 300]}],
            scale,
        )
        self.assertEqual(list_result[0]["box"], [300.0, 150.0, 1500.0, 900.0])

    def test_object_worker_keeps_frame_when_model_shape_is_unknown(self) -> None:
        worker = _InferenceWorker(
            DetectorConfig(enabled=False),
            "object",
            {},
            start_enabled=False,
        )
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        ipc_frame, scale = worker._prepare_object_frame_locked(frame)

        self.assertIs(ipc_frame, frame)
        self.assertIsNone(scale)

    def test_admission_percentiles_and_ipc_copy_bytes_are_reported(self) -> None:
        self.assertTrue(
            self.supervisor._enter_device_workload(
                InferenceWorkload.INCIDENT_INITIAL
            )
        )
        self.supervisor._leave_device_workload(InferenceWorkload.INCIDENT_INITIAL)
        device_status = self.supervisor.workload_status()["classes"][
            "incident_initial"
        ]
        self.assertIsNotNone(device_status["admission_wait_ms_p95"])
        self.assertIsNotNone(device_status["admission_wait_ms_p99"])

        worker = _InferenceWorker(
            DetectorConfig(enabled=False),
            "object",
            {},
            start_enabled=False,
        )
        frame = np.zeros((10, 12, 3), dtype=np.uint8)
        worker._frame_buffer = bytearray(frame.nbytes)
        worker._write_frame_locked(frame)
        ipc = worker.isolation_status()["ipc"]
        self.assertEqual(ipc["frame_copy_bytes_total"], frame.nbytes)
        self.assertEqual(ipc["samples"], 1)
        self.assertIsNotNone(ipc["frame_copy_ms_p95"])
        self.assertIsNotNone(ipc["frame_copy_ms_p99"])

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

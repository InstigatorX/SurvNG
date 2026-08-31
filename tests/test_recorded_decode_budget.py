from __future__ import annotations

import threading
import time
import unittest

from survng.app.config import DetectorConfig
from survng.app.motion_pipeline.recorded_decode_budget import RecordedDecodeBudget


class RecordedDecodeBudgetTest(unittest.TestCase):
    def test_process_capacity_blocks_until_release(self) -> None:
        budget = RecordedDecodeBudget(max_processes=2, memory_budget_bytes=64 << 20)
        first = budget.acquire_process(incident_epoch=1.0)
        second = budget.acquire_process(incident_epoch=2.0)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        blocked = budget.acquire_process(incident_epoch=3.0, deadline=time.monotonic() + 0.05)
        self.assertIsNone(blocked)
        assert first is not None
        first.release()
        third = budget.acquire_process(incident_epoch=3.0, deadline=time.monotonic() + 0.5)
        self.assertIsNotNone(third)
        assert second is not None and third is not None
        second.release()
        third.release()
        self.assertEqual(budget.status()["active_processes"], 0)

    def test_memory_reservation_bounds_concurrent_workflows(self) -> None:
        budget = RecordedDecodeBudget(
            max_processes=4,
            memory_budget_bytes=16 << 20,
            estimated_frame_bytes=8 << 20,
        )
        first = budget.reserve_workflow(maximum_frames=2, incident_epoch=1.0)
        self.assertIsNotNone(first)
        second = budget.reserve_workflow(
            maximum_frames=2,
            incident_epoch=2.0,
            deadline=time.monotonic() + 0.05,
        )
        self.assertIsNone(second)
        assert first is not None
        first.release()
        recovered = budget.reserve_workflow(maximum_frames=2, incident_epoch=2.0)
        self.assertIsNotNone(recovered)
        assert recovered is not None
        recovered.release()

    def test_expanded_burst_scales_memory_capacity_per_decode_process(self) -> None:
        """Each decoder admits one 14-frame full refinement workflow."""
        budget = RecordedDecodeBudget.from_detector_config(DetectorConfig())
        first = budget.reserve_workflow(maximum_frames=14, incident_epoch=1.0)
        second = budget.reserve_workflow(maximum_frames=14, incident_epoch=2.0)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(budget.status()["reserved_bytes"], 224 << 20)

        admitted = threading.Event()
        released = threading.Event()

        def wait_for_capacity() -> None:
            lease = budget.reserve_workflow(
                maximum_frames=14,
                incident_epoch=3.0,
                deadline=time.monotonic() + 1.0,
            )
            if lease is not None:
                admitted.set()
                lease.release()
            released.set()

        waiter = threading.Thread(target=wait_for_capacity)
        waiter.start()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline and budget.status()["waiting"] < 1:
            time.sleep(0.01)
        self.assertEqual(budget.status()["waiting"], 1)
        assert first is not None and second is not None
        first.release()
        self.assertTrue(admitted.wait(0.5))
        second.release()
        self.assertTrue(released.wait(0.5))
        waiter.join(timeout=0.5)
        self.assertEqual(budget.status()["reserved_bytes"], 0)

    def test_third_decode_process_adds_a_third_full_workflow_reservation(self) -> None:
        budget = RecordedDecodeBudget.from_detector_config(
            DetectorConfig(recorded_decode_max_processes=3)
        )
        leases = [
            budget.reserve_workflow(maximum_frames=14, incident_epoch=float(index))
            for index in range(3)
        ]
        self.assertTrue(all(lease is not None for lease in leases))
        self.assertEqual(budget.status()["memory_budget_bytes"], 336 << 20)
        self.assertEqual(budget.status()["reserved_bytes"], 336 << 20)
        for lease in leases:
            assert lease is not None
            lease.release()

    def test_legacy_memory_budget_does_not_scale_with_process_count(self) -> None:
        budget = RecordedDecodeBudget.from_detector_config(
            DetectorConfig(
                recorded_decode_max_processes=3,
                recorded_decode_memory_budget_mb=256,
                recorded_decode_memory_per_process_mb=None,
            )
        )
        status = budget.status()
        self.assertEqual(status["memory_budget_bytes"], 256 << 20)
        self.assertIsNone(status["memory_per_process_bytes"])

    def test_downscale_recharges_queued_workflow_to_the_current_ceiling(self) -> None:
        budget = RecordedDecodeBudget(
            max_processes=3,
            memory_budget_bytes=336 << 20,
            estimated_frame_bytes=8 << 20,
        )
        active = [
            budget.reserve_workflow(maximum_frames=14, incident_epoch=float(index))
            for index in range(3)
        ]
        self.assertTrue(all(lease is not None for lease in active))
        admitted = threading.Event()

        def wait_for_downscaled_capacity() -> None:
            lease = budget.reserve_workflow(
                maximum_frames=32,
                incident_epoch=4.0,
                deadline=time.monotonic() + 1.0,
            )
            if lease is not None:
                admitted.set()
                lease.release()

        waiter = threading.Thread(target=wait_for_downscaled_capacity)
        waiter.start()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline and budget.status()["waiting"] < 1:
            time.sleep(0.01)
        budget.reconfigure(
            max_processes=1,
            memory_budget_bytes=112 << 20,
            estimated_frame_bytes=8 << 20,
        )
        for lease in active:
            assert lease is not None
            lease.release()
        self.assertTrue(admitted.wait(0.5))
        waiter.join(timeout=0.5)

    def test_oversized_workflow_charges_full_budget_alone(self) -> None:
        budget = RecordedDecodeBudget(
            max_processes=2,
            memory_budget_bytes=16 << 20,
            estimated_frame_bytes=8 << 20,
        )
        lease = budget.reserve_workflow(maximum_frames=8, incident_epoch=1.0)
        self.assertIsNotNone(lease)
        status = budget.status()
        self.assertEqual(status["reserved_bytes"], 16 << 20)
        assert lease is not None
        lease.release()

    def test_older_incident_wins_process_wait_queue(self) -> None:
        budget = RecordedDecodeBudget(max_processes=1, memory_budget_bytes=64 << 20)
        holder = budget.acquire_process(incident_epoch=1.0)
        self.assertIsNotNone(holder)
        order: list[float] = []
        started = threading.Barrier(3)

        def wait_for(epoch: float) -> None:
            started.wait(1.0)
            lease = budget.acquire_process(
                incident_epoch=epoch,
                deadline=time.monotonic() + 1.0,
            )
            if lease is not None:
                order.append(epoch)
                time.sleep(0.02)
                lease.release()

        newer = threading.Thread(target=wait_for, args=(30.0,))
        older = threading.Thread(target=wait_for, args=(10.0,))
        newer.start()
        older.start()
        started.wait(1.0)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and budget.status()["waiting"] < 2:
            time.sleep(0.01)
        self.assertGreaterEqual(budget.status()["waiting"], 2)
        assert holder is not None
        holder.release()
        newer.join(timeout=1.0)
        older.join(timeout=1.0)
        self.assertEqual(order, [10.0, 30.0])

    def test_memory_waiter_does_not_block_free_process_slot(self) -> None:
        budget = RecordedDecodeBudget(
            max_processes=1,
            memory_budget_bytes=8 << 20,
            estimated_frame_bytes=8 << 20,
        )
        memory_holder = budget.reserve_workflow(maximum_frames=1, incident_epoch=1.0)
        self.assertIsNotNone(memory_holder)
        memory_waiting = threading.Event()

        def wait_for_memory() -> None:
            memory_waiting.set()
            lease = budget.reserve_workflow(
                maximum_frames=1,
                incident_epoch=2.0,
                deadline=time.monotonic() + 1.0,
            )
            if lease is not None:
                lease.release()

        waiter = threading.Thread(target=wait_for_memory)
        waiter.start()
        self.assertTrue(memory_waiting.wait(1.0))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and budget.status()["waiting"] < 1:
            time.sleep(0.01)

        process = budget.acquire_process(
            incident_epoch=3.0,
            deadline=time.monotonic() + 0.2,
        )

        self.assertIsNotNone(process)
        assert process is not None and memory_holder is not None
        process.release()
        memory_holder.release()
        waiter.join(1.0)

    def test_cancellation_exits_without_admission(self) -> None:
        budget = RecordedDecodeBudget(max_processes=1, memory_budget_bytes=64 << 20)
        holder = budget.acquire_process(incident_epoch=1.0)
        self.assertIsNotNone(holder)
        cancelled = False

        def is_cancelled() -> bool:
            return cancelled

        waiter = threading.Thread(
            target=lambda: budget.acquire_process(
                incident_epoch=2.0,
                deadline=time.monotonic() + 1.0,
                cancelled=is_cancelled,
            )
        )
        waiter.start()
        time.sleep(0.05)
        cancelled = True
        waiter.join(timeout=1.0)
        assert holder is not None
        holder.release()
        self.assertGreaterEqual(budget.status()["cancellations"], 1)

    def test_reconfigure_blocks_new_admissions_above_reduced_limit(self) -> None:
        budget = RecordedDecodeBudget(max_processes=2, memory_budget_bytes=64 << 20)
        first = budget.acquire_process(incident_epoch=1.0)
        second = budget.acquire_process(incident_epoch=2.0)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        budget.reconfigure(
            max_processes=1,
            memory_budget_bytes=64 << 20,
            estimated_frame_bytes=8 << 20,
        )
        blocked = budget.acquire_process(
            incident_epoch=3.0,
            deadline=time.monotonic() + 0.05,
        )
        self.assertIsNone(blocked)
        assert first is not None and second is not None
        first.release()
        second.release()

    def test_from_detector_config_defaults(self) -> None:
        budget = RecordedDecodeBudget.from_detector_config(DetectorConfig())
        status = budget.status()
        self.assertEqual(status["max_processes"], 2)
        self.assertEqual(status["memory_budget_bytes"], 224 << 20)
        self.assertEqual(status["estimated_frame_bytes"], 8 << 20)

    def test_wait_percentiles_and_ffmpeg_outcomes_are_reported(self) -> None:
        budget = RecordedDecodeBudget(max_processes=1, memory_budget_bytes=16 << 20)
        process = budget.acquire_process(incident_epoch=1.0)
        workflow = budget.reserve_workflow(maximum_frames=1, incident_epoch=1.0)
        self.assertIsNotNone(process)
        self.assertIsNotNone(workflow)
        assert process is not None and workflow is not None
        process.release()
        workflow.release()

        budget.record_ffmpeg("hardware", success=False)
        budget.record_ffmpeg("cpu", success=True)
        status = budget.status()

        self.assertIsNotNone(status["decode_process_wait_ms_p95"])
        self.assertIsNotNone(status["decode_process_wait_ms_p99"])
        self.assertIsNotNone(status["decode_memory_wait_ms_p95"])
        self.assertIsNotNone(status["decode_memory_wait_ms_p99"])
        self.assertEqual(status["ffmpeg_attempts"], {"hardware": 1, "cpu": 1})
        self.assertEqual(status["ffmpeg_successes"], {"hardware": 0, "cpu": 1})


if __name__ == "__main__":
    unittest.main()

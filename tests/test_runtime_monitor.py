from __future__ import annotations

import time
import unittest
from unittest.mock import Mock, patch

from survng.app.runtime_monitor import ApplicationRuntimeMonitor


def runtime_monitor(
    *,
    statuses: list[dict] | None = None,
    sample_interval_seconds: float = 0.01,
) -> tuple[ApplicationRuntimeMonitor, Mock, Mock, Mock]:
    inference = Mock()
    inference.detector.status.return_value = {
        "runtime": {},
        "workers": {},
    }
    inference.status.return_value = {}
    inference.semantic_search.status.return_value = {"state": "disabled"}
    events = Mock()
    state_events = Mock()
    monitor = ApplicationRuntimeMonitor(
        inference=inference,
        state_events=state_events,
        camera_statuses=Mock(return_value=statuses or []),
        sample_interval_seconds=sample_interval_seconds,
        poll_interval_seconds=0.01,
    )
    return monitor, inference, events, state_events


def test_operational_collector_persists_interval_deltas(tmp_path) -> None:
    from datetime import datetime, timezone

    from survng.app.runtime_monitor import OperationalTelemetryCollector

    collector = OperationalTelemetryCollector()
    sampled_at = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    status = {
        "id": "gate",
        "expected_enabled": True,
        "connected": True,
        "last_frame_age_seconds": 0.1,
        "capture_stats": {
            "live": {"fps": 10.0, "read_failures": 4},
            "main": {"fps": 20.0, "open_failures": 1},
        },
        "motion_qualification": {
            "analysis_frames_dropped": 2,
            "analysis_runtime": {"frames_sampled": 100},
            "event_runtime": {"episode": {"decision_counts": {"request_admitted": 3}}},
        },
    }
    _, first = collector.collect(
        [status],
        sampled_at=sampled_at,
        process_memory={},
        worker_memory={},
        system_runtime={},
        detector_runtime={},
    )
    status["capture_stats"]["live"]["read_failures"] = 6
    status["motion_qualification"]["analysis_runtime"]["frames_sampled"] = 125
    status["motion_qualification"]["event_runtime"]["episode"]["decision_counts"]["request_admitted"] = 4
    _, second = collector.collect(
        [status],
        sampled_at=sampled_at,
        process_memory={},
        worker_memory={},
        system_runtime={},
        detector_runtime={},
    )

    assert first[0].capture_interruptions == 0
    assert second[0].capture_interruptions == 2
    assert second[0].ema_frames_sampled == 25
    assert second[0].object_checks_admitted == 1


class ApplicationRuntimeMonitorTest(unittest.TestCase):
    def test_monitor_owns_periodic_telemetry_and_stops_cleanly(self) -> None:
        status = {
            "id": "gate",
            "running": True,
            "object_tracking": {"active": False, "worker_running": False},
        }
        monitor, inference, events, state_events = runtime_monitor(statuses=[status])

        monitor.start()
        deadline = time.monotonic() + 1.0
        while not inference.maintain.called and time.monotonic() < deadline:
            time.sleep(0.01)
        monitor.stop()

        self.assertFalse(monitor.running)
        inference.maintain.assert_called()
        state_events.publish.assert_any_call("camera_state", status)

    def test_start_and_stop_are_idempotent(self) -> None:
        monitor, _inference, _events, _state_events = runtime_monitor()

        monitor.start()
        first_thread = monitor._thread
        monitor.start()
        self.assertIs(monitor._thread, first_thread)
        monitor.stop()
        monitor.stop()

        self.assertFalse(monitor.running)

    def test_diagnostic_failure_does_not_stop_operational_maintenance(self) -> None:
        monitor, inference, _events, _state_events = runtime_monitor(
            statuses=[{"id": "gate", "connected": True}]
        )
        monitor._diagnostics = Mock()
        monitor._diagnostics.observe.side_effect = RuntimeError("diagnostic store busy")

        monitor.start()
        deadline = time.monotonic() + 1.0
        while not inference.maintain.called and time.monotonic() < deadline:
            time.sleep(0.01)
        monitor.stop()

        inference.maintain.assert_called()

    def test_worker_memory_deduplicates_shared_worker_pid(self) -> None:
        monitor, inference, _events, _state_events = runtime_monitor()
        inference.semantic_search.status.return_value = {
            "state": "ready",
            "worker_pid": 42,
        }
        detector = {
            "workers": {
                "object": {"worker_alive": True, "worker_pid": 42},
                "face": {"worker_alive": True, "worker_pid": 43},
            }
        }

        with patch(
            "survng.app.runtime_monitor.process_memory_status_for_pid",
            side_effect=lambda pid: {
                "rss_bytes": pid * 10,
                "pss_bytes": pid * 5,
                "threads": 2,
                "file_descriptors": 3,
            },
        ) as memory:
            status = monitor.worker_memory_status(detector_status=detector)

        self.assertEqual(memory.call_count, 2)
        self.assertEqual(set(status["workers"]), {"object", "face"})
        self.assertEqual(status["total_rss_bytes"], 850)

    def test_worker_memory_accounts_for_each_object_detector_process(self) -> None:
        monitor, inference, _events, _state_events = runtime_monitor()
        inference.semantic_search.status.return_value = {"state": "disabled"}
        detector = {
            "workers": {
                "object": {
                    "worker_alive": True,
                    "worker_pid": 41,
                    "worker_pids": [41, 42],
                },
            }
        }
        with patch(
            "survng.app.runtime_monitor.process_memory_status_for_pid",
            side_effect=lambda pid: {
                "rss_bytes": pid * 10,
                "pss_bytes": pid * 5,
                "threads": 2,
                "file_descriptors": 3,
            },
        ):
            status = monitor.worker_memory_status(detector_status=detector)

        self.assertEqual(set(status["workers"]), {"object-1", "object-2"})
        self.assertEqual(status["total_rss_bytes"], 830)

    def test_publish_camera_status_uses_one_snapshot(self) -> None:
        status = {"id": "gate", "running": True}
        monitor, _inference, _events, state_events = runtime_monitor(statuses=[status])

        monitor.publish_camera_status("gate")
        monitor.publish_camera_status("missing")

        state_events.publish.assert_called_once_with("camera_state", status)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app.config import CameraConfig, ObjectTrackingConfig
from survng.app.object_tracking_lifecycle import ObjectTrackingLifecycle


def _session(*, fps: float = 2.0) -> Mock:
    session = Mock()
    session.config = ObjectTrackingConfig(sample_fps=fps)
    session.stop.return_value = True
    session.running.return_value = False
    session.status.return_value = {"active": False}
    return session


def _lifecycle(
    initial: Mock,
    *,
    accepting: list[bool] | None = None,
) -> tuple[ObjectTrackingLifecycle, Mock, Mock, Mock]:
    camera = CameraConfig(
        id="gate",
        name="Gate",
        stream_url="rtsp://example.invalid/main",
    )
    factory = Mock()
    factory.create.return_value = initial
    frame_provider = Mock(return_value=None)
    catchup_provider = Mock(return_value=[])
    history = Mock()
    lifecycle = ObjectTrackingLifecycle(
        camera=camera,
        factory=factory,
        frame_provider=frame_provider,
        catchup_frame_provider=catchup_provider,
        prewarm_frame_provider=frame_provider,
        history=lambda: history,
        accepting=lambda: True if accepting is None else accepting[0],
        lifecycle_lock=threading.RLock(),
    )
    return lifecycle, factory, frame_provider, history


def test_factory_creation_and_prewarm_keep_frame_dependencies_injected() -> None:
    initial = _session()
    lifecycle, factory, frame_provider, _history = _lifecycle(initial)
    sample = (np.zeros((4, 6, 3), dtype=np.uint8), 10.0, 9.0)
    frame_provider.return_value = sample

    assert lifecycle.current() is initial
    assert lifecycle.prewarm() is sample
    assert factory.create.call_args.kwargs["camera"].id == "gate"
    assert factory.create.call_args.kwargs["frame_provider"] is frame_provider
    assert callable(factory.create.call_args.kwargs["catchup_frame_provider"])


def test_accepting_state_is_evaluated_when_session_is_resumed() -> None:
    state = [False]
    initial = _session()
    lifecycle, _factory, _frame_provider, _history = _lifecycle(
        initial,
        accepting=state,
    )

    lifecycle.sync_accepting()
    state[0] = True
    lifecycle.sync_accepting()

    assert initial.set_accepting.call_args_list[0].args == (False,)
    assert initial.set_accepting.call_args_list[1].args == (True,)


def test_pause_refuses_to_hide_a_session_that_did_not_stop() -> None:
    initial = _session()
    initial.stop.return_value = False
    lifecycle, _factory, _frame_provider, _history = _lifecycle(initial)

    with pytest.raises(RuntimeError, match="did not stop for gate"):
        lifecycle.pause()

    assert lifecycle.current() is initial


def test_replacement_stops_previous_resizes_history_and_applies_eligibility() -> None:
    initial = _session()
    replacement = _session(fps=3.0)
    lifecycle, _factory, _frame_provider, history = _lifecycle(initial)

    previous = lifecycle.replace(replacement)

    assert previous is initial
    assert lifecycle.current() is replacement
    initial.stop.assert_called_once_with()
    history.resize.assert_called_once_with(3.0)
    replacement.set_accepting.assert_called_once_with(True)


def test_replacement_failure_cleans_up_and_restores_previous_session() -> None:
    initial = _session()
    replacement = _session(fps=3.0)
    replacement.set_accepting.side_effect = RuntimeError("accept failed")
    lifecycle, _factory, _frame_provider, history = _lifecycle(initial)

    with pytest.raises(RuntimeError, match="accept failed"):
        lifecycle.replace(replacement)

    assert lifecycle.current() is initial
    replacement.stop.assert_called_once_with()
    assert history.resize.call_args_list[0].args == (3.0,)
    assert history.resize.call_args_list[1].args == (2.0,)
    initial.set_accepting.assert_called_once_with(True)


def test_replacement_preserves_original_error_when_restore_is_interrupted() -> None:
    initial = _session()
    initial.set_accepting.side_effect = KeyboardInterrupt("restore interrupted")
    replacement = _session(fps=3.0)
    replacement.set_accepting.side_effect = RuntimeError("accept failed")
    lifecycle, _factory, _frame_provider, _history = _lifecycle(initial)

    with pytest.raises(RuntimeError, match="accept failed"):
        lifecycle.replace(replacement)

    assert lifecycle.current() is initial


def test_status_running_and_sample_rate_follow_the_current_session() -> None:
    initial = _session(fps=2.5)
    initial.running.return_value = True
    initial.status.return_value = {"active": True, "frames_processed": 7}
    lifecycle, _factory, _frame_provider, _history = _lifecycle(initial)

    assert lifecycle.sample_fps() == 2.5
    assert lifecycle.running() is True
    assert lifecycle.status() == {"active": True, "frames_processed": 7}


def test_incident_handoff_filters_objects_and_starts_current_session_atomically() -> None:
    initial = _session()
    initial.start.return_value = True
    lifecycle, _factory, _frame_provider, _history = _lifecycle(initial)
    event_at = datetime.now(timezone.utc)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    objects = [
        {"label": "face"},
        {"label": "person", "incident_eligible": False},
        {"label": "person", "incident_eligible": True},
    ]

    assert lifecycle.enabled()
    assert lifecycle.has_trackable_objects(objects)
    assert lifecycle.start_incident(42, event_at, objects, frame) is True
    initial.start.assert_called_once_with(
        42,
        event_at,
        [{"label": "person", "incident_eligible": True}],
        frame,
    )


def test_read_only_sample_rate_does_not_wait_for_camera_lifecycle_lock() -> None:
    initial = _session(fps=3.0)
    lifecycle, _factory, _frame_provider, _history = _lifecycle(initial)
    completed = threading.Event()

    def read_sample_rate() -> None:
        assert lifecycle.sample_fps() == 3.0
        completed.set()

    with lifecycle.lifecycle_lock:
        reader = threading.Thread(target=read_sample_rate)
        reader.start()
        assert completed.wait(timeout=0.2)
    reader.join(timeout=1)


def test_compatibility_binding_rejects_replacement_of_active_session() -> None:
    initial = _session()
    initial.running.return_value = True
    lifecycle, _factory, _frame_provider, _history = _lifecycle(initial)

    with pytest.raises(RuntimeError, match="cannot replace active"):
        lifecycle.bind_for_compatibility(_session())

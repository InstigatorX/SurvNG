from __future__ import annotations

import threading
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


def test_status_running_and_sample_rate_follow_the_current_session() -> None:
    initial = _session(fps=2.5)
    initial.running.return_value = True
    initial.status.return_value = {"active": True, "frames_processed": 7}
    lifecycle, _factory, _frame_provider, _history = _lifecycle(initial)

    assert lifecycle.sample_fps() == 2.5
    assert lifecycle.running() is True
    assert lifecycle.status() == {"active": True, "frames_processed": 7}

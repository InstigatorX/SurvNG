from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

from survng.app.config import AppConfig
from survng.app.inference_lifecycle import InferenceLifecycle


def _lifecycle() -> InferenceLifecycle:
    service = object.__new__(InferenceLifecycle)
    config = AppConfig().detector
    service.storage_dir = Path("/storage")
    service.events = Mock()
    service.events.db_path = Path("/database/survng.sqlite3")
    service.appearance_index = Mock()
    service.semantic_index = Mock()
    service.event_publisher = Mock()
    service.tracking_burst_guard = Mock(return_value=True)
    service.database_dir = Path("/database")
    service.detector = Mock()
    service.detector.config = config
    service.face_recognizer = Mock()
    service.person_reidentifier = Mock()
    service.faces = Mock()
    service.semantic_search = Mock()
    service.tracking_limiter = Mock()
    service.tracking_factory = Mock()
    service.appearance_backfill = Mock()
    service._workers = {}
    service._workers_bound = False
    service._lock = threading.RLock()
    service._core_started = False
    service._auxiliary_started = False
    service._closed = False
    service._retired_cleanup = []
    return service


def test_constructor_failure_closes_every_completed_dependency() -> None:
    detector = Mock()
    faces = Mock()
    semantic = Mock()
    with (
        patch(
            "survng.app.inference_lifecycle.InferenceSupervisor",
            return_value=detector,
        ),
        patch("survng.app.inference_lifecycle.IsolatedFaceRecognizer"),
        patch("survng.app.inference_lifecycle.IsolatedPersonReidentifier"),
        patch("survng.app.inference_lifecycle.FaceStore", return_value=faces),
        patch(
            "survng.app.inference_lifecycle.build_semantic_search",
            return_value=semantic,
        ),
        patch.object(
            InferenceLifecycle,
            "_build_backfill",
            side_effect=RuntimeError("backfill construction failed"),
        ),
        pytest.raises(RuntimeError, match="backfill construction failed"),
    ):
        InferenceLifecycle(
            config=AppConfig().detector,
            semantic_config=AppConfig().semantic_search,
            storage_dir=Path("/storage"),
            events=Mock(),
            appearance_index=Mock(),
            semantic_index=Mock(),
            event_publisher=Mock(),
            tracking_burst_guard=Mock(return_value=True),
            database_dir=Path("/database"),
        )

    semantic.close.assert_called_once_with()
    faces.close.assert_called_once_with()
    detector.stop.assert_called_once_with()


def test_core_start_rolls_back_detector_when_face_queue_fails() -> None:
    service = _lifecycle()
    service.faces.start.side_effect = RuntimeError("face queue failed")

    with pytest.raises(RuntimeError, match="face queue failed"):
        service.start_core()

    service.detector.start.assert_called_once_with()
    service.faces.close.assert_called_once_with()
    service.detector.stop.assert_called_once_with()
    assert not service.status()["core_started"]


def test_workers_bind_once_before_start() -> None:
    service = _lifecycle()
    worker = Mock()

    service.bind_workers({"gate": worker})

    assert service.status()["bound_cameras"] == 1
    with pytest.raises(RuntimeError, match="already bound"):
        service.bind_workers({"yard": Mock()})


def test_auxiliary_start_rolls_back_backfill_when_search_fails() -> None:
    service = _lifecycle()
    service.semantic_search.start.side_effect = RuntimeError("search failed")

    with pytest.raises(RuntimeError, match="search failed"):
        service.start_auxiliary()

    service.appearance_backfill.start.assert_called_once_with()
    service.semantic_search.close.assert_called_once_with()
    service.appearance_backfill.close.assert_called_once_with()
    assert not service.status()["auxiliary_started"]


def test_close_attempts_every_owned_and_retired_generation() -> None:
    service = _lifecycle()
    retired = Mock(side_effect=RuntimeError("retired close failed"))
    service._retired_cleanup = [("retired search", retired)]
    service.faces.close.side_effect = RuntimeError("face close failed")

    with pytest.raises(RuntimeError, match="face recognition, retired search"):
        service.close()

    service.semantic_search.close.assert_called_once_with()
    service.appearance_backfill.close.assert_called_once_with()
    service.detector.stop.assert_called_once_with()
    retired.assert_called_once_with()
    assert service.status()["closed"]


def test_close_does_not_chain_secret_bearing_dependency_error() -> None:
    service = _lifecycle()
    service.faces.close.side_effect = RuntimeError(
        "https://admin:supersecret@example.invalid/model"
    )

    with pytest.raises(RuntimeError) as raised:
        service.close()

    assert "supersecret" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_tracking_reconfiguration_commits_one_coherent_generation() -> None:
    service = _lifecycle()
    worker = Mock()
    worker.camera.id = "gate"
    previous_session = Mock()
    replacement = Mock()
    worker.create_object_tracking_session.return_value = replacement
    worker.replace_object_tracking_session.return_value = previous_session
    service._workers = {"gate": worker}
    next_limiter = Mock()
    next_factory = Mock()
    next_backfill = Mock()
    service._build_limiter = Mock(return_value=next_limiter)
    service._build_tracking_factory = Mock(return_value=next_factory)
    service._build_backfill = Mock(return_value=next_backfill)
    service._auxiliary_started = True
    next_config = AppConfig().detector.model_copy(deep=True)
    next_config.tracking.sample_fps = 3.0

    service.reconfigure_tracking(next_config)

    worker.create_object_tracking_session.assert_called_once_with(next_factory)
    worker.replace_object_tracking_session.assert_called_once_with(replacement)
    service.detector.update_runtime_config.assert_called_once_with(next_config)
    next_backfill.start.assert_called_once_with()
    assert service.tracking_limiter is next_limiter
    assert service.tracking_factory is next_factory
    assert service.appearance_backfill is next_backfill


def test_tracking_swap_failure_restores_previous_and_closes_unbound_sessions() -> None:
    service = _lifecycle()
    first = Mock()
    first.camera.id = "gate"
    second = Mock()
    second.camera.id = "yard"
    old_first = Mock()
    replacement_first = Mock()
    replacement_second = Mock()
    first.create_object_tracking_session.return_value = replacement_first
    second.create_object_tracking_session.return_value = replacement_second
    first.replace_object_tracking_session.side_effect = [old_first, replacement_first]
    second.replace_object_tracking_session.side_effect = RuntimeError("swap failed")
    service._workers = {"gate": first, "yard": second}
    next_backfill = Mock()
    service._build_limiter = Mock(return_value=Mock())
    service._build_tracking_factory = Mock(return_value=Mock())
    service._build_backfill = Mock(return_value=next_backfill)

    with pytest.raises(RuntimeError, match="swap failed"):
        service.reconfigure_tracking(AppConfig().detector)

    assert first.replace_object_tracking_session.call_args_list == [
        call(replacement_first),
        call(old_first),
    ]
    replacement_second.stop.assert_called_once_with()
    next_backfill.close.assert_called_once_with()
    assert service.appearance_backfill is not next_backfill


def test_failed_previous_backfill_retirement_does_not_rollback_committed_state() -> None:
    service = _lifecycle()
    previous_backfill = service.appearance_backfill
    previous_backfill.close.side_effect = RuntimeError("busy")
    next_limiter = Mock()
    next_factory = Mock()
    next_backfill = Mock()
    service._build_limiter = Mock(return_value=next_limiter)
    service._build_tracking_factory = Mock(return_value=next_factory)
    service._build_backfill = Mock(return_value=next_backfill)

    service.reconfigure_tracking(AppConfig().detector)

    assert service.appearance_backfill is next_backfill
    assert service.status()["retired_cleanup_pending"] == 1


def test_role_refresh_rolls_back_inference_and_resumes_tracking() -> None:
    service = _lifecycle()
    worker = Mock()
    worker.camera.id = "gate"
    service._workers = {"gate": worker}
    previous = service.detector.config
    next_config = previous.model_copy(deep=True)
    next_config.device = "GPU"
    service.reconfigure_tracking = Mock(side_effect=RuntimeError("tracking failed"))

    with pytest.raises(RuntimeError, match="tracking failed"):
        service.reconfigure_roles(
            next_config,
            {"object"},
            refresh_tracking=True,
        )

    assert service.detector.reconfigure_roles.call_args_list == [
        call(next_config, {"object"}),
        call(previous, {"object"}),
    ]
    assert worker.pause_object_tracking_session.call_count == 2
    worker.resume_object_tracking_session.assert_called_once_with()


def test_semantic_replacement_start_failure_preserves_previous() -> None:
    service = _lifecycle()
    service._auxiliary_started = True
    previous = service.semantic_search
    replacement = Mock()
    replacement.start.side_effect = RuntimeError("load failed")

    with (
        patch(
            "survng.app.inference_lifecycle.build_semantic_search",
            return_value=replacement,
        ),
        pytest.raises(RuntimeError, match="load failed"),
    ):
        service.reconfigure_semantic_search(AppConfig().semantic_search)

    assert service.semantic_search is previous
    previous.close.assert_not_called()
    replacement.close.assert_called_once_with()


def test_semantic_retirement_failure_keeps_new_generation_and_defers_cleanup() -> None:
    service = _lifecycle()
    previous = service.semantic_search
    previous.close.side_effect = RuntimeError("busy")
    replacement = Mock()

    with patch(
        "survng.app.inference_lifecycle.build_semantic_search",
        return_value=replacement,
    ):
        service.reconfigure_semantic_search(AppConfig().semantic_search)

    assert service.semantic_search is replacement
    assert service.status()["retired_cleanup_pending"] == 1

    previous.close.side_effect = None
    service.maintain()

    assert service.status()["retired_cleanup_pending"] == 0
    assert previous.close.call_count == 2

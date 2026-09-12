from __future__ import annotations

import threading
import queue
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from survng.app.config import DetectorConfig, SemanticSearchConfig
from survng.app.inference_runtime.supervisor import InferenceSupervisor
from survng.app.inference_runtime.types import InferenceUnavailable, InferenceWorkload as W
from survng.app.semantic_search import OpenVinoManifestEncoder, SemanticSearchService
from survng.openvino_config import latency_compile_config


def supervisor_with_devices(object_device="GPU", auxiliary_device="CPU"):
    supervisor = InferenceSupervisor(DetectorConfig(
        enabled=False,
        device="GPU",
        face_recognition_device=auxiliary_device,
    ))
    for worker in supervisor._object_workers:
        worker._process = Mock()
        worker._process.is_alive.return_value = True
        worker._status["loaded_device"] = object_device
    supervisor._face._process = Mock()
    supervisor._face._process.is_alive.return_value = True
    supervisor._face._status.update(ready=True, device=auxiliary_device)
    return supervisor


def test_gpu_initial_overlaps_one_running_cpu_auxiliary_without_optional_queue():
    supervisor = supervisor_with_devices()
    entered = threading.Event()
    release = threading.Event()
    errors = []

    def embedding(*args, **kwargs):
        with supervisor._face._lock:
            entered.set()
            assert release.wait(2)
        return [1.0, 0.0]

    def run_embedding():
        try:
            supervisor.embed(np.zeros((2, 2, 3), dtype=np.uint8))
        except BaseException as exc:
            errors.append(exc)

    with patch.object(supervisor._face, "request", side_effect=embedding):
        thread = threading.Thread(target=run_embedding)
        thread.start()
        try:
            assert entered.wait(1)
            assert supervisor._cpu_optional_active == 1
            # Host work stays bounded even before security arrives.
            assert not supervisor._enter_device_workload(W.ENRICHMENT, cpu_only=True)
            # Busy workers cannot expose their actual device; an unknown new
            # optional request must not queue behind the single CPU slot.
            with pytest.raises(InferenceUnavailable):
                supervisor.embed(np.zeros((2, 2, 3), dtype=np.uint8))
            assert supervisor._enter_device_workload(W.INCIDENT_INITIAL, timeout=0)
            try:
                assert not supervisor._enter_device_workload(W.ENRICHMENT, cpu_only=True)
                assert not supervisor._enter_device_workload(W.OFFLINE, shed_optional=False, timeout=0)
            finally:
                supervisor._leave_device_workload(W.INCIDENT_INITIAL)
        finally:
            release.set()
            thread.join(1)
    assert not thread.is_alive()
    assert errors == []
    assert supervisor._cpu_optional_active == supervisor._optional_active == 0


@pytest.mark.parametrize("device", ["CPU", "", "AUTO", "MULTI:GPU,CPU"])
def test_cpu_fallback_or_unknown_object_device_keeps_conservative_exclusion(device):
    supervisor = supervisor_with_devices(object_device=device)
    assert supervisor._enter_device_workload(W.ENRICHMENT, cpu_only=True)
    assert not supervisor._enter_device_workload(W.INCIDENT_INITIAL, timeout=0)
    supervisor._leave_device_workload(W.ENRICHMENT, cpu_only=True)
    assert supervisor._enter_device_workload(W.INCIDENT_INITIAL, timeout=0)
    supervisor._leave_device_workload(W.INCIDENT_INITIAL)


def test_gpu_optional_and_offline_work_still_block_gpu_initial():
    supervisor = supervisor_with_devices(auxiliary_device="GPU")
    assert not supervisor._cpu_auxiliary(supervisor._face)
    for workload in (W.ENRICHMENT, W.OFFLINE):
        assert supervisor._enter_device_workload(workload)
        assert not supervisor._enter_device_workload(W.INCIDENT_INITIAL, timeout=0)
        supervisor._leave_device_workload(workload)


def test_cpu_security_depth_does_not_bypass_cpu_optional_on_gpu_object_host():
    supervisor = supervisor_with_devices()
    assert supervisor._enter_device_workload(W.ENRICHMENT, cpu_only=True)
    assert not supervisor._enter_device_workload(
        W.INCIDENT_INITIAL, timeout=0, security_device="CPU"
    )
    supervisor._leave_device_workload(W.ENRICHMENT, cpu_only=True)


def test_auxiliary_exception_releases_cpu_slot():
    supervisor = supervisor_with_devices()
    with patch.object(supervisor._face, "request", side_effect=RuntimeError("test failure")):
        with pytest.raises(RuntimeError, match="test failure"):
            supervisor.embed(np.zeros((2, 2, 3), dtype=np.uint8))
    assert supervisor._optional_active == supervisor._cpu_optional_active == 0


def test_unknown_and_mixed_auxiliary_devices_are_not_cpu_only():
    supervisor = supervisor_with_devices()
    supervisor._face._status["ready"] = False
    assert not supervisor._cpu_auxiliary(supervisor._face)
    worker = supervisor._reid
    worker.config.tracking.reid_enabled = True
    worker.config.tracking.vehicle_reid_enabled = True
    worker.config.tracking.reid_device = "CPU"
    worker.config.tracking.vehicle_reid_device = "GPU"
    worker._process = Mock()
    worker._status = {
        "person": {"enabled": True, "ready": True, "device": "CPU"},
        "vehicle": {"enabled": True, "ready": True, "device": "CPU"},
    }
    # Even a currently CPU-fallback vehicle can restart on its configured GPU.
    assert not supervisor._cpu_auxiliary(worker)


def test_actual_device_snapshot_is_nonblocking_and_ignores_dead_workers():
    supervisor = supervisor_with_devices()
    worker = supervisor._object_workers[0]
    locked = threading.Event()
    release = threading.Event()

    def hold_lock():
        with worker._lock:
            locked.set()
            assert release.wait(2)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    try:
        assert locked.wait(1)
        assert worker.admission_device() == ""
    finally:
        release.set()
        thread.join(1)
    assert worker.admission_device() == "GPU"
    worker._status["loaded_device"] = "CPU"
    status = worker.isolation_status()
    assert status["loaded_device"] == "CPU"
    assert status["fallback_active"] is False
    worker._process.is_alive.return_value = False
    assert worker.admission_device() == ""
    assert worker.isolation_status()["loaded_device"] == ""


def test_offline_admission_can_be_cancelled_behind_security():
    supervisor = supervisor_with_devices()
    assert supervisor._enter_device_workload(W.INCIDENT_INITIAL)
    cancelled = threading.Event()
    waiting = threading.Event()
    result = []
    original_wait = supervisor._device_condition.wait

    def wait(timeout):
        waiting.set()
        return original_wait(timeout)

    def request():
        try:
            with supervisor.offline_device_lease(cancel_event=cancelled):
                result.append("admitted")
        except InferenceUnavailable:
            result.append("cancelled")

    with patch.object(supervisor._device_condition, "wait", side_effect=wait):
        thread = threading.Thread(target=request)
        thread.start()
        assert waiting.wait(1)
        cancelled.set()
        thread.join(1)
    supervisor._leave_device_workload(W.INCIDENT_INITIAL)
    assert not thread.is_alive()
    assert result == ["cancelled"]
    assert supervisor._offline_active == supervisor._optional_active == 0


@pytest.mark.parametrize("device", ["GPU", "AUTO", "CPU"])
def test_semantic_compilation_uses_shared_device_policy(tmp_path, device):
    image_model = tmp_path / "image_encoder.xml"
    text_model = tmp_path / "text_encoder.xml"
    image_model.touch()
    text_model.touch()
    core = Mock()
    with (
        patch.dict("sys.modules", {"openvino": SimpleNamespace(Core=lambda: core)}),
        patch("survng.app.semantic_search._semantic_tokenizer", return_value=Mock()),
    ):
        OpenVinoManifestEncoder(tmp_path, {}, device, identity=Mock())
    assert [call.args for call in core.compile_model.call_args_list] == [
        (str(image_model), device, latency_compile_config(device)),
        (str(text_model), device, latency_compile_config(device)),
    ]


def test_semantic_startup_and_every_encoder_operation_hold_production_lease(tmp_path):
    supervisor = supervisor_with_devices()
    index = Mock()
    index.event_source_indexed.return_value = False
    index.event_source_keys.return_value = set()
    index.upsert.return_value = 2
    service = SemanticSearchService(SemanticSearchConfig(enabled=True), index, tmp_path, {})
    service.set_device_lease(supervisor.offline_device_lease)
    encoder = Mock()
    operations = []

    def assert_lease(operation):
        assert supervisor._offline_active == 1
        assert not supervisor._enter_device_workload(W.INCIDENT_INITIAL, timeout=0)
        operations.append(operation)

    def startup(*args):
        assert_lease("startup")
        return encoder

    def images(images):
        assert_lease("images")
        return np.ones((len(images), 3))

    def text(texts):
        assert_lease("text")
        return np.ones((len(texts), 3))

    encoder.encode_images.side_effect = images
    encoder.encode_text.side_effect = text
    service._state = "initializing"
    with (
        patch("survng.app.semantic_search.IsolatedOpenVinoManifestEncoder", side_effect=startup),
        patch.object(service, "_run"),
        patch.object(service, "_run_backfill"),
    ):
        service._initialize(Mock())
    assert service.encoder is encoder
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    with (
        patch("survng.app.semantic_search.event_snapshot_path", return_value=Path("test.jpg")),
        patch("survng.app.semantic_search.cv2.imread", return_value=image),
    ):
        service.index_event({"id": 1, "snapshot_path": "test.jpg", "objects_json": '[{"label":"person","box":{"x1":1,"y1":1,"x2":10,"y2":10}}]'})
    service.search_image(image)
    service.search_text("person")
    assert operations == ["startup", "images", "images", "images", "text"]
    assert supervisor._offline_active == supervisor._optional_active == 0
    encoder.encode_text.side_effect = RuntimeError("encoder failed")
    with pytest.raises(RuntimeError, match="encoder failed"):
        service.search_text("person")
    assert supervisor._offline_active == supervisor._optional_active == 0
    service.close()


def test_semantic_admission_failure_does_not_trigger_gpu_fallback(tmp_path):
    service = SemanticSearchService(SemanticSearchConfig(enabled=True), Mock(), tmp_path, {})

    @contextmanager
    def denied(**kwargs):
        service._stop.set()
        raise InferenceUnavailable("security busy")
        yield

    service.set_device_lease(denied)
    with patch("survng.app.semantic_search.IsolatedOpenVinoManifestEncoder") as encoder:
        service._initialize(Mock())
    encoder.assert_not_called()
    assert service._initialization_attempts == 0
    assert not service._fallback_active


@pytest.mark.parametrize("pause_at", ["admission", "request"])
def test_cpu_reconfiguration_cannot_reuse_stale_cpu_admission(pause_at):
    supervisor = supervisor_with_devices()
    worker = supervisor._face
    classified = threading.Event()
    resume = threading.Event()
    errors = []
    target = supervisor if pause_at == "admission" else worker
    method = "_enter_device_workload" if pause_at == "admission" else "request"
    original = getattr(target, method)

    def pause(*args, **kwargs):
        classified.set()
        assert resume.wait(2)
        return original(*args, **kwargs)

    def embed():
        try:
            supervisor.embed(np.zeros((2, 2, 3), dtype=np.uint8))
        except InferenceUnavailable as exc:
            errors.append(str(exc))

    config = supervisor.config.model_copy(deep=True)
    config.face_recognition_device = "GPU"
    with (
        patch.object(target, method, side_effect=pause),
        patch.object(worker, "_ensure_worker_locked") as ensure,
        patch.object(worker, "stop"),
    ):
        thread = threading.Thread(target=embed)
        thread.start()
        try:
            assert classified.wait(1)
            # Complete the real role reconfiguration while the caller carries
            # its earlier CPU classification. Disabled auxiliary => no startup.
            supervisor.reconfigure_roles(config, {"face"})
        finally:
            resume.set()
            thread.join(1)
    assert not thread.is_alive()
    assert errors and "reconfiguration" in errors[0]
    ensure.assert_not_called()  # Must not even compile/start a GPU replacement.
    assert supervisor._optional_active == supervisor._cpu_optional_active == 0


def test_cpu_admission_rechecks_actual_worker_after_startup():
    supervisor = supervisor_with_devices()
    worker = supervisor._face
    worker._status["device"] = "GPU"
    worker._connection = Mock()
    with patch.object(worker, "_ensure_worker_locked", return_value=True):
        with pytest.raises(InferenceUnavailable, match="loaded CPU"):
            worker.request("embed", require_cpu=True)
    worker._connection.send.assert_not_called()


def test_semantic_yields_to_waiting_security_between_images_without_reordering(tmp_path):
    supervisor = supervisor_with_devices()
    index = Mock()
    index.event_source_indexed.return_value = False
    index.event_source_keys.return_value = set()
    index.upsert.return_value = 2
    service = SemanticSearchService(SemanticSearchConfig(enabled=True), index, tmp_path, {})
    service.set_device_lease(supervisor.offline_device_lease)
    service.encoder = Mock()
    order = []
    security_thread = None

    def security():
        assert supervisor._enter_device_workload(W.INCIDENT_INITIAL)
        order.append("security")
        supervisor._leave_device_workload(W.INCIDENT_INITIAL)

    def encode(images):
        nonlocal security_thread
        assert len(images) == 1
        if not order:
            order.append("cover")
            security_thread = threading.Thread(target=security)
            security_thread.start()
            with supervisor._device_condition:
                assert supervisor._device_condition.wait_for(
                    lambda: supervisor._security_waiting == 1, timeout=1
                )
            return np.asarray([[1., 0., 0.]])
        assert order == ["cover", "security"]
        order.append("crop")
        return np.asarray([[0., 1., 0.]])

    service.encoder.encode_images.side_effect = encode
    with (
        patch("survng.app.semantic_search.event_snapshot_path", return_value=Path("test.jpg")),
        patch("survng.app.semantic_search.cv2.imread", return_value=np.zeros((20, 20, 3), dtype=np.uint8)),
    ):
        service.index_event({"id": 1, "snapshot_path": "test.jpg", "objects_json": '[{"label":"person","box":{"x1":1,"y1":1,"x2":10,"y2":10}}]'})
    security_thread.join(1)
    assert order == ["cover", "security", "crop"]
    evidence, embeddings, _identity = index.upsert.call_args.args
    assert [item.source_kind for item in evidence] == ["full_frame", "object_crop"]
    np.testing.assert_array_equal(embeddings, [[1., 0., 0.], [0., 1., 0.]])


def test_semantic_contention_retry_survives_full_queue_and_yields_to_live_events(tmp_path):
    service = SemanticSearchService(SemanticSearchConfig(enabled=True), Mock(), tmp_path, {})
    # Normal producers may queue one item; the sole consumer retains the
    # already-running history item using one bounded retry reservation.
    service._queue.maxsize = 1
    history = {"id": 1}
    live = {"id": 2}
    service._queue.put((1, next(service._queue_sequence), history))
    order = []

    def index(event):
        order.append(event["id"])
        if len(order) == 1:
            service._queue.put_nowait((0, next(service._queue_sequence), live))
            raise InferenceUnavailable("production is busy")
        if len(order) == 2:
            assert service._queue.qsize() == 1  # History retry is retained.
            with pytest.raises(queue.Full):
                service._queue.put_nowait((0, 99, {"id": 3}))
        if len(order) == 3:
            service._stop.set()

    with patch.object(service, "index_event", side_effect=index):
        service._run()
    assert order == [1, 2, 1]
    assert service._queue.qsize() == 0

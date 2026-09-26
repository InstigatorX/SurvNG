export function formatInferenceMs(value) {
  if (value == null || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${number.toFixed(number >= 100 ? 0 : 1)} ms`;
}

export function workerWeight(detectorConfig, workerId) {
  const weights = detectorConfig?.inference_worker_weights || {};
  if (Object.prototype.hasOwnProperty.call(weights, workerId)) {
    const override = Number(weights[workerId]);
    return Number.isFinite(override) ? override : 1;
  }
  const fallback = Number(detectorConfig?.inference_default_worker_weight);
  return Number.isFinite(fallback) ? fallback : 1;
}

export function inferenceTargetRows(detector, detectorConfig = {}) {
  const registry = detector?.remote_registry || {};
  const runtime = detector?.runtime || {};
  const loaded = Boolean(
    detector?.openvino_loaded
    || detector?.opencv_loaded
    || detector?.coreml_loaded
    || detector?.loaded_backend
  );
  const rows = [{
    id: "primary",
    name: "Primary",
    kind: "primary",
    ready: loaded,
    device: detector?.loaded_device || detector?.configured_device || "—",
    roles: ["object"],
    pending: Number(detector?.isolation?.pending_requests || 0),
    completed: Number(runtime.total_inferences || 0),
    completedLabel: "inferences",
    failed: Number(runtime.failed_inferences || 0),
    lastInferenceMs: runtime.last_inference_ms,
    averageInferenceMs: runtime.average_inference_ms,
    modelLoadMs: detector?.model_load_ms,
    leaseSeconds: null,
    weight: Number(detectorConfig.inference_primary_weight ?? 1),
  }];
  for (const worker of registry.workers || []) {
    const objectStatus = worker?.statuses?.object || {};
    const objectRuntime = objectStatus.runtime || {};
    rows.push({
      id: String(worker.worker_id || ""),
      name: worker.name || worker.worker_id || "Worker",
      kind: "worker",
      ready: Boolean(worker.ready),
      device: objectStatus.loaded_device || objectStatus.configured_device || "—",
      roles: Array.isArray(worker.roles) ? worker.roles : [],
      pending: Number(worker.pending_requests || 0),
      completed: Number(worker.completed_requests || 0),
      completedLabel: "requests",
      failed: Number(worker.failed_requests || 0),
      lastInferenceMs: objectRuntime.last_inference_ms,
      averageInferenceMs: objectRuntime.average_inference_ms,
      modelLoadMs: objectStatus.model_load_ms,
      leaseSeconds: Number(worker.lease_remaining_seconds),
      weight: workerWeight(detectorConfig, worker.worker_id),
    });
  }
  return rows;
}

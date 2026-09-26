export function formatCommit(value) {
  const text = String(value || "").trim();
  if (!text) return "—";
  return text.slice(0, 12);
}

export function formatInferenceMs(value) {
  if (value == null || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${number.toFixed(number >= 100 ? 0 : 1)} ms`;
}

export function formatAttemptOutcome(row) {
  if (!row?.lastOutcome) return "";
  const call = [row.lastRole, row.lastOperation].filter(Boolean).join(" ");
  if (row.lastOutcome === "completed") {
    return call ? `Completed ${call}` : "Completed";
  }
  if (row.lastOutcome === "rerouted") {
    const reason = row.lastError ? `: ${row.lastError}` : "";
    return `Rerouted${call ? ` ${call}` : ""}${reason}`;
  }
  if (row.lastOutcome === "failed") {
    const reason = row.lastError ? `: ${row.lastError}` : "";
    return `Failed${call ? ` ${call}` : ""}${reason}`;
  }
  return "";
}

export function formatRoleAttempts(attempts) {
  if (!attempts || typeof attempts !== "object") return "";
  return Object.entries(attempts).map(([role, counts]) => {
    const parts = [];
    const completed = Number(counts?.completed || 0);
    const rerouted = Number(counts?.rerouted || 0);
    const failed = Number(counts?.failed || 0);
    if (completed) parts.push(`${completed} completed`);
    if (rerouted) parts.push(`${rerouted} rerouted`);
    if (failed) parts.push(`${failed} failed`);
    return parts.length ? `${role}: ${parts.join(", ")}` : "";
  }).filter(Boolean).join(" · ");
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
    rerouted: null,
    lastOutcome: "",
    lastError: "",
    lastRole: "",
    lastOperation: "",
    lastRequestMs: null,
    roleAttempts: {},
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
      rerouted: Number(worker.rerouted_requests || 0),
      failed: Number(worker.failed_requests || 0),
      lastOutcome: String(worker.last_outcome || ""),
      lastError: String(worker.last_error || ""),
      lastRole: String(worker.last_role || ""),
      lastOperation: String(worker.last_operation || ""),
      lastRequestMs: worker.last_request_ms,
      roleAttempts: worker.role_attempts || {},
      lastInferenceMs: worker.last_inference_ms ?? objectRuntime.last_inference_ms,
      averageInferenceMs: worker.average_inference_ms ?? objectRuntime.average_inference_ms,
      modelLoadMs: objectStatus.model_load_ms,
      leaseSeconds: Number(worker.lease_remaining_seconds),
      weight: workerWeight(detectorConfig, worker.worker_id),
      softwareVersion: String(worker.software_version || ""),
      upgradePhase: String(worker.upgrade_phase || ""),
      upgradeDetail: String(worker.upgrade_detail || ""),
      codeMatches: Boolean(detector?.primary_sha) && worker.software_version === detector.primary_sha,
    });
  }
  return rows;
}

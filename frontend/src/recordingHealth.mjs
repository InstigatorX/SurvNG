const STALE_AFTER_MS = 90_000;

function finite(value) { return typeof value === "number" && Number.isFinite(value); }
function cameraConfigMap(config) {
  const list = Array.isArray(config?.cameras) ? config.cameras : [];
  return new Map(list.filter((camera) => camera?.id).map((camera) => [camera.id, camera]));
}

export function formatDuration(seconds) {
  if (!finite(seconds) || seconds < 0) return "Unavailable";
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${days ? `${days}d ` : ""}${hours}h ${minutes}m`;
}

export function recordingHealth({ cameras, appConfig, system, camerasUpdatedAt, systemUpdatedAt, camerasError = false, systemError = false, now = Date.now() } = {}) {
  const configById = cameraConfigMap(appConfig);
  const runtimeById = new Map(Array.isArray(cameras) ? cameras.filter((camera) => camera?.id).map((camera) => [camera.id, camera]) : []);
  const ids = new Set([...configById.keys(), ...runtimeById.keys()]);
  const cameraFresh = !camerasError && finite(camerasUpdatedAt) && now - camerasUpdatedAt <= STALE_AFTER_MS;
  const systemFresh = !systemError && finite(systemUpdatedAt) && now - systemUpdatedAt <= STALE_AFTER_MS;
  const cameraDataKnown = Array.isArray(cameras) && Array.isArray(appConfig?.cameras) && cameraFresh;
  const rows = [...ids].map((id) => {
    const config = configById.get(id);
    const runtime = runtimeById.get(id);
    const enabled = runtime?.expected_enabled ?? (config?.enabled !== false);
    const mainRequired = config ? config.record === true : runtime?.recording_configured === true;
    const subRequired = config ? config.record_sub === true : runtime?.record_sub_enabled === true;
    const intentionallyStopped = runtime?.recording_enabled === false;
    const expected = Boolean(enabled && !intentionallyStopped && (mainRequired || subRequired));
    const missing = [];
    if (expected && !runtime) missing.push("runtime status");
    else if (expected) {
      if (mainRequired && runtime.recording !== true) missing.push("main stream");
      if (subRequired && runtime.sub_recording !== true) missing.push("substream");
      if (runtime.running === false) missing.push("camera process");
      if (runtime.capture_connectivity === false || ["offline", "disconnected", "error"].includes(runtime.capture_connectivity)) missing.push("capture");
    }
    const state = !cameraDataKnown ? "stale" : !enabled ? "disabled" : !expected ? "paused" : !runtime ? "unavailable" : missing.length ? "attention" : "healthy";
    return { id, name: config?.name || runtime?.name || id, expected, mainRequired, subRequired, state, missing, runtime };
  });
  const expectedRows = rows.filter((row) => row.expected);
  const activeCount = cameraDataKnown ? expectedRows.filter((row) => row.state === "healthy").length : null;
  const retention = appConfig?.retention || {};
  const warningPercent = finite(retention.minimum_free_percent) ? retention.minimum_free_percent : 15;
  const emergencyPercent = finite(retention.emergency_free_percent) ? retention.emergency_free_percent : 5;
  const storage = system?.storage;
  const validStorage = storage?.available === true && finite(storage?.free_bytes) && finite(storage?.total_bytes)
    && storage.total_bytes > 0 && storage.free_bytes >= 0 && storage.free_bytes <= storage.total_bytes;
  const freePercent = validStorage ? (storage.free_bytes / storage.total_bytes) * 100 : null;
  let storageState = "unavailable";
  if (systemFresh && validStorage) {
    storageState = freePercent <= emergencyPercent ? "critical" : freePercent <= warningPercent ? "warning" : "healthy";
  }
  const cameraIssue = !cameraDataKnown || rows.some((row) => row.state === "attention" || row.state === "unavailable");
  const storageIssue = systemFresh && storageState !== "healthy";
  const systemReasons = [];
  if (systemFresh) {
    if (system?.lifecycle && system.lifecycle !== "running") systemReasons.push(`System is ${String(system.lifecycle).replaceAll("_", " ")}.`);
    if (system?.detector && system.detector.enabled !== false && !system.detector.loaded_backend) systemReasons.push("Detection is enabled but its backend is not loaded.");
    if (finite(system?.cameras?.enabled) && finite(system?.cameras?.online) && system.cameras.enabled > system.cameras.online) systemReasons.push(`${system.cameras.online} of ${system.cameras.enabled} enabled cameras are online.`);
  }
  const issues = Number(cameraIssue) + Number(!systemFresh || storageIssue) + Number(systemReasons.length > 0);
  return {
    rows, systemReasons, expectedCount: expectedRows.length, activeCount, issues, cameraFresh, cameraDataKnown, systemFresh,
    storage: { ...storage, state: storageState, freePercent, warningPercent, emergencyPercent },
    uptimeSeconds: system?.uptime_seconds,
  };
}

// Use the same explanations in hover cards and the touch-accessible Details panel.
export function recordingHealthContext(health) {
  const cameraReasons = !health.cameraDataKnown
    ? ["Camera status or configuration is unavailable or stale; recording cannot be confirmed."]
    : health.rows.filter((row) => row.state === "attention" || row.state === "unavailable")
      .map((row) => `${row.name}: missing ${row.missing.join(" and ") || "runtime status"}.`);
  const storage = health.storage;
  const storageReason = storage.state === "unavailable"
    ? !health.systemFresh ? "System status is unavailable or stale; free storage cannot be confirmed." : "Storage status is unavailable; free space cannot be confirmed."
    : storage.state === "critical" ? `Storage is critically low: ${storage.freePercent.toFixed(1)}% free, at or below the ${storage.emergencyPercent}% emergency threshold.`
    : storage.state === "warning" ? `Storage has ${storage.freePercent.toFixed(1)}% free, at or below the ${storage.warningPercent}% cleanup threshold.`
    : `Storage has ${storage.freePercent.toFixed(1)}% free, above the ${storage.warningPercent}% cleanup threshold.`;
  const excluded = health.rows.filter((row) => row.state === "paused" || row.state === "disabled").length;
  return {
    recording: [health.cameraDataKnown ? `${health.activeCount} of ${health.expectedCount} expected cameras have all configured recording streams active.` : cameraReasons[0],
      ...(health.cameraDataKnown ? cameraReasons : []),
      ...(excluded ? [`${excluded} paused or disabled camera${excluded === 1 ? " is" : "s are"} excluded from the recording count.`] : [])],
    storage: [storageReason, `Cleanup threshold: ${storage.warningPercent}% free. Emergency threshold: ${storage.emergencyPercent}% free.`],
    attention: [...cameraReasons, ...(storage.state !== "healthy" ? [storageReason] : []), ...health.systemReasons],
  };
}

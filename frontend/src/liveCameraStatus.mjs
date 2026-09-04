function cameraControlStatus({ kind, enabled, available = true, busy = false, error = "" }) {
  const label = kind === "recording" ? "Recording" : "Detection";
  const shortLabel = kind === "recording" ? "REC" : "DET";
  const detail = String(error || "").trim();

  if (detail) {
    return { kind, state: "error", shortLabel, symbol: "!", label: `${label} status error: ${detail}`, title: `${label} error: ${detail}` };
  }
  if (busy) {
    return { kind, state: "busy", shortLabel, symbol: "…", label: `${label} updating`, title: `${label} status is updating` };
  }
  if (!available) {
    return { kind, state: "unavailable", shortLabel, symbol: "—", label: `${label} unavailable`, title: `${label} is not configured for this camera` };
  }
  if (enabled) {
    return { kind, state: "on", shortLabel, symbol: "✓", label: `${label} on`, title: `${label} is on` };
  }
  return { kind, state: "off", shortLabel, symbol: "×", label: `${label} off`, title: `${label} is off` };
}

export function recordingStatus(camera, { busy = false, error = "" } = {}) {
  return cameraControlStatus({
    kind: "recording",
    enabled: Boolean(camera?.recording_enabled),
    available: Boolean(camera?.recording_configured),
    busy,
    error,
  });
}

export function detectionStatus(camera, { busy = false, error = "" } = {}) {
  return cameraControlStatus({
    kind: "detection",
    enabled: Boolean(camera?.detection_enabled),
    busy,
    error,
  });
}

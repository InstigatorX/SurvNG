/** Derive operator-facing live capture connectivity from camera status payloads. */

export function cameraCaptureConnectivity(camera) {
  const state = String(camera?.capture_connectivity || "").trim().toLowerCase();
  if (state === "healthy" || state === "reconnecting" || state === "offline" || state === "paused" || state === "asleep" || state === "wake_failed") {
    return state;
  }
  const running = Boolean(camera?.running);
  const connected = Boolean(camera?.connected);
  const captureRunning = Boolean(camera?.capture_running ?? running);
  if (!running) return "paused";
  if (connected) return "healthy";
  if (captureRunning) return "reconnecting";
  return "offline";
}

export function cameraConnectivityLabel(state) {
  switch (cameraCaptureConnectivity({ capture_connectivity: state })) {
    case "healthy":
      return "Live";
    case "reconnecting":
      return "Reconnecting";
    case "offline":
      return "Offline";
    case "asleep":
      return "Asleep";
    case "wake_failed":
      return "Wake failed";
    default:
      return "Paused";
  }
}

export function cameraConnectivityClass(state) {
  switch (cameraCaptureConnectivity({ capture_connectivity: state })) {
    case "healthy":
      return "healthy";
    case "reconnecting":
      return "attention";
    case "offline":
      return "offline";
    case "asleep":
      return "asleep";
    case "wake_failed":
      return "attention";
    default:
      return "disabled";
  }
}

export function cameraTileLiveState(camera) {
  const connectivity = cameraCaptureConnectivity(camera);
  if (connectivity === "healthy") return "LIVE";
  if (connectivity === "reconnecting") return "RECON";
  if (connectivity === "asleep") return "ASLEEP";
  if (connectivity === "wake_failed") return "WAKE";
  if (connectivity === "paused") return "OFF";
  return "OFFLINE";
}

/** Derive operator-facing live capture connectivity from camera status payloads. */

export function cameraCaptureConnectivity(camera) {
  const state = String(camera?.capture_connectivity || "").trim().toLowerCase();
  if (state === "healthy" || state === "reconnecting" || state === "offline" || state === "paused") {
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
    default:
      return "disabled";
  }
}

export function cameraTileLiveState(camera) {
  const connectivity = cameraCaptureConnectivity(camera);
  if (connectivity === "healthy") return "LIVE";
  if (connectivity === "reconnecting") return "RECON";
  if (connectivity === "paused") return "OFF";
  return "OFFLINE";
}

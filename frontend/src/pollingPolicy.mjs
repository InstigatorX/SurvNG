export function liveSnapshotRefreshMs({ running, visible, documentVisible, streamReady, transport, mobile, primary, userActive = true }) {
  if (!running || !visible || !documentVisible || !userActive) return null;
  if (streamReady && transport !== "snapshot") return null;
  if (!mobile) return 2000;
  return primary ? 2000 : 12000;
}

export function liveMediaShouldRun({ running, streamReady, mediaActive, transport }) {
  return Boolean(running && streamReady && mediaActive && ["webrtc", "mjpeg"].includes(transport));
}

export function logPayloadSignature(lines = []) {
  const values = Array.isArray(lines) ? lines : [];
  return JSON.stringify(values);
}

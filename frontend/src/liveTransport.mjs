export const WEBRTC_FAILURE_COOLDOWN_MS = 5 * 60 * 1000;

export function nextNativeFallbackSource(requestedSource, currentSource) {
  return requestedSource === "main" && currentSource === "main" ? "live" : null;
}

export function webRtcFailureKey(cameraId, source) {
  return `survng.webrtcFailure.v1.${cameraId}.${source}`;
}

export function shouldSkipWebRtc(failedAt, now = Date.now()) {
  const timestamp = Number(failedAt);
  return Number.isFinite(timestamp)
    && timestamp > 0
    && timestamp <= now
    && now - timestamp < WEBRTC_FAILURE_COOLDOWN_MS;
}

export function initialLiveTransport(cameraId, source, storage = globalThis.localStorage, now = Date.now()) {
  try {
    return shouldSkipWebRtc(storage.getItem(webRtcFailureKey(cameraId, source)), now)
      ? "mse"
      : "webrtc";
  } catch {
    return "webrtc";
  }
}

export function rememberWebRtcFailure(cameraId, source, storage = globalThis.localStorage, now = Date.now()) {
  try {
    storage.setItem(webRtcFailureKey(cameraId, source), String(now));
  } catch {
    // Storage may be unavailable in private or embedded browser contexts.
  }
}

export function clearWebRtcFailure(cameraId, source, storage = globalThis.localStorage) {
  try {
    storage.removeItem(webRtcFailureKey(cameraId, source));
  } catch {
    // Storage may be unavailable in private or embedded browser contexts.
  }
}

export function webRtcRetryDelay(cameraId, source, storage = globalThis.localStorage, now = Date.now()) {
  try {
    const failedAt = Number(storage.getItem(webRtcFailureKey(cameraId, source)));
    if (!shouldSkipWebRtc(failedAt, now)) return 0;
    return Math.max(0, WEBRTC_FAILURE_COOLDOWN_MS - (now - failedAt));
  } catch {
    return WEBRTC_FAILURE_COOLDOWN_MS;
  }
}

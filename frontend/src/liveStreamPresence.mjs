export const LIVE_STREAM_IDLE_MS = 15 * 60 * 1000;

export function liveStreamIdleLabel(idleMs = LIVE_STREAM_IDLE_MS) {
  const minutes = Math.max(1, Math.round(Number(idleMs) / 60000));
  return `${minutes} minute${minutes === 1 ? "" : "s"}`;
}

export function livePresenceActive({
  lastActivityAt,
  now,
  documentVisible,
  idleMs = LIVE_STREAM_IDLE_MS,
} = {}) {
  if (!documentVisible) return false;
  if (!Number.isFinite(lastActivityAt) || !Number.isFinite(now) || !Number.isFinite(idleMs)) return false;
  return now - lastActivityAt < idleMs;
}

export function nextLivePresence({
  lastActivityAt,
  now,
  documentVisible,
  idleMs = LIVE_STREAM_IDLE_MS,
} = {}) {
  const active = livePresenceActive({ lastActivityAt, now, documentVisible, idleMs });
  const elapsed = Number(now) - Number(lastActivityAt);
  const remaining = Number(idleMs) - elapsed;
  return {
    active,
    documentVisible: Boolean(documentVisible),
    nextCheckMs: documentVisible && active && Number.isFinite(remaining) ? Math.max(0, remaining) : null,
  };
}

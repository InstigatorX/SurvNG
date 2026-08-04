export const LIVE_DEFAULTS_INSTANCE_KEY = "survng.liveDefaults.serverInstance.v1";

const RESTART_RESET_PREFIXES = [
  "survng.streamMode.v3.",
  "survng.sourceMode.",
  "survng.liveOverlaySource.",
];

export function resetLiveDefaultsForServer(storage, instanceId) {
  const normalizedInstance = String(instanceId || "").trim();
  if (!storage || !normalizedInstance) return false;
  try {
    if (storage.getItem(LIVE_DEFAULTS_INSTANCE_KEY) === normalizedInstance) return false;
    const resetKeys = [];
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index);
      if (key && RESTART_RESET_PREFIXES.some((prefix) => key.startsWith(prefix))) resetKeys.push(key);
    }
    resetKeys.forEach((key) => storage.removeItem(key));
    storage.setItem(LIVE_DEFAULTS_INSTANCE_KEY, normalizedInstance);
    return true;
  } catch {
    return false;
  }
}


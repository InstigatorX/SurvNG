export const RUNTIME_STATE_PATHS = Object.freeze([
  "/api/cameras",
  "/api/config",
  "/api/system/status",
]);

async function requestRuntimePayload(request, path, signal) {
  try {
    const response = await request(path, { signal });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

export async function loadRuntimeState(request, { signal } = {}) {
  const [cameras, appConfig, system] = await Promise.all(
    RUNTIME_STATE_PATHS.map((path) => requestRuntimePayload(request, path, signal)),
  );
  return {
    cameras: Array.isArray(cameras) ? cameras : null,
    appConfig: appConfig && typeof appConfig === "object" ? appConfig : null,
    system: system && typeof system === "object" ? system : null,
  };
}

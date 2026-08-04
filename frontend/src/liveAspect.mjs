export const DEFAULT_LIVE_ASPECT = "16 / 9";

export function normalizedLiveSource(source) {
  return source === "main" ? "main" : "live";
}

export function liveAspectStorageKey(cameraId, source) {
  return `survng.liveAspect.${cameraId}.${normalizedLiveSource(source)}`;
}

export function aspectFromDimensions(width, height) {
  const safeWidth = Number(width);
  const safeHeight = Number(height);
  if (!Number.isFinite(safeWidth) || !Number.isFinite(safeHeight) || safeWidth <= 0 || safeHeight <= 0) {
    return null;
  }
  return `${Math.round(safeWidth)} / ${Math.round(safeHeight)}`;
}

export function validLiveAspect(aspect) {
  const [width, height, ...rest] = String(aspect || "").split("/").map((value) => Number(value.trim()));
  if (rest.length || !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }
  return `${width} / ${height}`;
}

export function cameraSourceAspect(camera, source) {
  const dimensions = camera?.stream_dimensions?.[normalizedLiveSource(source)];
  return aspectFromDimensions(dimensions?.width, dimensions?.height);
}

export function storedCameraAspect(storage, cameraId, source) {
  try {
    return validLiveAspect(storage?.getItem(liveAspectStorageKey(cameraId, source)));
  } catch {
    return null;
  }
}

export function initialCameraAspect(camera, source, storage) {
  return cameraSourceAspect(camera, source)
    || storedCameraAspect(storage, camera?.id, source)
    || DEFAULT_LIVE_ASPECT;
}


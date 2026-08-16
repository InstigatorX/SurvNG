export const DEFAULT_LIVE_FRAMING = Object.freeze({
  fit: "cover",
  focalX: 50,
  focalY: 50,
  zoom: 1,
});

function boundedNumber(value, minimum, maximum, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(minimum, Math.min(maximum, number));
}

export function normalizedLiveFraming(camera, source = "live") {
  const normalizedSource = String(source || "live").toLowerCase() === "main" ? "main" : "live";
  const value = camera?.live_view?.[normalizedSource] || {};
  return {
    fit: value.fit === "contain" ? "contain" : "cover",
    focalX: boundedNumber(value.focal_x, 0, 100, DEFAULT_LIVE_FRAMING.focalX),
    focalY: boundedNumber(value.focal_y, 0, 100, DEFAULT_LIVE_FRAMING.focalY),
    zoom: boundedNumber(value.zoom, 1, 3, DEFAULT_LIVE_FRAMING.zoom),
  };
}

export function liveFramingStyle(camera, source = "live") {
  const framing = normalizedLiveFraming(camera, source);
  return {
    "--live-object-fit": framing.fit,
    "--live-object-position": `${framing.focalX}% ${framing.focalY}%`,
    "--live-view-zoom": String(framing.zoom),
  };
}

export function camerasWithLiveFraming(statusCameras, configuredCameras) {
  const configById = new Map((configuredCameras || []).map((camera) => [String(camera?.id || ""), camera]));
  return (statusCameras || []).map((camera) => ({
    ...camera,
    live_view: configById.get(String(camera?.id || ""))?.live_view || camera?.live_view,
  }));
}

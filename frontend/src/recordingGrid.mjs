export function recordingCameraAspect(camera, source = "live") {
  const dimensions = camera?.stream_dimensions?.[source]
    || camera?.stream_dimensions?.live
    || camera?.stream_dimensions?.main;
  const width = Number(dimensions?.width);
  const height = Number(dimensions?.height);
  if (!(width > 0) || !(height > 0)) return 16 / 9;
  return Math.max(0.25, Math.min(4, width / height));
}

export function recordingGridBestEpoch(ranges, requestedEpoch, lookbackSeconds = 300) {
  const requested = Number(requestedEpoch);
  if (!Number.isFinite(requested)) return null;
  const windowStart = requested - Math.max(1, Number(lookbackSeconds) || 300);
  const normalized = (ranges || []).map((range) => ({
    cameraId: String(range.camera_id || ""),
    start: Number(range.start_epoch),
    end: Number(range.end_epoch),
  })).filter((range) => (
    range.cameraId && Number.isFinite(range.start) && Number.isFinite(range.end)
    && range.end > range.start && range.end > windowStart && range.start <= requested
  ));
  if (!normalized.length) return null;
  const candidates = new Set([requested]);
  normalized.forEach((range) => {
    candidates.add(Math.max(windowStart, range.start));
    // Stay clear of a lagging camera's indexed edge so playback does not lose
    // that tile immediately after the synchronized grid begins advancing.
    candidates.add(Math.min(requested, Math.max(range.start, range.end - 2)));
  });
  let best = null;
  for (const candidate of candidates) {
    if (!Number.isFinite(candidate) || candidate < windowStart || candidate > requested) continue;
    const cameraCount = new Set(normalized
      .filter((range) => range.start <= candidate && candidate < range.end)
      .map((range) => range.cameraId)).size;
    if (!best || cameraCount > best.cameraCount || (cameraCount === best.cameraCount && candidate > best.epoch)) {
      best = { epoch: candidate, cameraCount };
    }
  }
  return best?.cameraCount ? best.epoch : null;
}

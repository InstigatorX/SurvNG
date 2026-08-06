export function recordingCameraAspect(camera, source = "live") {
  const dimensions = camera?.stream_dimensions?.[source]
    || camera?.stream_dimensions?.live
    || camera?.stream_dimensions?.main;
  const width = Number(dimensions?.width);
  const height = Number(dimensions?.height);
  if (!(width > 0) || !(height > 0)) return 16 / 9;
  return Math.max(0.25, Math.min(4, width / height));
}

function partitionRows(items, rowCount) {
  const rows = [];
  let offset = 0;
  let remainingAspect = items.reduce((sum, item) => sum + item.aspect, 0);
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    const rowsRemaining = rowCount - rowIndex;
    const minimumRemainingItems = rowsRemaining - 1;
    const targetAspect = remainingAspect / rowsRemaining;
    const row = [];
    let rowAspect = 0;
    while (offset < items.length && items.length - offset > minimumRemainingItems) {
      const next = items[offset];
      if (row.length && rowAspect + next.aspect > targetAspect) {
        const currentDistance = Math.abs(targetAspect - rowAspect);
        const nextDistance = Math.abs(targetAspect - rowAspect - next.aspect);
        if (currentDistance <= nextDistance) break;
      }
      row.push(next);
      rowAspect += next.aspect;
      offset += 1;
    }
    if (!row.length && offset < items.length) {
      row.push(items[offset]);
      rowAspect += items[offset].aspect;
      offset += 1;
    }
    remainingAspect -= rowAspect;
    rows.push({ items: row, aspect: rowAspect });
  }
  return rows;
}

function portraitLayoutScale(aspect, portraitPriority) {
  if (!portraitPriority || aspect >= 1) return 1;
  // Portrait cameras otherwise inherit the common row height and become tiny,
  // narrow tiles. Give increasingly tall sources more visual area while
  // preserving their native aspect ratio and keeping the boost bounded.
  return 1 + Math.min(0.5, ((1 / Math.max(0.25, aspect)) - 1) * 0.6);
}

export function recordingGridLayout(
  cameras,
  source,
  width,
  height,
  gap = 6,
  aspectOverrides = {},
  { portraitPriority = false } = {},
) {
  if (!cameras.length || width <= 0 || height <= 0) return [];
  const items = cameras.map((camera) => {
    const nativeAspect = Number(aspectOverrides[camera.id]) > 0
      ? Number(aspectOverrides[camera.id])
      : recordingCameraAspect(camera, source);
    const scale = portraitLayoutScale(nativeAspect, portraitPriority);
    return {
      camera,
      nativeAspect,
      scale,
      // partitionRows balances rendered width, including the portrait boost.
      aspect: nativeAspect * scale,
    };
  });
  let best = null;
  const maximumRows = Math.min(items.length, 12);
  for (let rowCount = 1; rowCount <= maximumRows; rowCount += 1) {
    const rows = partitionRows(items, rowCount);
    const rowScales = rows.map((row) => Math.max(...row.items.map((item) => item.scale)));
    const heightLimit = (height - gap * (rows.length - 1))
      / rowScales.reduce((sum, scale) => sum + scale, 0);
    const widthLimit = Math.min(...rows.map((row) => (
      (width - gap * (row.items.length - 1)) / row.aspect
    )));
    const rowHeight = Math.max(1, Math.min(heightLimit, widthLimit));
    if (!best || rowHeight > best.rowHeight) best = { rows, rowHeight, rowScales };
  }

  const totalHeight = best.rowScales.reduce((sum, scale) => sum + scale * best.rowHeight, 0)
    + gap * (best.rows.length - 1);
  let y = Math.max(0, (height - totalHeight) / 2);
  const layout = [];
  best.rows.forEach((row, rowIndex) => {
    const renderedRowHeight = best.rowScales[rowIndex] * best.rowHeight;
    const rowWidth = row.aspect * best.rowHeight + gap * (row.items.length - 1);
    let x = Math.max(0, (width - rowWidth) / 2);
    for (const item of row.items) {
      const itemWidth = item.aspect * best.rowHeight;
      const itemHeight = item.scale * best.rowHeight;
      layout.push({
        camera: item.camera,
        x,
        y: y + (renderedRowHeight - itemHeight) / 2,
        width: itemWidth,
        height: itemHeight,
      });
      x += itemWidth + gap;
    }
    y += renderedRowHeight + gap;
  });
  return layout;
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

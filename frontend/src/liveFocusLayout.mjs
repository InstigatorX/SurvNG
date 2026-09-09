import { UNIFORM_LIVE_TILE_ASPECT } from "./liveWorkspace.mjs";

export function focusedLiveCameraId(cameras, storedCameraId) {
  const ids = (cameras || []).map((camera) => String(camera?.id || "")).filter(Boolean);
  const stored = String(storedCameraId || "");
  return ids.includes(stored) ? stored : ids[0] || "";
}

function fitAspect(width, height, aspect) {
  const fittedWidth = Math.min(width, height * aspect);
  const fittedHeight = fittedWidth / aspect;
  return {
    x: Math.max(0, (width - fittedWidth) / 2),
    y: Math.max(0, (height - fittedHeight) / 2),
    width: fittedWidth,
    height: fittedHeight,
  };
}

export function focusLiveMosaicLayout(cameras, width, height, storedCameraId, gap = 4, preferredAspect = UNIFORM_LIVE_TILE_ASPECT) {
  const items = [...(cameras || [])];
  const availableWidth = Number(width);
  const availableHeight = Number(height);
  const requestedGap = Number(gap);
  const requestedAspect = Number(preferredAspect);
  const gutter = Number.isFinite(requestedGap) ? Math.max(0, requestedGap) : 0;
  const primaryAspect = Number.isFinite(requestedAspect) && requestedAspect > 0 ? requestedAspect : UNIFORM_LIVE_TILE_ASPECT;
  if (!items.length || !Number.isFinite(availableWidth) || !Number.isFinite(availableHeight) || !(availableWidth > 0) || !(availableHeight > 0)) return [];

  const primaryId = focusedLiveCameraId(items, storedCameraId);
  if (items.length === 1) {
    return [{ camera: items[0], primary: true, ...fitAspect(availableWidth, availableHeight, primaryAspect) }];
  }

  const primary = items.find((camera) => String(camera.id) === primaryId);
  const supports = items.filter((camera) => String(camera.id) !== primaryId);
  const count = supports.length;
  const childAspect = UNIFORM_LIVE_TILE_ASPECT;
  let best = null;
  // Match the primary's height to a stack of child rows. Its width follows
  // its own aspect; remaining children can fill rows beneath the top band.
  // Bound the search for large installations while allowing long single rows.
  const maxRows = Math.min(count + 1, 16);
  const maxColumns = Math.min(count, 16);
  for (let primaryRows = 1; primaryRows <= maxRows; primaryRows += 1) {
    for (let sideColumns = 0; sideColumns <= maxColumns; sideColumns += 1) {
      const sideCount = Math.min(count, primaryRows * sideColumns);
      const belowCount = count - sideCount;
      for (let belowColumns = belowCount ? 1 : 0; belowColumns <= (belowCount ? Math.min(belowCount, maxColumns) : 0); belowColumns += 1) {
        const belowRows = belowCount ? Math.ceil(belowCount / belowColumns) : 0;
        const rows = primaryRows + belowRows;
        const topGapWidth = primaryAspect * (primaryRows - 1) * gutter + sideColumns * gutter;
        const topHeightLimit = (availableWidth - topGapWidth) / (primaryAspect * primaryRows + sideColumns * childAspect);
        const belowHeightLimit = belowCount ? (availableWidth - (belowColumns - 1) * gutter) / (belowColumns * childAspect) : Infinity;
        const cellHeight = Math.min(topHeightLimit, belowHeightLimit, (availableHeight - (rows - 1) * gutter) / rows);
        if (!(cellHeight > 0)) continue;
        const cellWidth = cellHeight * childAspect;
        const primaryHeight = primaryRows * cellHeight + (primaryRows - 1) * gutter;
        const primaryWidth = primaryHeight * primaryAspect;
        const primaryArea = primaryWidth * primaryHeight;
        const childArea = cellWidth * cellHeight;
        // A primary is worth roughly four child panes. Maximize the usable
        // size of both, so neither a tiny primary nor tiny children can win.
        const usableArea = Math.min(childArea, primaryArea / 4);
        const filledArea = primaryArea + count * childArea;
        if (!best || usableArea > best.usableArea + 0.001 || (Math.abs(usableArea - best.usableArea) <= 0.001 && filledArea > best.filledArea)) {
          best = { sideColumns, sideCount, belowColumns, belowCount, cellWidth, cellHeight, primaryWidth, primaryHeight, usableArea, filledArea,
            width: Math.max(primaryWidth + (sideColumns ? sideColumns * (cellWidth + gutter) : 0), belowColumns * (cellWidth + gutter) - (belowCount ? gutter : 0)),
            height: rows * cellHeight + (rows - 1) * gutter };
        }
      }
    }
  }
  if (!best) return [];
  const offsetX = Math.max(0, (availableWidth - best.width) / 2);
  const offsetY = Math.max(0, (availableHeight - best.height) / 2);
  return [
    { camera: primary, primary: true, x: offsetX, y: offsetY, width: best.primaryWidth, height: best.primaryHeight },
    ...supports.map((camera, index) => {
      const beside = index < best.sideCount;
      const localIndex = beside ? index : index - best.sideCount;
      const columns = beside ? best.sideColumns : best.belowColumns;
      return {
        camera, primary: false,
        x: offsetX + (beside ? best.primaryWidth + gutter : 0) + (localIndex % columns) * (best.cellWidth + gutter),
        y: offsetY + (beside ? 0 : best.primaryHeight + gutter) + Math.floor(localIndex / columns) * (best.cellHeight + gutter),
        width: best.cellWidth, height: best.cellHeight,
      };
    }),
  ];
}

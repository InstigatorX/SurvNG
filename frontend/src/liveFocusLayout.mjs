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

function packedCells(columns, supportCount) {
  const occupied = new Set(["0:0", "0:1", "1:0", "1:1"]);
  const cells = [];
  let row = 0;
  while (cells.length < supportCount) {
    for (let column = 0; column < columns && cells.length < supportCount; column += 1) {
      if (!occupied.has(`${row}:${column}`)) cells.push({ column, row });
    }
    row += 1;
  }
  return { cells, rows: Math.max(2, row) };
}

export function focusLiveMosaicLayout(cameras, width, height, storedCameraId, gap = 4, preferredAspect = 16 / 9) {
  const items = [...(cameras || [])];
  const availableWidth = Number(width);
  const availableHeight = Number(height);
  const requestedGap = Number(gap);
  const requestedAspect = Number(preferredAspect);
  const gutter = Number.isFinite(requestedGap) ? Math.max(0, requestedGap) : 0;
  const aspect = Number.isFinite(requestedAspect) && requestedAspect > 0 ? requestedAspect : 16 / 9;
  if (!items.length || !Number.isFinite(availableWidth) || !Number.isFinite(availableHeight) || !(availableWidth > 0) || !(availableHeight > 0)) return [];

  const primaryId = focusedLiveCameraId(items, storedCameraId);
  if (items.length === 1) {
    return [{ camera: items[0], primary: true, ...fitAspect(availableWidth, availableHeight, aspect) }];
  }

  const primary = items.find((camera) => String(camera.id) === primaryId);
  const supports = items.filter((camera) => String(camera.id) !== primaryId);
  let best = null;
  for (let columns = 2; columns <= 6; columns += 1) {
    const packed = packedCells(columns, supports.length);
    const widthPerCell = (availableWidth - gutter * (columns - 1)) / columns;
    const heightPerCell = (availableHeight - gutter * (packed.rows - 1)) / packed.rows;
    if (!(widthPerCell > 0) || !(heightPerCell > 0)) continue;
    const cellWidth = Math.min(widthPerCell, heightPerCell * aspect);
    const cellHeight = cellWidth / aspect;
    const candidate = {
      columns,
      rows: packed.rows,
      cells: packed.cells,
      cellWidth,
      cellHeight,
      primaryWidth: cellWidth * 2 + gutter,
      primaryHeight: cellHeight * 2 + gutter,
    };
    if (!best || candidate.primaryWidth * candidate.primaryHeight > best.primaryWidth * best.primaryHeight) best = candidate;
  }
  if (!best) return [];
  const gridWidth = best.columns * best.cellWidth + gutter * (best.columns - 1);
  const gridHeight = best.rows * best.cellHeight + gutter * (best.rows - 1);
  const offsetX = Math.max(0, (availableWidth - gridWidth) / 2);
  const offsetY = Math.max(0, (availableHeight - gridHeight) / 2);

  return [
    { camera: primary, primary: true, x: offsetX, y: offsetY, width: best.primaryWidth, height: best.primaryHeight },
    ...supports.map((camera, index) => ({
      camera,
      primary: false,
      x: offsetX + best.cells[index].column * (best.cellWidth + gutter),
      y: offsetY + best.cells[index].row * (best.cellHeight + gutter),
      width: best.cellWidth,
      height: best.cellHeight,
    })),
  ];
}

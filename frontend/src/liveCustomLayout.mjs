const COLUMN_COUNT = 12;
const ROW_COUNT = 4;
// Keep this in sync with --live-tile-gap in styles.css. The packer must use
// the rendered grid gap when translating a tile's pixel height into row spans.
const GRID_GAP = 4;
const PACK_ROW_HEIGHT = 2;

function defaultSize(camera, aspectOverrides = {}) {
  const aspect = Number(aspectOverrides[camera?.id]);
  const portrait = Number.isFinite(aspect) ? aspect < 0.9 : false;
  return portrait ? { columns: 3, rows: 2 } : { columns: 3, rows: 1 };
}

function clampInteger(value, minimum, maximum, fallback) {
  const number = Math.round(Number(value));
  return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback;
}

export function readLiveCustomLayout(rawValue, cameras, aspectOverrides = {}) {
  let parsed = {};
  try {
    parsed = JSON.parse(String(rawValue || "{}"));
  } catch {
    parsed = {};
  }
  const ids = (cameras || []).map((camera) => String(camera.id));
  const known = new Set(ids);
  const savedOrder = Array.isArray(parsed.order)
    ? parsed.order.map(String).filter((id, index, values) => known.has(id) && values.indexOf(id) === index)
    : [];
  const order = [...savedOrder, ...ids.filter((id) => !savedOrder.includes(id))];
  const camerasById = new Map((cameras || []).map((camera) => [String(camera.id), camera]));
  const sizes = {};
  for (const id of ids) {
    const fallback = defaultSize(camerasById.get(id), aspectOverrides);
    const saved = parsed.sizes?.[id] || {};
    sizes[id] = {
      columns: clampInteger(saved.columns, 2, COLUMN_COUNT, fallback.columns),
      rows: clampInteger(saved.rows, 1, ROW_COUNT, fallback.rows),
      aspectLocked: saved.aspectLocked === true,
    };
  }
  return { version: 1, order, sizes };
}

export function moveLiveCamera(order, sourceId, targetId, position = "before") {
  const source = String(sourceId || "");
  const target = String(targetId || "");
  if (!source || !target || source === target || !order.includes(source) || !order.includes(target)) {
    return [...order];
  }
  if (position === "swap") {
    const sourceIndex = order.indexOf(source);
    const targetIndex = order.indexOf(target);
    const next = [...order];
    next[sourceIndex] = target;
    next[targetIndex] = source;
    return next;
  }
  const next = order.filter((id) => id !== source);
  const targetIndex = next.indexOf(target);
  next.splice(targetIndex + (position === "after" ? 1 : 0), 0, source);
  return next;
}

export function liveCustomDropTarget(slots, clientX, clientY, sourceId) {
  const x = Number(clientX);
  const y = Number(clientY);
  const source = String(sourceId || "");
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Array.isArray(slots) || !source) return null;
  const normalized = slots
    .map((slot) => ({
      id: String(slot?.id || ""),
      left: Number(slot?.left),
      top: Number(slot?.top),
      width: Number(slot?.width),
      height: Number(slot?.height),
    }))
    .filter((slot) => slot.id && [slot.left, slot.top, slot.width, slot.height].every(Number.isFinite) && slot.width > 0 && slot.height > 0);
  const sourceSlot = normalized.find((slot) => slot.id === source);
  if (sourceSlot && x >= sourceSlot.left && x <= sourceSlot.left + sourceSlot.width
    && y >= sourceSlot.top && y <= sourceSlot.top + sourceSlot.height) {
    return { targetId: source, position: "original" };
  }
  const candidates = normalized.filter((slot) => slot.id !== source);
  if (!candidates.length) return null;
  const containing = candidates.find((slot) => (
    x >= slot.left && x <= slot.left + slot.width && y >= slot.top && y <= slot.top + slot.height
  ));
  if (containing) {
    const edgeX = Math.min(containing.width * 0.18, 36);
    const edgeY = Math.min(containing.height * 0.18, 36);
    const insideCenter = x >= containing.left + edgeX
      && x <= containing.left + containing.width - edgeX
      && y >= containing.top + edgeY
      && y <= containing.top + containing.height - edgeY;
    if (insideCenter) return { targetId: containing.id, position: "swap" };
  }
  const target = containing || candidates.reduce((nearest, slot) => {
    const distance = Math.hypot(x - (slot.left + slot.width / 2), y - (slot.top + slot.height / 2));
    return !nearest || distance < nearest.distance ? { slot, distance } : nearest;
  }, null)?.slot;
  if (!target) return null;
  const verticalIntent = Math.abs(y - (target.top + target.height / 2)) > target.height * 0.2;
  return {
    targetId: target.id,
    position: verticalIntent
      ? (y >= target.top + target.height / 2 ? "after" : "before")
      : (x >= target.left + target.width / 2 ? "after" : "before"),
  };
}

export function resizeLiveCamera(size, columnDelta, rowDelta) {
  return {
    columns: clampInteger(Number(size?.columns) + Number(columnDelta || 0), 2, COLUMN_COUNT, 3),
    rows: clampInteger(Number(size?.rows) + Number(rowDelta || 0), 1, ROW_COUNT, 1),
    aspectLocked: false,
  };
}

function spanPixels(count, unit, gap = GRID_GAP) {
  return count * unit + Math.max(0, count - 1) * gap;
}

export function resizeLiveCameraToAspect(
  size,
  pixelDeltaX,
  pixelDeltaY,
  metrics,
  aspect,
) {
  const ratio = Number(aspect);
  if (!(ratio > 0)) {
    return resizeLiveCamera(
      size,
      Math.round(Number(pixelDeltaX || 0) / Math.max(1, metrics?.columnWidth || 1)),
      Math.round(Number(pixelDeltaY || 0) / Math.max(1, metrics?.rowHeight || 1)),
    );
  }
  const gap = Math.max(0, Number(metrics?.gap) || GRID_GAP);
  const columnWidth = Math.max(1, Number(metrics?.columnWidth) || 1);
  const rowHeight = Math.max(1, Number(metrics?.rowHeight) || 1);
  const desiredWidth = Math.max(
    spanPixels(2, columnWidth, gap),
    spanPixels(Number(size?.columns) || 3, columnWidth, gap) + Number(pixelDeltaX || 0),
  );
  const desiredHeight = Math.max(
    rowHeight,
    spanPixels(Number(size?.rows) || 1, rowHeight, gap) + Number(pixelDeltaY || 0),
  );
  let best = null;
  const maximumHeight = spanPixels(ROW_COUNT, rowHeight, gap);
  for (let columns = 2; columns <= COLUMN_COUNT; columns += 1) {
    const width = spanPixels(columns, columnWidth, gap);
    const height = width / ratio;
    if (height > maximumHeight + 0.5) continue;
    const rows = clampInteger(
      Math.ceil((height + gap) / (rowHeight + gap)),
      1,
      ROW_COUNT,
      Number(size?.rows) || 1,
    );
    const score = Math.hypot(width - desiredWidth, height - desiredHeight);
    if (!best || score < best.score) best = { columns, rows, score };
  }
  return {
    columns: best?.columns || 3,
    rows: best?.rows || 1,
    aspectLocked: true,
  };
}

export function liveCustomGridMetrics(width, height, gap = GRID_GAP) {
  const safeWidth = Math.max(0, Number(width) || 0);
  const safeHeight = Math.max(0, Number(height) || 0);
  return {
    columnWidth: Math.max(1, (safeWidth - gap * (COLUMN_COUNT - 1)) / COLUMN_COUNT),
    rowHeight: Math.max(90, (safeHeight - gap * (ROW_COUNT - 1)) / ROW_COUNT),
    packRowHeight: PACK_ROW_HEIGHT,
    gap,
  };
}

export function liveCustomTilePlacement(size, metrics, aspect) {
  const gap = Math.max(0, Number(metrics?.gap) || GRID_GAP);
  const columnWidth = Math.max(1, Number(metrics?.columnWidth) || 1);
  const logicalRowHeight = Math.max(1, Number(metrics?.rowHeight) || 1);
  const packRowHeight = Math.max(1, Number(metrics?.packRowHeight) || PACK_ROW_HEIGHT);
  const columns = clampInteger(size?.columns, 2, COLUMN_COUNT, 3);
  const rows = clampInteger(size?.rows, 1, ROW_COUNT, 1);
  const measuredAspect = Number(aspect);
  const width = spanPixels(columns, columnWidth, gap);
  const height = size?.aspectLocked === true && measuredAspect > 0
    ? width / measuredAspect
    : spanPixels(rows, logicalRowHeight, gap);
  const packedRows = Math.max(1, Math.ceil((height + gap) / (packRowHeight + gap)));
  return { columns, packedRows, height };
}

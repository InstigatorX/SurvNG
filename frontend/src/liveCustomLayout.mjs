const COLUMN_COUNT = 12;

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
      rows: clampInteger(saved.rows, 1, 4, fallback.rows),
    };
  }
  return { version: 1, order, sizes };
}

export function moveLiveCamera(order, sourceId, targetId) {
  const source = String(sourceId || "");
  const target = String(targetId || "");
  if (!source || !target || source === target || !order.includes(source) || !order.includes(target)) {
    return [...order];
  }
  const next = order.filter((id) => id !== source);
  next.splice(next.indexOf(target), 0, source);
  return next;
}

export function resizeLiveCamera(size, columnDelta, rowDelta) {
  return {
    columns: clampInteger(Number(size?.columns) + Number(columnDelta || 0), 2, COLUMN_COUNT, 3),
    rows: clampInteger(Number(size?.rows) + Number(rowDelta || 0), 1, 4, 1),
  };
}

export function liveCustomGridMetrics(width, height, gap = 8) {
  const safeWidth = Math.max(0, Number(width) || 0);
  const safeHeight = Math.max(0, Number(height) || 0);
  return {
    columnWidth: Math.max(1, (safeWidth - gap * (COLUMN_COUNT - 1)) / COLUMN_COUNT),
    rowHeight: Math.max(90, (safeHeight - gap * 3) / 4),
  };
}

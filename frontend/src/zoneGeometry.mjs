function pointToSegmentDistanceSquared(point, start, end, scale) {
  const px = point.x * scale.x;
  const py = point.y * scale.y;
  const startX = start.x * scale.x;
  const startY = start.y * scale.y;
  const deltaX = (end.x - start.x) * scale.x;
  const deltaY = (end.y - start.y) * scale.y;
  const lengthSquared = (deltaX * deltaX) + (deltaY * deltaY);
  const projection = lengthSquared > 0
    ? Math.max(0, Math.min(1, (((px - startX) * deltaX) + ((py - startY) * deltaY)) / lengthSquared))
    : 0;
  const nearestX = startX + (projection * deltaX);
  const nearestY = startY + (projection * deltaY);
  return ((px - nearestX) ** 2) + ((py - nearestY) ** 2);
}

export function insertZonePointWithIndex(points, point, scale = { x: 1, y: 1 }) {
  const current = Array.isArray(points) ? points : [];
  if (current.length < 3) {
    return { points: [...current, point], insertionIndex: current.length };
  }

  let insertionIndex = current.length;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < current.length; index += 1) {
    const distance = pointToSegmentDistanceSquared(
      point,
      current[index],
      current[(index + 1) % current.length],
      scale,
    );
    if (distance < nearestDistance) {
      nearestDistance = distance;
      insertionIndex = index + 1;
    }
  }

  return {
    points: [
      ...current.slice(0, insertionIndex),
      point,
      ...current.slice(insertionIndex),
    ],
    insertionIndex,
  };
}

export function clampZoneView(view, frame, { min = 1, max = 8 } = {}) {
  const scale = Math.max(min, Math.min(max, Number(view?.scale) || 1));
  if (scale <= 1) return { scale: 1, x: 0, y: 0 };
  const width = Math.max(0, Number(frame?.width) || 0);
  const height = Math.max(0, Number(frame?.height) || 0);
  const xLimit = width * (scale - 1) / 2;
  const yLimit = height * (scale - 1) / 2;
  return {
    scale,
    x: Math.max(-xLimit, Math.min(xLimit, Number(view?.x) || 0)),
    y: Math.max(-yLimit, Math.min(yLimit, Number(view?.y) || 0)),
  };
}

export function zoomZoneView(view, frame, anchor, nextScale) {
  const currentScale = Number(view?.scale) > 0 ? Number(view.scale) : 1;
  const scale = Math.max(1, Math.min(8, Number(nextScale) || 1));
  const ratio = scale / currentScale;
  const anchorX = Number(anchor?.x) || 0;
  const anchorY = Number(anchor?.y) || 0;
  return clampZoneView({
    scale,
    x: anchorX - (anchorX - (Number(view?.x) || 0)) * ratio,
    y: anchorY - (anchorY - (Number(view?.y) || 0)) * ratio,
  }, frame);
}

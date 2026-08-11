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

export function insertZonePoint(points, point, scale = { x: 1, y: 1 }) {
  return insertZonePointWithIndex(points, point, scale).points;
}

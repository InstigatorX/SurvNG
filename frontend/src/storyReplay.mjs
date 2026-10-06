export function storyShotAt(plan, position) {
  return (plan?.shots || []).find((shot) => shot.offset <= position && position < shot.offset + shot.duration) || null;
}

export function storyCropAt(points, at) {
  const wide = { x: .5, y: .5, size: 1 };
  if (!points?.length || at < points[0].at || at > points.at(-1).at) return wide;
  let left = points[0];
  for (const right of points.slice(1)) {
    if (at <= right.at) {
      let u = Math.max(0, Math.min(1, (at - left.at) / Math.max(.001, right.at - left.at)));
      u = u * u * (3 - 2 * u);
      const size = 1 + (left.size + (right.size - left.size) * u - 1) * Math.min(1, Math.max(0, (at - 1) / 2));
      const half = size / 2;
      return { size, x: Math.max(half, Math.min(1 - half, left.x + (right.x - left.x) * u)), y: Math.max(half, Math.min(1 - half, left.y + (right.y - left.y) * u)) };
    }
    left = right;
  }
  return wide;
}

export function nextStoryPosition(plan, shot) {
  return Math.min(plan.duration, shot.offset + shot.duration);
}

export function motionAuditRegions(features) {
  const regions = Array.isArray(features?.motion_regions) ? features.motion_regions : [];
  return regions
    .filter((region) => Array.isArray(region) && region.length >= 4)
    .map((region) => region.slice(0, 4).map(Number))
    .filter(([x1, y1, x2, y2]) => (
      [x1, y1, x2, y2].every(Number.isFinite)
      && x1 >= 0 && y1 >= 0 && x2 <= 1 && y2 <= 1
      && x2 > x1 && y2 > y1
    ));
}

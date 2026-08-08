export function nextFaceReviewObservation(currentId, previous, refreshed) {
  const current = Number(currentId);
  const before = Array.isArray(previous) ? previous : [];
  const after = Array.isArray(refreshed) ? refreshed : [];
  if (!after.length) return null;

  const refreshedIndex = after.findIndex((item) => Number(item?.id) === current);
  if (refreshedIndex >= 0) {
    return after[refreshedIndex + 1] || null;
  }
  const previousIndex = Math.max(
    0,
    before.findIndex((item) => Number(item?.id) === current),
  );
  return after[Math.min(previousIndex, after.length - 1)] || null;
}

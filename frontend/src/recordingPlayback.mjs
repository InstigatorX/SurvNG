export function playbackRowsCoverEpoch(rows, epoch) {
  if (!Number.isFinite(epoch) || !Array.isArray(rows)) return false;
  return rows.some((row) => {
    const start = Number(row?.start_epoch);
    const end = Number(row?.end_epoch);
    return Number.isFinite(start) && Number.isFinite(end) && start <= epoch && epoch < end;
  });
}

export function describePlaybackError(error) {
  const details = [];
  const code = Number(error?.code);
  if (Number.isFinite(code)) details.push(`code ${code}`);
  const category = Number(error?.category);
  if (Number.isFinite(category)) details.push(`category ${category}`);
  if (error?.message) details.push(String(error.message));
  const data = Array.isArray(error?.data)
    ? error.data.filter((item) => ["string", "number"].includes(typeof item)).slice(0, 3)
    : [];
  if (data.length) details.push(data.join(", "));
  return [...new Set(details)].join(" · ") || "unknown media error";
}

export function isUnsupportedPlaybackError(error) {
  const code = Number(error?.code);
  const dataCodes = Array.isArray(error?.data) ? error.data.map(Number).filter(Number.isFinite) : [];
  const description = describePlaybackError(error).toLowerCase();
  return code === 4
    || dataCodes.includes(4)
    || /(?:codec|decode|format|media source).*(?:unsupported|not supported)/.test(description)
    || /(?:unsupported|not supported).*(?:codec|decode|format|media source)/.test(description);
}

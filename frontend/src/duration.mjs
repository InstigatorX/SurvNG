const MINUTE = 60;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
const WEEK = 7 * DAY;
const MONTH = 30 * DAY;

export function formatServerUptime(seconds) {
  if (!Number.isFinite(seconds) || seconds < MINUTE) return "Less than 1 min";
  let remaining = Math.floor(seconds);
  const units = [
    ["month", MONTH],
    ["week", WEEK],
    ["day", DAY],
    ["hour", HOUR],
    ["min", MINUTE],
  ];
  const parts = [];
  for (const [label, size] of units) {
    const value = Math.floor(remaining / size);
    remaining %= size;
    if (!value) continue;
    parts.push(`${value} ${label}${value === 1 ? "" : "s"}`);
  }
  return parts.join(", ") || "Less than 1 min";
}

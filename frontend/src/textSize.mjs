// Per-browser text size. Font sizes are rem-based (see the type scale on
// :root), so scaling the root font size scales all text. "default" leaves
// the root alone so the browser's own text-size setting still applies.
// The boot scripts in index.html, config.html and recordings.html apply the
// same mapping before React loads; keep them in sync with TEXT_SIZE_META.

export const TEXT_SIZES = Object.freeze(["default", "large", "larger"]);

export const TEXT_SIZE_META = Object.freeze({
  default: Object.freeze({ label: "Default", rootFontSize: "" }),
  large: Object.freeze({ label: "Large (+12%)", rootFontSize: "112.5%" }),
  larger: Object.freeze({ label: "Larger (+25%)", rootFontSize: "125%" }),
});

export const TEXT_SIZE_STORAGE_KEY = "survng.textSize";

export function normalizeTextSize(value) {
  return TEXT_SIZES.includes(value) ? value : "default";
}

/** Apply a text size to the document root and return the normalized value. */
export function applyTextSize(value, { documentRef = document } = {}) {
  const size = normalizeTextSize(value);
  const root = documentRef.documentElement;
  const rootFontSize = TEXT_SIZE_META[size].rootFontSize;
  if (rootFontSize) root.style.fontSize = rootFontSize;
  else root.style.removeProperty("font-size");
  root.dataset.textSize = size;
  return size;
}

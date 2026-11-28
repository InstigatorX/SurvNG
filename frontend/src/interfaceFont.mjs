// Per-browser interface font. The stylesheet maps each id to a font stack
// through the data-font attribute on <html>; "inter" is the default and sets
// no attribute. Every font is bundled, so none is fetched from another site,
// and the browser downloads only the one in use. The boot scripts in
// index.html, config.html and recordings.html apply the same ids before React
// loads; keep them in sync with INTERFACE_FONTS. The standalone ONVIF inspector
// has its own stylesheet and keeps its own font.

export const INTERFACE_FONTS = Object.freeze(["inter", "figtree", "manrope", "dm-sans", "source-sans", "system"]);

export const INTERFACE_FONT_META = Object.freeze({
  inter: Object.freeze({ label: "Inter (default)" }),
  figtree: Object.freeze({ label: "Figtree" }),
  manrope: Object.freeze({ label: "Manrope" }),
  "dm-sans": Object.freeze({ label: "DM Sans" }),
  "source-sans": Object.freeze({ label: "Source Sans" }),
  system: Object.freeze({ label: "System font" }),
});

export const INTERFACE_FONT_STORAGE_KEY = "survng.font";

export function normalizeInterfaceFont(value) {
  return INTERFACE_FONTS.includes(value) ? value : "inter";
}

/** Apply an interface font to the document root and return the normalized value. */
export function applyInterfaceFont(value, { documentRef = document } = {}) {
  const font = normalizeInterfaceFont(value);
  const root = documentRef.documentElement;
  if (font === "inter") delete root.dataset.font;
  else root.dataset.font = font;
  return font;
}

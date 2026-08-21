/** PWA chrome that follows SurvNG light / dark / auto without drawing under the status bar. */

export const BROWSER_APPEARANCE_THEME_KEY = "survng.theme";
export const BROWSER_APPEARANCE_THEMES = ["auto", "light", "dark"];

export const BROWSER_APPEARANCE_COLORS = Object.freeze({
  light: "#edf1f3",
  dark: "#071015",
});

export function normalizeBrowserAppearanceTheme(theme, fallback = "dark") {
  return BROWSER_APPEARANCE_THEMES.includes(theme) ? theme : fallback;
}

export function resolveBrowserAppearance(theme, prefersDark = false) {
  const normalized = normalizeBrowserAppearanceTheme(theme);
  if (normalized === "light") return "light";
  if (normalized === "dark") return "dark";
  return prefersDark ? "dark" : "light";
}

export function browserAppearanceChrome(appearance) {
  const resolved = appearance === "light" ? "light" : "dark";
  return {
    appearance: resolved,
    themeColor: BROWSER_APPEARANCE_COLORS[resolved],
    // Opaque Apple status bars: light → default, dark → black (not translucent).
    statusBarStyle: resolved === "light" ? "default" : "black",
    colorScheme: resolved,
  };
}

function ensureMeta(documentRef, name) {
  let meta = documentRef.querySelector(`meta[name="${name}"]`);
  if (!meta) {
    meta = documentRef.createElement("meta");
    meta.setAttribute("name", name);
    documentRef.head.appendChild(meta);
  }
  return meta;
}

export function applyBrowserAppearance(theme, {
  documentRef = document,
  matchMedia = window.matchMedia?.bind(window),
} = {}) {
  const prefersDark = Boolean(matchMedia?.("(prefers-color-scheme: dark)")?.matches);
  const chrome = browserAppearanceChrome(resolveBrowserAppearance(theme, prefersDark));
  const root = documentRef.documentElement;
  root.dataset.theme = normalizeBrowserAppearanceTheme(theme);
  ensureMeta(documentRef, "theme-color").setAttribute("content", chrome.themeColor);
  ensureMeta(documentRef, "apple-mobile-web-app-status-bar-style").setAttribute("content", chrome.statusBarStyle);
  ensureMeta(documentRef, "color-scheme").setAttribute("content", chrome.colorScheme);
  return chrome;
}

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const frontendRoot = join(dirname(fileURLToPath(import.meta.url)), "..");

for (const entry of ["index.html", "recordings.html", "config.html"]) {
  const html = readFileSync(join(frontendRoot, entry), "utf8");
  assert.ok(html.includes('localStorage.getItem("survng.theme")'), `${entry} reads the saved theme before startup`);
  assert.ok(html.includes('let theme = "dark";'), `${entry} defaults to dark before React loads`);
  assert.ok(html.includes("root.dataset.theme = theme;"), `${entry} applies the theme on the document root`);
  assert.ok(
    html.includes('if (themes.includes(stored)) theme = stored;'),
    `${entry} accepts light, dark, and auto from storage`,
  );
  const bootStart = html.indexOf('localStorage.getItem("survng.theme")');
  const moduleStart = html.indexOf('<script type="module"');
  assert.ok(bootStart !== -1 && moduleStart !== -1 && bootStart < moduleStart, `${entry} applies the theme before the application module`);
}

// The typefaces live in their own stylesheet, shared with the ONVIF inspector.
const styles = readFileSync(join(frontendRoot, "src", "fonts.css"), "utf8");
assert.ok(styles.includes("@fontsource-variable/inter/files/inter-latin-wght-normal.woff2"), "Inter Latin is bundled by the stylesheet");
assert.ok(styles.includes('font-family: "Inter Variable"'), "the bundled variable font is the primary family");
assert.match(styles, /:root \{[^}]*--font-ui: "Inter Variable"/, "and the default font stack starts with it");
assert.ok(readFileSync(join(frontendRoot, "src", "App.jsx"), "utf8").includes('import "./fonts.css";'), "the app loads the typefaces");

console.log("theme bootstrap tests passed");

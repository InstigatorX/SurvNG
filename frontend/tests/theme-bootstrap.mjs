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

const styles = readFileSync(join(frontendRoot, "src", "styles.css"), "utf8");
assert.ok(styles.includes("@fontsource-variable/inter/files/inter-latin-wght-normal.woff2"), "Inter Latin is bundled by the stylesheet");
assert.ok(styles.includes('font-family: "Inter Variable"'), "the bundled variable font is the primary family");

console.log("theme bootstrap tests passed");

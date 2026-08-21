import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { registerSurvngServiceWorker } from "../src/registerServiceWorker.mjs";

const directory = dirname(fileURLToPath(import.meta.url));
const frontendRoot = join(directory, "..");

for (const entry of ["index.html", "recordings.html", "config.html"]) {
  const html = readFileSync(join(frontendRoot, entry), "utf8");
  assert.match(html, /rel="manifest"/, `${entry} links the web app manifest`);
  assert.match(html, /apple-mobile-web-app-capable/, `${entry} enables iOS standalone install`);
  assert.match(html, /viewport-fit=cover/, `${entry} preserves safe-area on install`);
  assert.match(html, /mobile-web-app-capable/, `${entry} marks the app as installable`);
  assert.match(html, /apple-mobile-web-app-status-bar-style" content="black"/, `${entry} uses opaque status bar by default`);
  assert.doesNotMatch(html, /black-translucent/, `${entry} avoids Dynamic Island underlap`);
}

for (const icon of [
  "pwa-icon-192.png",
  "pwa-icon-512.png",
  "pwa-icon-maskable-192.png",
  "pwa-icon-maskable-512.png",
]) {
  assert.equal(existsSync(join(frontendRoot, "public", icon)), true, `${icon} must ship in public/`);
}

assert.equal(typeof registerSurvngServiceWorker, "function");

const appSource = readFileSync(join(frontendRoot, "src/App.jsx"), "utf8");
assert.match(appSource, /registerSurvngServiceWorker/);

console.log("pwa shell tests passed");

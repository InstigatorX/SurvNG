import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  applyBrowserAppearance,
  browserAppearanceChrome,
  resolveBrowserAppearance,
} from "../src/browserAppearance.mjs";

assert.equal(resolveBrowserAppearance("light", true), "light");
assert.equal(resolveBrowserAppearance("dark", false), "dark");
assert.equal(resolveBrowserAppearance("auto", true), "dark");
assert.equal(resolveBrowserAppearance("auto", false), "light");
assert.equal(resolveBrowserAppearance("nope", true), "dark");

assert.deepEqual(browserAppearanceChrome("light"), {
  appearance: "light",
  themeColor: "#edf1f3",
  statusBarStyle: "default",
  colorScheme: "light",
});
assert.deepEqual(browserAppearanceChrome("dark"), {
  appearance: "dark",
  themeColor: "#071015",
  statusBarStyle: "black",
  colorScheme: "dark",
});

const metas = new Map();
const documentRef = {
  documentElement: { dataset: {} },
  head: {
    appendChild(node) {
      metas.set(node.getAttribute("name"), node);
    },
  },
  querySelector(selector) {
    const match = /^meta\[name="([^"]+)"\]$/.exec(selector);
    return match ? metas.get(match[1]) || null : null;
  },
  createElement() {
    const attrs = new Map();
    return {
      setAttribute(name, value) {
        attrs.set(name, value);
      },
      getAttribute(name) {
        return attrs.get(name);
      },
    };
  },
};

const darkChrome = applyBrowserAppearance("dark", {
  documentRef,
  matchMedia: () => ({ matches: false }),
});
assert.equal(darkChrome.statusBarStyle, "black");
assert.equal(documentRef.documentElement.dataset.theme, "dark");
assert.equal(metas.get("theme-color").getAttribute("content"), "#071015");
assert.equal(metas.get("apple-mobile-web-app-status-bar-style").getAttribute("content"), "black");

const lightChrome = applyBrowserAppearance("auto", {
  documentRef,
  matchMedia: () => ({ matches: false }),
});
assert.equal(lightChrome.statusBarStyle, "default");
assert.equal(documentRef.documentElement.dataset.theme, "auto");
assert.equal(metas.get("theme-color").getAttribute("content"), "#edf1f3");
assert.equal(metas.get("apple-mobile-web-app-status-bar-style").getAttribute("content"), "default");

const directory = dirname(fileURLToPath(import.meta.url));
const frontendRoot = join(directory, "..");
for (const entry of ["index.html", "recordings.html", "config.html"]) {
  const html = readFileSync(join(frontendRoot, entry), "utf8");
  assert.match(html, /apple-mobile-web-app-status-bar-style" content="black"/, `${entry} defaults to opaque dark status bar`);
  assert.doesNotMatch(html, /black-translucent/, `${entry} must not use translucent status bar`);
  assert.match(html, /prefers-color-scheme: dark/, `${entry} resolves auto appearance before paint`);
  assert.match(html, /statusBar\.setAttribute\("content", appearance === "light" \? "default" : "black"\)/, `${entry} maps light/dark to opaque Apple styles`);
}

const appSource = readFileSync(join(frontendRoot, "src/App.jsx"), "utf8");
assert.match(appSource, /applyBrowserAppearance/);

console.log("browser appearance tests passed");

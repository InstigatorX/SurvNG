import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { applyTextSize, normalizeTextSize, TEXT_SIZE_META, TEXT_SIZE_STORAGE_KEY, TEXT_SIZES } from "../src/textSize.mjs";

function fakeDocument() {
  const properties = new Map();
  const root = {
    dataset: {},
    style: {
      set fontSize(value) { properties.set("font-size", value); },
      get fontSize() { return properties.get("font-size") || ""; },
      removeProperty(name) { properties.delete(name); },
    },
  };
  return { documentElement: root };
}

assert.deepEqual(TEXT_SIZES, ["default", "large", "larger"]);
assert.equal(normalizeTextSize("larger"), "larger");
assert.equal(normalizeTextSize("huge"), "default");
assert.equal(normalizeTextSize(null), "default");

const documentRef = fakeDocument();
assert.equal(applyTextSize("large", { documentRef }), "large");
assert.equal(documentRef.documentElement.style.fontSize, "112.5%");
assert.equal(documentRef.documentElement.dataset.textSize, "large");
assert.equal(applyTextSize("larger", { documentRef }), "larger");
assert.equal(documentRef.documentElement.style.fontSize, "125%");
// Default clears the override so the browser's own text size applies.
assert.equal(applyTextSize("default", { documentRef }), "default");
assert.equal(documentRef.documentElement.style.fontSize, "");
assert.equal(applyTextSize("bogus", { documentRef }), "default");

// The pre-React boot scripts must use the same key and the same root sizes.
for (const page of ["index.html", "config.html", "recordings.html"]) {
  const html = readFileSync(new URL(`../${page}`, import.meta.url), "utf8");
  assert.ok(html.includes(`localStorage.getItem("${TEXT_SIZE_STORAGE_KEY}")`), `${page} reads the text size`);
  for (const size of TEXT_SIZES.filter((item) => TEXT_SIZE_META[item].rootFontSize)) {
    assert.ok(html.includes(`${size}: "${TEXT_SIZE_META[size].rootFontSize}"`), `${page} maps ${size}`);
  }
}

// Text sizes only reach every view if font sizes are rem-based tokens.
const sourceRoot = new URL("../src/", import.meta.url);
const cssFiles = readdirSync(sourceRoot, { recursive: true }).filter((file) => String(file).endsWith(".css"));
assert.ok(cssFiles.length > 10);
for (const file of cssFiles) {
  const css = readFileSync(new URL(String(file), sourceRoot), "utf8");
  assert.equal(/font-size:\s*[0-9.]*[1-9][0-9.]*px/.test(css), false, `${file}: font sizes use the rem type scale`);
  assert.equal(/(^|[^-])font:[^;]*[0-9]px/.test(css), false, `${file}: font shorthands use the rem type scale`);
}
assert.match(readFileSync(new URL("typeScale.css", sourceRoot), "utf8"), /--text-base:\s*0\.9375rem/);
// Every entry point loads the type scale, or its var(--text-*) sizes are undefined.
for (const entry of ["App.jsx", "onvif/main.jsx"]) {
  assert.match(readFileSync(new URL(entry, sourceRoot), "utf8"), /typeScale\.css/, `${entry} imports the type scale`);
}

console.log("text size tests passed");

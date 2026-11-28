import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { applyInterfaceFont, INTERFACE_FONT_META, INTERFACE_FONT_STORAGE_KEY, INTERFACE_FONTS, normalizeInterfaceFont } from "../src/interfaceFont.mjs";

assert.deepEqual(INTERFACE_FONTS, ["inter", "figtree", "manrope", "dm-sans", "source-sans", "system"]);
assert.ok(INTERFACE_FONTS.length <= 6, "a short list, not a font picker");
for (const font of INTERFACE_FONTS) assert.ok(INTERFACE_FONT_META[font]?.label, `${font} has a label`);
assert.equal(normalizeInterfaceFont("manrope"), "manrope");
assert.equal(normalizeInterfaceFont("comic-sans"), "inter");
assert.equal(normalizeInterfaceFont(null), "inter");

const documentRef = { documentElement: { dataset: {} } };
assert.equal(applyInterfaceFont("figtree", { documentRef }), "figtree");
assert.equal(documentRef.documentElement.dataset.font, "figtree");
assert.equal(applyInterfaceFont("system", { documentRef }), "system");
assert.equal(documentRef.documentElement.dataset.font, "system");
// The default sets nothing, so the stylesheet's own default applies.
assert.equal(applyInterfaceFont("inter", { documentRef }), "inter");
assert.equal("font" in documentRef.documentElement.dataset, false);
assert.equal(applyInterfaceFont("bogus", { documentRef }), "inter");
assert.equal("font" in documentRef.documentElement.dataset, false);

// The pre-React boot scripts must use the same key and the same ids.
const optional = INTERFACE_FONTS.filter((font) => font !== "inter");
for (const page of ["index.html", "config.html", "recordings.html"]) {
  const html = readFileSync(new URL(`../${page}`, import.meta.url), "utf8");
  assert.ok(html.includes(`localStorage.getItem("${INTERFACE_FONT_STORAGE_KEY}")`), `${page} reads the font`);
  assert.ok(html.includes(`const fonts = ${JSON.stringify(optional).replaceAll(",", ", ")};`), `${page} lists the same fonts`);
}

// The stylesheet gives every optional font a stack, and every bundled face a file that exists.
const css = readFileSync(new URL("../src/fonts.css", import.meta.url), "utf8");
const appCss = readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
for (const font of optional) assert.match(css, new RegExp(`:root\\[data-font="${font}"\\] \\{\\s+--font-ui:`), `${font} has a font stack`);
assert.match(appCss, /font-family: var\(--font-ui\);/);
// The standalone ONVIF inspector loads the same faces and follows the same choice.
assert.match(readFileSync(new URL("../src/onvif/onvif.css", import.meta.url), "utf8"), /font-family: var\(--font-ui,/);
const inspector = readFileSync(new URL("../src/onvif/main.jsx", import.meta.url), "utf8");
assert.ok(inspector.includes('import "../fonts.css";') && inspector.includes("applyInterfaceFont("));
const faces = [...css.matchAll(/url\("(@fontsource-variable\/[^"]+)"\)/g)].map((match) => match[1]);
assert.equal(faces.length, 5, "Inter and four optional faces are bundled; the system font needs none");
for (const face of faces) assert.ok(existsSync(new URL(`../node_modules/${face}`, import.meta.url)), `${face} is installed`);
const packages = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).dependencies;
for (const face of faces) assert.ok(packages[face.split("/").slice(0, 2).join("/")], `${face} is a declared dependency`);

console.log("interface font tests passed");

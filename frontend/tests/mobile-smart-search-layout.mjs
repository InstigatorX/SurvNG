import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const searchCss = readFileSync(join(frontendRoot, "src/search/search.css"), "utf8");
const shellCss = readFileSync(join(frontendRoot, "src/shell/shell.css"), "utf8");
const mobileCss = readFileSync(join(frontendRoot, "src/shell/mobile.css"), "utf8");

assert.match(searchCss, /\.search-page:not\(:has\(\.search-history-panel\)\)/);
assert.match(searchCss, /grid-template-rows:\s*auto minmax\(0,\s*1fr\)/);
assert.match(searchCss, /align-content:\s*start/);
assert.match(shellCss, /\.search-page:not\(:has\(\.search-history-panel\)\)\s+\.semantic-search-workspace/);
assert.match(mobileCss, /grid-row:\s*2\s*!important/);

console.log("mobile smart search layout tests passed");

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const directory = dirname(fileURLToPath(import.meta.url));
const appSource = readFileSync(join(directory, "../src/App.jsx"), "utf8");

assert.match(appSource, /lazy,\s*Suspense/);
assert.match(appSource, /<Suspense fallback=\{<WorkspaceFallback \/>\}>/);

for (const specifier of [
  "./live/LivePage.jsx",
  "./incidents/IncidentsPage.jsx",
  "./timeline/TimelinePages.jsx",
  "./admin/ConfigPage.jsx",
  "./people/FacesPage.jsx",
]) {
  assert.doesNotMatch(
    appSource,
    new RegExp(`^import .* from "${specifier.replaceAll(".", "\\.")}"`, "m"),
    `${specifier} must not be a static App.jsx import`,
  );
  assert.match(appSource, new RegExp(`import\\("${specifier.replaceAll(".", "\\.")}"\\)`));
}

assert.match(appSource, /import \{ Shell \} from "\.\/shell\/Shell\.jsx"/);
assert.match(appSource, /import \{ AssistantPanel \} from "\.\/assistant\/AssistantPanel\.jsx"/);

console.log("workspace lazy-load contract tests passed");

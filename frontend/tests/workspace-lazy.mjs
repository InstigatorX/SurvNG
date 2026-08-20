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

for (const stylesheet of [
  "./timeline/timeline.css",
  "./live/live.css",
  "./timeline/investigation.css",
  "./incidents/incidents.css",
  "./search/search.css",
  "./people/people.css",
  "./admin/admin.css",
  "./admin/workspace.css",
]) {
  assert.doesNotMatch(
    appSource,
    new RegExp(`import "${stylesheet.replaceAll(".", "\\.")}"`),
    `${stylesheet} must load with its workspace chunk, not App.jsx`,
  );
}

assert.match(appSource, /import "\.\/styles\.css"/);
assert.match(
  readFileSync(join(directory, "../src/shell/Shell.jsx"), "utf8"),
  /import "\.\/shell\.css"/,
);
assert.match(
  readFileSync(join(directory, "../src/live/LivePage.jsx"), "utf8"),
  /import "\.\/live\.css"/,
);
assert.match(
  readFileSync(join(directory, "../src/incidents/IncidentsPage.jsx"), "utf8"),
  /import "\.\/incidents\.css"/,
);
assert.match(
  readFileSync(join(directory, "../src/timeline/TimelinePages.jsx"), "utf8"),
  /import "\.\/timeline\.css"/,
);
assert.match(
  readFileSync(join(directory, "../src/people/FacesPage.jsx"), "utf8"),
  /import "\.\/people\.css"/,
);
assert.match(
  readFileSync(join(directory, "../src/admin/ConfigPage.jsx"), "utf8"),
  /import "\.\/admin\.css"/,
);

console.log("workspace lazy-load contract tests passed");

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const directory = dirname(fileURLToPath(import.meta.url));
const appSource = [
  readFileSync(join(directory, "../src/live/LivePage.jsx"), "utf8"),
  readFileSync(join(directory, "../src/shared/evidence.jsx"), "utf8"),
  readFileSync(join(directory, "../src/incidents/IncidentsPage.jsx"), "utf8"),
  readFileSync(join(directory, "../src/timeline/TimelinePages.jsx"), "utf8"),
].join("\n");
const styles = [
  readFileSync(join(directory, "../src/styles.css"), "utf8"),
  readFileSync(join(directory, "../src/admin/admin.css"), "utf8"),
  readFileSync(join(directory, "../src/live/live.css"), "utf8"),
  readFileSync(join(directory, "../src/incidents/incidents.css"), "utf8"),
].join("\n");

for (const control of [
  "camera-open-target",
  "incident-card-open",
  "incident-preview-media-action",
  "recording-grid-focus-hit",
]) {
  const declaration = new RegExp(`className=["'][^"']*${control}[^"']*media-surface-action[^"']*["']`);
  assert.match(appSource, declaration, `${control} must use the shared transparent media-surface contract`);
}

assert.match(styles, /\.media-surface-action:hover:not\(:disabled\)[\s\S]*?background-color:\s*transparent;/);
assert.match(styles, /\.media-surface-action:focus-visible[\s\S]*?background-color:\s*transparent;/);
assert.match(appSource, /className="live-activity-select" onClick=\{\(\) => onSelect\(incident\)\}/);
assert.doesNotMatch(appSource, /showOpenAction|live-activity-actions/);
assert.match(appSource, /className="nav-button tile-control-button"[\s\S]*?liveActivityIncidentHref\(viewerEvent\)[\s\S]*?<Siren size=\{16\} \/> Incident/);

console.log("media surface interaction tests passed");

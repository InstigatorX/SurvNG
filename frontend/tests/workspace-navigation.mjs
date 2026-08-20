import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  canonicalWorkspacePath,
  canonicalWorkspaceUrl,
  DESKTOP_PRIMARY_WORKSPACES,
  MOBILE_PRIMARY_WORKSPACES,
  resolveWorkspace,
  systemHealthState,
  timelineHref,
  workspaceDefinition,
  workspaceHref,
} from "../src/workspaceNavigation.mjs";

assert.equal(resolveWorkspace("/").id, "live");
assert.equal(resolveWorkspace("/incidents").id, "incidents");
assert.equal(resolveWorkspace("/timeline").id, "timeline");
assert.equal(resolveWorkspace("/recordings/exports").id, "timeline");
assert.equal(resolveWorkspace("/recordings/search").id, "search");
assert.equal(resolveWorkspace("/search").id, "search");
assert.equal(resolveWorkspace("/faces").id, "people");
assert.equal(resolveWorkspace("/config").id, "admin");
assert.equal(resolveWorkspace("/unknown"), null);
assert.equal(resolveWorkspace("/live/unknown"), null);
assert.equal(resolveWorkspace("/people/unknown"), null);
assert.equal(resolveWorkspace("/timeline/unknown"), null);
assert.equal(resolveWorkspace("/recordings/unknown"), null);

assert.equal(canonicalWorkspacePath("/recordings"), "/timeline");
assert.equal(canonicalWorkspacePath("/recordings/exports"), "/timeline/exports");
assert.equal(canonicalWorkspacePath("/recordings/search"), "/search");
assert.equal(canonicalWorkspacePath("/faces"), "/people");
assert.equal(canonicalWorkspacePath("/config"), "/admin");
assert.equal(canonicalWorkspacePath("/live"), "/");
assert.equal(canonicalWorkspacePath("/live/unknown"), "/live/unknown");
assert.equal(canonicalWorkspacePath("/people/unknown"), "/people/unknown");
assert.equal(canonicalWorkspacePath("/timeline/unknown"), "/timeline/unknown");
assert.equal(canonicalWorkspacePath("/recordings/unknown"), "/recordings/unknown");
assert.equal(canonicalWorkspacePath("/incidents"), "/incidents");
assert.equal(
  canonicalWorkspaceUrl("/recordings", "?camera=gate&at=123.5", "#player"),
  "/timeline?camera=gate&at=123.5#player",
);
assert.equal(canonicalWorkspaceUrl("/faces", "status=unknown"), "/people?status=unknown");

assert.equal(workspaceDefinition("timeline").label, "Timeline");
assert.equal(workspaceDefinition("unknown"), null);
assert.throws(() => workspaceHref("unknown"), /Unknown SurvNG workspace/);
assert.equal(workspaceHref("incidents", { event_ids: "42,43" }), "/incidents?event_ids=42%2C43");
assert.equal(
  timelineHref({ cameraId: "front-door", epoch: 123.5, source: "main" }),
  "/timeline?camera=front-door&at=123.5&source=main",
);
assert.equal(timelineHref({ epoch: Number.NaN }), "/timeline");

assert.deepEqual(DESKTOP_PRIMARY_WORKSPACES, ["live", "incidents", "timeline", "search", "people"]);
assert.deepEqual(MOBILE_PRIMARY_WORKSPACES, ["live", "incidents", "timeline", "search", "more"]);

assert.deepEqual(
  systemHealthState({ lifecycle: "running", storage: { available: true }, detector: { enabled: true, loaded_backend: "openvino" }, cameras: { enabled: 2, online: 2, recording_expected: 2, recording: 2 } }),
  { healthy: true, severity: "healthy", label: "Healthy" },
);
assert.deepEqual(
  systemHealthState({ lifecycle: "starting", storage: { available: true }, detector: { enabled: false } }),
  { healthy: false, severity: "starting", label: "starting" },
);
assert.deepEqual(
  systemHealthState({ lifecycle: "running", storage: { available: true }, detector: { enabled: true, loaded_backend: "openvino" }, cameras: { enabled: 2, online: 1, recording_expected: 2, recording: 1 } }),
  { healthy: false, severity: "attention", label: "Needs attention" },
);
assert.deepEqual(
  systemHealthState({ lifecycle: "running", storage: { available: false }, detector: { enabled: false }, cameras: { enabled: 0, online: 0, recording_expected: 0, recording: 0 } }),
  { healthy: false, severity: "attention", label: "Needs attention" },
);

const stylesSource = [
  readFileSync(new URL("../src/styles.css", import.meta.url), "utf8"),
  readFileSync(new URL("../src/shell/shell.css", import.meta.url), "utf8"),
].join("\n");
assert.equal(stylesSource.includes("grid-template-columns: 176px minmax(0, 1fr)"), false);
assert.match(stylesSource, /\.app-shell\.workspace-rail-collapsed\s*\{\s*--workspace-rail-width:\s*68px;/);

console.log("workspace navigation contract tests passed");

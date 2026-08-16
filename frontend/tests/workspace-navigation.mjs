import assert from "node:assert/strict";
import {
  canonicalWorkspacePath,
  canonicalWorkspaceUrl,
  DESKTOP_PRIMARY_WORKSPACES,
  MOBILE_PRIMARY_WORKSPACES,
  resolveWorkspace,
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

assert.equal(canonicalWorkspacePath("/recordings"), "/timeline");
assert.equal(canonicalWorkspacePath("/recordings/exports"), "/timeline/exports");
assert.equal(canonicalWorkspacePath("/recordings/search"), "/search");
assert.equal(canonicalWorkspacePath("/faces"), "/people");
assert.equal(canonicalWorkspacePath("/config"), "/admin");
assert.equal(canonicalWorkspacePath("/live"), "/");
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

console.log("workspace navigation contract tests passed");

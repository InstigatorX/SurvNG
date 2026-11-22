import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  canonicalWorkspacePath,
  canonicalWorkspaceUrl,
  compactDesktopNavigation,
  DESKTOP_COMPACT_WORKSPACES,
  DESKTOP_PRIMARY_WORKSPACES,
  MOBILE_PRIMARY_WORKSPACES,
  resolveWorkspace,
  timelineHref,
  workspaceDefinition,
  workspaceHref,
} from "../src/workspaceNavigation.mjs";

assert.equal(resolveWorkspace("/").id, "live");
assert.equal(resolveWorkspace("/property").id, "property");
assert.equal(resolveWorkspace("/incidents").id, "incidents");
assert.equal(resolveWorkspace("/timeline").id, "timeline");
assert.equal(resolveWorkspace("/exports").id, "exports");
assert.equal(resolveWorkspace("/recordings/exports").id, "exports");
assert.equal(resolveWorkspace("/timeline/exports").id, "exports");
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
assert.equal(canonicalWorkspacePath("/recordings/exports"), "/exports");
assert.equal(canonicalWorkspacePath("/timeline/exports"), "/exports");
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

assert.equal(workspaceDefinition("property").label, "Property");
assert.equal(workspaceHref("property"), "/property");
assert.equal(workspaceDefinition("timeline").label, "Timeline");
assert.equal(workspaceDefinition("exports").label, "Exports");
assert.equal(workspaceDefinition("unknown"), null);
assert.throws(() => workspaceHref("unknown"), /Unknown SurvNG workspace/);
assert.equal(workspaceHref("incidents", { event_ids: "42,43" }), "/incidents?event_ids=42%2C43");
assert.equal(
  timelineHref({ cameraId: "front-door", epoch: 123.5, source: "main" }),
  "/timeline?camera=front-door&at=123.5&source=main",
);
assert.equal(
  timelineHref({ cameraId: "gate", epoch: 50, eventId: 42 }),
  "/timeline?camera=gate&at=50&event=42",
);
assert.equal(
  timelineHref({ cameraId: "gate", epoch: 50, eventId: 84, trailEventIds: [12, 84] }),
  "/timeline?camera=gate&at=50&event=84&trail=12%2C84",
);
assert.equal(timelineHref({ epoch: Number.NaN }), "/timeline");

assert.deepEqual(DESKTOP_PRIMARY_WORKSPACES, ["live", "property", "incidents", "timeline", "exports", "search", "people"]);
assert.deepEqual(MOBILE_PRIMARY_WORKSPACES, ["live", "incidents", "timeline", "search", "more"]);

const stylesSource = [
  readFileSync(new URL("../src/styles.css", import.meta.url), "utf8"),
  readFileSync(new URL("../src/shell/shell.css", import.meta.url), "utf8"),
].join("\n");
assert.equal(stylesSource.includes("grid-template-columns: 176px minmax(0, 1fr)"), false);
// Desktop navigation lives in the top bar; the left workspace rail is gone.
assert.equal(/\.workspace-sidebar\b/.test(stylesSource), false);
assert.equal(stylesSource.includes("--workspace-rail-width"), false);

assert.deepEqual(DESKTOP_COMPACT_WORKSPACES, ["live", "incidents", "timeline", "search", "admin"]);
const desktopIds = [...DESKTOP_PRIMARY_WORKSPACES, "admin"];
assert.deepEqual(compactDesktopNavigation(desktopIds, "live"), {
  inline: ["live", "incidents", "timeline", "search", "admin"],
  overflow: ["property", "exports", "people"],
});
// The active workspace stays inline in its usual position even when it normally sits under More.
assert.deepEqual(compactDesktopNavigation(desktopIds, "people"), {
  inline: ["live", "incidents", "timeline", "search", "people", "admin"],
  overflow: ["property", "exports"],
});
// Viewers have no Admin link; nothing is invented.
assert.deepEqual(compactDesktopNavigation(DESKTOP_PRIMARY_WORKSPACES, "exports").inline, ["live", "incidents", "timeline", "exports", "search"]);

console.log("workspace navigation contract tests passed");

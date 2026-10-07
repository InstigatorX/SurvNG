import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { parseTimelineView } from "../src/timelineWorkspace.mjs";
import {
  canonicalWorkspacePath,
  canonicalWorkspaceUrl,
  DESKTOP_PRIMARY_WORKSPACES,
  MOBILE_PRIMARY_WORKSPACES,
  resolveWorkspace,
  reviewMode,
  timelineHref,
  workspaceDefinition,
  workspaceHref,
} from "../src/workspaceNavigation.mjs";

assert.equal(resolveWorkspace("/").id, "live");
assert.equal(resolveWorkspace("/incidents").id, "review");
assert.equal(resolveWorkspace("/timeline").id, "review");
assert.equal(resolveWorkspace("/review").id, "review");
assert.equal(resolveWorkspace("/exports").id, "exports");
assert.equal(resolveWorkspace("/recordings/exports").id, "exports");
assert.equal(resolveWorkspace("/timeline/exports").id, "exports");
assert.equal(resolveWorkspace("/recordings/search").id, "review");
assert.equal(resolveWorkspace("/search").id, "review");
assert.equal(resolveWorkspace("/faces").id, "people");
assert.equal(resolveWorkspace("/config").id, "admin");
assert.equal(resolveWorkspace("/system").id, "admin");
assert.equal(resolveWorkspace("/unknown"), null);
assert.equal(resolveWorkspace("/live/unknown"), null);
assert.equal(resolveWorkspace("/people/unknown"), null);
assert.equal(resolveWorkspace("/timeline/unknown"), null);
assert.equal(resolveWorkspace("/recordings/unknown"), null);

assert.equal(canonicalWorkspacePath("/recordings"), "/review");
assert.equal(canonicalWorkspacePath("/recordings/exports"), "/exports");
assert.equal(canonicalWorkspacePath("/timeline/exports"), "/exports");
assert.equal(canonicalWorkspacePath("/recordings/search"), "/review");
assert.equal(canonicalWorkspacePath("/faces"), "/people");
assert.equal(canonicalWorkspacePath("/config"), "/admin");
assert.equal(canonicalWorkspacePath("/system"), "/admin");
assert.equal(canonicalWorkspacePath("/live"), "/");
assert.equal(canonicalWorkspacePath("/live/unknown"), "/live/unknown");
assert.equal(canonicalWorkspacePath("/people/unknown"), "/people/unknown");
assert.equal(canonicalWorkspacePath("/timeline/unknown"), "/timeline/unknown");
assert.equal(canonicalWorkspacePath("/recordings/unknown"), "/recordings/unknown");
assert.equal(canonicalWorkspacePath("/incidents"), "/review");
assert.equal(
  canonicalWorkspaceUrl("/recordings", "?camera=gate&at=123.5", "#player"),
  "/review?camera=gate&at=123.5&mode=timeline#player",
);
assert.equal(canonicalWorkspaceUrl("/incidents", "?event_ids=42"), "/review?event_ids=42");
assert.equal(canonicalWorkspaceUrl("/search", "?q=red+truck"), "/review?q=red+truck&mode=search");
assert.equal(canonicalWorkspaceUrl("/faces", "status=unknown"), "/people?status=unknown");
assert.equal(reviewMode("/timeline", "?camera=gate&at=123.5"), "timeline");
assert.equal(reviewMode("/review", "?event_ids=42"), "incidents");
assert.equal(reviewMode("/review", "?mode=search&q=truck"), "search");

const legacyTimeline = canonicalWorkspaceUrl("/timeline", "?camera=gate&at=123.5");
const legacyView = parseTimelineView(legacyTimeline.slice(legacyTimeline.indexOf("?")), "2026-08-27");
assert.equal(legacyView.cameraId, "gate");
assert.equal(legacyView.at, 123.5);

assert.equal(workspaceDefinition("review").label, "Review");
assert.equal(workspaceDefinition("admin").label, "System");
assert.equal(workspaceDefinition("exports").label, "Exports");
assert.equal(workspaceDefinition("unknown"), null);
assert.throws(() => workspaceHref("unknown"), /Unknown SurvNG workspace/);
assert.equal(workspaceHref("review", { event_ids: "42,43" }), "/review?event_ids=42%2C43");
assert.equal(
  timelineHref({ cameraId: "front-door", epoch: 123.5, source: "main" }),
  "/review?mode=timeline&camera=front-door&at=123.5&source=main",
);
assert.equal(
  timelineHref({ cameraId: "gate", epoch: 50, eventId: 42 }),
  "/review?mode=timeline&camera=gate&at=50&event=42",
);
assert.equal(
  timelineHref({ cameraId: "gate", epoch: 50, eventId: 84, trailEventIds: [12, 84] }),
  "/review?mode=timeline&camera=gate&at=50&event=84&trail=12%2C84",
);
assert.equal(timelineHref({ epoch: Number.NaN }), "/review?mode=timeline");

assert.deepEqual(DESKTOP_PRIMARY_WORKSPACES, ["live", "review", "people"]);
assert.deepEqual(MOBILE_PRIMARY_WORKSPACES, ["live", "review", "people"]);

const stylesSource = [
  readFileSync(new URL("../src/styles.css", import.meta.url), "utf8"),
  readFileSync(new URL("../src/shell/shell.css", import.meta.url), "utf8"),
].join("\n");
assert.equal(stylesSource.includes("grid-template-columns: 176px minmax(0, 1fr)"), false);
assert.match(stylesSource, /\.app-shell\.workspace-rail-collapsed\s*\{\s*--workspace-rail-width:\s*68px;/);

console.log("workspace navigation contract tests passed");

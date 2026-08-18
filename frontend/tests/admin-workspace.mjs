import assert from "node:assert/strict";
import { ADMIN_RESPONSIBILITY_GROUPS, adminDestination, adminWorkspaceId, adminWorkspaceSearch, cameraConfigDirtyState, comparableCameraSettings, comparableSystemConfig, configValuesEqual, nextTabId, preferredStoredValue, readAdminSubsection, readAdminWorkspace } from "../src/adminWorkspace.mjs";

assert.deepEqual(ADMIN_RESPONSIBILITY_GROUPS.map((group) => group.label), ["Configure", "Observe", "Act"]);
assert.deepEqual(ADMIN_RESPONSIBILITY_GROUPS[0].items.slice(0, 4).map((item) => item.label), ["Cameras", "Detection", "Storage", "Integrations"]);
assert.equal(adminDestination("general", { generalSection: "storage" }).id, "storage");
assert.equal(adminDestination("telemetry", { telemetrySection: "diagnostics" }).id, "diagnostics");
assert.equal(adminDestination("logs").id, "logs");

assert.equal(adminWorkspaceId("telemetry"), "telemetry");
assert.equal(adminWorkspaceId("invalid"), "general");
assert.equal(readAdminWorkspace("?section=audit&audit_id=12", "logs"), "audit");
assert.equal(readAdminWorkspace("", "logs"), "logs");
assert.equal(adminWorkspaceSearch("general", "?audit_id=12"), "");
assert.equal(adminWorkspaceSearch("general", "", { subsection: "storage" }), "?section=general&subsection=storage");
assert.equal(adminWorkspaceSearch("audit", "?audit_id=12"), "?section=audit&audit_id=12");
assert.equal(adminWorkspaceSearch("telemetry", "?audit_id=12"), "?section=telemetry");
assert.equal(adminWorkspaceSearch("cameras", "", { subsection: "zones", camera: "gate" }), "?section=cameras&subsection=zones&camera=gate");
assert.equal(readAdminSubsection("?subsection=zones", ["settings", "zones"], "settings"), "zones");
assert.equal(readAdminSubsection("?subsection=invalid", ["settings", "zones"], "settings"), "settings");
assert.equal(preferredStoredValue("telemetry", "logs", true), "telemetry");
assert.equal(preferredStoredValue("telemetry", "logs", false), "logs");
assert.deepEqual(comparableSystemConfig({ cameras: [{ id: "gate" }], detector: { confidence: 0.7 } }), { detector: { confidence: 0.7 } });
assert.equal(configValuesEqual({ a: 1 }, { a: 1 }), true);
assert.equal(configValuesEqual({ a: 1 }, { a: 2 }), false);
assert.deepEqual(comparableCameraSettings({ id: "gate", zones: [{ name: "drive" }] }), { id: "gate" });
assert.deepEqual(cameraConfigDirtyState(
  [{ id: "one", name: "Changed", zones: [] }, { id: "two", name: "Two", zones: [{ name: "new" }] }],
  [{ id: "one", name: "One", zones: [] }, { id: "two", name: "Two", zones: [] }],
), { settings: true, zones: true });
assert.deepEqual(cameraConfigDirtyState([{ id: "new", name: "New", zones: [] }], []), { settings: true, zones: false });
assert.equal(nextTabId(["one", "two", "three"], "two", "ArrowRight"), "three");
assert.equal(nextTabId(["one", "two", "three"], "one", "ArrowLeft"), "three");
assert.equal(nextTabId(["one", "two", "three"], "two", "Home"), "one");
assert.equal(nextTabId(["one", "two", "three"], "two", "End"), "three");
assert.equal(nextTabId(["one"], "one", "Tab"), null);

console.log("admin workspace tests passed");

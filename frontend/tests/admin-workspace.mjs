import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ADMIN_RESPONSIBILITY_GROUPS, adminDestination, ADMIN_NAV_GROUPS, adminWorkspaceId, adminWorkspaceSearch, baselineCamerasAfterSave, cameraConfigDirtyState, cameraSpeechDirty, comparableCameraSettings, comparableSystemConfig, configValuesEqual, dirtyCameraCount, nextTabId, normalizeTelemetrySection, perCameraDirtyState, preferredStoredValue, readAdminSubsection, readAdminWorkspace, resolveAdminWorkspace, telemetryLocationOptions } from "../src/adminWorkspace.mjs";

assert.deepEqual(ADMIN_RESPONSIBILITY_GROUPS.map((group) => group.label), ["Configure", "Observe", "Act"]);
assert.deepEqual(ADMIN_RESPONSIBILITY_GROUPS[0].items.slice(0, 5).map((item) => item.label), ["Cameras", "Detection", "Storage", "Integrations", "Access"]);
assert.deepEqual(ADMIN_RESPONSIBILITY_GROUPS[1].items.slice(0, 2).map((item) => item.label), ["Health", "Audit"]);
assert.equal(adminDestination("general", { generalSection: "storage" }).id, "storage");
assert.deepEqual(ADMIN_NAV_GROUPS.map((group) => group.label), ["", "Setup", "Observe", "Maintain"]);
assert.deepEqual(ADMIN_NAV_GROUPS.flatMap((group) => group.items.map((item) => item.id)), [
  "home", "cameras", "detection", "storage", "integrations", "homekit", "access", "server",
  "health", "audit", "logs", "tuneup", "diagnostics", "maintenance",
]);
assert.equal(adminDestination("home").label, "Overview");
assert.equal(adminDestination("home").id, "home");
assert.equal(adminDestination("telemetry", { telemetrySection: "diagnostics" }).id, "diagnostics");
assert.equal(adminDestination("telemetry", { telemetrySection: "health" }).id, "health");
assert.equal(adminDestination("telemetry", { telemetrySection: "occupancy" }).id, "health");
assert.equal(normalizeTelemetrySection("overview"), "health");
assert.equal(normalizeTelemetrySection("cameras"), "health");
assert.equal(normalizeTelemetrySection("occupancy"), "occupancy");
assert.equal(normalizeTelemetrySection("diagnostics"), "diagnostics");
assert.deepEqual(telemetryLocationOptions("health", "porch"), { camera: "porch" });
assert.deepEqual(telemetryLocationOptions("occupancy", "porch"), { subsection: "occupancy", camera: "porch" });
assert.deepEqual(telemetryLocationOptions("diagnostics"), { subsection: "diagnostics" });
assert.equal(adminDestination("logs").id, "logs");

assert.equal(adminWorkspaceId("telemetry"), "telemetry");
assert.equal(adminWorkspaceId("invalid"), "general");
assert.equal(readAdminWorkspace("?section=audit&audit_id=12", "logs"), "audit");
assert.equal(resolveAdminWorkspace("?section=general&subsection=motion-review", "general"), "calibration");
assert.equal(resolveAdminWorkspace("?section=audit", "general"), "audit");
assert.equal(readAdminWorkspace("", "logs"), "logs");
assert.equal(adminWorkspaceSearch("general", "?audit_id=12"), "");
assert.equal(adminWorkspaceSearch("general", "", { subsection: "storage" }), "?section=general&subsection=storage");
assert.equal(adminWorkspaceSearch("audit", "?audit_id=12"), "?section=audit&audit_id=12");
assert.equal(adminWorkspaceSearch("telemetry", "?audit_id=12"), "?section=telemetry");
assert.equal(adminWorkspaceSearch("telemetry", "", { subsection: "occupancy", camera: "porch" }), "?section=telemetry&subsection=occupancy&camera=porch");
assert.equal(adminWorkspaceSearch("cameras", "", { subsection: "zones", camera: "gate" }), "?section=cameras&subsection=zones&camera=gate");
assert.equal(adminWorkspaceSearch("audit", "", { camera: "gate" }), "?section=audit&camera=gate");
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
assert.deepEqual(perCameraDirtyState(
  [{ id: "one", name: "Changed", zones: [] }, { id: "two", name: "Two", zones: [{ name: "new" }] }],
  [{ id: "one", name: "One", zones: [] }, { id: "two", name: "Two", zones: [] }],
), {
  one: { settings: true, zones: false },
  two: { settings: false, zones: true },
});
assert.equal(dirtyCameraCount({
  one: { settings: true, zones: false },
  two: { settings: false, zones: false },
}), 1);
assert.deepEqual(cameraConfigDirtyState([{ id: "new", name: "New", zones: [] }], []), { settings: true, zones: false });
assert.equal(nextTabId(["one", "two", "three"], "two", "ArrowRight"), "three");
assert.equal(nextTabId(["one", "two", "three"], "one", "ArrowLeft"), "three");
assert.equal(nextTabId(["one", "two", "three"], "two", "Home"), "one");
assert.equal(nextTabId(["one", "two", "three"], "two", "End"), "three");
assert.equal(nextTabId(["one"], "one", "Tab"), null);

const configPageSource = readFileSync(new URL("../src/admin/ConfigPage.jsx", import.meta.url), "utf8");
assert.match(configPageSource, /detectionSection === "faces" \? <FaceCaptureSettings /);
const faceCaptureSource = readFileSync(new URL("../src/admin/FaceCaptureSettings.jsx", import.meta.url), "utf8");
assert.match(faceCaptureSource, /Face capture enabled/);
assert.match(faceCaptureSource, /Process previous incidents/);
const facesBlock = configPageSource.match(/detectionSection === "faces" \? <details[\s\S]*?: null\}/);
assert.ok(facesBlock, "previous face detection settings block is present");
assert.match(facesBlock[0], /Face detection enabled/);
assert.match(facesBlock[0], /Face Detector Model/);
assert.match(facesBlock[0], /Face Detection Confidence/);
assert.doesNotMatch(facesBlock[0], /Embedding Model|Landmark Model|Suggestion Threshold|References Per Person|Automatically identify|visit_auto_link|Model input profile|Recognition enabled/);
assert.doesNotMatch(configPageSource, /<h3>Face recognition<\/h3>/);
assert.match(configPageSource, /\["faces", "Faces"/);

// Camera speech switches are edited on the Detection page and saved with it.
assert.equal(cameraSpeechDirty([{ id: "gate", speech: { enabled: true } }], [{ id: "gate", speech: { enabled: false } }]), true);
assert.equal(cameraSpeechDirty([{ id: "gate", speech: { enabled: false }, name: "Renamed" }], [{ id: "gate", speech: { enabled: false }, name: "Gate" }]), false);
assert.equal(cameraSpeechDirty([], []), false);
assert.match(configPageSource, /\["speech", "Speech", Mic\]/);

console.log("admin workspace tests passed");

assert.equal(adminDestination("general", { generalSection: "homekit" }).id, "homekit");
const homekitBefore = { cameras: [{ id: "gate", homekit: { enabled: false } }] };
const homekitAfter = { cameras: [{ id: "gate", homekit: { enabled: true } }] };
assert.equal(configValuesEqual(comparableSystemConfig(homekitBefore), comparableSystemConfig(homekitAfter)), true);
assert.equal(cameraConfigDirtyState(homekitAfter.cameras, homekitBefore.cameras).settings, true);

// One Save sends every changed camera in one request. A change that reloads
// the manager restarts all cameras, so per-camera requests restarted them all
// once per camera (and a reload landing on the previous one could fail).
const saveAllBlock = configPageSource.match(/async function saveCurrentAdminChanges\(\) \{[\s\S]*?\n  \}\n/);
assert.ok(saveAllBlock, "saveCurrentAdminChanges is present");
assert.match(saveAllBlock[0], /saveCameras\(changedSettings\)/);
// Zones follow the id the batch saved, which changes for a new camera.
assert.match(saveAllBlock[0], /savedById\.get\(camera\.id\)\?\.id \?\? camera\.id/);
assert.doesNotMatch(saveAllBlock[0], /for \(const camera[^]*?saveCameras\(/);
assert.match(configPageSource, /fetch\("\/api\/config\/cameras", \{\s*method: "PUT"/);
assert.doesNotMatch(
  configPageSource,
  /fetch\(`\/api\/config\/cameras\/\$\{encodeURIComponent\(camera\.id\)\}`, \{\s*method: "PUT"/,
);

// After a batch save the baseline must hold every saved camera, even when a new
// camera's assigned id equals another new camera's client id.
{
  const zone = { id: "z1" };
  const baseline = [{ id: "front", name: "Front", zones: [zone] }];
  const sent = [
    { id: "front", name: "Front door", zones: [] },
    { id: "camera", name: "Camera 2", zones: [{ id: "drawn" }] },
    { id: "camera-2", name: "Camera 2", zones: [] },
  ];
  const savedById = new Map([
    ["front", { id: "front", name: "Front door" }],
    ["camera", { id: "camera-2", name: "Camera 2" }],
    ["camera-2", { id: "camera-2-2", name: "Camera 2" }],
  ]);
  const after = baselineCamerasAfterSave(baseline, sent, savedById);
  assert.deepEqual(after.map((camera) => camera.id), ["front", "camera-2", "camera-2-2"]);
  assert.equal(after[0].name, "Front door");
  assert.deepEqual(after[0].zones, [zone]); // saved zones stay in the baseline
  assert.notEqual(after[0].zones[0], zone); // and are copies, not shared state
  assert.deepEqual(baseline.map((camera) => camera.id), ["front"]); // input untouched
  assert.deepEqual(after[1].zones, []); // a new camera's zones are saved separately
  // Nothing is left looking unsaved.
  assert.equal(dirtyCameraCount(perCameraDirtyState(
    [
      { id: "front", name: "Front door", zones: [zone] },
      { id: "camera-2", name: "Camera 2", zones: [] },
      { id: "camera-2-2", name: "Camera 2", zones: [] },
    ],
    after,
  )), 0);
  // Its drawn zones still need their own save.
  assert.equal(dirtyCameraCount(perCameraDirtyState(
    [{ id: "camera-2", name: "Camera 2", zones: [{ id: "drawn" }] }],
    after,
  )), 1);
}

// Camera order after a batch save uses the saved ids and is saved only when it
// still differs from the saved baseline.
assert.match(saveAllBlock[0], /const orderIds = cameras\.map\(\(camera\) => savedById\.get\(camera\.id\)\?\.id \?\? camera\.id\)/);
assert.match(saveAllBlock[0], /if \(!configValuesEqual\(orderIds, savedOrder\)\) await saveCameraOrder\(orderIds\)/);
assert.doesNotMatch(saveAllBlock[0], /if \(cameraOrderDirty\)/);

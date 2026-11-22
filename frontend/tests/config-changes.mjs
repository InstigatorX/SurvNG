import assert from "node:assert/strict";
import {
  changeLabel,
  describeConfigChanges,
  formatChangeValue,
  humanizeKey,
  revertConfigChange,
  scopeConfigReview,
} from "../src/configChanges.mjs";

const baseline = {
  retention: { minimum_free_percent: 15 },
  mqtt: { host: "broker.local", password: "old-secret" },
  cameras: [
    {
      id: "driveway", name: "Driveway", rtsp_url: "rtsp://admin:hunter2@10.0.0.4/main",
      motion: { sensitivity: 0.42, ignore_lighting: true },
      zones: [{ id: "drive", name: "Driveway", points: [[0, 0], [1, 1]] }],
    },
    { id: "backyard", name: "Backyard", post_roll_seconds: 8, zones: [] },
    { id: "porch", name: "Porch", zones: [] },
  ],
};
const current = structuredClone(baseline);
current.retention.minimum_free_percent = 20;
current.mqtt.password = "new-secret";
current.cameras[0].rtsp_url = "rtsp://admin:hunter3@10.0.0.4/main";
current.cameras[0].motion.sensitivity = 0.55;
current.cameras[0].zones.push({ id: "street", name: "Street", points: [] });
current.cameras[1].post_roll_seconds = 12;
current.cameras.push({ id: "gate", name: "Side Gate", zones: [] });

const changes = describeConfigChanges(baseline, current);
assert.equal(changes.count, 7);
assert.deepEqual(changes.cameras.map((group) => [group.name, group.isNew, group.changes.length]), [
  ["Driveway", false, 3],
  ["Backyard", false, 1],
  ["Side Gate", true, 1],
]);
const [url, sensitivity, zone] = changes.cameras[0].changes;
// Credentials in stream URLs never appear in the review.
assert.deepEqual([url.label, url.before, url.after], ["RTSP URL", "rtsp://***@10.0.0.4/main", "rtsp://***@10.0.0.4/main"]);
assert.deepEqual([sensitivity.label, sensitivity.before, sensitivity.after], ["Motion › Sensitivity", "0.42", "0.55"]);
assert.deepEqual([zone.label, zone.kind, zone.text], ["Zones", "added", "+ Street"]);
assert.deepEqual(changes.system.map((change) => [change.label, change.before, change.after]), [
  ["Retention › Minimum free percent", "15", "20"],
  ["MQTT › Password", "hidden", "hidden"],
]);
assert.equal(changes.order, false);
assert.deepEqual(changes.removed, []);

// Undo one camera setting: only that value returns to the saved state.
const undone = revertConfigChange(current, baseline, sensitivity);
assert.equal(undone.cameras[0].motion.sensitivity, 0.42);
assert.equal(undone.cameras[0].zones.length, 2);
assert.equal(describeConfigChanges(baseline, undone).count, 6);
// Undo a zone addition removes just that zone; undo a new camera removes it.
assert.deepEqual(revertConfigChange(current, baseline, zone).cameras[0].zones.map((item) => item.id), ["drive"]);
const newCamera = changes.cameras[2].changes[0];
assert.deepEqual(revertConfigChange(current, baseline, newCamera).cameras.map((camera) => camera.id), ["driveway", "backyard", "porch"]);
// Undo a system setting.
assert.equal(revertConfigChange(current, baseline, changes.system[0]).retention.minimum_free_percent, 15);
// Undo restores removed zones in their original position and drops added keys.
const zoneEdited = structuredClone(baseline);
zoneEdited.cameras[0].zones = [];
zoneEdited.cameras[1].extra = true;
const editedChanges = describeConfigChanges(baseline, zoneEdited);
const removedZone = editedChanges.cameras[0].changes[0];
assert.deepEqual([removedZone.kind, removedZone.text], ["removed", "− Driveway"]);
assert.deepEqual(revertConfigChange(zoneEdited, baseline, removedZone).cameras[0].zones, baseline.cameras[0].zones);
const addedKey = editedChanges.cameras[1].changes[0];
assert.equal(addedKey.kind, "added");
assert.equal("extra" in revertConfigChange(zoneEdited, baseline, addedKey).cameras[1], false);

// Reordering and removal are reported, not mistaken for edits.
const reordered = structuredClone(baseline);
reordered.cameras = [reordered.cameras[1], reordered.cameras[0]];
const reorder = describeConfigChanges(baseline, reordered);
assert.equal(reorder.order, true);
assert.deepEqual(reorder.removed, [{ cameraId: "porch", name: "Porch" }]);
const swapped = structuredClone(baseline);
swapped.cameras = [swapped.cameras[1], swapped.cameras[0], swapped.cameras[2]];
assert.equal(describeConfigChanges(baseline, swapped).order, true);
assert.equal(describeConfigChanges(baseline, swapped).count, 1);

// No changes.
assert.deepEqual(describeConfigChanges(baseline, structuredClone(baseline)), { cameras: [], removed: [], system: [], order: false, count: 0 });

// Formatting.
assert.equal(humanizeKey("post_roll_seconds"), "Post roll seconds");
assert.equal(humanizeKey("onvif"), "ONVIF");
assert.equal(changeLabel(["homekit", "enabled"]), "HomeKit › Enabled");
assert.equal(formatChangeValue(true), "On");
assert.equal(formatChangeValue(undefined), "not set");
assert.equal(formatChangeValue(""), "empty");
assert.equal(formatChangeValue(["person", "car"]), "person, car");
assert.equal(formatChangeValue([{}, {}]), "2 items");
assert.equal(formatChangeValue("x", ["api_token"]), "hidden");
assert.equal(
  formatChangeValue("http://10.0.0.5/flv?channel=0&user=admin&password=hunter2&token=abc#x"),
  "http://10.0.0.5/flv?channel=0&user=admin&password=***&token=***#x",
);
assert.equal(formatChangeValue(0.1 + 0.2), "0.3");

// Scope: Cameras saves camera changes; Settings saves system and speech switches.
const speech = structuredClone(baseline);
speech.cameras[0].speech = { enabled: true };
speech.retention.minimum_free_percent = 25;
speech.cameras[1].post_roll_seconds = 30;
const speechReview = describeConfigChanges(baseline, speech);
const generalScope = scopeConfigReview(speechReview, "general");
assert.deepEqual(generalScope.cameras.map((group) => [group.name, group.changes.map((change) => change.label)]), [["Driveway", ["Speech"]]]);
assert.equal(generalScope.count, 2);
const cameraScope = scopeConfigReview(speechReview, "cameras");
assert.equal(cameraScope.system.length, 0);
assert.equal(cameraScope.count, 2);
assert.equal(scopeConfigReview(speechReview, "logs").count, 0);
assert.equal(scopeConfigReview(null, "cameras"), null);

console.log("config change review tests passed");

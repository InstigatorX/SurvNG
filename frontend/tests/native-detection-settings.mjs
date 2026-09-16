import assert from "node:assert/strict";
import { nativeDetectionError } from "../src/nativeDetectionSettings.mjs";
import { ADMIN_NAV_GROUPS, GENERAL_SECTION_LABELS } from "../src/adminWorkspace.mjs";
assert.equal(nativeDetectionError({}), "");
for (const value of [0, 6, 1.5]) assert.match(nativeDetectionError({native:{inference_interval:value}}), /Inference interval/);
assert.equal(nativeDetectionError({native:{inference_interval:5}}), "");
assert.equal(nativeDetectionError({ native: { stationary: { stationary_threshold: 0.05, moving_threshold: 0.05 } } }), "Stationary jitter threshold must be below the movement threshold.");
assert.match(nativeDetectionError({ native: { inference_requests: 1.5 } }), /whole number/);
assert.match(nativeDetectionError({ live_sample_fps: "" }), /frames per second/);
assert.match(nativeDetectionError({ native: { maximum_observation_age_seconds: NaN } }), /Maximum result age/);
assert.match(nativeDetectionError({ event_class_confirmation_frames: { person: 0 } }), /person/);
assert.match(nativeDetectionError({ event_class_confidence_thresholds: { car: 1 } }), /car/);
assert.equal(nativeDetectionError({ native: { maximum_tracks: 1024, stationary: { labels: [] } }, event_class_confirmation_frames: { person: 3 } }), "");
const destinations = ADMIN_NAV_GROUPS.flatMap((group) => group.items);
for (const retired of ["audit", "tuneup", "advisor"]) assert.ok(!destinations.some((item) => item.id === retired));
assert.ok(!("motion-review" in GENERAL_SECTION_LABELS));
console.log("Native detection settings passed");

assert.match(nativeDetectionError({live_sample_fps: .5, native:{batch_size:2}}), /Batch waiting time/);
assert.equal(nativeDetectionError({live_sample_fps:5, native:{batch_size:4}}), "");
assert.equal(nativeDetectionError({live_sample_fps:.5, native:{batch_size:1,inference_interval:5}}), "");

for (const tracking_classes of [null, [], ["person"]]) assert.equal(nativeDetectionError({native:{tracking_classes}}), "");
for (const tracking_classes of ["person", [""], [3]]) assert.match(nativeDetectionError({native:{tracking_classes}}), /Tracked classes/);

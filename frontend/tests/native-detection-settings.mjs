import assert from "node:assert/strict";
import { nativeDetectionError } from "../src/nativeDetectionSettings.mjs";

assert.equal(nativeDetectionError({}), "");
for (const value of [0, 6, 1.5]) assert.match(nativeDetectionError({native:{inference_interval:value}}), /Inference interval/);
assert.equal(nativeDetectionError({native:{inference_interval:5}}), "");
assert.match(nativeDetectionError({ native: { inference_requests: 1.5 } }), /whole number/);
assert.match(nativeDetectionError({ live_sample_fps: "" }), /frames per second/);
assert.match(nativeDetectionError({ native: { maximum_observation_age_seconds: NaN } }), /Maximum result age/);
assert.match(nativeDetectionError({ event_class_confirmation_frames: { person: 0 } }), /person/);
assert.match(nativeDetectionError({ event_class_confidence_thresholds: { car: 1 } }), /car/);
assert.equal(nativeDetectionError({ native: { maximum_tracks: 1024 }, event_class_confirmation_frames: { person: 3 } }), "");

assert.match(nativeDetectionError({live_sample_fps: .5, native:{batch_size:2}}), /Batch waiting time/);
assert.equal(nativeDetectionError({live_sample_fps:5, native:{batch_size:4}}), "");
assert.equal(nativeDetectionError({live_sample_fps:.5, native:{batch_size:1,inference_interval:5}}), "");

for (const tracking_classes of [null, [], ["person"]]) assert.equal(nativeDetectionError({native:{tracking_classes}}), "");
for (const tracking_classes of ["person", [""], [3]]) assert.match(nativeDetectionError({native:{tracking_classes}}), /Tracked classes/);

console.log("nativeDetectionSettings validation checks passed");

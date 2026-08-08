import assert from "node:assert/strict";
import { nextFaceReviewObservation } from "../src/faceReview.mjs";

const observations = [{ id: 10 }, { id: 20 }, { id: 30 }];

assert.equal(
  nextFaceReviewObservation(20, observations, [{ id: 10 }, { id: 30 }])?.id,
  30,
  "a removed suggestion advances into its former position",
);
assert.equal(
  nextFaceReviewObservation(20, observations, observations)?.id,
  30,
  "a confirmed face that remains under the current filter advances forward",
);
assert.equal(
  nextFaceReviewObservation(30, observations, [{ id: 10 }, { id: 20 }])?.id,
  20,
  "removing the final card selects the new final card",
);
assert.equal(nextFaceReviewObservation(20, observations, []), null);

console.log("face review navigation tests passed");

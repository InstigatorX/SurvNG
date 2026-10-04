import assert from "node:assert/strict";
import { recordingReviewActive, recordingReviewClosed, recordingReviewMinute, recordingReviewSummary, recordingReviewUrl } from "../src/recordingReview.mjs";

assert.equal(recordingReviewMinute(null), null);
assert.equal(recordingReviewMinute(undefined), null);
assert.equal(recordingReviewMinute(NaN), null);
assert.equal(recordingReviewMinute(-1), null);
assert.equal(recordingReviewMinute(125.99), 120);
assert.equal(recordingReviewMinute(179.99), 120);
assert.equal(recordingReviewMinute(180), 180);
assert.equal(recordingReviewClosed(null, 900), false);
assert.equal(recordingReviewClosed(120, 184.99), false);
assert.equal(recordingReviewClosed(120, 185), true);
assert.equal(recordingReviewActive("queued"), true);
assert.equal(recordingReviewActive("analyzing"), true);
for (const state of ["sampled", "partial", "failed", "unavailable", "unreviewed", undefined]) assert.equal(recordingReviewActive(state), false);
assert.equal(recordingReviewSummary({ state: "sampled" }), "Sampled");
assert.equal(recordingReviewSummary({ state: "partial" }), "Partial review");
const url = new URL(recordingReviewUrl("front/gate", "live", 120), "http://localhost");
assert.equal(url.pathname, "/api/cameras/front%2Fgate/recordings/review");
assert.equal(url.searchParams.get("source"), "live");
assert.equal(url.searchParams.get("epoch"), "120");
globalThis.window = { __SURVNG_BASE_PATH__: "/survng" };
globalThis.document = { documentElement: { dataset: {} } };
const { recordingDayUrl, recordingUpdatesUrl, recordingGridDayUrl, recordingGridUpdatesUrl } = await import("../src/shared/mediaUrls.js");
for (const [builder, args] of [
  [recordingDayUrl, ["gate", 120, 240, "main"]],
  [recordingUpdatesUrl, ["gate", 120, 240, 180, "main"]],
  [recordingGridDayUrl, [120, 240, "main"]],
  [recordingGridUpdatesUrl, [120, 240, 180, "main"]],
]) {
  assert.equal(new URL(builder(...args), "http://localhost").searchParams.has("review_only"), false);
  assert.equal(new URL(builder(...args, false, true), "http://localhost").searchParams.get("review_only"), "true");
}
console.log("recording review: fixed minute, closed-window guard, explicit states and escaped scope passed");

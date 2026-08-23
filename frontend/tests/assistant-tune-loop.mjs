import assert from "node:assert/strict";
import {
  assistantActionKey,
  assistantConfirmPostKind,
  buildApplyCameraReviewAction,
  buildEvaluationFollowupAction,
  buildStartCameraReviewAction,
  isAssistantConfirmPostAllowed,
  summarizeMotionAiReview,
} from "../src/assistant/assistantTuneLoop.mjs";

const start = buildStartCameraReviewAction("gate", "Gate");
assert.equal(start.kind, "confirm_post");
assert.equal(start.path, "/api/motion-ai-reviews");
assert.equal(start.body.camera_id, "gate");
assert.equal(assistantConfirmPostKind(start.path), "start_review");
assert.equal(isAssistantConfirmPostAllowed(start), true);
assert.match(assistantActionKey(start), /^confirm_post:/);

const apply = buildApplyCameraReviewAction({
  id: 9,
  camera_id: "gate",
  status: "completed",
  result: {
    can_apply: true,
    configuration_fingerprint: "abc",
    recommendations: [{ setting: "sensitivity", current: "balanced", proposed: "high", reasons: ["more misses"] }],
  },
});
assert.equal(apply.path, "/api/motion-ai-reviews/9/apply");
assert.equal(apply.body.changes[0].value, "high");
assert.equal(assistantConfirmPostKind(apply.path), "apply_review");

assert.equal(buildApplyCameraReviewAction({ id: 9, result: { can_apply: false, recommendations: [{ setting: "sensitivity" }] } }), null);

const followup = buildEvaluationFollowupAction({ id: 3, camera_id: "gate", status: "ready" });
assert.equal(followup.path, "/api/camera-intelligence/evaluations/3/follow-up");
assert.equal(buildEvaluationFollowupAction({ id: 3, status: "collecting" }), null);

assert.match(summarizeMotionAiReview({
  camera_id: "gate",
  status: "completed",
  analyzed: 8,
  failed: 1,
  result: { summary: "Looks consistent overall.", recommendations: [] },
}), /finished/);

assert.equal(isAssistantConfirmPostAllowed({ kind: "confirm_post", label: "Nope", path: "/api/config" }), false);
assert.equal(assistantConfirmPostKind("/api/incidents/1/ai-apply"), "");

console.log("assistant tune loop tests passed");

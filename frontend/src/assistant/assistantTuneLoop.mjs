/** Allowed assistant confirm_post targets for the Camera Advisor tune loop. */

const MOTION_REVIEW_START = "/api/motion-ai-reviews";
const MOTION_REVIEW_APPLY = /^\/api\/motion-ai-reviews\/(\d+)\/apply$/;
const EVALUATION_FOLLOWUP = /^\/api\/camera-intelligence\/evaluations\/(\d+)\/follow-up$/;

export function assistantConfirmPostKind(path) {
  const cleaned = String(path || "").trim();
  if (cleaned === MOTION_REVIEW_START) return "start_review";
  if (MOTION_REVIEW_APPLY.test(cleaned)) return "apply_review";
  if (EVALUATION_FOLLOWUP.test(cleaned)) return "evaluation_followup";
  return "";
}

export function isAssistantConfirmPostAllowed(action) {
  if (!action || typeof action !== "object") return false;
  if (String(action.kind || "") !== "confirm_post") return false;
  const label = String(action.label || "").trim();
  const path = String(action.path || "").trim();
  if (!label || !path) return false;
  return Boolean(assistantConfirmPostKind(path));
}

export function assistantActionKey(action) {
  if (!action || typeof action !== "object") return "";
  if (String(action.kind || "href") === "confirm_post") {
    return `confirm_post:${action.path}:${action.label}`;
  }
  return `href:${action.href}:${action.label}`;
}

export function buildStartCameraReviewAction(cameraId, cameraName = "", { hours = 24, imageLimit = 12 } = {}) {
  const id = String(cameraId || "").trim();
  const name = String(cameraName || id || "this camera").trim() || "this camera";
  return {
    kind: "confirm_post",
    label: `Start multi-sample review for ${name}`,
    path: MOTION_REVIEW_START,
    body: {
      camera_id: id,
      hours,
      record_limit: 100,
      image_limit: imageLimit,
    },
    confirm: `Start a Camera Advisor multi-sample review for ${name}? This inspects up to ${imageLimit} recent images and may take a few minutes. Nothing is applied automatically.`,
  };
}

export function buildApplyCameraReviewAction(review) {
  const reviewId = Number(review?.id || 0);
  const result = review?.result || {};
  const recommendations = Array.isArray(result.recommendations) ? result.recommendations : [];
  const cameraId = String(review?.camera_id || "").trim();
  if (!reviewId || !recommendations.length || !result.can_apply) return null;
  const count = recommendations.length;
  return {
    kind: "confirm_post",
    label: `Apply ${count} Camera Advisor recommendation${count === 1 ? "" : "s"}`,
    path: `/api/motion-ai-reviews/${reviewId}/apply`,
    body: {
      confirmed: true,
      configuration_fingerprint: String(result.configuration_fingerprint || ""),
      evaluation_hours: 24,
      changes: recommendations.map((recommendation) => ({
        scope: "camera",
        setting: recommendation.setting,
        value: recommendation.proposed ?? recommendation.value,
        reason: recommendation.reasons?.[0] || recommendation.reason || "Repeated review evidence supports this change.",
      })),
    },
    confirm: `Apply ${count} reviewed setting change${count === 1 ? "" : "s"} to ${cameraId || "this camera"}? SurvNG will restart that camera and start a 24-hour effectiveness check. Nothing else is changed.`,
    proposals: recommendations.map((recommendation) => ({
      setting: recommendation.setting,
      current: recommendation.current,
      proposed: recommendation.proposed ?? recommendation.value,
    })),
  };
}

export function buildEvaluationFollowupAction(evaluation, { imageLimit = 12 } = {}) {
  const evaluationId = Number(evaluation?.id || 0);
  const cameraId = String(evaluation?.camera_id || "").trim();
  if (!evaluationId || evaluation?.status !== "ready") return null;
  return {
    kind: "confirm_post",
    label: `Check whether the change helped${cameraId ? ` on ${cameraId}` : ""}`,
    path: `/api/camera-intelligence/evaluations/${evaluationId}/follow-up`,
    body: { image_limit: imageLimit },
    confirm: `Run the effectiveness follow-up for ${cameraId || "this camera"}? SurvNG will sample recent images and compare them to the pre-change review.`,
  };
}

export function summarizeMotionAiReview(review) {
  const result = review?.result || {};
  const analyzed = Number(review?.analyzed || result.analyzed || 0);
  const failed = Number(review?.failed || result.failed || 0);
  const recommendations = Array.isArray(result.recommendations) ? result.recommendations : [];
  const summary = String(result.summary || "").trim();
  const cameraId = String(review?.camera_id || "this camera");
  if (review?.status === "failed") {
    return `The multi-sample review for ${cameraId} failed${review.error ? `: ${review.error}` : "."}`;
  }
  const parts = [
    `Multi-sample review for ${cameraId} finished.`,
    summary || `Inspected ${analyzed} image${analyzed === 1 ? "" : "s"}${failed ? ` (${failed} skipped)` : ""}.`,
  ];
  if (recommendations.length) {
    parts.push(`Camera Advisor has ${recommendations.length} recommendation${recommendations.length === 1 ? "" : "s"} you can apply after confirmation.`);
  } else {
    parts.push("No repeated-evidence setting changes were recommended.");
  }
  return parts.join(" ");
}

export function summarizeEffectivenessEvaluation(evaluation) {
  const comparison = evaluation?.comparison || {};
  const summary = String(comparison.summary || evaluation?.error || "").trim();
  const cameraId = String(evaluation?.camera_id || "this camera");
  if (evaluation?.status === "completed" && summary) {
    return `Effectiveness check for ${cameraId}: ${summary}`;
  }
  if (evaluation?.status === "reviewing") {
    return `Reviewing post-change activity for ${cameraId}…`;
  }
  return `Effectiveness follow-up for ${cameraId} is running.`;
}

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  Check,
  CircleAlert,
  Download,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { browserStorage } from "../storage.mjs";
import { readAssistantHistory, writeAssistantHistory } from "../assistantStorage.mjs";
import { assistantContextLabel, assistantContextPrompts, snapshotAssistantContext } from "../assistantContext.mjs";
import {
  assistantCoachSeen,
  assistantComposerPlaceholder,
  assistantThinkingStages,
  assistantWelcomeCopy,
  markAssistantCoachSeen,
  stripAssistantCitationMarkers,
} from "../assistantMessage.mjs";
import { assistantEvidenceHref, assistantIncidentHref } from "../assistantNavigation.mjs";
import {
  assistantActionKey,
  assistantConfirmPostKind,
  buildApplyCameraReviewAction,
  buildEvaluationFollowupAction,
  isAssistantConfirmPostAllowed,
  summarizeEffectivenessEvaluation,
  summarizeMotionAiReview,
} from "./assistantTuneLoop.mjs";
import { assistantActiveExportIds, fetchAssistantExportJobs, mergeAssistantExportJobs } from "../exportPolling.mjs";
import { appUrl, fetch } from "../shared/api.js";
import { formatDateTime } from "../shared/format.js";
import { useStoredState, useViewportQuery } from "../shared/hooks.js";
import { useVisiblePolling } from "../visibilityPolling.mjs";

export const ASSISTANT_STORAGE_KEY = "survng.assistantConversation.v1";

export const assistantVisualVerdicts = {
  detection_consistent: "What SurvNG found matches the image",
  probable_missed_detection: "SurvNG likely missed something visible",
  probable_misclassification: "The object was likely labeled incorrectly",
  probable_false_positive: "The detection was likely a false alarm",
  uncertain: "The single image is inconclusive",
};

export const assistantDetectorAssessments = {
  consistent: "Matches the image",
  missed: "Likely missed a visible object",
  misclassified: "Likely used the wrong label",
  false_positive: "Likely detected something that is not there",
  uncertain: "Not enough visual evidence",
};

export const assistantTrackingAssessments = {
  consistent: "Followed the object normally",
  late: "Started following it late",
  lost: "Stopped following it too early",
  duplicate: "Likely counted one object more than once",
  unavailable: "No useful follow-up tracking was available",
  uncertain: "Not enough evidence to judge tracking",
};

export const assistantSettingLabels = {
  analysis_preset: "Motion analysis style",
  sensitivity: "Motion sensitivity",
  stationary_object_tolerance: "Stationary object policy",
  frame_width: "Motion analysis image size",
  borderline_rescue_enabled: "Second look at borderline motion",
  borderline_margin: "Borderline motion range",
};

export function readAssistantMessages() {
  return readAssistantHistory(browserStorage(window), ASSISTANT_STORAGE_KEY);
}

function AssistantMessageText({ content }) {
  return <div className="assistant-message-text">{stripAssistantCitationMarkers(content)}</div>;
}

function apiErrorDetail(payload, fallback) {
  const detail = payload?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  return fallback;
}

export function AssistantPanel({ pageContext, timeZone, askRequest = null, onAskRequestHandled = null }) {
  const [openValue, setOpenValue] = useStoredState("survng.assistantOpen.v1", "false");
  const open = openValue === "true";
  const [messages, setMessages] = useState(readAssistantMessages);
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [thinkingStage, setThinkingStage] = useState(0);
  const [applyingEvidenceId, setApplyingEvidenceId] = useState("");
  const [postingActionKey, setPostingActionKey] = useState("");
  const [error, setError] = useState(null);
  const [applyReview, setApplyReview] = useState(null);
  const [confirmPost, setConfirmPost] = useState(null);
  const [showCoach, setShowCoach] = useState(() => !assistantCoachSeen(browserStorage(window)));
  const bodyRef = useRef(null);
  const drawerRef = useRef(null);
  const applyDialogRef = useRef(null);
  const confirmDialogRef = useRef(null);
  const applyReturnFocusRef = useRef(null);
  const applyEvidenceRef = useRef(null);
  const confirmReturnFocusRef = useRef(null);
  const launcherRef = useRef(null);
  const composerRef = useRef(null);
  const followTranscriptRef = useRef(true);
  const returnFocusRef = useRef(null);
  const abortRef = useRef(null);
  const handledAskRef = useRef(null);
  const reviewPollRef = useRef({});
  const evaluationPollRef = useRef({});
  const modalOpen = Boolean(applyReview || confirmPost);
  const compactAssistant = useViewportQuery("(max-width: 1279px)");
  const currentContext = useMemo(() => snapshotAssistantContext(pageContext, timeZone), [pageContext, timeZone]);
  const currentContextLabel = assistantContextLabel(currentContext, (epoch) => formatDateTime(epoch, timeZone));
  const quickPrompts = assistantContextPrompts(currentContext).slice(0, 3);
  const welcomeCopy = assistantWelcomeCopy(currentContextLabel);
  const thinkingStages = useMemo(() => assistantThinkingStages(currentContextLabel), [currentContextLabel]);
  const composerPlaceholder = assistantComposerPlaceholder(currentContextLabel);
  const activeExportIds = assistantActiveExportIds(messages).join(",");

  useEffect(() => {
    writeAssistantHistory(browserStorage(window), ASSISTANT_STORAGE_KEY, messages);
  }, [messages]);

  async function loadAssistantStatus() {
    setError((current) => current?.kind === "status" ? null : current);
    try {
      const response = await fetch("/api/assistant/status");
      if (!response.ok) throw new Error("Assistant status unavailable");
      setStatus(await response.json());
    } catch (statusError) {
      setError({ message: statusError.message || "Assistant status unavailable", kind: "status" });
    }
  }

  useEffect(() => {
    if (!open) return;
    void loadAssistantStatus();
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const body = bodyRef.current;
    if (body && followTranscriptRef.current) body.scrollTop = body.scrollHeight;
  }, [messages, busy, open]);

  useEffect(() => {
    if (!open) return undefined;
    returnFocusRef.current = document.activeElement;
    window.requestAnimationFrame(() => composerRef.current?.focus());
    return () => {
      const target = returnFocusRef.current;
      if (target instanceof HTMLElement && target.isConnected) target.focus();
      else launcherRef.current?.focus();
    };
  }, [open]);

  useVisiblePolling(async (signal) => {
    const ids = activeExportIds.split(",").filter(Boolean);
    const updates = await fetchAssistantExportJobs(ids, fetch, { signal });
    if (updates.length) setMessages((current) => mergeAssistantExportJobs(current, updates));
  }, 2_000, Boolean(open && activeExportIds), { restartKey: activeExportIds });

  useEffect(() => {
    if (!open) return undefined;
    function handleAssistantKeyboard(event) {
      if (event.key === "Escape") {
        if (confirmPost) closeConfirmPost(false);
        else if (applyReview) closeApplyReview(false);
        else setOpenValue("false");
        return;
      }
      if (!compactAssistant || event.key !== "Tab" || modalOpen) return;
      const controls = [...(drawerRef.current?.querySelectorAll('button:not([disabled]), a[href], textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])') || [])];
      if (!controls.length) return;
      const first = controls[0];
      const last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
    window.addEventListener("keydown", handleAssistantKeyboard);
    return () => window.removeEventListener("keydown", handleAssistantKeyboard);
  }, [applyReview, compactAssistant, confirmPost, modalOpen, open, setOpenValue]);

  useEffect(() => {
    if (!applyReview && !confirmPost) return undefined;
    const dialog = (confirmPost ? confirmDialogRef.current : applyDialogRef.current);
    const controls = [...(dialog?.querySelectorAll('button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])') || [])];
    controls[0]?.focus();
    function trapModalFocus(event) {
      if (event.key !== "Tab" || !controls.length) return;
      const first = controls[0];
      const last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
    dialog?.addEventListener("keydown", trapModalFocus);
    return () => dialog?.removeEventListener("keydown", trapModalFocus);
  }, [applyReview, confirmPost]);

  useEffect(() => {
    if (!busy) {
      setThinkingStage(0);
      return undefined;
    }
    setThinkingStage(0);
    const timer = window.setInterval(() => {
      setThinkingStage((current) => (current + 1) % Math.max(1, thinkingStages.length));
    }, 2200);
    return () => window.clearInterval(timer);
  }, [busy, thinkingStages.length]);

  useEffect(() => {
    if (!askRequest?.id) return;
    setOpenValue("true");
    markAssistantCoachSeen(browserStorage(window));
    setShowCoach(false);
  }, [askRequest?.id, setOpenValue]);

  useEffect(() => {
    if (!askRequest?.id || !open || busy) return;
    if (handledAskRef.current === askRequest.id) return;
    handledAskRef.current = askRequest.id;
    const prompt = String(askRequest.prompt || "").trim() || "Analyze this incident";
    onAskRequestHandled?.();
    void sendMessage(prompt);
  }, [askRequest, open, busy]);

  useEffect(() => () => {
    abortRef.current?.abort();
    Object.values(reviewPollRef.current).forEach((timer) => window.clearInterval(timer));
    Object.values(evaluationPollRef.current).forEach((timer) => window.clearInterval(timer));
    reviewPollRef.current = {};
    evaluationPollRef.current = {};
  }, []);

  function dismissCoach() {
    markAssistantCoachSeen(browserStorage(window));
    setShowCoach(false);
  }

  function cancelInFlight() {
    abortRef.current?.abort();
  }

  function clearConversation() {
    if (messages.length && !window.confirm("Start a new assistant conversation? The current conversation will be cleared.")) return;
    abortRef.current?.abort();
    Object.values(reviewPollRef.current).forEach((timer) => window.clearInterval(timer));
    Object.values(evaluationPollRef.current).forEach((timer) => window.clearInterval(timer));
    reviewPollRef.current = {};
    evaluationPollRef.current = {};
    setMessages([]);
    setError(null);
  }

  function appendAssistantMessage(partial) {
    setMessages((current) => [...current, {
      id: `assistant-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      role: "assistant",
      content: "",
      evidence: [],
      suggestions: [],
      actions: [],
      reasoningTier: "fast",
      model: "",
      ...partial,
    }].slice(-30));
  }

  function markActionsSpent(messageId, action) {
    const key = assistantActionKey(action);
    setMessages((current) => current.map((message) => {
      if (message.id !== messageId) return message;
      return {
        ...message,
        actions: (message.actions || []).map((item) => assistantActionKey(item) === key ? { ...item, spent: true } : item),
        evidence: (message.evidence || []).map((item) => ({
          ...item,
          details: item.details?.next_actions ? {
            ...item.details,
            next_actions: item.details.next_actions.map((next) => assistantActionKey(next) === key ? { ...next, spent: true } : next),
          } : item.details,
        })),
      };
    }));
  }

  function openApplyReview(messageId, evidence, trigger) {
    applyReturnFocusRef.current = trigger || null;
    applyEvidenceRef.current = trigger?.closest?.(".assistant-evidence-card") || null;
    setApplyReview({ messageId, evidence });
  }

  function closeApplyReview(confirmed = false) {
    const target = confirmed ? applyEvidenceRef.current : applyReturnFocusRef.current;
    setApplyReview(null);
    window.requestAnimationFrame(() => {
      if (target instanceof HTMLElement && target.isConnected) target.focus();
      else composerRef.current?.focus();
    });
  }

  function openConfirmPost(action, messageId, trigger) {
    if (!isAssistantConfirmPostAllowed(action) || action.spent || postingActionKey) return;
    confirmReturnFocusRef.current = trigger || null;
    setConfirmPost({ action, messageId });
  }

  function closeConfirmPost(confirmed = false) {
    const target = confirmed ? null : confirmReturnFocusRef.current;
    setConfirmPost(null);
    window.requestAnimationFrame(() => {
      if (target instanceof HTMLElement && target.isConnected) target.focus();
      else composerRef.current?.focus();
    });
  }

  function stopReviewPoll(reviewId) {
    const key = String(reviewId);
    if (reviewPollRef.current[key]) {
      window.clearInterval(reviewPollRef.current[key]);
      delete reviewPollRef.current[key];
    }
  }

  function stopEvaluationPoll(evaluationId) {
    const key = String(evaluationId);
    if (evaluationPollRef.current[key]) {
      window.clearInterval(evaluationPollRef.current[key]);
      delete evaluationPollRef.current[key];
    }
  }

  async function pollMotionAiReview(reviewId) {
    const response = await fetch(`/api/motion-ai-reviews/${reviewId}`);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(apiErrorDetail(payload, "Unable to load the camera review"));
    return payload;
  }

  function startReviewPolling(reviewId) {
    const key = String(reviewId);
    if (reviewPollRef.current[key]) return;
    const tick = async () => {
      try {
        const review = await pollMotionAiReview(reviewId);
        if (["queued", "running"].includes(String(review.status || ""))) return;
        stopReviewPoll(reviewId);
        setMessages((current) => current.map((message) => (
          message.tuneLoop?.reviewId === Number(reviewId) && message.tuneLoop?.phase === "reviewing"
            ? { ...message, tuneLoop: { ...message.tuneLoop, phase: review.status === "completed" ? "awaiting_apply" : "failed" } }
            : message
        )));
        const applyAction = buildApplyCameraReviewAction(review);
        const advisorHref = review.camera_id
          ? `/admin?section=general&subsection=motion-review&camera=${encodeURIComponent(review.camera_id)}`
          : "/admin?section=general&subsection=motion-review";
        appendAssistantMessage({
          content: summarizeMotionAiReview(review),
          actions: [
            ...(applyAction ? [applyAction] : []),
            { label: "Open Camera Advisor", href: advisorHref },
          ],
          tuneLoop: { reviewId: Number(review.id || reviewId), cameraId: review.camera_id || "", phase: review.status === "completed" ? "awaiting_apply" : "failed" },
        });
      } catch (pollError) {
        stopReviewPoll(reviewId);
        setError({ message: pollError.message || "Unable to track the camera review", kind: "tune" });
      }
    };
    void tick();
    reviewPollRef.current[key] = window.setInterval(() => { void tick(); }, 4_000);
  }

  function startEvaluationPolling(evaluationId, cameraId) {
    const key = String(evaluationId);
    if (evaluationPollRef.current[key]) return;
    const tick = async () => {
      try {
        const response = await fetch(`/api/camera-intelligence/evaluations/latest?camera_id=${encodeURIComponent(cameraId)}`);
        const evaluation = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(apiErrorDetail(evaluation, "Unable to load the effectiveness check"));
        if (Number(evaluation.id || 0) !== Number(evaluationId)) return;
        if (evaluation.status === "collecting") return;
        if (evaluation.status === "ready") {
          stopEvaluationPoll(evaluationId);
          setMessages((current) => current.map((message) => (
            message.tuneLoop?.evaluationId === Number(evaluationId) && message.tuneLoop?.phase === "collecting"
              ? { ...message, tuneLoop: { ...message.tuneLoop, phase: "ready" } }
              : message
          )));
          const followup = buildEvaluationFollowupAction(evaluation);
          appendAssistantMessage({
            content: `Enough time has passed to check whether the change helped on ${cameraId || "this camera"}.`,
            actions: followup ? [followup] : [],
            tuneLoop: { evaluationId: Number(evaluationId), cameraId, phase: "ready" },
          });
          return;
        }
        if (evaluation.status === "completed") {
          stopEvaluationPoll(evaluationId);
          setMessages((current) => current.map((message) => (
            message.tuneLoop?.evaluationId === Number(evaluationId) && ["collecting", "followup_running", "ready"].includes(message.tuneLoop?.phase)
              ? { ...message, tuneLoop: { ...message.tuneLoop, phase: "done" } }
              : message
          )));
          appendAssistantMessage({
            content: summarizeEffectivenessEvaluation(evaluation),
            tuneLoop: { evaluationId: Number(evaluationId), cameraId, phase: "done" },
          });
          return;
        }
        if (evaluation.status === "reviewing") {
          setMessages((current) => current.map((message) => (
            message.tuneLoop?.evaluationId === Number(evaluationId) && message.tuneLoop?.phase !== "followup_running"
              ? { ...message, tuneLoop: { ...message.tuneLoop, phase: "followup_running" } }
              : message
          )));
        }
      } catch (pollError) {
        stopEvaluationPoll(evaluationId);
        setError({ message: pollError.message || "Unable to track the effectiveness check", kind: "tune" });
      }
    };
    void tick();
    evaluationPollRef.current[key] = window.setInterval(() => { void tick(); }, 8_000);
  }

  async function executeConfirmPost(action, messageId) {
    if (!isAssistantConfirmPostAllowed(action) || postingActionKey) return;
    const key = assistantActionKey(action);
    const kind = assistantConfirmPostKind(action.path);
    setPostingActionKey(key);
    setError(null);
    markActionsSpent(messageId, action);
    try {
      const response = await fetch(action.path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(action.body || {}),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(apiErrorDetail(payload, "Unable to run that Camera Advisor step"));
      if (kind === "start_review") {
        const reviewId = Number(payload.id || 0);
        const cameraId = String(payload.camera_id || action.body?.camera_id || "");
        appendAssistantMessage({
          content: `Started a multi-sample Camera Advisor review${cameraId ? ` for ${cameraId}` : ""}. I’ll update this chat when it finishes.`,
          tuneLoop: { reviewId, cameraId, phase: "reviewing" },
        });
        if (reviewId) startReviewPolling(reviewId);
        return;
      }
      if (kind === "apply_review") {
        const followUp = payload.follow_up;
        const evaluation = payload.effectiveness_evaluation || {};
        const evaluationId = Number(evaluation.id || 0);
        const cameraId = String(payload.camera_id || evaluation.camera_id || "");
        appendAssistantMessage({
          content: followUp?.message || `Applied Camera Advisor recommendations${cameraId ? ` on ${cameraId}` : ""}.`,
          suggestions: followUp?.suggestions || [],
          actions: followUp?.actions || [],
          tuneLoop: evaluationId ? { evaluationId, cameraId, phase: "collecting" } : undefined,
        });
        if (evaluationId && cameraId) startEvaluationPolling(evaluationId, cameraId);
        return;
      }
      if (kind === "evaluation_followup") {
        const evaluationId = Number(payload.id || action.path.match(/evaluations\/(\d+)/)?.[1] || 0);
        const cameraId = String(payload.camera_id || "");
        appendAssistantMessage({
          content: `Started the effectiveness follow-up${cameraId ? ` for ${cameraId}` : ""}. I’ll share the comparison when it finishes.`,
          tuneLoop: { evaluationId, cameraId, phase: "followup_running" },
        });
        if (evaluationId && cameraId) startEvaluationPolling(evaluationId, cameraId);
      }
    } catch (postError) {
      setError({ message: postError.message || "Unable to run that Camera Advisor step", kind: "tune" });
    } finally {
      setPostingActionKey("");
    }
  }

  function renderAssistantActions(actions, messageId) {
    if (!actions?.length) return null;
    return (
      <div className="assistant-next-actions">
        {actions.map((action) => {
          const key = assistantActionKey(action);
          if (String(action.kind || "href") === "confirm_post") {
            if (!isAssistantConfirmPostAllowed(action)) return null;
            return (
              <button
                type="button"
                key={key}
                disabled={Boolean(action.spent) || Boolean(postingActionKey) || busy}
                onClick={(event) => openConfirmPost(action, messageId, event.currentTarget)}
              >
                {postingActionKey === key ? "Working…" : action.label}
              </button>
            );
          }
          if (!action.href) return null;
          return <a key={key} href={appUrl(action.href)}>{action.label}</a>;
        })}
      </div>
    );
  }

  useEffect(() => {
    if (!open) {
      Object.values(reviewPollRef.current).forEach((timer) => window.clearInterval(timer));
      Object.values(evaluationPollRef.current).forEach((timer) => window.clearInterval(timer));
      reviewPollRef.current = {};
      evaluationPollRef.current = {};
      return undefined;
    }
    for (const message of messages) {
      const loop = message.tuneLoop || {};
      if (loop.phase === "reviewing" && Number(loop.reviewId) > 0 && !reviewPollRef.current[String(loop.reviewId)]) {
        startReviewPolling(Number(loop.reviewId));
      }
      if (["collecting", "followup_running"].includes(loop.phase) && Number(loop.evaluationId) > 0 && loop.cameraId && !evaluationPollRef.current[String(loop.evaluationId)]) {
        startEvaluationPolling(Number(loop.evaluationId), loop.cameraId);
      }
    }
    return undefined;
  }, [open, messages]);

  async function sendMessage(messageText = draft, contextOverride = null, { appendUser = true } = {}) {
    const content = String(messageText || "").trim();
    if (!content || busy) return;
    const submittedContext = contextOverride || snapshotAssistantContext(currentContext, timeZone);
    const userMessage = { id: `user-${Date.now()}`, role: "user", content, context: submittedContext };
    const prior = messages.slice(-12);
    if (appendUser) setMessages((current) => [...current, userMessage].slice(-30));
    setDraft("");
    setError(null);
    followTranscriptRef.current = true;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    try {
      const response = await fetch("/api/assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          message: content,
          history: prior.map(({ role, content: historyContent, context: historyContext, evidence: historyEvidence }) => ({
            role,
            content: historyContent,
            ...(historyContext ? { context: snapshotAssistantContext(historyContext, historyContext.time_zone || timeZone) } : {}),
            ...(historyEvidence?.length ? { evidence: historyEvidence.slice(0, 12).map((item) => ({ id: String(item.id || ""), kind: String(item.kind || ""), title: String(item.title || "") })).filter((item) => item.id) } : {}),
          })),
          context: submittedContext,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Assistant failed (${response.status})`);
      setStatus((current) => current ? { ...current, configured: true } : current);
      setMessages((current) => [...current, {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: payload.message || "No answer returned.",
        evidence: payload.evidence || [],
        citations: payload.citations || [],
        suggestions: payload.suggestions || [],
        actions: payload.actions || [],
        reasoningTier: payload.reasoning_tier || "fast",
        model: payload.model || "",
        context: submittedContext,
      }].slice(-30));
    } catch (sendError) {
      if (sendError?.name === "AbortError") return;
      const statusCode = Number(String(sendError.message || "").match(/\((\d+)\)/)?.[1]);
      const message = statusCode === 429 ? "The AI provider is busy. Wait a moment, then retry."
        : [502, 503, 504].includes(statusCode) ? "The AI provider is temporarily unavailable. Retry when it recovers."
          : sendError.message || "Assistant request failed";
      setError({ message, kind: "request", content, context: submittedContext });
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setBusy(false);
    }
  }

  async function applyVisualProposals(messageId, evidence) {
    const details = evidence?.details || {};
    const changes = details.advice?.changes || [];
    if (!changes.length || applyingEvidenceId) return;
    setApplyingEvidenceId(evidence.id);
    setError(null);
    try {
      const response = await fetch(`/api/incidents/${details.event_id}/ai-apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          changes,
          confirmed: true,
          configuration_fingerprint: details.configuration_fingerprint || "",
          recommendation_proof: details.recommendation_proof || "",
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Unable to apply the reviewed settings");
      setMessages((current) => {
        const next = current.map((message) => message.id !== messageId ? message : {
          ...message,
          evidence: (message.evidence || []).map((item) => item.id !== evidence.id ? item : {
            ...item,
            details: { ...item.details, can_apply: false, applied: payload.applied || [] },
          }),
        });
        const followUp = payload.follow_up;
        if (!followUp?.message) return next;
        return [...next, {
          id: `assistant-apply-${Date.now()}`,
          role: "assistant",
          content: followUp.message,
          evidence: [],
          suggestions: followUp.suggestions || [],
          actions: followUp.actions || [],
          reasoningTier: "fast",
          model: "",
        }].slice(-30);
      });
    } catch (applyError) {
      setError({ message: applyError.message || "Unable to apply the reviewed settings", kind: "apply" });
    } finally {
      setApplyingEvidenceId("");
    }
  }

  return (
    <>
      <div className="assistant-launcher-wrap">
        <button ref={launcherRef} type="button" className={`assistant-launcher ${open ? "open" : ""}`} onClick={() => { dismissCoach(); setOpenValue(open ? "false" : "true"); }} aria-label={open ? "Close SurvNG Assistant" : "Open SurvNG Assistant"} aria-expanded={open} aria-controls="survng-assistant" title="SurvNG Assistant">
          {open ? <X size={22} /> : <Sparkles size={22} />}
        </button>
        {showCoach && !open ? <div className="assistant-coach" role="status">
          <span>Ask about this camera or incident</span>
          <button type="button" onClick={dismissCoach} aria-label="Dismiss tip">Got it</button>
        </div> : null}
      </div>
      {open && compactAssistant ? <div className="assistant-backdrop" aria-hidden="true" onClick={() => setOpenValue("false")} /> : null}
      {open ? <aside id="survng-assistant" ref={drawerRef} className="assistant-drawer" role={modalOpen ? undefined : compactAssistant ? "dialog" : "complementary"} aria-modal={!modalOpen && compactAssistant || undefined} aria-hidden={modalOpen || undefined} inert={modalOpen || undefined} aria-labelledby="assistant-heading">
        <header className="assistant-head">
          <div><strong id="assistant-heading"><Sparkles size={17} /> SurvNG Assistant</strong><small>{status?.configured === false ? "AI provider needs setup" : "Answers from your cameras & incidents"}</small></div>
          <div>
            <button type="button" onClick={clearConversation} disabled={busy} aria-label="Start a new assistant conversation" title="New conversation"><Trash2 size={16} /></button>
            <button type="button" onClick={() => setOpenValue("false")} aria-label="Close SurvNG Assistant"><X size={17} /></button>
          </div>
        </header>
        <div className="assistant-context" aria-live="polite">
          <strong>Using</strong><span>{currentContextLabel}</span>
          {status ? <span title={status.fast_model === status.reasoning_model ? "Everyday and detailed questions use this model" : `Everyday: ${status.fast_model} · Detailed: ${status.reasoning_model}`}>{status.fast_model === status.reasoning_model ? "Configured AI" : "Everyday + detailed AI"}</span> : null}
        </div>
        <div className="assistant-body" ref={bodyRef} onScroll={(event) => { const body = event.currentTarget; followTranscriptRef.current = body.scrollHeight - body.scrollTop - body.clientHeight < 72; }}>
          {!messages.length ? <div className="assistant-welcome">
            <Sparkles size={26} />
            <strong>{welcomeCopy.title}</strong>
            <p>{welcomeCopy.body}</p>
            <div className="assistant-welcome-actions">
              {quickPrompts.map((suggestion) => <button type="button" key={suggestion} disabled={busy || status?.configured === false} onClick={() => sendMessage(suggestion)}>{suggestion}</button>)}
            </div>
          </div> : null}
          {messages.map((message) => <article className={`assistant-message ${message.role}`} key={message.id}>
            {message.role === "user" && message.context ? <small className="assistant-turn-context">Asked about {assistantContextLabel(message.context, (epoch) => formatDateTime(epoch, message.context.time_zone || timeZone))}</small> : null}
            {message.role === "assistant"
              ? <AssistantMessageText content={message.content} />
              : <div className="assistant-message-text">{message.content}</div>}
            {message.role === "assistant" && (message.model || message.reasoningTier) ? <small className="assistant-model-tier" title={message.model || undefined}>{message.reasoningTier === "deep" ? "Took a closer look" : "Quick answer"}</small> : null}
            {message.evidence?.length ? <div className="assistant-evidence">
              {message.evidence.map((item) => <div key={item.id} data-evidence-id={item.id} className={`assistant-evidence-card ${item.details ? "has-details" : ""}`} tabIndex={-1}>
                {item.image_url ? <a className="assistant-evidence-image" href={appUrl(assistantEvidenceHref(item))} aria-label={`Open ${item.title || "incident"}`} title="Open incident"><img src={appUrl(item.image_url)} alt={item.title || "Incident evidence"} loading="lazy" /></a> : null}
                {assistantEvidenceHref(item) ? <a href={appUrl(assistantEvidenceHref(item))}><span title={item.id}>Open in SurvNG</span><strong>{item.title}</strong><small>{item.summary}</small></a> : <div className="assistant-evidence-summary"><span>Evidence</span><strong>{item.title}</strong><small>{item.summary}</small></div>}
                {item.details?.timeline ? <div className="assistant-timeline">
                  {item.details.timeline.matches?.length ? item.details.timeline.matches.map((match) => <a className="assistant-timeline-link" href={appUrl(assistantIncidentHref(match.event_id))} key={match.event_id} title="Open incident"><span>{formatDateTime(match.start_at, timeZone)}</span><strong>{match.camera_id}</strong><small>{({ confirmed_identity: "Confirmed face", automatic_identity: "Automatic face match", possible_identity: "Possible face", appearance_similarity: `Visually similar ${match.appearance_similarity != null ? `${Math.round(Number(match.appearance_similarity) * 100)}%` : "appearance"}`, context_candidate: "Nearby matching class" })[match.match_strength] || "Possible connection"}</small></a>) : <small>No related incidents were found in this time window.</small>}
                  <p>{item.details.timeline.limitations?.[3]}</p>
                </div> : null}
                {item.details?.media_export ? <div className={`assistant-media-export ${item.details.media_export.status}`}>
                  <div><strong>{item.details.media_export.phase || item.details.media_export.status}</strong><span>{Math.round(Number(item.details.media_export.progress) || 0)}%</span></div>
                  <i><b style={{ width: `${Math.max(0, Math.min(100, Number(item.details.media_export.progress) || 0))}%` }} /></i>
                  {item.details.media_export.error ? <small>{item.details.media_export.error}</small> : null}
                  <a className="assistant-export-download" href={appUrl("/exports")}>View in Exports</a>
                  {item.details.media_export.status === "completed" && item.details.media_export.download_url ? <a className="assistant-export-download" href={appUrl(item.details.media_export.download_url)}><Download size={14} />Download MP4</a> : null}
                </div> : null}
                {item.details?.advice ? <div className="assistant-visual-review">
                  <div><strong>{assistantVisualVerdicts[item.details.advice.verdict] || "The image is inconclusive"}</strong><span>{Math.round(Number(item.details.advice.confidence || 0) * 100)}%</span></div>
                  <p>{item.details.advice.summary}</p>
                  {item.details.advice.visible_subjects?.length ? <small>Visible in this image: {item.details.advice.visible_subjects.join(", ")}</small> : null}
                  <dl>
                    <div><dt>Object recognition</dt><dd>{assistantDetectorAssessments[item.details.advice.detector_assessment] || assistantDetectorAssessments.uncertain}</dd></div>
                    <div><dt>Follow-up tracking</dt><dd>{assistantTrackingAssessments[item.details.advice.tracking_assessment] || assistantTrackingAssessments.uncertain}</dd></div>
                  </dl>
                  {item.details.proposals?.length ? <div className="assistant-proposals">
                    {item.details.proposals.map((proposal) => <div key={`${proposal.scope}-${proposal.setting}`}>
                      <strong>This camera · {assistantSettingLabels[proposal.setting] || String(proposal.setting).replaceAll("_", " ")}</strong>
                      <span><code>{String(proposal.current)}</code><ArrowRight size={13} /><code>{String(proposal.proposed)}</code></span>
                      <small>{proposal.reason}</small>
                    </div>)}
                  </div> : <small>No bounded setting changes recommended from this image.</small>}
                  {item.details.applied?.length ? <div className="assistant-applied"><Check size={14} /> Applied after confirmation</div> : null}
                  {item.details.can_apply && !item.details.applied?.length ? <button type="button" className="assistant-apply" disabled={Boolean(applyingEvidenceId)} onClick={(event) => openApplyReview(message.id, item, event.currentTarget)}>{applyingEvidenceId === item.id ? "Applying…" : "Apply proposed changes"}</button> : null}
                  {item.details.proposals?.length && !item.details.can_apply && !item.details.applied?.length ? <small>Enable “Allow confirmed changes” in Admin to apply these proposals.</small> : null}
                </div> : null}
                {item.details?.next_actions?.length ? renderAssistantActions(item.details.next_actions, message.id) : null}
              </div>)}
            </div> : null}
            {message.actions?.length ? renderAssistantActions(message.actions, message.id) : null}
            {message.suggestions?.length ? <div className="assistant-suggestions">
              {message.suggestions.map((suggestion) => <button type="button" key={suggestion} onClick={() => sendMessage(suggestion)}>{suggestion}</button>)}
            </div> : null}
          </article>)}
          {busy ? <div className="assistant-thinking" aria-live="polite"><span /><span /><span /> {thinkingStages[thinkingStage] || thinkingStages[0]}{busy ? <button type="button" className="assistant-stop" onClick={cancelInFlight}>Stop</button> : null}</div> : null}
          {error ? <div className="assistant-error" role="alert"><CircleAlert size={15} /><span>{error.message}</span>{error.kind === "request" ? <button type="button" onClick={() => sendMessage(error.content, error.context, { appendUser: false })}>Retry</button> : error.kind === "status" ? <button type="button" onClick={() => void loadAssistantStatus()}>Retry</button> : null}</div> : null}
          {status && !status.configured ? <div className="assistant-error" role="alert"><CircleAlert size={15} /><span>Configure and enable the AI provider to use the assistant.</span><a href={appUrl("/admin?section=general&subsection=mqtt&detail=ai-provider#ai-provider-settings")}>Open AI Provider settings</a></div> : null}
        </div>
        {messages.length ? <div className="assistant-quick-actions" role="group" aria-label={`Suggestions for ${currentContextLabel}`}>
          {quickPrompts.map((suggestion) => <button type="button" key={suggestion} disabled={busy} onClick={() => sendMessage(suggestion)}>{suggestion}</button>)}
        </div> : null}
        <form className="assistant-compose" onSubmit={(event) => { event.preventDefault(); sendMessage(); }}>
          <textarea ref={composerRef} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(); } }} placeholder={composerPlaceholder} rows="2" maxLength="8000" disabled={busy || status?.configured === false} />
          <button type="submit" disabled={busy || !draft.trim()}>Send</button>
        </form>
      </aside> : null}
      {applyReview ? <div ref={applyDialogRef} className="assistant-apply-dialog" role="dialog" aria-modal="true" aria-labelledby="assistant-apply-title">
        <div>
          <h2 id="assistant-apply-title">Apply proposed changes?</h2>
          <p>{applyReview.evidence.details?.camera_id || "This camera"} will restart after these settings are applied.</p>
          <ul>{(applyReview.evidence.details?.proposals || []).map((change) => <li key={change.setting}><strong>{assistantSettingLabels[change.setting] || change.setting}</strong><span>{String(change.current)} → {String(change.proposed)}</span></li>)}</ul>
          <footer><button type="button" onClick={() => closeApplyReview(false)}>Cancel</button><button type="button" onClick={() => { const review = applyReview; closeApplyReview(true); void applyVisualProposals(review.messageId, review.evidence); }}>Confirm and apply</button></footer>
        </div>
      </div> : null}
      {confirmPost ? <div ref={confirmDialogRef} className="assistant-apply-dialog" role="dialog" aria-modal="true" aria-labelledby="assistant-confirm-post-title">
        <div>
          <h2 id="assistant-confirm-post-title">{assistantConfirmPostKind(confirmPost.action.path) === "apply_review" ? "Apply Camera Advisor recommendations?" : "Confirm Camera Advisor step?"}</h2>
          <p>{confirmPost.action.confirm || confirmPost.action.label}</p>
          {confirmPost.action.proposals?.length ? <ul>{confirmPost.action.proposals.map((change) => <li key={change.setting}><strong>{assistantSettingLabels[change.setting] || change.setting}</strong><span>{String(change.current)} → {String(change.proposed)}</span></li>)}</ul> : null}
          <footer>
            <button type="button" onClick={() => closeConfirmPost(false)}>Cancel</button>
            <button type="button" onClick={() => {
              const pending = confirmPost;
              closeConfirmPost(true);
              void executeConfirmPost(pending.action, pending.messageId);
            }}>Confirm</button>
          </footer>
        </div>
      </div> : null}
    </>
  );
}

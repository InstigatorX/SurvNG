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
import { assistantEvidenceHref, assistantIncidentHref } from "../assistantNavigation.mjs";
import { appUrl, fetch } from "../shared/api.js";
import { formatDateTime } from "../shared/format.js";
import { useStoredState, useViewportQuery } from "../shared/hooks.js";

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

export function AssistantPanel({ pageContext, timeZone }) {
  const [openValue, setOpenValue] = useStoredState("survng.assistantOpen.v1", "false");
  const open = openValue === "true";
  const [messages, setMessages] = useState(readAssistantMessages);
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [applyingEvidenceId, setApplyingEvidenceId] = useState("");
  const [error, setError] = useState(null);
  const [applyReview, setApplyReview] = useState(null);
  const bodyRef = useRef(null);
  const drawerRef = useRef(null);
  const applyDialogRef = useRef(null);
  const applyReturnFocusRef = useRef(null);
  const applyEvidenceRef = useRef(null);
  const launcherRef = useRef(null);
  const composerRef = useRef(null);
  const followTranscriptRef = useRef(true);
  const returnFocusRef = useRef(null);
  const compactAssistant = useViewportQuery("(max-width: 1279px)");
  const currentContext = useMemo(() => snapshotAssistantContext(pageContext, timeZone), [pageContext, timeZone]);
  const currentContextLabel = assistantContextLabel(currentContext, (epoch) => formatDateTime(epoch, timeZone));
  const quickPrompts = assistantContextPrompts(currentContext).slice(0, 3);
  const activeExportIds = [...new Set(messages.flatMap((message) =>
    (message.evidence || [])
      .map((item) => item.details?.media_export)
      .filter((job) => job?.id && ["queued", "running", "cancelling"].includes(job.status))
      .map((job) => job.id)
  ))].sort().join(",");

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

  useEffect(() => {
    if (!open || !activeExportIds) return undefined;
    let cancelled = false;
    const ids = activeExportIds.split(",").filter(Boolean);
    async function refreshExports() {
      if (document.hidden) return;
      const updates = await Promise.all(ids.map(async (id) => {
        try {
          const response = await fetch(`/api/exports/${encodeURIComponent(id)}`);
          return response.ok ? await response.json() : null;
        } catch {
          return null;
        }
      }));
      if (cancelled) return;
      const byId = new Map(updates.filter(Boolean).map((job) => [job.id, job]));
      if (!byId.size) return;
      setMessages((current) => current.map((message) => ({
        ...message,
        evidence: (message.evidence || []).map((item) => {
          const previous = item.details?.media_export;
          const update = previous?.id ? byId.get(previous.id) : null;
          if (!update) return item;
          return {
            ...item,
            details: {
              ...item.details,
              media_export: {
                ...previous,
                status: update.status,
                phase: update.phase,
                progress: update.progress,
                error: update.error,
                output_name: update.output_name,
                size_bytes: update.size_bytes,
                download_url: update.download_url,
              },
            },
          };
        }),
      })));
    }
    refreshExports();
    const timer = window.setInterval(refreshExports, 2000);
    const onVisibility = () => { if (!document.hidden) void refreshExports(); };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [activeExportIds, open]);

  useEffect(() => {
    if (!open) return undefined;
    function handleAssistantKeyboard(event) {
      if (event.key === "Escape") {
        if (applyReview) closeApplyReview(false);
        else setOpenValue("false");
        return;
      }
      if (!compactAssistant || event.key !== "Tab" || applyReview) return;
      const controls = [...(drawerRef.current?.querySelectorAll('button:not([disabled]), a[href], textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])') || [])];
      if (!controls.length) return;
      const first = controls[0];
      const last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
    window.addEventListener("keydown", handleAssistantKeyboard);
    return () => window.removeEventListener("keydown", handleAssistantKeyboard);
  }, [applyReview, compactAssistant, open, setOpenValue]);

  useEffect(() => {
    if (!applyReview) return undefined;
    const dialog = applyDialogRef.current;
    const controls = [...(dialog?.querySelectorAll('button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])') || [])];
    controls[0]?.focus();
    function trapApplyFocus(event) {
      if (event.key !== "Tab" || !controls.length) return;
      const first = controls[0];
      const last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
    dialog?.addEventListener("keydown", trapApplyFocus);
    return () => dialog?.removeEventListener("keydown", trapApplyFocus);
  }, [applyReview]);

  function clearConversation() {
    if (messages.length && !window.confirm("Start a new assistant conversation? The current conversation will be cleared.")) return;
    setMessages([]);
    setError(null);
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
    setBusy(true);
    try {
      const response = await fetch("/api/assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
        reasoningTier: payload.reasoning_tier || "fast",
        model: payload.model || "",
        context: submittedContext,
      }].slice(-30));
    } catch (sendError) {
      const statusCode = Number(String(sendError.message || "").match(/\((\d+)\)/)?.[1]);
      const message = statusCode === 429 ? "The AI provider is busy. Wait a moment, then retry."
        : [502, 503, 504].includes(statusCode) ? "The AI provider is temporarily unavailable. Retry when it recovers."
          : sendError.message || "Assistant request failed";
      setError({ message, kind: "request", content, context: submittedContext });
    } finally {
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
      setMessages((current) => current.map((message) => message.id !== messageId ? message : {
        ...message,
        evidence: (message.evidence || []).map((item) => item.id !== evidence.id ? item : {
          ...item,
          details: { ...item.details, can_apply: false, applied: payload.applied || [] },
        }),
      }));
    } catch (applyError) {
      setError({ message: applyError.message || "Unable to apply the reviewed settings", kind: "apply" });
    } finally {
      setApplyingEvidenceId("");
    }
  }

  return (
    <>
      <button ref={launcherRef} type="button" className={`assistant-launcher ${open ? "open" : ""}`} onClick={() => setOpenValue(open ? "false" : "true")} aria-label={open ? "Close SurvNG Assistant" : "Open SurvNG Assistant"} aria-expanded={open} aria-controls="survng-assistant" title="SurvNG Assistant">
        {open ? <X size={22} /> : <Sparkles size={22} />}
      </button>
      {open && compactAssistant ? <div className="assistant-backdrop" aria-hidden="true" onClick={() => setOpenValue("false")} /> : null}
      {open ? <aside id="survng-assistant" ref={drawerRef} className="assistant-drawer" role={applyReview ? undefined : compactAssistant ? "dialog" : "complementary"} aria-modal={!applyReview && compactAssistant || undefined} aria-hidden={applyReview || undefined} inert={applyReview || undefined} aria-labelledby="assistant-heading">
        <header className="assistant-head">
          <div><strong id="assistant-heading"><Sparkles size={17} /> SurvNG Assistant</strong><small>{status?.configured === false ? "AI provider needs setup" : "Grounded in SurvNG evidence"}</small></div>
          <div>
            <button type="button" onClick={clearConversation} disabled={busy} aria-label="Start a new assistant conversation" title="New conversation"><Trash2 size={16} /></button>
            <button type="button" onClick={() => setOpenValue("false")} aria-label="Close SurvNG Assistant"><X size={17} /></button>
          </div>
        </header>
        <div className="assistant-context" aria-live="polite">
          <strong>Current view</strong><span>{currentContextLabel}</span>
          {status ? <span title={status.fast_model === status.reasoning_model ? "Everyday and detailed questions use this model" : `Everyday: ${status.fast_model} · Detailed: ${status.reasoning_model}`}>{status.fast_model === status.reasoning_model ? status.fast_model : "Everyday + detailed AI"}</span> : null}
        </div>
        <div className="assistant-body" ref={bodyRef} onScroll={(event) => { const body = event.currentTarget; followTranscriptRef.current = body.scrollHeight - body.scrollTop - body.clientHeight < 72; }}>
          {!messages.length ? <div className="assistant-welcome">
            <Sparkles size={26} />
            <strong>What would you like to know?</strong>
            <p>I can search incidents, trace related activity, review a selected incident, inspect camera health, explain settings, and create recording exports or timelapses.</p>
          </div> : null}
          {messages.map((message) => <article className={`assistant-message ${message.role}`} key={message.id}>
            {message.role === "user" && message.context ? <small className="assistant-turn-context">Asked about {assistantContextLabel(message.context, (epoch) => formatDateTime(epoch, message.context.time_zone || timeZone))}</small> : null}
            <div className="assistant-message-text">{message.content}</div>
            {message.role === "assistant" && (message.model || message.reasoningTier) ? <small className="assistant-model-tier">{message.reasoningTier === "deep" ? "Detailed analysis" : "Quick answer"}{message.model ? ` · ${message.model}` : ""}</small> : null}
            {message.evidence?.length ? <div className="assistant-evidence">
              {message.evidence.map((item) => <div key={item.id} className={`assistant-evidence-card ${item.details ? "has-details" : ""}`} tabIndex={-1}>
                {item.image_url ? <a className="assistant-evidence-image" href={appUrl(assistantEvidenceHref(item))} aria-label={`Open ${item.title || "incident"}`} title="Open incident"><img src={appUrl(item.image_url)} alt={item.title || "Incident evidence"} loading="lazy" /></a> : null}
                {assistantEvidenceHref(item) ? <a href={appUrl(assistantEvidenceHref(item))}><span title={item.id}>Open evidence</span><strong>{item.title}</strong><small>{item.summary}</small></a> : <div className="assistant-evidence-summary"><span>Evidence</span><strong>{item.title}</strong><small>{item.summary}</small></div>}
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
              </div>)}
            </div> : null}
            {message.suggestions?.length ? <div className="assistant-suggestions">
              {message.suggestions.map((suggestion) => <button type="button" key={suggestion} onClick={() => sendMessage(suggestion)}>{suggestion}</button>)}
            </div> : null}
          </article>)}
          {busy ? <div className="assistant-thinking"><span /><span /><span /> Gathering SurvNG evidence…</div> : null}
          {error ? <div className="assistant-error" role="alert"><CircleAlert size={15} /><span>{error.message}</span>{error.kind === "request" ? <button type="button" onClick={() => sendMessage(error.content, error.context, { appendUser: false })}>Retry</button> : error.kind === "status" ? <button type="button" onClick={() => void loadAssistantStatus()}>Retry</button> : null}</div> : null}
          {status && !status.configured ? <div className="assistant-error" role="alert"><CircleAlert size={15} /><span>Configure and enable the AI provider to use the assistant.</span><a href={appUrl("/admin?section=general&subsection=detection&detail=ai-provider")}>Open AI Provider settings</a></div> : null}
        </div>
        <div className="assistant-quick-actions" role="group" aria-label={`Suggestions for ${currentContextLabel}`}>
          {quickPrompts.map((suggestion) => <button type="button" key={suggestion} disabled={busy} onClick={() => sendMessage(suggestion)}>{suggestion}</button>)}
        </div>
        <form className="assistant-compose" onSubmit={(event) => { event.preventDefault(); sendMessage(); }}>
          <textarea ref={composerRef} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(); } }} placeholder="Ask about SurvNG…" rows="2" maxLength="8000" disabled={busy || status?.configured === false} />
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
    </>
  );
}

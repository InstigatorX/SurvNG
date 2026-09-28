import React, { useCallback, useEffect, useState } from "react";
import { nextTabId } from "../adminWorkspace.mjs";
import { tuneupOutcome, tuneupValue } from "../detectionTuneup.mjs";
import { formatDateTime } from "../shared/format.js";
import { fetch } from "../shared/api.js";

const SECTIONS = ["briefing", "monitoring", "history"];

const AUTOMATIC_CLASS_CATALOG = [
  { id: "motion_sensitivity", label: "Motion sensitivity", detail: "How readily motion qualifies, including stationary-object tolerance." },
  { id: "motion_analysis", label: "Motion analysis", detail: "Frame size, sample rate, and the timing windows around a trigger." },
  { id: "visual_backup", label: "Visual backup", detail: "Rescue score, persistence, warmup, cooldown, and how often a rescue may fire." },
  { id: "borderline_rescue", label: "Borderline rescue", detail: "Whether near-threshold motion is checked again, and how close it must be." },
  { id: "tracking", label: "Tracking", detail: "Sample rate, how long a track is kept, how many cameras can track at once, and re-identification." },
  { id: "restore_inherit", label: "Restore inheritance", detail: "Put a camera back on the system value when that does not change what is in effect." },
];

function classAllowed(classId, allowed) {
  const selected = new Set(allowed || []);
  if (selected.has(classId)) return true;
  return classId === "visual_backup" && selected.has("visual_backup_cooldown");
}

async function readPayload(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === "string"
      ? detail
      : Array.isArray(detail)
        ? detail.map((item) => item?.msg || item).filter(Boolean).join("; ")
        : "";
    throw new Error(message || `System review could not be updated (${response.status})`);
  }
  return payload;
}

function visibleNotes(notes) {
  return (notes || []).filter((note) => note?.kind !== "grace" && !String(note?.text || "").includes("45-second"));
}

const FINDING_VERDICTS = {
  likely_miss: "Likely miss",
  likely_false_alarm: "Likely false alarm",
  likely_misclassification: "Likely wrong label",
};

function reviewFindings(result) {
  const reports = result?.camera_reports || {};
  return (result?.camera_summaries || []).flatMap((summary) => {
    const samples = (reports[summary.camera_id]?.samples || []).filter((sample) => FINDING_VERDICTS[sample.verdict] && sample.summary);
    return samples.map((sample) => ({
      cameraId: summary.camera_id,
      cameraName: summary.camera_name || summary.camera_id,
      verdict: sample.verdict,
      summary: sample.summary,
    }));
  });
}

function settingLabel(setting) {
  return String(setting || "setting").split(".").pop().replaceAll("_", " ");
}

function reviewProgress(briefing, cameras) {
  const progress = briefing?.result?.progress || {};
  const total = Number(progress.total) || 0;
  const completed = Number(progress.completed) || 0;
  const cameraName = progress.camera_name || cameras.find((item) => item.id === progress.camera_id)?.name || "";
  const phase = progress.phase || (briefing?.status === "queued" ? "Waiting to start" : briefing?.status === "cancelling" ? "Cancelling" : "");
  const imagesTotal = Number(progress.images_total) || 0;
  const imagesDone = Number(progress.images_done) || 0;
  const parts = [];
  if (phase && cameraName) parts.push(`${phase} · ${cameraName}`);
  else if (phase) parts.push(phase);
  else if (cameraName) parts.push(cameraName);
  if (phase && total) parts.push(`camera ${Math.min(Number(progress.camera_index) || completed + (cameraName ? 1 : 0), total)} of ${total}`);
  else if (total) parts.push(`Finished ${completed} of ${total} cameras`);
  if (imagesTotal) parts.push(`image ${Math.min(imagesDone, imagesTotal)} of ${imagesTotal}`);
  const skipped = Object.keys(briefing?.result?.camera_errors || {}).length;
  if (skipped) parts.push(`${skipped} skipped`);
  const imageShare = imagesTotal ? Math.min(1, imagesDone / imagesTotal) : 0;
  const currentShare = cameraName && total && completed < total ? imageShare : 0;
  const percent = total ? Math.round(Math.min(1, (completed + currentShare) / total) * 100) : 0;
  return { text: parts.join(" · ") || "Starting", percent };
}

function nextLabel(review, timeZone) {
  if (!review || review.cadence === "off") return "Not scheduled";
  if (review.cadence === "daily") {
    return review.next_daily_at ? `Next daily ${formatDateTime(review.next_daily_at, timeZone)}` : "Daily";
  }
  const weekly = review.next_weekly_at ? formatDateTime(review.next_weekly_at, timeZone) : "soon";
  return `Next weekly ${weekly}`;
}

export function SystemReviewPanel({ cameras = [], timeZone, onCommandBarChange = null }) {
  const [review, setReview] = useState(null);
  const [cadence, setCadence] = useState("weekly");
  const [section, setSection] = useState("briefing");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const load = useCallback(async () => {
    const response = await fetch("/api/system-review");
    const payload = await readPayload(response);
    setReview(payload);
    if (payload?.cadence) setCadence(payload.cadence);
    setError("");
    return payload;
  }, []);

  const briefing = review?.briefing || null;
  const result = briefing?.result || {};
  const suggestions = (result.recommendations || []).filter((item) => !item.dismissed);
  const findings = reviewFindings(result);
  const running = ["queued", "running", "cancelling"].includes(briefing?.status);
  const progress = running ? reviewProgress(briefing, cameras) : null;
  const monitors = (review?.change_sets || []).filter((item) => item.action === "apply" && ["collecting", "reviewing", "evaluated", "evaluation_failed"].includes(item.status));

  useEffect(() => {
    let cancelled = false;
    load().catch((loadError) => {
      if (!cancelled) setError(loadError.message || "System review could not be loaded");
    });
    return () => { cancelled = true; };
  }, [load]);

  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setInterval(() => {
      load().catch((loadError) => setError(loadError.message || "System review could not be loaded"));
    }, 4000);
    return () => window.clearInterval(timer);
  }, [running, load]);

  async function saveSettings(next) {
    const previous = cadence;
    if (next.cadence) setCadence(next.cadence);
    setBusy(true);
    try {
      const payload = await readPayload(await fetch("/api/system-review/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cadence: next.cadence ?? cadence,
          automatic_classes: next.automatic_classes ?? review?.automatic_classes ?? [],
        }),
      }));
      setReview(payload);
      if (payload?.cadence) setCadence(payload.cadence);
      setError("");
    } catch (saveError) {
      setCadence(previous);
      setError(saveError.message);
    } finally {
      setBusy(false);
    }
  }

  async function reviewNow() {
    setBusy(true);
    try {
      await readPayload(await fetch("/api/system-review/run?review_pass=weekly", { method: "POST" }));
      await load();
      setSection("briefing");
    } catch (runError) {
      setError(runError.message);
    } finally {
      setBusy(false);
    }
  }

  async function applySuggestion(item) {
    if (!briefing?.id || !briefing.configuration_fingerprint) return;
    setBusy(true);
    try {
      await readPayload(await fetch(`/api/calibration/runs/${briefing.id}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recommendation_ids: [item.id],
          confirmed: true,
          configuration_fingerprint: briefing.configuration_fingerprint,
        }),
      }));
      await load();
      setSection("monitoring");
    } catch (applyError) {
      setError(applyError.message);
    } finally {
      setBusy(false);
    }
  }

  async function dismissSuggestion(item) {
    setBusy(true);
    try {
      setReview(await readPayload(await fetch(`/api/system-review/runs/${briefing.id}/dismiss`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recommendation_id: item.id }),
      })));
      setError("");
    } catch (dismissError) {
      setError(dismissError.message);
    } finally {
      setBusy(false);
    }
  }

  function toggleAutomatic(classId, enabled) {
    const current = new Set(review?.automatic_classes || []);
    if (classId === "visual_backup") current.delete("visual_backup_cooldown");
    if (enabled) current.add(classId);
    else current.delete(classId);
    void saveSettings({ automatic_classes: [...current] });
  }

  useEffect(() => {
    if (!onCommandBarChange) return undefined;
    return () => onCommandBarChange(null);
  }, [onCommandBarChange]);

  useEffect(() => {
    if (!onCommandBarChange) return;
    onCommandBarChange({
      cadence,
      onCadence: (value) => { void saveSettings({ cadence: value }); },
      nextLabel: nextLabel(review, timeZone),
      progressLabel: progress?.text || "",
      error,
      reviewNow: () => { void reviewNow(); },
      reviewBusy: busy || running,
      starting: busy && !running,
      refresh: () => { void load().catch((loadError) => setError(loadError.message || "System review could not be loaded")); },
    });
  }, [busy, cadence, error, load, onCommandBarChange, review, running, timeZone]);

  function onTabsKeyDown(event) {
    const next = nextTabId(SECTIONS, section, event.key);
    if (!next) return;
    event.preventDefault();
    setSection(next);
    window.requestAnimationFrame(() => document.getElementById(`system-review-tab-${next}`)?.focus());
  }

  return (
    <div className="detection-settings subsection-workspace system-review">
      <nav className="admin-section-tabs camera-section-tabs detection-subsection-tabs" role="tablist" aria-label="System review sections" onKeyDown={onTabsKeyDown}>
        <button id="system-review-tab-briefing" type="button" role="tab" aria-selected={section === "briefing"} className={section === "briefing" ? "active" : ""} onClick={() => setSection("briefing")}>Briefing</button>
        <button id="system-review-tab-monitoring" type="button" role="tab" aria-selected={section === "monitoring"} className={section === "monitoring" ? "active" : ""} onClick={() => setSection("monitoring")}>Monitoring</button>
        <button id="system-review-tab-history" type="button" role="tab" aria-selected={section === "history"} className={section === "history" ? "active" : ""} onClick={() => setSection("history")}>History</button>
      </nav>
      <div className="detection-settings-content" role="tabpanel">
        {error ? <div className="error-banner" role="alert">{error}</div> : null}
        {section === "briefing" ? (
          <>
            <section className="telemetry-section">
              <div className="telemetry-section-head">
                <div>
                  <h3>Latest briefing</h3>
                  <p>{running ? progress.text : result.summary || "No briefing yet. Use Review now, or wait for the next scheduled pass."}</p>
                </div>
                {briefing?.completed_at ? <span>{formatDateTime(briefing.completed_at, timeZone)}</span> : null}
              </div>
              {running ? <div className="motion-ai-review-progress system-review-progress" role="status" aria-label={progress.text}><div><i style={{ width: `${progress.percent}%` }} /></div></div> : null}
              <div className="telemetry-diagnostic-controls">
                <label><span>Cadence</span>
                  <select aria-label="System review cadence" value={cadence} onChange={(event) => void saveSettings({ cadence: event.target.value })}>
                    <option value="weekly">Weekly</option>
                    <option value="daily">Daily</option>
                    <option value="off">Off</option>
                  </select>
                </label>
                <span>{nextLabel(review, timeZone)}</span>
              </div>
              {visibleNotes(result.site_notes).length ? <div className="telemetry-health-event-list">{visibleNotes(result.site_notes).map((note) => <div key={note.text}><span>{note.text}</span></div>)}</div> : null}
            </section>
            {findings.length ? <section className="telemetry-section">
              <div className="telemetry-section-head"><div><h3>What the samples showed</h3><p>These are the samples that were not a clean match. A suggestion appears only when the same setting change is repeated.</p></div></div>
              <div className="telemetry-diagnostic-list system-review-findings">{findings.map((item) => (
                <article className="telemetry-diagnostic-card" key={`${item.cameraId}-${item.verdict}-${item.summary}`}>
                  <div>
                    <strong>{item.cameraName}</strong>
                    <span>{FINDING_VERDICTS[item.verdict]} · {item.summary}</span>
                  </div>
                </article>
              ))}</div>
            </section> : null}
            <section className="telemetry-section">
              <div className="telemetry-section-head"><div><h3>Suggestions</h3><p>Nothing is written until you apply it. Allowed classes apply on the weekly run and roll back if the monitored outcome is worse.</p></div></div>
              {review && !review.apply_enabled ? <p className="telemetry-diagnostic-empty">Applying stays off until AI recommendation changes are allowed in Server settings.</p> : null}
              {suggestions.length ? <div className="system-review-list">{suggestions.map((item) => {
                const camera = cameras.find((entry) => entry.id === item.camera_id);
                const allowed = Boolean(item.automatic_class && classAllowed(item.automatic_class, review?.automatic_classes));
                return (
                  <article key={item.id}>
                    <div>
                      <strong>{settingLabel(item.setting)}</strong>
                      <span>{camera?.name || (item.camera_id ? item.camera_id : "Every inheriting camera")}</span>
                    </div>
                    <p>{tuneupValue(item.current ?? item.current_effective)} → {tuneupValue(item.proposed)}</p>
                    {item.expected_benefit ? <small>{item.expected_benefit}</small> : null}
                    {item.downside ? <small>{item.downside}</small> : null}
                    {item.compute_impact ? <small>Compute: {item.compute_impact}</small> : null}
                    <div className="button-row">
                      <button type="button" className="primary" disabled={busy || !review?.apply_enabled} onClick={() => void applySuggestion(item)}>Apply</button>
                      <button type="button" disabled={busy} onClick={() => void dismissSuggestion(item)}>Dismiss</button>
                      {item.automatic_class ? <label className="compact-toggle"><input type="checkbox" checked={allowed} disabled={busy} onChange={(event) => toggleAutomatic(item.automatic_class, event.target.checked)} /><span>Allow this class automatically</span></label> : null}
                    </div>
                  </article>
                );
              })}</div> : <p className="telemetry-diagnostic-empty">{findings.length ? "No setting was repeated often enough to suggest a change." : "No suggestions in this briefing."}</p>}
            </section>
            <section className="telemetry-section">
              <div className="telemetry-section-head"><div><h3>Automatic classes</h3><p>The review can tune motion and tracking across the site. Turn a class on to apply it during the weekly run. Confidence, zone requirements, and anything that changes what can open an incident stay manual.</p></div></div>
              <div className="system-review-toggles">
                {AUTOMATIC_CLASS_CATALOG.map((item) => (
                  <label className="compact-toggle system-review-class" key={item.id}>
                    <input type="checkbox" checked={classAllowed(item.id, review?.automatic_classes)} disabled={busy} onChange={(event) => toggleAutomatic(item.id, event.target.checked)} />
                    <span><strong>{item.label}</strong><small>{item.detail}</small></span>
                  </label>
                ))}
              </div>
            </section>
          </>
        ) : null}
        {section === "monitoring" ? (
          <section className="telemetry-section">
            <div className="telemetry-section-head"><div><h3>After a change</h3><p>Applied suggestions stay in the existing monitoring window.</p></div></div>
            {monitors.length ? <div className="telemetry-diagnostic-list">{monitors.map((item) => {
              const [outcome, tone] = tuneupOutcome(item);
              return (
                <article className="telemetry-diagnostic-card" key={item.id}>
                  <div>
                    <strong>{item.status === "collecting" ? "Monitoring changes" : outcome}</strong>
                    <span>{formatDateTime(item.created_at, timeZone)} · {item.evaluation?.summary || `${item.changes?.length || 0} changes`}</span>
                  </div>
                  <em className={tone}>{String(item.status).replaceAll("_", " ")}</em>
                </article>
              );
            })}</div> : <p className="telemetry-diagnostic-empty">No changes from these reviews are being monitored.</p>}
          </section>
        ) : null}
        {section === "history" ? (
          <section className="telemetry-section">
            <div className="telemetry-section-head"><div><h3>History</h3><p>Daily evidence passes and weekly sample reviews.</p></div></div>
            {(review?.history || []).length ? <div className="telemetry-diagnostic-list">{review.history.map((item) => (
              <article className="telemetry-diagnostic-card" key={item.id}>
                <div>
                  <strong>{item.mode === "system_weekly" ? "Weekly review" : "Daily evidence"}</strong>
                  <span>{formatDateTime(item.created_at, timeZone)} · {item.id === briefing?.id && progress ? progress.text : item.status}{item.summary && !(item.id === briefing?.id && progress) ? ` · ${item.summary}` : ""}{item.error ? ` · ${item.error}` : ""}</span>
                </div>
              </article>
            ))}</div> : <p className="telemetry-diagnostic-empty">No reviews yet.</p>}
          </section>
        ) : null}
      </div>
    </div>
  );
}

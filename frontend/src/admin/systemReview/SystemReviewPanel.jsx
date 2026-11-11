import React, { useCallback, useEffect, useState } from "react";
import { formatDateTime } from "../../shared/format.js";
import { fetch } from "../../shared/api.js";
import { findingsByCamera } from "../../systemReviewView.mjs";
import { ChangeFeed } from "./ChangeFeed.jsx";
import { EvidenceStrip } from "./EvidenceStrip.jsx";
import { LeftAlone } from "./LeftAlone.jsx";
import { PastReviews } from "./PastReviews.jsx";
import { ReviewHeader } from "./ReviewHeader.jsx";
import { SuggestionCard } from "./SuggestionCard.jsx";

async function readPayload(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === "string"
      ? detail
      : Array.isArray(detail)
        ? detail.map((item) => item?.msg || item).filter(Boolean).join("; ")
        : typeof detail?.message === "string"
          ? detail.message
          : "";
    throw new Error(message || `System review could not be updated (${response.status})`);
  }
  return payload;
}

function reviewAdvisories(result) {
  const advisories = result?.advisories || {};
  const lines = [];
  if (advisories.object_safety && !String(advisories.object_safety).startsWith("No broad")) lines.push(advisories.object_safety);
  if (advisories.manual_review && !String(advisories.manual_review).startsWith("No pipeline")) lines.push(advisories.manual_review);
  return lines;
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
    return review.next_daily_at ? `Next daily check ${formatDateTime(review.next_daily_at, timeZone)}` : "Daily";
  }
  const weekly = review.next_weekly_at ? formatDateTime(review.next_weekly_at, timeZone) : "soon";
  return `Next weekly review ${weekly}`;
}

export function SystemReviewPanel({ cameras = [], timeZone, onCommandBarChange = null }) {
  const [review, setReview] = useState(null);
  const [pastRun, setPastRun] = useState(null);
  const [cadence, setCadence] = useState("weekly");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    const payload = await readPayload(await fetch("/api/system-review"));
    setReview(payload);
    if (payload?.cadence) setCadence(payload.cadence);
    setError("");
    return payload;
  }, []);

  const live = review?.briefing || null;
  const shown = pastRun || live;
  const result = shown?.result || {};
  const past = Boolean(pastRun && pastRun.id !== live?.id);
  const suggestions = (result.recommendations || []).filter((item) => !item.dismissed);
  const findings = findingsByCamera(result);
  const running = ["queued", "running", "cancelling"].includes(live?.status);
  const progress = running ? reviewProgress(live, cameras) : null;

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
      setPastRun(null);
      await load();
    } catch (runError) {
      setError(runError.message);
    } finally {
      setBusy(false);
    }
  }

  async function applySuggestion(item) {
    if (!shown?.id || !shown.configuration_fingerprint || past) return;
    setBusy(true);
    try {
      await readPayload(await fetch(`/api/calibration/runs/${shown.id}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recommendation_ids: [item.id],
          confirmed: true,
          configuration_fingerprint: shown.configuration_fingerprint,
        }),
      }));
      setPastRun(null);
      await load();
    } catch (applyError) {
      setError(applyError.message);
    } finally {
      setBusy(false);
    }
  }

  async function dismissSuggestion(item) {
    if (!shown?.id || past) return;
    setBusy(true);
    try {
      setReview(await readPayload(await fetch(`/api/system-review/runs/${shown.id}/dismiss`, {
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

  async function rollbackChangeSet(item) {
    const names = (item.changes || []).map((change) => `${change.label} on ${change.camera_name}`).join(", ");
    const count = item.changes?.length || 0;
    const prompt = count
      ? `Put ${count} setting${count === 1 ? "" : "s"} back? ${names}`
      : "Put the settings from this change back?";
    if (!window.confirm(prompt)) return;
    setBusy(true);
    try {
      await readPayload(await fetch(`/api/calibration/change-sets/${item.id}/rollback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmed: true }),
      }));
      await load();
      setError("");
    } catch (rollbackError) {
      setError(rollbackError.message);
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

  function toggleAll(enabled) {
    const ids = enabled ? (review?.automatic_class_catalog || []).map((item) => item.id) : [];
    void saveSettings({ automatic_classes: ids });
  }

  async function openPast(item) {
    if (!item?.id || item.id === live?.id) {
      setPastRun(null);
      return;
    }
    setBusy(true);
    try {
      setPastRun(await readPayload(await fetch(`/api/system-review/runs/${item.id}`)));
      setError("");
    } catch (openError) {
      setError(openError.message);
    } finally {
      setBusy(false);
    }
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

  return (
    <div className="detection-settings subsection-workspace system-review">
      <div className="detection-settings-content system-review-page">
        {error ? <div className="error-banner" role="alert">{error}</div> : null}
        <ReviewHeader
          glance={live?.result?.at_a_glance}
          completedAt={live?.completed_at ? formatDateTime(live.completed_at, timeZone) : ""}
          running={running}
          progress={progress}
          nextLabel={nextLabel(review, timeZone)}
          level={review?.automatic_level || "off"}
          catalog={review?.automatic_class_catalog || []}
          classes={review?.automatic_classes || []}
          busy={busy}
          onToggleAll={toggleAll}
          onToggleClass={toggleAutomatic}
        />
        {past ? (
          <p className="system-review-past-banner">
            Showing the review from {formatDateTime(shown?.created_at, timeZone)}.
            {" "}
            <button type="button" onClick={() => setPastRun(null)}>Back to the latest review</button>
          </p>
        ) : null}
        <section className="telemetry-section">
          <div className="telemetry-section-head">
            <div>
              <h3>Suggested changes</h3>
              <p>Nothing changes until you apply it, unless the switch above is on for that kind of change.</p>
            </div>
          </div>
          {review && !review.apply_enabled ? <p className="telemetry-diagnostic-empty">Applying stays off until AI recommendation changes are allowed in Server settings.</p> : null}
          {suggestions.length ? <div className="system-review-list">{suggestions.map((item) => (
            <SuggestionCard
              key={item.id}
              item={item}
              busy={busy}
              past={past}
              applyEnabled={Boolean(review?.apply_enabled)}
              onApply={applySuggestion}
              onDismiss={dismissSuggestion}
            />
          ))}</div> : <p className="telemetry-diagnostic-empty">{findings.length ? "No setting was repeated often enough to suggest a change." : "No suggestions in this review."}</p>}
        </section>
        {findings.length ? (
          <section className="telemetry-section">
            <div className="telemetry-section-head">
              <div>
                <h3>What the review looked at</h3>
                <p>Pictures that did not look like a clean match. A suggestion appears only when the same change shows up more than once.</p>
              </div>
            </div>
            <div className="system-review-findings">{findings.map((group) => (
              <article key={group.cameraId}>
                <strong>{group.cameraName}</strong>
                <EvidenceStrip samples={group.samples} />
              </article>
            ))}</div>
          </section>
        ) : null}
        <ChangeFeed changes={review?.changes || []} timeZone={timeZone} busy={busy} onRollback={rollbackChangeSet} />
        <LeftAlone
          notes={result.site_notes}
          excluded={result.excluded_cameras}
          advisories={reviewAdvisories(result)}
          guide={review?.setting_guide}
        />
        <PastReviews
          history={review?.history || []}
          activeId={shown?.id}
          timeZone={timeZone}
          busy={busy}
          onOpen={(item) => { void openPast(item); }}
        />
      </div>
    </div>
  );
}

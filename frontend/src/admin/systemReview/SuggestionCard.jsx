import React from "react";
import { adminWorkspaceSearch } from "../../adminWorkspace.mjs";
import { appUrl } from "../../shared/api.js";
import { autoStatusLabel, effectText, formatReviewValue } from "../../systemReviewView.mjs";
import { EvidenceStrip } from "./EvidenceStrip.jsx";

function settingsHref(plain) {
  const where = plain?.where || {};
  const cameraId = where.scope === "camera" ? where.cameras?.[0]?.camera_id : "";
  if (cameraId) {
    return appUrl(`/admin${adminWorkspaceSearch("cameras", "", { subsection: "motion", camera: cameraId })}`);
  }
  return appUrl(`/admin${adminWorkspaceSearch("general", "", { subsection: "detection" })}`);
}

export function SuggestionCard({ item, busy, applyEnabled, past, onApply, onDismiss }) {
  const plain = item.plain || {};
  const guide = plain.guide || {};
  const where = plain.where || {};
  const unit = guide.unit || "";
  const current = item.current_effective ?? item.current;
  const support = Number(item.support_count) || 0;
  const why = [
    support ? `Seen in ${support} snapshot${support === 1 ? "" : "s"}` : "",
    item.expected_benefit || "",
  ].filter(Boolean).join(". ");
  const link = settingsHref(plain);
  const linkLabel = where.scope === "camera"
    ? `Open ${where.cameras?.[0]?.camera_name || "camera"} motion settings`
    : "Open site detection settings";
  const effect = effectText(item, guide);
  const showCompute = item.compute_impact && !["Minimal.", "None."].includes(item.compute_impact);

  return (
    <article className="system-review-card">
      <header>
        <strong>{plain.title || guide.label || "Suggested change"}</strong>
        {plain.auto_status ? <em className={`system-review-badge ${plain.auto_status}`}>{autoStatusLabel(plain.auto_status)}</em> : null}
      </header>
      <p>{where.summary} <a href={link}>{linkLabel}</a></p>
      <p>
        <span>{guide.label || "Setting"}</span>
        {" "}
        {formatReviewValue(current, unit)} → {formatReviewValue(item.proposed, unit)}
      </p>
      {guide.what ? <p>{guide.what}</p> : null}
      {effect ? <p>{effect}</p> : null}
      {why ? <p>{why.endsWith(".") ? why : `${why}.`}</p> : null}
      <EvidenceStrip samples={item.evidence || []} />
      {item.downside ? <small>Trade-off: {item.downside}</small> : null}
      {showCompute ? <small>Computer use: {item.compute_impact}</small> : null}
      {where.cameras?.length > 1 ? (
        <details className="system-review-cameras">
          <summary>Each camera</summary>
          <ul>
            {where.cameras.map((camera) => (
              <li key={camera.camera_id || camera.camera_name}>
                <strong>{camera.camera_name}</strong>
                <span>{camera.changes ? `${formatReviewValue(camera.current, unit)} → ${formatReviewValue(camera.proposed, unit)}` : "Keeps its own setting"}</span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
      <div className="button-row">
        <button type="button" className="primary" disabled={busy || past || !applyEnabled} onClick={() => onApply(item)}>Apply</button>
        {past ? null : <button type="button" disabled={busy} onClick={() => onDismiss(item)}>Not now</button>}
      </div>
    </article>
  );
}

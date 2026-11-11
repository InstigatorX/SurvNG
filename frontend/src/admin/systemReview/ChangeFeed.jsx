import React from "react";
import { formatDateTime } from "../../shared/format.js";
import { formatReviewValue, originLabel, outcomeLabel, phaseLabel } from "../../systemReviewView.mjs";

export function ChangeFeed({ changes = [], timeZone, busy, onRollback }) {
  return (
    <section className="telemetry-section">
      <div className="telemetry-section-head">
        <div>
          <h3>Changes made</h3>
          <p>After a change is applied, SurvNG watches the cameras and then compares them with the review from before.</p>
        </div>
      </div>
      {changes.length ? <div className="system-review-feed">{changes.map((item) => {
        const canRollback = item.action === "apply" && item.phase !== "rolled_back" && item.phase !== "checking";
        return (
          <article className="system-review-card" key={item.id}>
            <header>
              <strong>{originLabel(item.origin)}</strong>
              <em className={`system-review-badge ${item.phase}`}>{phaseLabel(item.phase)}</em>
            </header>
            <p>
              {item.created_at ? formatDateTime(item.created_at, timeZone) : "Time not recorded"}
              {item.phase === "watching" && item.watch_until ? ` · watching until ${formatDateTime(item.watch_until, timeZone)}` : ""}
              {item.outcome ? ` · ${outcomeLabel(item.outcome)}` : ""}
            </p>
            {item.changes?.length ? (
              <ul className="system-review-change-list">
                {item.changes.map((change) => (
                  <li key={change.id || `${change.setting}-${change.camera_id}`}>
                    <strong>{change.camera_name}</strong>
                    <span>{change.label}: {formatReviewValue(change.before, change.unit)} → {formatReviewValue(change.after, change.unit)}</span>
                  </li>
                ))}
              </ul>
            ) : null}
            {item.cameras?.length ? (
              <ul className="system-review-change-list">
                {item.cameras.map((camera) => (
                  <li key={camera.camera_id}><strong>{camera.camera_name}</strong><span>{outcomeLabel(camera.outcome)}</span></li>
                ))}
              </ul>
            ) : null}
            {item.summary ? <p>{item.summary}</p> : null}
            {item.rollback_error ? <p className="system-review-attention">Rollback needs attention: {item.rollback_error}</p> : null}
            {item.undoes_id ? <p>This put an earlier change back.</p> : null}
            {item.undone_by?.length ? <p>A later rollback restored the previous values.</p> : null}
            {canRollback ? (
              <div className="button-row">
                <button type="button" disabled={busy} onClick={() => onRollback(item)}>Roll back</button>
              </div>
            ) : null}
          </article>
        );
      })}</div> : <p className="telemetry-diagnostic-empty">No changes from these reviews yet.</p>}
    </section>
  );
}

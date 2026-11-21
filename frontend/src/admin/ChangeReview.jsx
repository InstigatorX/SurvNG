import React, { useEffect, useRef } from "react";
import { RefreshCcw, X } from "lucide-react";
import "./studio.css";

function ChangeRow({ change, onUndo, disabled }) {
  return (
    <div className="change-review-row">
      <span className="change-review-label">{change.label || "Camera"}</span>
      {change.text
        ? <span className={`change-review-value ${change.kind === "removed" ? "old" : "new"}`}>{change.text}</span>
        : change.kind === "added" && change.before === "not set"
          ? <span className="change-review-value new">{change.after}</span>
          : <span className="change-review-values"><span className="change-review-value old">{change.before}</span><span aria-hidden="true">→</span><span className="sr-only">changes to</span><span className="change-review-value new">{change.after}</span></span>}
      <button type="button" className="change-review-undo" onClick={() => onUndo(change)} disabled={disabled} aria-label={`Undo ${change.label || "change"}`}>Undo</button>
    </div>
  );
}

/**
 * "Review before saving": every unsaved change, grouped by camera, with
 * per-change undo. Saving and discarding use the page's existing handlers.
 */
export function ChangeReviewPanel({ review, impact, saving = false, onUndo, onSave, onDiscard, onClose }) {
  const headingRef = useRef(null);
  useEffect(() => {
    headingRef.current?.focus();
  }, []);
  // Undo removes the row that had focus; keep focus inside the panel.
  const undo = (change) => {
    onUndo(change);
    headingRef.current?.focus();
  };
  const total = review.count;
  const cameraCount = review.cameras.length + review.removed.length;
  return (
    <aside className="change-review" aria-labelledby="change-review-title" onKeyDown={(event) => { if (event.key === "Escape") { event.stopPropagation(); onClose(); } }}>
      <header>
        <div>
          <h2 id="change-review-title" ref={headingRef} tabIndex={-1}>Review before saving</h2>
          <p>{total ? `${total} change${total === 1 ? "" : "s"}${cameraCount ? ` on ${cameraCount} camera${cameraCount === 1 ? "" : "s"}` : ""}. ${impact}` : "There are no unsaved changes."}</p>
        </div>
        <button type="button" className="change-review-close" onClick={onClose} aria-label="Close review"><X size={16} /></button>
      </header>
      <div className="change-review-body">
        {review.cameras.map((group) => (
          <section key={group.cameraId} aria-label={group.name}>
            <h3>{group.name}{group.isNew ? <em>New</em> : null}</h3>
            {group.changes.map((change, index) => <ChangeRow key={`${change.label}-${change.zone || ""}-${index}`} change={change} onUndo={undo} disabled={saving} />)}
          </section>
        ))}
        {review.removed.length ? (
          <section aria-label="Removed cameras">
            <h3>Removed</h3>
            {review.removed.map((camera) => <div className="change-review-row" key={camera.cameraId}><span className="change-review-label">{camera.name}</span><span className="change-review-value old">Removed</span></div>)}
          </section>
        ) : null}
        {review.order ? (
          <section aria-label="Camera order">
            <h3>Camera order</h3>
            <div className="change-review-row"><span className="change-review-label">Order</span><span className="change-review-value new">Changed</span></div>
          </section>
        ) : null}
        {review.system.length ? (
          <section aria-label="System settings">
            <h3>Settings</h3>
            {review.system.map((change, index) => <ChangeRow key={`${change.label}-${index}`} change={change} onUndo={undo} disabled={saving} />)}
          </section>
        ) : null}
      </div>
      <footer>
        <button type="button" className="primary" onClick={onSave} disabled={!total || saving}>{saving ? <><RefreshCcw className="spin" size={15} aria-hidden="true" />Saving…</> : `Save ${total} change${total === 1 ? "" : "s"}`}</button>
        <button type="button" onClick={onDiscard} disabled={!total || saving}>Discard all</button>
        <p>Saved changes appear under Recently changed on the Overview.</p>
      </footer>
    </aside>
  );
}

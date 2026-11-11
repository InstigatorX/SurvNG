import React from "react";
import { formatDateTime } from "../../shared/format.js";

function reviewTitle(item) {
  if (item.mode === "system_weekly") return "Weekly review";
  if (item.mode === "system_daily") return "Daily check";
  return "Review";
}

export function PastReviews({ history = [], activeId, timeZone, busy, onOpen }) {
  return (
    <section className="telemetry-section">
      <div className="telemetry-section-head">
        <div>
          <h3>Past reviews</h3>
          <p>Open an earlier check. Suggestions from the latest review are the ones you can apply.</p>
        </div>
      </div>
      {history.length ? <div className="system-review-feed">{history.map((item) => (
        <button
          type="button"
          className={item.id === activeId ? "system-review-past active" : "system-review-past"}
          key={item.id}
          disabled={busy}
          onClick={() => onOpen(item)}
        >
          <strong>{reviewTitle(item)}</strong>
          <span>{formatDateTime(item.created_at, timeZone)} · {item.summary || item.status}{item.error ? ` · ${item.error}` : ""}</span>
        </button>
      ))}</div> : <p className="telemetry-diagnostic-empty">No reviews yet.</p>}
    </section>
  );
}

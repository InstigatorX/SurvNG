import React, { useEffect, useRef } from "react";
import { glanceSentence } from "../../systemReviewView.mjs";

export function ReviewHeader({
  glance,
  completedAt,
  running,
  progress,
  nextLabel,
  level,
  catalog,
  classes,
  busy,
  onToggleAll,
  onToggleClass,
}) {
  const automatic = useRef(null);
  useEffect(() => {
    if (automatic.current) automatic.current.indeterminate = level === "custom";
  }, [level]);

  return (
    <section className="telemetry-section system-review-glance">
      <div className="telemetry-section-head">
        <div>
          <h3>{running ? "Review in progress" : "Latest review"}</h3>
          <p>{running ? progress?.text : glanceSentence(glance)}</p>
        </div>
        {completedAt ? <span>{completedAt}</span> : null}
      </div>
      {running ? (
        <div className="motion-ai-review-progress system-review-progress" role="status" aria-label={progress?.text || "Review in progress"}>
          <div><i style={{ width: `${progress?.percent || 0}%` }} /></div>
        </div>
      ) : null}
      {nextLabel ? <p className="system-review-schedule">{nextLabel}</p> : null}
      <label className="compact-toggle system-review-switch">
        <input
          ref={automatic}
          type="checkbox"
          checked={level === "on"}
          disabled={busy}
          onChange={(event) => onToggleAll(level === "custom" ? true : event.target.checked)}
        />
        <span>
          <strong>Let SurvNG make small motion adjustments on its own</strong>
          <small>On the weekly run, allowed changes are applied and watched. If they make results worse, SurvNG puts the old values back and turns that kind of change off.</small>
        </span>
      </label>
      {level === "custom" ? <p className="system-review-schedule">Some kinds of changes are on and some are off. The switch turns them all on or all off.</p> : null}
      <details className="system-review-advanced">
        <summary>Advanced</summary>
        <div className="system-review-toggles">
          {(catalog || []).map((item) => (
            <label className="compact-toggle system-review-class" key={item.id}>
              <input
                type="checkbox"
                checked={(classes || []).includes(item.id) || (item.id === "visual_backup" && (classes || []).includes("visual_backup_cooldown"))}
                disabled={busy}
                onChange={(event) => onToggleClass(item.id, event.target.checked)}
              />
              <span><strong>{item.label}</strong><small>{item.detail}</small></span>
            </label>
          ))}
        </div>
      </details>
    </section>
  );
}

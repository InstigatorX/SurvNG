import React from "react";

export default function NativeBudgetTelemetry({ budget, detectionEnabled, connected }) {
  if (detectionEnabled === false) return <p>Adaptive inference: inactive · AI detection disabled</p>;
  if (connected === false) return <p>Adaptive inference: unavailable · camera disconnected</p>;
  if (budget?.mode === "disabled") return <p>Adaptive inference: off · fixed-rate checking; motion wake-up inactive</p>;
  if (!["idle", "active"].includes(budget?.mode)) return <p>Adaptive inference: waiting for telemetry…</p>;

  const number = (value) => value == null ? "—" : Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 });
  // Native counters are sparse: an absent counter is zero after sampling starts.
  const count = (key) => number(budget[key] ?? (budget.sampled_frames > 0 ? 0 : null));
  const skippedPercent = budget.sampled_frames > 0
    ? `${number(100 * (budget.skipped_frames ?? 0) / budget.sampled_frames)}%`
    : "—";
  return <div className="native-budget-telemetry">
    <h4>Adaptive inference · {budget.mode === "idle" ? "Idle" : "Active"}</h4>
    <dl className="telemetry-details">
      <div><dt>AI check rate</dt><dd>{number(budget.target_fps)}/sec target · {number(budget.idle_fps)} idle / {number(budget.active_fps)} active</dd></div>
      <div><dt>Active hold remaining</dt><dd>{number(budget.cooldown_remaining_seconds)} sec</dd></div>
      <div><dt>AI inputs</dt><dd>{count("admitted_frames")} admitted · {count("skipped_frames")} skipped ({skippedPercent}) · {count("sampled_frames")} sampled</dd></div>
      <div><dt>Motion wake-up</dt><dd>{budget.motion_enabled == null ? "Setting unavailable" : budget.motion_enabled ? "Enabled" : "Disabled"} · {count("motion_wakes")} requests</dd></div>
      <div><dt>Object wake-up</dt><dd>{count("object_wakes")} requests</dd></div>
      <div><dt>Excluded motion</dt><dd>{count("excluded_motion_regions")} rectangles suppressed</dd></div>
      <div><dt>Mode changes</dt><dd>{count("active_transitions")} to active · {count("idle_transitions")} to idle</dd></div>
    </dl>
    <small>Totals since this camera’s pipeline started. Wake requests include keeping it active. Skipped inputs are not measured GPU savings.</small>
  </div>;
}

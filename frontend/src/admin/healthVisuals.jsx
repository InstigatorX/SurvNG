import React from "react";

/** Sparkline: area, line and an emphasized last point. Values are scaled to the series maximum. */
export function Sparkline({ values = [], label, className = "admin-overview-spark", width = 120, height = 24, emptyText = "No activity yet" }) {
  if (values.length < 2) return <span className="admin-overview-spark-empty">{emptyText}</span>;
  const max = Math.max(1e-9, ...values);
  const step = width / (values.length - 1);
  const points = values.map((value, index) => [index * step, height - 2 - (Math.max(0, value) / max) * (height - 6)]);
  const line = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const [lastX, lastY] = points.at(-1);
  return (
    <svg className={className} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label={label}>
      <path d={`${line} L${width} ${height} L0 ${height} Z`} className="area" />
      <path d={line} className="line" />
      <circle cx={lastX} cy={lastY} r="2.5" className="point" />
    </svg>
  );
}

/** One bar per hour, colored by availability. */
export function UptimeStrip({ slots = [], name }) {
  if (!slots.length) return <span className="admin-overview-muted">No history yet</span>;
  return (
    <div className="admin-overview-uptime" role="img" aria-label={`${name} availability by hour, last 24 hours`}>
      {slots.map((slot, index) => <span key={slot.startedAt || index} className={slot.tone} title={slot.label} />)}
    </div>
  );
}

import React from "react";
import { formatTimeOnly } from "../shared/format.js";
import { fraction, objectName, timelineTicks } from "../incidentStory.mjs";

const ROW_HEIGHT = 21;

function barLabel(bar, lane, lanes, cameraName) {
  const name = objectName(bar);
  if (bar.uncertain) return `${name}?`;
  if (bar.state === "stationary") return `${name} · stationary`;
  const sameLane = lane.bars.filter((item) => item.objectId === bar.objectId);
  const following = sameLane[sameLane.indexOf(bar) + 1];
  if (bar.state === "moving" && following?.state === "stationary") return `${name} arrives`;
  // Name the camera where this object was seen first, when it came from elsewhere.
  const earlier = lanes.flatMap((item) => item.bars).filter((item) => item.objectId === bar.objectId && item.start < bar.start && item.camera_id !== bar.camera_id);
  if (earlier.length && sameLane[0] === bar) return `${name} · same as ${cameraName(earlier[0].camera_id)}`;
  return name;
}

export function IncidentStoryline({ line, cameraName, selectedId, cursor, timeZone, hint, onBar }) {
  const { start, end, lanes } = line;
  const playLeft = `${fraction(cursor, start, end) * 100}%`;
  const showCursor = Number.isFinite(cursor);
  return <section className="story-storyline" aria-label="Storyline">
    <h2>Storyline</h2>
    {lanes.length ? <div className="story-storyline-grid">
      <span aria-hidden="true" />
      <div className="story-axis" aria-hidden="true">
        {timelineTicks(start, end).map((tick, index, ticks) => <span key={index} className={index === ticks.length - 1 ? "end" : index % 2 ? "minor" : ""} style={index === ticks.length - 1 ? undefined : { left: `${tick.fraction * 100}%` }}>{formatTimeOnly(tick.epoch, timeZone)}</span>)}
      </div>
      {lanes.map((lane, laneIndex) => <React.Fragment key={lane.key}>
        <span className="story-lane-name" title={cameraName(lane.cameraId)}>{cameraName(lane.cameraId)}{lane.repeated ? <small> {formatTimeOnly(lane.start, timeZone)}</small> : null}</span>
        <div className="story-lane" style={{ height: `${lane.rows * (ROW_HEIGHT + 4) + 6}px` }} role="group" aria-label={`${cameraName(lane.cameraId)} episode`}>
          {lane.bars.map((bar, index) => {
            const left = fraction(bar.start, start, end) * 100;
            const width = Math.max(1.2, (fraction(bar.end, start, end) * 100) - left);
            const dimmed = selectedId != null && String(bar.objectId) !== String(selectedId);
            const label = barLabel(bar, lane, lanes, cameraName);
            return <button key={`${bar.objectId}-${index}`} type="button"
              className={`story-bar ${bar.kind}${bar.state === "stationary" ? " stationary" : ""}${dimmed ? " dimmed" : ""}${String(bar.objectId) === String(selectedId) ? " selected" : ""}`}
              style={{ left: `${Math.min(left, 100 - width)}%`, width: `${width}%`, top: `${5 + bar.row * (ROW_HEIGHT + 4)}px`, height: `${ROW_HEIGHT - 4}px` }}
              aria-label={`${label}, ${cameraName(bar.camera_id)}, ${formatTimeOnly(bar.start, timeZone)} to ${formatTimeOnly(bar.end, timeZone)}`}
              onClick={(event) => {
                const rect = event.currentTarget.getBoundingClientRect();
                const ratio = event.detail && rect.width > 0 ? Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) : 0;
                onBar?.(bar, bar.start + (bar.end - bar.start) * ratio);
              }}>{label}</button>;
          })}
          {!lane.bars.length ? <span className="story-lane-empty">No objects observed{lane.coverage?.state && lane.coverage.state !== "sampled" ? " · analysis incomplete" : ""}</span> : null}
          {showCursor ? <span className={`story-playhead${laneIndex === 0 ? " first" : ""}`} style={{ left: playLeft }} aria-hidden="true" /> : null}
        </div>
      </React.Fragment>)}
    </div> : <p className="story-muted">No camera episodes are available for this incident.</p>}
    <p className="story-hint">{hint}</p>
  </section>;
}

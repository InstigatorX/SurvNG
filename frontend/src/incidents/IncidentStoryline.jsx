import React, { useLayoutEffect, useRef, useState } from "react";
import { formatTimeOnly } from "../shared/format.js";
import { fraction, layoutStoryLane, objectName, timelineTicks } from "../incidentStory.mjs";

const BAR_HEIGHT = 24;
const ROW_HEIGHT = BAR_HEIGHT + 4;

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
  const axis = useRef(null);
  const [trackWidth, setTrackWidth] = useState(0);
  const hasLanes = lanes.length > 0;
  useLayoutEffect(() => {
    const track = axis.current;
    if (!track) return undefined;
    const measure = () => setTrackWidth(track.getBoundingClientRect().width);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(track);
    return () => observer.disconnect();
  }, [hasLanes]);
  const playLeft = `${fraction(cursor, start, end) * 100}%`;
  const showCursor = Number.isFinite(cursor);
  return <section className="story-storyline" aria-label="Storyline">
    <h2>Storyline</h2>
    {lanes.length ? <div className="story-storyline-grid">
      <span aria-hidden="true" />
      <div ref={axis} className="story-axis" aria-hidden="true">
        {timelineTicks(start, end).map((tick, index, ticks) => <span key={index} className={index === ticks.length - 1 ? "end" : index % 2 ? "minor" : ""} style={index === ticks.length - 1 ? undefined : { left: `${tick.fraction * 100}%` }}>{formatTimeOnly(tick.epoch, timeZone)}</span>)}
      </div>
      {lanes.map((lane, laneIndex) => {
        const layout = layoutStoryLane(lane.bars, start, end, trackWidth);
        return <React.Fragment key={lane.key}>
        <span className="story-lane-name" title={cameraName(lane.cameraId)}>{cameraName(lane.cameraId)}{lane.repeated ? <small> {formatTimeOnly(lane.start, timeZone)}</small> : null}</span>
        <div className="story-lane" style={{ height: layout.rows * ROW_HEIGHT + 6 }} role="group" aria-label={`${cameraName(lane.cameraId)} episode`}>
          {layout.bars.map(({ bar, row, left, width, startX, endX }, index) => {
            const dimmed = selectedId != null && String(bar.objectId) !== String(selectedId);
            const label = barLabel(bar, lane, lanes, cameraName);
            const description = `${label}, ${cameraName(bar.camera_id)}, ${formatTimeOnly(bar.start, timeZone)} to ${formatTimeOnly(bar.end, timeZone)}`;
            return <button key={`${bar.objectId}-${index}`} type="button"
              className={`story-bar ${bar.kind}${bar.state === "stationary" ? " stationary" : ""}${dimmed ? " dimmed" : ""}${String(bar.objectId) === String(selectedId) ? " selected" : ""}`}
              style={{ left, width, minWidth: 0, top: 5 + row * ROW_HEIGHT, height: BAR_HEIGHT }}
              title={description} aria-label={description}
              onClick={(event) => {
                const rect = event.currentTarget.getBoundingClientRect();
                const ratio = event.detail && endX > startX ? Math.max(0, Math.min(1, (left + event.clientX - rect.left - startX) / (endX - startX))) : 0;
                onBar?.(bar, bar.start + (bar.end - bar.start) * ratio);
              }}>{label}</button>;
          })}
          {!lane.bars.length ? <span className="story-lane-empty">No objects observed{lane.coverage?.state && lane.coverage.state !== "sampled" ? " · analysis incomplete" : ""}</span> : null}
          {showCursor ? <span className={`story-playhead${laneIndex === 0 ? " first" : ""}`} style={{ left: playLeft }} aria-hidden="true" /> : null}
        </div>
      </React.Fragment>;
      })}
    </div> : <p className="story-muted">No camera episodes are available for this incident.</p>}
    <p className="story-hint">{hint}</p>
  </section>;
}

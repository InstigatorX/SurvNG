// Harness for tests/scrub-frame-react.mjs: mounts the real scrub-frame React
// pieces one scenario at a time (?scenario=hook|overlay|rail|edge|classic) and
// records what they report on `window.log`. `window.h` exposes controls.
import React, { Profiler, useLayoutEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import { useScrubStrips } from "../../src/timeline/useScrubStrips.js";
import { ScrubFrameOverlay } from "../../src/timeline/ScrubFrameOverlay.jsx";
import { EdgeRail, useVerticalScrub } from "../../src/timeline/TimelineRail.jsx";
import { RecordingTimeline } from "../../src/timeline/TimelinePages.jsx";
import { scrubStripTile } from "../../src/scrubStrips.mjs";
import "../../src/styles.css";
import "../../src/timeline/timeline.css";

const params = new URLSearchParams(location.search);
const scenario = params.get("scenario") || "hook";
const log = [];
const h = {};
window.log = log;
window.h = h;
// Mutations of harness state from the test are flushed synchronously, so the
// next assertion sees the rendered result.
const now = (update) => flushSync(update);

// The strip image: TILE_COLUMNS x TILE_ROWS tiles, tile i filled with grey
// level 20 + 20 * i, so the canvas the overlay draws says which tile it is.
const TILE_COLUMNS = 4;
const TILE_ROWS = 3;
const TILE_W = 40;
const TILE_H = 30;
function stripCanvas() {
  const canvas = document.createElement("canvas");
  canvas.width = TILE_COLUMNS * TILE_W;
  canvas.height = TILE_ROWS * TILE_H;
  const context = canvas.getContext("2d");
  for (let i = 0; i < TILE_COLUMNS * TILE_ROWS; i += 1) {
    const level = 20 + 20 * i;
    context.fillStyle = `rgb(${level},${level},${level})`;
    context.fillRect((i % TILE_COLUMNS) * TILE_W, Math.floor(i / TILE_COLUMNS) * TILE_H, TILE_W, TILE_H);
  }
  return canvas;
}
h.stripPng = () => stripCanvas().toDataURL("image/png").split(",")[1];

// Which tile the overlay's canvas shows, or null when nothing is drawn.
h.overlay = () => {
  const frame = document.querySelector(".recording-scrub-frame");
  const canvas = frame?.querySelector("canvas");
  let tile = null;
  if (canvas && canvas.width > 1) {
    const [level] = canvas.getContext("2d").getImageData(Math.floor(canvas.width / 2), Math.floor(canvas.height / 2), 1, 1).data;
    tile = level === 0 ? null : Math.round((level - 20) / 20);
  }
  return {
    up: Boolean(frame),
    drawn: frame?.dataset.drawn ?? null,
    loading: Boolean(frame?.classList.contains("loading")),
    gap: Boolean(frame?.classList.contains("gap")),
    text: frame?.textContent ?? "",
    status: document.querySelector('[role="status"]')?.textContent ?? null,
    statusPresent: Boolean(document.querySelector('[role="status"]')),
    canvas: canvas ? [canvas.width, canvas.height] : null,
    opacity: canvas ? getComputedStyle(canvas).opacity : null,
    visibility: canvas ? getComputedStyle(canvas).visibility : null,
    tile,
  };
};
h.sleep = (ms) => new Promise((done) => setTimeout(done, ms));

const describe = (frame, options) => (frame
  ? {
    frame: true, epoch: frame.epoch, start: frame.strip?.start ?? null, source: frame.strip?.source ?? null,
    index: frame.tile?.index ?? null, gap: frame.gap, loading: frame.loading, held: frame.held,
  }
  : { frame: false, committed: Boolean(options?.committed) });

// ---- useScrubStrips, wired to a real overlay as the timeline page does.
function Scrubber({ cameraId, source, overlayRef }) {
  const scrub = useScrubStrips({
    cameraId, source,
    onFrame: (frame, options) => {
      log.push(describe(frame, options));
      overlayRef.current?.show(frame, options);
    },
  });
  h.scrub = (epoch, options) => now(() => scrub(epoch, options));
  return null;
}
function HookHarness() {
  const overlayRef = useRef(null);
  const [cameraId, setCameraId] = useState(params.get("camera") ?? "cam1");
  const [source, setSource] = useState("main");
  const [mounted, setMounted] = useState(true);
  const [seeking, setSeeking] = useState(false);
  Object.assign(h, {
    setCamera: (value) => now(() => setCameraId(value)),
    setSource: (value) => now(() => setSource(value)),
    unmount: () => now(() => setMounted(false)),
    setSeeking: (value) => now(() => setSeeking(value)),
  });
  return (
    <div className="recording-hero-media" style={{ position: "relative", width: 320, height: 180 }}>
      <ScrubFrameOverlay ref={overlayRef} seeking={seeking} />
      {mounted ? <Scrubber cameraId={cameraId} source={source} overlayRef={overlayRef} /> : null}
    </div>
  );
}

// ---- ScrubFrameOverlay alone, fed frames built here.
const strips = {};
async function strip(id) {
  if (!strips[id]) {
    const image = new Image();
    image.src = stripCanvas().toDataURL("image/png");
    await image.decode();
    strips[id] = { start: 1000, end: 1010, step: 1, count: 10, columns: TILE_COLUMNS, rows: TILE_ROWS, image };
  }
  return strips[id];
}
h.renders = 0;
function OverlayHarness() {
  const overlayRef = useRef(null);
  const [seeking, setSeeking] = useState(params.get("seeking") === "true");
  // spec: { strip: id | null, index, gap, loading, held }; null clears.
  h.show = async (spec, options) => {
    let frame = null;
    if (spec) {
      const found = spec.strip ? await strip(spec.strip) : null;
      frame = {
        epoch: 1000 + (spec.index ?? 0), strip: found,
        tile: found ? scrubStripTile(found, 1000 + spec.index, found.image.naturalWidth, found.image.naturalHeight) : null,
        gap: Boolean(spec.gap), loading: Boolean(spec.loading), held: Boolean(spec.held),
      };
    }
    now(() => overlayRef.current.show(frame, options));
  };
  h.setSeeking = (value) => now(() => setSeeking(value));
  return (
    <div className="recording-hero-media" style={{ position: "relative", width: 320, height: 180 }}>
      <Profiler id="overlay" onRender={() => { h.renders += 1; }}>
        <ScrubFrameOverlay ref={overlayRef} seeking={seeking} />
      </Profiler>
    </div>
  );
}

// ---- useVerticalScrub on a bare scroller.
const RAIL_TOP = 100000;
const RAIL_BOTTOM = RAIL_TOP - 86400;
function Rail({ playhead, onSeek }) {
  const scrub = useVerticalScrub({
    topEpoch: RAIL_TOP, bottomEpoch: RAIL_BOTTOM, spanSeconds: 3600, playhead, onSeek,
    onScrub: (epoch, options) => {
      log.push(epoch === null ? { end: true, committed: Boolean(options?.committed) } : { epoch });
      if (epoch !== null && h.unmountOnScrub) setTimeout(() => h.unmount(), 0);
    },
  });
  return (
    <div className="rail-scroller" {...scrub.scrollerProps} style={{ height: 400, width: 200, overflowY: "auto" }}>
      <div style={{ height: scrub.contentHeight }} />
    </div>
  );
}
function RailHarness() {
  const [playhead, setPlayhead] = useState(RAIL_TOP - 7200);
  const [mounted, setMounted] = useState(true);
  Object.assign(h, {
    setPlayhead: (value) => now(() => setPlayhead(value)),
    unmount: () => now(() => setMounted(false)),
  });
  const onSeek = (epoch) => {
    log.push({ seek: epoch });
    // The player arrives there a moment later.
    setTimeout(() => setPlayhead(epoch), 50);
  };
  return mounted ? <Rail playhead={playhead} onSeek={onSeek} /> : null;
}

// ---- EdgeRail.
const DAY_START = 86400 * 20000;
function EdgeHarness() {
  const [mounted, setMounted] = useState(true);
  h.unmount = () => now(() => setMounted(false));
  const withScrub = params.get("scrub") !== "false";
  return (
    <div style={{ position: "relative", width: 120, height: 640 }}>
      <style>{`.edge-rail{position:absolute;inset:0 0 0 60px;width:44px}.edge-rail-track{position:absolute;inset:48px 0}.edge-rail-day{position:absolute;left:0;right:0;height:44px}.edge-rail-day.next{top:0}.edge-rail-day.previous{bottom:0}.edge-rail-loupe{position:absolute;right:50px}`}</style>
      {mounted ? (
        <EdgeRail
          cameraId={params.get("camera") ?? "cam1"} source="main"
          dayStart={DAY_START} dayEnd={DAY_START + 86400} topEpoch={DAY_START + 86400}
          events={[]} recordings={[]} windowStart={DAY_START + 3600} windowEnd={DAY_START + 7200}
          playhead={DAY_START + 3600} timeZone="UTC"
          onSeek={(epoch) => log.push({ seek: epoch })}
          onScrub={withScrub ? (epoch, options) => log.push(epoch === null ? { end: true, committed: Boolean(options?.committed) } : { epoch }) : null}
        />
      ) : null}
    </div>
  );
}

// ---- RecordingTimeline (classic scrubber).
const CLASSIC_START = 86400 * 20000;
function ClassicHarness() {
  const [mounted, setMounted] = useState(true);
  h.unmount = () => now(() => setMounted(false));
  return (
    <div style={{ width: 800, padding: 20 }}>
      {mounted ? (
        <RecordingTimeline
          cameraId="cam1" source="main" startEpoch={CLASSIC_START} endEpoch={CLASSIC_START + 3600}
          recordings={[]} events={[]} evidenceFilter="all" laneVisibility={{ object: true, motion: true }}
          selectedEventId={null} onEventSelect={() => {}} playhead={CLASSIC_START + 1200} timeZone="UTC"
          onSeek={(epoch) => log.push({ seek: epoch })}
          onScrub={(epoch, options) => log.push(epoch === null ? { end: true, committed: Boolean(options?.committed) } : { epoch })}
          onPanViewport={(delta) => log.push({ pan: delta })}
          windowHours={1} exportRange={null}
        />
      ) : null}
    </div>
  );
}

h.constants = { TILE_COLUMNS, TILE_ROWS, RAIL_TOP, RAIL_BOTTOM, DAY_START, CLASSIC_START };
const scenarios = { hook: HookHarness, overlay: OverlayHarness, rail: RailHarness, edge: EdgeHarness, classic: ClassicHarness };
const Scenario = scenarios[scenario];
function Ready() {
  useLayoutEffect(() => { document.body.dataset.ready = "true"; }, []);
  return null;
}
createRoot(document.getElementById("root")).render(<><Scenario /><Ready /></>);

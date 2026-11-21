import React, { useEffect, useRef, useState } from "react";

const MEDIA_EVENTS = ["loadstart", "waiting", "stalled", "playing", "pause", "seeking", "seeked", "ended", "error"];

function emptyStats() {
  return {
    waits: 0, stalls: 0, waitMs: 0, longestWaitMs: 0,
    seeks: 0, lastSeekMs: null, longestSeekMs: 0,
    loads: 0, errors: 0, lastError: "",
  };
}

function srcKind(src) {
  if (!src) return "none";
  if (/\.m3u8(\?|$)/.test(src)) return "HLS playlist";
  if (/segment\.mp4/.test(src)) return "per-file MP4";
  return "other";
}

function bufferedAhead(video) {
  if (!video?.buffered) return 0;
  for (let index = 0; index < video.buffered.length; index += 1) {
    if (video.buffered.start(index) <= video.currentTime + 0.05 && video.currentTime <= video.buffered.end(index)) {
      return Math.max(0, video.buffered.end(index) - video.currentTime);
    }
  }
  return 0;
}

const seconds = (ms) => (ms === null ? "—" : `${(ms / 1000).toFixed(2)} s`);

/**
 * On-device playback readout for testing transports (for example native HLS on
 * an iPhone). It only observes: media events from every video inside
 * `containerRef` are counted, and the visible video is sampled twice a second.
 */
export function PlaybackDiagnostics({ containerRef, transport, override, playlistStart }) {
  const statsRef = useRef(emptyStats());
  const waitStartRef = useRef(new Map());
  const seekStartRef = useRef(new Map());
  const [, setTick] = useState(0);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    const onMediaEvent = (event) => {
      const video = event.target;
      if (!(video instanceof HTMLMediaElement)) return;
      // Hidden standby elements preload the next file; only the visible one counts.
      if (getComputedStyle(video).visibility === "hidden") return;
      const stats = statsRef.current;
      const now = performance.now();
      if (event.type === "loadstart") stats.loads += 1;
      if (event.type === "waiting" || event.type === "stalled") {
        if (event.type === "waiting") stats.waits += 1; else stats.stalls += 1;
        if (!waitStartRef.current.has(video)) waitStartRef.current.set(video, now);
      }
      if (event.type === "playing" || event.type === "pause" || event.type === "ended") {
        const started = waitStartRef.current.get(video);
        if (started !== undefined) {
          const waited = now - started;
          stats.waitMs += waited;
          stats.longestWaitMs = Math.max(stats.longestWaitMs, waited);
          waitStartRef.current.delete(video);
        }
      }
      if (event.type === "seeking") {
        stats.seeks += 1;
        seekStartRef.current.set(video, now);
      }
      if (event.type === "seeked") {
        const started = seekStartRef.current.get(video);
        if (started !== undefined) {
          stats.lastSeekMs = now - started;
          stats.longestSeekMs = Math.max(stats.longestSeekMs, stats.lastSeekMs);
          seekStartRef.current.delete(video);
        }
      }
      if (event.type === "error") {
        stats.errors += 1;
        stats.lastError = `${video.error?.code ?? "?"} ${video.error?.message || "media error"}`.trim();
      }
    };
    for (const name of MEDIA_EVENTS) container.addEventListener(name, onMediaEvent, true);
    const timer = window.setInterval(() => setTick((tick) => tick + 1), 500);
    return () => {
      for (const name of MEDIA_EVENTS) container.removeEventListener(name, onMediaEvent, true);
      window.clearInterval(timer);
    };
  }, [containerRef]);

  const videos = [...(containerRef.current?.querySelectorAll("video") || [])];
  const video = videos.find((element) => getComputedStyle(element).visibility !== "hidden") || videos[0];
  const stats = statsRef.current;
  return (
    <aside className="playback-diagnostics" aria-label="Playback diagnostics">
      <strong>{override && override !== transport ? `forced ${override}, fell back to ${transport}` : `${transport}${override ? " (forced)" : ""}`} · {srcKind(video?.currentSrc || video?.getAttribute("src"))}</strong>
      {playlistStart ? <span>Startup via EXT-X-START</span> : null}
      <span>{video ? `${video.paused ? "paused" : "playing"} · t ${video.currentTime.toFixed(2)} · ready ${video.readyState} · ahead ${bufferedAhead(video).toFixed(1)} s` : "no video"}</span>
      <span>waits {stats.waits} · stalls {stats.stalls} · waited {seconds(stats.waitMs)} (max {seconds(stats.longestWaitMs)})</span>
      <span>seeks {stats.seeks} · last {seconds(stats.lastSeekMs)} · max {seconds(stats.longestSeekMs)} · loads {stats.loads}</span>
      {stats.errors ? <span className="playback-diagnostics-error">errors {stats.errors} · {stats.lastError}</span> : null}
      <button type="button" onClick={() => { statsRef.current = emptyStats(); waitStartRef.current.clear(); seekStartRef.current.clear(); setTick((tick) => tick + 1); }}>Reset</button>
    </aside>
  );
}

export function playbackDiagnosticsEnabled(override, win = typeof window === "undefined" ? null : window) {
  try {
    const setting = new URLSearchParams(win?.location?.search || "").get("playbackdebug");
    return setting !== "0" && (setting === "1" || Boolean(override));
  } catch {
    return Boolean(override);
  }
}

import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { RecordingHlsVideo } from "../../src/shared/RecordingHlsVideo.jsx";

const fixture = await window.fetch("/fixture-config.json").then((response) => response.json());
const initialName = new URLSearchParams(window.location.search).get("fixture");
const events = [];
const frames = [];
const ids = new WeakMap();
let nextId = 0;
let playRun = 0;
const identify = (video) => video && (ids.get(video) || (ids.set(video, ++nextId), nextId));
const nativeHls = Boolean(document.createElement("video").canPlayType("application/vnd.apple.mpegurl"));

function summarizeFrames() {
  let minPeakRgb = 255;
  let blackFrames = 0;
  let maxPlayingIntervalMs = 0;
  let previousPresentedFrame = null;
  const colors = [];
  for (let index = 0; index < frames.length; index += 1) {
    const frame = frames[index];
    const peak = Math.max(...frame.rgb);
    minPeakRgb = Math.min(minPeakRgb, peak);
    if (peak < 70) blackFrames += 1;
    const color = peak < 70 ? "black" : ["red", "green", "blue"][frame.rgb.indexOf(peak)];
    if (colors.at(-1) !== color) colors.push(color);
    if (frame.sampling === "rvfc") {
      const previous = previousPresentedFrame;
      if (previous && !previous.paused && !frame.paused && previous.playRun === frame.playRun && previous.id === frame.id && previous.source === frame.source && frame.time > previous.time && frame.time - previous.time < 1) {
        maxPlayingIntervalMs = Math.max(maxPlayingIntervalMs, frame.wallTime - previous.wallTime);
      }
      previousPresentedFrame = frame;
    }
  }
  return { colors, blackFrames, minPeakRgb: frames.length ? minPeakRgb : null, maxPlayingIntervalMs: Math.round(maxPlayingIntervalMs) };
}

function Fixture() {
  const videoRef = useRef(null);
  const [config, setConfig] = useState({ name: fixture.playlists[initialName] ? initialName : "h264", revision: 0, offline: false, playing: false, seek: 0 });
  const [status, setStatus] = useState({});
  const [error, setError] = useState(null);
  const sampler = useRef({});
  const desired = useRef({ ...config });
  const src = `/media/${fixture.playlists[config.name]}?revision=${config.revision}${config.offline ? "&offline=1" : ""}`;
  const currentSource = useRef(src);
  currentSource.current = src;
  function snapshot() {
    const video = videoRef.current;
    const activeFrames = frames.filter((frame) => frame.id === identify(video) && frame.source === currentSource.current);
    return { source: currentSource.current, transport: nativeHls ? "Native HLS" : "Shaka MSE", id: identify(video), time: video?.currentTime || 0,
      rate: video?.playbackRate, duration: Number.isFinite(video?.duration) ? video.duration : null, paused: video?.paused ?? true,
      playIntent: desired.current.playing,
      readyState: video?.readyState ?? 0, seeking: video?.seeking ?? false, ended: video?.ended ?? false,
      metadata: events.filter((event) => event.name === "metadata").length,
      ready: events.filter((event) => event.name === "ready").length, frames: frames.length,
      activeFrames: activeFrames.length, requestedTime: desired.current.seek, lastActiveFrame: activeFrames.at(-1) || null,
      sampler: { ...sampler.current },
      frameSummary: summarizeFrames(),
    };
  }
  function record(name, video = videoRef.current, extra = {}) {
    events.push({ name, source: currentSource.current, id: identify(video), time: video?.currentTime || 0, ownsRef: video === videoRef.current, ...extra });
    setStatus(snapshot());
  }
  function play() {
    desired.current.playing = true;
    videoRef.current?.play().catch((failure) => setError({ message: failure.message }));
  }
  function pause() { desired.current.playing = false; videoRef.current?.pause(); }
  function seek(time, playing = !videoRef.current?.paused) {
    desired.current.seek = time;
    videoRef.current.currentTime = time;
    if (playing) play(); else pause();
  }
  function change(patch) {
    setError(null);
    desired.current = { ...desired.current, ...patch };
    setConfig((previous) => ({ ...previous, playing: desired.current.playing, seek: desired.current.seek, ...patch }));
  }
  function switchWindow(index, playing) {
    const name = index === 1 && fixture.playlists.gap ? "gap" : "h264";
    const target = index === 0 ? 10.25 : 20.25;
    const targetSource = `/media/${fixture.playlists[name]}?revision=${index}`;
    if (targetSource === currentSource.current) seek(target, playing);
    else change({ name, revision: index, seek: target, playing, offline: false });
  }
  window.hlsFixture = { events, frames, snapshot, video: () => videoRef.current, change, seek, play, pause, error,
    capabilities: { nativeHls, hevcMse: Boolean(window.MediaSource?.isTypeSupported('video/mp4; codecs="hvc1.1.6.L93.B0"')) },
  };
  useEffect(() => {
    const video = videoRef.current;
    let running = true;
    let frameHandle = null;
    let lastCallbackAt = performance.now();
    const diagnostic = { rvfcSupported: typeof video?.requestVideoFrameCallback === "function", callbacks: 0,
      rvfcSamples: 0, fallbackSamples: 0, rejectedCallbacks: 0, canvasErrors: 0, lastError: "", lastFailure: "" };
    sampler.current = diagnostic;
    const canvas = document.createElement("canvas"); canvas.width = canvas.height = 1;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const sample = (sampling) => {
      try {
        if (sampling === "rvfc") { diagnostic.callbacks += 1; lastCallbackAt = performance.now(); }
        if (!running || !video || video !== videoRef.current || currentSource.current !== src) {
          diagnostic.rejectedCallbacks += 1;
          return;
        }
        if (video.readyState < 2 || !video.videoWidth || video.seeking) return;
        ctx.drawImage(video, 0, 0, 1, 1);
        const rgb = [...ctx.getImageData(0, 0, 1, 1).data].slice(0, 3);
        diagnostic.lastError = "";
        const previous = frames.at(-1);
        if (sampling !== "rvfc" && previous?.id === identify(video) && previous.source === src && previous.time === video.currentTime && previous.rgb.every((value, index) => value === rgb[index])) return;
        frames.push({ id: identify(video), time: video.currentTime, wallTime: performance.now(), paused: video.paused, playRun, sampling,
          rgb, source: src });
        if (sampling === "rvfc") diagnostic.rvfcSamples += 1;
        else diagnostic.fallbackSamples += 1;
      } catch (failure) {
        // A native media surface may temporarily reject canvas reads. Expose
        // that failure and keep observing; never invent a decoded pixel/frame.
        diagnostic.canvasErrors += 1;
        diagnostic.lastError = `${failure.name}: ${failure.message}`;
        diagnostic.lastFailure = diagnostic.lastError;
      } finally {
        // A transient ref mismatch or canvas failure must not permanently
        // terminate observation of this playlist's captured element.
        if (running && sampling === "rvfc") frameHandle = video.requestVideoFrameCallback(() => sample("rvfc"));
      }
    };
    if (diagnostic.rvfcSupported) frameHandle = video.requestVideoFrameCallback(() => sample("rvfc"));
    const timer = window.setInterval(() => {
      // Safari can expose RVFC without delivering native-HLS callbacks. Pixel
      // checks after callback silence prove colors, not rendered frame cadence.
      if (!diagnostic.rvfcSupported || performance.now() - lastCallbackAt > 500) sample("poll");
      setStatus(snapshot());
    }, 150);
    return () => {
      running = false; window.clearInterval(timer);
      if (frameHandle !== null) video?.cancelVideoFrameCallback?.(frameHandle);
    };
  }, [src]);
  return <main style={{ maxWidth: 840, margin: "24px auto", padding: 16, fontFamily: "system-ui", color: "#17212b" }}>
    <h1>Recording HLS browser checks</h1>
    <p>{fixture.description} Clip boundaries: {fixture.manifest[config.name]?.boundaries.join(", ") || "—"} seconds. Colored footage makes resets and missing frames visible.</p>
    <label>Recording fixture <select aria-label="Recording fixture" value={config.name} onChange={(event) => change({ name: event.target.value, seek: 0, revision: config.revision + 1, offline: false })}>
      {Object.keys(fixture.playlists).map((name) => <option key={name} value={name}>{({ h264: "H.264 · red / green / blue", hevc: "HEVC · green / blue", mixed: "Mixed H.264 and HEVC", gap: "H.264 with recording-time gap", unknown: "Unknown codec metadata · H.264 discontinuities" })[name] || name}</option>)}
    </select></label>
    <div id="stage" style={{ background: "#050708", margin: "16px 0", width: "100%", aspectRatio: "16/9" }}>
      <RecordingHlsVideo ref={videoRef} src={src} muted controls playsInline preload="auto" startTime={config.seek} bufferingGoal={40}
        style={{ display: "block", width: "100%", height: "100%" }}
        onReady={(_player, video) => {
          record("ready", video);
          if (Math.abs(video.currentTime - desired.current.seek) > 0.05) video.currentTime = desired.current.seek;
          if (desired.current.playing) play();
        }}
        onLoadedMetadata={(event) => record("metadata", event.currentTarget)}
        onPlay={(event) => { playRun += 1; record("play", event.currentTarget); }}
        onPause={(event) => record("pause", event.currentTarget)}
        onSeeking={(event) => record("seeking", event.currentTarget)}
        onSeeked={(event) => record("seeked", event.currentTarget)}
        onEnded={(event) => { desired.current.playing = false; record("ended", event.currentTarget); }}
        onError={(failure) => {
          const value = { category: failure.category, code: failure.code, message: failure.message || String(failure) };
          record("error", videoRef.current, value); setError(value);
        }}
      />
    </div>
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      <button onClick={play}>Play</button><button onClick={pause}>Pause</button>
      <button onClick={() => seek(0, true)}>Play from start</button>
      <button onClick={() => seek(9.75)}>Seek 9.75s</button><button onClick={() => seek(10.25)}>Seek 10.25s</button>
      <button onClick={() => seek(20.25)}>Seek 20.25s</button>
      <button onClick={() => change({ revision: config.revision + 1, seek: 10.25, offline: false })}>Switch playlist URL</button>
      <button onClick={() => change({ revision: config.revision + 1, offline: true })}>Simulate network failure</button>
      <button onClick={() => change({ revision: config.revision + 1, offline: false })}>Retry playlist</button>
      <button onClick={() => switchWindow(0, false)}>Window A · paused</button>
      <button onClick={() => switchWindow(1, false)}>Window B · paused</button>
      <button onClick={() => switchWindow(0, true)}>Window A · playing</button>
      <button onClick={() => switchWindow(1, true)}>Window B · playing</button>
      {[1, 2, 4].map((rate) => <button key={rate} aria-label={`Playback speed ${rate}×`} onClick={() => { videoRef.current.playbackRate = rate; }}>{rate}×</button>)}
    </div>
    <p role="status">{status.transport} · {status.paused ? "Paused" : "Playing"} · {Number(status.time || 0).toFixed(2)} / {status.duration?.toFixed(2) || "—"} seconds · {status.rate || 1}× · Video {status.id || "—"} · Seeking {status.seeking ? "yes" : "no"} · Metadata {status.metadata || 0} · Ready {status.ready || 0} · Current-source pixel samples {status.activeFrames || 0} · Total pixel samples {status.frames || 0}</p>
    <p data-testid="frame-summary">Sampled colors: {status.frameSummary?.colors.join(" → ") || "—"} · Black pixel samples: {status.frameSummary?.blackFrames || 0} · Largest RVFC playing-frame interval: {status.frameSummary?.maxPlayingIntervalMs ? `${status.frameSummary.maxPlayingIntervalMs} ms` : "—"}</p>
    <p data-testid="sampler-status">RVFC supported: {status.sampler?.rvfcSupported ? "yes" : "no"} · Callbacks: {status.sampler?.callbacks || 0} · RVFC samples: {status.sampler?.rvfcSamples || 0} · Fallback samples: {status.sampler?.fallbackSamples || 0} · Canvas failures: {status.sampler?.canvasErrors || 0} · Last sampling failure: {status.sampler?.lastFailure || "none"}</p>
    <p role="alert">{error ? `Playback error: category ${error.category ?? "—"}, code ${error.code ?? "—"}: ${error.message}` : "No playback errors"}</p>
    <details><summary>Diagnostics</summary><pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify({ ...status, error }, null, 2)}</pre></details>
  </main>;
}
createRoot(document.getElementById("root")).render(<Fixture />);

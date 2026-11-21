import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Camera, Maximize2 } from "lucide-react";
import { gridPlaybackNeedsSeek, playbackMediaTimeForEpoch, recordingPlaybackTimeline } from "../recordingPlayback.mjs";
import { ALL_CAMERAS_WINDOW_SECONDS, allCamerasWindowStart } from "../recordingGrid.mjs";
import { focusLiveMosaicLayout } from "../liveFocusLayout.mjs";
import { uniformLiveGridLayout } from "../liveWorkspace.mjs";
import { fetch } from "../shared/api.js";
import { recordingDayHlsUrl, recordingPreviewUrl, recordingWindowUrl } from "../shared/mediaUrls.js";
import { RecordingHlsVideo } from "../shared/RecordingHlsVideo.jsx";

// Each tile plays one 15-minute window playlist (ALL_CAMERAS_WINDOW_SECONDS).
// A window this recent may still be growing; a tile nearing the end of what
// it loaded re-reads it early, so new footage is ready before it runs out.
const LIVE_WINDOW_GRACE_SECONDS = 120;
const LIVE_PREFETCH_SECONDS = 12;
const LIVE_REFRESH_MS = 3000;
// A camera with nothing in a live window (offline) is re-read less often.
const EMPTY_REFRESH_MS = 30000;
// Pulling a lagging tile back to the clock rebuffers it; a slow tile gets at
// most one correction per interval so it cannot stutter on repeated seeks.
const DRIFT_CORRECTION_INTERVAL_MS = 3000;
// A held frame is a bridge across a reload, never a stale picture: it goes
// when the playhead moves away from it or the reload takes too long.
const HOLD_MAX_DRIFT_SECONDS = 3;
const HOLD_MAX_MS = 4000;

/**
 * One camera in the All cameras grid. The page playhead is the only clock:
 * the tile plays its camera's window playlist through the shared recording
 * player (native HLS starts at EXT-X-START), applies play/pause at once, and
 * re-seeks only after its start has settled and only when it is off the clock.
 */
function CameraGridTile({ camera, source, epoch, playing, rate, primary, canFocus, onFocus, placement }) {
  const videoRef = useRef(null);
  const previousEpochRef = useRef(null);
  const lastCorrectionRef = useRef(-Infinity);
  const loadKeyRef = useRef("");
  const playbackRef = useRef(null);
  const holdRef = useRef(null);
  const holdEpochRef = useRef(null);
  const [holding, setHolding] = useState(false);
  const [playback, setPlayback] = useState(null);
  const [error, setError] = useState("");
  const [readyManifest, setReadyManifest] = useState("");
  const [revision, setRevision] = useState(0);
  const windowStart = allCamerasWindowStart(epoch);
  const windowEnd = windowStart + ALL_CAMERAS_WINDOW_SECONDS;
  const timeline = useMemo(() => recordingPlaybackTimeline(playback?.rows || []), [playback?.rows]);
  const exactMediaTime = playbackMediaTimeForEpoch(timeline, epoch);
  // A short gap between files keeps the previous frame rather than blanking.
  const mediaTime = Number.isFinite(exactMediaTime) ? exactMediaTime : playbackMediaTimeForEpoch(timeline, epoch, 1.25);
  const briefGap = !Number.isFinite(exactMediaTime) && Number.isFinite(mediaTime);
  const hasCoverage = Number.isFinite(mediaTime);
  // Loaded outside this camera's coverage, the player still starts, on the
  // next recording (or the last one); the clock moves it once coverage begins.
  const startMediaTime = useMemo(() => {
    if (!timeline.length) return null;
    const loadEpoch = playback?.loadEpoch;
    const exact = playbackMediaTimeForEpoch(timeline, loadEpoch, 1.25);
    if (Number.isFinite(exact)) return exact;
    return (timeline.find((row) => row.end_epoch > loadEpoch) || timeline[timeline.length - 1]).media_start;
  }, [playback?.loadEpoch, timeline]);
  const manifestUrl = playback && Number.isFinite(startMediaTime)
    ? recordingDayHlsUrl(camera.id, playback.start, playback.end, playback.source, startMediaTime)
    : "";
  const showingVideo = Boolean(manifestUrl) && readyManifest === manifestUrl && hasCoverage;
  const previewEpoch = Math.floor(Math.max(0, Number(epoch) || 0) / 2) * 2;
  const lastEnd = timeline.length ? timeline[timeline.length - 1].end_epoch : null;
  const nearLoadedTail = Boolean(playback) && (lastEnd === null || (epoch >= lastEnd - LIVE_PREFETCH_SECONDS && lastEnd < windowEnd - 1));
  playbackRef.current = playback;

  // Reloading the player (a new window, or a live window that grew) keeps the
  // last frame on screen until the new start settles, instead of blanking.
  function holdFrame() {
    const video = videoRef.current;
    const canvas = holdRef.current;
    if (!video || !canvas || video.readyState < 2 || !video.videoWidth) return;
    try {
      const scale = Math.min(1, 640 / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      // Some browsers draw HLS video as black; then the still is the better cover.
      const { data } = context.getImageData(0, Math.floor(canvas.height / 2), canvas.width, 1);
      let lit = false;
      for (let index = 0; index < data.length && !lit; index += 4) lit = data[index] + data[index + 1] + data[index + 2] > 24;
      holdEpochRef.current = epoch;
      setHolding(lit);
    } catch (holdError) {
      // A frame that cannot be captured only loses the hold; the still covers.
      console.debug("All cameras: frame hold unavailable", holdError);
      setHolding(false);
    }
  }

  useEffect(() => {
    if (!holding) return undefined;
    if (showingVideo || error || (playback && !hasCoverage) || Math.abs(epoch - holdEpochRef.current) > HOLD_MAX_DRIFT_SECONDS) {
      setHolding(false);
      return undefined;
    }
    const timer = window.setTimeout(() => setHolding(false), HOLD_MAX_MS);
    return () => window.clearTimeout(timer);
  }, [epoch, error, hasCoverage, holding, playback, showingVideo]);

  useEffect(() => {
    if (!camera?.id || !windowStart) return undefined;
    const controller = new AbortController();
    let cancelled = false;
    const loadKey = `${camera.id}:${source}:${windowStart}`;
    const refresh = loadKeyRef.current === loadKey;
    loadKeyRef.current = loadKey;
    if (!refresh) {
      // Only playback running into the next window keeps its frame; after a
      // jump the old frame would be the wrong picture, so the still covers.
      if (Math.abs(epoch - (previousEpochRef.current ?? -Infinity)) <= HOLD_MAX_DRIFT_SECONDS) holdFrame();
      setPlayback(null);
      setError("");
      setReadyManifest("");
    }
    const candidates = source === "main" ? ["main", "live"] : ["live", "main"];
    const next = (candidate, payload) => ({
      source: candidate,
      start: Number(payload.start_epoch),
      end: Number(payload.end_epoch),
      rows: payload.recordings,
      loadEpoch: epoch,
    });
    async function load() {
      let failure = "";
      for (const candidate of candidates) {
        const response = await fetch(recordingWindowUrl(camera.id, windowStart, windowEnd, candidate), { signal: controller.signal });
        if (!response.ok) {
          failure = `Recording unavailable (${response.status})`;
          continue;
        }
        const payload = await response.json();
        if (!(payload.recordings || []).length) continue;
        if (cancelled) return;
        setError("");
        // A refresh only replaces what is playing when the window grew.
        const current = playbackRef.current;
        if (refresh && current?.rows?.length && current.source === candidate) {
          const grown = Math.max(...payload.recordings.map((row) => Number(row.end_epoch)));
          if (!(grown > Number(current.rows[current.rows.length - 1]?.end_epoch) + 0.5)) return;
          holdFrame();
        }
        setPlayback(next(candidate, payload));
        return;
      }
      if (cancelled) return;
      if (failure) setError(failure);
      if (!refresh) setPlayback({ source, start: windowStart, end: windowEnd, rows: [], loadEpoch: epoch });
    }
    load().catch((loadError) => {
      if (!cancelled && loadError.name !== "AbortError") setError("Recording unavailable");
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
    // epoch is read once per load; the clock effect follows it after.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera?.id, revision, source, windowStart]);

  // Near live, the window grows after it was read: re-read it as the
  // playhead nears the end of what this tile has (rarely, with nothing yet).
  useEffect(() => {
    if (!nearLoadedTail || windowEnd < Date.now() / 1000 - LIVE_WINDOW_GRACE_SECONDS) return undefined;
    const timer = window.setTimeout(() => setRevision((value) => value + 1), lastEnd === null ? EMPTY_REFRESH_MS : LIVE_REFRESH_MS);
    return () => window.clearTimeout(timer);
  }, [lastEnd, nearLoadedTail, revision, windowEnd]);

  useEffect(() => {
    const video = videoRef.current;
    const previousEpoch = previousEpochRef.current;
    previousEpochRef.current = epoch;
    if (!video) return;
    video.playbackRate = Number(rate) || 1;
    const shouldPlay = playing && hasCoverage && !briefGap;
    if (readyManifest !== manifestUrl) {
      // The shared player owns the start position until it settles.
      if (shouldPlay) video.play().catch(() => {});
      else video.pause();
      return;
    }
    // Rows can promise slightly more than the media holds; never restart it.
    const pastMedia = Number.isFinite(video.duration) && mediaTime >= video.duration - 0.25;
    if (!hasCoverage || briefGap || pastMedia) {
      video.pause();
      return;
    }
    const epochDelta = Number.isFinite(previousEpoch) ? epoch - previousEpoch : null;
    if (gridPlaybackNeedsSeek({ currentTime: video.currentTime, targetTime: mediaTime, playing, epochDelta })) {
      // Scrubs and pauses seek at once; drift on a running clock is rate limited.
      const driftOnly = playing && Number.isFinite(epochDelta) && epochDelta >= 0 && epochDelta <= 2.5;
      const now = performance.now();
      if (!driftOnly || (!video.seeking && now - lastCorrectionRef.current >= DRIFT_CORRECTION_INTERVAL_MS)) {
        if (driftOnly) lastCorrectionRef.current = now;
        video.currentTime = mediaTime;
      }
    }
    if (shouldPlay && !video.ended) video.play().catch(() => {});
    else video.pause();
  }, [briefGap, epoch, hasCoverage, manifestUrl, mediaTime, playing, rate, readyManifest]);

  const status = error || (playback && !hasCoverage ? "No recording" : "");
  const fallback = playback?.rows?.length && playback.source !== source ? (playback.source === "main" ? "Main" : "Sub") : "";
  return (
    <article className={`all-cameras-tile${primary ? " primary" : ""}${hasCoverage ? "" : " gap"}`} data-camera-id={camera.id} style={placement}>
      {/* The still covers loading and gaps; while video shows, it stops requesting frames. */}
      {showingVideo || holding ? null : <img className="all-cameras-still" src={recordingPreviewUrl(camera.id, previewEpoch, source)} alt="" />}
      {manifestUrl ? (
        <RecordingHlsVideo
          ref={videoRef}
          src={manifestUrl}
          startTime={startMediaTime}
          bufferingGoal={10}
          autoPlay={playing && hasCoverage}
          muted
          playsInline
          preload="auto"
          onReady={(_player, video) => {
            setError("");
            setReadyManifest(manifestUrl);
            // Follow the clock if it moved on while the start settled.
            if (Number.isFinite(mediaTime) && Math.abs(video.currentTime - mediaTime) > 0.25) video.currentTime = mediaTime;
            if (playing && hasCoverage) video.play().catch(() => {});
          }}
          onError={() => setError("Playback unavailable")}
        />
      ) : null}
      {/* After the video: while a tile reloads, its held frame covers the player. */}
      <canvas ref={holdRef} className="all-cameras-still all-cameras-hold" hidden={!holding || showingVideo} aria-hidden="true" />
      {status ? <span className="all-cameras-status">{status}</span> : null}
      <button
        type="button"
        className="all-cameras-tile-hit media-surface-action"
        onClick={() => onFocus(camera.id)}
        disabled={!canFocus}
        aria-pressed={canFocus ? primary : undefined}
        aria-label={canFocus ? `Focus ${camera.name}` : camera.name}
      >
        <span className="all-cameras-label">
          <Camera size={13} aria-hidden="true" />{camera.name}
          {fallback ? <em title={`Playing the ${fallback} stream; the selected one has no recording here`}>{fallback}</em> : null}
        </span>
        {canFocus && !primary ? <Maximize2 size={14} aria-hidden="true" /> : null}
      </button>
    </article>
  );
}

const GRID_GAP = 3;

/**
 * All cameras: the cameras with recording at the Timeline playhead, in the
 * spot of the single-camera player, laid out like Live (Automatic's uniform
 * grid, or Focus's mosaic around one primary camera). Tiles stay in camera
 * order in the DOM and are only positioned, so a layout change never moves or
 * reloads a tile that stays shown; a tile that leaves (another page, another
 * set) stops its player, which is what keeps fewer streams running.
 */
export function AllCamerasGrid({ cameras, source, epoch, playing, rate, layoutMode, focusCameraId, canFocus, onFocusCamera }) {
  const gridRef = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const grid = gridRef.current;
    if (!grid) return undefined;
    const measure = () => setSize((current) => (current.width === grid.clientWidth && current.height === grid.clientHeight ? current : { width: grid.clientWidth, height: grid.clientHeight }));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(grid);
    return () => observer.disconnect();
  }, [cameras.length > 0]);
  const placements = useMemo(() => {
    const layout = layoutMode === "focus"
      ? focusLiveMosaicLayout(cameras, size.width, size.height, focusCameraId, GRID_GAP)
      : uniformLiveGridLayout(cameras, size.width, size.height, GRID_GAP);
    // Live's uniform grid is top-aligned; in the player's spot it is centred.
    const bottom = layout.reduce((edge, item) => Math.max(edge, item.y + item.height), 0);
    const offsetY = layoutMode === "focus" ? 0 : Math.max(0, (size.height - bottom) / 2);
    return new Map(layout.map((item) => [item.camera.id, {
      primary: Boolean(item.primary),
      style: { left: item.x, top: item.y + offsetY, width: item.width, height: item.height },
    }]));
  }, [cameras, focusCameraId, layoutMode, size.height, size.width]);
  if (!cameras.length) return <div className="all-cameras-grid empty"><span className="all-cameras-status">No camera has recording in this 15-minute window</span></div>;
  return (
    <div ref={gridRef} className={`all-cameras-grid layout-${layoutMode}`} aria-label="All cameras">
      {cameras.map((camera) => {
        const placement = placements.get(camera.id);
        return (
          <CameraGridTile
            key={camera.id}
            camera={camera}
            source={source}
            epoch={epoch}
            playing={playing}
            rate={rate}
            primary={layoutMode === "focus" && Boolean(placement?.primary)}
            canFocus={canFocus}
            onFocus={onFocusCamera}
            placement={placement ? placement.style : { visibility: "hidden" }}
          />
        );
      })}
    </div>
  );
}

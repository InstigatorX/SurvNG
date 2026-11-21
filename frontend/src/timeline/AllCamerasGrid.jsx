import React, { useEffect, useMemo, useRef, useState } from "react";
import { Camera, Maximize2, Minimize2 } from "lucide-react";
import { gridPlaybackNeedsSeek, playbackMediaTimeForEpoch, recordingPlaybackTimeline } from "../recordingPlayback.mjs";
import { allCamerasGridColumns } from "../recordingGrid.mjs";
import { fetch } from "../shared/api.js";
import { recordingDayHlsUrl, recordingPreviewUrl, recordingWindowUrl } from "../shared/mediaUrls.js";
import { RecordingHlsVideo } from "../shared/RecordingHlsVideo.jsx";

// Each tile plays one 15-minute window playlist, the same unit Timeline uses.
const WINDOW_SECONDS = 15 * 60;
// A window this recent may still be growing; a tile at its end re-reads it.
const LIVE_WINDOW_GRACE_SECONDS = 120;
const LIVE_REFRESH_MS = 5000;
// A camera with nothing in a live window (offline) is re-read less often.
const EMPTY_REFRESH_MS = 30000;
// Pulling a lagging tile back to the clock rebuffers it; a slow tile gets at
// most one correction per interval so it cannot stutter on repeated seeks.
const DRIFT_CORRECTION_INTERVAL_MS = 3000;

/**
 * One camera in the All cameras grid. The page playhead is the only clock:
 * the tile plays its camera's window playlist through the shared recording
 * player (native HLS starts at EXT-X-START), applies play/pause at once, and
 * re-seeks only after its start has settled and only when it is off the clock.
 */
function CameraGridTile({ camera, source, epoch, playing, rate, enlarged, onToggle }) {
  const videoRef = useRef(null);
  const previousEpochRef = useRef(null);
  const lastCorrectionRef = useRef(-Infinity);
  const loadKeyRef = useRef("");
  const [playback, setPlayback] = useState(null);
  const [error, setError] = useState("");
  const [readyManifest, setReadyManifest] = useState("");
  const [revision, setRevision] = useState(0);
  const windowStart = Math.floor(Math.max(0, Number(epoch) || 0) / WINDOW_SECONDS) * WINDOW_SECONDS;
  const windowEnd = windowStart + WINDOW_SECONDS;
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
  const atLoadedTail = Boolean(playback) && (lastEnd === null || (epoch >= lastEnd - 1 && lastEnd < windowEnd - 1));

  useEffect(() => {
    if (!camera?.id || !windowStart) return undefined;
    const controller = new AbortController();
    let cancelled = false;
    const loadKey = `${camera.id}:${source}:${windowStart}`;
    const refresh = loadKeyRef.current === loadKey;
    loadKeyRef.current = loadKey;
    if (!refresh) {
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
        setPlayback((current) => {
          if (!refresh || !current?.rows?.length || current.source !== candidate) return next(candidate, payload);
          const grown = Math.max(...payload.recordings.map((row) => Number(row.end_epoch)));
          return grown > Number(current.rows[current.rows.length - 1]?.end_epoch) + 0.5 ? next(candidate, payload) : current;
        });
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

  // Near live, the window grows after it was read: re-read it while the
  // playhead waits at the end of what this tile has.
  useEffect(() => {
    if (!atLoadedTail || windowEnd < Date.now() / 1000 - LIVE_WINDOW_GRACE_SECONDS) return undefined;
    const timer = window.setTimeout(() => setRevision((value) => value + 1), lastEnd === null ? EMPTY_REFRESH_MS : LIVE_REFRESH_MS);
    return () => window.clearTimeout(timer);
  }, [atLoadedTail, lastEnd, revision, windowEnd]);

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
    <article className={`all-cameras-tile${enlarged ? " enlarged" : ""}${hasCoverage ? "" : " gap"}`} data-camera-id={camera.id}>
      {/* The still covers loading and gaps; while video shows, it stops requesting frames. */}
      {showingVideo ? null : <img className="all-cameras-still" src={recordingPreviewUrl(camera.id, previewEpoch, source)} alt="" />}
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
      {status ? <span className="all-cameras-status">{status}</span> : null}
      <button
        type="button"
        className="all-cameras-tile-hit media-surface-action"
        onClick={() => onToggle(camera.id)}
        aria-pressed={enlarged}
        aria-label={enlarged ? `Show ${camera.name} in the grid` : `Enlarge ${camera.name}`}
      >
        <span className="all-cameras-label">
          <Camera size={13} aria-hidden="true" />{camera.name}
          {fallback ? <em title={`Playing the ${fallback} stream; the selected one has no recording here`}>{fallback}</em> : null}
        </span>
        {enlarged ? <Minimize2 size={14} aria-hidden="true" /> : <Maximize2 size={14} aria-hidden="true" />}
      </button>
    </article>
  );
}

/**
 * All cameras: every camera's recording at the Timeline playhead, in the
 * spot of the single-camera player. Tapping a tile enlarges it (the others
 * move to a strip below); tapping it again returns to the even grid.
 */
export function AllCamerasGrid({ cameras, source, epoch, playing, rate, enlargedCameraId, onEnlargedChange }) {
  const enlarged = cameras.find((camera) => camera.id === enlargedCameraId) || null;
  const others = enlarged ? cameras.filter((camera) => camera.id !== enlarged.id) : cameras;
  const toggle = (cameraId) => onEnlargedChange(cameraId === enlargedCameraId ? null : cameraId);
  const tile = (camera, isEnlarged) => (
    <CameraGridTile
      key={camera.id}
      camera={camera}
      source={source}
      epoch={epoch}
      playing={playing}
      rate={rate}
      enlarged={isEnlarged}
      onToggle={toggle}
    />
  );
  if (!cameras.length) return <div className="all-cameras-grid empty"><span className="all-cameras-status">No cameras</span></div>;
  // Every tile stays in one container in camera order, so enlarging only
  // changes CSS placement and never reloads a tile's video.
  const strip = enlarged && others.length > 0;
  const columns = strip ? others.length : enlarged ? 1 : allCamerasGridColumns(cameras.length);
  return (
    <div className={`all-cameras-grid${strip ? " has-enlarged" : ""}`} style={{ "--all-cameras-columns": columns }} aria-label="All cameras">
      {cameras.map((camera) => tile(camera, camera.id === enlarged?.id))}
    </div>
  );
}

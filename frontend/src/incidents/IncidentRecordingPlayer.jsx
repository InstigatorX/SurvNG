import React, { useEffect, useRef, useState } from "react";
import { Pause, Play, SkipBack, SkipForward, Volume2, VolumeX, X } from "lucide-react";
import {
  isRecordingCompatibilityError,
  playbackMediaTimeForEpoch,
  recordingPlaybackTransport,
  recordingPlayableEpoch,
  recordingSegmentAt,
  recordingSegmentLocalTime,
  recordingEpochAfterSegment,
  seekVideoToTime,
  supportsNativeRecordingHls,
} from "../recordingPlayback.mjs";
import { fetch } from "../shared/api.js";
import { formatTimeOnly } from "../shared/format.js";
import { recordingDayHlsUrl, recordingSegmentUrl, recordingWindowUrl } from "../shared/mediaUrls.js";
import { NativeRecordingVideo } from "../shared/NativeRecordingVideo.jsx";
import { RecordingHlsVideo } from "../shared/RecordingHlsVideo.jsx";
import { DebugDetectionOverlay, StoredTrackVideoOverlay } from "../shared/evidence.jsx";
import { storedObjectTracks } from "../objectTrackReplay.mjs";

const WINDOW_SECONDS = 15 * 60;

export function incidentRecordingBounds(event) {
  if (!event?.camera_id) return null;
  const sceneStart = Number(event.scene_clip_window?.start);
  const sceneEnd = Number(event.scene_clip_window?.end);
  if (Number.isFinite(sceneStart) && Number.isFinite(sceneEnd) && sceneEnd > sceneStart) {
    return { cameraId: event.camera_id, startEpoch: sceneStart, endEpoch: sceneEnd };
  }
  const startAt = Date.parse(event.start_at);
  const startEpoch = Number.isFinite(Number(event.start_epoch))
    ? Number(event.start_epoch)
    : (Number.isFinite(startAt) ? startAt / 1000 : null);
  const endAt = Date.parse(event.end_at);
  const endEpoch = Number.isFinite(Number(event.last_epoch))
    ? Number(event.last_epoch)
    : Number.isFinite(Number(event.end_epoch))
      ? Number(event.end_epoch)
      : (Number.isFinite(endAt) ? endAt / 1000 : null);
  if (Number.isFinite(startEpoch)) {
    const end = Number.isFinite(endEpoch) && endEpoch > startEpoch ? endEpoch : startEpoch + 1;
    return { cameraId: event.camera_id, startEpoch, endEpoch: end };
  }
  const created = Date.parse(event.created_at);
  if (!Number.isFinite(created)) return null;
  return { cameraId: event.camera_id, startEpoch: created / 1000, endEpoch: created / 1000 + 1 };
}

function playbackWindowForEpoch(epoch) {
  const start = Math.floor(Math.max(0, epoch) / WINDOW_SECONDS) * WINDOW_SECONDS;
  return { start, end: start + WINDOW_SECONDS };
}

// Match the timeline playlist clock: media time is concatenated segment duration.
function withMediaTimeline(rows) {
  let mediaOffset = 0;
  return (rows || [])
    .map((item) => ({
      ...item,
      start_epoch: Number(item.start_epoch),
      end_epoch: Number(item.end_epoch),
    }))
    .filter((item) => Number.isFinite(item.start_epoch) && Number.isFinite(item.end_epoch) && item.end_epoch > item.start_epoch)
    .sort((left, right) => left.start_epoch - right.start_epoch)
    .map((item) => {
      const duration = Math.max(0.01, Number(item.duration_seconds) || item.end_epoch - item.start_epoch);
      const result = { ...item, media_start: mediaOffset, media_end: mediaOffset + duration };
      mediaOffset += duration;
      return result;
    });
}

function rowsOverlapping(rows, startEpoch, endEpoch) {
  return rows.filter((row) => row.end_epoch > startEpoch && row.start_epoch < endEpoch);
}

function epochForMediaTime(rows, mediaTime) {
  if (!Number.isFinite(mediaTime)) return null;
  const clip = rows.find((item) => item.media_start <= mediaTime && mediaTime < item.media_end) || rows[rows.length - 1];
  if (!clip) return null;
  const offset = Math.max(0, Math.min(clip.end_epoch - clip.start_epoch, mediaTime - clip.media_start));
  return clip.start_epoch + offset;
}

function overlayClock(rows, mediaTime, segment) {
  if (segment) return { windowStartEpoch: segment.start_epoch, mediaStartTime: 0 };
  const clip = rows.find((item) => item.media_start <= mediaTime && mediaTime < item.media_end) || rows[0];
  if (!clip) return { windowStartEpoch: null, mediaStartTime: null };
  return { windowStartEpoch: clip.start_epoch - clip.media_start, mediaStartTime: 0 };
}

export function IncidentRecordingPlayer({
  cameraId,
  startEpoch,
  endEpoch,
  source = "main",
  autoPlay = true,
  analysisMode = "clean",
  depthLayer = "both",
  trackingEvent = null,
  timeZone,
  confidence = 0.35,
  onAnalysisStats,
  onEnded,
  onClose,
}) {
  const videoRef = useRef(null);
  const endedRef = useRef(false);
  const scrubbingRef = useRef(false);
  const appliedSeekRef = useRef(startEpoch);
  const readySourceRef = useRef("");
  const ignorePauseRef = useRef(0);
  const [nativeHls] = useState(supportsNativeRecordingHls);
  const [transport, setTransport] = useState(() => recordingPlaybackTransport({ nativeHls: supportsNativeRecordingHls(), rate: 1 }));
  const [loaded, setLoaded] = useState(null);
  const [targetEpoch, setTargetEpoch] = useState(startEpoch);
  const [playhead, setPlayhead] = useState(startEpoch);
  const [playing, setPlaying] = useState(autoPlay);
  const [muted, setMuted] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [clock, setClock] = useState({ windowStartEpoch: null, mediaStartTime: null });
  const safeEnd = Number.isFinite(endEpoch) && endEpoch > startEpoch ? endEpoch : startEpoch + 1;
  const fetchEpoch = targetEpoch < startEpoch || targetEpoch > safeEnd ? startEpoch : targetEpoch;
  const windowStart = playbackWindowForEpoch(fetchEpoch).start;

  useEffect(() => {
    endedRef.current = false;
    readySourceRef.current = "";
    ignorePauseRef.current = performance.now() + 1500;
    appliedSeekRef.current = startEpoch;
    setTargetEpoch(startEpoch);
    setPlayhead(startEpoch);
    setPlaying(autoPlay);
    setError("");
    setTransport(recordingPlaybackTransport({ nativeHls, rate: 1 }));
  }, [autoPlay, cameraId, endEpoch, nativeHls, source, startEpoch]);

  useEffect(() => {
    if (!cameraId || !Number.isFinite(startEpoch)) return undefined;
    const controller = new AbortController();
    let cancelled = false;
    const requested = playbackWindowForEpoch(fetchEpoch);
    const candidates = source === "live" ? ["live", "main"] : ["main", "live"];
    setLoading(true);
    setError("");
    async function load() {
      for (const candidate of candidates) {
        const response = await fetch(
          recordingWindowUrl(cameraId, requested.start, requested.end, candidate),
          { signal: controller.signal },
        );
        if (!response.ok) continue;
        const payload = await response.json();
        const rows = rowsOverlapping(withMediaTimeline(payload.recordings || []), startEpoch, safeEnd);
        const playable = recordingPlayableEpoch(rows, fetchEpoch);
        if (!rows.length || !Number.isFinite(playable)) continue;
        if (cancelled) return;
        setLoaded({
          cameraId,
          source: payload.source || candidate,
          start: Number(payload.start_epoch),
          end: Number(payload.end_epoch),
          rows,
          seekEpoch: playable,
          revision: Date.now(),
        });
        setPlayhead(playable);
        setLoading(false);
        return;
      }
      if (!cancelled) {
        setLoaded(null);
        setLoading(false);
        setError("No recording window found");
      }
    }
    load().catch((loadError) => {
      if (!cancelled && loadError.name !== "AbortError") {
        setLoaded(null);
        setLoading(false);
        setError("No recording window found");
      }
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [cameraId, safeEnd, source, startEpoch, windowStart]);

  const rows = loaded?.rows || [];
  const playable = recordingPlayableEpoch(rows, loaded?.seekEpoch);
  // HLS owns a whole window; native playback owns the segment at the latest
  // requested epoch, including seeks and automatic segment handoffs.
  const segment = transport === "hls" || Number(loaded?.start) !== windowStart
    ? null : recordingSegmentAt(rows, recordingPlayableEpoch(rows, targetEpoch));
  const mediaTime = transport === "hls" ? playbackMediaTimeForEpoch(rows, playable) : null;
  const manifestUrl = transport === "hls" && loaded && Number.isFinite(mediaTime)
    ? `${recordingDayHlsUrl(loaded.cameraId, loaded.start, loaded.end, loaded.source, mediaTime)}&reload=${loaded.revision}`
    : "";
  const segmentUrl = segment
    ? `${recordingSegmentUrl(loaded.cameraId, segment.start_epoch, loaded.source, transport === "transcode")}&reload=${loaded.revision}`
    : "";
  const playbackUrl = manifestUrl || segmentUrl;

  function mediaEpoch(video) {
    if (!video || !rows.length) return null;
    if (transport !== "hls" && segment) return segment.start_epoch + Number(video.currentTime || 0);
    return epochForMediaTime(rows, video.currentTime);
  }

  function seekVideo(video, epoch) {
    if (!video || !rows.length || !Number.isFinite(epoch)) return;
    const next = recordingPlayableEpoch(rows, epoch);
    if (!Number.isFinite(next)) return;
    // A cross-segment seek is applied when the new source delivers metadata,
    // never to the outgoing element using the incoming segment's clock.
    if (transport !== "hls" && (!segment || next < segment.start_epoch || next >= segment.end_epoch)) return;
    const local = transport === "hls"
      ? playbackMediaTimeForEpoch(rows, next)
      : recordingSegmentLocalTime(segment, next, video);
    if (!Number.isFinite(local)) return;
    if (Math.abs(Number(video.currentTime || 0) - local) > 0.35) seekVideoToTime(video, local);
    setClock(overlayClock(rows, transport === "hls" ? local : (segment ? segment.media_start + local : local), transport === "hls" ? null : segment));
  }

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !loaded || scrubbingRef.current || readySourceRef.current !== playbackUrl) return;
    if (Math.abs(targetEpoch - appliedSeekRef.current) <= 0.35) return;
    if (Math.abs(playbackWindowForEpoch(targetEpoch).start - Number(loaded.start)) > 1) return;
    appliedSeekRef.current = targetEpoch;
    seekVideo(video, targetEpoch);
    if (playing) video.play?.().catch(() => { });
  }, [loaded, targetEpoch, transport, playbackUrl]);

  function finishEpisode() {
    if (endedRef.current) return;
    endedRef.current = true;
    onEnded?.();
  }

  function handleTimeUpdate(event) {
    if (scrubbingRef.current || readySourceRef.current !== playbackUrl) return;
    const video = event.currentTarget;
    const epoch = mediaEpoch(video);
    if (!Number.isFinite(epoch) || epoch < startEpoch - 0.5) return;
    setPlayhead(epoch);
    setClock(overlayClock(rows, transport === "hls" ? video.currentTime : (segment ? (segment.media_start || 0) + video.currentTime : video.currentTime), transport === "hls" ? null : segment));
    if (epoch >= safeEnd - 0.05) {
      video.pause?.();
      finishEpisode();
      return;
    }
    if (loaded && epoch >= loaded.end - 0.25 && safeEnd > loaded.end + 0.05) setTargetEpoch(Math.min(safeEnd, loaded.end + 0.01));
  }

  function handleEnded(event) {
    const video = event?.currentTarget;
    if (!video?.ended) {
      finishEpisode();
      return;
    }
    const epoch = mediaEpoch(video);
    if (transport !== "hls") {
      const next = recordingEpochAfterSegment(segment, rows);
      if (Number.isFinite(next) && next < safeEnd) {
        setTargetEpoch(next);
        setPlaying(true);
        return;
      }
    }
    if (loaded && Number.isFinite(epoch) && epoch < safeEnd - 0.25 && safeEnd > loaded.end - 0.05) {
      setTargetEpoch(Math.min(safeEnd, loaded.end + 0.01));
      setPlaying(true);
      return;
    }
    finishEpisode();
  }

  function handleError(mediaError) {
    if (transport !== "transcode" && isRecordingCompatibilityError(mediaError)) {
      setTargetEpoch(playhead);
      setTransport("transcode");
      return;
    }
    setError("No recording window found");
  }

  function scrubTo(epoch) {
    const next = Math.max(startEpoch, Math.min(safeEnd, Number(epoch)));
    if (!Number.isFinite(next)) return;
    endedRef.current = false;
    setTargetEpoch(next);
    setPlayhead(next);
    setPlaying(true);
    if (loaded && next >= loaded.start && next < loaded.end) seekVideo(videoRef.current, next);
  }

  function togglePlayback() {
    const video = videoRef.current;
    if (playing) {
      video?.pause?.();
      setPlaying(false);
      return;
    }
    endedRef.current = false;
    setPlaying(true);
    video?.play?.().catch(() => { });
  }

  const tracks = storedObjectTracks(trackingEvent);
  const showTracks = analysisMode === "tracks" && trackingEvent?.object_tracking;
  const showAnalysis = analysisMode === "ai" || analysisMode === "depth";

  return (
    <div
      className="incident-recording-player incident-video-layer"
      data-camera-id={cameraId}
      data-start-epoch={startEpoch}
      data-end-epoch={safeEnd}
      onClick={(event) => event.stopPropagation()}
    >
      {manifestUrl ? (
        <RecordingHlsVideo
          ref={videoRef}
          src={manifestUrl}
          startTime={mediaTime}
          autoPlay={playing}
          muted={muted}
          playsInline
          preload="auto"
          onReady={(_player, video) => { readySourceRef.current = playbackUrl; ignorePauseRef.current = performance.now() + 1500; seekVideo(video, targetEpoch); }}
          onTimeUpdate={handleTimeUpdate}
          onEnded={handleEnded}
          onError={handleError}
          onPlay={() => setPlaying(true)}
          onPause={() => { if (!scrubbingRef.current && performance.now() >= ignorePauseRef.current) setPlaying(false); }}
        />
      ) : null}
      {segmentUrl ? (
        <NativeRecordingVideo
          ref={videoRef}
          src={segmentUrl}
          autoPlay={playing}
          muted={muted}
          onLoadedMetadata={(event) => { readySourceRef.current = playbackUrl; ignorePauseRef.current = performance.now() + 1500; seekVideo(event.currentTarget, targetEpoch); }}
          onTimeUpdate={handleTimeUpdate}
          onEnded={handleEnded}
          onError={(event) => handleError(event.currentTarget.error)}
          onPlay={() => setPlaying(true)}
          onPause={() => { if (!scrubbingRef.current && performance.now() >= ignorePauseRef.current) setPlaying(false); }}
        />
      ) : null}
      {showTracks ? (
        <StoredTrackVideoOverlay
          videoRef={videoRef}
          tracks={tracks}
          tracking={trackingEvent.object_tracking}
          coordinateSize={{
            width: Number(trackingEvent.object_tracking.frame_width),
            height: Number(trackingEvent.object_tracking.frame_height),
          }}
          windowStartEpoch={clock.windowStartEpoch}
          mediaStartTime={clock.mediaStartTime}
          mediaKey={manifestUrl || segmentUrl}
          sampleFps={trackingEvent.object_tracking.sample_fps}
          lostTimeoutSeconds={trackingEvent.object_tracking.lost_timeout_seconds}
        />
      ) : null}
      <DebugDetectionOverlay
        videoRef={videoRef}
        active={showAnalysis && playing && !loading}
        depth={analysisMode === "depth"}
        depthLayer={depthLayer}
        confidence={confidence}
        onStats={onAnalysisStats}
      />
      {loading ? <div className="incident-video-status preparing">Loading recording…</div> : null}
      {error ? <div className="incident-video-status">{error}</div> : null}
      <div className="incident-recording-controls">
        <button type="button" onClick={() => scrubTo(playhead - 10)} aria-label="Back 10 seconds"><SkipBack size={16} /></button>
        <button type="button" className="primary" onClick={togglePlayback} aria-label={playing ? "Pause recording" : "Play recording"}>
          {playing ? <Pause size={16} /> : <Play size={16} fill="currentColor" />}
        </button>
        <button type="button" onClick={() => scrubTo(playhead + 10)} aria-label="Forward 10 seconds"><SkipForward size={16} /></button>
        <button type="button" onClick={() => setMuted((current) => !current)} aria-label={muted ? "Unmute recording" : "Mute recording"} aria-pressed={muted}>
          {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
        </button>
        <input
          type="range"
          min={startEpoch}
          max={safeEnd}
          step="0.1"
          value={Math.max(startEpoch, Math.min(safeEnd, playhead))}
          aria-label="Incident recording position"
          onPointerDown={() => { scrubbingRef.current = true; }}
          onPointerUp={() => { scrubbingRef.current = false; }}
          onChange={(event) => scrubTo(event.target.value)}
        />
        <time>{formatTimeOnly(playhead, timeZone)}</time>
      </div>
      {onClose ? <button type="button" className="incident-recording-close" onClick={onClose} aria-label="Close playback"><X size={18} /></button> : null}
    </div>
  );
}

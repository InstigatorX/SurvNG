import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { nativePlaylistStart, nativePlaylistStartStep, seekVideoToTime, supportsNativeRecordingHls } from "../recordingPlayback.mjs";
import { fetch } from "./api.js";
import { ShakaVideo } from "./media.jsx";

const NATIVE_START_EVENTS = ["loadeddata", "canplay", "playing", "seeking", "seeked", "timeupdate"];

// Let Safari own the playlist and buffer across recording boundaries. Other
// browsers use Shaka's MSE path; neither path requests a video transcode.
//
// Native startup at an offset follows the playlist's EXT-X-START: writing
// currentTime while Safari loads can interrupt the seek it is already doing.
// The element stays covered and muted until the position settles (one
// correction at most, via nativePlaylistStartStep, shared with Timeline), and
// onReady reports the settled element. onStartNeedsPlay fires when a paused
// start cannot be judged until playback begins (including a rejected play()).
// While the start settles the video itself is hidden (not covered), so any
// poster or background behind it stays visible.
const NATIVE_START_WATCHDOG_MS = 4000;
export const RecordingHlsVideo = forwardRef(function RecordingHlsVideo({
  src, startTime, bufferingGoal, mimeType = "application/vnd.apple.mpegurl",
  onReady, onError, onStartNeedsPlay, muted = false, ...videoProps
}, forwardedRef) {
  const videoRef = useRef(null);
  const callbacks = useRef({});
  callbacks.current = {
    src, startTime, onReady, onError, onStartNeedsPlay, muted, autoPlay: Boolean(videoProps.autoPlay),
  };
  const [nativeHls] = useState(supportsNativeRecordingHls);
  // The source whose playlist start is still settling (set per load below).
  const [startingSrc, setStartingSrc] = useState(null);
  const covering = nativeHls && Boolean(src) && startingSrc === src;
  const playRequested = Boolean(videoProps.autoPlay);
  useImperativeHandle(forwardedRef, () => videoRef.current);

  useEffect(() => {
    if (nativeHls && videoRef.current) videoRef.current.muted = Boolean(muted) || covering;
  }, [nativeHls, muted, covering]);

  // Play requested while the start is still settling (for example the user
  // pressed play on a paused, unpositioned start): frames only arrive while
  // playing, so start the muted lead-in instead of waiting for onReady.
  useEffect(() => {
    const video = videoRef.current;
    if (!covering || !playRequested || !video?.paused) return;
    video.play?.().catch(() => callbacks.current.onStartNeedsPlay?.(video));
  }, [covering, playRequested]);

  useEffect(() => {
    if (!nativeHls) return undefined;
    const video = videoRef.current;
    const controller = new AbortController();
    let checking = false;
    let disposed = false;
    let probeTimer;
    let readyReported = false;
    let watchdog;
    const isCurrent = () => Boolean(src) && !disposed && callbacks.current.src === src
      && videoRef.current === video && video.getAttribute("src") === src;
    const target = Number(callbacks.current.startTime);
    const start = src && Number.isFinite(target) && target > 0.05 ? nativePlaylistStart(target) : null;
    setStartingSrc(start ? src : null);
    const stopStartWatch = () => {
      for (const name of NATIVE_START_EVENTS) video.removeEventListener(name, checkStart);
      window.clearTimeout(watchdog);
    };
    const needsPlay = () => callbacks.current.onStartNeedsPlay?.(video);
    const reportReady = (atTarget = true) => {
      if (!isCurrent() || readyReported || video.readyState < 1) return;
      readyReported = true;
      stopStartWatch();
      if (start) {
        const wasPlaying = !video.paused;
        video.muted = Boolean(callbacks.current.muted);
        setStartingSrc(null);
        // Same rule as Timeline: Safari can pause muted autoplay when it
        // becomes audible, or after the correction seek.
        if (callbacks.current.autoPlay && video.paused && (wasPlaying || (atTarget && start.corrected))) {
          video.play?.().catch(needsPlay);
        }
      }
      callbacks.current.onReady?.(null, video);
    };
    function checkStart(event) {
      if (!start || !isCurrent() || readyReported) return;
      let step = nativePlaylistStartStep(start, video, event.type);
      if (step.action === "needs-play") {
        needsPlay();
        return;
      }
      if (step.action === "correct") {
        try {
          seekVideoToTime(video, start.mediaTime, { allowFastSeek: false });
          return;
        } catch (error) {
          console.warn("Native HLS startup correction failed", error);
          step = { action: "settle", atTarget: false };
        }
      }
      if (step.action === "settle") reportReady(step.atTarget);
    }
    const ready = () => {
      if (!isCurrent()) return;
      if (!start) {
        reportReady();
        return;
      }
      // Frames are presented only while playing; play muted while hidden. A
      // rejected play (autoplay blocked) means the user has to start it.
      if (callbacks.current.autoPlay && video.paused) video.play?.().catch(needsPlay);
      checkStart({ type: "loadedmetadata" });
      // A start that never settles (Safari stuck seeking or never buffering)
      // must not stay hidden and muted forever: ask for play if paused,
      // otherwise reveal it where it is.
      watchdog = window.setTimeout(() => {
        if (!isCurrent() || readyReported) return;
        if (video.paused) needsPlay();
        else reportReady(false);
      }, NATIVE_START_WATCHDOG_MS);
    };
    const failed = async () => {
      if (checking || !isCurrent() || !video.error) return;
      checking = true;
      const error = { code: video.error?.code, message: video.error?.message || "HLS playback failed" };
      // Safari also reports format errors for inaccessible playlists. Do not
      // start an encoder to work around authentication or a failed request.
      if (error.code === 3 || error.code === 4) {
        probeTimer = window.setTimeout(() => controller.abort(), 10_000);
        try {
          const response = await fetch(src, { signal: controller.signal });
          if (!response.ok || !(await response.text()).trimStart().startsWith("#EXTM3U")) {
            error.category = 1;
            error.message = `Recording playlist unavailable (${response.status})`;
          }
        } catch {
          error.category = 1;
          error.message = "Recording playlist request failed";
        } finally {
          window.clearTimeout(probeTimer);
        }
      }
      if (isCurrent()) {
        callbacks.current.onError?.(error);
      }
      checking = false;
    };
    if (start) {
      video.muted = true;
      for (const name of NATIVE_START_EVENTS) video.addEventListener(name, checkStart);
    }
    video.addEventListener("loadedmetadata", ready);
    video.addEventListener("error", failed);
    // Own source changes here so cleanup releases the previous playlist before
    // loading the next one. Keep the element that received the user's play gesture.
    if (src) video.setAttribute("src", src);
    else video.removeAttribute("src");
    video.load();
    return () => {
      disposed = true;
      video.removeEventListener("loadedmetadata", ready);
      video.removeEventListener("error", failed);
      stopStartWatch();
      controller.abort();
      window.clearTimeout(probeTimer);
      // Reset the media resource, including stale buffers and pending seeks,
      // without replacing the element or its Safari playback permission.
      video.pause();
      video.removeAttribute("src");
      video.load();
    };
  }, [nativeHls, src]);

  if (!nativeHls) {
    return <ShakaVideo {...videoProps} ref={videoRef} src={src} mimeType={mimeType} muted={muted}
      startTime={startTime} bufferingGoal={bufferingGoal} onReady={onReady} onError={onError} />;
  }
  const className = [videoProps.className, covering ? "recording-startup-pending" : ""].filter(Boolean).join(" ");
  return <video {...videoProps} className={className || undefined} ref={videoRef} />;
});

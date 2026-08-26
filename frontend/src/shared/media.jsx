import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { clearWebRtcFailure, initialLiveTransport, nextNativeFallbackSource, rememberWebRtcFailure, webRtcRetryDelay } from "../liveTransport.mjs";
import { appUrl, fetch } from "./api.js";
import { DEFAULT_TIME_ZONE, PREFER_NATIVE_HLS } from "./constants.js";
import { dateKeyForTimeZone, addDaysToDateKey, zonedDateSecondToEpoch } from "./datetime.js";
import { recordingDayUrl, recordingDayHlsUrl } from "./mediaUrls.js";

export let shakaImport;
export function loadShaka() {
  if (!shakaImport) shakaImport = import("shaka-player").then((module) => module.default || module);
  return shakaImport;
}

// Touch browsers are considerably more reliable with a normal MP4 resource
// than with an fMP4 HLS playlist through MSE.  Timeline uses the same policy.
export function prefersNativeMobilePlayback() {
  return PREFER_NATIVE_HLS
    || (typeof window !== "undefined" && Boolean(window.matchMedia?.("(pointer: coarse)").matches));
}

export const ShakaVideo = forwardRef(function ShakaVideo({
  src,
  mimeType,
  startTime = null,
  bufferingGoal = 20,
  autoPlay = false,
  muted = false,
  controls = false,
  onReady,
  onError,
  ...videoProps
}, forwardedRef) {
  const videoRef = useRef(null);
  const [runtime, setRuntime] = useState(null);
  const [nativeControlsVisible, setNativeControlsVisible] = useState(prefersNativeMobilePlayback);
  const callbacksRef = useRef({ onReady, onError });
  useImperativeHandle(forwardedRef, () => videoRef.current);

  useEffect(() => {
    // Touch / iOS need controls immediately; deferred reveal leaves a paused frame with no play affordance.
    setNativeControlsVisible(prefersNativeMobilePlayback());
  }, [controls, src]);

  useEffect(() => {
    callbacksRef.current = { onReady, onError };
  }, [onReady, onError]);

  useEffect(() => {
    let disposed = false;
    let nextPlayer = null;
    let shaka = null;
    let handleError = null;
    loadShaka().then((loadedShaka) => {
      if (disposed) return;
      shaka = loadedShaka;
      shaka.polyfill.installAll();
      if (!shaka.Player.isBrowserSupported()) {
        callbacksRef.current.onError?.(new Error("This browser does not support Shaka Player"));
        return;
      }
      nextPlayer = new shaka.Player();
      handleError = (event) => callbacksRef.current.onError?.(event.detail || event);
      nextPlayer.addEventListener("error", handleError);
      nextPlayer.configure({
        streaming: {
          preferNativeHls: PREFER_NATIVE_HLS,
          bufferingGoal,
          rebufferingGoal: 1,
        },
      });
      return nextPlayer.attach(videoRef.current).then(() => {
        if (!disposed) setRuntime({ player: nextPlayer, shaka });
      });
    }).catch((error) => callbacksRef.current.onError?.(error));
    return () => {
      disposed = true;
      if (nextPlayer && handleError) nextPlayer.removeEventListener("error", handleError);
      nextPlayer?.destroy();
    };
  }, [bufferingGoal]);

  useEffect(() => {
    if (!runtime?.player || !src) return undefined;
    let cancelled = false;
    runtime.player.load(src, Number.isFinite(startTime) ? startTime : null, mimeType).then(() => {
      if (cancelled) return;
      callbacksRef.current.onReady?.(runtime.player, videoRef.current);
      if (autoPlay) videoRef.current?.play().catch(() => { });
    }).catch((error) => {
      if (!cancelled && error?.code !== runtime.shaka.util.Error.Code.LOAD_INTERRUPTED) {
        callbacksRef.current.onError?.(error);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [runtime, src, mimeType, startTime]);

  const { onPointerDown, onFocus, ...restVideoProps } = videoProps;
  return (
    <video
      ref={videoRef}
      muted={muted}
      {...restVideoProps}
      controls={Boolean(controls && nativeControlsVisible)}
      onPointerDown={(event) => {
        if (controls) setNativeControlsVisible(true);
        onPointerDown?.(event);
      }}
      onFocus={(event) => {
        if (controls) setNativeControlsVisible(true);
        onFocus?.(event);
      }}
      onError={(event) => {
        const mediaError = event.currentTarget.error;
        callbacksRef.current.onError?.(mediaError
          ? { code: mediaError.code, message: mediaError.message || "Native media playback failed" }
          : new Error("Native media playback failed"));
      }}
    />
  );
});

export function RecordingFallback({ cameraId, source, timeZone, muted, controls, onReady, onError }) {
  const [playback, setPlayback] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const day = dateKeyForTimeZone(Date.now(), timeZone || DEFAULT_TIME_ZONE);
    const nextDay = addDaysToDateKey(day, 1);
    const start = zonedDateSecondToEpoch(day, 0, timeZone || DEFAULT_TIME_ZONE);
    const end = zonedDateSecondToEpoch(nextDay, 0, timeZone || DEFAULT_TIME_ZONE);
    const candidates = source === "live" ? ["live", "main"] : ["main"];

    async function load() {
      for (const candidate of candidates) {
        const response = await fetch(recordingDayUrl(cameraId, start, end, candidate));
        if (!response.ok) continue;
        const payload = await response.json();
        const rows = Array.isArray(payload) ? payload : payload.recordings;
        if (!Array.isArray(rows) || !rows.length) continue;
        const duration = rows.reduce((total, row) => total + Math.max(0, Number(row.duration_seconds) || 0), 0);
        if (!cancelled) {
          setPlayback({
            src: recordingDayHlsUrl(cameraId, start, end, candidate),
            startTime: Math.max(0, duration - 3),
          });
        }
        return;
      }
      throw new Error("No near-live recording is available");
    }

    load().catch((error) => !cancelled && onError?.(error));
    return () => { cancelled = true; };
  }, [cameraId, source, timeZone, onError]);

  if (!playback) return null;
  return (
    <ShakaVideo
      src={playback.src}
      mimeType="application/vnd.apple.mpegurl"
      startTime={playback.startTime}
      bufferingGoal={6}
      autoPlay
      muted={muted}
      controls={controls}
      playsInline
      onReady={(_player, video) => onReady?.(video, "recording", source)}
      onError={onError}
    />
  );
}

export const WebRtcLive = forwardRef(function WebRtcLive({
  cameraId,
  source = "live",
  timeZone = DEFAULT_TIME_ZONE,
  muted = true,
  controls = false,
  showPoster = true,
  onReady,
  onStageChange,
}, forwardedRef) {
  const videoRef = useRef(null);
  const [stage, setStage] = useState(() => initialLiveTransport(cameraId, source));
  const [deliverySource, setDeliverySource] = useState(source);
  const [snapshotToken, setSnapshotToken] = useState(() => Date.now());
  const [videoReady, setVideoReady] = useState(false);
  const [nativeControlsVisible, setNativeControlsVisible] = useState(false);
  useImperativeHandle(forwardedRef, () => videoRef.current);

  useEffect(() => {
    setStage(initialLiveTransport(cameraId, source));
    setDeliverySource(source);
    setVideoReady(false);
    setNativeControlsVisible(false);
  }, [cameraId, source]);

  useEffect(() => {
    setVideoReady(false);
    setNativeControlsVisible(false);
  }, [controls, stage]);

  useEffect(() => {
    onStageChange?.(stage, deliverySource);
  }, [deliverySource, stage, onStageChange]);

  useEffect(() => {
    if (stage !== "webrtc") return undefined;
    let disposed = false;
    let mediaReady = false;
    let socket = null;
    let peer = null;
    let disconnectTimer = null;
    const outgoingCandidates = [];
    const incomingCandidates = [];
    const video = videoRef.current;
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const fallback = () => {
      if (!disposed) {
        rememberWebRtcFailure(cameraId, deliverySource);
        setVideoReady(false);
        setStage("mse");
      }
    };
    const failTimer = window.setTimeout(() => !mediaReady && fallback(), 2500);

    const markMediaReady = () => {
      mediaReady = true;
      setVideoReady(true);
      window.clearTimeout(failTimer);
      clearWebRtcFailure(cameraId, deliverySource);
    };
    video?.addEventListener("loadeddata", markMediaReady);

    try {
      peer = new RTCPeerConnection({
        bundlePolicy: "max-bundle",
        iceServers: [{ urls: ["stun:stun.cloudflare.com:3478", "stun:stun.l.google.com:19302"] }],
      });
      peer.addTransceiver("video", { direction: "recvonly" });
      peer.addTransceiver("audio", { direction: "recvonly" });
      const media = new MediaStream();
      peer.addEventListener("track", (event) => {
        media.addTrack(event.track);
        if (videoRef.current) videoRef.current.srcObject = media;
      });
      peer.addEventListener("icecandidate", (event) => {
        const candidate = event.candidate?.candidate || "";
        if (socket?.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: "webrtc/candidate", value: candidate }));
        } else {
          outgoingCandidates.push(candidate);
        }
      });
      peer.addEventListener("connectionstatechange", () => {
        if (peer.connectionState === "connected") {
          window.clearTimeout(disconnectTimer);
          disconnectTimer = null;
        } else if (peer.connectionState === "failed" && !disposed) {
          fallback();
        } else if (peer.connectionState === "disconnected" && !disposed && !disconnectTimer) {
          disconnectTimer = window.setTimeout(fallback, 1000);
        }
      });
      socket = new WebSocket(`${protocol}//${location.host}${appUrl(`/api/cameras/${encodeURIComponent(cameraId)}/webrtc?source=${encodeURIComponent(deliverySource)}`)}`);
      socket.addEventListener("open", async () => {
        try {
          const offer = await peer.createOffer();
          await peer.setLocalDescription(offer);
          if (socket.readyState !== WebSocket.OPEN) return;
          socket.send(JSON.stringify({ type: "webrtc/offer", value: offer.sdp }));
          outgoingCandidates.splice(0).forEach((candidate) => {
            socket.send(JSON.stringify({ type: "webrtc/candidate", value: candidate }));
          });
        } catch (_error) {
          fallback();
        }
      });
      socket.addEventListener("message", (event) => {
        let message;
        try {
          message = JSON.parse(event.data);
        } catch {
          fallback();
          return;
        }
        if (message.type === "webrtc/answer") {
          peer.setRemoteDescription({ type: "answer", sdp: message.value }).then(async () => {
            for (const candidate of incomingCandidates.splice(0)) {
              await peer.addIceCandidate({ candidate, sdpMid: "0" });
            }
          }).catch(fallback);
        } else if (message.type === "webrtc/candidate" && message.value) {
          if (peer.remoteDescription) {
            peer.addIceCandidate({ candidate: message.value, sdpMid: "0" }).catch(fallback);
          } else {
            incomingCandidates.push(message.value);
          }
        } else if (message.type === "error") {
          fallback();
        }
      });
      socket.addEventListener("error", fallback);
      socket.addEventListener("close", () => !disposed && fallback());
    } catch (_error) {
      fallback();
    }

    return () => {
      disposed = true;
      window.clearTimeout(failTimer);
      window.clearTimeout(disconnectTimer);
      socket?.close();
      peer?.close();
      video?.removeEventListener("loadeddata", markMediaReady);
      if (videoRef.current) videoRef.current.srcObject = null;
    };
  }, [cameraId, deliverySource, stage]);

  useEffect(() => {
    if (stage !== "mse") return undefined;
    const MediaSourceApi = window.ManagedMediaSource || window.MediaSource;
    if (!MediaSourceApi) {
      const fallbackSource = nextNativeFallbackSource(source, deliverySource);
      setVideoReady(false);
      if (fallbackSource) {
        setDeliverySource(fallbackSource);
        setStage(initialLiveTransport(cameraId, fallbackSource));
      } else {
        setStage("mjpeg");
      }
      return undefined;
    }

    let disposed = false;
    let socket = null;
    let mediaSource = null;
    let sourceBuffer = null;
    let objectUrl = "";
    let socketOpen = false;
    let mediaSourceOpen = false;
    let requestSent = false;
    let ready = false;
    let queuedBytes = 0;
    const queue = [];
    const video = videoRef.current;
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const fallback = () => {
      if (disposed) return;
      const fallbackSource = nextNativeFallbackSource(source, deliverySource);
      setVideoReady(false);
      if (fallbackSource) {
        setDeliverySource(fallbackSource);
        setStage(initialLiveTransport(cameraId, fallbackSource));
        setSnapshotToken(Date.now());
      } else {
        setStage("mjpeg");
      }
    };
    const markReady = () => {
      ready = true;
      setVideoReady(true);
      window.clearTimeout(failTimer);
    };
    let lastDataAt = Date.now();
    const failTimer = window.setTimeout(() => !ready && fallback(), 6000);
    const stallTimer = window.setInterval(() => {
      if (ready && Date.now() - lastDataAt > 8000) fallback();
    }, 2000);
    const codecs = [
      "avc1.640029",
      "avc1.64002A",
      "avc1.640033",
      "hvc1.1.6.L153.B0",
      "mp4a.40.2",
      "mp4a.40.5",
      "flac",
      "opus",
    ].filter((codec) => MediaSourceApi.isTypeSupported(`video/mp4; codecs="${codec}"`)).join();

    if (!video || !codecs) {
      window.clearTimeout(failTimer);
      window.clearInterval(stallTimer);
      fallback();
      return undefined;
    }

    function sendRequest() {
      if (requestSent || !socketOpen || !mediaSourceOpen) return;
      requestSent = true;
      socket.send(JSON.stringify({ type: "mse", value: codecs }));
    }

    function pump() {
      if (!sourceBuffer || sourceBuffer.updating || mediaSource?.readyState !== "open") return;
      try {
        if (sourceBuffer.buffered.length) {
          const end = sourceBuffer.buffered.end(sourceBuffer.buffered.length - 1);
          const start = sourceBuffer.buffered.start(0);
          if (end - start > 10) {
            sourceBuffer.remove(start, end - 5);
            return;
          }
          const gap = end - video.currentTime;
          if (gap > 5) video.currentTime = Math.max(start, end - 1);
          video.playbackRate = gap > 1 ? 1.1 : 1;
        }
        const data = queue.shift();
        if (!data) return;
        queuedBytes -= data.byteLength;
        sourceBuffer.appendBuffer(data);
      } catch (_error) {
        fallback();
      }
    }

    try {
      mediaSource = new MediaSourceApi();
      mediaSource.addEventListener("sourceopen", () => {
        mediaSourceOpen = true;
        sendRequest();
      }, { once: true });
      video.disableRemotePlayback = true;
      if ("ManagedMediaSource" in window && mediaSource instanceof window.ManagedMediaSource) {
        video.srcObject = mediaSource;
      } else {
        objectUrl = URL.createObjectURL(mediaSource);
        video.src = objectUrl;
      }
      video.play().catch(() => { });
      video.addEventListener("loadeddata", markReady, { once: true });

      socket = new WebSocket(`${protocol}//${location.host}${appUrl(`/api/cameras/${encodeURIComponent(cameraId)}/mse?source=${encodeURIComponent(deliverySource)}`)}`);
      socket.binaryType = "arraybuffer";
      socket.addEventListener("open", () => {
        socketOpen = true;
        sendRequest();
      });
      socket.addEventListener("message", (event) => {
        if (typeof event.data === "string") {
          let message;
          try {
            message = JSON.parse(event.data);
          } catch {
            fallback();
            return;
          }
          if (message.type === "error") {
            fallback();
          } else if (message.type === "mse" && !sourceBuffer) {
            try {
              sourceBuffer = mediaSource.addSourceBuffer(message.value);
              sourceBuffer.mode = "segments";
              sourceBuffer.addEventListener("updateend", pump);
              sourceBuffer.addEventListener("error", fallback);
              pump();
            } catch (_error) {
              fallback();
            }
          }
          return;
        }
        const data = event.data;
        if (!(data instanceof ArrayBuffer)) return;
        lastDataAt = Date.now();
        queue.push(data);
        queuedBytes += data.byteLength;
        if (queuedBytes > 8 * 1024 * 1024) {
          fallback();
          return;
        }
        pump();
      });
      socket.addEventListener("error", fallback);
      socket.addEventListener("close", () => !disposed && fallback());
    } catch (_error) {
      fallback();
    }

    return () => {
      disposed = true;
      window.clearTimeout(failTimer);
      window.clearInterval(stallTimer);
      socket?.close();
      queue.length = 0;
      if (video) {
        video.removeEventListener("loadeddata", markReady);
        video.pause();
        video.removeAttribute("src");
        video.srcObject = null;
        video.load();
      }
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [cameraId, deliverySource, source, stage]);

  useEffect(() => {
    if (stage === "webrtc") return undefined;
    const delay = Math.max(1000, webRtcRetryDelay(cameraId, deliverySource));
    const timer = window.setTimeout(() => {
      setVideoReady(false);
      setDeliverySource(source);
      setStage(initialLiveTransport(cameraId, source));
      setSnapshotToken(Date.now());
    }, delay);
    return () => window.clearTimeout(timer);
  }, [cameraId, deliverySource, source, stage]);

  useEffect(() => {
    if (stage !== "snapshot") return undefined;
    const timer = window.setInterval(() => setSnapshotToken(Date.now()), 2000);
    return () => window.clearInterval(timer);
  }, [stage]);

  const posterDeliverySource = stage === "mjpeg"
    ? deliverySource
    : deliverySource === "main" ? "live" : deliverySource;

  return (
    <div className={`live-stack ${videoReady ? "video-ready" : ""} ${showPoster ? "" : "external-poster"}`} data-stage={stage} data-video-ready={videoReady ? "true" : "false"}>
      {showPoster || !["webrtc", "mse", "recording"].includes(stage) ? (
        <img
          className="live-poster"
          src={appUrl(stage === "mjpeg"
            ? `/api/cameras/${cameraId}/stream.mjpg?source=${deliverySource}&fps=1&t=${snapshotToken}`
            : `/api/cameras/${cameraId}/snapshot.jpg?source=${posterDeliverySource}&t=${snapshotToken}`)}
          alt=""
          onLoad={(event) => ["mjpeg", "snapshot"].includes(stage) && onReady?.(event.currentTarget, stage, posterDeliverySource)}
        />
      ) : null}
      {["webrtc", "mse"].includes(stage) ? (
        <video
          ref={videoRef}
          className="live-video"
          muted={muted}
          controls={Boolean(controls && nativeControlsVisible)}
          autoPlay
          playsInline
          disableRemotePlayback
          onPointerDown={() => controls && setNativeControlsVisible(true)}
          onFocus={() => controls && setNativeControlsVisible(true)}
          onLoadedData={(event) => {
            event.currentTarget.play().catch(() => { });
            setVideoReady(true);
            onReady?.(event.currentTarget, stage, deliverySource);
          }}
        />
      ) : null}
      {stage === "recording" ? (
        <RecordingFallback
          cameraId={cameraId}
          source={deliverySource}
          timeZone={timeZone}
          muted={muted}
          controls={controls}
          onReady={onReady}
          onError={() => setStage("snapshot")}
        />
      ) : null}
    </div>
  );
});

import React, { forwardRef, useContext, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  ArrowUpDown,
  Bike,
  Bot,
  BusFront,
  Camera,
  CarFront,
  Cat,
  Check,
  CircleAlert,
  ChevronLeft,
  ChevronRight,
  Copy,
  CircleDot,
  Clock3,
  Crop,
  Cog,
  Download,
  Dog,
  Cpu,
  Film,
  Gauge,
  Grid2X2,
  Images,
  GripVertical,
  HardDrive,
  Search,
  ListTree,
  Monitor,
  Moon,
  Pause,
  Play,
  Plus,
  Power,
  Radar,
  Radio,
  RefreshCcw,
  RotateCcw,
  Save,
  ScanFace,
  ShieldCheck,
  Sparkles,
  Siren,
  SkipBack,
  SkipForward,
  Sun,
  Trash2,
  Truck,
  Undo2,
  UserRound,
  UserPlus,
  Users,
  Rows3,
  Video,
  Wrench,
  X,
} from "lucide-react";
import "./styles.css";
import {
  buildMotionDecisionFusion,
  MOTION_BEHAVIOR_OPTIONS,
  motionBehaviorOption,
  motionBehaviorSettings,
  motionBehaviorValue,
  motionModeInfo,
  readMotionDecisionFusion,
} from "./motionDecisionConfig.mjs";
import {
  availableQualificationPresets,
  motionAnalysisPresetSelectionUseful,
  presetQualificationGraph,
  readMotionAnalysisPreset,
} from "./motionAnalysisConfig.mjs";
import {
  clearWebRtcFailure,
  initialLiveTransport,
  nextNativeFallbackSource,
  rememberWebRtcFailure,
  webRtcRetryDelay,
} from "./liveTransport.mjs";
import {
  aspectFromDimensions,
  cameraSourceAspect,
  initialCameraAspect,
  liveAspectStorageKey,
  normalizedLiveSource,
  validLiveAspect,
} from "./liveAspect.mjs";
import { resetLiveDefaultsForServer } from "./liveDefaults.mjs";
import { browserStorage, readStoredValue, removeStoredValue, writeStoredValue } from "./storage.mjs";
import { readAssistantHistory, writeAssistantHistory } from "./assistantStorage.mjs";
import { assistantEvidenceHref, assistantIncidentHref } from "./assistantNavigation.mjs";
import { containedFrameTransform, hlsPlaybackOffset, hlsProgramStartEpoch, incidentTrackingSource, playbackEpochAt, storedObjectTracks, trackFrameAt } from "./objectTrackReplay.mjs";
import { describePlaybackError, gridPlaybackNeedsSeek, isUnsupportedPlaybackError, mergeRecordingAvailability, playbackMediaTimeForEpoch, playbackRowsCoverEpoch } from "./recordingPlayback.mjs";
import { recordingCameraAspect, recordingGridBestEpoch, recordingGridLayout } from "./recordingGrid.mjs";
import { liveCustomDropTarget, liveCustomGridMetrics, liveCustomTilePlacement, moveLiveCamera, readLiveCustomLayout, resizeLiveCameraToAspect } from "./liveCustomLayout.mjs";
import { adjacentIncident, createIncidentPageCache, incidentDetectionFrameSize, incidentDetailQuery, incidentEvidenceFrames, incidentMosaicEvents, incidentMosaicPage, incidentObjectIconName, incidentProgressiveImageWidth, incidentThumbnailPageSize, incidentTrackingFrameSize, incidentZoomLayout, incidentsNewestFirst, incidentTriggerLabel, linkedIncidentEventFilter, retainFocusedIncident, showIncidentCardAnnotations } from "./incidentNavigation.mjs";
import { motionAuditRegions } from "./motionAudit.mjs";
import { addSemanticSearchHistory, clearSemanticSearchSession, readSemanticSearchHistory, readSemanticSearchSession, semanticSearchResultsForCamera, writeSemanticSearchHistory, writeSemanticSearchSession } from "./semanticSearchState.mjs";
import { mapWithConcurrency, rankSemanticIncidentDetails, semanticIncidentRequest } from "./incidentSemanticSearch.mjs";
import { insertZonePointWithIndex } from "./zoneGeometry.mjs";
import { relatedEvidenceLabel, relatedIncidentThumbnailPath, relatedIncidentsPath, visibleRelatedAppearances } from "./relatedIncidents.mjs";
import { nextFaceReviewObservation } from "./faceReview.mjs";

const DEFAULT_TIME_ZONE = "America/New_York";
const MEDIA_STORAGE_ROLES = [
  ["recordings", "Recordings"],
  ["snapshots", "Snapshots"],
  ["motion_audits", "Motion audits"],
  ["clips", "Clips"],
  ["exports", "Exports"],
];
const LEGACY_INCIDENT_FILTER_KEYS = [
  "survng.liveEventFilter.v2",
  "survng.incidentDay.v1",
  "survng.incidentCameraFilter.v1",
  "survng.incidentObjectFilter.v1",
  "survng.incidentZoneFilter.v1",
];
const US_TIME_ZONES = [
  ["America/New_York", "Eastern"],
  ["America/Chicago", "Central"],
  ["America/Denver", "Mountain"],
  ["America/Phoenix", "Arizona"],
  ["America/Los_Angeles", "Pacific"],
  ["America/Anchorage", "Alaska"],
  ["Pacific/Honolulu", "Hawaii"],
];

const THEMES = ["auto", "light", "dark"];
const THEME_META = {
  auto: { label: "Auto", icon: Monitor },
  light: { label: "Light", icon: Sun },
  dark: { label: "Dark", icon: Moon },
};

const APP_BASE_PATH = String(window.__SURVNG_BASE_PATH__ || "").replace(/\/+$/, "");
const PREFER_NATIVE_HLS = /iPad|iPhone|iPod/.test(navigator.userAgent)
  || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
document.documentElement.dataset.embedded = window.self !== window.top ? "true" : "false";

function appUrl(path = "/") {
  if (typeof path !== "string" || !path.startsWith("/") || path.startsWith("//")) return path;
  if (APP_BASE_PATH && (path === APP_BASE_PATH || path.startsWith(`${APP_BASE_PATH}/`))) return path;
  return `${APP_BASE_PATH}${path}`;
}

function appPathname() {
  const pathname = window.location.pathname;
  if (!APP_BASE_PATH || (!pathname.startsWith(`${APP_BASE_PATH}/`) && pathname !== APP_BASE_PATH)) return pathname;
  return pathname.slice(APP_BASE_PATH.length) || "/";
}

function incidentRecordingContext(item) {
  if (!item?.camera_id || !item?.created_at) return null;
  const epoch = new Date(item.created_at).getTime() / 1000;
  if (!Number.isFinite(epoch)) return null;
  return { cameraId: item.camera_id, epoch };
}

function recordingsHref(context) {
  if (!context?.cameraId || !Number.isFinite(context?.epoch)) return appUrl("/recordings");
  const params = new URLSearchParams({
    camera: context.cameraId,
    at: String(Math.round(context.epoch * 1000) / 1000),
  });
  return appUrl(`/recordings?${params.toString()}`);
}

const fetch = (resource, options) => window.fetch(
  typeof resource === "string" ? appUrl(resource) : resource,
  options,
);
const SECRET_PLACEHOLDER = "__SURVNG_SECRET_SET__";

function clearMaskedUrlPassword(value) {
  if (!value || !value.includes(SECRET_PLACEHOLDER)) return value || "";
  return value.replace(`:${SECRET_PLACEHOLDER}@`, "@");
}

function clearMaskedSecret(value) {
  return value === SECRET_PLACEHOLDER ? "" : value || "";
}

function secretInputValue(value) {
  return value === SECRET_PLACEHOLDER ? "" : value || "";
}

function secretInputHint(value, fallback = "") {
  return value === SECRET_PLACEHOLDER ? "Saved — type to replace" : fallback;
}

let shakaImport;
function loadShaka() {
  if (!shakaImport) shakaImport = import("shaka-player").then((module) => module.default || module);
  return shakaImport;
}

const ShakaVideo = forwardRef(function ShakaVideo({
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
  const [nativeControlsVisible, setNativeControlsVisible] = useState(false);
  const callbacksRef = useRef({ onReady, onError });
  useImperativeHandle(forwardedRef, () => videoRef.current);

  useEffect(() => {
    setNativeControlsVisible(false);
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
      if (autoPlay) videoRef.current?.play().catch(() => {});
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

function RecordingFallback({ cameraId, source, timeZone, muted, controls, onReady, onError }) {
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

const WebRtcLive = forwardRef(function WebRtcLive({
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
      video.play().catch(() => {});
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
            event.currentTarget.play().catch(() => {});
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

function eventSnapshotUrl(event) {
  if (event?.snapshot_url) return appUrl(event.snapshot_url);
  const eventId = Number(event?.representative_event_id || event?.id);
  return Number.isFinite(eventId) ? appUrl(`/api/events/${eventId}/snapshot.jpg`) : "";
}

function eventSnapshotDownloadUrl(event) {
  const snapshotUrl = eventSnapshotUrl(event);
  return snapshotUrl ? `${snapshotUrl}?download=true` : "";
}

function eventThumbnailUrl(event, width = 720, quality = 82) {
  if (event?.snapshot_url) return appUrl(event.snapshot_url);
  const eventId = Number(event?.representative_event_id || event?.id);
  return Number.isFinite(eventId) ? appUrl(`/api/events/${eventId}/thumbnail.jpg?width=${width}&quality=${quality}`) : "";
}

function useStoredState(key, initialValue) {
  const [value, setValue] = useState(() => readStoredValue(browserStorage(window), key, initialValue));
  useEffect(() => {
    writeStoredValue(browserStorage(window), key, value);
  }, [key, value]);
  return [value, setValue];
}

function mediaStorageConfigurationError(mediaStorage) {
  const locations = mediaStorage?.locations || [];
  if (!locations.length) return "";
  const enabledRoles = new Set(
    locations
      .filter((location) => location.enabled !== false)
      .flatMap((location) => location.roles || []),
  );
  const missing = MEDIA_STORAGE_ROLES
    .filter(([role]) => !enabledRoles.has(role))
    .map(([, label]) => label);
  return missing.length
    ? `At least one enabled media location must accept: ${missing.join(", ")}.`
    : "";
}

function clearLegacyIncidentFilterStorage() {
  const storage = browserStorage(window);
  LEGACY_INCIDENT_FILTER_KEYS.forEach((key) => removeStoredValue(storage, key));
}

function formatDateTime(value, timeZone) {
  if (!value) return "--";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    timeZoneName: "short",
  }).format(date);
}

function formatTimeOnly(value, timeZone) {
  if (!value) return "--";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-US", {
    timeZone,
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function formatExportHandleTime(value, timeZone) {
  if (!value) return "--:--:--";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-US", {
    timeZone,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).format(date);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "--";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = Math.max(0, bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)}${units[unit]}`;
}

function formatMilliseconds(value) {
  if (!Number.isFinite(value)) return "--";
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)}ms`;
}

function formatRate(value) {
  if (!Number.isFinite(value)) return "--";
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)}/s`;
}

function formatAge(seconds) {
  if (!Number.isFinite(seconds)) return "--";
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 1) return "instant";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
}

function formatCompactDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "calculating";
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

function isMobileViewport() {
  return typeof window !== "undefined" && window.matchMedia("(max-width: 760px)").matches;
}

function preferredStreamSource() {
  return isMobileViewport() ? "live" : "main";
}

function sourceLabel(source) {
  return source === "main" ? "Main" : "Sub";
}

const LIVE_TRANSPORT_LABELS = {
  webrtc: "WebRTC",
  mse: "MSE",
  mjpeg: "MJPEG",
  recording: "Recording",
  snapshot: "Snapshot",
};

function liveTransportLabel(stage) {
  return LIVE_TRANSPORT_LABELS[stage] || "Connecting";
}

function dateKeyForTimeZone(value, timeZone) {
  const date = typeof value === "number"
    ? new Date(value > 100_000_000_000 ? value : value * 1000)
    : new Date(value || Date.now());
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);
  const part = (type) => parts.find((item) => item.type === type)?.value || "";
  return `${part("year")}-${part("month")}-${part("day")}`;
}

function addDaysToDateKey(dateKey, days) {
  const [year, month, day] = dateKey.split("-").map(Number);
  const date = new Date(Date.UTC(year, month - 1, day + days));
  return date.toISOString().slice(0, 10);
}

function zonedDateSecondToEpoch(dateKey, seconds, timeZone) {
  const [year, month, day] = dateKey.split("-").map(Number);
  const desired = Date.UTC(year, month - 1, day, 0, 0, 0) + Math.max(0, seconds) * 1000;
  let guess = desired;
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone,
    hourCycle: "h23",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  for (let index = 0; index < 3; index += 1) {
    const parts = formatter.formatToParts(new Date(guess));
    const value = (type) => Number(parts.find((item) => item.type === type)?.value || 0);
    const actual = Date.UTC(value("year"), value("month") - 1, value("day"), value("hour"), value("minute"), value("second"));
    guess += desired - actual;
  }
  return guess / 1000;
}

function slugify(value) {
  return String(value || "camera")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    || "camera";
}

function uniqueCameraId(cameras, base) {
  const existing = new Set(cameras.map((camera) => camera.id));
  let candidate = slugify(base);
  let index = 2;
  while (existing.has(candidate)) {
    candidate = `${slugify(base)}-${index}`;
    index += 1;
  }
  return candidate;
}

function streamUrlDefaults(url) {
  try {
    const parsed = new URL(url);
    if (!parsed.hostname) return {};
    return {
      scheme: parsed.protocol.replace(":", "").toLowerCase(),
      host: parsed.hostname,
      port: parsed.port ? Number(parsed.port) : null,
      username: decodeURIComponent(parsed.username || ""),
      password: decodeURIComponent(parsed.password || ""),
      channel: Number(parsed.searchParams.get("channel") || parsed.searchParams.get("chn") || 0),
    };
  } catch {
    return {};
  }
}

function inferredBackendLabel(camera) {
  if (camera.video_backend === "baichuan_native" && camera.baichuan?.enabled) {
    return "Reolink Baichuan (URL fallback)";
  }
  const defaults = streamUrlDefaults(camera.stream_url || camera.live_stream_url || "");
  if (defaults.scheme === "reolink") return "Reolink Baichuan";
  if (defaults.scheme === "rtsp") return "RTSP";
  if (defaults.scheme) return defaults.scheme.toUpperCase();
  return "URL";
}

function cameraWithDerivedConnection(camera) {
  const defaults = streamUrlDefaults(camera.stream_url || camera.live_stream_url || "");
  if (!defaults.host) return camera;
  const isReolink = defaults.scheme === "reolink";
  const isRtsp = defaults.scheme === "rtsp";
  const nativeSelected = camera.video_backend === "baichuan_native" && camera.baichuan?.enabled;
  return {
    ...camera,
    video_backend: isReolink || nativeSelected ? "baichuan_native" : isRtsp ? "url" : camera.video_backend,
    onvif: {
      ...camera.onvif,
      host: camera.onvif?.host || defaults.host,
      username: camera.onvif?.username || defaults.username,
      password: camera.onvif?.password || defaults.password,
    },
    baichuan: {
      ...camera.baichuan,
      enabled: isReolink || nativeSelected,
      host: isReolink ? defaults.host : camera.baichuan?.host || defaults.host,
      port: isReolink ? defaults.port || 9000 : camera.baichuan?.port || 9000,
      username: isReolink ? defaults.username : camera.baichuan?.username || defaults.username,
      password: isReolink ? defaults.password : camera.baichuan?.password || defaults.password,
      channel: isReolink ? defaults.channel || 0 : camera.baichuan?.channel || 0,
    },
  };
}

function camerasWithGeneratedIds(cameras) {
  const used = new Set();
  return (cameras || []).map((camera) => {
    const base = slugify(camera.name || camera.id || "camera") || "camera";
    let id = base;
    let index = 2;
    while (used.has(id)) {
      id = `${base}-${index}`;
      index += 1;
    }
    used.add(id);
    return { ...cameraWithDerivedConnection(camera), id };
  });
}

function defaultCamera(cameras, seed = {}) {
  const id = uniqueCameraId(cameras, seed.id || seed.name || "camera");
  const host = seed.onvif?.host || seed.baichuan?.host || "";
  return {
    id,
    name: seed.name ? `${seed.name} Copy` : "New Camera",
    video_backend: seed.video_backend || "url",
    stream_url: clearMaskedUrlPassword(seed.stream_url),
    live_stream_url: clearMaskedUrlPassword(seed.live_stream_url),
    record: seed.record ?? true,
    record_sub: seed.record_sub ?? false,
    retention: {
      main_days: seed.retention?.main_days ?? null,
      live_days: seed.retention?.live_days ?? null,
    },
    require_incident_zone: seed.require_incident_zone ?? null,
    object_activity_attribution: seed.object_activity_attribution || "inherit",
    motion_qualification: {
      ...defaultCameraMotionQualification(),
      mode: seed.motion_qualification?.mode || "inherit",
      sensitivity: seed.motion_qualification?.sensitivity || "inherit",
      stationary_object_tolerance: seed.motion_qualification?.stationary_object_tolerance || "inherit",
      illumination_filter_enabled: seed.motion_qualification?.illumination_filter_enabled ?? null,
      frame_width: seed.motion_qualification?.frame_width ?? null,
      visual_backup_grace_seconds: seed.motion_qualification?.visual_backup_grace_seconds ?? null,
      visual_backup_min_score: seed.motion_qualification?.visual_backup_min_score ?? null,
      visual_backup_min_consecutive: seed.motion_qualification?.visual_backup_min_consecutive ?? null,
      visual_backup_cooldown_seconds: seed.motion_qualification?.visual_backup_cooldown_seconds ?? null,
      visual_backup_max_triggers_5m: seed.motion_qualification?.visual_backup_max_triggers_5m ?? null,
      borderline_rescue_enabled: seed.motion_qualification?.borderline_rescue_enabled ?? null,
      borderline_margin: seed.motion_qualification?.borderline_margin ?? null,
      suppression_verification_rate: seed.motion_qualification?.suppression_verification_rate ?? null,
      spatial_alignment: structuredClone(seed.motion_qualification?.spatial_alignment || {}),
      pipeline: structuredClone(seed.motion_qualification?.pipeline || {}),
    },
    zones: structuredClone(seed.zones || []),
    onvif: {
      enabled: seed.onvif?.enabled || false,
      host,
      port: seed.onvif?.port || 8000,
      username: seed.onvif?.username || "",
      password: clearMaskedSecret(seed.onvif?.password),
    },
    baichuan: {
      enabled: seed.baichuan?.enabled || false,
      host,
      port: seed.baichuan?.port || 9000,
      username: seed.baichuan?.username || "",
      password: clearMaskedSecret(seed.baichuan?.password),
      channel: seed.baichuan?.channel || 0,
    },
  };
}

function defaultCameraMotionQualification() {
  return {
    mode: "inherit",
    sensitivity: "inherit",
    stationary_object_tolerance: "inherit",
    illumination_filter_enabled: null,
    frame_width: null,
    visual_backup_grace_seconds: null,
    visual_backup_min_score: null,
    visual_backup_min_consecutive: null,
    visual_backup_cooldown_seconds: null,
    visual_backup_max_triggers_5m: null,
    borderline_rescue_enabled: null,
    borderline_margin: null,
    suppression_verification_rate: null,
    spatial_alignment: {},
    pipeline: {
      qualification: null,
      observation: null,
      fusion: null,
    },
  };
}

function cameraMotionQualificationInherited(qualification) {
  const value = qualification || {};
  const alignment = value.spatial_alignment || {};
  const pipeline = value.pipeline || {};
  return (
    (value.mode == null || value.mode === "inherit")
    && (value.sensitivity == null || value.sensitivity === "inherit")
    && (value.stationary_object_tolerance == null || value.stationary_object_tolerance === "inherit")
    && value.illumination_filter_enabled == null
    && value.frame_width == null
    && value.visual_backup_grace_seconds == null
    && value.visual_backup_min_score == null
    && value.visual_backup_min_consecutive == null
    && value.visual_backup_cooldown_seconds == null
    && value.visual_backup_max_triggers_5m == null
    && value.borderline_rescue_enabled == null
    && value.borderline_margin == null
    && value.suppression_verification_rate == null
    && (alignment.mode == null || alignment.mode === "auto")
    && Number(alignment.confidence ?? 0) === 0
    && Number(alignment.scale_x ?? 1) === 1
    && Number(alignment.scale_y ?? 1) === 1
    && Number(alignment.offset_x ?? 0) === 0
    && Number(alignment.offset_y ?? 0) === 0
    && pipeline.qualification == null
    && pipeline.observation == null
    && pipeline.fusion == null
  );
}

const ASSISTANT_STORAGE_KEY = "survng.assistantConversation.v1";

const assistantVisualVerdicts = {
  detection_consistent: "What SurvNG found matches the image",
  probable_missed_detection: "SurvNG likely missed something visible",
  probable_misclassification: "The object was likely labeled incorrectly",
  probable_false_positive: "The detection was likely a false alarm",
  uncertain: "The single image is inconclusive",
};

const assistantDetectorAssessments = {
  consistent: "Matches the image",
  missed: "Likely missed a visible object",
  misclassified: "Likely used the wrong label",
  false_positive: "Likely detected something that is not there",
  uncertain: "Not enough visual evidence",
};

const assistantTrackingAssessments = {
  consistent: "Followed the object normally",
  late: "Started following it late",
  lost: "Stopped following it too early",
  duplicate: "Likely counted one object more than once",
  unavailable: "No useful follow-up tracking was available",
  uncertain: "Not enough evidence to judge tracking",
};

const assistantSettingLabels = {
  analysis_preset: "Motion analysis style",
  sensitivity: "Motion sensitivity",
  stationary_object_tolerance: "Stationary object policy",
  frame_width: "Motion analysis image size",
  borderline_rescue_enabled: "Second look at borderline motion",
  borderline_margin: "Borderline motion range",
};

function readAssistantMessages() {
  return readAssistantHistory(browserStorage(window), ASSISTANT_STORAGE_KEY);
}

function AssistantPanel({ pageContext, timeZone }) {
  const [openValue, setOpenValue] = useStoredState("survng.assistantOpen.v1", "false");
  const open = openValue === "true";
  const [messages, setMessages] = useState(readAssistantMessages);
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [applyingEvidenceId, setApplyingEvidenceId] = useState("");
  const [error, setError] = useState("");
  const bodyRef = useRef(null);
  const activeExportIds = [...new Set(messages.flatMap((message) =>
    (message.evidence || [])
      .map((item) => item.details?.media_export)
      .filter((job) => job?.id && ["queued", "running", "cancelling"].includes(job.status))
      .map((job) => job.id)
  ))].sort().join(",");

  useEffect(() => {
    writeAssistantHistory(browserStorage(window), ASSISTANT_STORAGE_KEY, messages);
  }, [messages]);

  useEffect(() => {
    if (!open) return;
    fetch("/api/assistant/status")
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Assistant status unavailable")))
      .then(setStatus)
      .catch((statusError) => setError(statusError.message || "Assistant status unavailable"));
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const body = bodyRef.current;
    if (body) body.scrollTop = body.scrollHeight;
  }, [messages, busy, open]);

  useEffect(() => {
    if (!open || !activeExportIds) return undefined;
    let cancelled = false;
    const ids = activeExportIds.split(",").filter(Boolean);
    async function refreshExports() {
      const updates = await Promise.all(ids.map(async (id) => {
        try {
          const response = await fetch(`/api/exports/${encodeURIComponent(id)}`);
          return response.ok ? await response.json() : null;
        } catch {
          return null;
        }
      }));
      if (cancelled) return;
      const byId = new Map(updates.filter(Boolean).map((job) => [job.id, job]));
      if (!byId.size) return;
      setMessages((current) => current.map((message) => ({
        ...message,
        evidence: (message.evidence || []).map((item) => {
          const previous = item.details?.media_export;
          const update = previous?.id ? byId.get(previous.id) : null;
          if (!update) return item;
          return {
            ...item,
            details: {
              ...item.details,
              media_export: {
                ...previous,
                status: update.status,
                phase: update.phase,
                progress: update.progress,
                error: update.error,
                output_name: update.output_name,
                size_bytes: update.size_bytes,
                download_url: update.download_url,
              },
            },
          };
        }),
      })));
    }
    refreshExports();
    const timer = window.setInterval(refreshExports, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeExportIds, open]);

  useEffect(() => {
    if (!open) return undefined;
    function closeOnEscape(event) {
      if (event.key === "Escape") setOpenValue("false");
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [open, setOpenValue]);

  function clearConversation() {
    setMessages([]);
    setError("");
  }

  async function sendMessage(messageText = draft) {
    const content = String(messageText || "").trim();
    if (!content || busy) return;
    const userMessage = { id: `user-${Date.now()}`, role: "user", content };
    const prior = messages.slice(-12);
    setMessages((current) => [...current, userMessage].slice(-30));
    setDraft("");
    setError("");
    setBusy(true);
    try {
      const response = await fetch("/api/assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: content,
          history: prior.map(({ role, content: historyContent }) => ({ role, content: historyContent })),
          context: {
            page: pageContext?.page || "live",
            camera_id: pageContext?.camera_id || "",
            incident_event_id: pageContext?.incident_event_id || null,
            recording_epoch: pageContext?.recording_epoch || null,
            filters: pageContext?.filters || {},
            time_zone: timeZone,
          },
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Assistant failed (${response.status})`);
      setStatus((current) => current ? { ...current, configured: true } : current);
      setMessages((current) => [...current, {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: payload.message || "No answer returned.",
        evidence: payload.evidence || [],
        citations: payload.citations || [],
        suggestions: payload.suggestions || [],
        reasoningTier: payload.reasoning_tier || "fast",
        model: payload.model || "",
      }].slice(-30));
    } catch (sendError) {
      setError(sendError.message || "Assistant request failed");
    } finally {
      setBusy(false);
    }
  }

  async function applyVisualProposals(messageId, evidence) {
    const details = evidence?.details || {};
    const changes = details.advice?.changes || [];
    if (!changes.length || applyingEvidenceId) return;
    const cameraLabel = details.camera_id || "this camera";
    if (!window.confirm(`Apply ${changes.length} reviewed motion setting${changes.length === 1 ? "" : "s"} for ${cameraLabel}? Camera workers will restart.`)) return;
    setApplyingEvidenceId(evidence.id);
    setError("");
    try {
      const response = await fetch(`/api/incidents/${details.event_id}/ai-apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          changes,
          confirmed: true,
          configuration_fingerprint: details.configuration_fingerprint || "",
          recommendation_proof: details.recommendation_proof || "",
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Unable to apply the reviewed settings");
      setMessages((current) => current.map((message) => message.id !== messageId ? message : {
        ...message,
        evidence: (message.evidence || []).map((item) => item.id !== evidence.id ? item : {
          ...item,
          details: { ...item.details, can_apply: false, applied: payload.applied || [] },
        }),
      }));
    } catch (applyError) {
      setError(applyError.message || "Unable to apply the reviewed settings");
    } finally {
      setApplyingEvidenceId("");
    }
  }

  return (
    <>
      <button type="button" className={`assistant-launcher ${open ? "open" : ""}`} onClick={() => setOpenValue(open ? "false" : "true")} aria-label={open ? "Close SurvNG Assistant" : "Open SurvNG Assistant"} title="SurvNG Assistant">
        {open ? <X size={22} /> : <Sparkles size={22} />}
      </button>
      {open ? <aside className="assistant-drawer" role="dialog" aria-label="SurvNG Assistant">
        <header className="assistant-head">
          <div><strong><Sparkles size={17} /> SurvNG Assistant</strong><small>Grounded analysis · exports on request</small></div>
          <div>
            <button type="button" onClick={clearConversation} aria-label="Clear assistant conversation" title="Clear conversation"><Trash2 size={16} /></button>
            <button type="button" onClick={() => setOpenValue("false")} aria-label="Close SurvNG Assistant"><X size={17} /></button>
          </div>
        </header>
        <div className="assistant-context">
          <span>{pageContext?.page || "live"}</span>
          {pageContext?.camera_id ? <span>{pageContext.camera_id}</span> : null}
          {pageContext?.incident_event_id ? <span>event #{pageContext.incident_event_id}</span> : null}
          {status ? <span title={status.fast_model === status.reasoning_model ? "Everyday and detailed questions use this model" : `Everyday: ${status.fast_model} · Detailed: ${status.reasoning_model}`}>{status.fast_model === status.reasoning_model ? status.fast_model : "Everyday + detailed AI"}</span> : null}
        </div>
        <div className="assistant-body" ref={bodyRef}>
          {!messages.length ? <div className="assistant-welcome">
            <Sparkles size={26} />
            <strong>What would you like to know?</strong>
            <p>I can search incidents, trace related activity, review a selected incident, inspect camera health, explain settings, and create recording exports or timelapses.</p>
            <div>
              {[
                "Is everything healthy?",
                ...(pageContext?.incident_event_id ? ["Trace this incident across cameras", "Visually analyze this incident"] : []),
                ...(pageContext?.camera_id ? [`Create a timelapse for ${pageContext.camera_id} from 8 AM to 8 PM yesterday`] : []),
                "Find person incidents from the last 24 hours",
              ].map((suggestion) => <button type="button" key={suggestion} onClick={() => sendMessage(suggestion)}>{suggestion}</button>)}
            </div>
          </div> : null}
          {messages.map((message) => <article className={`assistant-message ${message.role}`} key={message.id}>
            <div className="assistant-message-text">{message.content}</div>
            {message.role === "assistant" && (message.model || message.reasoningTier) ? <small className="assistant-model-tier">{message.reasoningTier === "deep" ? "Detailed analysis" : "Quick answer"}{message.model ? ` · ${message.model}` : ""}</small> : null}
            {message.evidence?.length ? <div className="assistant-evidence">
              {message.evidence.map((item) => <div key={item.id} className={`assistant-evidence-card ${item.details ? "has-details" : ""}`}>
                {item.image_url ? <a className="assistant-evidence-image" href={appUrl(assistantEvidenceHref(item))} aria-label={`Open ${item.title || "incident"}`} title="Open incident"><img src={appUrl(item.image_url)} alt={item.title || "Incident evidence"} loading="lazy" /></a> : null}
                <a href={assistantEvidenceHref(item) ? appUrl(assistantEvidenceHref(item)) : undefined}><span title={item.id}>Source</span><strong>{item.title}</strong><small>{item.summary}</small></a>
                {item.details?.timeline ? <div className="assistant-timeline">
                  {item.details.timeline.matches?.length ? item.details.timeline.matches.map((match) => <a className="assistant-timeline-link" href={appUrl(assistantIncidentHref(match.event_id))} key={match.event_id} title="Open incident"><span>{formatDateTime(match.start_at)}</span><strong>{match.camera_id}</strong><small>{({ confirmed_identity: "Confirmed face", possible_identity: "Possible face", appearance_similarity: `Visually similar ${match.appearance_similarity != null ? `${Math.round(Number(match.appearance_similarity) * 100)}%` : "appearance"}`, context_candidate: "Nearby matching class" })[match.match_strength] || "Possible connection"}</small></a>) : <small>No related incidents were found in this time window.</small>}
                  <p>{item.details.timeline.limitations?.[3]}</p>
                </div> : null}
                {item.details?.media_export ? <div className={`assistant-media-export ${item.details.media_export.status}`}>
                  <div><strong>{item.details.media_export.phase || item.details.media_export.status}</strong><span>{Math.round(Number(item.details.media_export.progress) || 0)}%</span></div>
                  <i><b style={{ width: `${Math.max(0, Math.min(100, Number(item.details.media_export.progress) || 0))}%` }} /></i>
                  {item.details.media_export.error ? <small>{item.details.media_export.error}</small> : null}
                  {item.details.media_export.status === "completed" && item.details.media_export.download_url ? <a className="assistant-export-download" href={item.details.media_export.download_url}><Download size={14} />Download MP4</a> : null}
                </div> : null}
                {item.details?.advice ? <div className="assistant-visual-review">
                  <div><strong>{assistantVisualVerdicts[item.details.advice.verdict] || "The image is inconclusive"}</strong><span>{Math.round(Number(item.details.advice.confidence || 0) * 100)}%</span></div>
                  <p>{item.details.advice.summary}</p>
                  {item.details.advice.visible_subjects?.length ? <small>Visible in this image: {item.details.advice.visible_subjects.join(", ")}</small> : null}
                  <dl>
                    <div><dt>Object recognition</dt><dd>{assistantDetectorAssessments[item.details.advice.detector_assessment] || assistantDetectorAssessments.uncertain}</dd></div>
                    <div><dt>Follow-up tracking</dt><dd>{assistantTrackingAssessments[item.details.advice.tracking_assessment] || assistantTrackingAssessments.uncertain}</dd></div>
                  </dl>
                  {item.details.proposals?.length ? <div className="assistant-proposals">
                    {item.details.proposals.map((proposal) => <div key={`${proposal.scope}-${proposal.setting}`}>
                      <strong>This camera · {assistantSettingLabels[proposal.setting] || String(proposal.setting).replaceAll("_", " ")}</strong>
                      <span><code>{String(proposal.current)}</code><ArrowRight size={13} /><code>{String(proposal.proposed)}</code></span>
                      <small>{proposal.reason}</small>
                    </div>)}
                  </div> : <small>No bounded setting changes recommended from this image.</small>}
                  {item.details.applied?.length ? <div className="assistant-applied"><Check size={14} /> Applied after confirmation</div> : null}
                  {item.details.can_apply && !item.details.applied?.length ? <button type="button" className="assistant-apply" disabled={Boolean(applyingEvidenceId)} onClick={() => applyVisualProposals(message.id, item)}>{applyingEvidenceId === item.id ? "Applying…" : "Review and apply"}</button> : null}
                  {item.details.proposals?.length && !item.details.can_apply && !item.details.applied?.length ? <small>Enable “Allow confirmed changes” in Admin to apply these proposals.</small> : null}
                </div> : null}
              </div>)}
            </div> : null}
            {message.suggestions?.length ? <div className="assistant-suggestions">
              {message.suggestions.map((suggestion) => <button type="button" key={suggestion} onClick={() => sendMessage(suggestion)}>{suggestion}</button>)}
            </div> : null}
          </article>)}
          {busy ? <div className="assistant-thinking"><span /><span /><span /> Gathering SurvNG evidence…</div> : null}
          {error ? <div className="assistant-error"><CircleAlert size={15} /> {error}</div> : null}
          {status && !status.configured ? <div className="assistant-error"><CircleAlert size={15} /> Configure and enable the AI provider under Admin → Object Detection.</div> : null}
        </div>
        <form className="assistant-compose" onSubmit={(event) => { event.preventDefault(); sendMessage(); }}>
          <textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(); } }} placeholder="Ask about SurvNG…" rows="2" maxLength="8000" disabled={busy} />
          <button type="submit" disabled={busy || !draft.trim()}>Send</button>
        </form>
      </aside> : null}
    </>
  );
}

function Shell({ page, theme, recordingContext, children }) {
  const shellRef = useRef(null);
  const topbarRef = useRef(null);
  const isLive = page === "live";
  const isRecordings = page === "recordings";
  const isConfig = page === "config";
  const isIncidents = page === "incidents";
  const isFaces = page === "faces";
  useLayoutEffect(() => {
    const shell = shellRef.current;
    const topbar = topbarRef.current;
    if (!shell || !topbar) return undefined;
    const updateTopbarHeight = () => {
      shell.style.setProperty(
        "--topbar-height",
        `${Math.ceil(topbar.getBoundingClientRect().height)}px`,
      );
    };
    updateTopbarHeight();
    const observer = typeof ResizeObserver === "function"
      ? new ResizeObserver(updateTopbarHeight)
      : null;
    observer?.observe(topbar);
    window.addEventListener("resize", updateTopbarHeight);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", updateTopbarHeight);
    };
  }, []);
  return (
    <div ref={shellRef} className={`app-shell page-${page}`}>
      <header ref={topbarRef} className="topbar">
        <div className="brand-block">
          <div className="brand-mark">
            <img src={appUrl("/static/favicon.svg")} alt="" aria-hidden="true" />
          </div>
          <div className="brand-title">
            <h1>SurvNG</h1>
          </div>
          {!isConfig && !isRecordings && !isIncidents && !isFaces ? <LiveHeaderStats /> : null}
        </div>
        <div className="top-actions">
          <nav className="topnav primary-nav" aria-label="Primary">
            <a className={`nav-button ${isLive ? "active" : ""}`} aria-current={isLive ? "page" : undefined} href={appUrl("/")}><Video size={16} /> Live</a>
            <a className={`nav-button incidents-nav ${isIncidents ? "active" : ""}`} aria-current={isIncidents ? "page" : undefined} href={appUrl("/incidents")}><Siren size={16} /> Incidents</a>
            <a className={`nav-button ${isRecordings ? "active" : ""}`} aria-current={isRecordings ? "page" : undefined} href={recordingsHref(recordingContext)}><Film size={16} /> Recordings</a>
          </nav>
          <nav className="topnav utility-nav" aria-label="Additional">
            <a className={`nav-button ${isFaces ? "active" : ""}`} aria-current={isFaces ? "page" : undefined} href={appUrl("/faces")} aria-label="Faces"><ScanFace size={16} /><span>Faces</span></a>
            <a className={`nav-button ${isConfig ? "active" : ""}`} aria-current={isConfig ? "page" : undefined} href={appUrl("/config")} aria-label="Admin"><Cog size={16} /><span>Admin</span></a>
          </nav>
        </div>
      </header>
      {children}
    </div>
  );
}

const APP_EVENT_TYPES = ["camera_state", "cameras_state", "motion", "object", "incident", "system_state"];
const INCIDENT_REFRESH_FALLBACK_MS = 15_000;
const appEventListeners = new Set();
let appEventSource = null;
let appEventCloseTimer = null;

function subscribeAppEvents(listener) {
  appEventListeners.add(listener);
  if (appEventCloseTimer) {
    window.clearTimeout(appEventCloseTimer);
    appEventCloseTimer = null;
  }
  if (!appEventSource) {
    appEventSource = new EventSource(appUrl("/api/events/stream"));
    APP_EVENT_TYPES.forEach((type) => {
      appEventSource.addEventListener(type, (event) => {
        let data = null;
        try {
          data = JSON.parse(event.data);
        } catch {
          return;
        }
        appEventListeners.forEach((current) => current({ type, data, id: event.lastEventId }));
      });
    });
  }
  return () => {
    appEventListeners.delete(listener);
    if (!appEventListeners.size && appEventSource) {
      appEventCloseTimer = window.setTimeout(() => {
        if (!appEventListeners.size && appEventSource) {
          appEventSource.close();
          appEventSource = null;
        }
      }, 1000);
    }
  };
}

function useAppEvents(handler) {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;
  useEffect(() => subscribeAppEvents((event) => handlerRef.current(event)), []);
}

function LiveHeaderStats() {
  const [stats, setStats] = useState({
    resources: null,
    storage: null,
    detector: null,
    cameras: null,
  });

  async function loadSystem() {
    try {
      const systemResponse = await fetch("/api/system/status");
      if (!systemResponse.ok) return;
      const system = await systemResponse.json();
      setStats((current) => ({
        ...current,
        resources: system.resources || null,
        storage: system.storage || null,
        detector: system.detector || null,
        cameras: system.cameras || null,
      }));
    } catch {
      // Keep the last known status; the next event or interval retries.
    }
  }

  useAppEvents(({ type, data }) => {
    if (type === "system_state") {
      setStats((current) => ({
        ...current,
        resources: data.resources || null,
        storage: data.storage || null,
        detector: data.detector || null,
        cameras: data.cameras || null,
      }));
    }
  });

  useEffect(() => {
    void loadSystem();
    const timer = window.setInterval(() => void loadSystem(), 60_000);
    return () => {
      window.clearInterval(timer);
    };
  }, []);

  const detector = stats.detector || {};
  const runtime = detector.runtime || {};
  const inferenceStages = runtime.stages || {};
  const isolation = detector.isolation || {};
  const inferenceWorkers = detector.workers || {};
  const objectWorker = inferenceWorkers.object || isolation;
  const faceWorker = inferenceWorkers.face || {};
  const lastStages = inferenceStages.last_ms || {};
  const averageStages = inferenceStages.average_ms || {};
  const storageLabel = stats.storage ? `${formatBytes(stats.storage.free_bytes)} free` : "--";
  const memoryLabel = stats.resources ? formatBytes(stats.resources.application_memory_bytes) : "--";
  const cpuLabel = Number.isFinite(stats.resources?.cpu_load_percent) ? `${stats.resources.cpu_load_percent.toFixed(1)}%` : "--";
  const cameraLabel = stats.cameras ? `${stats.cameras.recording}/${stats.cameras.total} rec` : "--";

  return (
    <div className="header-stats" aria-label="System summary">
      <span className="header-stat"><HardDrive size={15} /><small>Storage</small><strong>{storageLabel}</strong></span>
      <span className="header-stat"><Monitor size={15} /><small>Memory</small><strong>{memoryLabel}</strong></span>
      <span className="header-stat"><Activity size={15} /><small>CPU</small><strong>{cpuLabel}</strong></span>
      <span className="header-stat infer-stat" tabIndex={0}>
        <Cpu size={15} /><small>Infer</small><strong>{formatMilliseconds(runtime.last_inference_ms)}</strong>
        <span className="infer-tooltip" role="tooltip">
          <span className="infer-tooltip-head"><strong>OpenVINO latency</strong><small>{detector.loaded_device || detector.configured_device || "device"} · {detector.performance_hint || "default"}</small></span>
          <span className="infer-tooltip-summary">
            <span><small>Average</small><strong>{formatMilliseconds(runtime.average_inference_ms)}</strong></span>
            <span><small>Detection rate</small><strong>{formatRate(runtime.detection_fps)} det/s</strong></span>
          </span>
          <span className="infer-tooltip-row labels"><b>Stage</b><b>Last</b><b>Average</b></span>
          {[["Queue", "queue"], ["Preprocess", "preprocess"], ["Accelerator", "inference"], ["Postprocess", "postprocess"], ["Total", "total"]].map(([label, key]) => (
            <span className="infer-tooltip-row" key={key}><span>{label}</span><strong>{formatMilliseconds(lastStages[key])}</strong><strong>{formatMilliseconds(averageStages[key])}</strong></span>
          ))}
          <span className="infer-tooltip-foot">{objectWorker.configured_workers || 1} detector process{(objectWorker.configured_workers || 1) === 1 ? "" : "es"} · mmap {detector.mmap_enabled ? "on" : "off"} · cache {detector.cache_enabled ? "on" : "off"} · warm-up {formatMilliseconds(detector.warmup_ms)}</span>
          <span className="infer-tooltip-foot">object {objectWorker.configured_workers > 1 ? `${objectWorker.alive_workers || 0}/${objectWorker.configured_workers} online` : objectWorker.worker_alive ? `#${objectWorker.worker_pid}` : "offline"} · {objectWorker.configured_device || detector.configured_device || "device"} · {objectWorker.pending_requests || 0} queued · restarts {objectWorker.restart_count ?? 0}{objectWorker.fallback_active ? " · CPU fallback" : ""}</span>
          <span className="infer-tooltip-foot">face {faceWorker.enabled ? (faceWorker.worker_alive ? `#${faceWorker.worker_pid}` : "offline") : "disabled"} · {faceWorker.configured_device || "AUTO"} · gen {faceWorker.generation ?? "--"} · restarts {faceWorker.restart_count ?? 0}{faceWorker.fallback_active ? " · CPU fallback" : ""}</span>
        </span>
      </span>
      <span className="header-stat"><Camera size={15} /><small>Cameras</small><strong>{cameraLabel}</strong></span>
    </div>
  );
}

function usePollingData() {
  const [cameras, setCameras] = useState([]);
  const [appConfig, setAppConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const loadSequence = useRef(0);

  async function load() {
    const sequence = ++loadSequence.current;
    try {
      const [cameraResponse, configResponse] = await Promise.all([
        fetch("/api/cameras"),
        fetch("/api/config"),
      ]);
      if (!cameraResponse.ok) throw new Error(`Camera status failed (${cameraResponse.status})`);
      const cameraPayload = await cameraResponse.json();
      const configPayload = configResponse.ok ? await configResponse.json() : null;
      if (sequence !== loadSequence.current) return;
      if (Array.isArray(cameraPayload)) setCameras(cameraPayload);
      if (configPayload) setAppConfig(configPayload);
    } catch {
      // SSE updates may still populate the page; periodic polling retries.
    } finally {
      if (sequence === loadSequence.current) setLoading(false);
    }
  }

  useAppEvents(({ type, data }) => {
    if (type === "cameras_state" && Array.isArray(data)) {
      setCameras(data);
      setLoading(false);
    } else if (type === "camera_state" && data?.id) {
      setCameras((current) => {
        const index = current.findIndex((camera) => camera.id === data.id);
        if (index < 0) return [...current, data];
        const next = [...current];
        next[index] = data;
        return next;
      });
    }
  });

  useEffect(() => {
    load();
    const timer = window.setInterval(load, 60_000);
    return () => {
      loadSequence.current += 1;
      window.clearInterval(timer);
    };
  }, []);

  return { cameras, appConfig, loading, refresh: load };
}

function useIncidentDetails() {
  const incidentDetailCacheRef = useRef(null);
  if (!incidentDetailCacheRef.current) {
    incidentDetailCacheRef.current = createIncidentPageCache(async (query) => {
      const response = await fetch(`/api/incidents/detail?${query}`);
      if (!response.ok) throw new Error("Unable to load incident details");
      return response.json();
    });
  }
  const [incidentDetails, setIncidentDetails] = useState({});
  const incidentSelectionRequestRef = useRef(0);
  const [selectedEvent, setSelectedEvent] = useState(null);

  async function loadIncidentDetail(incident) {
    const query = incidentDetailQuery(incident);
    if (!query) return incident;
    const cached = incidentDetails[query];
    if (cached) return cached;
    try {
      const detail = await incidentDetailCacheRef.current.load(query);
      setIncidentDetails((current) => ({ ...current, [query]: detail }));
      return detail;
    } catch {
      return incident;
    }
  }

  async function openIncidentOverlay(incident) {
    const request = ++incidentSelectionRequestRef.current;
    setSelectedEvent(incident);
    const detail = await loadIncidentDetail(incident);
    if (request === incidentSelectionRequestRef.current) setSelectedEvent(detail);
  }

  function closeIncidentOverlay() {
    incidentSelectionRequestRef.current += 1;
    setSelectedEvent(null);
  }

  return {
    incidentDetailCacheRef,
    incidentDetails,
    setIncidentDetails,
    incidentSelectionRequestRef,
    selectedEvent,
    setSelectedEvent,
    openIncidentOverlay,
    closeIncidentOverlay,
  };
}

const STREAM_MODES = ["motion", "mjpeg", "webrtc"];
const STREAM_LABELS = {
  motion: "Auto",
  mjpeg: "MJPEG",
  webrtc: "WebRTC",
};
const MOTION_WEBRTC_HOLD_MS = 30_000;

function mediaAspect(element) {
  const width = element?.videoWidth || element?.naturalWidth || 0;
  const height = element?.videoHeight || element?.naturalHeight || 0;
  return aspectFromDimensions(width, height) || "16 / 9";
}

function mediaAspectRatio(aspect) {
  const [width, height] = String(aspect || "").split("/").map((value) => Number(value.trim()));
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return 16 / 9;
  }
  return width / height;
}

function CameraTile({ camera, timeZone, refresh, onOpen, onAspectChange, layout, customLayout = false, customStyle, resizeHandleProps = {}, startDelayMs = 0, dragHandleProps = {}, resizing = false, aspectSnapped = false }) {
  const [streamMode, setStreamMode] = useStoredState(`survng.streamMode.v3.${camera.id}`, "motion");
  const [sourceMode, setSourceMode] = useStoredState(`survng.sourceMode.${camera.id}`, "live");
  const [motionWindowNow, setMotionWindowNow] = useState(() => Date.now());
  const normalizedStreamMode = STREAM_MODES.includes(streamMode) ? streamMode : "motion";
  const lastMotionMs = new Date(camera.last_motion_at || 0).getTime();
  const motionActive = camera.running && Number.isFinite(lastMotionMs) && motionWindowNow - lastMotionMs <= MOTION_WEBRTC_HOLD_MS;
  const activeTransport = normalizedStreamMode === "motion" ? (motionActive ? "webrtc" : "snapshot") : normalizedStreamMode;
  const [deliveredSource, setDeliveredSource] = useState(sourceMode === "main" ? "main" : "live");
  const [aspect, setAspect] = useState(() => initialCameraAspect(camera, sourceMode, browserStorage(window)));
  const [mjpegToken, setMjpegToken] = useState(() => String(Date.now()));
  const [snapshotToken, setSnapshotToken] = useState(() => String(Date.now()));
  const [streamReady, setStreamReady] = useState(false);
  const [recordingBusy, setRecordingBusy] = useState(false);
  const [recordingError, setRecordingError] = useState("");
  const [detectionBusy, setDetectionBusy] = useState(false);
  const [detectionError, setDetectionError] = useState("");
  const [cameraActionBusy, setCameraActionBusy] = useState(false);
  const [cameraActionError, setCameraActionError] = useState("");
  const shouldUseWebRtc = camera.running && streamReady && activeTransport === "webrtc";
  const shouldUseMjpegStream = camera.running && streamReady && activeTransport === "mjpeg";
  const cameraConnected = camera.connected ?? camera.running;

  useEffect(() => {
    onAspectChange?.(camera.id, mediaAspectRatio(aspect));
  }, [aspect, camera.id, onAspectChange]);

  useEffect(() => {
    if (!STREAM_MODES.includes(streamMode)) setStreamMode("motion");
  }, [streamMode, setStreamMode]);

  useEffect(() => {
    const now = Date.now();
    setMotionWindowNow(now);
    if (!Number.isFinite(lastMotionMs)) return undefined;
    const remaining = MOTION_WEBRTC_HOLD_MS - (now - lastMotionMs);
    if (remaining <= 0) return undefined;
    const timer = window.setTimeout(() => setMotionWindowNow(Date.now()), remaining + 50);
    return () => window.clearTimeout(timer);
  }, [lastMotionMs]);

  useEffect(() => {
    setMjpegToken(String(Date.now()));
    setSnapshotToken(String(Date.now()));
    setStreamReady(false);
  }, [camera.id, sourceMode, activeTransport]);

  const authoritativeAspect = cameraSourceAspect(camera, deliveredSource);

  useEffect(() => {
    const nextSource = activeTransport === "webrtc" ? deliveredSource : normalizedLiveSource(sourceMode);
    const nextAspect = cameraSourceAspect(camera, nextSource)
      || initialCameraAspect(camera, nextSource, browserStorage(window));
    setAspect(nextAspect);
  }, [camera.id, sourceMode, activeTransport, deliveredSource, authoritativeAspect]);

  function rememberAspect(media, source, activate = true) {
    const normalizedSource = normalizedLiveSource(source);
    const measuredAspect = validLiveAspect(mediaAspect(media));
    if (!measuredAspect) return;
    writeStoredValue(
      browserStorage(window),
      liveAspectStorageKey(camera.id, normalizedSource),
      measuredAspect,
    );
    if (activate && !cameraSourceAspect(camera, normalizedSource)) setAspect(measuredAspect);
  }

  useEffect(() => {
    setStreamReady(false);
    setSnapshotToken(String(Date.now()));
    if (!camera.running) return undefined;
    const timer = window.setTimeout(() => setStreamReady(true), startDelayMs);
    return () => window.clearTimeout(timer);
  }, [camera.id, camera.running, sourceMode, activeTransport, startDelayMs]);

  useEffect(() => {
    if (!camera.running) return undefined;
    if (camera.running && streamReady && activeTransport !== "snapshot") return undefined;
    const timer = window.setInterval(
      () => setSnapshotToken(String(Date.now())),
      isMobileViewport() ? 8000 : 2000,
    );
    return () => window.clearInterval(timer);
  }, [camera.running, streamReady, activeTransport]);

  async function post(action) {
    if (cameraActionBusy) return;
    setCameraActionBusy(true);
    setCameraActionError("");
    try {
      const response = await fetch(`/api/cameras/${camera.id}/${action}`, { method: "POST" });
      if (!response.ok) throw new Error(`Camera control failed (${response.status})`);
      await refresh();
    } catch (error) {
      setCameraActionError(error instanceof Error ? error.message : "Camera control failed");
    } finally {
      setCameraActionBusy(false);
    }
  }

  async function toggleRecording() {
    if (recordingBusy || !camera.recording_configured) return;
    setRecordingBusy(true);
    setRecordingError("");
    try {
      const response = await fetch(`/api/cameras/${camera.id}/recording`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !camera.recording_enabled }),
      });
      if (!response.ok) throw new Error(`Recording control failed (${response.status})`);
      await refresh();
    } catch (error) {
      setRecordingError(error instanceof Error ? error.message : "Recording control failed");
    } finally {
      setRecordingBusy(false);
    }
  }

  async function toggleDetection() {
    if (detectionBusy) return;
    setDetectionBusy(true);
    setDetectionError("");
    try {
      const response = await fetch(`/api/cameras/${camera.id}/detection`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !camera.detection_enabled }),
      });
      if (!response.ok) throw new Error(`Detection control failed (${response.status})`);
      await refresh();
    } catch (error) {
      setDetectionError(error instanceof Error ? error.message : "Detection control failed");
    } finally {
      setDetectionBusy(false);
    }
  }

  function cycleStreamMode() {
    const index = STREAM_MODES.indexOf(normalizedStreamMode);
    setStreamMode(STREAM_MODES[(index + 1) % STREAM_MODES.length]);
  }

  function toggleSourceMode() {
    const nextSource = sourceMode === "main" ? "live" : "main";
    setDeliveredSource(nextSource);
    setAspect(initialCameraAspect(camera, nextSource, browserStorage(window)));
    setSourceMode(nextSource);
  }

  const posterSource = activeTransport === "webrtc" && sourceMode === "main" ? "live" : sourceMode;
  const imageUrl = appUrl(shouldUseMjpegStream
    ? `/api/cameras/${camera.id}/stream.mjpg?source=${sourceMode}&t=${mjpegToken}`
    : `/api/cameras/${camera.id}/snapshot.jpg?source=${posterSource}&t=${snapshotToken}`);

  return (
    <article
      className={`bento-card camera-tile ${layout ? "viewport-layout" : ""} ${customLayout ? "custom-layout-tile" : ""} ${motionActive ? "motion-active" : ""} ${resizing ? "resizing" : ""} ${aspectSnapped ? "aspect-snapped" : ""}`}
      data-motion-active={motionActive ? "true" : "false"}
      data-camera-id={camera.id}
      style={customLayout ? customStyle : layout ? {
        left: `${layout.x}px`,
        top: `${layout.y}px`,
        width: `${layout.width}px`,
        height: `${layout.height}px`,
      } : undefined}
    >
      <div
        className="video-frame camera-open-target"
        style={{ "--media-aspect": aspect }}
        role="button"
        tabIndex={0}
        onClick={() => onOpen(camera)}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onOpen(camera);
          }
        }}
        aria-label={`Open ${camera.name} live view`}
      >
        {!camera.running ? (
          <div className="camera-offline-state" role="img" aria-label={`${camera.name} is powered off`}>
            <Power size={24} />
          </div>
        ) : (
          <>
            <img
              className="camera-tile-poster"
              src={imageUrl}
              alt={`${camera.name} ${sourceMode === "main" ? "main" : "sub"} live stream`}
              onLoad={(event) => rememberAspect(
                event.currentTarget,
                posterSource,
                activeTransport !== "webrtc" || normalizedLiveSource(posterSource) === normalizedLiveSource(deliveredSource),
              )}
            />
            {shouldUseWebRtc ? (
              <div className="camera-live-layer">
                <WebRtcLive
                  cameraId={camera.id}
                  source={sourceMode}
                  timeZone={timeZone}
                  muted
                  showPoster={false}
                  onStageChange={(_stage, nextSource) => setDeliveredSource(normalizedLiveSource(nextSource))}
                  onReady={(media, _stage, readySource) => rememberAspect(media, readySource || deliveredSource)}
                />
              </div>
            ) : null}
          </>
        )}
        <div
          className="tile-header camera-hud"
          onClick={(event) => event.stopPropagation()}
          onKeyDown={(event) => event.stopPropagation()}
        >
          <div className="tile-title">
            <h2>{camera.name}</h2>
          </div>
          <div className="tile-controls" aria-label={`${camera.name} controls`}>
            {dragHandleProps.onPointerDown ? <button
              type="button"
              className="tile-control-button icon-only camera-drag-handle"
              title="Drag to move camera"
              aria-label={`Move ${camera.name}`}
              {...dragHandleProps}
            >
              <GripVertical size={16} />
            </button> : null}
            <span className={`status-pill hud-status ${cameraConnected ? "ok" : "bad"}`} title={!camera.running ? "Powered off" : cameraConnected ? "Connected" : "Waiting for a fresh frame"}>
              <CircleDot size={13} /> <span className="hud-full-label">{cameraConnected ? "online" : "offline"}</span>
            </span>
            <button type="button" className="tile-control-button" onClick={toggleSourceMode} title="Switch main/sub stream">
              <Radio size={15} />
              <span className="hud-full-label">{sourceMode === "main" ? "Main" : "Sub"}</span>
              <span className="hud-short-label">{sourceMode === "main" ? "M" : "S"}</span>
            </button>
            <button
              type="button"
              className="tile-control-button"
              onClick={cycleStreamMode}
              title={normalizedStreamMode === "motion" ? `Automatic motion switching: ${activeTransport === "webrtc" ? "WebRTC active" : "snapshot idle"}` : "Cycle transport: Auto, MJPEG, WebRTC"}
            >
              <span className="hud-full-label">
                {normalizedStreamMode === "motion" ? `Auto ${activeTransport === "webrtc" ? "RTC" : "Snap"}` : STREAM_LABELS[normalizedStreamMode]}
              </span>
              <span className="hud-short-label">
                {normalizedStreamMode === "motion" ? (activeTransport === "webrtc" ? "RTC" : "Snap") : normalizedStreamMode === "mjpeg" ? "MJ" : "RTC"}
              </span>
            </button>
            <button
              type="button"
              className={`status-pill hud-toggle hud-icon ${camera.recording_enabled ? "ok" : ""} ${recordingError ? "bad" : ""}`}
              onClick={toggleRecording}
              disabled={recordingBusy || !camera.recording_configured}
              title={recordingError || (!camera.recording_configured ? "Recording is not configured for this camera" : camera.recording_enabled ? "Stop recording" : "Start recording")}
              aria-label={`${camera.recording_enabled ? "Stop" : "Start"} recording ${camera.name}`}
            >
              <Video size={13} /> <span className="hud-full-label">{recordingBusy ? "..." : "rec"}</span>
            </button>
            <button
              type="button"
              className={`status-pill hud-toggle hud-icon ${camera.detection_enabled ? "ok" : ""} ${detectionError ? "bad" : ""}`}
              onClick={toggleDetection}
              disabled={detectionBusy}
              title={detectionError || (camera.detection_enabled ? "Stop motion and object detection" : "Start motion and object detection")}
              aria-label={`${camera.detection_enabled ? "Stop" : "Start"} motion and object detection for ${camera.name}`}
            >
              <Radar size={13} /> <span className="hud-full-label">{detectionBusy ? "..." : "detect"}</span>
            </button>
            <button
              type="button"
              className={`tile-control-button icon-only ${camera.running ? "danger" : ""} ${cameraActionError ? "bad" : ""}`}
              onClick={() => post(camera.running ? "camera/stop" : "camera/start")}
              disabled={cameraActionBusy}
              title={cameraActionError || (camera.running ? "Stop camera" : "Start camera")}
              aria-label={`${camera.running ? "Stop" : "Start"} ${camera.name}`}
            >
              {cameraActionBusy ? <RefreshCcw className="spin" size={16} /> : <Power size={16} />}
            </button>
          </div>
        </div>
      </div>
      {customLayout ? <button
        type="button"
        className="camera-resize-handle"
        title="Drag to resize camera"
        aria-label={`Resize ${camera.name}`}
        {...resizeHandleProps}
      /> : null}
      {resizing ? <span className="camera-resize-hint">{aspectSnapped ? "Fit video" : "Free size"}</span> : null}
    </article>
  );
}

function LiveCameraOverlay({ camera, timeZone, onClose }) {
  const [source, setSource] = useStoredState(
    `survng.liveOverlaySource.${camera.id}`,
    preferredStreamSource(),
  );
  const [mediaReady, setMediaReady] = useState(false);
  const [transport, setTransport] = useState("webrtc");
  const activeSource = source === "main" ? "main" : "live";
  const [deliveredSource, setDeliveredSource] = useState(activeSource);
  const [aspect, setAspect] = useState(() => initialCameraAspect(camera, activeSource, browserStorage(window)));

  useEffect(() => {
    setMediaReady(false);
    setTransport("webrtc");
    setDeliveredSource(activeSource);
    setAspect(initialCameraAspect(camera, activeSource, browserStorage(window)));
  }, [camera.id, activeSource]);

  const authoritativeOverlayAspect = cameraSourceAspect(camera, deliveredSource);

  useEffect(() => {
    if (authoritativeOverlayAspect) setAspect(authoritativeOverlayAspect);
  }, [authoritativeOverlayAspect]);

  useEffect(() => {
    function onKey(event) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  function rememberAspect(media, sourceName = activeSource) {
    const nextAspect = mediaAspect(media);
    if (!cameraSourceAspect(camera, sourceName)) setAspect(nextAspect);
    writeStoredValue(
      browserStorage(window),
      liveAspectStorageKey(camera.id, sourceName),
      nextAspect,
    );
  }

  return (
    <div className="live-overlay" role="dialog" aria-modal="true" aria-label={`${camera.name} full live view`}>
      <button className="live-overlay-backdrop" type="button" onClick={onClose} aria-label="Close live view" />
      <section
        className="live-overlay-panel"
        style={{
          "--media-aspect": aspect,
          "--media-ratio": mediaAspectRatio(aspect),
        }}
      >
        <div className="live-overlay-head">
          <div>
            <h2>{camera.name}</h2>
            <div className="live-overlay-subtitle">
              <span>{sourceLabel(activeSource)} stream</span>
              {deliveredSource !== activeSource ? <span>using {sourceLabel(deliveredSource)} fallback</span> : null}
              <span className="live-transport-badge" aria-label={`Stream transport ${liveTransportLabel(transport)}`}>
                {liveTransportLabel(transport)}
              </span>
            </div>
          </div>
          <button type="button" className="tile-control-button" onClick={() => setSource(activeSource === "main" ? "live" : "main")} aria-label="Switch live stream">
            <Radio size={15} /> {sourceLabel(activeSource)}
          </button>
          <button type="button" className="tile-control-button icon-only" onClick={onClose} aria-label="Close live view">
            <X size={18} />
          </button>
        </div>
        <div className="live-overlay-media">
          {!mediaReady ? (
            <div className="live-media-status" role="status" aria-live="polite">
              <RefreshCcw className="spin" size={20} />
              <span>Connecting live stream...</span>
            </div>
          ) : null}
          <WebRtcLive
            cameraId={camera.id}
            source={activeSource}
            timeZone={timeZone}
            muted
            controls
            onStageChange={(nextTransport, nextSource) => {
              setTransport(nextTransport);
              setDeliveredSource(nextSource);
              if (nextSource !== deliveredSource) setMediaReady(false);
            }}
            onReady={(media, readyTransport, readySource) => {
              rememberAspect(media, readySource || deliveredSource);
              setMediaReady(true);
              if (readyTransport) setTransport(readyTransport);
            }}
          />
        </div>
      </section>
    </div>
  );
}

function eventObjects(event) {
  return event.objects || [];
}

function eventEpoch(event) {
  const explicit = Number(event?.created_epoch);
  if (Number.isFinite(explicit)) return explicit;
  const parsed = new Date(event?.created_at || 0).getTime() / 1000;
  return Number.isFinite(parsed) ? parsed : null;
}

function incidentClipWindow(event, before, after) {
  const anchor = eventEpoch(event);
  const children = event?.events || [];
  const childEpochs = children.map(eventEpoch).filter(Number.isFinite);
  const explicitStart = Number(event?.start_epoch);
  const explicitEnd = Number(event?.last_epoch);
  const start = Number.isFinite(explicitStart) ? explicitStart : childEpochs.length ? Math.min(...childEpochs) : anchor;
  const end = Number.isFinite(explicitEnd) ? explicitEnd : childEpochs.length ? Math.max(...childEpochs) : anchor;
  return {
    before: Math.max(0, before + (Number.isFinite(anchor) && Number.isFinite(start) ? anchor - start : 0)),
    after: Math.max(0, after + (Number.isFinite(anchor) && Number.isFinite(end) ? end - anchor : 0)),
  };
}

function incidentLabels(incident) {
  const labels = Array.isArray(incident.labels)
    ? incident.labels
    : eventObjects(incident).filter((object) => object.incident_eligible !== false).map((object) => object.label).filter(Boolean);
  return Array.from(new Set(labels.filter(Boolean)));
}

function IncidentObjectIcon({ label, size = 14 }) {
  const icons = {
    person: UserRound,
    face: ScanFace,
    car: CarFront,
    truck: Truck,
    bus: BusFront,
    bike: Bike,
    cat: Cat,
    dog: Dog,
    mower: Bot,
    object: CircleDot,
  };
  const Icon = icons[incidentObjectIconName(label)] || CircleDot;
  return <Icon size={size} strokeWidth={2.2} aria-hidden="true" />;
}

function IncidentObjectBadges({ labels }) {
  if (!labels.length) {
    return <span className="pill quiet object-icon-pill" aria-label="Motion only" title="Motion only"><Radar size={14} strokeWidth={2.2} aria-hidden="true" /></span>;
  }
  return labels.slice(0, 3).map((label) => (
    <span className="pill object-icon-pill" key={label} aria-label={label} title={label}>
      <IncidentObjectIcon label={label} />
    </span>
  ));
}

function hasDetectedObjects(event) {
  if (typeof event.has_objects === "boolean") return event.has_objects;
  return eventObjects(event).some((object) => object.label && object.incident_eligible !== false) || incidentLabels(event).length > 0;
}

function incidentZones(incident) {
  const zones = Array.isArray(incident.zones)
    ? incident.zones
    : eventObjects(incident).filter((object) => object.incident_eligible !== false).flatMap((object) => object.zones || []);
  return Array.from(new Set(zones.filter(Boolean)));
}


function objectBoxes(event, incidentEligibleOnly = false) {
  return eventObjects(event)
    .map((object) => ({ object, box: object?.box }))
    .filter(({ object, box }) => object?.label && object.snapshot_visible !== false && (!incidentEligibleOnly || object.incident_eligible !== false) && box && [box.x1, box.y1, box.x2, box.y2].every((value) => Number.isFinite(Number(value))))
    .map(({ object, box }) => ({
      label: object.label,
      confidence: object.confidence,
      maskPolygon: Array.isArray(object.mask_polygon)
        ? object.mask_polygon.filter((point) => Array.isArray(point) && point.length >= 2).map((point) => [Number(point[0]), Number(point[1])])
        : [],
      x1: Number(box.x1),
      y1: Number(box.y1),
      x2: Number(box.x2),
      y2: Number(box.y2),
    }))
    .filter((box) => box.x2 > box.x1 && box.y2 > box.y1);
}

function SnapshotImage({ event, alt, iconSize = 24, className = "", layerStyle = null, zoom = null, allowObjectFocus = true, showAnnotations = true, showTracking = false, incidentEligibleOnly = false, thumbnail = false, progressive = false, fullResolution = false, highQualityZoom = false, onRequestFullResolution, onImageSize, children }) {
  const boxes = objectBoxes(event, incidentEligibleOnly);
  const tracks = storedObjectTracks(event);
  const boxCoordinateSize = incidentDetectionFrameSize(event);
  const trackCoordinateSize = incidentTrackingFrameSize(event);
  const coordinateSize = showTracking ? trackCoordinateSize : boxCoordinateSize;
  const progressiveImageKey = `${event?.representative_event_id || event?.id || "none"}:${event?.snapshot_path || "none"}:${event?.snapshot_url || "stored"}`;
  const frameRef = useRef(null);
  const [imageSize, setImageSize] = useState(() => coordinateSize);
  const [loadedImageKey, setLoadedImageKey] = useState("");
  const [frameSize, setFrameSize] = useState(null);
  const [objectFocused, setObjectFocused] = useState(false);
  const [progressiveState, setProgressiveState] = useState({ key: "", base: false, intermediate: false, full: false });
  const progressiveReady = progressiveState.key === progressiveImageKey ? progressiveState : { base: false, intermediate: false, full: false };
  const imageReady = loadedImageKey === progressiveImageKey;
  const devicePixelRatio = Math.max(1, Math.min(4, Number(window.devicePixelRatio) || 1));
  const displayPixelWidth = (frameSize?.width || 0) * devicePixelRatio;
  const progressiveWidth = incidentProgressiveImageWidth(frameSize?.width, devicePixelRatio);
  const progressiveQuality = progressiveWidth > 1280 ? 90 : 86;
  const shouldLoadFullResolution = fullResolution || highQualityZoom || displayPixelWidth > 2560;
  const zoomLayout = useMemo(() => incidentZoomLayout(frameSize, zoom), [frameSize, zoom?.scale, zoom?.x, zoom?.y]);
  const renderingFrameSize = zoomLayout
    ? { width: zoomLayout.width, height: zoomLayout.height }
    : frameSize;
  const renderedImage = useMemo(() => {
    if (!imageSize?.width || !imageSize?.height || !renderingFrameSize?.width || !renderingFrameSize?.height) return null;
    const scale = Math.min(renderingFrameSize.width / imageSize.width, renderingFrameSize.height / imageSize.height);
    const width = imageSize.width * scale;
    const height = imageSize.height * scale;
    return {
      x: (renderingFrameSize.width - width) / 2,
      y: (renderingFrameSize.height - height) / 2,
      width,
      height,
      scale,
    };
  }, [imageSize, renderingFrameSize?.height, renderingFrameSize?.width]);
  const canFocus = allowObjectFocus && showAnnotations && boxes.length > 0 && renderedImage;

  useLayoutEffect(() => {
    setObjectFocused(false);
    setImageSize(coordinateSize);
  }, [progressiveImageKey]);

  useLayoutEffect(() => {
    const frame = frameRef.current;
    if (!frame) return undefined;
    function updateFrameSize() {
      const rect = frame.getBoundingClientRect();
      if (rect.width && rect.height) setFrameSize({ width: rect.width, height: rect.height });
    }
    updateFrameSize();
    const observer = new ResizeObserver(updateFrameSize);
    observer.observe(frame);
    return () => observer.disconnect();
  }, []);

  useLayoutEffect(() => {
    if (!coordinateSize || imageReady) return;
    setImageSize(coordinateSize);
  }, [coordinateSize?.height, coordinateSize?.width, imageReady]);

  function recordImageLoad(image, imageKey = progressiveImageKey) {
    if (image.naturalWidth && image.naturalHeight) {
      const size = { width: image.naturalWidth, height: image.naturalHeight };
      setImageSize(size);
      setLoadedImageKey(imageKey);
      onImageSize?.(size);
    }
  }

  function onImageLoad(loadEvent, imageKey = progressiveImageKey) {
    recordImageLoad(loadEvent.currentTarget, imageKey);
  }

  function markProgressiveReady(stage, loadEvent) {
    const image = loadEvent.currentTarget;
    if (!image.isConnected) return;
    recordImageLoad(image, progressiveImageKey);
    setProgressiveState((current) => ({
      ...(current.key === progressiveImageKey ? current : { key: progressiveImageKey, base: false, intermediate: false, full: false }),
      key: progressiveImageKey,
      [stage]: true,
    }));
  }

  const renderedBoxes = useMemo(() => {
    if (!renderedImage || !frameSize) return [];
    const sourceWidth = boxCoordinateSize?.width || imageSize?.width;
    const sourceHeight = boxCoordinateSize?.height || imageSize?.height;
    if (!sourceWidth || !sourceHeight) return [];
    const scaleX = renderedImage.width / sourceWidth;
    const scaleY = renderedImage.height / sourceHeight;
    return boxes.map((box) => ({
      ...box,
      left: renderedImage.x + box.x1 * scaleX,
      top: renderedImage.y + box.y1 * scaleY,
      width: (box.x2 - box.x1) * scaleX,
      height: (box.y2 - box.y1) * scaleY,
      maskPoints: box.maskPolygon.map(([x, y]) => `${renderedImage.x + x * scaleX},${renderedImage.y + y * scaleY}`).join(" "),
    })).filter((box) => box.width > 0 && box.height > 0);
  }, [boxes, boxCoordinateSize?.height, boxCoordinateSize?.width, frameSize, imageSize, renderedImage]);

  const renderedTracks = useMemo(() => {
    if (!renderedImage || !frameSize) return [];
    const sourceWidth = trackCoordinateSize?.width || imageSize?.width;
    const sourceHeight = trackCoordinateSize?.height || imageSize?.height;
    if (!sourceWidth || !sourceHeight) return [];
    const scaleX = renderedImage.width / sourceWidth;
    const scaleY = renderedImage.height / sourceHeight;
    return tracks.map((track) => ({
      ...track,
      left: renderedImage.x + track.x1 * scaleX,
      top: renderedImage.y + track.y1 * scaleY,
      width: (track.x2 - track.x1) * scaleX,
      height: (track.y2 - track.y1) * scaleY,
      pathPoints: track.trajectory.map(([, x, y]) => `${renderedImage.x + x * scaleX},${renderedImage.y + y * scaleY}`).join(" "),
    })).filter((track) => track.width > 0 && track.height > 0);
  }, [frameSize, imageSize, renderedImage, trackCoordinateSize?.height, trackCoordinateSize?.width, tracks]);

  const focusStyle = useMemo(() => {
    if (!canFocus || !frameSize || !renderedBoxes.length) return null;
    const minX = Math.max(0, Math.min(...renderedBoxes.map((box) => box.left)));
    const minY = Math.max(0, Math.min(...renderedBoxes.map((box) => box.top)));
    const maxX = Math.min(frameSize.width, Math.max(...renderedBoxes.map((box) => box.left + box.width)));
    const maxY = Math.min(frameSize.height, Math.max(...renderedBoxes.map((box) => box.top + box.height)));
    const boxWidth = Math.max(1, maxX - minX);
    const boxHeight = Math.max(1, maxY - minY);
    const padX = Math.max(frameSize.width * 0.04, boxWidth * 0.35);
    const padY = Math.max(frameSize.height * 0.04, boxHeight * 0.35);
    const cropX1 = Math.max(0, minX - padX);
    const cropY1 = Math.max(0, minY - padY);
    const cropX2 = Math.min(frameSize.width, maxX + padX);
    const cropY2 = Math.min(frameSize.height, maxY + padY);
    const cropWidth = Math.max(1, cropX2 - cropX1);
    const cropHeight = Math.max(1, cropY2 - cropY1);
    const centerX = cropX1 + cropWidth / 2;
    const centerY = cropY1 + cropHeight / 2;
    const scale = Math.min(5.5, Math.max(1, Math.min((frameSize.width * 0.82) / cropWidth, (frameSize.height * 0.82) / cropHeight)));
    return {
      transform: `translate3d(${frameSize.width / 2 - centerX * scale}px, ${frameSize.height / 2 - centerY * scale}px, 0) scale(${scale})`,
      transformOrigin: "0 0",
    };
  }, [canFocus, frameSize, renderedBoxes]);

  const zoomLayerStyle = zoomLayout ? {
    inset: "auto",
    left: `${zoomLayout.left}px`,
    top: `${zoomLayout.top}px`,
    width: `${zoomLayout.width}px`,
    height: `${zoomLayout.height}px`,
  } : null;
  const activeLayerStyle = objectFocused && focusStyle ? focusStyle : zoomLayerStyle || layerStyle;
  const aspect = imageSize ? `${imageSize.width} / ${imageSize.height}` : undefined;
  const prefersHighQualityRaster = highQualityZoom || objectFocused;

  return (
    <div ref={frameRef} className={`snapshot-frame ${objectFocused ? "object-focused" : ""} ${prefersHighQualityRaster ? "high-quality-zoom" : ""} ${className}`} style={aspect ? { "--snapshot-aspect": aspect } : undefined}>
      <div className="snapshot-layer" style={activeLayerStyle || undefined}>
        {event?.snapshot_path && eventSnapshotUrl(event) ? (
          progressive ? (
            <div className={`snapshot-progressive-stack ${progressiveReady.full ? "full-resolution-ready" : ""}`}>
              <img
                key={`${progressiveImageKey}-base`}
                className="snapshot-progressive-base"
                src={eventThumbnailUrl(event)}
                alt={alt}
                decoding="async"
                onLoad={(loadEvent) => markProgressiveReady("base", loadEvent)}
              />
              {progressiveReady.base ? (
                <img
                  key={`${progressiveImageKey}-intermediate`}
                  className={`snapshot-progressive-image snapshot-intermediate-image ${progressiveReady.intermediate ? "ready" : ""}`}
                  src={eventThumbnailUrl(event, progressiveWidth, progressiveQuality)}
                  alt=""
                  aria-hidden="true"
                  decoding="async"
                  onLoad={(loadEvent) => markProgressiveReady("intermediate", loadEvent)}
                />
              ) : null}
              {shouldLoadFullResolution ? (
                <img
                  key={`${progressiveImageKey}-full`}
                  className={`snapshot-progressive-image snapshot-full-resolution-image ${progressiveReady.full ? "ready" : ""}`}
                  src={eventSnapshotUrl(event)}
                  alt=""
                  aria-hidden="true"
                  decoding="sync"
                  fetchPriority="high"
                  onLoad={(loadEvent) => markProgressiveReady("full", loadEvent)}
                />
              ) : null}
            </div>
          ) : <img className={thumbnail ? "snapshot-thumbnail-image" : "snapshot-original-image"} src={thumbnail ? eventThumbnailUrl(event) : eventSnapshotUrl(event)} alt={alt} loading={thumbnail ? "lazy" : undefined} decoding="async" onLoad={(loadEvent) => onImageLoad(loadEvent, progressiveImageKey)} />
        ) : <div className="empty-thumb"><Camera size={iconSize} /></div>}
        {imageReady && showAnnotations && (!showTracking || !renderedTracks.length) && renderedBoxes.length ? (
          <div className="object-box-layer" aria-hidden="true">
            <svg className="object-mask-layer" viewBox={`0 0 ${frameSize.width} ${frameSize.height}`} preserveAspectRatio="none">
              {renderedBoxes.filter((box) => box.maskPoints).map((box, index) => (
                <polygon key={`mask-${box.label}-${index}`} points={box.maskPoints} />
              ))}
            </svg>
            {renderedBoxes.map((box, index) => (
              <span
                className="object-box"
                key={`${box.label}-${index}-${box.x1}-${box.y1}`}
                style={{ left: `${box.left}px`, top: `${box.top}px`, width: `${box.width}px`, height: `${box.height}px` }}
              >
                <strong>{box.label}{box.confidence ? ` ${(box.confidence * 100).toFixed(0)}%` : ""}</strong>
              </span>
            ))}
          </div>
        ) : null}
        {imageReady && showTracking && renderedTracks.length ? (
          <div className="object-track-layer" aria-hidden="true">
            <svg viewBox={`0 0 ${frameSize.width} ${frameSize.height}`} preserveAspectRatio="none" aria-hidden="true">
              {renderedTracks.map((track) => (
                <g className={`object-track-color-${Math.abs(track.trackId) % 6}`} key={`trail-${track.trackId}`}>
                  {track.pathPoints ? <polyline className="object-track-trail" points={track.pathPoints} vectorEffect="non-scaling-stroke" /> : null}
                  <circle className="object-track-end" cx={track.left + track.width / 2} cy={track.top + track.height / 2} r="4" vectorEffect="non-scaling-stroke" />
                </g>
              ))}
            </svg>
            {renderedTracks.map((track) => (
              <span
                className={`object-track-box object-track-color-${Math.abs(track.trackId) % 6}`}
                key={`track-${track.trackId}`}
                style={{ left: `${track.left}px`, top: `${track.top}px`, width: `${track.width}px`, height: `${track.height}px` }}
              >
                <strong>#{track.trackId} {track.label}</strong>
              </span>
            ))}
          </div>
        ) : null}
      </div>
      {canFocus ? (
        <button
          type="button"
          className={`snapshot-focus-button ${objectFocused ? "active" : ""}`}
          onClick={(event) => {
            event.stopPropagation();
            if (!objectFocused) onRequestFullResolution?.();
            setObjectFocused((focused) => !focused);
          }}
          title={objectFocused ? "Show full snapshot" : "Focus detected objects"}
          aria-label={objectFocused ? "Show full snapshot" : "Focus detected objects"}
        >
          <Crop size={14} />
        </button>
      ) : null}
      {children}
    </div>
  );
}

function StoredTrackVideoOverlay({ videoRef, tracks, coordinateSize, windowStartEpoch, mediaStartTime, mediaKey, sampleFps, lostTimeoutSeconds }) {
  const layerRef = useRef(null);
  const [playbackEpoch, setPlaybackEpoch] = useState(null);
  const [layerSize, setLayerSize] = useState(null);

  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return undefined;
    function updateLayerSize() {
      const rect = layer.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) setLayerSize({ width: rect.width, height: rect.height });
    }
    updateLayerSize();
    const observer = new ResizeObserver(updateLayerSize);
    observer.observe(layer);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(windowStartEpoch) || !Number.isFinite(mediaStartTime)) return undefined;
    let timer = null;
    let stopped = false;

    function update() {
      const epoch = playbackEpochAt(windowStartEpoch, video.currentTime, mediaStartTime);
      if (epoch !== null) setPlaybackEpoch(epoch);
    }

    function schedule() {
      if (stopped) return;
      update();
      if (!video.paused && !video.ended) timer = window.setTimeout(schedule, 100);
    }

    function onPlaying() {
      if (timer !== null) window.clearTimeout(timer);
      schedule();
    }

    function onSeeked() {
      update();
    }

    video.addEventListener("playing", onPlaying);
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("timeupdate", update);
    if (!video.paused) onPlaying();
    return () => {
      stopped = true;
      if (timer !== null) window.clearTimeout(timer);
      video.removeEventListener("playing", onPlaying);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("timeupdate", update);
    };
  }, [videoRef, windowStartEpoch, mediaStartTime, mediaKey]);

  const visibleTracks = useMemo(() => {
    if (!Number.isFinite(playbackEpoch)) return [];
    const holdSeconds = Math.max(0.5, Number(lostTimeoutSeconds) || 3);
    return tracks.flatMap((track) => {
      const frame = trackFrameAt(track, playbackEpoch, { holdSeconds, sampleFps });
      return frame ? [{ ...track, ...frame }] : [];
    });
  }, [lostTimeoutSeconds, playbackEpoch, sampleFps, tracks]);

  const secondsUntilTracking = useMemo(() => {
    if (!Number.isFinite(playbackEpoch) || visibleTracks.length) return null;
    const nextEpoch = tracks
      .flatMap((track) => track.boxHistory.length ? [track.boxHistory[0][0]] : [])
      .filter((epoch) => epoch > playbackEpoch)
      .sort((left, right) => left - right)[0];
    return Number.isFinite(nextEpoch) ? Math.max(1, Math.ceil(nextEpoch - playbackEpoch)) : null;
  }, [playbackEpoch, tracks, visibleTracks.length]);

  const coordinateTransform = useMemo(
    () => containedFrameTransform(layerSize, coordinateSize),
    [coordinateSize?.height, coordinateSize?.width, layerSize],
  );
  const renderedTracks = useMemo(() => {
    if (!coordinateTransform) return [];
    const { x, y, scale } = coordinateTransform;
    return visibleTracks.map((track) => ({
      ...track,
      renderedBox: [
        x + track.box[0] * scale,
        y + track.box[1] * scale,
        x + track.box[2] * scale,
        y + track.box[3] * scale,
      ],
      renderedPath: track.path.map(([pathX, pathY]) => [x + pathX * scale, y + pathY * scale]),
    }));
  }, [coordinateTransform, visibleTracks]);

  if (!coordinateSize?.width || !coordinateSize?.height || !tracks.some((track) => track.boxHistory.length)) return null;
  return (
    <div ref={layerRef} className="object-track-video-layer" aria-hidden="true">
      {layerSize ? <svg viewBox={`0 0 ${layerSize.width} ${layerSize.height}`} preserveAspectRatio="none" aria-hidden="true">
        {renderedTracks.map((track) => (
          <g className={`object-track-color-${Math.abs(track.trackId) % 6} ${track.recovery ? "reid-recovered" : ""} ${track.estimated ? "track-estimated" : ""}`} key={`video-path-${track.trackId}`}>
            {track.renderedPath.length > 1 ? <polyline className="object-track-trail" points={track.renderedPath.map(([x, y]) => `${x},${y}`).join(" ")} vectorEffect="non-scaling-stroke" /> : null}
            <rect className="object-track-video-box" x={track.renderedBox[0]} y={track.renderedBox[1]} width={track.renderedBox[2] - track.renderedBox[0]} height={track.renderedBox[3] - track.renderedBox[1]} vectorEffect="non-scaling-stroke" />
          </g>
        ))}
      </svg> : null}
      {renderedTracks.map((track) => (
        <span
          className={`object-track-video-label object-track-color-${Math.abs(track.trackId) % 6} ${track.recovery ? "reid-recovered" : ""} ${track.estimated ? "track-estimated" : ""}`}
          key={`video-label-${track.trackId}`}
          style={{ left: `${track.renderedBox[0]}px`, top: `${track.renderedBox[1]}px` }}
        >
          #{track.trackId} {track.label}{track.estimated ? " · estimated" : ""}{track.recovery ? ` · ReID ${Math.round(track.recovery.similarity * 100)}%` : ""}
        </span>
      ))}
      {secondsUntilTracking ? <span className="object-track-video-waiting">Tracking begins in {secondsUntilTracking}s</span> : null}
    </div>
  );
}


function IncidentClipLayer({ event, trackingEvent, active, analysisMode = "clean", onAnalysisStats, onEnded }) {
  const videoRef = useRef(null);
  const [clipInfo, setClipInfo] = useState(null);
  const [clipLoading, setClipLoading] = useState(false);
  const [clipError, setClipError] = useState("");
  const [playback, setPlayback] = useState(null);
  const [playbackOriginTime, setPlaybackOriginTime] = useState(null);
  const storedTracks = storedObjectTracks(trackingEvent || event);

  useEffect(() => {
    let cancelled = false;
    async function loadClipSettings() {
      const eventId = Number(event?.representative_event_id || event?.id);
      if (!active || !Number.isFinite(eventId)) {
        setClipInfo(null);
        setPlayback(null);
        setPlaybackOriginTime(null);
        setClipLoading(false);
        setClipError(active ? "No event video available" : "");
        return;
      }
      setClipInfo(null);
      setPlayback(null);
      setPlaybackOriginTime(null);
      setClipLoading(true);
      setClipError("");
      const info = await loadIncidentClipInfo(event, () => cancelled);
      if (!info) return;
      setClipInfo(info);
      setPlayback({ url: info.streamUrl, mimeType: "application/vnd.apple.mpegurl" });
    }
    loadClipSettings();
    return () => { cancelled = true; };
  }, [active, event?.id, event?.representative_event_id, event?.start_epoch, event?.last_epoch]);

  if (!active) return null;
  return (
    <div className="incident-video-layer" onClick={(event) => event.stopPropagation()}>
      {clipInfo && playback && !clipError ? (
        <>
          <ShakaVideo
            ref={videoRef}
            src={playback.url}
            mimeType={playback.mimeType}
            autoPlay
            controls
            playsInline
            preload="metadata"
            onReady={(_player, video) => {
              setPlaybackOriginTime(0);
              if (clipInfo.playbackStartOffset > 0 && video) {
                const seekToIncident = () => {
                  const targetTime = clipInfo.playbackStartOffset;
                  video.currentTime = Number.isFinite(video.duration)
                    ? Math.min(targetTime, Math.max(0, video.duration - 0.25))
                    : targetTime;
                };
                if (video.paused) video.addEventListener("playing", seekToIncident, { once: true });
                else seekToIncident();
              }
              setClipLoading(false);
              setClipError("");
            }}
            onError={() => {
              if (playback.url !== clipInfo.downloadUrl) {
                setClipLoading(true);
                setPlaybackOriginTime(null);
                setClipInfo((current) => current ? {
                  ...current,
                  windowStartEpoch: current.requestedWindowStartEpoch,
                  playbackStartOffset: current.initialPlaybackOffset,
                } : current);
                setPlayback({ url: clipInfo.downloadUrl, mimeType: "video/mp4" });
              } else {
                setClipLoading(false);
                setClipError("No recording window found");
              }
            }}
            onEnded={onEnded}
          />
          {analysisMode === "tracks" && storedTracks.length ? (
            <StoredTrackVideoOverlay
              videoRef={videoRef}
              tracks={storedTracks}
              coordinateSize={{
                width: Number(trackingEvent?.object_tracking?.frame_width),
                height: Number(trackingEvent?.object_tracking?.frame_height),
              }}
              windowStartEpoch={clipInfo.windowStartEpoch}
              mediaStartTime={playbackOriginTime}
              mediaKey={playback.url}
              sampleFps={trackingEvent?.object_tracking?.sample_fps}
              lostTimeoutSeconds={trackingEvent?.object_tracking?.lost_timeout_seconds}
            />
          ) : null}
          <DebugDetectionOverlay videoRef={videoRef} active={analysisMode === "ai"} confidence={0.35} onStats={onAnalysisStats} />
          {clipLoading ? <div className="incident-video-status preparing">Preparing incident video...</div> : null}
        </>
      ) : (
        <div className="incident-video-status">{clipLoading ? "Preparing video..." : clipError || "No event video available"}</div>
      )}
    </div>
  );
}

function IncidentCard({ incident, timeZone, expanded, selected = false, thumbnailAnnotations = true, desktopWorkspace = false, analysisMode = "clean", replayRequest = 0, onAnalysisStats, onToggle, onSelect, onPreviewChange, onImageSize }) {
  const rawEvents = incident.events || [];
  const motionObservations = incident.motion_observations || [];
  const showSubEvents = rawEvents.length > 1 || motionObservations.length > 0;
  const [selectedPreview, setSelectedPreview] = useState(null);
  const [workspaceView, setWorkspaceView] = useState(() => {
    try {
      const stored = window.sessionStorage.getItem("survng.incidentWorkspaceView.v1");
      return ["mosaic", "evidence"].includes(stored) ? stored : "focus";
    } catch { return "focus"; }
  });
  const [selectedEvidence, setSelectedEvidence] = useState(null);
  const [mosaicPageIndex, setMosaicPageIndex] = useState(0);
  const [subEventsOpen, setSubEventsOpen] = useState(false);
  const [inlineVideoActive, setInlineVideoActive] = useState(false);
  const [snapshotZoom, setSnapshotZoom] = useState({ scale: 1, x: 0, y: 0 });
  const previewRef = useRef(null);
  const snapshotZoomRef = useRef(snapshotZoom);
  const panGestureRef = useRef({ pointerId: null, startX: 0, startY: 0, panX: 0, panY: 0, moved: false });
  const replayRequestRef = useRef(replayRequest);
  const evidenceSource = selectedPreview || incident;
  const evidenceFrames = useMemo(() => incidentEvidenceFrames(evidenceSource), [evidenceSource]);
  const evidenceItems = useMemo(() => evidenceFrames.map((frame) => {
    if (frame.kind === "snapshot") return {
      ...frame,
      event: {
        ...evidenceSource,
        created_at: new Date(frame.epoch * 1000).toISOString(),
      },
    };
    const snapshotUrl = `/api/cameras/${encodeURIComponent(incident.camera_id)}/recordings/preview.jpg?epoch=${encodeURIComponent(frame.epoch)}&source=main&width=1280&exact=true`;
    return {
      ...frame,
      event: {
        ...evidenceSource,
        snapshot_path: "recording-evidence",
        snapshot_url: snapshotUrl,
        created_at: new Date(frame.epoch * 1000).toISOString(),
        objects: [],
        object_tracking: null,
      },
    };
  }), [evidenceFrames, evidenceSource, incident.camera_id]);
  const preview = selectedEvidence?.event || selectedPreview || incident;
  const mosaicEvents = useMemo(() => incidentMosaicEvents(incident), [incident]);
  const mosaic = useMemo(() => incidentMosaicPage(mosaicEvents, mosaicPageIndex), [mosaicEvents, mosaicPageIndex]);
  const canShowMosaic = desktopWorkspace && expanded && mosaicEvents.length > 1;
  const canShowEvidence = desktopWorkspace && expanded && evidenceItems.length > 0;
  const activeWorkspaceView = workspaceView === "mosaic" && !canShowMosaic
    ? "focus"
    : workspaceView === "evidence" && !canShowEvidence ? "focus" : workspaceView;
  const trackingPreview = incidentTrackingSource(preview, incident) || preview;
  const labels = incidentLabels(incident);
  const eventCount = incident.event_count || rawEvents.length || 1;
  const observationCount = Number(incident.motion_observation_count || motionObservations.length || 0);
  const countText = `${eventCount} ${eventCount === 1 ? "event" : "events"}${observationCount ? ` · ${observationCount} additional motion update${observationCount === 1 ? "" : "s"}` : ""}`;
  const triggerLabel = incidentTriggerLabel(incident);
  const triggerTitle = triggerLabel === "EMA" ? "EMA visual backup trigger" : "Camera motion trigger";
  const incidentTimeline = [
    ...rawEvents.map((event) => ({ kind: "event", item: event })),
    ...motionObservations.map((observation) => ({ kind: "activity", item: observation })),
  ].sort((left, right) => Date.parse(right.item.created_at || 0) - Date.parse(left.item.created_at || 0));
  const timeText = incident.start_at && incident.end_at && incident.start_at !== incident.end_at
    ? `${formatDateTime(incident.start_at, timeZone)} - ${formatDuration(incident.duration_seconds)}`
    : formatDateTime(incident.created_at, timeZone);
  const previewTimeText = preview.created_at ? formatDateTime(preview.created_at, timeZone) : timeText;

  useEffect(() => {
    if (!expanded) {
      setSelectedPreview(null);
      setSubEventsOpen(false);
      setInlineVideoActive(false);
    }
  }, [expanded]);

  useEffect(() => {
    setSelectedPreview(null);
    setSelectedEvidence(null);
    setMosaicPageIndex(0);
    setSubEventsOpen(false);
    setInlineVideoActive(false);
    replayRequestRef.current = replayRequest;
  }, [incident.id]);

  useEffect(() => {
    try { window.sessionStorage.setItem("survng.incidentWorkspaceView.v1", workspaceView); } catch { /* Session preference is optional. */ }
  }, [workspaceView]);

  useEffect(() => {
    setInlineVideoActive(false);
    resetSnapshotZoom();
  }, [preview.id, preview.created_at]);

  useEffect(() => {
    const previousRequest = replayRequestRef.current;
    replayRequestRef.current = replayRequest;
    if (expanded && replayRequest > previousRequest) setInlineVideoActive(true);
  }, [expanded, replayRequest]);

  useEffect(() => {
    if (activeWorkspaceView !== "focus") setInlineVideoActive(false);
  }, [activeWorkspaceView]);

  useEffect(() => {
    if (!expanded || !onPreviewChange) return;
    const representative = rawEvents.find((event) => Number(event.id) === Number(incident.representative_event_id));
    onPreviewChange(Number((selectedPreview || representative || incident).id));
  }, [expanded, incident, onPreviewChange, rawEvents, selectedPreview]);

  function toggle() {
    onToggle(incident.id);
  }

  function onKey(keyEvent) {
    if (keyEvent.key === "Enter" || keyEvent.key === " ") {
      keyEvent.preventDefault();
      toggle();
    }
  }

  function openPreview(pointerEvent) {
    pointerEvent.stopPropagation();
    if (panGestureRef.current.moved) {
      panGestureRef.current.moved = false;
      return;
    }
    if (desktopWorkspace && expanded && snapshotZoomRef.current.scale > 1) return;
    if (expanded) setInlineVideoActive(true);
    else toggle();
  }

  function clampSnapshotZoom(nextZoom) {
    const scale = Math.max(1, Math.min(6, nextZoom.scale));
    if (scale === 1) return { scale: 1, x: 0, y: 0 };
    const box = previewRef.current?.getBoundingClientRect();
    const limitX = box ? box.width * (scale - 1) / 2 : 0;
    const limitY = box ? box.height * (scale - 1) / 2 : 0;
    return {
      scale,
      x: Math.max(-limitX, Math.min(limitX, nextZoom.x || 0)),
      y: Math.max(-limitY, Math.min(limitY, nextZoom.y || 0)),
    };
  }

  function updateSnapshotZoom(updater) {
    setSnapshotZoom((current) => {
      const candidate = typeof updater === "function" ? updater(current) : updater;
      const next = clampSnapshotZoom(candidate);
      snapshotZoomRef.current = next;
      return next;
    });
  }

  function resetSnapshotZoom() {
    const reset = { scale: 1, x: 0, y: 0 };
    snapshotZoomRef.current = reset;
    setSnapshotZoom(reset);
  }

  function onPreviewWheel(wheelEvent) {
    if (!desktopWorkspace || !expanded) return;
    wheelEvent.preventDefault();
    const box = previewRef.current?.getBoundingClientRect();
    if (!box) return;
    const delta = Math.max(-120, Math.min(120, wheelEvent.deltaY));
    const factor = Math.exp(-delta * 0.0017);
    setInlineVideoActive(false);
    updateSnapshotZoom((current) => {
      const nextScale = Math.max(1, Math.min(6, current.scale * factor));
      if (nextScale === 1) return { scale: 1, x: 0, y: 0 };
      const anchorX = wheelEvent.clientX - box.left - box.width / 2;
      const anchorY = wheelEvent.clientY - box.top - box.height / 2;
      const scaleRatio = nextScale / current.scale;
      return {
        scale: nextScale,
        x: anchorX - (anchorX - current.x) * scaleRatio,
        y: anchorY - (anchorY - current.y) * scaleRatio,
      };
    });
  }

  function onPreviewPointerDown(pointerEvent) {
    if (!desktopWorkspace || !expanded || pointerEvent.pointerType === "touch" || snapshotZoomRef.current.scale <= 1) return;
    pointerEvent.preventDefault();
    pointerEvent.currentTarget.setPointerCapture(pointerEvent.pointerId);
    const current = snapshotZoomRef.current;
    panGestureRef.current = { pointerId: pointerEvent.pointerId, startX: pointerEvent.clientX, startY: pointerEvent.clientY, panX: current.x, panY: current.y, moved: false };
  }

  function onPreviewPointerMove(pointerEvent) {
    const gesture = panGestureRef.current;
    if (gesture.pointerId !== pointerEvent.pointerId || snapshotZoomRef.current.scale <= 1) return;
    pointerEvent.preventDefault();
    const dx = pointerEvent.clientX - gesture.startX;
    const dy = pointerEvent.clientY - gesture.startY;
    if (Math.abs(dx) + Math.abs(dy) > 4) gesture.moved = true;
    updateSnapshotZoom({ scale: snapshotZoomRef.current.scale, x: gesture.panX + dx, y: gesture.panY + dy });
  }

  function onPreviewPointerUp(pointerEvent) {
    if (panGestureRef.current.pointerId !== pointerEvent.pointerId) return;
    panGestureRef.current.pointerId = null;
  }

  function openOverlay(pointerEvent) {
    pointerEvent.stopPropagation();
    if (!onSelect) return;
    onSelect({
      ...preview,
      start_epoch: incident.start_epoch,
      last_epoch: incident.last_epoch,
      start_at: incident.start_at,
      end_at: incident.end_at,
      event_count: eventCount,
      events: rawEvents,
    });
  }

  function selectWorkspaceView(view) {
    setWorkspaceView(view);
    if (view !== "focus") {
      setInlineVideoActive(false);
      resetSnapshotZoom();
    }
  }

  function selectMosaicEvent(event) {
    setSelectedPreview(event);
    setSelectedEvidence(null);
    setInlineVideoActive(false);
    setWorkspaceView("focus");
  }

  function selectEvidenceItem(item) {
    setSelectedEvidence(item);
    setInlineVideoActive(false);
    setWorkspaceView("focus");
  }

  return (
    <article
      className={`incident-card ${hasDetectedObjects(incident) ? "has-objects" : ""} ${expanded ? "expanded" : ""} ${selected ? "selected" : ""}`}
      role="button"
      tabIndex={0}
      aria-current={selected ? "true" : undefined}
      onClick={toggle}
      onKeyDown={onKey}
      title={`${incident.camera_id} ${timeText}`}
    >
      <div
        ref={previewRef}
        className={`incident-preview ${activeWorkspaceView !== "focus" ? "mosaic-view" : ""} ${desktopWorkspace && expanded && activeWorkspaceView === "focus" ? "zoomable" : ""} ${snapshotZoom.scale > 1 ? "zoomed" : ""}`}
        onClick={activeWorkspaceView === "focus" ? openPreview : (event) => event.stopPropagation()}
        onDoubleClick={(pointerEvent) => {
          if (!desktopWorkspace || !expanded) return;
          pointerEvent.preventDefault();
          pointerEvent.stopPropagation();
          resetSnapshotZoom();
        }}
        onWheel={activeWorkspaceView === "focus" ? onPreviewWheel : undefined}
        onPointerDown={activeWorkspaceView === "focus" ? onPreviewPointerDown : undefined}
        onPointerMove={activeWorkspaceView === "focus" ? onPreviewPointerMove : undefined}
        onPointerUp={activeWorkspaceView === "focus" ? onPreviewPointerUp : undefined}
        onPointerCancel={activeWorkspaceView === "focus" ? onPreviewPointerUp : undefined}
        aria-label={activeWorkspaceView === "mosaic" ? "Incident event mosaic" : expanded ? "Play selected event video" : "Expand incident"}
        title={desktopWorkspace && expanded && activeWorkspaceView === "focus" ? (snapshotZoom.scale > 1 ? "Drag to pan. Double-click to reset zoom." : "Scroll to zoom. Click to play event video.") : undefined}
      >
        {activeWorkspaceView === "mosaic" ? (
          <div className={`incident-mosaic incident-mosaic-${mosaic.items.length}`} role="group" aria-label={`Events ${mosaic.page * 6 + 1} through ${mosaic.page * 6 + mosaic.items.length} of ${mosaicEvents.length}`}>
            {mosaic.items.map((event, index) => {
              const eventLabels = incidentLabels(event);
              const eventTrigger = incidentTriggerLabel(event);
              return (
                <button
                  type="button"
                  className="incident-mosaic-tile"
                  key={`${event.id || "event"}-${index}`}
                  onClick={(clickEvent) => { clickEvent.stopPropagation(); selectMosaicEvent(event); }}
                  aria-label={`Focus event at ${formatTimeOnly(event.created_at || incident.created_at, timeZone)}`}
                >
                  <SnapshotImage event={event} alt="incident event snapshot" className="incident-mosaic-snapshot" progressive thumbnail allowObjectFocus={false} showAnnotations showTracking={false}>
                    <div className="incident-mosaic-hud">
                      <time>{formatTimeOnly(event.created_at || incident.created_at, timeZone)}</time>
                      <div className="pill-row compact"><IncidentObjectBadges labels={eventLabels} /></div>
                      <span className={`incident-trigger-source trigger-${eventTrigger.toLowerCase()}`}>{eventTrigger}</span>
                    </div>
                  </SnapshotImage>
                </button>
              );
            })}
            {mosaic.pageCount > 1 ? (
              <div className="incident-mosaic-pager" onClick={(event) => event.stopPropagation()}>
                <button type="button" onClick={() => setMosaicPageIndex((page) => Math.max(0, page - 1))} disabled={mosaic.page === 0} aria-label="Previous mosaic events"><ChevronLeft size={15} /></button>
                <span>{mosaic.page + 1} / {mosaic.pageCount}</span>
                <button type="button" onClick={() => setMosaicPageIndex((page) => Math.min(mosaic.pageCount - 1, page + 1))} disabled={mosaic.page >= mosaic.pageCount - 1} aria-label="Next mosaic events"><ChevronRight size={15} /></button>
              </div>
            ) : null}
          </div>
        ) : activeWorkspaceView === "evidence" ? (
          <div className={`incident-evidence incident-evidence-${evidenceItems.length}`} role="group" aria-label="Incident evidence frames">
            {evidenceItems.map((item) => (
              <button type="button" className="incident-evidence-tile" key={item.key} onClick={(event) => { event.stopPropagation(); selectEvidenceItem(item); }} aria-label={`Focus ${item.label.toLowerCase()} frame`}>
                <SnapshotImage event={item.event} alt={`${item.label} evidence frame`} className="incident-evidence-snapshot" thumbnail allowObjectFocus={false} showAnnotations={item.kind === "snapshot"} showTracking={false}>
                  <div className="incident-evidence-hud">
                    <strong>{item.label}</strong>
                    <time>{formatTimeOnly(item.event.created_at, timeZone)}</time>
                    {item.confidence > 0 ? <span>{Math.round(item.confidence * 100)}%</span> : null}
                  </div>
                </SnapshotImage>
              </button>
            ))}
          </div>
        ) : (
          <SnapshotImage
            event={preview}
            alt="incident snapshot"
            zoom={desktopWorkspace && expanded ? snapshotZoom : null}
            highQualityZoom={desktopWorkspace && expanded && snapshotZoom.scale > 1}
            showAnnotations={desktopWorkspace && expanded ? true : showIncidentCardAnnotations(expanded, thumbnailAnnotations)}
            showTracking={false}
            incidentEligibleOnly={!expanded}
            thumbnail={!desktopWorkspace || !expanded}
            onImageSize={expanded && onImageSize ? (size) => onImageSize({
              ...size,
              eventId: Number(preview.representative_event_id || preview.id),
            }) : undefined}
          >
            {!desktopWorkspace || !expanded ? (
              <div className="incident-snapshot-hud">
                <div className="incident-snapshot-main">
                  <strong>{incident.camera_id}</strong>
                  <time>{expanded ? previewTimeText : timeText}</time>
                </div>
                <div className="pill-row compact incident-labels">
                  <IncidentObjectBadges labels={labels} />
                </div>
              </div>
            ) : null}
            <IncidentClipLayer
              event={incident}
              trackingEvent={trackingPreview}
              active={expanded && inlineVideoActive}
              analysisMode={analysisMode}
              onAnalysisStats={onAnalysisStats}
              onEnded={() => setInlineVideoActive(false)}
            />
            {desktopWorkspace
              ? (!expanded ? <span className={`event-count incident-trigger-source trigger-${triggerLabel.toLowerCase()}`} aria-label={`${triggerTitle}. ${countText}`} title={`${triggerTitle} · ${countText}`}>{triggerLabel}</span> : null)
              : <button type="button" className={`event-count incident-trigger-source trigger-${triggerLabel.toLowerCase()}`} onClick={openOverlay} onKeyDown={(event) => event.stopPropagation()} aria-label={`Open ${triggerTitle.toLowerCase()} incident`} title={`${triggerTitle} · Open incident`}>{triggerLabel}</button>}
          </SnapshotImage>
        )}
        {canShowMosaic || canShowEvidence ? (
          <div className="incident-workspace-view-toggle" role="group" aria-label="Incident image layout" onClick={(event) => event.stopPropagation()}>
            <button type="button" className={activeWorkspaceView === "focus" ? "active" : ""} onClick={() => selectWorkspaceView("focus")} aria-pressed={activeWorkspaceView === "focus"} title="Focus selected event"><Crop size={14} /><span>Focus</span></button>
            {canShowMosaic ? <button type="button" className={activeWorkspaceView === "mosaic" ? "active" : ""} onClick={() => selectWorkspaceView("mosaic")} aria-pressed={activeWorkspaceView === "mosaic"} title="Show all incident events"><Grid2X2 size={14} /><span>Mosaic</span></button> : null}
            {canShowEvidence ? <button type="button" className={activeWorkspaceView === "evidence" ? "active" : ""} onClick={() => selectWorkspaceView("evidence")} aria-pressed={activeWorkspaceView === "evidence"} title="Compare trigger, detection, selected, and tracking frames"><Images size={14} /><span>Evidence</span></button> : null}
          </div>
        ) : null}
      </div>
      {expanded && showSubEvents ? (
        <div className="incident-meta">
          <div className="incident-detail" onClick={(event) => event.stopPropagation()}>
            <button
              className="incident-events-toggle"
              type="button"
              onClick={() => setSubEventsOpen((open) => !open)}
              aria-expanded={subEventsOpen}
            >
              <span>{countText}</span>
              <strong>{subEventsOpen ? "Hide" : "Show"}</strong>
            </button>
            {subEventsOpen ? (
              <div className="incident-events">
                {incidentTimeline.map(({ kind, item }, index) => {
                  if (kind === "activity") {
                    const activityLabel = item.reason === "event_state_cooldown" ? "motion during cooldown" : "continued motion";
                    return (
                      <div className="incident-activity-row" key={`activity-${item.id || index}`}>
                        <span>{formatTimeOnly(item.created_at || incident.created_at, timeZone)}</span>
                        <strong>{activityLabel}</strong>
                      </div>
                    );
                  }
                  const event = item;
                  const eventLabels = incidentLabels(event);
                  const eventLabelText = eventLabels.length ? eventLabels.join(", ") : "motion";
                  const isActive = (preview.id || incident.id) === event.id && (preview.created_at || incident.created_at) === event.created_at;
                  return (
                    <button type="button" key={`${event.id || "event"}-${index}`} className={isActive ? "active" : ""} onClick={() => { setSelectedPreview(event); setSelectedEvidence(null); setInlineVideoActive(false); }}>
                      <span>{formatTimeOnly(event.created_at || incident.created_at, timeZone)}</span>
                      <strong>{eventLabelText}</strong>
                    </button>
                  );
                })}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </article>
  );
}

function RelatedAppearanceIncidents({ anchorEventId, selectedEventId, loadingEventId, cameraNameById, timeZone, onSelect, onReturn }) {
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!Number.isInteger(Number(anchorEventId)) || Number(anchorEventId) <= 0) {
      setMatches([]);
      return undefined;
    }
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    fetch(appUrl(relatedIncidentsPath(anchorEventId)), {
      signal: controller.signal,
    })
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Related incidents unavailable")))
      .then((payload) => {
        if (!cancelled) setMatches(visibleRelatedAppearances(payload, anchorEventId, 8));
      })
      .catch((requestError) => {
        if (!cancelled && requestError?.name !== "AbortError") {
          setMatches([]);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [anchorEventId]);

  if (!loading && !matches.length) return null;

  return (
    <section className="incident-related">
      <div className="incident-related-head">
        <h3>Related incidents</h3>
        {selectedEventId ? <button type="button" onClick={onReturn}>Selected incident</button> : null}
      </div>
      {loading ? <p>Finding related incidents…</p> : null}
      {matches.length ? <div className="incident-related-grid">
        {matches.map((match) => {
          const eventId = Number(match.event_id);
          const selected = eventId === Number(selectedEventId);
          const pending = eventId === Number(loadingEventId);
          return (
            <button type="button" className={selected ? "selected" : ""} key={eventId} onClick={() => onSelect(match)} disabled={pending} aria-pressed={selected} title={match.route_name ? `${match.route_name}: ${relatedEvidenceLabel(match)}` : `Preview related incident from ${cameraNameById.get(match.camera_id) || match.camera_id}`}>
              <img src={appUrl(relatedIncidentThumbnailPath(eventId))} alt={`${cameraNameById.get(match.camera_id) || match.camera_id} related incident`} loading="lazy" />
              <span><strong>{cameraNameById.get(match.camera_id) || match.camera_id}</strong><b>{relatedEvidenceLabel(match)}</b></span>
              <small>{pending ? "Loading…" : formatDateTime(match.created_at, timeZone)}</small>
            </button>
          );
        })}
      </div> : null}
    </section>
  );
}

function IncidentInspector({ incident, faceEvent, anchorEventId, selectedRelatedEventId, relatedLoadingEventId, cameraNameById, appConfig, timeZone, imageSize, analysisMode = "clean", analysisStats, onAnalysisModeChange, onFaceOpen, onRelatedSelect, onRelatedReturn }) {
  if (!incident) return <aside className="incident-inspector"><div className="empty-state">Select an incident.</div></aside>;
  const inspectedEvent = faceEvent || incident;
  const objects = eventObjects(inspectedEvent).filter((object) => object.label && object.incident_eligible !== false);
  const incidentTracking = incidentTrackingSource(inspectedEvent, incident)?.object_tracking;
  const objectTracks = incidentTracking?.tracks || [];
  const faces = faceEvent?.faces || [];
  const zones = incidentZones(inspectedEvent);
  const eventId = Number(inspectedEvent.representative_event_id || inspectedEvent.id);
  const before = Number(appConfig?.event_clip_before_seconds ?? 5);
  const after = Number(appConfig?.event_clip_after_seconds ?? 5);
  const window = incidentClipWindow(incident, before, after);
  const clipUrl = Number.isFinite(eventId) ? eventClipUrl(eventId, window.before, window.after) : "";

  return (
    <aside className="incident-inspector">
      <div className="incident-inspector-head">
        <div><strong>{incident.camera_id}</strong><time>{formatDateTime(inspectedEvent.created_at || incident.created_at, timeZone)}</time></div>
      </div>
      <RelatedAppearanceIncidents anchorEventId={anchorEventId} selectedEventId={selectedRelatedEventId} loadingEventId={relatedLoadingEventId} cameraNameById={cameraNameById} timeZone={timeZone} onSelect={onRelatedSelect} onReturn={onRelatedReturn} />
      <section className="incident-replay-analysis">
        <h3>Replay analysis</h3>
        <div className="incident-analysis-modes" role="group" aria-label="Replay analysis mode">
          <button type="button" className={analysisMode === "clean" ? "active" : ""} onClick={() => onAnalysisModeChange("clean")} title="Replay without an analysis overlay"><Play size={14} /> Clean</button>
          <button type="button" className={analysisMode === "tracks" ? "active" : ""} onClick={() => onAnalysisModeChange("tracks")} disabled={!objectTracks.length} title={objectTracks.length ? "Replay stored object tracks" : "No stored tracks for this incident"}><ListTree size={14} /> Tracks</button>
          <button type="button" className={analysisMode === "ai" ? "active" : ""} onClick={() => onAnalysisModeChange("ai")} title="Run OpenVINO detection while replaying"><Activity size={14} /> AI</button>
        </div>
        {analysisMode === "tracks" ? <small>{objectTracks.length} stored track{objectTracks.length === 1 ? "" : "s"} · {Number(incidentTracking?.sample_fps || 0) || "?"} FPS</small> : null}
        {analysisMode === "ai" && analysisStats ? <small className={analysisStats.error ? "analysis-error" : ""}>{analysisStats.error || `${analysisStats.inferenceMs ?? "--"} ms · ${analysisStats.objects ?? 0} current objects`}</small> : null}
      </section>
      <section>
        <h3>Objects</h3>
        {objects.length ? objects.map((object, index) => {
          const box = object.box || {};
          return (
            <div className="inspector-detection" key={`${object.label}-${index}`}>
              <div><strong>{object.label}</strong><span>{Math.round(Number(object.confidence || 0) * 100)}%</span></div>
              <code>{Math.round(Number(box.x1 || 0))}, {Math.round(Number(box.y1 || 0))} → {Math.round(Number(box.x2 || 0))}, {Math.round(Number(box.y2 || 0))}</code>
              {object.zones?.length ? <small>{object.zones.join(", ")}</small> : null}
            </div>
          );
        }) : <p>No eligible object detections.</p>}
      </section>
      <section>
        <h3>Faces</h3>
        {faces.length ? faces.map((face, index) => (
          <button type="button" className={`inspector-face ${face.status || "unknown"}`} key={`${face.status}-${face.name}-${index}`} onClick={() => onFaceOpen(face)}>
            <strong>{face.name || "Unknown"}</strong>
            <span>{Math.round(Number(face.confidence || 0) * 100)}%{Number(face.candidate_count || 0) > 1 ? ` · ${face.candidate_count} frames` : ""}</span>
          </button>
        )) : <p>No recognized faces.</p>}
      </section>
      <section>
        <h3>Zones</h3>
        <div className="pill-row">{zones.length ? zones.map((zone) => <span className="pill" key={zone}>{zone}</span>) : <span className="pill quiet">none</span>}</div>
      </section>
      <section>
        <h3>Incident</h3>
        <dl>
          <div><dt>Events</dt><dd>{incident.event_count || incident.events?.length || 1}</dd></div>
          <div><dt>Selected trigger</dt><dd>{incidentTriggerLabel(inspectedEvent)}</dd></div>
          <div><dt>Additional motion</dt><dd>{incident.motion_observation_count || incident.motion_observations?.length || 0}</dd></div>
          <div><dt>Duration</dt><dd>{formatDuration(incident.duration_seconds || 0)}</dd></div>
          <div><dt>Start</dt><dd>{formatTimeOnly(incident.start_at || incident.created_at, timeZone)}</dd></div>
          <div><dt>End</dt><dd>{formatTimeOnly(incident.end_at || incident.created_at, timeZone)}</dd></div>
          <div><dt>Loaded image</dt><dd>{imageSize?.width && imageSize?.height ? `${imageSize.width} × ${imageSize.height} px` : "—"}</dd></div>
        </dl>
      </section>
      <div className="incident-inspector-actions">
        {clipUrl ? <a href={clipUrl} download={`survng-${incident.camera_id}-${eventId}.mp4`}><Download size={15} /> Video</a> : null}
        {inspectedEvent.snapshot_path && eventSnapshotDownloadUrl(inspectedEvent) ? <a href={eventSnapshotDownloadUrl(inspectedEvent)}><Download size={15} /> Snapshot</a> : null}
      </div>
    </aside>
  );
}

function detectionIou(left, right) {
  const x1 = Math.max(Number(left.x1), Number(right.x1));
  const y1 = Math.max(Number(left.y1), Number(right.y1));
  const x2 = Math.min(Number(left.x2), Number(right.x2));
  const y2 = Math.min(Number(left.y2), Number(right.y2));
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const leftArea = Math.max(0, left.x2 - left.x1) * Math.max(0, left.y2 - left.y1);
  const rightArea = Math.max(0, right.x2 - right.x1) * Math.max(0, right.y2 - right.y1);
  return intersection / Math.max(1, leftArea + rightArea - intersection);
}

function DebugDetectionOverlay({ videoRef, active, confidence = 0.35, onStats }) {
  const canvasRef = useRef(null);
  const captureRef = useRef(document.createElement("canvas"));
  const tracksRef = useRef([]);
  const nextTrackIdRef = useRef(1);

  useEffect(() => {
    if (!active) {
      tracksRef.current = [];
      const canvas = canvasRef.current;
      canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
      return undefined;
    }
    let disposed = false;
    let timer = null;
    let controller = null;

    function updateTracks(detections) {
      const now = performance.now();
      const available = [...tracksRef.current];
      const next = detections.map((object) => {
        let bestIndex = -1;
        let bestScore = 0.25;
        available.forEach((track, index) => {
          if (!track || track.label !== object.label) return;
          const score = detectionIou(track.box, object.box);
          if (score > bestScore) {
            bestScore = score;
            bestIndex = index;
          }
        });
        const previous = bestIndex >= 0 ? available.splice(bestIndex, 1)[0] : null;
        return {
          id: previous?.id || nextTrackIdRef.current++,
          label: object.label,
          confidence: Number(object.confidence) || 0,
          box: object.box,
          seenAt: now,
        };
      });
      tracksRef.current = [...next, ...available.filter((track) => now - track.seenAt < 1200)].slice(0, 40);
      return next;
    }

    function draw(tracks, frameWidth, frameHeight) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;
      const width = Math.max(1, video.clientWidth);
      const height = Math.max(1, video.clientHeight);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const context = canvas.getContext("2d");
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);
      const scale = Math.min(width / frameWidth, height / frameHeight);
      const offsetX = (width - frameWidth * scale) / 2;
      const offsetY = (height - frameHeight * scale) / 2;
      context.font = "700 12px system-ui, sans-serif";
      context.lineWidth = 2;
      tracks.forEach((track) => {
        const x = offsetX + track.box.x1 * scale;
        const y = offsetY + track.box.y1 * scale;
        const boxWidth = (track.box.x2 - track.box.x1) * scale;
        const boxHeight = (track.box.y2 - track.box.y1) * scale;
        const label = `#${track.id} ${track.label} ${Math.round(track.confidence * 100)}%`;
        const labelWidth = context.measureText(label).width + 10;
        context.strokeStyle = "#2dd4bf";
        context.fillStyle = "rgba(13, 148, 136, 0.88)";
        context.strokeRect(x, y, boxWidth, boxHeight);
        context.fillRect(x, Math.max(0, y - 20), labelWidth, 20);
        context.fillStyle = "#ffffff";
        context.fillText(label, x + 5, Math.max(14, y - 6));
      });
    }

    async function sample() {
      const video = videoRef.current;
      if (disposed) return;
      if (!video || video.readyState < 2 || video.paused || document.hidden) {
        timer = window.setTimeout(sample, 350);
        return;
      }
      const capture = captureRef.current;
      const width = Math.min(960, video.videoWidth || 960);
      const height = Math.max(1, Math.round(width * (video.videoHeight || 540) / (video.videoWidth || 960)));
      capture.width = width;
      capture.height = height;
      capture.getContext("2d").drawImage(video, 0, 0, width, height);
      try {
        const blob = await new Promise((resolve) => capture.toBlob(resolve, "image/jpeg", 0.78));
        if (!blob || disposed) return;
        controller = new AbortController();
        const response = await fetch(`/api/detector/frame?confidence=${Number(confidence).toFixed(2)}`, {
          method: "POST",
          headers: { "Content-Type": "image/jpeg" },
          body: blob,
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        if (disposed) return;
        const tracks = updateTracks(payload.objects || []);
        draw(tracks, payload.width || width, payload.height || height);
        onStats?.({ inferenceMs: payload.elapsed_ms, objects: tracks.length, tracks: tracks.map((track) => track.id) });
      } catch (error) {
        if (!disposed && error.name !== "AbortError") onStats?.({ error: error.message || "Detection failed" });
      }
      if (!disposed) timer = window.setTimeout(sample, 500);
    }

    sample();
    return () => {
      disposed = true;
      controller?.abort();
      window.clearTimeout(timer);
    };
  }, [active, confidence, videoRef, onStats]);

  return <canvas ref={canvasRef} className="event-detection-canvas" aria-hidden="true" />;
}

function EventOverlay({ event, events, timeZone, onClose, onSelect, onRefresh }) {
  const clipVideoRef = useRef(null);
  const mediaRef = useRef(null);
  const comparisonPanelRef = useRef(null);
  const gestureRef = useRef({ mode: null, pointerId: null, startX: 0, startY: 0, panX: 0, panY: 0, moved: false, pinchDistance: 0, scale: 1 });
  const [clipInfo, setClipInfo] = useState(null);
  const [clipLoading, setClipLoading] = useState(false);
  const [clipError, setClipError] = useState("");
  const [playback, setPlayback] = useState(null);
  const [playbackOriginTime, setPlaybackOriginTime] = useState(null);
  const [videoActive, setVideoActive] = useState(false);
  const [detectionDebug, setDetectionDebug] = useState(false);
  const [detectionDebugStats, setDetectionDebugStats] = useState(null);
  const [trackingVisible, setTrackingVisible] = useState(false);
  const [trackingComparison, setTrackingComparison] = useState(null);
  const [trackingComparisonEngine, setTrackingComparisonEngine] = useState(null);
  const [trackingComparisonLoading, setTrackingComparisonLoading] = useState(false);
  const [trackingComparisonError, setTrackingComparisonError] = useState("");
  const [trackingComparisonHistory, setTrackingComparisonHistory] = useState({ items: [], summary: null });
  const [trackingVerdictLoading, setTrackingVerdictLoading] = useState(false);
  const [analysisToolsOpen, setAnalysisToolsOpen] = useState(false);
  const [zoom, setZoom] = useState({ scale: 1, x: 0, y: 0 });
  const [mediaSize, setMediaSize] = useState(() => incidentTrackingFrameSize(event));
  const [fullSnapshotRequested, setFullSnapshotRequested] = useState(false);
  const zoomRef = useRef(zoom);
  const [manualConfidence, setManualConfidence] = useStoredState("survng.manualDetectionConfidence.v1", "0.35");
  const [manualDetection, setManualDetection] = useState(null);
  const [manualLoading, setManualLoading] = useState(false);
  const [manualError, setManualError] = useState("");
  const trackingSource = incidentTrackingSource(event);
  const incidentTrackingEvent = trackingSource && trackingSource !== event ? trackingSource : null;
  const viewerEvent = incidentTrackingEvent ? {
    ...incidentTrackingEvent,
    start_epoch: event.start_epoch,
    last_epoch: event.last_epoch,
    start_at: event.start_at,
    end_at: event.end_at,
    event_count: event.event_count,
    events: event.events || [],
  } : event;
  const displayedEvent = manualDetection ? { ...viewerEvent, objects: manualDetection.objects || [] } : viewerEvent;
  const comparisonTracking = trackingComparisonEngine ? trackingComparison?.engines?.[trackingComparisonEngine] : null;
  const trackingEvent = comparisonTracking
    ? { ...displayedEvent, object_tracking: {
      ...comparisonTracking,
      sample_fps: trackingComparison.sample_fps,
      lost_timeout_seconds: trackingComparison.lost_timeout_seconds,
      frame_width: trackingComparison.frame_width,
      frame_height: trackingComparison.frame_height,
    } }
    : displayedEvent;
  const storedTracks = storedObjectTracks(trackingEvent);
  const reidDiagnostics = trackingEvent.object_tracking?.reid_diagnostics || {};
  const reidAttemptReasons = reidDiagnostics.inference_attempts_by_reason || {};
  const replayTrackCount = storedTracks.filter((track) => track.boxHistory.length).length;
  const manualConfidenceNumber = Number(manualConfidence);
  const safeManualConfidence = Number.isFinite(manualConfidenceNumber) ? Math.max(0.01, Math.min(0.99, manualConfidenceNumber)) : 0.35;
  const manualEventId = Number(viewerEvent.representative_event_id || viewerEvent.id);
  const downloadName = `survng-${String(viewerEvent.camera_id || "camera")}-${String(viewerEvent.created_at || viewerEvent.id || "event").replace(/[^0-9A-Za-z_-]+/g, "-")}.mp4`;

  useEffect(() => {
    setTrackingVisible(false);
  }, [event.id]);

  useEffect(() => {
    let cancelled = false;
    async function loadClipSettings() {
      const eventId = Number(viewerEvent.representative_event_id || viewerEvent.id);
      if (!Number.isFinite(eventId)) {
        setClipInfo(null);
        setClipLoading(false);
        setClipError("No event video available");
        return;
      }
      setClipInfo(null);
      setPlayback(null);
      setPlaybackOriginTime(null);
      setClipLoading(true);
      setClipError("");
      setVideoActive(false);
      const info = await loadIncidentClipInfo(viewerEvent, () => cancelled);
      if (!info) return;
      setClipInfo(info);
      setPlayback({ url: info.streamUrl, mimeType: "application/vnd.apple.mpegurl" });
    }
    loadClipSettings();
    return () => { cancelled = true; };
  }, [viewerEvent.id, viewerEvent.representative_event_id, viewerEvent.start_epoch, viewerEvent.last_epoch]);

  useEffect(() => {
    let cancelled = false;
    const cameraId = String(event.camera_id || "");
    setTrackingComparisonHistory({ items: [], summary: null });
    if (!cameraId) return undefined;
    fetch(`/api/tracking-comparisons?camera_id=${encodeURIComponent(cameraId)}&limit=10`)
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.detail || "Comparison history unavailable");
        if (!cancelled) setTrackingComparisonHistory(payload);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [event.camera_id]);

  function playEventClip() {
    if (!clipInfo || clipError) return;
    setVideoActive(true);
    const video = clipVideoRef.current;
    if (!video) return;
    if (video.ended || video.currentTime >= Math.max(0, video.duration - 0.1)) video.currentTime = 0;
    video.play().catch(() => {});
  }


  function clampZoom(nextZoom) {
    const scale = Math.max(1, Math.min(6, nextZoom.scale));
    if (scale === 1) return { scale: 1, x: 0, y: 0 };
    const box = mediaRef.current?.getBoundingClientRect();
    const limitX = box ? box.width * (scale - 1) / 2 : 0;
    const limitY = box ? box.height * (scale - 1) / 2 : 0;
    return {
      scale,
      x: Math.max(-limitX, Math.min(limitX, nextZoom.x || 0)),
      y: Math.max(-limitY, Math.min(limitY, nextZoom.y || 0)),
    };
  }

  function updateZoom(updater) {
    setZoom((current) => {
      const candidate = typeof updater === "function" ? updater(current) : updater;
      const next = clampZoom(candidate);
      zoomRef.current = next;
      return next;
    });
  }

  function zoomAround(clientX, clientY, factor) {
    const box = mediaRef.current?.getBoundingClientRect();
    if (!box) return;
    setVideoActive(false);
    setFullSnapshotRequested(true);
    updateZoom((current) => {
      const nextScale = Math.max(1, Math.min(6, current.scale * factor));
      if (nextScale === 1) return { scale: 1, x: 0, y: 0 };
      const anchorX = clientX - box.left - box.width / 2;
      const anchorY = clientY - box.top - box.height / 2;
      const scaleRatio = nextScale / current.scale;
      return {
        scale: nextScale,
        x: anchorX - (anchorX - current.x) * scaleRatio,
        y: anchorY - (anchorY - current.y) * scaleRatio,
      };
    });
  }

  function onMediaWheel(wheelEvent) {
    if (videoActive) return;
    wheelEvent.preventDefault();
    const delta = Math.max(-120, Math.min(120, wheelEvent.deltaY));
    zoomAround(wheelEvent.clientX, wheelEvent.clientY, Math.exp(-delta * 0.0017));
  }

  function touchDistance(touches) {
    if (touches.length < 2) return 0;
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.hypot(dx, dy);
  }

  function touchCenter(touches) {
    return {
      x: (touches[0].clientX + touches[1].clientX) / 2,
      y: (touches[0].clientY + touches[1].clientY) / 2,
    };
  }

  function onMediaTouchStart(touchEvent) {
    if (videoActive) return;
    if (touchEvent.touches.length === 2) {
      touchEvent.preventDefault();
      setFullSnapshotRequested(true);
      const center = touchCenter(touchEvent.touches);
      const current = zoomRef.current;
      gestureRef.current = { mode: "pinch", pointerId: null, pinchDistance: touchDistance(touchEvent.touches), scale: current.scale, startX: center.x, startY: center.y, panX: current.x, panY: current.y, moved: true };
    } else if (touchEvent.touches.length === 1 && zoomRef.current.scale > 1) {
      touchEvent.preventDefault();
      const touch = touchEvent.touches[0];
      const current = zoomRef.current;
      gestureRef.current = { mode: "touch-pan", pointerId: null, pinchDistance: 0, scale: current.scale, startX: touch.clientX, startY: touch.clientY, panX: current.x, panY: current.y, moved: false };
    }
  }

  function onMediaTouchMove(touchEvent) {
    if (videoActive) return;
    const gesture = gestureRef.current;
    if (touchEvent.touches.length === 2 && gesture.mode === "pinch" && gesture.pinchDistance) {
      touchEvent.preventDefault();
      const box = mediaRef.current?.getBoundingClientRect();
      if (!box) return;
      const distance = touchDistance(touchEvent.touches);
      const center = touchCenter(touchEvent.touches);
      const nextScale = Math.max(1, Math.min(6, gesture.scale * (distance / gesture.pinchDistance)));
      const scaleRatio = nextScale / gesture.scale;
      const anchorX = gesture.startX - box.left - box.width / 2;
      const anchorY = gesture.startY - box.top - box.height / 2;
      updateZoom({
        scale: nextScale,
        x: anchorX - (anchorX - gesture.panX) * scaleRatio + center.x - gesture.startX,
        y: anchorY - (anchorY - gesture.panY) * scaleRatio + center.y - gesture.startY,
      });
      return;
    }
    if (touchEvent.touches.length === 1 && gesture.mode === "touch-pan" && zoomRef.current.scale > 1) {
      touchEvent.preventDefault();
      const touch = touchEvent.touches[0];
      const dx = touch.clientX - gesture.startX;
      const dy = touch.clientY - gesture.startY;
      if (Math.abs(dx) + Math.abs(dy) > 4) gesture.moved = true;
      updateZoom({ scale: zoomRef.current.scale, x: gesture.panX + dx, y: gesture.panY + dy });
    }
  }

  function onMediaTouchEnd(touchEvent) {
    if (touchEvent.touches.length === 1 && zoomRef.current.scale > 1) {
      const touch = touchEvent.touches[0];
      const current = zoomRef.current;
      gestureRef.current = { mode: "touch-pan", pointerId: null, pinchDistance: 0, scale: current.scale, startX: touch.clientX, startY: touch.clientY, panX: current.x, panY: current.y, moved: true };
      return;
    }
    gestureRef.current.mode = null;
    gestureRef.current.pinchDistance = 0;
  }

  function onMediaPointerDown(pointerEvent) {
    if (videoActive || pointerEvent.pointerType === "touch" || zoomRef.current.scale <= 1) return;
    pointerEvent.preventDefault();
    pointerEvent.currentTarget.setPointerCapture(pointerEvent.pointerId);
    const current = zoomRef.current;
    gestureRef.current = { mode: "pointer-pan", pointerId: pointerEvent.pointerId, startX: pointerEvent.clientX, startY: pointerEvent.clientY, panX: current.x, panY: current.y, moved: false, pinchDistance: 0, scale: current.scale };
  }

  function onMediaPointerMove(pointerEvent) {
    const gesture = gestureRef.current;
    if (gesture.pointerId !== pointerEvent.pointerId || zoomRef.current.scale <= 1) return;
    pointerEvent.preventDefault();
    const dx = pointerEvent.clientX - gesture.startX;
    const dy = pointerEvent.clientY - gesture.startY;
    if (Math.abs(dx) + Math.abs(dy) > 4) gesture.moved = true;
    updateZoom({ scale: zoomRef.current.scale, x: gesture.panX + dx, y: gesture.panY + dy });
  }

  function onMediaPointerUp(pointerEvent) {
    if (gestureRef.current.pointerId === pointerEvent.pointerId) {
      gestureRef.current.pointerId = null;
      gestureRef.current.mode = null;
    }
  }

  function onMediaClick() {
    if (gestureRef.current.moved) {
      gestureRef.current.moved = false;
      return;
    }
    if (zoom.scale > 1 || videoActive) return;
    if (clipInfo && !clipError) playEventClip();
  }

  function resetZoom() {
    const reset = { scale: 1, x: 0, y: 0 };
    zoomRef.current = reset;
    setZoom(reset);
  }

  useLayoutEffect(() => {
    resetZoom();
    setMediaSize(incidentTrackingFrameSize(trackingEvent));
    setFullSnapshotRequested(false);
    setDetectionDebug(false);
    setDetectionDebugStats(null);
    setTrackingVisible(false);
    setAnalysisToolsOpen(false);
    setTrackingComparison(null);
    setTrackingComparisonEngine(null);
    setTrackingComparisonLoading(false);
    setTrackingComparisonError("");
    setManualDetection(null);
    setManualError("");
    setManualLoading(false);
  }, [event.id]);

  useEffect(() => {
    if (!trackingComparison || !comparisonPanelRef.current) return undefined;
    const frame = window.requestAnimationFrame(() => {
      comparisonPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [trackingComparison]);

  const mediaStyle = useMemo(() => {
    if (!mediaSize?.width || !mediaSize?.height) return undefined;
    const ratio = mediaSize.width / mediaSize.height;
    return {
      "--snapshot-aspect": `${mediaSize.width} / ${mediaSize.height}`,
      "--event-media-fit-width": `${ratio * 72}vh`,
      "--event-media-mobile-fit-width": `${ratio * 70}dvh`,
      "--event-panel-fit-width": `calc(${ratio * 72}vh + 24px)`,
    };
  }, [mediaSize]);

  async function runManualDetection() {
    if (!Number.isFinite(manualEventId)) {
      setManualError("No event id available");
      return;
    }
    setManualLoading(true);
    setManualError("");
    try {
      const response = await fetch(`/api/events/${manualEventId}/detect?confidence=${safeManualConfidence.toFixed(2)}`, { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Manual detection failed");
      setManualDetection(payload);
      if (onRefresh) onRefresh();
      setVideoActive(false);
      resetZoom();
    } catch (error) {
      setManualError(error.message || "Manual detection failed");
    } finally {
      setManualLoading(false);
    }
  }

  async function runTrackingComparison() {
    if (!Number.isFinite(manualEventId) || trackingComparisonLoading) return;
    setAnalysisToolsOpen(true);
    setTrackingComparisonLoading(true);
    setTrackingComparisonError("");
    setTrackingComparisonEngine(null);
    try {
      const response = await fetch(`/api/events/${manualEventId}/tracking-comparison?duration_seconds=30`, { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Tracking comparison failed");
      setTrackingComparison(payload);
      setTrackingComparisonHistory((current) => ({
        items: payload.comparison
          ? [payload.comparison, ...current.items.filter((item) => Number(item.id) !== Number(payload.comparison.id))].slice(0, 10)
          : current.items,
        summary: payload.evidence_summary || current.summary,
      }));
    } catch (error) {
      setTrackingComparisonError(error.message || "Tracking comparison failed");
    } finally {
      setTrackingComparisonLoading(false);
    }
  }

  async function saveTrackingVerdict(verdict) {
    const comparisonId = Number(trackingComparison?.comparison_id);
    if (!Number.isFinite(comparisonId) || trackingVerdictLoading) return;
    setTrackingVerdictLoading(true);
    setTrackingComparisonError("");
    try {
      const response = await fetch(`/api/tracking-comparisons/${comparisonId}/verdict`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verdict }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not save comparison verdict");
      setTrackingComparison((current) => ({ ...current, verdict }));
      setTrackingComparisonHistory((current) => {
        const comparison = payload.comparison;
        const items = comparison
          ? [comparison, ...current.items.filter((item) => Number(item.id) !== Number(comparison.id))].slice(0, 10)
          : current.items;
        return { items, summary: payload.summary || current.summary };
      });
    } catch (error) {
      setTrackingComparisonError(error.message || "Could not save comparison verdict");
    } finally {
      setTrackingVerdictLoading(false);
    }
  }

  async function replayTrackingComparison(implementation) {
    const engine = trackingComparison?.engines?.[implementation];
    if (!engine || !clipInfo) return;
    const after = Math.max(0.1, Number(trackingComparison.requested_duration_seconds || trackingComparison.duration_seconds || 0));
    const anchorEpoch = eventEpoch(viewerEvent);
    const requestedWindowStartEpoch = Number.isFinite(anchorEpoch) ? anchorEpoch : null;
    const streamUrl = eventStreamUrl(manualEventId, 0, after);
    const timelineStartEpoch = Number.isFinite(requestedWindowStartEpoch)
      ? await eventStreamTimelineStart(streamUrl, requestedWindowStartEpoch)
      : null;
    const nextClip = {
      ...clipInfo,
      before: 0,
      after,
      duration: after,
      streamUrl,
      downloadUrl: eventClipUrl(manualEventId, 0, after),
      windowStartEpoch: timelineStartEpoch,
      requestedWindowStartEpoch,
      initialPlaybackOffset: 0,
      playbackStartOffset: hlsPlaybackOffset(requestedWindowStartEpoch, timelineStartEpoch, 0),
    };
    setTrackingComparisonEngine(implementation);
    setTrackingVisible(true);
    setDetectionDebug(false);
    setClipError("");
    setClipLoading(true);
    setPlaybackOriginTime(null);
    setClipInfo(nextClip);
    setPlayback({
      url: nextClip.streamUrl,
      mimeType: "application/vnd.apple.mpegurl",
      key: `comparison-${implementation}-${Date.now()}`,
    });
    setVideoActive(true);
  }

  useEffect(() => {
    function onKey(keyEvent) {
      if (keyEvent.key === "Escape") {
        onClose();
        return;
      }
      if (keyEvent.key !== "ArrowLeft" && keyEvent.key !== "ArrowRight") return;
      const direction = keyEvent.key === "ArrowRight" ? 1 : -1;
      const nextIncident = adjacentIncident(events, event, direction);
      if (!nextIncident) return;
      keyEvent.preventDefault();
      onSelect(nextIncident);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [event.id, events, onClose, onSelect]);

  return (
    <div className="event-overlay" role="dialog" aria-modal="true" aria-label="Event image">
      <button className="live-overlay-backdrop" type="button" onClick={onClose} aria-label="Close event image" />
      <section className="event-overlay-panel" style={mediaStyle}>
        <div className="event-detail-head">
          <div>
            <h2>{viewerEvent.camera_id}</h2>
            <time>{formatDateTime(viewerEvent.created_at, timeZone)}</time>
          </div>
          <div className="overlay-actions">
            <button
              type="button"
              className="tile-control-button"
              onClick={runTrackingComparison}
              disabled={trackingComparisonLoading || !Number.isFinite(manualEventId)}
              title="Run Hybrid and FastTrack on the same 30-second recording window"
            >
              <Gauge size={16} /> {trackingComparisonLoading ? "Comparing" : "Compare"}
            </button>
            <button
                type="button"
                className={`tile-control-button tracking-trail-toggle ${trackingVisible ? "active" : ""}`}
                onClick={() => setTrackingVisible((visible) => !visible)}
                disabled={!storedTracks.length}
                title={storedTracks.length ? `${trackingVisible ? "Hide" : "Show"} stored object tracking${replayTrackCount ? " on the snapshot and video" : " on the event snapshot"}` : "No stored tracks for this event; run Compare to generate an offline replay"}
                aria-label={storedTracks.length ? `${trackingVisible ? "Hide" : "Show"} stored object tracks` : "Stored object tracks unavailable"}
                aria-pressed={trackingVisible}
              >
                <ListTree size={16} /> Tracks
            </button>
            <button
              type="button"
              className={`tile-control-button debug-detection-toggle ${detectionDebug ? "active" : ""}`}
                onClick={() => {
                  if (!videoActive) playEventClip();
                  setDetectionDebug((enabled) => {
                    if (!enabled) setTrackingVisible(false);
                    return !enabled;
                  });
              }}
              disabled={!clipInfo || Boolean(clipError)}
              title="Toggle real-time OpenVINO detection and tracking"
              aria-pressed={detectionDebug}
            >
              <Activity size={16} /> AI
            </button>
            {clipInfo && !clipError ? (
              <a className="tile-control-button icon-only" href={clipInfo.downloadUrl} download={downloadName} title="Download event video" aria-label="Download event video">
                <Download size={18} />
              </a>
            ) : (
              <span className="tile-control-button icon-only disabled" title={clipError || "Event video not ready"} aria-label="Event video not ready">
                <Download size={18} />
              </span>
            )}
            <button type="button" className="tile-control-button icon-only" onClick={onClose} aria-label="Close event image">
              <X size={18} />
            </button>
          </div>
        </div>
        <div
          ref={mediaRef}
          className={`event-detail-media event-detail-play-target ${zoom.scale > 1 ? "zoomed" : ""}`}
          role="button"
          tabIndex={0}
          onClick={onMediaClick}
          onDoubleClick={resetZoom}
          onWheel={onMediaWheel}
          onTouchStart={onMediaTouchStart}
          onTouchMove={onMediaTouchMove}
          onTouchEnd={onMediaTouchEnd}
          onTouchCancel={onMediaTouchEnd}
          onPointerDown={onMediaPointerDown}
          onPointerMove={onMediaPointerMove}
          onPointerUp={onMediaPointerUp}
          onPointerCancel={onMediaPointerUp}
          onKeyDown={(keyEvent) => {
            if (keyEvent.key === "0") resetZoom();
            if ((keyEvent.key === "Enter" || keyEvent.key === " ") && clipInfo && !clipError && zoom.scale === 1) {
              keyEvent.preventDefault();
              playEventClip();
            }
          }}
          title={zoom.scale > 1 ? "Drag to pan. Double-click to reset zoom." : clipError || (clipLoading ? "Preparing event clip" : "Scroll or pinch to zoom. Click to play event video.")}
        >
          <SnapshotImage
            event={trackingEvent}
            alt="selected event snapshot"
            iconSize={42}
            className="event-snapshot-frame"
            zoom={zoom}
            allowObjectFocus={zoom.scale === 1 && !videoActive}
            progressive
            fullResolution={fullSnapshotRequested}
            highQualityZoom={zoom.scale > 1}
            onRequestFullResolution={() => setFullSnapshotRequested(true)}
            showAnnotations
            showTracking={trackingVisible && !manualDetection}
            onImageSize={setMediaSize}
          />
          {videoActive && clipInfo && playback && !clipError ? (
            <>
              <ShakaVideo
                key={playback.key || playback.url}
                className="event-video-layer"
                ref={clipVideoRef}
                src={playback.url}
                mimeType={playback.mimeType}
                autoPlay
                controls
                playsInline
                preload="metadata"
                onReady={(_player, video) => {
                  setPlaybackOriginTime(0);
                  const playbackStartOffset = Math.max(0, Number(clipInfo.playbackStartOffset) || 0);
                  if (playbackStartOffset > 0 && video) {
                    const seekToSelectedEvent = () => {
                      const targetTime = playbackStartOffset;
                      video.currentTime = Number.isFinite(video.duration)
                        ? Math.min(targetTime, Math.max(0, video.duration - 0.25))
                        : targetTime;
                    };
                    if (video.paused) video.addEventListener("playing", seekToSelectedEvent, { once: true });
                    else seekToSelectedEvent();
                  }
                  setClipLoading(false);
                  setClipError("");
                }}
                onError={() => {
                  if (playback.url !== clipInfo.downloadUrl) {
                    setClipLoading(true);
                    setPlaybackOriginTime(null);
                    setClipInfo((current) => current ? {
                      ...current,
                      windowStartEpoch: current.requestedWindowStartEpoch,
                      playbackStartOffset: current.initialPlaybackOffset,
                    } : current);
                    setPlayback({ url: clipInfo.downloadUrl, mimeType: "video/mp4" });
                  } else {
                    setClipLoading(false);
                    setVideoActive(false);
                    setClipError("No recording window found");
                  }
                }}
                onClick={(event) => event.stopPropagation()}
              />
              <DebugDetectionOverlay
                videoRef={clipVideoRef}
                active={detectionDebug}
                confidence={safeManualConfidence}
                onStats={setDetectionDebugStats}
              />
              {trackingVisible ? (
                <StoredTrackVideoOverlay
                  videoRef={clipVideoRef}
                  tracks={storedTracks}
                  coordinateSize={{
                    width: Number(trackingEvent.object_tracking?.frame_width) || mediaSize?.width,
                    height: Number(trackingEvent.object_tracking?.frame_height) || mediaSize?.height,
                  }}
                  windowStartEpoch={clipInfo.windowStartEpoch}
                  mediaStartTime={playbackOriginTime}
                  mediaKey={playback.key || playback.url}
                  sampleFps={trackingEvent.object_tracking?.sample_fps}
                  lostTimeoutSeconds={trackingEvent.object_tracking?.lost_timeout_seconds}
                />
              ) : null}
              {clipLoading ? <div className="event-video-preparing">Preparing incident video...</div> : null}
            </>
          ) : null}
          {videoActive && detectionDebug && detectionDebugStats ? (
            <div className={`event-detection-stats ${detectionDebugStats.error ? "error" : ""}`}>
              {detectionDebugStats.error
                ? detectionDebugStats.error
                : `${detectionDebugStats.inferenceMs ?? "--"} ms · ${detectionDebugStats.objects ?? 0} objects`}
            </div>
          ) : null}
        </div>
        <div className="event-detail-body">
          <details
            className="event-analysis-details"
            open={analysisToolsOpen}
            onToggle={(toggleEvent) => setAnalysisToolsOpen(toggleEvent.currentTarget.open)}
          >
            <summary>Analysis tools &amp; diagnostics</summary>
            <div className="event-analysis-details-body">
          {storedTracks.length ? (
            <div className="event-track-summary">
              <span className="muted">{trackingComparisonEngine ? "Comparison replay" : "Stored tracking"}</span>
              <strong>{storedTracks.length} track{storedTracks.length === 1 ? "" : "s"} · {String(trackingEvent.object_tracking?.implementation || "tracker").replaceAll("_", " ")} · {Number(trackingEvent.object_tracking?.sample_fps || 0) || "?"} FPS</strong>
              <small>{replayTrackCount ? `${replayTrackCount} track${replayTrackCount === 1 ? "" : "s"} can replay over video. Dashed, faded boxes are estimated positions during the configured lost-object grace period. Snapshot boxes mark each track's last stored position.` : "Paths show sampled object centers over time. The box marks each track's last stored position."}</small>
              {Number(reidDiagnostics.inference_attempts || 0) || Number(reidDiagnostics.reid_avoided_geometry_matches || 0) ? (
                <div className="tracking-comparison-shared">
                  <span>Appearance checks <strong>{Number(reidDiagnostics.inference_attempts || 0)}</strong></span>
                  <span>Checks avoided <strong>{Number(reidDiagnostics.reid_avoided_geometry_matches || 0)}</strong></span>
                  {Object.entries(reidAttemptReasons).map(([reason, count]) => <span key={reason}>{String(reason).replaceAll("_", " ")} <strong>{count}</strong></span>)}
                </div>
              ) : null}
            </div>
          ) : null}
          {trackingComparisonError ? <div className="tracking-comparison-error">{trackingComparisonError}</div> : null}
          {trackingComparison ? (
            <div className="tracking-comparison-panel" ref={comparisonPanelRef}>
              <div className="tracking-comparison-head">
                <div><span className="muted">Same-frame comparison</span><strong>{trackingComparison.frames_processed} frames · {Number(trackingComparison.duration_seconds || 0).toFixed(1)}s · {trackingComparison.sample_fps} FPS · {(Number(trackingComparison.elapsed_ms || 0) / 1000).toFixed(1)}s analysis</strong></div>
                <small>Detection and appearance extraction are shared by both engines. Extra track IDs are a comparison signal, not ground-truth identity accuracy.</small>
              </div>
              <div className="tracking-comparison-shared"><span>Recording decode <strong>{trackingComparison.average_frame_decode_ms} ms/frame</strong></span><span>OpenVINO detection <strong>{trackingComparison.average_detection_ms_per_frame} ms/frame</strong></span>{Number(trackingComparison.appearance_ms || 0) > 0 ? <span>Appearance extraction <strong>{trackingComparison.average_appearance_ms_per_frame} ms/frame</strong></span> : null}{trackingComparison.appearance_failures ? <span>Appearance failures <strong>{trackingComparison.appearance_failures}</strong></span> : null}<span>Clip preparation <strong>{(Number(trackingComparison.clip_preparation_ms || 0) / 1000).toFixed(1)}s</strong></span></div>
              <div className="tracking-comparison-grid">
                {["survng_hybrid", "ultralytics_fasttrack"].map((implementation) => {
                  const engine = trackingComparison.engines?.[implementation];
                  if (!engine) return null;
                  const comparisonEvent = { ...viewerEvent, object_tracking: {
                    ...engine,
                    sample_fps: trackingComparison.sample_fps,
                    frame_width: trackingComparison.frame_width,
                    frame_height: trackingComparison.frame_height,
                  } };
                  return (
                    <article className={`tracking-comparison-card ${trackingComparisonEngine === implementation ? "active" : ""}`} key={implementation}>
                      <header><strong>{implementation === "survng_hybrid" ? "SurvNG Hybrid" : "FastTrack"}</strong><span>{engine.average_ms_per_frame} ms/frame · {engine.initialization_ms} ms init</span></header>
                      <SnapshotImage event={comparisonEvent} alt={`${implementation} tracking result`} allowObjectFocus={false} showAnnotations={false} showTracking />
                      <dl>
                        <div><dt>Tracks</dt><dd>{engine.track_count}</dd></div>
                        <div><dt>Extra track IDs</dt><dd>{engine.fragmentation_proxy}</dd></div>
                        <div><dt>Observations</dt><dd>{engine.observations}</dd></div>
                        <div><dt>ReID recoveries</dt><dd>{engine.reid_recoveries}</dd></div>
                        <div><dt>Geometry matches</dt><dd>{engine.reid_diagnostics?.association_counts?.geometry || 0}</dd></div>
                      </dl>
                      <button type="button" className="tile-control-button" onClick={() => replayTrackingComparison(implementation)}><Play size={15} /> Replay this result</button>
                    </article>
                  );
                })}
              </div>
              <div className="tracking-comparison-verdict">
                <div><span className="muted">Your visual review</span><strong>Which replay kept identities most accurately?</strong></div>
                <div className="tracking-comparison-verdict-actions">
                  {[
                    ["survng_hybrid", "Hybrid looked better"],
                    ["ultralytics_fasttrack", "FastTrack looked better"],
                    ["inconclusive", "No clear winner"],
                  ].map(([verdict, label]) => (
                    <button type="button" className={`tile-control-button ${trackingComparison.verdict === verdict ? "active" : ""}`} disabled={trackingVerdictLoading} onClick={() => saveTrackingVerdict(verdict)} key={verdict}>{label}</button>
                  ))}
                </div>
              </div>
            </div>
          ) : null}
          {trackingComparisonHistory.summary?.total ? (
            <div className="tracking-comparison-history">
              <div className="tracking-comparison-head">
                <div><span className="muted">{event.camera_id} comparison evidence</span><strong>{trackingComparisonHistory.summary.reviewed} reviewed · {trackingComparisonHistory.summary.verdicts?.unreviewed || 0} awaiting review</strong></div>
                <small>SurvNG records your judgment but never changes the configured tracker automatically.</small>
              </div>
              <div className="tracking-comparison-shared">
                <span>Hybrid better <strong>{trackingComparisonHistory.summary.verdicts?.survng_hybrid || 0}</strong></span>
                <span>FastTrack better <strong>{trackingComparisonHistory.summary.verdicts?.ultralytics_fasttrack || 0}</strong></span>
                {trackingComparisonHistory.summary.verdicts?.ultralytics_botsort ? <span>BoT-SORT better (historic) <strong>{trackingComparisonHistory.summary.verdicts.ultralytics_botsort}</strong></span> : null}
                <span>No clear winner <strong>{trackingComparisonHistory.summary.verdicts?.inconclusive || 0}</strong></span>
              </div>
              {trackingComparisonHistory.items.some((item) => item.verdict) ? (
                <div className="tracking-comparison-history-list">
                  {trackingComparisonHistory.items.filter((item) => item.verdict).slice(0, 5).map((item) => (
                    <div key={item.id}>
                      <time>{formatDateTime(item.event_created_at || item.created_at, timeZone)}</time>
                      <strong>{item.verdict === "survng_hybrid" ? "Hybrid" : item.verdict === "ultralytics_fasttrack" ? "FastTrack" : item.verdict === "ultralytics_deepocsort" ? "Deep OC-SORT (historic)" : item.verdict === "ultralytics_botsort" ? "BoT-SORT (historic)" : "No clear winner"}</strong>
                      <span>{item.result?.frames_processed || 0} frames</span>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}
          <div className="manual-detect-panel" onClick={(event) => event.stopPropagation()}>
            <div className="manual-detect-head">
              <div>
                <span className="muted">Manual OpenVINO</span>
                <strong>{Math.round(safeManualConfidence * 100)}% confidence</strong>
              </div>
              <button type="button" className="tile-control-button" onClick={runManualDetection} disabled={manualLoading || !Number.isFinite(manualEventId)}>
                <Search size={15} /> {manualLoading ? "Running" : "Run"}
              </button>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.01"
              value={safeManualConfidence}
              onChange={(event) => setManualConfidence(event.target.value)}
              aria-label="Manual detection confidence"
            />
            {manualError ? <span className="manual-debug error">{manualError}</span> : null}
            {manualDetection ? (
              <div className="manual-debug">
                <span>{manualDetection.object_count} objects</span>
                <span>{manualDetection.elapsed_ms} ms</span>
                <span>{manualDetection.detector?.loaded_backend || "detector"}</span>
                <span>{manualDetection.detector?.loaded_device || manualDetection.detector?.configured_device || "device"}</span>
                <span>{manualDetection.snapshot_width}x{manualDetection.snapshot_height}</span>
                {manualDetection.labels?.length ? <code>{manualDetection.labels.join(", ")}</code> : <code>no labels</code>}
              </div>
            ) : null}
          </div>
            </div>
          </details>
        </div>
      </section>
    </div>
  );
}

function IncidentsPage({ timeZone, onRecordingContextChange, onAssistantContextChange }) {
  const { cameras, appConfig, refresh: refreshBase } = usePollingData();
  const thumbnailAnnotations = appConfig?.incident_thumbnail_annotations ?? false;
  const [eventFilter, setEventFilter] = useState("object");
  const [incidentCameraFilter, setIncidentCameraFilter] = useState("all");
  const [incidentObjectFilter, setIncidentObjectFilter] = useState("all");
  const [incidentZoneFilter, setIncidentZoneFilter] = useState("all");
  const [incidentDensity, setIncidentDensity] = useStoredState("survng.incidentDensity.v1", "compact");
  const today = dateKeyForTimeZone(Date.now(), timeZone);
  const [incidentDay, setIncidentDay] = useState(today);
  const [incidents, setIncidents] = useState([]);
  const [incidentTotal, setIncidentTotal] = useState(0);
  const [incidentFacets, setIncidentFacets] = useState({ camera_ids: [], labels: [], zones: [] });
  const [incidentLoading, setIncidentLoading] = useState(true);
  const [incidentLoadError, setIncidentLoadError] = useState("");
  const [incidentRefreshToken, setIncidentRefreshToken] = useState(0);
  const [semanticIncidentQuery, setSemanticIncidentQuery] = useState("");
  const [semanticIncidentActiveQuery, setSemanticIncidentActiveQuery] = useState("");
  const [semanticIncidentResults, setSemanticIncidentResults] = useState([]);
  const [semanticIncidentLoading, setSemanticIncidentLoading] = useState(false);
  const [semanticIncidentError, setSemanticIncidentError] = useState("");
  const semanticIncidentRequestRef = useRef(null);
  const incidentLoadedQueryRef = useRef("");
  const incidentEventRefreshTimer = useRef(null);
  const {
    incidentDetailCacheRef,
    incidentDetails,
    setIncidentDetails,
    incidentSelectionRequestRef,
    selectedEvent,
    setSelectedEvent,
    openIncidentOverlay,
    closeIncidentOverlay,
  } = useIncidentDetails();
  const [focusedFaceEventId, setFocusedFaceEventId] = useState(null);
  const [linkedIncidentDetail, setLinkedIncidentDetail] = useState(null);
  const [linkedIncidentEventId, setLinkedIncidentEventId] = useState(null);
  const [selectedFace, setSelectedFace] = useState(null);
  const [facePeople, setFacePeople] = useState([]);
  const [expandedIncidentId, setExpandedIncidentId] = useState(null);
  const [incidentPage, setIncidentPage] = useState(0);
  const incidentRailListRef = useRef(null);
  const [incidentRailSize, setIncidentRailSize] = useState({ width: 0, height: 0 });
  const [desktopAnalysisMode, setDesktopAnalysisMode] = useStoredState("survng.incidentDesktopAnalysis.v1", "clean");
  const [desktopAnalysisStats, setDesktopAnalysisStats] = useState(null);
  const [desktopReplayRequest, setDesktopReplayRequest] = useState(0);
  const [focusedImageSize, setFocusedImageSize] = useState(null);
  const [relatedPreviewIncident, setRelatedPreviewIncident] = useState(null);
  const [relatedPreviewEventId, setRelatedPreviewEventId] = useState(null);
  const [relatedPreviewLoadingEventId, setRelatedPreviewLoadingEventId] = useState(null);
  const relatedPreviewRequestRef = useRef(0);
  const mobileView = isMobileViewport();
  const incidentRailReady = mobileView || (incidentRailSize.width > 0 && incidentRailSize.height > 0);
  const incidentsPerPage = mobileView ? 12 : incidentThumbnailPageSize({ ...incidentRailSize, density: incidentDensity });
  const previousIncidentsPerPageRef = useRef(incidentsPerPage);
  const cameraNameById = useMemo(() => new Map(cameras.map((camera) => [camera.id, camera.name || camera.id])), [cameras]);
  const incidentCameraOptions = incidentFacets.camera_ids || [];
  const incidentObjectOptions = incidentFacets.labels || [];
  const incidentZoneOptions = incidentFacets.zones || [];
  const semanticIncidentActive = Boolean(semanticIncidentActiveQuery);
  const incidentResultSource = semanticIncidentActive ? semanticIncidentResults : incidents;
  const displayedIncidentTotal = semanticIncidentActive ? semanticIncidentResults.length : incidentTotal;
  const displayedIncidentLoading = semanticIncidentActive ? semanticIncidentLoading : incidentLoading;
  const displayedIncidentError = semanticIncidentActive ? semanticIncidentError : incidentLoadError;
  const visibleIncidents = semanticIncidentActive
    ? incidentResultSource.slice(incidentPage * incidentsPerPage, (incidentPage + 1) * incidentsPerPage)
    : incidentResultSource;
  const explicitlyFocusedSummary = visibleIncidents.find((incident) => incident.id === expandedIncidentId)
    || (linkedIncidentDetail?.id === expandedIncidentId ? linkedIncidentDetail : null);
  const focusedSummary = mobileView ? explicitlyFocusedSummary : explicitlyFocusedSummary || visibleIncidents[0] || null;
  const focusedDetailQuery = incidentDetailQuery(focusedSummary);
  const focusedIncident = focusedSummary ? incidentDetails[focusedDetailQuery] || focusedSummary : null;
  const focusedEvent = (focusedIncident?.events || []).find((event) => Number(event.id) === Number(focusedFaceEventId))
    || (focusedIncident?.events || []).find((event) => Number(event.id) === Number(focusedIncident.representative_event_id))
    || focusedIncident;
  const relatedAnchorEvent = incidentTrackingSource(focusedEvent, focusedIncident) || focusedEvent;
  const relatedAnchorEventId = Number(relatedAnchorEvent?.representative_event_id || relatedAnchorEvent?.id) || null;
  const displayedIncident = relatedPreviewIncident || focusedIncident;
  const displayedEvent = (displayedIncident?.events || []).find((event) => Number(event.id) === Number(focusedFaceEventId))
    || (displayedIncident?.events || []).find((event) => Number(event.id) === Number(displayedIncident?.representative_event_id))
    || displayedIncident;
  const focusedSnapshotEvent = displayedEvent;
  const focusedSnapshotEventId = Number(focusedSnapshotEvent?.representative_event_id || focusedSnapshotEvent?.id);
  const focusedLoadedImageSize = Number(focusedImageSize?.eventId) === focusedSnapshotEventId ? focusedImageSize : null;
  const galleryIncidents = visibleIncidents;
  const incidentPageCount = Math.max(1, Math.ceil(displayedIncidentTotal / incidentsPerPage));
  const clampedIncidentPage = Math.min(incidentPage, incidentPageCount - 1);
  const pagedIncidents = galleryIncidents;

  useEffect(() => {
    onAssistantContextChange?.({
      page: "incidents",
      camera_id: focusedIncident?.camera_id || (incidentCameraFilter === "all" ? "" : incidentCameraFilter),
      incident_event_id: Number(focusedEvent?.representative_event_id || focusedEvent?.id) || null,
      filters: {
        day: incidentDay,
        event_type: eventFilter,
        camera: incidentCameraFilter,
        object: incidentObjectFilter,
        zone: incidentZoneFilter,
      },
    });
  }, [eventFilter, focusedEvent?.id, focusedEvent?.representative_event_id, focusedIncident?.camera_id, incidentCameraFilter, incidentDay, incidentObjectFilter, incidentZoneFilter, onAssistantContextChange]);

  useEffect(() => {
    const eventIds = new URLSearchParams(window.location.search).get("event_ids");
    if (!eventIds) return;
    const query = new URLSearchParams({ event_ids: eventIds, gap_seconds: "45" });
    fetch(`/api/incidents/detail?${query}`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Linked incident unavailable")))
      .then((detail) => {
        const requestedEventId = Number(String(eventIds).split(",")[0]);
        if (mobileView) {
          setSelectedEvent(detail);
          return;
        }
        setLinkedIncidentDetail(detail);
        setLinkedIncidentEventId(Number.isInteger(requestedEventId) ? requestedEventId : Number(detail.representative_event_id) || null);
        setExpandedIncidentId(detail.id);
        setFocusedFaceEventId(Number.isInteger(requestedEventId) ? requestedEventId : Number(detail.representative_event_id) || null);
        const linkedEpoch = new Date(detail.start_at || detail.created_at || 0).getTime();
        if (Number.isFinite(linkedEpoch) && linkedEpoch > 0) setIncidentDay(dateKeyForTimeZone(linkedEpoch, timeZone));
        setEventFilter(linkedIncidentEventFilter(detail));
      })
      .catch(() => {});
  }, [mobileView, timeZone]);

  useEffect(() => {
    clearLegacyIncidentFilterStorage();
  }, []);

  useEffect(() => {
    if (mobileView) return undefined;
    const rail = incidentRailListRef.current;
    if (!rail) return undefined;
    function updateRailSize() {
      const rect = rail.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) setIncidentRailSize({ width: rect.width, height: rect.height });
    }
    updateRailSize();
    const observer = new ResizeObserver(updateRailSize);
    observer.observe(rail);
    return () => observer.disconnect();
  }, [mobileView]);

  function refresh() {
    refreshBase();
    incidentDetailCacheRef.current.clear();
    setIncidentDetails({});
    setIncidentRefreshToken((value) => value + 1);
  }

  async function runSemanticIncidentSearch(event, requestedQuery = semanticIncidentQuery) {
    event?.preventDefault();
    const queryText = String(requestedQuery || "").trim();
    if (!queryText) return;
    semanticIncidentRequestRef.current?.abort();
    const controller = new AbortController();
    semanticIncidentRequestRef.current = controller;
    setSemanticIncidentQuery(queryText);
    setSemanticIncidentActiveQuery(queryText);
    setSemanticIncidentResults([]);
    setSemanticIncidentLoading(true);
    setSemanticIncidentError("");
    setIncidentPage(0);
    setEventFilter("object");
    try {
      const nextDay = addDaysToDateKey(incidentDay || today, 1);
      const startEpoch = zonedDateSecondToEpoch(incidentDay || today, 0, timeZone);
      const endEpoch = zonedDateSecondToEpoch(nextDay, 0, timeZone);
      const response = await fetch("/api/semantic-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify(semanticIncidentRequest({
          query: queryText,
          cameraFilter: incidentCameraFilter,
          objectFilter: incidentObjectFilter,
          startAt: new Date(startEpoch * 1000).toISOString(),
          endAt: new Date(endEpoch * 1000).toISOString(),
        })),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Semantic incident search failed");
      const hits = Array.isArray(payload.results) ? payload.results : [];
      const hydrated = await mapWithConcurrency(hits, 6, async (hit) => {
        const eventId = Number(hit?.event?.id);
        if (!Number.isInteger(eventId) || eventId <= 0) return null;
        const detailResponse = await fetch(`/api/incidents/by-event/${eventId}`, { signal: controller.signal });
        if (!detailResponse.ok) return null;
        const detail = await detailResponse.json();
        return {
          ...detail,
          semantic_search: {
            query: queryText,
            score: Number(hit.score),
            rank_score: Number(hit.rank_score ?? hit.score),
            match_strength: hit.match_strength || "visual_similarity",
            component_scores: hit.component_scores || null,
            evidence: hit.evidence || null,
            event_id: eventId,
          },
        };
      });
      if (semanticIncidentRequestRef.current !== controller) return;
      setSemanticIncidentResults(rankSemanticIncidentDetails(hydrated, incidentZoneFilter));
    } catch (error) {
      if (error?.name !== "AbortError") setSemanticIncidentError(error.message || "Semantic incident search failed");
    } finally {
      if (semanticIncidentRequestRef.current === controller) {
        semanticIncidentRequestRef.current = null;
        setSemanticIncidentLoading(false);
      }
    }
  }

  function resetSemanticIncidentSearch() {
    semanticIncidentRequestRef.current?.abort();
    semanticIncidentRequestRef.current = null;
    setSemanticIncidentQuery("");
    setSemanticIncidentActiveQuery("");
    setSemanticIncidentResults([]);
    setSemanticIncidentLoading(false);
    setSemanticIncidentError("");
    setIncidentPage(0);
  }

  useEffect(() => () => semanticIncidentRequestRef.current?.abort(), []);

  useEffect(() => {
    if (!semanticIncidentActiveQuery) return;
    runSemanticIncidentSearch(null, semanticIncidentActiveQuery);
  }, [incidentCameraFilter, incidentDay, incidentObjectFilter, incidentZoneFilter]);

  useAppEvents(({ type }) => {
    if (type !== "incident" || incidentDay !== today || incidentPage !== 0 || document.hidden) return;
    if (incidentEventRefreshTimer.current) return;
    incidentEventRefreshTimer.current = window.setTimeout(
      () => {
        incidentEventRefreshTimer.current = null;
        if (focusedDetailQuery) {
          incidentDetailCacheRef.current.invalidate(focusedDetailQuery);
          incidentDetailCacheRef.current.load(focusedDetailQuery).then((detail) => {
            setIncidentDetails((current) => ({ ...current, [focusedDetailQuery]: detail }));
          }).catch(() => {
            // Keep the existing detail visible; the next event or fallback poll retries.
          });
        }
        setIncidentRefreshToken((value) => value + 1);
      },
      1000,
    );
  });

  useEffect(() => () => window.clearTimeout(incidentEventRefreshTimer.current), []);

  useEffect(() => {
    if (incidentDay !== today || incidentPage !== 0) return undefined;
    const timer = window.setInterval(() => {
      if (!document.hidden) setIncidentRefreshToken((value) => value + 1);
    }, INCIDENT_REFRESH_FALLBACK_MS);
    return () => window.clearInterval(timer);
  }, [incidentDay, incidentPage, today]);

  async function openFaceReview(face) {
    const observationId = Number(face?.observation_id);
    if (!Number.isFinite(observationId)) return;
    try {
      const [observationResponse, peopleResponse] = await Promise.all([
        fetch(`/api/faces/observations/${observationId}`),
        fetch("/api/faces/people"),
      ]);
      if (!observationResponse.ok || !peopleResponse.ok) return;
      setFacePeople(await peopleResponse.json());
      setSelectedFace(await observationResponse.json());
    } catch {
      // Leave the current incident visible when face details are unavailable.
    }
  }

  useEffect(() => {
    if (incidentDay > today) setIncidentDay(today);
  }, [incidentDay, setIncidentDay, today]);

  useEffect(() => {
    if (!incidentRailReady) return undefined;
    let cancelled = false;
    async function loadIncidentPage() {
      const query = new URLSearchParams({
        day: incidentDay || today,
        time_zone: timeZone,
        event_type: eventFilter,
        limit: String(incidentsPerPage),
        offset: String(incidentPage * incidentsPerPage),
        gap_seconds: "45",
      });
      if (incidentCameraFilter !== "all") query.set("camera_id", incidentCameraFilter);
      if (incidentObjectFilter !== "all") query.set("object_label", incidentObjectFilter);
      if (incidentZoneFilter !== "all") query.set("zone", incidentZoneFilter);
      const queryKey = query.toString();
      const foregroundLoad = incidentLoadedQueryRef.current !== queryKey;
      if (foregroundLoad) {
        setIncidentLoading(true);
        setIncidentLoadError("");
      }
      try {
        const response = await fetch(`/api/incidents/search?${query}`);
        if (!response.ok) throw new Error("Unable to load incidents");
        const payload = await response.json();
        if (cancelled) return;
        incidentLoadedQueryRef.current = queryKey;
        const items = payload.items || [];
        setIncidents(items);
        setIncidentTotal(Number(payload.total || 0));
        setIncidentFacets(payload.facets || { camera_ids: [], labels: [], zones: [] });
        setIncidentLoadError("");
        if (!mobileView && items.length) {
          setExpandedIncidentId((current) => current || items[0].id);
        }
      } catch (error) {
        if (!cancelled && foregroundLoad) {
          setIncidents([]);
          setIncidentTotal(0);
          setIncidentLoadError(error.message || "Unable to load incidents");
        }
      } finally {
        if (!cancelled && foregroundLoad) setIncidentLoading(false);
      }
    }
    loadIncidentPage();
    return () => {
      cancelled = true;
    };
  }, [incidentDay, today, timeZone, eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, incidentPage, incidentsPerPage, incidentRefreshToken, incidentRailReady]);

  useEffect(() => {
    setIncidentPage(0);
  }, [eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, incidentDay, incidentDensity]);

  useEffect(() => {
    const previousPageSize = previousIncidentsPerPageRef.current;
    if (previousPageSize !== incidentsPerPage) {
      setIncidentPage((page) => Math.floor(page * previousPageSize / incidentsPerPage));
      previousIncidentsPerPageRef.current = incidentsPerPage;
    }
  }, [incidentsPerPage]);

  useEffect(() => {
    if (incidentPage >= incidentPageCount) setIncidentPage(Math.max(0, incidentPageCount - 1));
  }, [incidentPage, incidentPageCount]);

  useEffect(() => {
    setFocusedFaceEventId(
      linkedIncidentDetail?.id === focusedIncident?.id ? linkedIncidentEventId : null,
    );
    setDesktopAnalysisStats(null);
    relatedPreviewRequestRef.current += 1;
    setRelatedPreviewIncident(null);
    setRelatedPreviewEventId(null);
    setRelatedPreviewLoadingEventId(null);
  }, [focusedIncident?.id, linkedIncidentDetail?.id, linkedIncidentEventId]);

  useEffect(() => {
    if (!focusedSummary || !focusedDetailQuery || incidentDetails[focusedDetailQuery]) return;
    let cancelled = false;
    incidentDetailCacheRef.current.load(focusedDetailQuery).then((detail) => {
      if (!cancelled) setIncidentDetails((current) => ({ ...current, [focusedDetailQuery]: detail }));
    }).catch(() => {
      // The compact incident remains usable if investigation details fail.
    });
    return () => { cancelled = true; };
  }, [focusedSummary?.id, focusedDetailQuery, incidentDetails]);

  useEffect(() => {
    if (desktopAnalysisMode !== "tracks") return;
    const trackingEvent = incidentTrackingSource(displayedEvent, displayedIncident);
    if (!storedObjectTracks(trackingEvent).length) setDesktopAnalysisMode("clean");
  }, [desktopAnalysisMode, displayedEvent, displayedIncident, setDesktopAnalysisMode]);

  useEffect(() => {
    const context = incidentRecordingContext(selectedEvent || focusedEvent);
    if (context) onRecordingContextChange(context);
  }, [selectedEvent?.id, focusedEvent?.id, focusedEvent?.created_at, focusedEvent?.camera_id, onRecordingContextChange]);

  useEffect(() => {
    if (expandedIncidentId
      && linkedIncidentDetail?.id !== expandedIncidentId
      && !visibleIncidents.some((incident) => incident.id === expandedIncidentId)) {
      setExpandedIncidentId(null);
    }
  }, [expandedIncidentId, linkedIncidentDetail?.id, visibleIncidents]);

  useEffect(() => {
    function onKey(keyEvent) {
      if (keyEvent.key === "Escape" && expandedIncidentId && !selectedEvent) {
        keyEvent.preventDefault();
        setExpandedIncidentId(null);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expandedIncidentId, selectedEvent]);

  function toggleIncident(incidentId) {
    if (mobileView) {
      const incident = visibleIncidents.find((candidate) => candidate.id === incidentId);
      if (incident) openIncidentOverlay(incident);
      return;
    }
    if (incidentId !== linkedIncidentDetail?.id) {
      setLinkedIncidentDetail(null);
      setLinkedIncidentEventId(null);
    }
    setExpandedIncidentId(incidentId);
  }

  const focusedIndex = focusedIncident ? visibleIncidents.findIndex((incident) => incident.id === focusedIncident.id) : -1;

  function moveFocus(direction) {
    if (!visibleIncidents.length) return;
    const nextIndex = Math.max(0, Math.min(visibleIncidents.length - 1, focusedIndex + direction));
    setExpandedIncidentId(visibleIncidents[nextIndex].id);
  }

  function selectDesktopAnalysisMode(mode) {
    setDesktopAnalysisMode(mode);
    setDesktopAnalysisStats(null);
    setDesktopReplayRequest((request) => request + 1);
  }

  async function selectRelatedIncident(match) {
    const eventId = Number(match?.event_id);
    if (!Number.isInteger(eventId) || eventId <= 0) return;
    const request = ++relatedPreviewRequestRef.current;
    setRelatedPreviewLoadingEventId(eventId);
    try {
      const response = await fetch(`/api/incidents/by-event/${eventId}`);
      if (!response.ok) throw new Error("Related incident unavailable");
      const detail = await response.json();
      if (request !== relatedPreviewRequestRef.current) return;
      setRelatedPreviewIncident(detail);
      setRelatedPreviewEventId(eventId);
      setFocusedFaceEventId(eventId);
      setDesktopAnalysisMode("clean");
      setDesktopAnalysisStats(null);
    } catch {
      // Keep the currently displayed incident if a stale related event was removed.
    } finally {
      if (request === relatedPreviewRequestRef.current) setRelatedPreviewLoadingEventId(null);
    }
  }

  function returnToSelectedIncident() {
    relatedPreviewRequestRef.current += 1;
    setRelatedPreviewIncident(null);
    setRelatedPreviewEventId(null);
    setRelatedPreviewLoadingEventId(null);
    setFocusedFaceEventId(Number(focusedIncident?.representative_event_id || focusedIncident?.id) || null);
    setDesktopAnalysisMode("clean");
    setDesktopAnalysisStats(null);
  }

  useEffect(() => {
    if (mobileView || selectedEvent) return undefined;
    function onIncidentArrow(event) {
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName))) return;
      if (event.key === "ArrowLeft" && focusedIndex > 0) {
        event.preventDefault();
        moveFocus(-1);
      }
      if (event.key === "ArrowRight" && focusedIndex >= 0 && focusedIndex < visibleIncidents.length - 1) {
        event.preventDefault();
        moveFocus(1);
      }
    }
    window.addEventListener("keydown", onIncidentArrow);
    return () => window.removeEventListener("keydown", onIncidentArrow);
  }, [mobileView, selectedEvent, focusedIndex, visibleIncidents]);

  const semanticIncidentControl = (
    <div className={`incident-semantic-search ${semanticIncidentActive ? "active" : ""} ${semanticIncidentError ? "error" : ""}`}>
      <form onSubmit={runSemanticIncidentSearch} role="search" aria-label="Semantic incident search">
        <Search size={15} aria-hidden="true" />
        <input value={semanticIncidentQuery} onChange={(event) => setSemanticIncidentQuery(event.target.value)} placeholder='Describe an incident, e.g. “white delivery truck”' aria-label="Describe incidents to find" disabled={semanticIncidentLoading} />
        {semanticIncidentActive ? <button type="button" className="secondary" onClick={resetSemanticIncidentSearch} aria-label="Reset semantic incident search">Reset</button> : null}
        <button type="submit" disabled={semanticIncidentLoading || !semanticIncidentQuery.trim()}>{semanticIncidentLoading ? "Searching…" : "Search"}</button>
      </form>
      {semanticIncidentError ? <small>{semanticIncidentError}</small> : semanticIncidentActive ? <small>Showing visual matches for “{semanticIncidentActiveQuery}” using the selected day and filters.</small> : null}
    </div>
  );

  if (!mobileView) {
    return (
      <main className="incidents-desktop-page with-inspector">
        <section className="bento-card incidents-desktop-shell">
          <div className="incidents-desktop-toolbar">
            <div className="incident-filter-toggle compact" aria-label="Incident type filter">
              <button className={eventFilter === "object" ? "active" : ""} onClick={() => setEventFilter("object")}>Object</button>
              <button className={eventFilter === "motion" ? "active" : ""} onClick={() => { resetSemanticIncidentSearch(); setEventFilter("motion"); }}>Motion</button>
            </div>
            <div className="incident-filter-selects desktop">
              <label className="incident-day-field"><span>Day</span><input type="date" value={incidentDay} max={today} onChange={(event) => setIncidentDay(event.target.value || today)} aria-label="Incident day" /></label>
              <label><span>Camera</span><select value={incidentCameraFilter} onChange={(event) => setIncidentCameraFilter(event.target.value)}><option value="all">All cameras</option>{incidentCameraOptions.map((id) => <option value={id} key={id}>{cameraNameById.get(id) || id}</option>)}</select></label>
              <label><span>Object</span><select value={incidentObjectFilter} onChange={(event) => setIncidentObjectFilter(event.target.value)}><option value="all">All objects</option>{incidentObjectOptions.map((label) => <option value={label} key={label}>{label}</option>)}</select></label>
              <label><span>Zone</span><select value={incidentZoneFilter} onChange={(event) => setIncidentZoneFilter(event.target.value)}><option value="all">All zones</option>{incidentZoneOptions.map((zone) => <option value={zone} key={zone}>{zone}</option>)}</select></label>
            </div>
            {semanticIncidentControl}
            <span className="shown-bubble">{displayedIncidentTotal} {semanticIncidentActive ? "matches" : "shown"}</span>
          </div>

          <div className="incidents-desktop-workspace">
            <aside className={`incident-rail ${incidentDensity}`}>
              <div className="incident-rail-head">
                <strong>Incidents</strong>
                <div className="density-control" aria-label="Thumbnail density">
                  <button type="button" className={incidentDensity === "compact" ? "active" : ""} onClick={() => setIncidentDensity("compact")} title="Compact thumbnails" aria-label="Compact thumbnails"><Grid2X2 size={15} /></button>
                  <button type="button" className={incidentDensity === "comfortable" ? "active" : ""} onClick={() => setIncidentDensity("comfortable")} title="Comfortable thumbnails" aria-label="Comfortable thumbnails"><Rows3 size={15} /></button>
                </div>
              </div>
              <div className="incident-rail-list" ref={incidentRailListRef}>
                {displayedIncidentLoading && !galleryIncidents.length ? <div className="empty-state">{semanticIncidentActive ? "Searching indexed incidents..." : "Loading incidents..."}</div> : null}
                {!galleryIncidents.length && displayedIncidentError ? <div className="empty-state">{displayedIncidentError}</div> : null}
                {galleryIncidents.length ? pagedIncidents.map((incident) => (
                  <IncidentCard key={incident.id} incident={incident} timeZone={timeZone} expanded={false} selected={incident.id === focusedIncident?.id} thumbnailAnnotations={thumbnailAnnotations} desktopWorkspace onToggle={toggleIncident} />
                )) : null}
                {!displayedIncidentLoading && !displayedIncidentError && !galleryIncidents.length ? <div className="empty-state">{semanticIncidentActive ? "No semantic matches for the selected filters." : "No other incidents."}</div> : null}
              </div>
              <div className={`incident-pager ${displayedIncidentTotal > incidentsPerPage ? "" : "placeholder"}`} aria-label="Incident pages" aria-hidden={displayedIncidentTotal <= incidentsPerPage}>
                <button type="button" onClick={() => setIncidentPage((page) => Math.max(0, page - 1))} disabled={clampedIncidentPage === 0}>Prev</button>
                <span>{clampedIncidentPage + 1} / {incidentPageCount}</span>
                <button type="button" onClick={() => setIncidentPage((page) => Math.min(incidentPageCount - 1, page + 1))} disabled={clampedIncidentPage >= incidentPageCount - 1}>Next</button>
              </div>
            </aside>

            <section className="incident-investigation">
              <div className="incident-focus-nav">
                <span>{focusedIndex >= 0 ? `${incidentPage * incidentsPerPage + focusedIndex + 1} of ${displayedIncidentTotal}` : "No incident selected"}</span>
                {relatedPreviewIncident ? <button type="button" onClick={returnToSelectedIncident}>Viewing related appearance · return to selected incident</button> : null}
              </div>
              <div className="incident-desktop-focus">
                {focusedIncident ? (
                  <>
                    <button type="button" className="incident-focus-arrow previous" onClick={() => moveFocus(-1)} disabled={focusedIndex <= 0} title="Previous incident" aria-label="Previous incident"><ChevronLeft size={26} /></button>
                    <button type="button" className="incident-focus-arrow next" onClick={() => moveFocus(1)} disabled={focusedIndex < 0 || focusedIndex >= visibleIncidents.length - 1} title="Next incident" aria-label="Next incident"><ChevronRight size={26} /></button>
                  </>
                ) : null}
                {displayedIncident ? <IncidentCard key={`${focusedIncident?.id || "none"}:${displayedIncident.id || displayedIncident.representative_event_id}`} incident={displayedIncident} timeZone={timeZone} expanded thumbnailAnnotations={thumbnailAnnotations} desktopWorkspace analysisMode={desktopAnalysisMode} replayRequest={desktopReplayRequest} onAnalysisStats={setDesktopAnalysisStats} onToggle={toggleIncident} onPreviewChange={setFocusedFaceEventId} onImageSize={setFocusedImageSize} /> : <div className="empty-state">No incidents match the current filters.</div>}
              </div>
            </section>

            <IncidentInspector incident={displayedIncident} faceEvent={displayedEvent} anchorEventId={relatedAnchorEventId} selectedRelatedEventId={relatedPreviewEventId} relatedLoadingEventId={relatedPreviewLoadingEventId} cameraNameById={cameraNameById} appConfig={appConfig} timeZone={timeZone} imageSize={focusedLoadedImageSize} analysisMode={desktopAnalysisMode} analysisStats={desktopAnalysisStats} onAnalysisModeChange={selectDesktopAnalysisMode} onFaceOpen={openFaceReview} onRelatedSelect={selectRelatedIncident} onRelatedReturn={returnToSelectedIncident} />
          </div>
        </section>
        {selectedFace ? <FaceReviewDialog observation={selectedFace} people={facePeople} timeZone={timeZone} onClose={() => setSelectedFace(null)} onUpdated={() => { setSelectedFace(null); refresh(); }} /> : null}
      </main>
    );
  }

  return (
    <main className="bento-grid incidents-grid">
      <section className="bento-card events-zone incidents-page-zone">
        <div className="section-head compact incident-head">
          <div><h2>Incidents</h2></div>
          <div className="incident-head-actions">
            <div className="incident-filter-toggle compact" aria-label="Incident type filter">
              <button className={eventFilter === "object" ? "active" : ""} onClick={() => setEventFilter("object")}>Object</button>
              <button className={eventFilter === "motion" ? "active" : ""} onClick={() => { resetSemanticIncidentSearch(); setEventFilter("motion"); }}>Motion</button>
            </div>
            <span className="shown-bubble">{displayedIncidentTotal} {semanticIncidentActive ? "matches" : "shown"}</span>
          </div>
        </div>
        <div className="event-filter incident-filter-panel" aria-label="Incident filters">
          <div className="incident-filter-selects">
            <label>
              <span>Day</span>
              <input type="date" value={incidentDay} max={today} onChange={(event) => setIncidentDay(event.target.value || today)} aria-label="Incident day" />
            </label>
            <label>
              <span>Camera</span>
              <select value={incidentCameraFilter} onChange={(event) => setIncidentCameraFilter(event.target.value)}>
                <option value="all">All cameras</option>
                {incidentCameraOptions.map((id) => <option value={id} key={id}>{cameraNameById.get(id) || id}</option>)}
              </select>
            </label>
            <label>
              <span>Object</span>
              <select value={incidentObjectFilter} onChange={(event) => setIncidentObjectFilter(event.target.value)}>
                <option value="all">All objects</option>
                {incidentObjectOptions.map((label) => <option value={label} key={label}>{label}</option>)}
              </select>
            </label>
            <label>
              <span>Zone</span>
              <select value={incidentZoneFilter} onChange={(event) => setIncidentZoneFilter(event.target.value)}>
                <option value="all">All zones</option>
                {incidentZoneOptions.map((zone) => <option value={zone} key={zone}>{zone}</option>)}
              </select>
            </label>
          </div>
          {semanticIncidentControl}
        </div>
        <div className="incident-gallery">
          {displayedIncidentLoading ? <div className="empty-state">{semanticIncidentActive ? "Searching indexed incidents..." : "Loading incidents..."}</div> : null}
          {!displayedIncidentLoading && displayedIncidentError ? <div className="empty-state">{displayedIncidentError}</div> : null}
          {!displayedIncidentLoading && !displayedIncidentError && visibleIncidents.length
            ? pagedIncidents.map((incident) => (
              <IncidentCard
                key={incident.id}
                incident={incident}
                timeZone={timeZone}
                expanded={false}
                thumbnailAnnotations={thumbnailAnnotations}
                onToggle={toggleIncident}
                onSelect={openIncidentOverlay}
              />
            ))
            : null}
          {!displayedIncidentLoading && !displayedIncidentError && !visibleIncidents.length ? <div className="empty-state">{semanticIncidentActive ? "No semantic matches for the selected filters." : "No incidents match the current filters."}</div> : null}
        </div>
        {displayedIncidentTotal > incidentsPerPage ? (
          <div className="incident-pager" aria-label="Incident pages">
            <button type="button" onClick={() => setIncidentPage((page) => Math.max(0, page - 1))} disabled={clampedIncidentPage === 0}>Prev</button>
            <span>{clampedIncidentPage + 1} / {incidentPageCount}</span>
            <button type="button" onClick={() => setIncidentPage((page) => Math.min(incidentPageCount - 1, page + 1))} disabled={clampedIncidentPage >= incidentPageCount - 1}>Next</button>
          </div>
        ) : null}
      </section>
      {selectedEvent ? <EventOverlay event={selectedEvent} events={visibleIncidents} timeZone={timeZone} onClose={closeIncidentOverlay} onSelect={openIncidentOverlay} onRefresh={refresh} /> : null}
    </main>
  );
}

function LivePage({ timeZone, onRecordingContextChange, onAssistantContextChange }) {
  const { cameras, appConfig, refresh: refreshBase } = usePollingData();
  const thumbnailAnnotations = appConfig?.incident_thumbnail_annotations ?? false;
  const [eventFilter, setEventFilter] = useState("object");
  const [incidentCameraFilter, setIncidentCameraFilter] = useState("all");
  const [incidentObjectFilter, setIncidentObjectFilter] = useState("all");
  const [incidentZoneFilter, setIncidentZoneFilter] = useState("all");
  const [cameraOrder] = useStoredState("survng.liveCameraOrder.v1", "[]");
  const [liveLayoutMode, setLiveLayoutMode] = useStoredState("survng.liveLayoutMode.v1", "auto");
  const [customLayoutValue, setCustomLayoutValue] = useStoredState("survng.liveCustomLayout.v1", "{}");
  const [customSizePreview, setCustomSizePreview] = useState({});
  const [resizingCameraId, setResizingCameraId] = useState("");
  const customMoveCleanupRef = useRef(null);
  const liveCameraGridRef = useRef(null);
  const [liveCameraGridSize, setLiveCameraGridSize] = useState({ width: 0, height: 0 });
  const [liveCameraAspects, setLiveCameraAspects] = useState({});
  const {
    incidentDetailCacheRef,
    incidentDetails,
    setIncidentDetails,
    incidentSelectionRequestRef,
    selectedEvent,
    setSelectedEvent,
    openIncidentOverlay,
    closeIncidentOverlay,
  } = useIncidentDetails();
  const [expandedIncidentId, setExpandedIncidentId] = useState(null);
  const [expandedCamera, setExpandedCamera] = useState(null);
  const [liveDefaultsReady, setLiveDefaultsReady] = useState(false);
  const [liveDefaultsInstance, setLiveDefaultsInstance] = useState("");
  const linkedCameraIdRef = useRef(new URLSearchParams(window.location.search).get("camera") || "");
  const [incidentPage, setIncidentPage] = useState(0);
  const [incidents, setIncidents] = useState([]);
  const [incidentFacets, setIncidentFacets] = useState({ camera_ids: [], labels: [], zones: [] });
  const [incidentHasMore, setIncidentHasMore] = useState(false);
  const [incidentLoading, setIncidentLoading] = useState(true);
  const [incidentLoadError, setIncidentLoadError] = useState("");
  const [incidentRefreshToken, setIncidentRefreshToken] = useState(0);
  const incidentLoadedQueryRef = useRef("");
  const incidentEventRefreshTimer = useRef(null);
  const incidentFeedCacheRef = useRef(null);
  if (!incidentFeedCacheRef.current) {
    incidentFeedCacheRef.current = createIncidentPageCache(async (query) => {
      const response = await fetch(`/api/incidents/feed?${query}`);
      if (!response.ok) throw new Error("Unable to load recent incidents");
      return response.json();
    });
  }
  const liveIncidentGalleryRef = useRef(null);
  const liveIncidentZoneRef = useRef(null);
  const [liveIncidentGallerySize, setLiveIncidentGallerySize] = useState({ width: 0, height: 0 });
  const liveIncidentGalleryReady = liveIncidentGallerySize.width > 0 && liveIncidentGallerySize.height > 0;
  const incidentsPerPage = liveIncidentGalleryReady
    ? incidentThumbnailPageSize({ ...liveIncidentGallerySize, density: "compact", columns: 2, gap: 10, horizontalPadding: 24 })
    : 12;
  const orderedCameras = useMemo(() => {
    let order = [];
    try {
      order = JSON.parse(cameraOrder);
    } catch {
      order = [];
    }
    const cameraById = new Map(cameras.map((camera) => [camera.id, camera]));
    const sorted = order.map((id) => cameraById.get(id)).filter(Boolean);
    const seen = new Set(sorted.map((camera) => camera.id));
    return [...sorted, ...cameras.filter((camera) => !seen.has(camera.id))];
  }, [cameras, cameraOrder]);
  const normalizedLayoutMode = liveLayoutMode === "custom" ? "custom" : "auto";
  const customLayout = useMemo(
    () => readLiveCustomLayout(customLayoutValue, cameras, liveCameraAspects),
    [cameras, customLayoutValue, liveCameraAspects],
  );
  const displayedCameras = orderedCameras;
  const liveCameraLayout = useMemo(
    () => recordingGridLayout(
      orderedCameras,
      "live",
      liveCameraGridSize.width,
      liveCameraGridSize.height,
      8,
      liveCameraAspects,
      { portraitPriority: true, portraitRowSpan: 2 },
    ),
    [liveCameraAspects, liveCameraGridSize.height, liveCameraGridSize.width, orderedCameras],
  );
  const liveCameraLayoutById = useMemo(
    () => new Map(liveCameraLayout.map((item) => [item.camera.id, item])),
    [liveCameraLayout],
  );
  const liveCameraLayoutReady = liveCameraLayout.length === orderedCameras.length && orderedCameras.length > 0;
  const customGridMetrics = useMemo(
    () => liveCustomGridMetrics(liveCameraGridSize.width, liveCameraGridSize.height),
    [liveCameraGridSize.height, liveCameraGridSize.width],
  );
  const updateLiveCameraAspect = React.useCallback((cameraId, aspect) => {
    const normalized = Number(aspect);
    if (!(normalized > 0)) return;
    setLiveCameraAspects((current) => (
      Math.abs(Number(current[cameraId]) - normalized) < 0.0001
        ? current
        : { ...current, [cameraId]: normalized }
    ));
  }, []);

  useLayoutEffect(() => {
    const grid = liveCameraGridRef.current;
    if (!grid) return undefined;
    const update = () => {
      const next = {
        width: Math.max(0, grid.clientWidth),
        height: Math.max(0, grid.clientHeight),
      };
      setLiveCameraGridSize((current) => (
        current.width === next.width && current.height === next.height ? current : next
      ));
    };
    update();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", update);
      return () => window.removeEventListener("resize", update);
    }
    const observer = new ResizeObserver(update);
    observer.observe(grid);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    let cancelled = false;
    async function syncLiveDefaults() {
      try {
        const response = await fetch("/api/system/status", { cache: "no-store" });
        if (!response.ok) return;
        const payload = await response.json();
        const instanceId = String(payload.instance_id || "");
        if (!instanceId || cancelled) return;
        const reset = resetLiveDefaultsForServer(browserStorage(window), instanceId);
        if (cancelled) return;
        if (reset) setExpandedCamera(null);
        setLiveDefaultsInstance(instanceId);
      } catch {
        // A reconnecting server is expected to be temporarily unavailable.
      } finally {
        if (!cancelled) setLiveDefaultsReady(true);
      }
    }
    void syncLiveDefaults();
    const timer = window.setInterval(syncLiveDefaults, 15_000);
    window.addEventListener("focus", syncLiveDefaults);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      window.removeEventListener("focus", syncLiveDefaults);
    };
  }, []);
  useEffect(() => {
    const linkedCameraId = linkedCameraIdRef.current;
    if (!linkedCameraId || !cameras.length) return;
    const linkedCamera = cameras.find((camera) => camera.id === linkedCameraId);
    if (linkedCamera) setExpandedCamera(linkedCamera);
    linkedCameraIdRef.current = "";
  }, [cameras]);
  const cameraNameById = useMemo(() => new Map(cameras.map((camera) => [camera.id, camera.name || camera.id])), [cameras]);
  const incidentCameraOptions = incidentFacets.camera_ids || [];
  const incidentObjectOptions = incidentFacets.labels || [];
  const incidentZoneOptions = incidentFacets.zones || [];
  const visibleIncidents = incidents;
  const [retainedFocusedIncident, setRetainedFocusedIncident] = useState(null);
  const listedFocusedIncident = retainFocusedIncident(visibleIncidents, expandedIncidentId);
  const focusedSummary = listedFocusedIncident
    || retainFocusedIncident([], expandedIncidentId, retainedFocusedIncident);
  const focusedDetailQuery = incidentDetailQuery(focusedSummary);
  const focusedIncident = focusedSummary ? incidentDetails[focusedDetailQuery] || focusedSummary : null;
  const galleryIncidents = visibleIncidents;
  const incidentPageCount = incidentPage + (incidentHasMore ? 2 : 1);
  const clampedIncidentPage = incidentPage;
  const pagedIncidents = galleryIncidents;

  useEffect(() => {
    const contextualIncident = selectedEvent || focusedIncident;
    onAssistantContextChange?.({
      page: "live",
      camera_id: expandedCamera?.id || contextualIncident?.camera_id || (incidentCameraFilter === "all" ? "" : incidentCameraFilter),
      incident_event_id: Number(contextualIncident?.representative_event_id || contextualIncident?.id) || null,
      filters: {
        event_type: eventFilter,
        camera: incidentCameraFilter,
        object: incidentObjectFilter,
        zone: incidentZoneFilter,
      },
    });
  }, [eventFilter, expandedCamera?.id, focusedIncident?.camera_id, focusedIncident?.id, focusedIncident?.representative_event_id, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, onAssistantContextChange, selectedEvent?.camera_id, selectedEvent?.id, selectedEvent?.representative_event_id]);

  useEffect(() => {
    clearLegacyIncidentFilterStorage();
  }, []);

  useEffect(() => {
    const zone = liveIncidentZoneRef.current;
    const gallery = liveIncidentGalleryRef.current;
    if (!zone || !gallery) return undefined;
    function updateGallerySize() {
      const zoneRect = zone.getBoundingClientRect();
      const rect = gallery.getBoundingClientRect();
      const fixedHeight = [...zone.children]
        .filter((child) => child !== gallery && !child.classList.contains("incident-focus"))
        .reduce((total, child) => total + child.getBoundingClientRect().height, 0);
      const availableHeight = Math.max(0, zoneRect.height - fixedHeight);
      if (rect.width <= 0 || availableHeight <= 0) return;
      const next = {
        width: Math.round(rect.width),
        height: Math.round(availableHeight),
      };
      setLiveIncidentGallerySize((current) => (
        current.width === next.width && current.height === next.height ? current : next
      ));
    }
    updateGallerySize();
    const observer = new ResizeObserver(updateGallerySize);
    observer.observe(zone);
    observer.observe(gallery);
    return () => observer.disconnect();
  }, []);

  function saveCustomLayout(order, sizes) {
    setCustomLayoutValue(JSON.stringify({ version: 1, order, sizes }));
  }

  function beginCustomMove(event, cameraId) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    customMoveCleanupRef.current?.(false);
    const source = String(cameraId);
    const originalOrder = [...customLayout.order];
    let latestOrder = originalOrder;
    let changed = false;
    let activeTarget = null;
    let animationFrame = 0;
    let pendingPoint = null;
    const sourceTile = event.currentTarget.closest(".camera-tile");
    const sourceRect = sourceTile?.getBoundingClientRect();
    const grid = liveCameraGridRef.current;
    const startScrollLeft = grid?.scrollLeft || 0;
    const startScrollTop = grid?.scrollTop || 0;
    const tiles = [...(grid?.querySelectorAll(".camera-tile[data-camera-id]") || [])];
    const tilesById = new Map(tiles.map((tile) => [String(tile.dataset.cameraId), tile]));
    const dragSlots = tiles.map((tile) => {
      const rect = tile.getBoundingClientRect();
      return { id: String(tile.dataset.cameraId), left: rect.left, top: rect.top, width: rect.width, height: rect.height };
    });
    const offsetX = sourceRect ? event.clientX - sourceRect.left : 24;
    const offsetY = sourceRect ? event.clientY - sourceRect.top : 20;
    let ghost = null;
    if (sourceTile && sourceRect) {
      ghost = document.createElement("div");
      ghost.className = "camera-drag-ghost";
      ghost.setAttribute("aria-hidden", "true");
      const poster = sourceTile.querySelector(".camera-tile-poster")?.cloneNode(true);
      if (poster) ghost.appendChild(poster);
      const label = document.createElement("strong");
      label.textContent = sourceTile.querySelector(".tile-header h2")?.textContent || source;
      ghost.appendChild(label);
      ghost.style.width = `${sourceRect.width}px`;
      ghost.style.height = `${sourceRect.height}px`;
      ghost.style.transform = `translate3d(${event.clientX - offsetX}px, ${event.clientY - offsetY}px, 0)`;
      document.body.appendChild(ghost);
    }
    const applyOrder = (order) => order.forEach((id, index) => {
      const tile = tilesById.get(String(id));
      if (tile) tile.style.order = String(index);
    });
    const markTarget = (targetId) => {
      activeTarget?.classList.remove("drag-over");
      activeTarget = targetId && targetId !== source ? tilesById.get(String(targetId)) : null;
      activeTarget?.classList.add("drag-over");
    };
    sourceTile?.classList.add("dragging");
    document.documentElement.classList.add("camera-layout-dragging");
    event.currentTarget.setPointerCapture?.(event.pointerId);

    const renderPoint = () => {
      animationFrame = 0;
      const point = pendingPoint;
      if (!point) return;
      pendingPoint = null;
      if (ghost) ghost.style.transform = `translate3d(${point.x - offsetX}px, ${point.y - offsetY}px, 0)`;
      const gridRect = grid?.getBoundingClientRect();
      if (!gridRect || point.x < gridRect.left || point.x > gridRect.right || point.y < gridRect.top || point.y > gridRect.bottom) {
        markTarget("");
        return;
      }
      const scrollOffsetX = startScrollLeft - (grid?.scrollLeft || 0);
      const scrollOffsetY = startScrollTop - (grid?.scrollTop || 0);
      const target = liveCustomDropTarget(
        dragSlots.map((slot) => ({ ...slot, left: slot.left + scrollOffsetX, top: slot.top + scrollOffsetY })),
        point.x,
        point.y,
        source,
      );
      if (!target) {
        markTarget("");
        return;
      }
      const next = target.position === "original"
        ? originalOrder
        : moveLiveCamera(originalOrder, source, target.targetId, target.position);
      if (next.some((id, index) => id !== latestOrder[index])) {
        latestOrder = next;
        changed = latestOrder.some((id, index) => id !== originalOrder[index]);
        applyOrder(latestOrder);
      }
      markTarget(target.targetId);
    };

    const onMove = (moveEvent) => {
      moveEvent.preventDefault();
      pendingPoint = { x: moveEvent.clientX, y: moveEvent.clientY };
      if (!animationFrame) animationFrame = window.requestAnimationFrame(renderPoint);
    };

    let finished = false;
    const finish = (commit) => {
      if (finished) return;
      finished = true;
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerCancel);
      window.removeEventListener("blur", onPointerCancel);
      ghost?.remove();
      activeTarget?.classList.remove("drag-over");
      sourceTile?.classList.remove("dragging");
      document.documentElement.classList.remove("camera-layout-dragging");
      if (commit && changed) saveCustomLayout(latestOrder, customLayout.sizes);
      else applyOrder(originalOrder);
      if (customMoveCleanupRef.current === finish) customMoveCleanupRef.current = null;
    };
    const onPointerUp = (pointerEvent) => {
      pendingPoint = { x: pointerEvent.clientX, y: pointerEvent.clientY };
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
      renderPoint();
      finish(true);
    };
    const onPointerCancel = () => finish(false);
    window.addEventListener("pointermove", onMove, { passive: false });
    window.addEventListener("pointerup", onPointerUp, { once: true });
    window.addEventListener("pointercancel", onPointerCancel, { once: true });
    window.addEventListener("blur", onPointerCancel, { once: true });
    customMoveCleanupRef.current = finish;
  }

  function beginCustomResize(event, cameraId) {
    event.preventDefault();
    event.stopPropagation();
    const start = customSizePreview[cameraId] || customLayout.sizes[cameraId];
    const startX = event.clientX;
    const startY = event.clientY;
    let latest = start;
    setResizingCameraId(cameraId);
    document.documentElement.classList.add("camera-layout-resizing");
    event.currentTarget.setPointerCapture?.(event.pointerId);
    const onMove = (moveEvent) => {
      latest = resizeLiveCameraToAspect(
        start,
        moveEvent.clientX - startX,
        moveEvent.clientY - startY,
        customGridMetrics,
        liveCameraAspects[cameraId],
      );
      setCustomSizePreview((current) => ({ ...current, [cameraId]: latest }));
    };
    const onEnd = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onEnd);
      window.removeEventListener("pointercancel", onEnd);
      window.removeEventListener("blur", onEnd);
      const sizes = { ...customLayout.sizes, ...customSizePreview, [cameraId]: latest };
      saveCustomLayout(customLayout.order, sizes);
      setCustomSizePreview({});
      setResizingCameraId("");
      document.documentElement.classList.remove("camera-layout-resizing");
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onEnd, { once: true });
    window.addEventListener("pointercancel", onEnd, { once: true });
    window.addEventListener("blur", onEnd, { once: true });
  }

  useEffect(() => () => {
    customMoveCleanupRef.current?.(false);
    document.documentElement.classList.remove("camera-layout-resizing");
    document.documentElement.classList.remove("camera-layout-dragging");
    document.querySelectorAll(".camera-drag-ghost").forEach((ghost) => ghost.remove());
  }, []);

  useEffect(() => {
    incidentFeedCacheRef.current.clear();
    incidentLoadedQueryRef.current = "";
    setIncidents([]);
    setIncidentHasMore(false);
    setIncidentLoading(true);
    setIncidentLoadError("");
    setIncidentPage(0);
    setExpandedIncidentId(null);
    setRetainedFocusedIncident(null);
  }, [eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter]);

  useEffect(() => {
    incidentFeedCacheRef.current.clear();
    setIncidentPage(0);
  }, [incidentsPerPage]);

  function refreshIncidents() {
    incidentSelectionRequestRef.current += 1;
    incidentFeedCacheRef.current.clear();
    incidentDetailCacheRef.current.clear();
    setIncidentDetails({});
    setIncidentRefreshToken((value) => value + 1);
  }

  useAppEvents(({ type }) => {
    if (type !== "incident" || incidentPage !== 0 || document.hidden) return;
    window.clearTimeout(incidentEventRefreshTimer.current);
    incidentEventRefreshTimer.current = window.setTimeout(() => {
      incidentEventRefreshTimer.current = null;
      incidentFeedCacheRef.current.clear();
      if (focusedDetailQuery) {
        incidentDetailCacheRef.current.invalidate(focusedDetailQuery);
        incidentDetailCacheRef.current.load(focusedDetailQuery).then((detail) => {
          setIncidentDetails((current) => ({ ...current, [focusedDetailQuery]: detail }));
        }).catch(() => {
          // Keep the existing detail visible; the next event or fallback poll retries.
        });
      }
      setIncidentRefreshToken((value) => value + 1);
    }, 1000);
  });

  useEffect(() => {
    if (incidentPage !== 0) return undefined;
    const timer = window.setInterval(() => {
      if (document.hidden) return;
      incidentFeedCacheRef.current.clear();
      setIncidentRefreshToken((value) => value + 1);
    }, INCIDENT_REFRESH_FALLBACK_MS);
    return () => window.clearInterval(timer);
  }, [incidentPage]);

  useEffect(() => () => {
    incidentSelectionRequestRef.current += 1;
    window.clearTimeout(incidentEventRefreshTimer.current);
  }, []);

  useEffect(() => {
    if (!liveIncidentGalleryReady) return undefined;
    let cancelled = false;
    function feedQuery(page) {
      const query = new URLSearchParams({
        event_type: eventFilter,
        limit: String(incidentsPerPage),
        offset: String(page * incidentsPerPage),
        gap_seconds: "45",
      });
      if (incidentCameraFilter !== "all") query.set("camera_id", incidentCameraFilter);
      if (incidentObjectFilter !== "all") query.set("object_label", incidentObjectFilter);
      if (incidentZoneFilter !== "all") query.set("zone", incidentZoneFilter);
      return query.toString();
    }
    async function loadIncidentFeed() {
      const query = feedQuery(incidentPage);
      const cachedPayload = incidentFeedCacheRef.current.peek(query);
      const foregroundLoad = incidentLoadedQueryRef.current !== query;
      if (foregroundLoad && !cachedPayload) setIncidentLoading(true);
      setIncidentLoadError("");
      try {
        const payload = cachedPayload || await incidentFeedCacheRef.current.load(query);
        if (cancelled) return;
        incidentLoadedQueryRef.current = query;
        setIncidents(incidentsNewestFirst(payload.items || []));
        setIncidentFacets(payload.facets || { camera_ids: [], labels: [], zones: [] });
        setIncidentHasMore(Boolean(payload.has_more));
        const previousQuery = incidentPage > 0 ? feedQuery(incidentPage - 1) : "";
        const nextQuery = payload.has_more ? feedQuery(incidentPage + 1) : "";
        for (const adjacentQuery of [previousQuery, nextQuery].filter(Boolean)) {
          incidentFeedCacheRef.current.load(adjacentQuery).catch(() => {
            // A foreground navigation will retry a failed speculative request.
          });
        }
        incidentFeedCacheRef.current.retain([previousQuery, query, nextQuery]);
      } catch (error) {
        if (cancelled) return;
        setIncidentLoadError(error.message || "Unable to load recent incidents");
      } finally {
        if (!cancelled && foregroundLoad) setIncidentLoading(false);
      }
    }
    loadIncidentFeed();
    return () => {
      cancelled = true;
    };
  }, [eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, incidentPage, incidentsPerPage, incidentRefreshToken, liveIncidentGalleryReady]);

  useEffect(() => {
    if (!focusedSummary || !focusedDetailQuery || incidentDetails[focusedDetailQuery]) return;
    let cancelled = false;
    incidentDetailCacheRef.current.load(focusedDetailQuery).then((detail) => {
      if (!cancelled) setIncidentDetails((current) => ({ ...current, [focusedDetailQuery]: detail }));
    }).catch(() => {
      // The compact incident remains usable if investigation details fail.
    });
    return () => { cancelled = true; };
  }, [focusedSummary?.id, focusedDetailQuery, incidentDetails]);

  useEffect(() => {
    const context = incidentRecordingContext(selectedEvent || focusedIncident);
    if (context) onRecordingContextChange(context);
  }, [selectedEvent?.id, focusedIncident?.id, focusedIncident?.created_at, focusedIncident?.camera_id, onRecordingContextChange]);

  useEffect(() => {
    if (listedFocusedIncident) setRetainedFocusedIncident(listedFocusedIncident);
  }, [listedFocusedIncident]);

  useEffect(() => {
    function onKey(keyEvent) {
      if (keyEvent.key === "Escape" && expandedIncidentId && !selectedEvent) {
        keyEvent.preventDefault();
        setExpandedIncidentId(null);
        setRetainedFocusedIncident(null);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expandedIncidentId, selectedEvent]);

  function toggleIncident(incidentId) {
    const closing = String(expandedIncidentId) === String(incidentId);
    if (closing) {
      setExpandedIncidentId(null);
      setRetainedFocusedIncident(null);
      return;
    }
    const incident = visibleIncidents.find((candidate) => String(candidate.id) === String(incidentId));
    setRetainedFocusedIncident(incident || null);
    setExpandedIncidentId(incidentId);
  }

  function changeIncidentPage(nextPage) {
    setExpandedIncidentId(null);
    setRetainedFocusedIncident(null);
    setIncidentPage(Math.max(0, nextPage));
  }

  return (
    <main className="bento-grid live-grid">
      <section className="bento-card camera-zone live-camera-zone">
        <div className="live-layout-switch" aria-label="Live camera layout">
          <button
            type="button"
            className={normalizedLayoutMode === "auto" ? "active" : ""}
            onClick={() => setLiveLayoutMode("auto")}
            title="Automatic layout"
            aria-label="Use automatic live layout"
          ><Grid2X2 size={16} /></button>
          <button
            type="button"
            className={normalizedLayoutMode === "custom" ? "active" : ""}
            onClick={() => setLiveLayoutMode("custom")}
            title="Custom layout"
            aria-label="Use custom live layout"
          ><GripVertical size={16} /></button>
        </div>
        <div
          ref={liveCameraGridRef}
          className={`camera-grid live-camera-grid${normalizedLayoutMode === "custom" ? " custom-layout" : liveCameraLayoutReady ? " viewport-layout" : ""}`}
          style={normalizedLayoutMode === "custom" ? { "--custom-pack-row-height": `${customGridMetrics.packRowHeight}px` } : undefined}
        >
          {liveDefaultsReady ? displayedCameras.map((camera, index) => (
            <CameraTile
              key={`${camera.id}:${liveDefaultsInstance}`}
              camera={camera}
              timeZone={timeZone}
              refresh={refreshBase}
              onOpen={setExpandedCamera}
              onAspectChange={updateLiveCameraAspect}
              layout={normalizedLayoutMode === "auto" ? liveCameraLayoutById.get(camera.id) : null}
              customLayout={normalizedLayoutMode === "custom"}
              customStyle={normalizedLayoutMode === "custom" ? (() => {
                const size = customSizePreview[camera.id] || customLayout.sizes[camera.id];
                const measuredAspect = Number(liveCameraAspects[camera.id]);
                const placement = liveCustomTilePlacement(size, customGridMetrics, measuredAspect);
                return {
                  order: customLayout.order.indexOf(String(camera.id)),
                  gridColumn: `span ${placement.columns}`,
                  gridRow: `span ${placement.packedRows}`,
                  height: `${placement.height}px`,
                };
              })() : undefined}
              startDelayMs={index * 450}
              resizing={resizingCameraId === camera.id}
              aspectSnapped={Boolean((customSizePreview[camera.id] || customLayout.sizes[camera.id]).aspectLocked)}
              dragHandleProps={normalizedLayoutMode === "custom" ? {
                onPointerDown: (event) => beginCustomMove(event, camera.id),
              } : {}}
              resizeHandleProps={normalizedLayoutMode === "custom" ? {
                onPointerDown: (event) => beginCustomResize(event, camera.id),
              } : {}}
            />
          )) : null}
        </div>
      </section>
      <section className="bento-card events-zone" ref={liveIncidentZoneRef}>
        <div className="section-head compact incident-head">
          <div><h2>Recent Incidents</h2></div>
          <div className="incident-head-actions">
            <div className="incident-filter-toggle compact" aria-label="Incident type filter">
              <button className={eventFilter === "object" ? "active" : ""} onClick={() => setEventFilter("object")}>Object</button>
              <button className={eventFilter === "motion" ? "active" : ""} onClick={() => setEventFilter("motion")}>Motion</button>
            </div>
            <span className="shown-bubble">{visibleIncidents.length} shown</span>
          </div>
        </div>
        <div className="event-filter incident-filter-panel" aria-label="Incident filters">
          <div className="incident-filter-selects">
            <label>
              <span>Camera</span>
              <select value={incidentCameraFilter} onChange={(event) => setIncidentCameraFilter(event.target.value)}>
                <option value="all">All cameras</option>
                {incidentCameraOptions.map((id) => <option value={id} key={id}>{cameraNameById.get(id) || id}</option>)}
              </select>
            </label>
            <label>
              <span>Object</span>
              <select value={incidentObjectFilter} onChange={(event) => setIncidentObjectFilter(event.target.value)}>
                <option value="all">All objects</option>
                {incidentObjectOptions.map((label) => <option value={label} key={label}>{label}</option>)}
              </select>
            </label>
            <label>
              <span>Zone</span>
              <select value={incidentZoneFilter} onChange={(event) => setIncidentZoneFilter(event.target.value)}>
                <option value="all">All zones</option>
                {incidentZoneOptions.map((zone) => <option value={zone} key={zone}>{zone}</option>)}
              </select>
            </label>
          </div>
        </div>
        {focusedIncident ? (
          <div className="incident-focus">
            <IncidentCard
              incident={focusedIncident}
              timeZone={timeZone}
              expanded
              thumbnailAnnotations={thumbnailAnnotations}
              onToggle={toggleIncident}
              onSelect={openIncidentOverlay}
            />
          </div>
        ) : null}
        <div className="incident-gallery" ref={liveIncidentGalleryRef}>
          {incidentLoading && !visibleIncidents.length ? <div className="empty-state">Loading {eventFilter} incidents...</div> : null}
          {!visibleIncidents.length && incidentLoadError ? <div className="empty-state">{incidentLoadError}</div> : null}
          {visibleIncidents.length
            ? pagedIncidents.map((incident) => (
              <IncidentCard
                key={incident.id}
                incident={incident}
                timeZone={timeZone}
                expanded={false}
                selected={incident.id === focusedIncident?.id}
                thumbnailAnnotations={thumbnailAnnotations}
                onToggle={toggleIncident}
                onSelect={openIncidentOverlay}
              />
            ))
            : null}
          {!incidentLoading && !incidentLoadError && !visibleIncidents.length ? <div className="empty-state">No incidents match the current filters.</div> : null}
        </div>
        <div className={`incident-pager ${incidentPage > 0 || incidentHasMore ? "" : "placeholder"}`} aria-label="Incident pages" aria-hidden={incidentPage === 0 && !incidentHasMore}>
          <button type="button" onClick={() => changeIncidentPage(incidentPage - 1)} disabled={clampedIncidentPage === 0}>Prev</button>
          <span>{clampedIncidentPage + 1} / {incidentPageCount}</span>
          <button type="button" onClick={() => changeIncidentPage(incidentPage + 1)} disabled={!incidentHasMore}>Next</button>
        </div>
      </section>
      {selectedEvent ? <EventOverlay event={selectedEvent} events={visibleIncidents} timeZone={timeZone} onClose={closeIncidentOverlay} onSelect={openIncidentOverlay} onRefresh={refreshIncidents} /> : null}
      {expandedCamera ? <LiveCameraOverlay camera={expandedCamera} timeZone={timeZone} onClose={() => setExpandedCamera(null)} /> : null}
    </main>
  );
}

function eventClipUrl(eventId, before = 5, after = 5, source = "main") {
  const params = new URLSearchParams({ before: before.toFixed(3), after: after.toFixed(3), source });
  return appUrl(`/api/events/${eventId}/clip.mp4?${params.toString()}`);
}

function eventStreamUrl(eventId, before = 5, after = 5, source = "main") {
  const params = new URLSearchParams({ before: before.toFixed(3), after: after.toFixed(3), source });
  return appUrl(`/api/events/${eventId}/stream.m3u8?${params.toString()}`);
}

async function eventStreamTimelineStart(streamUrl, requestedWindowStartEpoch) {
  try {
    const response = await fetch(streamUrl);
    if (!response.ok) return requestedWindowStartEpoch;
    return hlsProgramStartEpoch(await response.text()) ?? requestedWindowStartEpoch;
  } catch {
    return requestedWindowStartEpoch;
  }
}

async function loadIncidentClipInfo(event, isCancelled = () => false) {
  const eventId = Number(event?.representative_event_id || event?.id);
  if (!Number.isFinite(eventId)) return null;
  let before = 5;
  let after = 5;
  try {
    const response = await fetch("/api/event-clip/settings");
    if (response.ok) {
      const settings = await response.json();
      before = Number(settings.before_seconds ?? before);
      after = Number(settings.after_seconds ?? after);
    }
  } catch {
    // Defaults keep incident playback useful if settings are temporarily unavailable.
  }
  if (isCancelled()) return null;
  const safeBefore = Number.isFinite(before) ? before : 5;
  const safeAfter = Number.isFinite(after) ? after : 5;
  const window = incidentClipWindow(event, safeBefore, safeAfter);
  const anchorEpoch = eventEpoch(event);
  const requestedWindowStartEpoch = Number.isFinite(anchorEpoch) ? anchorEpoch - window.before : null;
  const streamUrl = eventStreamUrl(eventId, window.before, window.after);
  const timelineStartEpoch = Number.isFinite(requestedWindowStartEpoch)
    ? await eventStreamTimelineStart(streamUrl, requestedWindowStartEpoch)
    : null;
  if (isCancelled()) return null;
  const initialPlaybackOffset = Math.max(0, window.before - safeBefore);
  return {
    streamUrl,
    downloadUrl: eventClipUrl(eventId, window.before, window.after),
    before: window.before,
    after: window.after,
    duration: window.before + window.after,
    windowStartEpoch: timelineStartEpoch,
    requestedWindowStartEpoch,
    initialPlaybackOffset,
    playbackStartOffset: hlsPlaybackOffset(requestedWindowStartEpoch, timelineStartEpoch, initialPlaybackOffset),
  };
}

function recordingDayUrl(cameraId, startEpoch, endEpoch, source) {
  const params = new URLSearchParams({
    start_epoch: startEpoch.toFixed(3),
    end_epoch: endEpoch.toFixed(3),
    source,
  });
  return appUrl(`/api/cameras/${cameraId}/recordings/day?${params.toString()}`);
}

function recordingWindowUrl(cameraId, startEpoch, endEpoch, source) {
  const params = new URLSearchParams({
    start_epoch: startEpoch.toFixed(3),
    end_epoch: endEpoch.toFixed(3),
    source,
  });
  return appUrl(`/api/cameras/${cameraId}/recordings/window?${params.toString()}`);
}

function recordingUpdatesUrl(cameraId, startEpoch, endEpoch, afterEpoch, source) {
  const params = new URLSearchParams({
    start_epoch: startEpoch.toFixed(3),
    end_epoch: endEpoch.toFixed(3),
    after_epoch: afterEpoch.toFixed(3),
    source,
  });
  return appUrl(`/api/cameras/${cameraId}/recordings/updates?${params.toString()}`);
}

function recordingDayHlsUrl(cameraId, startEpoch, endEpoch, source) {
  const params = new URLSearchParams({
    start_epoch: startEpoch.toFixed(3),
    end_epoch: endEpoch.toFixed(3),
    source,
  });
  return appUrl(`/api/cameras/${cameraId}/recordings/day.m3u8?${params.toString()}`);
}

function recordingGridDayUrl(startEpoch, endEpoch, source) {
  const params = new URLSearchParams({
    start_epoch: startEpoch.toFixed(3),
    end_epoch: endEpoch.toFixed(3),
    source,
  });
  return appUrl(`/api/recordings/grid/day?${params.toString()}`);
}

function recordingGridUpdatesUrl(startEpoch, endEpoch, afterEpoch, source) {
  const params = new URLSearchParams({
    start_epoch: startEpoch.toFixed(3),
    end_epoch: endEpoch.toFixed(3),
    after_epoch: afterEpoch.toFixed(3),
    source,
  });
  return appUrl(`/api/recordings/grid/updates?${params.toString()}`);
}

function recordingPreviewUrl(cameraId, epoch, source) {
  const params = new URLSearchParams({
    epoch: epoch.toFixed(3),
    source,
  });
  return appUrl(`/api/cameras/${cameraId}/recordings/preview.jpg?${params.toString()}`);
}

function mergeRecordingEvents(current, updates) {
  const byId = new Map(current.map((event) => [event.id, event]));
  updates.forEach((event) => byId.set(event.id, event));
  return [...byId.values()]
    .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    .slice(-5000);
}

function recordingIncidentEpoch(incident) {
  const direct = Number(incident?.start_epoch);
  if (Number.isFinite(direct)) return direct;
  const parsed = new Date(incident?.start_at || incident?.created_at || "").getTime() / 1000;
  return Number.isFinite(parsed) ? parsed : null;
}

function recordingIncidentEndEpoch(incident) {
  const direct = Number(incident?.last_epoch);
  if (Number.isFinite(direct)) return direct;
  const parsed = new Date(incident?.end_at || incident?.created_at || "").getTime() / 1000;
  return Number.isFinite(parsed) ? parsed : recordingIncidentEpoch(incident);
}

function recordingPlaybackTimeline(rows) {
  let mediaOffset = 0;
  return (rows || [])
    .map((item) => ({
      ...item,
      start_epoch: Number(item.start_epoch),
      end_epoch: Number(item.end_epoch),
    }))
    .filter((item) => Number.isFinite(item.start_epoch) && Number.isFinite(item.end_epoch))
    .sort((left, right) => left.start_epoch - right.start_epoch)
    .map((item) => {
      const duration = Math.max(0.01, Number(item.duration_seconds) || item.end_epoch - item.start_epoch);
      const result = { ...item, media_start: mediaOffset, media_end: mediaOffset + duration };
      mediaOffset += duration;
      return result;
    });
}

function RecordingGridTile({ camera, source, epoch, playing, focused, layout, onAspectChange, onFocus, onSelect }) {
  const videoRef = useRef(null);
  const previousEpochRef = useRef(null);
  const [playback, setPlayback] = useState(null);
  const [error, setError] = useState("");
  const bucket = Math.floor(Math.max(0, Number(epoch) || 0) / (15 * 60)) * 15 * 60;
  const retryScope = `${camera?.id || ""}:${source}:${bucket}`;
  const [retryState, setRetryState] = useState({ scope: retryScope, count: 0 });
  const retryCount = retryState.scope === retryScope ? retryState.count : 0;
  const timeline = useMemo(
    () => recordingPlaybackTimeline(playback?.rows || []),
    [playback?.rows],
  );
  const exactMediaTime = playbackMediaTimeForEpoch(timeline, epoch);
  const mediaTime = Number.isFinite(exactMediaTime)
    ? exactMediaTime
    : playbackMediaTimeForEpoch(timeline, epoch, 1.25);
  const briefGap = !Number.isFinite(exactMediaTime) && Number.isFinite(mediaTime);
  const hasCoverage = Number.isFinite(mediaTime);

  useEffect(() => {
    if (!camera?.id || !bucket) return undefined;
    const controller = new AbortController();
    let cancelled = false;
    setPlayback(null);
    setError("");
    const candidates = source === "main" ? ["main", "live"] : ["live", "main"];
    async function load() {
      for (const candidate of candidates) {
        const response = await fetch(
          recordingWindowUrl(camera.id, bucket, bucket + 15 * 60, candidate),
          { signal: controller.signal },
        );
        if (!response.ok) continue;
        const payload = await response.json();
        if (!(payload.recordings || []).length) continue;
        if (!cancelled) {
          setPlayback({
            source: candidate,
            start: Number(payload.start_epoch),
            end: Number(payload.end_epoch),
            rows: payload.recordings,
            targetEpoch: epoch,
          });
        }
        return;
      }
      if (!cancelled) setError("No recording at this time");
    }
    load().catch((loadError) => {
      if (!cancelled && loadError.name !== "AbortError") setError("Recording unavailable");
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [bucket, camera?.id, retryCount, source]);

  useEffect(() => {
    const nearLive = Math.abs(Date.now() / 1000 - epoch) <= 5 * 60;
    if (retryCount >= 4 || !nearLive || (!error && (!playback || hasCoverage))) return undefined;
    const timer = window.setTimeout(() => setRetryState((current) => ({
      scope: retryScope,
      count: current.scope === retryScope ? current.count + 1 : 1,
    })), 3000);
    return () => window.clearTimeout(timer);
  }, [epoch, error, hasCoverage, playback, retryCount, retryScope]);

  useEffect(() => {
    const video = videoRef.current;
    const previousEpoch = previousEpochRef.current;
    previousEpochRef.current = epoch;
    if (!video || !Number.isFinite(mediaTime)) {
      video?.pause();
      return;
    }
    if (briefGap) {
      video.pause();
      return;
    }
    if (gridPlaybackNeedsSeek({
      currentTime: video.currentTime,
      targetTime: mediaTime,
      playing,
      epochDelta: Number.isFinite(previousEpoch) ? epoch - previousEpoch : null,
    })) video.currentTime = mediaTime;
    if (playing) video.play().catch(() => {});
    else video.pause();
  }, [briefGap, mediaTime, playing]);

  const manifestUrl = playback
    ? recordingDayHlsUrl(camera.id, playback.start, playback.end, playback.source)
    : "";
  const initialMediaTime = playbackMediaTimeForEpoch(timeline, playback?.targetEpoch, 1.25);
  const displayedSource = playback?.source || source;
  const aspect = recordingCameraAspect(camera, displayedSource);
  useEffect(() => {
    onAspectChange(camera.id, aspect);
  }, [aspect, camera.id, onAspectChange]);
  return (
    <article
      className={`recording-grid-tile${hasCoverage ? "" : " gap"}${focused ? " focused" : ""}`}
      style={{
        "--recording-aspect": aspect,
        ...(!focused && layout ? {
          left: `${layout.x}px`,
          top: `${layout.y}px`,
          width: `${layout.width}px`,
          height: `${layout.height}px`,
        } : {}),
      }}
    >
      {manifestUrl ? <ShakaVideo
        ref={videoRef}
        src={manifestUrl}
        mimeType="application/vnd.apple.mpegurl"
        startTime={Number.isFinite(initialMediaTime) ? initialMediaTime : 0}
        bufferingGoal={20}
        muted
        playsInline
        preload="auto"
        onReady={(_player, video) => {
          setError("");
          if (Number.isFinite(mediaTime)) video.currentTime = mediaTime;
          if (playing && Number.isFinite(mediaTime)) video.play().catch(() => {});
        }}
        onError={() => setError("Playback unavailable")}
      /> : null}
      <button
        type="button"
        className="recording-grid-focus-hit"
        onClick={() => onFocus(camera.id)}
        aria-label={focused ? `Restore ${camera.name} grid tile` : `Enlarge ${camera.name} recording`}
        title={focused ? "Return to camera grid" : "Enlarge camera"}
      />
      <button type="button" className="recording-grid-camera" onClick={() => onSelect(camera.id)} title={`Open ${camera.name} recording`}>
        <Camera size={14} /><span>{camera.name}</span>
        {playback?.source && playback.source !== source ? <em>{playback.source === "live" ? "Sub" : "Main"}</em> : null}
      </button>
      {!playback && !error ? <div className="recording-grid-status"><RefreshCcw className="spin" size={17} />Loading</div> : null}
      {(!hasCoverage || error) && playback ? <div className="recording-grid-status"><Film size={17} />{error || "No recording at this time"}</div> : null}
      {error && !playback ? <div className="recording-grid-status"><Film size={17} />{error}</div> : null}
    </article>
  );
}

function RecordingCameraGrid({ cameras, source, epoch, playing, onSelect }) {
  const [focusedCameraId, setFocusedCameraId] = useState("");
  const [gridSize, setGridSize] = useState({ width: 0, height: 0 });
  const [aspectOverrides, setAspectOverrides] = useState({});
  const gridRef = useRef(null);
  const layout = useMemo(
    () => recordingGridLayout(cameras, source, gridSize.width, gridSize.height, 6, aspectOverrides),
    [aspectOverrides, cameras, gridSize.height, gridSize.width, source],
  );

  const updateCameraAspect = React.useCallback((cameraId, aspect) => {
    setAspectOverrides((current) => (
      current[cameraId] === aspect ? current : { ...current, [cameraId]: aspect }
    ));
  }, []);

  useLayoutEffect(() => {
    const grid = gridRef.current;
    if (!grid) return undefined;
    const update = () => {
      const next = { width: grid.clientWidth, height: grid.clientHeight };
      setGridSize((current) => (
        current.width === next.width && current.height === next.height ? current : next
      ));
    };
    update();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", update);
      return () => window.removeEventListener("resize", update);
    }
    const observer = new ResizeObserver(update);
    observer.observe(grid);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!focusedCameraId) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setFocusedCameraId("");
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [focusedCameraId]);

  function focusCamera(cameraId) {
    setFocusedCameraId((current) => current === cameraId ? "" : cameraId);
    gridRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }

  const items = layout.length
    ? layout
    : cameras.map((camera) => ({ camera, x: 0, y: 0, width: 0, height: 0 }));
  return <div ref={gridRef} className={`recording-camera-grid${focusedCameraId ? " has-focus" : ""}`}>
    {items.map((item) => <RecordingGridTile
      key={item.camera.id}
      camera={item.camera}
      source={source}
      epoch={epoch}
      playing={playing}
      focused={item.camera.id === focusedCameraId}
      layout={item}
      onAspectChange={updateCameraAspect}
      onFocus={focusCamera}
      onSelect={onSelect}
    />)}
    {focusedCameraId ? <button type="button" className="recording-grid-close-focus" onClick={() => setFocusedCameraId("")} aria-label="Return to all camera tiles"><X size={18} /></button> : null}
  </div>;
}

const ALL_RECORDING_CAMERAS_ID = "all";

function RecordingSectionSwitcher({ mode, cameraId = "" }) {
  const query = cameraId ? `?camera=${encodeURIComponent(cameraId)}` : "";
  return (
    <div className="recordings-section-switcher" aria-label="Recording section">
      <a className={mode === "history" ? "active" : ""} href={appUrl(`/recordings${query}`)}><Film size={14} />History</a>
      <a className={mode === "search" ? "active" : ""} href={appUrl("/recordings/search")}><Search size={14} />Smart Search</a>
      <a className={mode === "exports" ? "active" : ""} href={appUrl("/recordings/exports")}><Download size={14} />Exports</a>
    </div>
  );
}

function RecordingCameraRail({ subtitle, children }) {
  return (
    <aside className="recordings-v2-cameras" aria-label="Cameras">
      <header className="recordings-camera-header">
        <strong>Cameras</strong>
        <small>{subtitle}</small>
      </header>
      <div className="recordings-camera-list">{children}</div>
    </aside>
  );
}

function SemanticSearchPage({ timeZone, onAssistantContextChange }) {
  const initialQuery = useMemo(() => new URLSearchParams(window.location.search), []);
  const localStorage = useMemo(() => browserStorage(window), []);
  const sessionStorage = useMemo(() => {
    try { return window.sessionStorage; } catch { return null; }
  }, []);
  const initialQueryText = initialQuery.get("q") || "";
  const initialCameraId = initialQuery.get("camera") || "";
  const restoredSearch = useMemo(
    () => readSemanticSearchSession(sessionStorage, initialQueryText, initialCameraId),
    [initialCameraId, initialQueryText, sessionStorage],
  );
  const [cameras, setCameras] = useState([]);
  const [cameraId, setCameraId] = useState(initialCameraId);
  const [query, setQuery] = useState(initialQueryText);
  const [results, setResults] = useState(() => restoredSearch?.results || []);
  const [searchHistory, setSearchHistory] = useState(() => readSemanticSearchHistory(localStorage));
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const searchRequestRef = useRef(null);
  const visibleResults = useMemo(
    () => semanticSearchResultsForCamera(results, cameraId),
    [cameraId, results],
  );

  useEffect(() => {
    Promise.all([fetch("/api/cameras").then((r) => r.json()), fetch("/api/semantic-search/status").then((r) => r.json())])
      .then(([cameraRows, semanticStatus]) => { setCameras(cameraRows || []); setStatus(semanticStatus); })
      .catch((reason) => setError(reason.message || "Could not load Smart Search."));
  }, []);
  useEffect(() => {
    onAssistantContextChange?.({ page: "recordings", camera_id: cameraId, filters: { semantic_query: query } });
  }, [cameraId, onAssistantContextChange, query]);
  useEffect(() => () => {
    const activeRequest = searchRequestRef.current;
    searchRequestRef.current = null;
    activeRequest?.abort();
  }, []);

  async function runSearch(event, requestedQuery = query, requestedCameraId = cameraId) {
    event?.preventDefault();
    const searchQuery = String(requestedQuery || "").trim();
    const searchCameraId = String(requestedCameraId || "");
    if (!searchQuery) return;
    searchRequestRef.current?.abort();
    const controller = new AbortController();
    searchRequestRef.current = controller;
    setQuery(searchQuery);
    setCameraId(searchCameraId);
    setLoading(true); setError("");
    try {
      const response = await fetch("/api/semantic-search", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, camera_ids: searchCameraId ? [searchCameraId] : [], limit: 100 }),
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Smart Search failed.");
      const nextResults = payload.results || [];
      setResults(nextResults);
      writeSemanticSearchSession(sessionStorage, { query: searchQuery, cameraId: searchCameraId, results: nextResults });
      setSearchHistory((current) => {
        const next = addSemanticSearchHistory(current, {
          query: searchQuery,
          cameraId: searchCameraId,
          searchedAt: new Date().toISOString(),
        });
        writeSemanticSearchHistory(localStorage, next);
        return next;
      });
      const params = new URLSearchParams({ q: searchQuery });
      if (searchCameraId) params.set("camera", searchCameraId);
      window.history.replaceState(null, "", appUrl(`/recordings/search?${params.toString()}`));
    } catch (reason) {
      if (reason?.name !== "AbortError") setError(reason.message || "Smart Search failed.");
    } finally {
      if (searchRequestRef.current === controller) {
        searchRequestRef.current = null;
        setLoading(false);
      }
    }
  }

  function resetSearch() {
    searchRequestRef.current?.abort();
    searchRequestRef.current = null;
    setLoading(false);
    setQuery("");
    setCameraId("");
    setResults([]);
    setError("");
    clearSemanticSearchSession(sessionStorage);
    window.history.replaceState(null, "", appUrl("/recordings/search"));
  }

  function selectCamera(nextCameraId) {
    if (loading) return;
    const selectedCameraId = String(nextCameraId || "");
    setCameraId(selectedCameraId);
    if (!query.trim()) return;
    writeSemanticSearchSession(sessionStorage, { query: query.trim(), cameraId: selectedCameraId, results });
    const params = new URLSearchParams({ q: query.trim() });
    if (selectedCameraId) params.set("camera", selectedCameraId);
    window.history.replaceState(null, "", appUrl(`/recordings/search?${params.toString()}`));
  }

  return <main className="recordings-v2-page semantic-search-page">
    <nav className="recordings-tabs"><RecordingSectionSwitcher mode="search" cameraId={cameraId} /></nav>
    <RecordingCameraRail subtitle="Filter search results">
      <button type="button" className={!cameraId ? "active" : ""} onClick={() => selectCamera("")} disabled={loading}><Search size={16} /><span>All cameras</span><i /></button>
      {cameras.map((camera) => <button type="button" key={camera.id} className={cameraId === camera.id ? "active" : ""} onClick={() => selectCamera(camera.id)} disabled={loading}><Camera size={16} /><span>{camera.name}</span><i className={camera.running ? "online" : ""} /></button>)}
    </RecordingCameraRail>
    <section className="semantic-search-workspace">
      <header><div><h2>Smart Search</h2><p>Describe what you remember. SurvNG searches locally indexed incident images.</p></div><span className={`semantic-status ${status?.state || ""}`}>{status?.state === "ready" ? `${Number(status.event_count || 0).toLocaleString()} incidents indexed` : status?.state || "Loading"}</span></header>
      <form onSubmit={runSearch}><Search size={20} /><input value={query} onChange={(event) => setQuery(event.target.value)} disabled={loading} placeholder='Try “person in a red jacket” or “white delivery truck”' /><div className="semantic-search-actions"><button type="button" className="secondary" onClick={resetSearch} disabled={!query && !cameraId && !results.length && !error}>Reset</button><button type="submit" disabled={loading || !query.trim()}>{loading ? "Searching…" : "Search"}</button></div></form>
      {searchHistory.length ? <div className="semantic-search-history" aria-label="Recent Smart Search history">
        <span><Clock3 size={14} />Recent</span>
        <div>{searchHistory.map((item) => {
          const cameraName = item.cameraId ? cameras.find((camera) => camera.id === item.cameraId)?.name || item.cameraId : "All cameras";
          return <button type="button" key={`${item.query.toLocaleLowerCase()}-${item.cameraId}`} onClick={() => void runSearch(null, item.query, item.cameraId)} disabled={loading} title={`Search ${cameraName}`}><strong>{item.query}</strong><small>{cameraName}</small></button>;
        })}</div>
      </div> : null}
      {error ? <div className="semantic-search-error"><CircleAlert size={17} />{error}</div> : null}
      <div className="semantic-search-results">
        {visibleResults.map((result) => {
          const item = result.event || {};
          const context = incidentRecordingContext(item);
          const matchLabel = ({ strong_match: "Strong match", possible_match: "Possible match" })[result.match_strength] || "Visually similar";
          return <article key={item.id}><a href={appUrl(`/incidents?event_ids=${item.id}`)}><img src={result.snapshot_url} alt="" loading="lazy" /><span title={`Raw visual similarity ${Number(result.score || 0).toFixed(3)}`}>{matchLabel}</span></a><footer><div><strong>{cameras.find((camera) => camera.id === item.camera_id)?.name || item.camera_id}</strong><small>{formatDateTime(new Date(item.created_at).getTime() / 1000, timeZone)}</small></div><a href={recordingsHref(context)}><Play size={14} />View recording</a></footer></article>;
        })}
        {!loading && !error && !visibleResults.length ? <div className="semantic-search-empty"><Search size={28} /><strong>{results.length && cameraId ? "No matching results from this camera" : "Search indexed incidents by appearance"}</strong><span>{results.length && cameraId ? "Choose All cameras or another camera to widen the current results." : "Results link to the exact incident and recording time."}</span></div> : null}
      </div>
    </section>
  </main>;
}

function RecordingsPage({ timeZone, onAssistantContextChange }) {
  const initialQuery = useMemo(() => new URLSearchParams(window.location.search), []);
  const queryAt = Number(initialQuery.get("at"));
  const initialEpoch = Number.isFinite(queryAt) && queryAt > 0 ? queryAt : null;
  const videoRef = useRef(null);
  const desiredEpochRef = useRef(initialEpoch);
  const autoplayRef = useRef(false);
  const codecFallbackRef = useRef(false);
  const playbackRequestRef = useRef(0);
  const latestAvailabilityRef = useRef(null);
  const pendingSeekEpochRef = useRef(null);
  const pendingSeekModeRef = useRef(null);
  const playbackRetryRef = useRef({ attempts: 0, timer: null });
  const gridRefreshCursorRef = useRef(null);
  const today = dateKeyForTimeZone(Date.now(), timeZone);
  const queryDate = initialQuery.get("date") || (initialEpoch ? dateKeyForTimeZone(initialEpoch * 1000, timeZone) : today);
  const querySource = initialQuery.get("source");
  const [cameras, setCameras] = useState([]);
  const [cameraId, setCameraId] = useState(initialQuery.get("camera") || "");
  const [source, setSource] = useState(querySource === "live" || querySource === "main"
    ? querySource
    : initialQuery.get("camera") === ALL_RECORDING_CAMERAS_ID ? "live" : preferredStreamSource());
  const [date, setDate] = useState(/^\d{4}-\d{2}-\d{2}$/.test(queryDate) && queryDate <= today ? queryDate : today);
  const [recordings, setRecordings] = useState([]);
  const [playbackDetail, setPlaybackDetail] = useState(null);
  const [events, setEvents] = useState([]);
  const [eventFilter, setEventFilter] = useState("object");
  const [incidentRangeHours, setIncidentRangeHours] = useState(1);
  const [availableSources, setAvailableSources] = useState([]);
  const [playhead, setPlayhead] = useState(null);
  const [loading, setLoading] = useState(true);
  const [playbackError, setPlaybackError] = useState("");
  const [playbackErrorStage, setPlaybackErrorStage] = useState("");
  const [playbackNotice, setPlaybackNotice] = useState("");
  const [playbackBlocked, setPlaybackBlocked] = useState(false);
  const [playbackWindow, setPlaybackWindow] = useState(null);
  const [playbackWindowRevision, setPlaybackWindowRevision] = useState(0);
  const [manifestRetryToken, setManifestRetryToken] = useState(0);
  const [recordingIndexRevision, setRecordingIndexRevision] = useState(0);
  const [followTarget, setFollowTarget] = useState(null);
  const [exportRange, setExportRange] = useState(null);
  const [exportKind, setExportKind] = useState("recording");
  const [exportOptions, setExportOptions] = useState({ interval: 30, fps: 30, clipHeight: 0, timelapseHeight: 720 });
  const [exportLabel, setExportLabel] = useState("");
  const [exportJob, setExportJob] = useState(null);
  const [exportError, setExportError] = useState("");
  const [exportSubmitting, setExportSubmitting] = useState(false);
  const [gridPlaying, setGridPlaying] = useState(false);

  const isAllCameras = cameraId === ALL_RECORDING_CAMERAS_ID;
  const activeCameraId = isAllCameras
    ? ALL_RECORDING_CAMERAS_ID
    : cameras.some((camera) => camera.id === cameraId) ? cameraId : cameras[0]?.id || "";
  const dayStart = useMemo(() => zonedDateSecondToEpoch(date, 0, timeZone), [date, timeZone]);
  const nextDate = addDaysToDateKey(date, 1);
  const dayEnd = useMemo(() => zonedDateSecondToEpoch(nextDate, 0, timeZone), [nextDate, timeZone]);
  const daySeconds = Math.max(1, dayEnd - dayStart);

  useEffect(() => {
    onAssistantContextChange?.({
      page: "recordings",
      camera_id: isAllCameras ? "" : activeCameraId,
      recording_epoch: Number(playhead) || initialEpoch || dayStart,
      filters: { date, source, event_type: eventFilter, camera_view: isAllCameras ? "all" : "single" },
    });
  }, [activeCameraId, date, dayStart, eventFilter, initialEpoch, isAllCameras, onAssistantContextChange, playhead, source]);

  const timeline = useMemo(() => {
    return recordingPlaybackTimeline(recordings);
  }, [recordings]);
  latestAvailabilityRef.current = timeline[timeline.length - 1]?.end_epoch ?? null;
  const playbackTimeline = useMemo(() => {
    if (!playbackDetail) return [];
    return recordingPlaybackTimeline(playbackDetail.rows);
  }, [playbackDetail]);
  const loadedPlaybackWindow = playbackDetail;
  const manifestUrl = !isAllCameras && activeCameraId && playbackDetail && playbackTimeline.length
    ? `${recordingDayHlsUrl(activeCameraId, playbackDetail.start, playbackDetail.end, source)}&reload=${playbackDetail.revision || 0}-${manifestRetryToken}`
    : "";
  const manifestStartTime = useMemo(() => {
    if (!playbackTimeline.length) return null;
    const retainedEpoch = desiredEpochRef.current;
    const initialEpoch = Number.isFinite(retainedEpoch) && retainedEpoch >= dayStart && retainedEpoch < dayEnd
      ? retainedEpoch
      : date === today ? Date.now() / 1000 : timeline[0].start_epoch;
    return epochToPlaybackMediaTime(initialEpoch);
  }, [manifestUrl, playbackTimeline]);

  const filteredEvents = useMemo(() => events
    .filter((event) => eventFilter === "object" ? Boolean(event.has_objects) : !event.has_objects)
    .map((event) => ({ ...event, incident_epoch: recordingIncidentEpoch(event) }))
    .filter((event) => Number.isFinite(event.incident_epoch))
    .sort((left, right) => left.incident_epoch - right.incident_epoch), [eventFilter, events]);

  const nearbyEvents = useMemo(() => {
    if (!Number.isFinite(playhead)) return [];
    if (incidentRangeHours >= 24) return filteredEvents;
    const maximumDistance = incidentRangeHours * 30 * 60;
    return filteredEvents
      .map((event) => ({ ...event, distance: Math.abs(event.incident_epoch - playhead) }))
      .filter((event) => event.distance <= maximumDistance)
      .sort((left, right) => left.incident_epoch - right.incident_epoch)
  }, [filteredEvents, incidentRangeHours, playhead]);

  useEffect(() => {
    if (!exportJob?.id || !["queued", "running", "cancelling"].includes(exportJob.status)) return undefined;
    let stopped = false;
    let timer = null;
    const refresh = async () => {
      try {
        const response = await fetch(appUrl(`/api/exports/${exportJob.id}`));
        if (!response.ok) throw new Error(`Export status failed (${response.status})`);
        const next = await response.json();
        if (!stopped) {
          setExportJob(next);
          if (["queued", "running", "cancelling"].includes(next.status)) {
            timer = window.setTimeout(refresh, 1000);
          }
        }
      } catch (error) {
        if (!stopped) {
          setExportError(error.message || "Unable to refresh export status");
          timer = window.setTimeout(refresh, 2500);
        }
      }
    };
    timer = window.setTimeout(refresh, 700);
    return () => {
      stopped = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [exportJob?.id, exportJob?.status]);

  useEffect(() => {
    setExportRange(null);
    setExportJob(null);
    setExportError("");
    setExportLabel("");
  }, [activeCameraId, date, source]);

  useEffect(() => {
    if (!activeCameraId) return undefined;
    const controller = new AbortController();
    fetch(appUrl("/api/exports?limit=100"), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Export history failed (${response.status})`);
        return response.json();
      })
      .then((payload) => {
        const active = (payload.exports || []).find((job) => (
          job.camera_id === activeCameraId
          && job.source === source
          && ["queued", "running", "cancelling"].includes(job.status)
        ));
        if (!active || controller.signal.aborted) return;
        setExportKind(active.kind === "timelapse" ? "timelapse" : "recording");
        setExportRange({ start: Number(active.start_epoch), end: Number(active.end_epoch) });
        setExportOptions({
          interval: Number(active.options?.sample_interval_seconds) || 30,
          fps: Number(active.options?.output_fps) || 30,
          clipHeight: active.kind === "recording" ? Number(active.options?.height) || 0 : 0,
          timelapseHeight: active.kind === "timelapse" ? Number(active.options?.height) || 720 : 720,
        });
        setExportLabel(active.label || "");
        setExportJob(active);
      })
      .catch((error) => {
        if (error.name !== "AbortError") console.warn("Unable to restore active export", error);
      });
    return () => controller.abort();
  }, [activeCameraId, date, source]);

  function snapToRecording(epoch) {
    if (!timeline.length) return null;
    const bounded = Math.max(dayStart, Math.min(dayEnd - 0.01, epoch));
    const latest = timeline[timeline.length - 1];
    if (bounded >= latest.end_epoch) return latest.end_epoch - 0.01;
    const containing = timeline.find((item) => item.start_epoch <= bounded && bounded < item.end_epoch);
    if (containing) return bounded;
    let nearest = timeline[0].start_epoch;
    let distance = Math.abs(nearest - bounded);
    timeline.forEach((item) => {
      [item.start_epoch, item.end_epoch - 0.01].forEach((candidate) => {
        const nextDistance = Math.abs(candidate - bounded);
        if (nextDistance < distance) {
          nearest = candidate;
          distance = nextDistance;
        }
      });
    });
    return nearest;
  }

  function mediaTimeToEpoch(mediaTime) {
    const clip = playbackTimeline.find((item) => item.media_start <= mediaTime && mediaTime < item.media_end)
      || playbackTimeline[playbackTimeline.length - 1];
    if (!clip) return null;
    return clip.start_epoch + Math.max(0, Math.min(clip.end_epoch - clip.start_epoch, mediaTime - clip.media_start));
  }

  function epochToPlaybackMediaTime(epoch) {
    if (!playbackTimeline.length) return null;
    const clip = playbackTimeline.find((item) => item.start_epoch <= epoch && epoch < item.end_epoch)
      || playbackTimeline.reduce((nearest, item) => {
        if (!nearest) return item;
        return Math.abs(item.start_epoch - epoch) < Math.abs(nearest.start_epoch - epoch) ? item : nearest;
      }, null);
    return clip.media_start + Math.max(0, Math.min(clip.media_end - clip.media_start - 0.01, epoch - clip.start_epoch));
  }

  function windowAround(epoch) {
    const windowSeconds = 15 * 60;
    const bucket = Math.floor(Math.max(0, epoch - dayStart) / windowSeconds);
    const start = dayStart + bucket * windowSeconds;
    return {
      start,
      end: Math.min(dayEnd, start + windowSeconds),
    };
  }

  function requestPlaybackWindow(nextWindow) {
    setPlaybackWindow(nextWindow);
    setPlaybackWindowRevision((revision) => revision + 1);
  }

  function requestRecordingPlay(video, showBlocked = true) {
    if (!video) return;
    video.play().then(() => {
      setPlaybackBlocked(false);
    }).catch((error) => {
      if (showBlocked && error?.name === "NotAllowedError") {
        setPlaybackBlocked(true);
        setPlaybackNotice("Tap to play recording");
      }
    });
  }

  function playAt(epoch, autoplay = true) {
    const target = snapToRecording(epoch);
    if (target === null || !activeCameraId) return;
    if (isAllCameras) {
      autoplayRef.current = autoplay;
      setGridPlaying(autoplay);
      setPlaybackError("");
      setPlaybackErrorStage("");
      setPlaybackNotice("");
      setPlayhead(target);
      desiredEpochRef.current = target;
      return;
    }
    autoplayRef.current = autoplay;
    setFollowTarget(null);
    setPlaybackError("");
    setPlaybackErrorStage("");
    setPlayhead(target);
    desiredEpochRef.current = target;
    const nextWindow = windowAround(target);
    const inCurrentWindow = loadedPlaybackWindow
      && target >= loadedPlaybackWindow.start
      && target < loadedPlaybackWindow.end;
    const coveredByCurrentManifest = playbackRowsCoverEpoch(playbackTimeline, target);
    const video = videoRef.current;
    if (autoplay && video) requestRecordingPlay(video, false);
    const mediaTime = epochToPlaybackMediaTime(target);
    if (inCurrentWindow && coveredByCurrentManifest && video && Number.isFinite(mediaTime)) {
      pendingSeekEpochRef.current = target;
      pendingSeekModeRef.current = "local";
      playbackRequestRef.current += 1;
      setPlaybackWindow(null);
      setPlaybackNotice("Seeking...");
      video.currentTime = mediaTime;
      if (autoplay) requestRecordingPlay(video);
      else setPlaybackNotice("");
    } else {
      pendingSeekEpochRef.current = target;
      pendingSeekModeRef.current = "window";
      setPlaybackNotice("Loading recording...");
      requestPlaybackWindow(nextWindow);
    }
  }

  useEffect(() => {
    if (!isAllCameras || !gridPlaying || !timeline.length) return undefined;
    let previous = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now();
      const elapsed = Math.max(0, Math.min(2, (now - previous) / 1000));
      previous = now;
      setPlayhead((current) => {
        const base = Number.isFinite(current) ? current : timeline[0].start_epoch;
        const requested = base + elapsed;
        const containing = timeline.find((item) => item.start_epoch <= requested && requested < item.end_epoch);
        let next = containing ? requested : null;
        if (!Number.isFinite(next)) {
          const following = timeline.find((item) => item.end_epoch > requested);
          next = following ? Math.max(requested, following.start_epoch) : null;
        }
        if (!Number.isFinite(next) || next >= dayEnd) {
          setGridPlaying(false);
          return base;
        }
        desiredEpochRef.current = next;
        return next;
      });
    }, 500);
    return () => window.clearInterval(timer);
  }, [dayEnd, gridPlaying, isAllCameras, timeline]);

  useEffect(() => {
    if (!isAllCameras || !gridPlaying) return undefined;
    const pauseWhenHidden = () => {
      if (document.hidden) setGridPlaying(false);
    };
    document.addEventListener("visibilitychange", pauseWhenHidden);
    return () => document.removeEventListener("visibilitychange", pauseWhenHidden);
  }, [gridPlaying, isAllCameras]);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/cameras", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Camera status failed (${response.status})`);
        return response.json();
      })
      .then(setCameras)
      .catch((error) => {
        if (error.name !== "AbortError") {
          setPlaybackErrorStage("index");
          setPlaybackError("Unable to load cameras");
        }
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!activeCameraId) return undefined;
    const controller = new AbortController();
    setLoading(true);
    setPlaybackError("");
    setPlaybackErrorStage("");
    setPlaybackBlocked(false);
    setRecordings([]);
    playbackRequestRef.current += 1;
    setPlaybackDetail(null);
    setEvents([]);
    setAvailableSources([]);
    setPlaybackWindow(null);
    setGridPlaying(false);
    gridRefreshCursorRef.current = null;
    setManifestRetryToken(0);
    setFollowTarget(null);
    pendingSeekEpochRef.current = null;
    pendingSeekModeRef.current = null;
    if (Number.isFinite(playhead)) desiredEpochRef.current = playhead;
    setPlayhead(null);
    const video = videoRef.current;
    if (video) {
      video.pause();
    }
    codecFallbackRef.current = false;
    if (playbackRetryRef.current.timer) window.clearTimeout(playbackRetryRef.current.timer);
    playbackRetryRef.current = { attempts: 0, timer: null };
    const indexUrl = isAllCameras
      ? recordingGridDayUrl(dayStart, dayEnd, source)
      : recordingDayUrl(activeCameraId, dayStart, dayEnd, source);
    fetch(indexUrl, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Recording index failed (${response.status})`);
        return response.json();
      })
      .then((payload) => {
        const nextAvailableSources = payload.available_sources || [];
        const nextAvailability = payload.availability || payload.recordings || [];
        setAvailableSources(nextAvailableSources);
        if (!nextAvailability.length && source === "main" && nextAvailableSources.includes("live")) {
          codecFallbackRef.current = true;
          setPlaybackNotice(isAllCameras
            ? "No Main recordings exist for this day; using Sub."
            : "No Main recording exists for this day; using Sub.");
          setSource("live");
          return;
        }
        setRecordings(nextAvailability);
        setEvents(payload.incidents || payload.events || []);
        if (isAllCameras) gridRefreshCursorRef.current = Date.now() / 1000;
      })
      .catch((error) => {
        if (error.name !== "AbortError") {
          setPlaybackErrorStage("index");
          setPlaybackError(error.message || "Unable to load recordings");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => {
      controller.abort();
      if (playbackRetryRef.current.timer) window.clearTimeout(playbackRetryRef.current.timer);
      playbackRetryRef.current = { attempts: 0, timer: null };
    };
  }, [activeCameraId, isAllCameras, source, dayStart, dayEnd, recordingIndexRevision]);

  useEffect(() => {
    if (!activeCameraId || isAllCameras || date !== today) return undefined;
    let stopped = false;
    let inFlight = false;
    const refresh = async () => {
      if (stopped || inFlight) return;
      inFlight = true;
      try {
        const afterEpoch = Number.isFinite(latestAvailabilityRef.current)
          ? latestAvailabilityRef.current
          : dayStart;
        const response = await fetch(
          recordingUpdatesUrl(activeCameraId, dayStart, dayEnd, afterEpoch, source),
        );
        if (!response.ok) throw new Error(`Recording update failed (${response.status})`);
        const payload = await response.json();
        if (stopped) return;
        const additions = payload.availability || [];
        if (additions.length) {
          setRecordings((current) => mergeRecordingAvailability(current, additions));
        }
        const eventUpdates = payload.incidents || payload.events || [];
        if (eventUpdates.length) {
          setEvents((current) => mergeRecordingEvents(current, eventUpdates));
        }
      } catch {
        // The next poll retries without disrupting active playback.
      } finally {
        inFlight = false;
      }
    };
    const timer = window.setInterval(refresh, 10_000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [activeCameraId, isAllCameras, source, date, today, dayStart, dayEnd]);

  useEffect(() => {
    if (!isAllCameras || date !== today) return undefined;
    let stopped = false;
    let inFlight = false;
    const refresh = async () => {
      if (stopped || inFlight) return;
      inFlight = true;
      try {
        const requestStartedAt = Date.now() / 1000;
        const afterEpoch = Number.isFinite(gridRefreshCursorRef.current)
          ? gridRefreshCursorRef.current
          : Math.max(dayStart, requestStartedAt - 120);
        const response = await fetch(recordingGridUpdatesUrl(
          dayStart, dayEnd, afterEpoch, source,
        ));
        if (!response.ok) return;
        const payload = await response.json();
        if (stopped) return;
        const additions = payload.availability || payload.recordings || [];
        if (additions.length) {
          setRecordings((current) => mergeRecordingAvailability(current, additions));
        }
        const eventUpdates = payload.incidents || payload.events || [];
        if (eventUpdates.length) {
          setEvents((current) => mergeRecordingEvents(current, eventUpdates));
        }
        gridRefreshCursorRef.current = requestStartedAt;
      } catch {
        // The next refresh retries without interrupting synchronized playback.
      } finally {
        inFlight = false;
      }
    };
    const timer = window.setInterval(refresh, 10_000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [date, dayEnd, dayStart, isAllCameras, source, today]);

  useEffect(() => {
    if (!activeCameraId || isAllCameras || !playbackWindow) return undefined;
    const controller = new AbortController();
    const requestId = ++playbackRequestRef.current;
    const requestedWindow = { ...playbackWindow };
    fetch(
      recordingWindowUrl(activeCameraId, requestedWindow.start, requestedWindow.end, source),
      { signal: controller.signal },
    )
      .then(async (response) => {
        if (!response.ok) throw new Error(`Recording window failed (${response.status})`);
        return response.json();
      })
      .then((payload) => {
        if (controller.signal.aborted || requestId !== playbackRequestRef.current) return;
        const rows = payload.recordings || [];
        if (!rows.length) throw new Error("No recording segments exist in this window");
        setPlaybackDetail({
          start: Number(payload.start_epoch),
          end: Number(payload.end_epoch),
          rows,
          revision: requestId,
        });
      })
      .catch((error) => {
        if (error.name !== "AbortError" && requestId === playbackRequestRef.current) {
          pendingSeekEpochRef.current = null;
          pendingSeekModeRef.current = null;
          setPlaybackNotice("");
          setPlaybackErrorStage("window");
          setPlaybackError(error.message || "Unable to load recording window");
        }
      });
    return () => controller.abort();
  }, [activeCameraId, isAllCameras, source, playbackWindow?.start, playbackWindow?.end, playbackWindowRevision]);

  useEffect(() => {
    if (!timeline.length || Number.isFinite(playhead)) return;
    const retainedEpoch = desiredEpochRef.current;
    let initialEpoch = Number.isFinite(retainedEpoch) && retainedEpoch >= dayStart && retainedEpoch < dayEnd
      ? retainedEpoch
      : date === today ? Date.now() / 1000 : timeline[0].start_epoch;
    if (isAllCameras && !Number.isFinite(retainedEpoch) && date === today) {
      initialEpoch = recordingGridBestEpoch(timeline, initialEpoch) ?? initialEpoch;
    }
    playAt(initialEpoch, false);
  }, [date, dayEnd, dayStart, isAllCameras, playhead, timeline, today]);

  useEffect(() => {
    if (!Number.isFinite(followTarget) || !timeline.length) return;
    const nextRange = timeline.find((item) => item.end_epoch > followTarget);
    if (!nextRange) return;
    const nextEpoch = Math.max(followTarget, nextRange.start_epoch);
    playAt(nextEpoch, true);
  }, [followTarget, timeline]);

  useEffect(() => {
    if (!activeCameraId) return;
    const params = new URLSearchParams({ camera: activeCameraId, date, source });
    const retainedEpoch = desiredEpochRef.current;
    if (Number.isFinite(retainedEpoch) && retainedEpoch >= dayStart && retainedEpoch < dayEnd) {
      params.set("at", String(Math.round(retainedEpoch * 1000) / 1000));
    }
    window.history.replaceState(null, "", appUrl(`/recordings?${params.toString()}`));
  }, [activeCameraId, date, source, dayStart, dayEnd]);

  function handleRecordingReady(_player, video) {
    if (playbackRetryRef.current.timer) window.clearTimeout(playbackRetryRef.current.timer);
    playbackRetryRef.current = { attempts: 0, timer: null };
    const retained = Number.isFinite(pendingSeekEpochRef.current)
      ? pendingSeekEpochRef.current
      : desiredEpochRef.current;
    const target = Number.isFinite(retained) ? snapToRecording(retained) : snapToRecording(Date.now() / 1000);
    const mediaTime = epochToPlaybackMediaTime(target);
    if (Number.isFinite(mediaTime)) video.currentTime = mediaTime;
    if (Number.isFinite(target)) {
      desiredEpochRef.current = target;
      setPlayhead(target);
    }
    pendingSeekEpochRef.current = null;
    pendingSeekModeRef.current = null;
    setPlaybackError("");
    setPlaybackErrorStage("");
    setPlaybackNotice("");
    if (autoplayRef.current) requestRecordingPlay(video);
  }

  function handleRecordingTimeUpdate(event) {
    if (Number.isFinite(pendingSeekEpochRef.current)) return;
    const epoch = mediaTimeToEpoch(event.currentTarget.currentTime);
    if (!Number.isFinite(epoch)) return;
    desiredEpochRef.current = epoch;
    setPlayhead(epoch);
  }

  function handleRecordingSeeked(event) {
    if (pendingSeekModeRef.current === "local") {
      const epoch = mediaTimeToEpoch(event.currentTarget.currentTime);
      if (Number.isFinite(epoch)) {
        desiredEpochRef.current = epoch;
        setPlayhead(epoch);
      }
      pendingSeekEpochRef.current = null;
      pendingSeekModeRef.current = null;
    }
    setPlaybackNotice("");
  }

  function handleRecordingError(error) {
    const detail = describePlaybackError(error);
    console.warn("Recording playback error", { camera: activeCameraId, source, detail, error });
    if (source === "main" && availableSources.includes("live") && !codecFallbackRef.current && isUnsupportedPlaybackError(error)) {
      codecFallbackRef.current = true;
      setPlaybackNotice(`Main stream is not supported by this browser; using Sub. (${detail})`);
      setSource("live");
      return;
    }
    if (playbackRetryRef.current.timer) return;
    if (playbackRetryRef.current.attempts < 4 && manifestUrl) {
      playbackRetryRef.current.attempts += 1;
      const attempt = playbackRetryRef.current.attempts;
      const delay = Math.min(5_000, 750 * (2 ** (attempt - 1)));
      setPlaybackError("");
      setPlaybackErrorStage("");
      setPlaybackNotice(`Playback interrupted (${detail}). Retrying ${attempt}/4...`);
      playbackRetryRef.current.timer = window.setTimeout(() => {
        playbackRetryRef.current.timer = null;
        setManifestRetryToken((token) => token + 1);
      }, delay);
      return;
    }
    setPlaybackNotice("");
    setPlaybackErrorStage("media");
    setPlaybackError(`Playback failed: ${detail}`);
  }

  function retryRecordingPlayback() {
    if (playbackRetryRef.current.timer) window.clearTimeout(playbackRetryRef.current.timer);
    playbackRetryRef.current = { attempts: 0, timer: null };
    setPlaybackError("");
    setPlaybackErrorStage("");
    setPlaybackNotice("Retrying recording...");
    if (playbackErrorStage === "window" && playbackWindow) {
      requestPlaybackWindow(playbackWindow);
      return;
    }
    if (playbackErrorStage === "index") {
      if (!activeCameraId) {
        window.location.reload();
        return;
      }
      setRecordingIndexRevision((revision) => revision + 1);
      return;
    }
    if (manifestUrl) {
      setManifestRetryToken((token) => token + 1);
      return;
    }
    const target = Number.isFinite(desiredEpochRef.current) ? desiredEpochRef.current : Date.now() / 1000;
    requestPlaybackWindow(windowAround(target));
  }

  function continueRecordingPlayback() {
    const lastSegment = playbackTimeline[playbackTimeline.length - 1];
    if (!lastSegment) return;
    const nextEpoch = lastSegment.end_epoch + 0.01;
    const nextRange = timeline.find((item) => item.end_epoch > nextEpoch);
    if (nextRange) {
      playAt(Math.max(nextEpoch, nextRange.start_epoch), true);
      return;
    }
    if (date === today) {
      desiredEpochRef.current = nextEpoch;
      autoplayRef.current = true;
      setFollowTarget(nextEpoch);
      setPlaybackNotice("Waiting for the next recording...");
    }
  }

  function changeDate(next) {
    setDate(next > today ? today : next);
  }

  function toggleExport() {
    if (exportJob && ["queued", "running", "cancelling"].includes(exportJob.status)) return;
    if (exportRange) {
      setExportRange(null);
      setExportJob(null);
      setExportError("");
      return;
    }
    const center = Number.isFinite(playhead) ? playhead : timeline[0]?.start_epoch ?? dayStart;
    const start = Math.max(dayStart, center - 5 * 60);
    const end = Math.min(dayEnd, Math.max(start + 30, center + 5 * 60));
    setExportRange({ start, end });
    setExportJob(null);
    setExportError("");
    setExportLabel("");
  }

  const exportActive = Boolean(exportJob && ["queued", "running", "cancelling"].includes(exportJob.status));

  async function startExport() {
    if (!activeCameraId || !exportRange || exportSubmitting) return;
    setExportSubmitting(true);
    setExportError("");
    try {
      const response = await fetch(appUrl("/api/exports"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kind: exportKind,
          camera_id: activeCameraId,
          source,
          start_epoch: exportRange.start,
          end_epoch: exportRange.end,
          sample_interval_seconds: exportOptions.interval,
          output_fps: exportOptions.fps,
          height: exportKind === "timelapse" ? exportOptions.timelapseHeight : exportOptions.clipHeight,
          label: exportLabel.trim(),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Export failed (${response.status})`);
      setExportJob(payload);
    } catch (error) {
      setExportError(error.message || "Unable to start export");
    } finally {
      setExportSubmitting(false);
    }
  }

  async function cancelExport() {
    if (!exportJob?.id) return;
    try {
      const response = await fetch(appUrl(`/api/exports/${exportJob.id}`), { method: "DELETE" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Cancel failed (${response.status})`);
      setExportJob(payload.deleted ? null : payload);
    } catch (error) {
      setExportError(error.message || "Unable to cancel export");
    }
  }

  return (
    <main className="recordings-v2-page">
      <nav className="recordings-tabs"><RecordingSectionSwitcher mode="history" cameraId={activeCameraId} /></nav>
      <RecordingCameraRail subtitle="Choose recording source">
        <button type="button" className={isAllCameras ? "active" : ""} onClick={() => { setCameraId(ALL_RECORDING_CAMERAS_ID); setSource("live"); }}>
          <Grid2X2 size={16} />
          <span>All Cameras</span>
          <i className={cameras.some((camera) => camera.recording || camera.sub_recording) ? "online" : ""} />
        </button>
        {cameras.map((camera) => (
          <button key={camera.id} type="button" className={camera.id === activeCameraId ? "active" : ""} onClick={() => setCameraId(camera.id)}>
            <Camera size={16} />
            <span>{camera.name}</span>
            <i className={(source === "main" ? camera.recording : camera.sub_recording) ? "online" : ""} />
          </button>
        ))}
      </RecordingCameraRail>

      <section className="recordings-v2-workspace">
        <div className={`recordings-v2-player${isAllCameras ? " all-camera-grid" : ""}`}>
          {isAllCameras && Number.isFinite(playhead) ? <RecordingCameraGrid
            cameras={cameras}
            source={source}
            epoch={playhead}
            playing={gridPlaying}
            onSelect={(selectedCameraId) => setCameraId(selectedCameraId)}
          /> : null}
          {!isAllCameras && manifestUrl ? (
            <ShakaVideo
              ref={videoRef}
              src={manifestUrl}
              mimeType="application/vnd.apple.mpegurl"
              startTime={manifestStartTime}
              bufferingGoal={8}
              controls
              playsInline
              preload="auto"
              onReady={handleRecordingReady}
              onError={handleRecordingError}
              onTimeUpdate={handleRecordingTimeUpdate}
              onSeeked={handleRecordingSeeked}
              onEnded={continueRecordingPlayback}
              onPlay={() => {
                autoplayRef.current = true;
                setPlaybackBlocked(false);
                setPlaybackNotice("");
              }}
              onPause={(event) => {
                if (!event.currentTarget.ended && !Number.isFinite(pendingSeekEpochRef.current)) {
                  autoplayRef.current = false;
                }
              }}
            />
          ) : null}
          {loading ? <div className="recordings-v2-message"><Film size={28} />Loading recordings</div> : null}
          {!loading && !timeline.length ? <div className="recordings-v2-message"><Film size={28} />No recordings on this day</div> : null}
          {playbackError ? <div className="recordings-v2-error"><span>{playbackError}</span><button type="button" onClick={retryRecordingPlayback}><RefreshCcw size={14} />Retry</button></div> : null}
          {playbackNotice && !playbackError ? <div className="recordings-v2-notice">{playbackNotice}</div> : null}
          {!isAllCameras && playbackBlocked && !playbackError ? (
            <button type="button" className="recordings-v2-play" onClick={() => requestRecordingPlay(videoRef.current)}>
              <Play size={22} fill="currentColor" />
              Play recording
            </button>
          ) : null}
          <div className="recordings-v2-player-source" aria-label="Recording stream">
            <button type="button" className={source === "main" ? "active" : ""} onClick={() => setSource("main")} disabled={availableSources.length > 0 && !availableSources.includes("main")}>Main</button>
            <button type="button" className={source === "live" ? "active" : ""} onClick={() => setSource("live")} disabled={availableSources.length > 0 && !availableSources.includes("live")}>Sub</button>
          </div>
          {isAllCameras && Number.isFinite(playhead) ? <div className="recording-grid-controls">
            <button type="button" onClick={() => playAt(playhead - 10, gridPlaying)} aria-label="Back 10 seconds"><SkipBack size={16} /></button>
            <button type="button" className="primary" onClick={() => setGridPlaying((current) => !current)}>{gridPlaying ? <Pause size={17} /> : <Play size={17} fill="currentColor" />}{gridPlaying ? "Pause" : "Play"}</button>
            <button type="button" onClick={() => playAt(playhead + 10, gridPlaying)} aria-label="Forward 10 seconds"><SkipForward size={16} /></button>
            <time>{formatDateTime(playhead, timeZone)}</time>
          </div> : null}
        </div>

        <div className="recordings-v2-controls">
          <RecordingTimeline
            cameraId={isAllCameras ? "" : activeCameraId}
            source={source}
            previewManifestUrl={manifestUrl}
            previewStartTime={manifestStartTime}
            previewTimeline={playbackTimeline}
            startEpoch={dayStart}
            endEpoch={dayEnd}
            recordings={timeline}
            events={filteredEvents}
            playhead={playhead ?? dayStart}
            timeZone={timeZone}
            onSeek={(epoch) => playAt(epoch, true)}
            exportRange={isAllCameras ? null : exportRange}
            onExportRangeChange={isAllCameras || exportJob ? null : setExportRange}
          />
          {exportRange ? (
            <section className="recordings-v2-export-panel" aria-label="Export recording">
              <div className="recordings-v2-export-kind" role="group" aria-label="Export type">
                <button type="button" className={exportKind === "recording" ? "active" : ""} onClick={() => setExportKind("recording")} disabled={Boolean(exportJob)}>Video clip</button>
                <button type="button" className={exportKind === "timelapse" ? "active" : ""} onClick={() => setExportKind("timelapse")} disabled={Boolean(exportJob)}>Timelapse</button>
              </div>
              <div className="recordings-v2-export-range-label">
                <span><b>Start</b>{formatDateTime(exportRange.start, timeZone)}</span>
                <span><b>End</b>{formatDateTime(exportRange.end, timeZone)}</span>
                <span><b>Length</b>{formatDuration(exportRange.end - exportRange.start)}</span>
              </div>
              <div className="recordings-v2-export-options">
                <label className="export-name-field"><span>Name</span><input value={exportLabel} onChange={(event) => setExportLabel(event.target.value)} placeholder="Optional export name" maxLength="120" disabled={Boolean(exportJob)} /></label>
                {exportKind === "timelapse" ? <>
                  <label><span>Capture every</span><select value={exportOptions.interval} onChange={(event) => setExportOptions((current) => ({ ...current, interval: Number(event.target.value) }))} disabled={Boolean(exportJob)}><option value="5">5 sec</option><option value="10">10 sec</option><option value="30">30 sec</option><option value="60">1 min</option><option value="300">5 min</option></select></label>
                  <label><span>Playback</span><select value={exportOptions.fps} onChange={(event) => setExportOptions((current) => ({ ...current, fps: Number(event.target.value) }))} disabled={Boolean(exportJob)}><option value="24">24 FPS</option><option value="30">30 FPS</option><option value="60">60 FPS</option></select></label>
                </> : null}
                <label><span>Resolution</span><select value={exportKind === "timelapse" ? exportOptions.timelapseHeight : exportOptions.clipHeight} onChange={(event) => setExportOptions((current) => ({ ...current, [exportKind === "timelapse" ? "timelapseHeight" : "clipHeight"]: Number(event.target.value) }))} disabled={Boolean(exportJob)}><option value="0">Original</option><option value="2160">2160p</option><option value="1440">1440p</option><option value="1080">1080p</option><option value="720">720p</option><option value="480">480p</option></select></label>
              </div>
              {exportKind === "recording" ? <p>Creates one broadly compatible H.264 MP4. Width follows the camera aspect ratio; gaps are skipped and source-format changes are joined automatically.</p> : <p>Resolution is the output height; width follows the camera aspect ratio.</p>}
              {exportJob ? (
                <div className={`recordings-v2-export-status ${exportJob.status}`}>
                  <span><b>{exportJob.phase || exportJob.status}</b><small>{Math.round(Number(exportJob.progress) || 0)}%</small></span>
                  <i><b style={{ width: `${Math.max(0, Math.min(100, Number(exportJob.progress) || 0))}%` }} /></i>
                  {exportJob.error ? <em>{exportJob.error}</em> : null}
                </div>
              ) : null}
              {exportError ? <div className="recordings-v2-export-error"><CircleAlert size={15} />{exportError}</div> : null}
              <div className="recordings-v2-export-actions">
                {exportJob?.status === "completed" && exportJob.download_url ? <a className="nav-button" href={exportJob.download_url}><Download size={15} />Download</a> : null}
                {exportJob?.status === "completed" ? <a className="nav-button" href={appUrl(`/recordings/exports?camera=${encodeURIComponent(activeCameraId)}`)}><Film size={15} />Export Center</a> : null}
                {exportJob && ["queued", "running", "cancelling"].includes(exportJob.status) ? <button type="button" onClick={cancelExport} disabled={exportJob.status === "cancelling"}><X size={15} />{exportJob.status === "cancelling" ? "Cancelling" : "Cancel"}</button> : null}
                {!exportJob ? <button type="button" className="primary" onClick={startExport} disabled={exportSubmitting}><Download size={15} />{exportSubmitting ? "Starting..." : `Start ${exportKind === "timelapse" ? "timelapse" : "export"}`}</button> : null}
                {exportJob && ["completed", "failed", "cancelled"].includes(exportJob.status) ? <button type="button" onClick={() => { setExportJob(null); setExportError(""); setExportLabel(""); }}>New export</button> : null}
              </div>
            </section>
          ) : null}
        </div>

        <div className="recordings-v2-incidents">
          <div className="recordings-v2-incidents-toolbar">
            <div className="recordings-v2-incidents-tools">
              <div className="recordings-v2-date">
                <button type="button" onClick={() => changeDate(addDaysToDateKey(date, -1))} aria-label="Previous day"><SkipBack size={15} /></button>
                <input type="date" value={date} max={today} onChange={(event) => changeDate(event.target.value || today)} aria-label="Recording day" />
                <button type="button" onClick={() => changeDate(addDaysToDateKey(date, 1))} disabled={date >= today} aria-label="Next day"><SkipForward size={15} /></button>
                <button type="button" onClick={() => changeDate(today)} disabled={date === today}>Today</button>
              </div>
              <div className="recordings-v2-event-filter" aria-label="Recording incident type">
                <button type="button" className={eventFilter === "object" ? "active" : ""} onClick={() => setEventFilter("object")}><CircleDot size={14} />Object</button>
                <button type="button" className={eventFilter === "motion" ? "active" : ""} onClick={() => setEventFilter("motion")}><Radar size={14} />Motion</button>
              </div>
              <label className="recordings-v2-range">
                <span>Range</span>
                <select value={incidentRangeHours} onChange={(event) => setIncidentRangeHours(Number(event.target.value))} aria-label="Incident thumbnail time range">
                  <option value="1">1 hour</option>
                  <option value="2">2 hours</option>
                  <option value="4">4 hours</option>
                  <option value="8">8 hours</option>
                  <option value="12">12 hours</option>
                  <option value="24">Full day</option>
                </select>
              </label>
              <button
                type="button"
                className={`recordings-v2-export-toggle${exportRange ? " active" : ""}`}
                onClick={toggleExport}
                disabled={isAllCameras || !timeline.length || exportActive}
              >
                <Download size={15} />{isAllCameras ? "Select camera to export" : exportActive ? "Export running" : exportRange ? "Close export" : "Export"}
              </button>
            </div>
            <span>{nearbyEvents.length.toLocaleString()} of {filteredEvents.length.toLocaleString()} {eventFilter} incident{filteredEvents.length === 1 ? "" : "s"} · {incidentRangeHours >= 24 ? "full day" : `${incidentRangeHours} hour${incidentRangeHours === 1 ? "" : "s"} around current time`}</span>
          </div>
          <div className="recordings-v2-events">
            {nearbyEvents.length ? nearbyEvents.map((event) => (
              <button
                key={event.id}
                type="button"
                className={event.has_objects ? "object" : "motion"}
                onClick={() => playAt(event.incident_epoch, true)}
                title={`${formatDateTime(event.incident_epoch, timeZone)} · ${event.labels?.length ? event.labels.join(", ") : "Motion"}`}
              >
                <span className="recordings-v2-event-image">
                  <Radar size={20} />
                  {event.snapshot_path ? <img src={eventThumbnailUrl(event, 240, 72)} alt="" loading="lazy" decoding="async" onError={(loadEvent) => { loadEvent.currentTarget.hidden = true; }} /> : null}
                </span>
                <span className="recordings-v2-event-caption">
                  <time>{formatTimeOnly(event.incident_epoch, timeZone).replace(/:\d{2}(?=\s)/, "")}</time>
                  <b>{isAllCameras ? `${cameras.find((camera) => camera.id === event.camera_id)?.name || event.camera_id} · ` : ""}{event.labels?.length ? event.labels.join(", ") : "Motion"}</b>
                </span>
              </button>
            )) : <div className="recordings-v2-no-events"><Radar size={17} />No {eventFilter} incidents {incidentRangeHours >= 24 ? "on this day" : `within ${incidentRangeHours === 1 ? "30 minutes" : `${incidentRangeHours / 2} hours`} of this time`}</div>}
          </div>
        </div>
      </section>
    </main>
  );
}

function exportStatusLabel(status) {
  return ({
    queued: "Queued",
    running: "Creating",
    cancelling: "Cancelling",
    completed: "Ready",
    failed: "Failed",
    cancelled: "Cancelled",
  })[status] || String(status || "Unknown");
}

function ExportCenterPage({ timeZone, onAssistantContextChange }) {
  const initialQuery = useMemo(() => new URLSearchParams(window.location.search), []);
  const [cameras, setCameras] = useState([]);
  const [cameraId, setCameraId] = useState(initialQuery.get("camera") || "");
  const [kind, setKind] = useState("all");
  const [status, setStatus] = useState("all");
  const [protectedOnly, setProtectedOnly] = useState(false);
  const [exportsList, setExportsList] = useState([]);
  const [total, setTotal] = useState(0);
  const [summary, setSummary] = useState({});
  const [limit, setLimit] = useState(50);
  const [selectedId, setSelectedId] = useState("");
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState([]);
  const [editLabel, setEditLabel] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionBusy, setActionBusy] = useState("");
  const [revision, setRevision] = useState(0);

  const filteredExports = exportsList;
  const selected = filteredExports.find((item) => item.id === selectedId) || filteredExports[0] || null;
  const activeCount = Math.max(
    Number(summary.active) || 0,
    exportsList.filter((item) => ["queued", "running", "cancelling"].includes(item.status)).length,
  );
  const readyBytes = exportsList
    .filter((item) => item.status === "completed")
    .reduce((sum, item) => sum + (Number(item.size_bytes) || 0), 0);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/cameras", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Camera status failed (${response.status})`);
        return response.json();
      })
      .then(setCameras)
      .catch((loadError) => {
        if (loadError.name !== "AbortError") setError(loadError.message || "Unable to load cameras");
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams({ limit: String(limit) });
    if (cameraId) params.set("camera_id", cameraId);
    if (kind !== "all") params.set("kind", kind);
    if (status !== "all") params.set("status", status);
    if (protectedOnly) params.set("protected", "true");
    setLoading(true);
    setError("");
    fetch(`/api/exports?${params.toString()}`, { signal: controller.signal })
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.detail || `Export list failed (${response.status})`);
        return payload;
      })
      .then((payload) => {
        setExportsList(payload.exports || []);
        setTotal(Number(payload.total) || 0);
      })
      .catch((loadError) => {
        if (loadError.name !== "AbortError") setError(loadError.message || "Unable to load exports");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [cameraId, kind, limit, protectedOnly, revision, status]);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/exports/summary", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Export summary failed (${response.status})`);
        return response.json();
      })
      .then(setSummary)
      .catch((loadError) => {
        if (loadError.name !== "AbortError") setError(loadError.message || "Unable to load export summary");
      });
    return () => controller.abort();
  }, [revision]);

  useEffect(() => {
    setLimit(50);
    setSelectedIds([]);
    setSelectionMode(false);
  }, [cameraId, kind, protectedOnly, status]);

  useEffect(() => {
    const interval = window.setInterval(() => setRevision((current) => current + 1), activeCount ? 2_000 : 15_000);
    return () => window.clearInterval(interval);
  }, [activeCount]);

  useEffect(() => {
    if (!selectedId || !filteredExports.some((item) => item.id === selectedId)) {
      setSelectedId(filteredExports[0]?.id || "");
    }
  }, [filteredExports, selectedId]);

  useEffect(() => {
    setEditLabel(selected?.label || "");
  }, [selected?.id, selected?.label]);

  useEffect(() => {
    const params = new URLSearchParams();
    if (cameraId) params.set("camera", cameraId);
    window.history.replaceState(null, "", appUrl(`/recordings/exports${params.size ? `?${params.toString()}` : ""}`));
  }, [cameraId]);

  useEffect(() => {
    onAssistantContextChange?.({
      page: "recordings",
      view: "exports",
      camera_id: cameraId || selected?.camera_id || "",
      export_id: selected?.id || "",
      filters: { kind, status, protected: protectedOnly },
    });
  }, [cameraId, kind, onAssistantContextChange, protectedOnly, selected?.camera_id, selected?.id, status]);

  async function setExportProtection(item, nextProtected) {
    if (!item?.id || actionBusy) return;
    setActionBusy(item.id);
    setError("");
    try {
      const response = await fetch(`/api/exports/${encodeURIComponent(item.id)}/protection`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ protected: nextProtected }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Protection update failed (${response.status})`);
      setExportsList((current) => current.map((candidate) => candidate.id === payload.id ? payload : candidate));
      setRevision((current) => current + 1);
    } catch (actionError) {
      setError(actionError.message || "Unable to update export protection");
    } finally {
      setActionBusy("");
    }
  }

  async function saveExportLabel(item) {
    if (!item?.id || actionBusy) return;
    setActionBusy(item.id);
    setError("");
    try {
      const response = await fetch(`/api/exports/${encodeURIComponent(item.id)}/metadata`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: editLabel.trim() }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Rename failed (${response.status})`);
      setExportsList((current) => current.map((candidate) => candidate.id === payload.id ? payload : candidate));
    } catch (actionError) {
      setError(actionError.message || "Unable to rename export");
    } finally {
      setActionBusy("");
    }
  }

  function toggleExportSelection(itemId) {
    setSelectedIds((current) => current.includes(itemId)
      ? current.filter((id) => id !== itemId)
      : [...current, itemId]);
  }

  async function runBatchAction(action) {
    if (!selectedIds.length || actionBusy) return;
    if (action === "delete" && !window.confirm(`Permanently delete ${selectedIds.length} selected export${selectedIds.length === 1 ? "" : "s"}? Protected selections will also be deleted.`)) return;
    setActionBusy("batch");
    setError("");
    try {
      const response = await fetch("/api/exports/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: selectedIds, action }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Batch action failed (${response.status})`);
      if (payload.errors?.length) {
        setError(`${payload.errors.length} export${payload.errors.length === 1 ? "" : "s"} could not be updated`);
      }
      setSelectedIds([]);
      setSelectionMode(false);
      setSelectedId("");
      setRevision((current) => current + 1);
    } catch (actionError) {
      setError(actionError.message || "Unable to update selected exports");
    } finally {
      setActionBusy("");
    }
  }

  async function removeExport(item) {
    if (!item?.id || actionBusy) return;
    const active = ["queued", "running", "cancelling"].includes(item.status);
    const prompt = active
      ? `Cancel this ${item.kind === "timelapse" ? "timelapse" : "video export"}?`
      : `Permanently delete ${item.output_name || "this export"}?${item.protected ? " It is currently protected." : ""}`;
    if (!window.confirm(prompt)) return;
    setActionBusy(item.id);
    setError("");
    try {
      const force = item.protected && !active ? "?force=true" : "";
      const response = await fetch(`/api/exports/${encodeURIComponent(item.id)}${force}`, { method: "DELETE" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Delete failed (${response.status})`);
      if (payload.deleted) {
        setExportsList((current) => current.filter((candidate) => candidate.id !== item.id));
        setSelectedId("");
      } else {
        setExportsList((current) => current.map((candidate) => candidate.id === payload.id ? payload : candidate));
      }
      setRevision((current) => current + 1);
    } catch (actionError) {
      setError(actionError.message || "Unable to remove export");
    } finally {
      setActionBusy("");
    }
  }

  return (
    <main className="recordings-v2-page export-center-page">
      <nav className="recordings-tabs"><RecordingSectionSwitcher mode="exports" cameraId={cameraId} /></nav>
      <RecordingCameraRail subtitle="Filter saved exports">
        <button type="button" className={!cameraId ? "active" : ""} onClick={() => setCameraId("")}>
          <Film size={16} /><span>All exports</span><i className={Number(summary.total) ? "online" : ""} />
        </button>
        {cameras.map((camera) => (
          <button key={camera.id} type="button" className={camera.id === cameraId ? "active" : ""} onClick={() => setCameraId(camera.id)}>
            <Camera size={16} /><span>{camera.name}</span><i className={(camera.recording || camera.sub_recording) ? "online" : ""} />
          </button>
        ))}
      </RecordingCameraRail>

      <section className="export-center-workspace">
        <header className="export-center-toolbar">
          <div className="export-center-filters">
            <label>Type<select value={kind} onChange={(event) => setKind(event.target.value)}><option value="all">All exports</option><option value="recording">Video clips</option><option value="timelapse">Timelapses</option></select></label>
            <label>Status<select value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">Any status</option><option value="completed">Ready</option><option value="active">In progress</option><option value="failed">Failed</option><option value="cancelled">Cancelled</option></select></label>
            <button type="button" className={protectedOnly ? "active" : ""} onClick={() => setProtectedOnly((current) => !current)}><ShieldCheck size={15} />Protected</button>
            <button type="button" className={selectionMode ? "active" : ""} onClick={() => { setSelectionMode((current) => !current); setSelectedIds([]); }}><Check size={15} />Select</button>
            <button type="button" onClick={() => setRevision((current) => current + 1)} disabled={loading}><RefreshCcw className={loading ? "spin" : ""} size={15} />Refresh</button>
          </div>
          <span>{filteredExports.length.toLocaleString()} shown · {total.toLocaleString()} matching · {Number(summary.total || 0).toLocaleString()} overall · {formatBytes(Number(summary.bytes) || readyBytes)} / {formatBytes(Number(summary.max_storage_bytes))}</span>
        </header>

        <section className="export-center-review">
          <div className="export-center-player">
            {selected?.status === "completed" && selected.media_url ? <video key={selected.id} src={selected.media_url} controls playsInline preload="metadata" /> : null}
            {!selected && !loading ? <div className="export-center-empty"><Film size={30} /><strong>No exports match these filters</strong><span>Create a clip or timelapse from recording history or ask the SurvNG Assistant.</span></div> : null}
            {selected && selected.status !== "completed" ? <div className={`export-center-job-state ${selected.status}`}><RefreshCcw className={["queued", "running", "cancelling"].includes(selected.status) ? "spin" : ""} size={30} /><strong>{selected.phase || exportStatusLabel(selected.status)}</strong><span>{selected.error || `${Math.round(Number(selected.progress) || 0)}% complete`}</span></div> : null}
          </div>
          <aside className="export-center-details">
            {selected ? <>
              <header><div><strong>{selected.label || (selected.kind === "timelapse" ? "Timelapse" : "Video clip")}</strong><span className={`export-status ${selected.status}`}>{exportStatusLabel(selected.status)}</span></div><small>{selected.output_name || `${selected.camera_id} export`}</small></header>
              <div className="export-center-rename"><input value={editLabel} onChange={(event) => setEditLabel(event.target.value)} placeholder="Add a useful name" maxLength="120" /><button type="button" onClick={() => saveExportLabel(selected)} disabled={actionBusy === selected.id || editLabel.trim() === (selected.label || "")}><Save size={14} />Save</button></div>
              <dl>
                <div><dt>Camera</dt><dd>{cameras.find((camera) => camera.id === selected.camera_id)?.name || selected.camera_id}</dd></div>
                <div><dt>Stream</dt><dd>{selected.source === "live" ? "Sub" : "Main"}</dd></div>
                <div><dt>From</dt><dd>{formatDateTime(Number(selected.start_epoch), timeZone)}</dd></div>
                <div><dt>To</dt><dd>{formatDateTime(Number(selected.end_epoch), timeZone)}</dd></div>
                <div><dt>Duration</dt><dd>{formatDuration(Number(selected.end_epoch) - Number(selected.start_epoch))}</dd></div>
                <div><dt>Resolution</dt><dd>{Number(selected.options?.height) > 0 ? `${Number(selected.options.height)}p` : Number(selected.options?.width) > 0 ? `${Number(selected.options.width)}px wide (legacy)` : "Original"}</dd></div>
                {selected.kind === "timelapse" ? <div><dt>Timelapse</dt><dd>Every {formatDuration(Number(selected.options?.sample_interval_seconds) || 30)} · {Number(selected.options?.output_fps) || 30} FPS</dd></div> : null}
                <div><dt>File size</dt><dd>{formatBytes(Number(selected.size_bytes))}</dd></div>
                <div><dt>Created</dt><dd>{formatDateTime(selected.created_at, timeZone)}</dd></div>
                <div><dt>Created by</dt><dd>{selected.origin === "assistant" ? "SurvNG Assistant" : "Recording viewer"}</dd></div>
                <div><dt>Retention</dt><dd>{selected.protected ? "Protected" : selected.expires_at ? `Until ${formatDateTime(selected.expires_at, timeZone)}` : "While active"}</dd></div>
              </dl>
              <div className="export-center-actions">
                {selected.status === "completed" && selected.download_url ? <a className="primary" href={selected.download_url}><Download size={15} />Download</a> : null}
                {selected.status === "completed" ? <button type="button" onClick={() => setExportProtection(selected, !selected.protected)} disabled={actionBusy === selected.id}><ShieldCheck size={15} />{selected.protected ? "Unprotect" : "Protect"}</button> : null}
                <button type="button" className="danger" onClick={() => removeExport(selected)} disabled={actionBusy === selected.id || selected.status === "cancelling"}><Trash2 size={15} />{["queued", "running"].includes(selected.status) ? "Cancel" : "Delete"}</button>
              </div>
            </> : null}
          </aside>
        </section>

        {error ? <div className="export-center-error"><CircleAlert size={15} />{error}</div> : null}
        {selectionMode ? <section className="export-center-batch" aria-label="Selected export actions"><strong>{selectedIds.length} selected</strong><button type="button" onClick={() => runBatchAction("protect")} disabled={!selectedIds.length || Boolean(actionBusy)}><ShieldCheck size={14} />Protect</button><button type="button" onClick={() => runBatchAction("unprotect")} disabled={!selectedIds.length || Boolean(actionBusy)}><ShieldCheck size={14} />Unprotect</button><button type="button" className="danger" onClick={() => runBatchAction("delete")} disabled={!selectedIds.length || Boolean(actionBusy)}><Trash2 size={14} />Delete</button><button type="button" onClick={() => { setSelectedIds([]); setSelectionMode(false); }}>Done</button></section> : null}
        <section className="export-center-library" aria-label="Saved exports">
          {filteredExports.map((item) => (
            <button key={item.id} type="button" className={`${item.id === selected?.id && !selectionMode ? "active" : ""} ${selectedIds.includes(item.id) ? "selected" : ""} ${item.status}`} onClick={() => selectionMode ? toggleExportSelection(item.id) : setSelectedId(item.id)} aria-pressed={selectionMode ? selectedIds.includes(item.id) : undefined}>
              <span className="export-center-card-icon">{item.kind === "timelapse" ? <Clock3 size={23} /> : <Film size={23} />}</span>
              <span className="export-center-card-copy"><strong>{item.label || cameras.find((camera) => camera.id === item.camera_id)?.name || item.camera_id}</strong><small>{item.label ? `${cameras.find((camera) => camera.id === item.camera_id)?.name || item.camera_id} · ` : ""}{item.kind === "timelapse" ? "Timelapse" : "Video clip"} · {formatDateTime(Number(item.start_epoch), timeZone)}</small><small>{formatDuration(Number(item.end_epoch) - Number(item.start_epoch))} · {formatBytes(Number(item.size_bytes))}</small></span>
              <span className={`export-center-card-status ${item.status}`}>{item.protected ? <ShieldCheck size={13} /> : null}{exportStatusLabel(item.status)}</span>
            </button>
          ))}
          {loading && !filteredExports.length ? <div className="export-center-loading"><RefreshCcw className="spin" size={20} />Loading exports</div> : null}
          {!loading && exportsList.length < total && limit < 1000 ? <button type="button" className="export-center-load-more" onClick={() => setLimit((current) => Math.min(1000, current + 50))}><Plus size={18} /><span>Load older</span><small>{(total - exportsList.length).toLocaleString()} remaining</small></button> : null}
        </section>
      </section>
    </main>
  );
}

function RecordingTimeline({ cameraId, source, previewManifestUrl, previewStartTime, previewTimeline, startEpoch, endEpoch, recordings, events, playhead, timeZone, onSeek, exportRange, onExportRangeChange }) {
  const duration = Math.max(1, endEpoch - startEpoch);
  const offset = Math.max(0, Math.min(duration, playhead - startEpoch));
  const [draft, setDraft] = useState(offset);
  const [scrubbing, setScrubbing] = useState(false);
  const [preview, setPreview] = useState({ epoch: null, url: "", loading: false, gap: false, unavailable: false, mode: "jpeg" });
  const [localPreviewEnabled, setLocalPreviewEnabled] = useState(false);
  const [localPreviewReady, setLocalPreviewReady] = useState(false);
  const [localPreviewFrameReady, setLocalPreviewFrameReady] = useState(false);
  const draftRef = useRef(offset);
  const dragRef = useRef(null);
  const previewTimerRef = useRef(null);
  const previewAbortRef = useRef(null);
  const previewLastRequestRef = useRef({ epoch: null, at: 0 });
  const previewUrlRef = useRef("");
  const previewInFlightRef = useRef(false);
  const previewPendingRef = useRef(null);
  const previewHideTimerRef = useRef(null);
  const previewGenerationRef = useRef(0);
  const localPreviewRef = useRef(null);
  const localPreviewCanvasRef = useRef(null);
  const localPreviewTargetRef = useRef(null);
  const exportDragRef = useRef(null);
  const ticks = useMemo(() => {
    const formatter = new Intl.DateTimeFormat("en-US", {
      timeZone,
      hour: "numeric",
      minute: "2-digit",
    });
    const partFormatter = new Intl.DateTimeFormat("en-US", {
      timeZone,
      hourCycle: "h23",
      hour: "2-digit",
      minute: "2-digit",
    });
    const items = [];
    const first = Math.ceil(startEpoch / (15 * 60)) * 15 * 60;
    for (let epoch = first; epoch < endEpoch; epoch += 15 * 60) {
      const parts = partFormatter.formatToParts(new Date(epoch * 1000));
      const minute = Number(parts.find((item) => item.type === "minute")?.value || 0);
      const kind = minute === 0 ? "hour" : minute === 30 ? "half" : "quarter";
      items.push({
        epoch,
        kind,
        label: kind === "hour" ? formatter.format(new Date(epoch * 1000)).replace(":00", "") : "",
      });
    }
    return items;
  }, [endEpoch, startEpoch, timeZone]);
  const eventMarkers = useMemo(() => (events || []).map((event) => {
    const start = recordingIncidentEpoch(event);
    const end = recordingIncidentEndEpoch(event);
    if (!Number.isFinite(start) || start >= endEpoch || (Number.isFinite(end) && end < startEpoch)) return null;
    const boundedStart = Math.max(startEpoch, start);
    const boundedEnd = Math.min(endEpoch, Math.max(boundedStart + 1, Number.isFinite(end) ? end : start + 1));
    return {
      id: event.id,
      hasObjects: Boolean(event.has_objects),
      left: ((boundedStart - startEpoch) / duration) * 100,
      width: ((boundedEnd - boundedStart) / duration) * 100,
    };
  }).filter(Boolean), [duration, endEpoch, events, startEpoch]);
  useEffect(() => {
    if (dragRef.current) return;
    draftRef.current = offset;
    setDraft(offset);
  }, [offset]);
  useEffect(() => () => {
    previewGenerationRef.current += 1;
    if (previewTimerRef.current) window.clearTimeout(previewTimerRef.current);
    if (previewHideTimerRef.current) window.clearTimeout(previewHideTimerRef.current);
    previewAbortRef.current?.abort();
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);
  useEffect(() => {
    previewGenerationRef.current += 1;
    previewLastRequestRef.current = { epoch: null, at: 0 };
    previewPendingRef.current = null;
    previewInFlightRef.current = false;
    if (previewTimerRef.current) window.clearTimeout(previewTimerRef.current);
    if (previewHideTimerRef.current) window.clearTimeout(previewHideTimerRef.current);
    previewAbortRef.current?.abort();
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = "";
    setScrubbing(false);
    setLocalPreviewEnabled(false);
    setLocalPreviewReady(false);
    setLocalPreviewFrameReady(false);
    localPreviewTargetRef.current = null;
    setPreview({ epoch: null, url: "", loading: false, gap: false, unavailable: false, mode: "jpeg" });
  }, [cameraId, source, startEpoch]);
  useEffect(() => {
    setLocalPreviewReady(false);
    setLocalPreviewFrameReady(false);
    localPreviewTargetRef.current = null;
  }, [previewManifestUrl]);
  const percent = (draft / duration) * 100;

  function hidePreviewAfterDelay() {
    if (previewHideTimerRef.current) window.clearTimeout(previewHideTimerRef.current);
    previewHideTimerRef.current = window.setTimeout(() => {
      previewHideTimerRef.current = null;
      if (!dragRef.current) setScrubbing(false);
    }, 1400);
  }

  async function requestPreview(request) {
    if (previewInFlightRef.current) {
      previewPendingRef.current = request;
      return;
    }
    const generation = previewGenerationRef.current;
    previewInFlightRef.current = true;
    previewLastRequestRef.current = { epoch: request.bucket, at: performance.now() };
    const controller = new AbortController();
    previewAbortRef.current = controller;
    setPreview((current) => ({
      ...current,
      epoch: request.epoch,
      loading: true,
      gap: false,
      unavailable: false,
      mode: "jpeg",
    }));
    try {
      const response = await fetch(recordingPreviewUrl(cameraId, request.epoch, source), {
        signal: controller.signal,
      });
      if (controller.signal.aborted || generation !== previewGenerationRef.current) return;
      if (response.status === 404) {
        setPreview((current) => ({
          ...current,
          epoch: request.epoch,
          loading: false,
          gap: true,
          unavailable: false,
          mode: "jpeg",
        }));
      } else {
        if (!response.ok) throw new Error(`Preview failed (${response.status})`);
        const objectUrl = URL.createObjectURL(await response.blob());
        if (controller.signal.aborted || generation !== previewGenerationRef.current) {
          URL.revokeObjectURL(objectUrl);
          return;
        }
        if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
        previewUrlRef.current = objectUrl;
        setPreview({
          epoch: request.epoch,
          url: objectUrl,
          loading: false,
          gap: false,
          unavailable: false,
          mode: "jpeg",
        });
      }
    } catch (error) {
      if (error.name !== "AbortError" && generation === previewGenerationRef.current) {
        setPreview((current) => ({
          ...current,
          epoch: request.epoch,
          loading: false,
          gap: false,
          unavailable: true,
          mode: "jpeg",
        }));
      }
    } finally {
      if (generation !== previewGenerationRef.current) return;
      previewInFlightRef.current = false;
      const pending = previewPendingRef.current;
      previewPendingRef.current = null;
      if (pending && pending.bucket !== request.bucket) {
        if (!requestLocalPreview(pending)) requestPreview(pending);
      } else if (!dragRef.current) {
        setScrubbing(true);
        hidePreviewAfterDelay();
      }
    }
  }

  function requestLocalPreview(request) {
    const video = localPreviewRef.current;
    const mediaTime = playbackMediaTimeForEpoch(previewTimeline, request.epoch);
    if (!localPreviewReady || previewInFlightRef.current || !video || !Number.isFinite(mediaTime)) {
      return false;
    }
    previewLastRequestRef.current = { epoch: request.bucket, at: performance.now() };
    previewPendingRef.current = null;
    const target = { epoch: request.epoch, mediaTime };
    localPreviewTargetRef.current = target;
    setPreview((current) => ({
      ...current,
      loading: true,
      gap: false,
      unavailable: false,
      mode: localPreviewFrameReady ? "local" : current.mode,
    }));
    video.pause();
    if (Math.abs(video.currentTime - mediaTime) <= 0.04 && video.readyState >= 2) {
      publishLocalPreviewFrame(video, target);
    } else {
      video.currentTime = mediaTime;
    }
    return true;
  }

  function handleLocalPreviewReady(_player, video) {
    video.pause();
    setLocalPreviewReady(true);
    const target = localPreviewTargetRef.current;
    if (target && Number.isFinite(target.mediaTime)) video.currentTime = target.mediaTime;
  }

  function handleLocalPreviewSeeked(event) {
    const target = localPreviewTargetRef.current;
    if (!target || Math.abs(event.currentTarget.currentTime - target.mediaTime) > 0.08) return;
    publishLocalPreviewFrame(event.currentTarget, target);
  }

  function publishLocalPreviewFrame(video, target) {
    const publish = () => {
      if (localPreviewTargetRef.current !== target || video.readyState < 2) return;
      const canvas = localPreviewCanvasRef.current;
      const sourceWidth = Number(video.videoWidth) || 0;
      const sourceHeight = Number(video.videoHeight) || 0;
      if (!canvas || !sourceWidth || !sourceHeight) return;
      const width = Math.min(480, sourceWidth);
      const height = Math.max(1, Math.round(sourceHeight * (width / sourceWidth)));
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
      canvas.getContext("2d", { alpha: false })?.drawImage(video, 0, 0, width, height);
      setLocalPreviewFrameReady(true);
      setPreview((current) => ({
        ...current,
        epoch: target.epoch,
        loading: false,
        gap: false,
        unavailable: false,
        mode: "local",
      }));
      if (!dragRef.current) {
        setScrubbing(true);
        hidePreviewAfterDelay();
      }
    };
    if (typeof video.requestVideoFrameCallback === "function") {
      let published = false;
      const fallback = window.setTimeout(() => {
        if (!published) publish();
      }, 80);
      video.requestVideoFrameCallback(() => {
        published = true;
        window.clearTimeout(fallback);
        publish();
      });
    } else {
      window.requestAnimationFrame(publish);
    }
  }

  function handleLocalPreviewError() {
    setLocalPreviewReady(false);
    setLocalPreviewFrameReady(false);
    localPreviewTargetRef.current = null;
    previewLastRequestRef.current = { epoch: null, at: 0 };
    if (dragRef.current) schedulePreview(draftRef.current, true);
  }

  function schedulePreview(value, immediate = false) {
    if (!cameraId) return;
    const requestedEpoch = startEpoch + Math.max(0, Math.min(duration, Number(value) || 0));
    const localMediaTime = localPreviewReady && !previewInFlightRef.current
      ? playbackMediaTimeForEpoch(previewTimeline, requestedEpoch)
      : null;
    const previewBucket = Number.isFinite(localMediaTime)
      ? `local:${Math.floor(requestedEpoch * 2)}`
      : `jpeg:${Math.floor((requestedEpoch - startEpoch) / 5)}`;
    if (previewLastRequestRef.current.epoch === previewBucket) {
      previewPendingRef.current = null;
      return;
    }
    if (previewTimerRef.current) window.clearTimeout(previewTimerRef.current);
    const elapsed = performance.now() - previewLastRequestRef.current.at;
    const delay = immediate ? 0 : Math.max(0, 250 - elapsed);
    previewTimerRef.current = window.setTimeout(() => {
      previewTimerRef.current = null;
      const request = { epoch: requestedEpoch, bucket: previewBucket };
      if (!requestLocalPreview(request)) requestPreview(request);
    }, delay);
  }

  function updateDraft(value, requestPreview = false) {
    const next = Math.max(0, Math.min(duration, Number(value) || 0));
    draftRef.current = next;
    setDraft(next);
    if (requestPreview) schedulePreview(next);
    return next;
  }

  function commit(value) {
    const next = updateDraft(value);
    onSeek(startEpoch + next);
  }

  function pointerValue(event, drag) {
    const pointerX = Math.max(0, Math.min(drag.width, event.clientX - drag.left));
    if (drag.precise) {
      return drag.startValue + (pointerX - drag.startX);
    }
    return (pointerX / drag.width) * duration;
  }

  function startDrag(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.width) return;
    const pointerX = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
    const playheadX = (draftRef.current / duration) * rect.width;
    const grabRadius = event.pointerType === "touch" ? 24 : 14;
    const drag = {
      pointerId: event.pointerId,
      left: rect.left,
      width: rect.width,
      startX: pointerX,
      startValue: draftRef.current,
      precise: Math.abs(pointerX - playheadX) <= grabRadius,
    };
    dragRef.current = drag;
    if (previewHideTimerRef.current) window.clearTimeout(previewHideTimerRef.current);
    if (previewManifestUrl) setLocalPreviewEnabled(true);
    setScrubbing(true);
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    if (!drag.precise) updateDraft(pointerValue(event, drag), true);
    else schedulePreview(draftRef.current, true);
  }

  function moveDrag(event) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    updateDraft(pointerValue(event, drag), true);
  }

  function finishDrag(event, cancelled = false) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
    if (previewTimerRef.current) window.clearTimeout(previewTimerRef.current);
    previewTimerRef.current = null;
    if (!cancelled) schedulePreview(pointerValue(event, drag), true);
    hidePreviewAfterDelay();
    if (cancelled) updateDraft(offset);
    else commit(pointerValue(event, drag));
  }

  function exportEpochAtPointer(event, drag) {
    const pointerX = Math.max(0, Math.min(drag.width, event.clientX - drag.left));
    return startEpoch + (pointerX / drag.width) * duration;
  }

  function updateExportHandle(kind, epoch) {
    if (!exportRange) return;
    const minimumGap = 1;
    const next = kind === "start"
      ? { ...exportRange, start: Math.max(startEpoch, Math.min(exportRange.end - minimumGap, epoch)) }
      : { ...exportRange, end: Math.min(endEpoch, Math.max(exportRange.start + minimumGap, epoch)) };
    onExportRangeChange?.(next);
  }

  function startExportDrag(kind, event) {
    const track = event.currentTarget.parentElement;
    const rect = track?.getBoundingClientRect();
    if (!rect?.width) return;
    exportDragRef.current = { kind, pointerId: event.pointerId, left: rect.left, width: rect.width };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function moveExportDrag(event) {
    const drag = exportDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    updateExportHandle(drag.kind, exportEpochAtPointer(event, drag));
    event.preventDefault();
  }

  function finishExportDrag(event) {
    const drag = exportDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    updateExportHandle(drag.kind, exportEpochAtPointer(event, drag));
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    exportDragRef.current = null;
    event.preventDefault();
  }

  const exportStartPercent = exportRange ? ((exportRange.start - startEpoch) / duration) * 100 : 0;
  const exportEndPercent = exportRange ? ((exportRange.end - startEpoch) / duration) * 100 : 0;

  return (
    <div className="recordings-v2-timeline">
      <div className="recordings-v2-track">
        <div className="recordings-v2-ticks" aria-hidden="true">
          {ticks.map((tick) => (
            <span className={tick.kind} key={tick.epoch} style={{ left: `${((tick.epoch - startEpoch) / duration) * 100}%` }}>
              {tick.label ? <small>{tick.label}</small> : null}
            </span>
          ))}
        </div>
        {recordings.map((item) => (
          <span
            key={`${item.start_epoch}:${item.end_epoch}`}
            style={{
              left: `${((Math.max(startEpoch, item.start_epoch) - startEpoch) / duration) * 100}%`,
              width: `${((Math.min(endEpoch, item.end_epoch) - Math.max(startEpoch, item.start_epoch)) / duration) * 100}%`,
            }}
          />
        ))}
        <div className="recordings-v2-event-markers" aria-hidden="true">
          {eventMarkers.map((event) => <b key={event.id} className={event.hasObjects ? "object" : "motion"} style={{ left: `${event.left}%`, width: `${event.width}%` }} />)}
        </div>
        {exportRange ? (
          <>
            <div
              className="recordings-v2-export-selection"
              style={{ left: `${exportStartPercent}%`, width: `${Math.max(0, exportEndPercent - exportStartPercent)}%` }}
              aria-hidden="true"
            />
            <button
              type="button"
              className="recordings-v2-export-handle start"
              style={{ left: `${exportStartPercent}%` }}
              onPointerDown={(event) => startExportDrag("start", event)}
              onPointerMove={moveExportDrag}
              onPointerUp={finishExportDrag}
              onPointerCancel={finishExportDrag}
              disabled={!onExportRangeChange}
              aria-label={`Export start ${formatTimeOnly(exportRange.start, timeZone)}`}
            ><span>{formatExportHandleTime(exportRange.start, timeZone)}</span></button>
            <button
              type="button"
              className="recordings-v2-export-handle end"
              style={{ left: `${exportEndPercent}%` }}
              onPointerDown={(event) => startExportDrag("end", event)}
              onPointerMove={moveExportDrag}
              onPointerUp={finishExportDrag}
              onPointerCancel={finishExportDrag}
              disabled={!onExportRangeChange}
              aria-label={`Export end ${formatTimeOnly(exportRange.end, timeZone)}`}
            ><span>{formatExportHandleTime(exportRange.end, timeZone)}</span></button>
          </>
        ) : null}
        {localPreviewEnabled && previewManifestUrl ? (
          <ShakaVideo
            ref={localPreviewRef}
            src={previewManifestUrl}
            mimeType="application/vnd.apple.mpegurl"
            startTime={previewStartTime}
            bufferingGoal={2}
            muted
            playsInline
            preload="metadata"
            className="recordings-v2-local-preview-decoder"
            tabIndex={-1}
            aria-label="Local recording scrub preview"
            onReady={handleLocalPreviewReady}
            onError={handleLocalPreviewError}
            onSeeked={handleLocalPreviewSeeked}
          />
        ) : null}
        <div
          className={`recordings-v2-scrub-preview${scrubbing ? " active" : ""}${preview.loading ? " loading" : ""}${preview.mode === "local" ? " local" : ""}`}
          style={{ left: `${Math.max(8, Math.min(92, percent))}%` }}
          role="status"
          aria-live="polite"
          aria-hidden={!scrubbing}
        >
          <div>
            <canvas ref={localPreviewCanvasRef} aria-label="Decoded recording preview frame" />
            {preview.mode === "jpeg" && preview.url && !preview.gap && !preview.unavailable ? <img src={preview.url} alt="Recording preview" /> : null}
            {preview.mode === "jpeg" && preview.gap ? <span><Film size={20} />No recording</span> : null}
            {preview.mode === "jpeg" && preview.unavailable ? <span><RefreshCcw size={18} />Preview unavailable</span> : null}
            {preview.mode === "jpeg" && !preview.url && !preview.gap && !preview.unavailable ? <span><RefreshCcw size={18} />Loading preview</span> : null}
          </div>
          <time>{formatTimeOnly(Number.isFinite(preview.epoch) ? preview.epoch : startEpoch + draft, timeZone)}</time>
        </div>
        <i style={{ left: `${percent}%` }} />
        <output style={{ left: `${Math.max(4, Math.min(96, percent))}%` }}>{formatTimeOnly(startEpoch + draft, timeZone)}</output>
        <input
          type="range"
          min="0"
          max={duration}
          step="0.1"
          value={draft}
          onChange={(event) => {
            if (!dragRef.current) updateDraft(event.target.value);
          }}
          onPointerDown={startDrag}
          onPointerMove={moveDrag}
          onPointerUp={(event) => finishDrag(event)}
          onPointerCancel={(event) => finishDrag(event, true)}
          onKeyUp={(event) => commit(event.currentTarget.value)}
          aria-label="24 hour recording timeline"
        />
      </div>
    </div>
  );
}

function MotionAnalysisPresetEditor({
  qualification,
  inherited = false,
  catalog,
  onSetInherited,
  onChange,
}) {
  const presets = availableQualificationPresets(catalog);
  const parsed = readMotionAnalysisPreset(qualification, catalog);
  const selectedValue = inherited
    ? "inherit"
    : parsed.custom
      ? "custom"
      : parsed.preset?.id || "";

  function selectPreset(value) {
    if (value === "inherit") {
      onSetInherited?.(true);
      return;
    }
    const preset = presets.find((candidate) => candidate.id === value);
    if (preset) onChange(presetQualificationGraph(preset));
  }

  if (!motionAnalysisPresetSelectionUseful(catalog)) {
    return parsed.custom && !inherited ? (
      <div className="motion-analysis-warning motion-analysis-custom-notice">
        <strong>Advanced custom motion pipeline</strong>
        <span>This externally configured pipeline remains active and protected from guided settings.</span>
      </div>
    ) : null;
  }

  return (
    <div className="motion-analysis-preset">
      <label>Motion analysis method<select value={selectedValue} onChange={(event) => selectPreset(event.target.value)} disabled={!presets.length}>
        {inherited ? <option value="inherit">Use global setting</option> : null}
        {parsed.custom && !inherited ? <option value="custom">Advanced custom pipeline</option> : null}
        {!presets.length ? <option value="">Loading available methods...</option> : null}
        {presets.map((preset) => (
          <option key={preset.id} value={preset.id}>{preset.label}{preset.recommended ? " (Recommended)" : ""}</option>
        ))}
      </select></label>
      {parsed.custom && !inherited ? <div className="motion-analysis-warning">This advanced pipeline is protected. Selecting another method will replace only the motion-analysis stages.</div> : null}
    </div>
  );
}

function MotionDecisionEditor({
  fusion,
  mode,
  globalMode = "camera",
  inherited = false,
  inheritedFusion,
  onSetInherited,
  onModeChange,
  onChange,
  onRestoreDefaults,
  configurationInherited,
  cameraName,
}) {
  const parsed = readMotionDecisionFusion(fusion);
  const inheritedParsed = readMotionDecisionFusion(inheritedFusion);
  const effective = inherited ? inheritedParsed : parsed;
  const settings = effective.settings;
  const effectiveMode = mode === "inherit" ? globalMode : mode;
  const legacyMode = ["audit", "off", "enforce"].includes(effectiveMode);
  const fullyInherited = Boolean(onSetInherited && inherited && mode === "inherit");
  const custom = (!inherited && parsed.custom) || (fullyInherited && inheritedParsed.custom);
  const effectiveBehavior = motionBehaviorValue(effectiveMode, settings);
  const selectedBehavior = fullyInherited
    ? "inherit"
    : custom
      ? "custom"
      : legacyMode
        ? `legacy:${effectiveMode}`
        : effectiveBehavior;
  const behaviorInfo = custom
    ? {
      status: fullyInherited ? "Global advanced configuration" : "Advanced custom configuration",
      description: "This pipeline was created outside the guided editor. Selecting a standard behavior will replace its decision stages.",
    }
    : legacyMode
      ? motionModeInfo(effectiveMode)
      : motionBehaviorOption(effectiveBehavior);
  const statusLabel = onRestoreDefaults
    ? configurationInherited ? "Inherited" : "Custom"
    : fullyInherited ? "Inherited" : custom ? "Advanced" : legacyMode ? "Legacy" : parsed.usesDefaults ? "Recommended default" : "Customized";

  function updateSettings(patch) {
    onChange(buildMotionDecisionFusion({ ...settings, ...patch }));
  }

  function selectBehavior(value) {
    if (value === "inherit") {
      onModeChange("inherit");
      onSetInherited?.(true);
      return;
    }
    if (value === "custom" || value.startsWith("legacy:")) return;
    const next = motionBehaviorSettings(settings, value);
    onModeChange(next.mode);
    onSetInherited?.(false);
    onChange(buildMotionDecisionFusion(next.settings));
  }

  return (
    <div className={`motion-decision-editor${custom || legacyMode ? " motion-decision-custom" : ""}`}>
      <div className="motion-decision-heading">
        <div>
          <strong>Motion behavior</strong>
          <span>{cameraName ? `Choose what can start object detection for ${cameraName}.` : "Choose what can start object detection."}</span>
        </div>
        <div className="motion-decision-status-actions">
          <span className="motion-decision-status">{statusLabel}</span>
          {onRestoreDefaults ? <button type="button" className="motion-decision-status motion-defaults-action" onClick={onRestoreDefaults} title="Restore all motion settings for this camera to global inheritance">Defaults</button> : null}
        </div>
      </div>

      <div className="motion-behavior-row">
        <label>What starts object detection?<select value={selectedBehavior} onChange={(event) => selectBehavior(event.target.value)}>
          {onSetInherited ? <option value="inherit">Use global setting</option> : null}
          {custom ? <option value="custom">Advanced custom configuration</option> : null}
          {legacyMode ? <option value={`legacy:${effectiveMode}`}>{motionModeInfo(effectiveMode).label}</option> : null}
          {MOTION_BEHAVIOR_OPTIONS.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}
        </select></label>
        <div className={`motion-decision-mode mode-${effectiveMode}`}>
          <strong>{fullyInherited && !custom ? `Global · ${behaviorInfo.status}` : behaviorInfo.status}</strong>
          <span>{behaviorInfo.description}</span>
        </div>
      </div>

    </div>
  );
}

const TelemetryInterruptionsContext = React.createContext([]);

function TelemetryTrend({ title, history, series, timeZone, interruptions = null, maximum = null, valueFormatter = (value) => `${value}` }) {
  const [hoverState, setHoverState] = useState(null);
  const sharedInterruptions = useContext(TelemetryInterruptionsContext);
  const chartInterruptions = interruptions || sharedInterruptions;
  const numericValue = (raw) => {
    if (raw == null || raw === "") return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  };
  const values = series.flatMap((item) => history.map((point) => numericValue(point[item.key])).filter((value) => value != null));
  const top = maximum || Math.max(1, ...values) * 1.12;
  const width = 100;
  const height = 42;
  const sampleTimes = history.map((point) => new Date(point?.sampled_at || 0).getTime());
  const firstAt = Number.isFinite(sampleTimes[0]) ? sampleTimes[0] : 0;
  const lastAt = Number.isFinite(sampleTimes.at(-1)) ? sampleTimes.at(-1) : firstAt;
  const timeSpan = Math.max(0, lastAt - firstAt);
  const xForIndex = (index) => {
    if (history.length <= 1) return width;
    const sampledAt = sampleTimes[index];
    return timeSpan > 0 && Number.isFinite(sampledAt)
      ? ((sampledAt - firstAt) / timeSpan) * width
      : (index / (history.length - 1)) * width;
  };
  const segmentsFor = (key) => {
    const segments = [];
    let segment = [];
    history.forEach((point, index) => {
      const value = numericValue(point[key]);
      if (value == null) {
        if (segment.length) segments.push(segment);
        segment = [];
      }
      if (value != null) {
        const x = xForIndex(index);
        const y = height - Math.min(height, (Math.max(0, value) / top) * height);
        segment.push(`${x.toFixed(2)},${y.toFixed(2)}`);
      }
    });
    if (segment.length) segments.push(segment);
    return segments;
  };
  const xForTime = (value) => {
    const timestamp = new Date(value || 0).getTime();
    if (!Number.isFinite(timestamp) || timeSpan <= 0) return null;
    return Math.max(0, Math.min(width, ((timestamp - firstAt) / timeSpan) * width));
  };
  const visibleInterruptions = chartInterruptions.map((item) => {
    const startX = xForTime(item.start_at);
    const markerX = xForTime(item.marker_at || item.start_at);
    const endX = xForTime(item.end_at);
    const actualStartX = Math.min(startX ?? 0, endX ?? 0);
    const actualWidth = Math.abs((endX ?? 0) - (startX ?? 0));
    const displayWidth = Math.max(0.18, actualWidth);
    const displayStartX = Math.max(
      0,
      Math.min(
        width - displayWidth,
        actualWidth >= 0.18
          ? actualStartX
          : (markerX ?? actualStartX) - (displayWidth / 2),
      ),
    );
    return {
      ...item,
      startX,
      markerX,
      endX,
      displayStartX,
      displayWidth,
    };
  }).filter((item) => item.startX != null && item.endX != null && new Date(item.end_at).getTime() >= firstAt && new Date(item.start_at).getTime() <= lastAt);
  const coordinatesFor = (key, index) => {
    const value = numericValue(history[index]?.[key]);
    if (value == null) return null;
    return {
      value,
      x: xForIndex(index),
      y: height - Math.min(height, (Math.max(0, value) / top) * height),
    };
  };
  const latestValue = (key) => {
    for (let index = history.length - 1; index >= 0; index -= 1) {
      const value = numericValue(history[index]?.[key]);
      if (value != null) return value;
    }
    return null;
  };
  const formatBoundary = (value) => (
    lastAt - firstAt >= 24 * 60 * 60 * 1000
      ? formatDateTime(value, timeZone)
      : formatTimeOnly(value, timeZone)
  );
  const selectedIndex = Number.isInteger(hoverState?.index) && hoverState.index >= 0 && hoverState.index < history.length
    ? hoverState.index
    : null;
  const selectedPoint = selectedIndex == null ? null : history[selectedIndex];
  const selectedPointX = selectedIndex == null
    ? 0
    : xForIndex(selectedIndex);
  const hoverX = hoverState?.x ?? selectedPointX;
  const interruptionHitTolerance = hoverState?.hitToleranceX ?? 0.25;
  const hoverInterruption = visibleInterruptions.find((item) => (
    hoverX >= item.displayStartX - interruptionHitTolerance
    && hoverX <= item.displayStartX + item.displayWidth + interruptionHitTolerance
  )) || null;
  const tooltipAlignment = hoverX < 25 ? "start" : hoverX > 75 ? "end" : "center";
  const selectHoverIndex = (index) => {
    if (!history.length) return;
    const boundedIndex = Math.max(0, Math.min(history.length - 1, index));
    setHoverState({ index: boundedIndex, x: xForIndex(boundedIndex), hitToleranceX: 0.25 });
  };
  const updateHover = (event) => {
    if (!history.length) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left) / Math.max(1, bounds.width)));
    if (timeSpan <= 0) {
      setHoverState({
        index: Math.round(ratio * Math.max(0, history.length - 1)),
        x: ratio * width,
        hitToleranceX: (6 / Math.max(1, bounds.width)) * width,
      });
      return;
    }
    const targetTime = firstAt + ratio * timeSpan;
    let nearestIndex = 0;
    for (let index = 1; index < sampleTimes.length; index += 1) {
      if (Math.abs(sampleTimes[index] - targetTime) < Math.abs(sampleTimes[nearestIndex] - targetTime)) {
        nearestIndex = index;
      }
    }
    setHoverState({
      index: nearestIndex,
      x: ratio * width,
      hitToleranceX: (6 / Math.max(1, bounds.width)) * width,
    });
  };
  const handleChartKeyDown = (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "Home") selectHoverIndex(0);
    else if (event.key === "End") selectHoverIndex(history.length - 1);
    else {
      const currentIndex = Number.isInteger(hoverState?.index)
        ? hoverState.index
        : history.length - 1;
      selectHoverIndex(currentIndex + (event.key === "ArrowLeft" ? -1 : 1));
    }
  };
  return (
    <article className={`telemetry-trend${selectedPoint ? " has-tooltip" : ""}`}>
      <header><strong>{title}</strong><div className="telemetry-trend-values" aria-label="Chart lines">{series.map((item) => <span className={item.className || ""} key={item.key}><i /><b>{item.label}</b><em>{latestValue(item.key) == null ? "--" : valueFormatter(latestValue(item.key), item.key)}</em></span>)}</div></header>
      <div
        className="telemetry-trend-chart"
        tabIndex={history.length ? 0 : -1}
        aria-label={`${title}. Point, tap, or use the arrow keys to inspect values.`}
        onPointerMove={updateHover}
        onPointerDown={updateHover}
        onFocus={(event) => {
          if (hoverState == null && event.currentTarget.matches(":focus-visible")) {
            selectHoverIndex(history.length - 1);
          }
        }}
        onBlur={() => setHoverState(null)}
        onKeyDown={handleChartKeyDown}
        onPointerLeave={(event) => {
          if (event.pointerType === "mouse") setHoverState(null);
        }}
      >
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label={`${title} trend`}>
          <line x1="0" y1={height / 2} x2={width} y2={height / 2} />
          {visibleInterruptions.map((item, index) => <g className={`telemetry-interruption ${item.kind}`} key={`${item.start_at}-${index}`}>
            <rect x={item.displayStartX} y="0" width={item.displayWidth} height={height} />
          </g>)}
          {series.flatMap((item) => segmentsFor(item.key).map((points, index) => <polyline className={item.className || ""} key={`${item.key}-${index}`} points={points.join(" ")} />))}
          {selectedPoint ? <line className="telemetry-trend-cursor" x1={hoverX} y1="0" x2={hoverX} y2={height} /> : null}
          {selectedPoint ? series.map((item) => {
            const coordinates = coordinatesFor(item.key, selectedIndex);
            return coordinates ? <ellipse className={`telemetry-trend-point ${item.className || ""}`} key={item.key} cx={coordinates.x} cy={coordinates.y} rx="0.75" ry="1.4" /> : null;
          }) : null}
        </svg>
        {selectedPoint ? (
          <div className={`telemetry-trend-tooltip ${tooltipAlignment}`} style={{ left: `${hoverX}%` }} role="status">
            <time>{hoverInterruption ? `Restarted ${formatDateTime(hoverInterruption.marker_at, timeZone)}` : formatDateTime(selectedPoint.sampled_at, timeZone)}</time>
            {hoverInterruption ? <div className={`telemetry-trend-interruption-detail ${hoverInterruption.kind}`}><strong>{hoverInterruption.title}</strong><small>{formatDuration(hoverInterruption.duration_seconds || 0)} · {hoverInterruption.description}</small></div> : null}
            {series.map((item) => {
              const value = numericValue(selectedPoint[item.key]);
              return <span className={item.className || ""} key={item.key}><i />{item.label}<strong>{value == null ? "--" : valueFormatter(value, item.key)}</strong></span>;
            })}
          </div>
        ) : null}
      </div>
      <footer><time>{history[0] ? formatBoundary(history[0].sampled_at) : "Now"}</time><span>{history.length} sample{history.length === 1 ? "" : "s"}</span><time>{history.at(-1) ? formatBoundary(history.at(-1).sampled_at) : "Now"}</time></footer>
    </article>
  );
}

function formatRecorderTimestampHealth(sources) {
  const entries = Object.entries(sources || {});
  if (!entries.length) return "Stable · no discontinuities";
  return entries.map(([source, health]) => {
    const details = [
      `${Number(health.discontinuities || 0)} jump${Number(health.discontinuities || 0) === 1 ? "" : "s"}`,
      `${Number(health.epoch_rollovers || 0)} recovered`,
    ];
    if (health.rollover_pending) details.push("recovery pending");
    if (Number(health.rollover_failures || 0)) details.push(`${health.rollover_failures} failed`);
    if (Number(health.rate_limited || 0)) details.push(`${health.rate_limited} rate-limited`);
    return `${source}: ${details.join(", ")}`;
  }).join(" · ");
}

function TelemetryContinuity({ data }) {
  if (!data) return null;
  const summary = data.interruption_summary || {};
  const parts = [
    summary.controlled ? `${summary.controlled} controlled restart${summary.controlled === 1 ? "" : "s"}` : "",
    summary.unexpected ? `${summary.unexpected} unexpected restart${summary.unexpected === 1 ? "" : "s"}` : "",
    summary.unknown ? `${summary.unknown} unexplained gap${summary.unknown === 1 ? "" : "s"}` : "",
  ].filter(Boolean);
  return (
    <div className={`telemetry-interruption-summary telemetry-header-continuity ${summary.unexpected ? "danger" : summary.unknown ? "warning" : "healthy"}`}>
      <Clock3 size={16} />
      <span><strong>Service continuity · 24h</strong><em>{parts.length ? `${parts.join(" · ")} · ${formatDuration(summary.duration_seconds || 0)} unavailable` : "No interruptions"}</em></span>
    </div>
  );
}

function TelemetryViewer({ data, cameraId, timeZone }) {
  if (!data) return <div className="empty-state">Waiting for telemetry...</div>;
  const selected = cameraId ? data.cameras?.find((camera) => camera.id === cameraId) : null;
  const activity = selected?.activity || data.activity;
  const lastHour = activity?.last_hour || {};
  const lastDay = activity?.last_24h || {};
  const runtime = data.detector?.runtime || {};
  const objectWorkers = data.detector?.workers?.object || data.detector?.isolation || {};
  const semantic = data.semantic_search || {};
  const faceRecognition = data.face_recognition || {};
  const gpu = data.gpu || {};
  const storage = data.system?.storage || {};
  const memory = data.system?.memory || {};
  const serviceMemory = data.system?.service_memory || {};
  const workerMemory = data.system?.worker_memory || {};
  const memoryMaintenance = data.system?.memory_maintenance || {};
  const hourly = activity?.hourly || [];
  const runtimeShort = data.runtime_history?.short || [];
  const runtimeLong = data.runtime_history?.long || [];
  const capacityShort = data.tracking_capacity_history?.short || [];
  const capacityLong = data.tracking_capacity_history?.long || [];
  const memoryShort = data.process_memory_history?.short || [];
  const memoryLong = data.process_memory_history?.long || [];
  const appearanceBackfill = data.appearance_backfill || {};
  const backfillCounts = appearanceBackfill.counts || {};
  const capacityTotals = capacityShort.reduce((total, point) => ({
    attempts: total.attempts + Number(point.attempts || 0),
    waited: total.waited + Number(point.waited || 0),
    skipped: total.skipped + Number(point.skipped || 0),
    waitMax: Math.max(total.waitMax, Number(point.wait_seconds_max || 0)),
  }), { attempts: 0, waited: 0, skipped: 0, waitMax: 0 });
  const maxHourly = Math.max(1, ...hourly.map((item) => Number(item.events) || 0));
  const topLabels = Object.entries(lastDay.labels || {}).sort((left, right) => right[1] - left[1]).slice(0, 5);
  const shownCameras = selected ? [selected] : (data.cameras || []);
  const activityAttribution = shownCameras.reduce((total, camera) => {
    const status = camera.object_tracking?.object_activity_attribution || {};
    total.evaluated += Number(status.evaluated || 0);
    total.active += Number(status.active || 0);
    total.sceneContext += Number(status.scene_context || 0);
    total.indeterminate += Number(status.indeterminate || 0);
    total.enforced += Number(status.enforced_suppressions || 0);
    total.detectorAdmissions += Number(status.detector_admissions || 0);
    total.confidenceRejections += Number(status.confidence_rejections || 0);
    total.zoneRejections += Number(status.zone_rejections || 0);
    total.temporalRejections += Number(status.temporal_rejections || 0);
    if (status.mode) total.modes.add(status.mode);
    return total;
  }, { evaluated: 0, active: 0, sceneContext: 0, indeterminate: 0, enforced: 0, detectorAdmissions: 0, confidenceRejections: 0, zoneRejections: 0, temporalRejections: 0, modes: new Set() });
  const selectedCapture = selected?.capture || {};
  const selectedReadFailures = [selectedCapture.live, selectedCapture.main]
    .reduce((total, source) => total + Number(source?.read_failures || 0), 0);
  const selectedOpenFailures = [selectedCapture.live, selectedCapture.main]
    .reduce((total, source) => total + Number(source?.open_failures || 0), 0);
  const runtimeTotals = runtimeShort.reduce((total, point) => ({
    analyzed: total.analyzed + Number(point.analysis_frames_sampled || 0),
    superseded: total.superseded + Number(point.analysis_frames_dropped || 0),
    interruptions: total.interruptions + Number(point.capture_interruptions || 0),
    eventLoss: total.eventLoss + Number(point.event_delivery_failures || 0),
    availabilitySum: total.availabilitySum + Number(point.camera_availability_percent ?? 0),
    availabilitySamples: total.availabilitySamples + (point.camera_availability_percent == null ? 0 : 1),
    minimumAvailability: Math.min(
      total.minimumAvailability,
      Number(point.camera_availability_percent ?? 100),
    ),
  }), {
    analyzed: 0,
    superseded: 0,
    interruptions: 0,
    eventLoss: 0,
    availabilitySum: 0,
    availabilitySamples: 0,
    minimumAvailability: 100,
  });
  const averageAvailability = runtimeTotals.availabilitySamples
    ? runtimeTotals.availabilitySum / runtimeTotals.availabilitySamples
    : null;
  const analysisTotal = runtimeTotals.analyzed + runtimeTotals.superseded;
  const analysisCoverage = analysisTotal
    ? (runtimeTotals.analyzed / analysisTotal) * 100
    : null;
  const formatCoverage = (value) => value == null
    ? "--"
    : `${Number(value).toFixed(Number(value) >= 99.95 ? 2 : 1)}%`;
  return (
    <TelemetryInterruptionsContext.Provider value={selected ? [] : (data.interruptions || [])}>
    <div className="telemetry-viewer">
      <div className={`telemetry-summary-grid${selected ? " camera-summary" : " overview-summary"}`}>
        <article><span>Events · 1h</span><strong>{Number(lastHour.events || 0).toLocaleString()}</strong><small>{Number(lastDay.events || 0).toLocaleString()} in the shown 24-hour window</small></article>
        <article><span>Object incidents · 24h</span><strong>{Number(lastDay.object_incidents || 0).toLocaleString()}</strong><small>{Number(lastDay.objects || 0).toLocaleString()} eligible object detections</small></article>
        {selected ? <>
          <article><span>Live video</span><strong>{selected.connected ? "Available" : "Unavailable"}</strong><small>Last frame {formatAge(selected.last_frame_age_seconds)}</small></article>
          <article><span>Stream interruptions · since restart</span><strong>{(selectedReadFailures + selectedOpenFailures).toLocaleString()}</strong><small>{selectedReadFailures.toLocaleString()} interrupted reads · {selectedOpenFailures.toLocaleString()} failed connections</small></article>
          <article><span>Tracking · 2h</span><strong>{capacityTotals.skipped ? `${capacityTotals.skipped} skipped` : "No skips"}</strong><small>{capacityTotals.attempts} sessions · {capacityTotals.waited} waited · longest {capacityTotals.waitMax.toFixed(1)}s</small></article>
          <article><span>EMA coverage · 2h</span><strong>{analysisTotal ? formatCoverage(analysisCoverage) : "Not active"}</strong><small>{runtimeTotals.eventLoss ? `${runtimeTotals.eventLoss} events lost` : "No events lost"}</small></article>
        </> : <>
          <article><span>Camera uptime · 2h</span><strong>{formatCoverage(averageAvailability)}</strong><small>Lowest minute {formatCoverage(runtimeTotals.minimumAvailability)} · {runtimeTotals.interruptions ? `${runtimeTotals.interruptions.toLocaleString()} recovered stream issues` : "no stream interruptions"}</small></article>
          <article><span>EMA coverage · 2h</span><strong>{analysisTotal ? formatCoverage(analysisCoverage) : "Not active"}</strong><small>{analysisTotal ? (runtimeTotals.superseded ? `${runtimeTotals.superseded.toLocaleString()} stale frames skipped to stay current` : "Every sampled frame analyzed") : "No EMA samples in this window"}{runtimeTotals.eventLoss ? ` · ${runtimeTotals.eventLoss} events lost` : " · no events lost"}</small></article>
          <article><span>Detector response</span><strong>{formatMilliseconds(runtime.average_inference_ms)}</strong><small>{objectWorkers.alive_workers || (objectWorkers.worker_alive ? 1 : 0)}/{objectWorkers.configured_workers || 1} workers online · {Number(runtime.failed_inferences || 0) ? `${Number(runtime.failed_inferences).toLocaleString()} failures` : "no failures"}</small></article>
          <article><span>GPU</span><strong>{gpu.available ? "Available" : "Unavailable"}</strong><small>{Number.isFinite(gpu.utilization_percent) ? `${gpu.utilization_percent}% busy now` : "Collecting activity"}</small></article>
          <article><span>Storage free</span><strong>{formatBytes(storage.free_bytes)}</strong><small>{storage.used_percent || 0}% used of {formatBytes(storage.total_bytes)}</small></article>
          <article><span>Tracking · 2h</span><strong>{capacityTotals.skipped ? `${capacityTotals.skipped} skipped` : "No skips"}</strong><small>{capacityTotals.waited} delayed · {Number(backfillCounts.completed || 0).toLocaleString()} recovered · {Number(backfillCounts.queued || 0).toLocaleString()} waiting</small></article>
          <article><span>SurvNG uptime</span><strong>{formatDuration(data.system?.uptime_seconds || 0)}</strong><small>Since the last service start</small></article>
          <article><span>CPU demand</span><strong>{data.system?.load_average?.one ?? "--"}</strong><small>Across {data.system?.cpu_count || 1} cores</small></article>
          <article><span>Host memory</span><strong>{formatBytes(memory.available_bytes)}</strong><small>{memory.used_percent || 0}% currently used</small></article>
          <article><span>Application memory</span><strong>{formatBytes(serviceMemory.application_bytes)}</strong><small>SurvNG and AI workers</small></article>
          <article><span>File cache</span><strong>{formatBytes(serviceMemory.reclaimable_file_cache_bytes)}</strong><small>Released automatically as needed</small></article>
          <article><span>Local databases</span><strong>{formatBytes(data.system?.database?.bytes)}</strong><small>Events, indexes, and runtime state</small></article>
        </>}
      </div>

      {!selected ? <details className="telemetry-technical telemetry-system-technical">
        <summary>Technical system diagnostics</summary>
        <dl className="telemetry-details">
          <div><dt>CPU load · 1 / 5 / 15 min</dt><dd>{data.system?.load_average?.one ?? "--"} / {data.system?.load_average?.five ?? "--"} / {data.system?.load_average?.fifteen ?? "--"}</dd></div>
          <div><dt>Working set / service total</dt><dd>{formatBytes(serviceMemory.working_set_bytes)} / {formatBytes(serviceMemory.total_bytes)}</dd></div>
          <div><dt>Main / inference-worker RSS</dt><dd>{formatBytes(data.system?.process_rss_bytes)} / {formatBytes(workerMemory.total_rss_bytes)}</dd></div>
          <div><dt>Allocator live / retained</dt><dd>{formatBytes(data.system?.process_memory?.malloc?.allocated_bytes)} / {formatBytes(data.system?.process_memory?.malloc?.free_bytes)}</dd></div>
          <div><dt>Allocator trims</dt><dd>{Number(memoryMaintenance.successful_trims || 0).toLocaleString()} <small>{formatBytes(memoryMaintenance.reclaimed_total_bytes)} reclaimed</small></dd></div>
          <div><dt>Threads / open files</dt><dd>{Number(data.system?.process_memory?.threads || 0).toLocaleString()} / {Number(data.system?.process_memory?.file_descriptors || 0).toLocaleString()}</dd></div>
          <div><dt>Detector backend / device</dt><dd>{data.detector?.loaded_backend || "Not loaded"} / {data.detector?.loaded_device || data.detector?.configured_device || "--"}</dd></div>
          <div><dt>Object detector processes</dt><dd>{(objectWorkers.worker_pids || [objectWorkers.worker_pid]).filter(Boolean).join(", ") || "None"}</dd></div>
          <div><dt>Per-detector response</dt><dd>{(runtime.workers || []).length ? runtime.workers.map((worker) => `#${worker.index} ${formatMilliseconds(worker.average_inference_ms)} · ${Number(worker.queue_depth || 0)} queued`).join(" · ") : "Waiting for samples"}</dd></div>
          <div><dt>Inference requests / object hits</dt><dd>{Number(runtime.total_inferences || 0).toLocaleString()} / {Number(runtime.object_hit_inferences || 0).toLocaleString()}</dd></div>
        </dl>
      </details> : null}

      <section className="telemetry-section">
        <div className="telemetry-section-head"><div><h3>Events by hour{selected ? ` · ${selected.name}` : ""}</h3></div></div>
        <div className="telemetry-hourly" aria-label="Events per hour">
          {hourly.map((item, index) => (
            <div className="telemetry-hour" key={item.started_at} title={`${formatDateTime(item.started_at, timeZone)}: ${item.events} events, ${item.object_incidents} object incidents`}>
              <div className="telemetry-hour-bars">
                <i style={{ height: `${Math.max(3, (Number(item.events) / maxHourly) * 100)}%` }} />
                <b style={{ height: `${Math.max(0, (Number(item.object_incidents) / maxHourly) * 100)}%` }} />
              </div>
              {(index % 4 === 0 || index === hourly.length - 1) ? <time>{formatTimeOnly(item.started_at, timeZone).replace(/:00(?=\s)/, "")}</time> : <time />}
            </div>
          ))}
        </div>
        <div className="telemetry-legend"><span><i /> Events</span><span><i className="objects" /> Object incidents</span></div>
      </section>

      <section className="telemetry-section">
        <div className="telemetry-section-head"><div><h3>{selected ? `${selected.name} object tracking` : "Object tracking"}</h3></div></div>
        <div className="telemetry-trend-grid two-column">
          <TelemetryTrend title="Tracking · 2 hours" history={capacityShort} timeZone={timeZone} valueFormatter={(value) => Math.round(value).toLocaleString()} series={[{ key: "attempts", label: "Requested", className: "rate" }, { key: "waited", label: "Delayed", className: "warning" }, { key: "skipped", label: "Skipped", className: "danger" }]} />
          <TelemetryTrend title="Tracking · 7 days" history={capacityLong} timeZone={timeZone} valueFormatter={(value) => Math.round(value).toLocaleString()} series={[{ key: "attempts", label: "Requested", className: "rate" }, { key: "waited", label: "Delayed", className: "warning" }, { key: "skipped", label: "Skipped", className: "danger" }]} />
        </div>
      </section>

      <section className="telemetry-section">
        <div className="telemetry-section-head"><div><h3>Camera reliability{selected ? ` · ${selected.name}` : ""}</h3></div></div>
        <div className="telemetry-trend-grid two-column">
          <TelemetryTrend title="Availability · 2 hours" history={runtimeShort} timeZone={timeZone} maximum={100} valueFormatter={formatCoverage} series={[{ key: "camera_availability_percent", label: "Available", className: "rate" }]} />
          <TelemetryTrend title="Stream interruptions · 2 hours" history={runtimeShort} timeZone={timeZone} valueFormatter={(value) => Math.round(value).toLocaleString()} series={[{ key: "capture_interruptions", label: "Interruptions", className: "danger" }]} />
          <TelemetryTrend title="Availability · 7 days" history={runtimeLong} timeZone={timeZone} maximum={100} valueFormatter={formatCoverage} series={[{ key: "camera_availability_percent", label: "Available", className: "rate" }]} />
          <TelemetryTrend title="Stream interruptions · 7 days" history={runtimeLong} timeZone={timeZone} valueFormatter={(value) => Math.round(value).toLocaleString()} series={[{ key: "capture_interruptions", label: "Interruptions", className: "danger" }]} />
        </div>
      </section>

      <section className="telemetry-section">
        <div className="telemetry-section-head"><div><h3>Enhanced motion analysis{selected ? ` · ${selected.name}` : ""}</h3></div></div>
        <div className="telemetry-trend-grid two-column">
          <TelemetryTrend title="EMA coverage · 2 hours" history={runtimeShort} timeZone={timeZone} maximum={100} valueFormatter={formatCoverage} series={[{ key: "analysis_coverage_percent", label: "Coverage", className: "rate" }]} />
          <TelemetryTrend title="EMA coverage · 7 days" history={runtimeLong} timeZone={timeZone} maximum={100} valueFormatter={formatCoverage} series={[{ key: "analysis_coverage_percent", label: "Coverage", className: "rate" }]} />
          <TelemetryTrend title="EMA rescue path · 2 hours" history={runtimeShort} timeZone={timeZone} valueFormatter={(value) => Math.round(value).toLocaleString()} series={[{ key: "ema_credible_episodes", label: "Credible", className: "secondary" }, { key: "object_checks_admitted", label: "Admitted", className: "warning" }, { key: "object_checks_completed", label: "Completed", className: "rate" }]} />
          <TelemetryTrend title="EMA rescue path · 7 days" history={runtimeLong} timeZone={timeZone} valueFormatter={(value) => Math.round(value).toLocaleString()} series={[{ key: "ema_credible_episodes", label: "Credible", className: "secondary" }, { key: "object_checks_admitted", label: "Admitted", className: "warning" }, { key: "object_checks_completed", label: "Completed", className: "rate" }]} />
        </div>
      </section>

      {!selected ? <section className="telemetry-section">
        <div className="telemetry-section-head"><div><h3>System performance</h3></div></div>
        <div className="telemetry-trend-grid two-column">
          <TelemetryTrend title="Host demand · 2 hours" history={runtimeShort} timeZone={timeZone} maximum={100} valueFormatter={(value) => `${value.toFixed(1)}%`} series={[{ key: "cpu_load_percent", label: "CPU", className: "cpu" }, { key: "memory_used_percent", label: "Memory", className: "memory" }]} />
          <TelemetryTrend title="Detector response · 2 hours" history={runtimeShort} timeZone={timeZone} valueFormatter={(value) => formatMilliseconds(value)} series={[{ key: "inference_ms", label: "Response", className: "inference" }]} />
          <TelemetryTrend title="Host demand · 7 days" history={runtimeLong} timeZone={timeZone} maximum={100} valueFormatter={(value) => `${value.toFixed(1)}%`} series={[{ key: "cpu_load_percent", label: "CPU", className: "cpu" }, { key: "memory_used_percent", label: "Memory", className: "memory" }]} />
          <TelemetryTrend title="Detector response · 7 days" history={runtimeLong} timeZone={timeZone} valueFormatter={(value) => formatMilliseconds(value)} series={[{ key: "inference_ms", label: "Response", className: "inference" }]} />
        </div>
      </section> : null}

      {!selected ? <section className="telemetry-section">
        <div className="telemetry-section-head"><div><h3>Memory stability</h3></div></div>
        <div className="telemetry-trend-grid two-column">
          <TelemetryTrend title="Application memory · 24 hours" history={memoryShort} timeZone={timeZone} valueFormatter={(value) => formatBytes(value)} series={[{ key: "rss_bytes", label: "SurvNG", className: "process-memory" }, { key: "worker_rss_bytes", label: "AI workers", className: "secondary" }]} />
          <TelemetryTrend title="Application memory · 7 days" history={memoryLong} timeZone={timeZone} valueFormatter={(value) => formatBytes(value)} series={[{ key: "rss_bytes", label: "SurvNG", className: "process-memory" }, { key: "worker_rss_bytes", label: "AI workers", className: "secondary" }]} />
        </div>
      </section> : null}

      <div className={`telemetry-activity-grid${selected ? " camera-only" : ""}`}>
        <section className="telemetry-section">
          <div className="telemetry-section-head"><div><h3>{selected ? `${selected.name} activity` : "Object activity"}</h3></div></div>
          <dl className="telemetry-details">
            <div><dt>Top labels · 24h</dt><dd>{topLabels.length ? topLabels.map(([label, count]) => `${label} ${count}`).join(" · ") : "None"}</dd></div>
            <div><dt>Activity attribution · since restart</dt><dd>{activityAttribution.evaluated.toLocaleString()} checked <small>{activityAttribution.active.toLocaleString()} active · {activityAttribution.sceneContext.toLocaleString()} scene context · {activityAttribution.indeterminate.toLocaleString()} uncertain</small></dd></div>
            <div><dt>Context prevented from labeling incidents</dt><dd>{activityAttribution.enforced.toLocaleString()} <small>{activityAttribution.modes.size ? [...activityAttribution.modes].join(" / ").replaceAll("_", " ") : "waiting for detections"}</small></dd></div>
            <div><dt>Object admission · since restart</dt><dd>{activityAttribution.detectorAdmissions.toLocaleString()} detector-eligible <small>{activityAttribution.confidenceRejections.toLocaleString()} low confidence · {activityAttribution.zoneRejections.toLocaleString()} zone-rejected · {activityAttribution.temporalRejections.toLocaleString()} unconfirmed · {activityAttribution.enforced.toLocaleString()} scene context</small></dd></div>
          </dl>
        </section>
        {!selected ? <section className="telemetry-section">
          <div className="telemetry-section-head"><div><h3>Semantic search</h3></div></div>
          <dl className="telemetry-details">
            <div><dt>Status</dt><dd>{String(semantic.state || (semantic.enabled ? "starting" : "disabled")).replaceAll("_", " ")}{semantic.device ? ` · ${semantic.device}` : ""}</dd></div>
            <div><dt>Indexed incidents</dt><dd>{Number(semantic.event_count || 0).toLocaleString()}</dd></div>
            <div><dt>Search evidence</dt><dd>{Number(semantic.evidence_count || 0).toLocaleString()} <small>whole images and object crops</small></dd></div>
            <div><dt>Queue / added since restart</dt><dd>{Number(semantic.queue_depth || 0).toLocaleString()} / {Number(semantic.indexed_since_start || 0).toLocaleString()}</dd></div>
            {semantic.error || semantic.reason ? <div><dt>Last issue</dt><dd>{semantic.error || semantic.reason}</dd></div> : null}
          </dl>
        </section> : null}
        {!selected ? <section className="telemetry-section">
          <div className="telemetry-section-head"><div><h3>Face recognition</h3></div></div>
          <dl className="telemetry-details">
            <div><dt>Recognizable faces</dt><dd>{Number(faceRecognition.actionable_observations || 0).toLocaleString()} <small>{Number(faceRecognition.known || 0).toLocaleString()} identified · {Number(faceRecognition.unknown || 0).toLocaleString()} unknown</small></dd></div>
            <div><dt>Identification rate</dt><dd>{Number(faceRecognition.identified_percent || 0).toFixed(1)}%</dd></div>
            <div><dt>Unusable faces</dt><dd>{Number(faceRecognition.too_small || 0).toLocaleString()} <small>{Number(faceRecognition.processing_failed || 0).toLocaleString()} failures</small></dd></div>
            <div><dt>Candidate frames / multi-frame tracks</dt><dd>{Number(faceRecognition.candidate_frames || 0).toLocaleString()} / {Number(faceRecognition.multi_frame_tracks || 0).toLocaleString()}</dd></div>
            <div><dt>Recognition queue</dt><dd>{Number(faceRecognition.recognition?.queue_depth || 0).toLocaleString()} <small>{Number(faceRecognition.recognition?.pending || 0).toLocaleString()} pending · {Number(faceRecognition.recognition?.failed || 0).toLocaleString()} failed</small></dd></div>
          </dl>
        </section> : null}
      </div>

      {selected ? <section className="telemetry-section">
        <div className="telemetry-section-head"><div><h3>Camera configuration &amp; storage</h3></div></div>
        <div className="telemetry-camera-grid">
          {shownCameras.map((camera) => {
            const analysisRuntime = camera.motion?.analysis_runtime || {};
            const performance = camera.performance || {};
            const analyzed = Number(analysisRuntime.frames_sampled || 0);
            const superseded = Number(analysisRuntime.mailbox_replacements || camera.motion?.analysis_frames_dropped || 0);
            const objectActivity = camera.object_tracking?.object_activity_attribution || {};
            const onvifIssues = Number(camera.onvif?.poll_errors || 0) + Number(camera.onvif?.poll_timeouts || 0) + Number(camera.onvif?.renewal_errors || 0);
            const expected = camera.expected_enabled ?? (camera.lifecycle?.enabled !== false);
            const fresh = camera.connected && (camera.frame_fresh ?? Number(camera.last_frame_age_seconds || 0) <= 5);
            const cameraEventStatus = !camera.onvif?.enabled
              ? "Disabled"
              : !camera.onvif?.connected
                ? "Unavailable"
                : onvifIssues
                  ? `Connected · ${onvifIssues.toLocaleString()} recovered issues`
                  : "Healthy";
            const statusClass = !expected ? "disabled" : fresh ? "healthy" : camera.connected ? "attention" : "offline";
            const statusLabel = !expected ? "Paused" : fresh ? "Healthy" : camera.connected ? "Stale video" : "Offline";
            return <article className="telemetry-camera-card" key={camera.id}>
              <header><div><strong>{camera.name}</strong><small>{camera.id}</small></div><span className={statusClass}>{statusLabel}</span></header>
              <dl>
                <div><dt>Recording / detection</dt><dd>{camera.recording ? "On" : "Off"} / {camera.detection_enabled ? "On" : "Off"}</dd></div>
                <div><dt>Recording timeline</dt><dd>{formatRecorderTimestampHealth(camera.recording_timestamps)}</dd></div>
                <div><dt>Used-Recordings</dt><dd>{formatBytes(camera.storage?.recording_bytes)}</dd></div>
                <div><dt>Used-Snapshots</dt><dd>{formatBytes(camera.storage?.snapshot_bytes)}</dd></div>
                <div><dt>Processing health</dt><dd>{performance.summary || "Collecting a representative processing sample"}</dd></div>
                <div><dt>Camera event connection</dt><dd>{cameraEventStatus}</dd></div>
              </dl>
              <details className="telemetry-technical">
                <summary>Technical diagnostics</summary>
                <dl>
                  <div><dt>Lifecycle / workers</dt><dd>{camera.lifecycle?.phase || "unknown"} · {camera.lifecycle?.active_worker_count || 0} active</dd></div>
                  <div><dt>Live decoded FPS</dt><dd>{Number(camera.capture?.live?.fps || 0).toFixed(1)}</dd></div>
                  <div><dt>Main decoder starts</dt><dd>{Number(camera.capture?.main?.starts || 0).toLocaleString()}</dd></div>
                  <div><dt>Read / open failures</dt><dd>{Number(camera.capture?.live?.read_failures || 0) + Number(camera.capture?.main?.read_failures || 0)} / {Number(camera.capture?.live?.open_failures || 0) + Number(camera.capture?.main?.open_failures || 0)}</dd></div>
                  <div><dt>Capture-to-analysis p95 / p99</dt><dd>{formatMilliseconds(analysisRuntime.capture_to_analysis_p95_ms)} / {formatMilliseconds(analysisRuntime.capture_to_analysis_p99_ms)}</dd></div>
                  <div><dt>Performance gates</dt><dd>{(performance.checks || []).map((check) => `${check.label}: ${Number(check.value || 0).toFixed(check.unit === "%" ? 1 : 2)}${check.unit}`).join(" · ") || "Waiting for samples"}</dd></div>
                  <div><dt>Analyzed / stale skipped / deferred</dt><dd>{analyzed.toLocaleString()} / {superseded.toLocaleString()} / {Number(analysisRuntime.analysis_slot_deferrals || 0).toLocaleString()}</dd></div>
                  <div><dt>Motion passed / rejected / suppressed</dt><dd>{camera.motion?.passed || 0} / {camera.motion?.rejected || 0} / {camera.motion?.suppressed || 0}</dd></div>
                  <div><dt>Object admission / confidence / zone / confirmation / context</dt><dd>{Number(objectActivity.detector_admissions || 0).toLocaleString()} / {Number(objectActivity.confidence_rejections || 0).toLocaleString()} / {Number(objectActivity.zone_rejections || 0).toLocaleString()} / {Number(objectActivity.temporal_rejections || 0).toLocaleString()} / {Number(objectActivity.enforced_suppressions || 0).toLocaleString()}</dd></div>
                  <div><dt>Event queue peak / evicted / rejected / retry lost</dt><dd>{camera.motion?.event_runtime?.queue_high_water || 0} / {camera.motion?.event_runtime?.evicted || 0} / {camera.motion?.event_runtime?.rejected || 0} / {camera.motion?.event_runtime?.retries_dropped || 0}</dd></div>
                  <div><dt>EMA requests · admitted / merged / failed</dt><dd>{camera.motion?.event_runtime?.episode?.decision_counts?.request_admitted || 0} / {camera.motion?.event_runtime?.episode?.decision_counts?.merged_with_request || 0} / {camera.motion?.event_runtime?.episode?.decision_counts?.detector_failed || 0}</dd></div>
                  <div><dt>ONVIF notices / renewals / issues</dt><dd>{camera.onvif?.notifications || 0} / {camera.onvif?.renewals || 0} / {onvifIssues}</dd></div>
                  <div><dt>Tracking waits / longest / timeouts</dt><dd>{camera.tracking?.capacity_waits || 0} / {Number(camera.tracking?.capacity_wait_seconds_max || 0).toFixed(1)}s / {camera.tracking?.capacity_timeouts || 0}</dd></div>
                  <div><dt>ReID checks / recoveries / failures</dt><dd>{camera.tracking?.reid_attempts || 0} / {camera.tracking?.reid_recoveries || 0} / {camera.tracking?.reid_failures || 0}</dd></div>
                </dl>
              </details>
            </article>
          })}
        </div>
      </section> : null}
      <p className="telemetry-footnote">Availability, interruptions, EMA coverage, event delivery, and tracking capacity are the primary health signals. “Stale skipped” means a newer frame replaced an older pending sample so analysis stayed current; it matters only when coverage drops persistently. One-minute detail is retained for 48 hours, with compact summaries retained longer.</p>
    </div>
    </TelemetryInterruptionsContext.Provider>
  );
}

function MaintenanceViewer({ state }) {
  if (!state || state.status === "idle") {
    return <div className="empty-state">Run a storage scan to compare files on disk with SurvNG’s local databases.</div>;
  }
  if (state.status === "running" || state.status === "cancelling") {
    const progress = state.progress || {};
    const percent = Number.isFinite(progress.total) && progress.total > 0 ? Math.min(100, Math.round(Number(progress.current || 0) / progress.total * 100)) : null;
    return <div className="maintenance-running" role="status"><RefreshCcw className="spin" size={20} /><div><strong>{state.status === "cancelling" ? "Cancelling safely…" : state.mode === "repair" ? "Repairing storage records…" : state.full ? "Running full storage scan…" : "Running quick storage check…"}</strong><span>{progress.phase || "Starting"}{percent != null ? ` · ${percent}%` : progress.current ? ` · ${Number(progress.current).toLocaleString()} checked` : ""}</span></div></div>;
  }
  if (state.status === "cancelled") {
    return <div className="maintenance-result-banner warning"><CircleAlert size={20} /><div><strong>Maintenance cancelled</strong><span>No media files were deleted. Run a quick check whenever you are ready.</span></div></div>;
  }
  if (state.status === "failed") {
    return <div className="error-banner"><strong>Maintenance failed</strong><span>{state.error || "Check Logs for details."}</span></div>;
  }
  const result = state.result || {};
  const summary = result.summary || {};
  const repairs = result.repairs || {};
  const missingReferences = (summary.missing_event_snapshots || 0) + (summary.missing_event_recordings || 0) + (summary.missing_motion_snapshots || 0) + (summary.missing_face_snapshots || 0);
  const databaseIssues = (summary.missing_index_rows || 0) + (summary.unindexed_recording_files || 0) + missingReferences;
  const fullScan = result.full === true;
  const cameraRows = Object.entries(summary.per_camera || {});
  const repaired = Object.values(repairs).reduce((total, value) => total + (Number(value) || 0), 0);
  return (
    <div className="maintenance-viewer">
      <div className={`maintenance-result-banner ${databaseIssues ? "warning" : "healthy"}`}>
        {databaseIssues ? <CircleAlert size={20} /> : <CircleDot size={20} />}
        <div><strong>{databaseIssues ? `${databaseIssues.toLocaleString()} database mismatch${databaseIssues === 1 ? "" : "es"} found` : fullScan ? "Storage records are consistent" : "Quick check found no mismatches"}</strong><span>{fullScan ? "Full library checked." : `${Number(summary.index_rows_scanned || 0).toLocaleString()} newest index rows and ${Number(summary.recording_hours_scanned || 0)} recent recording hours checked.`} {result.note}</span></div>
      </div>
      <div className="telemetry-summary-grid maintenance-summary-grid">
        <article><span>{fullScan ? "Recording files" : "Recent files checked"}</span><strong>{Number(summary.recording_files || 0).toLocaleString()}</strong><small>{Number(summary.indexed_recordings || 0).toLocaleString()} total indexed · {Number(summary.recent_recording_files || 0).toLocaleString()} active/recent protected</small></article>
        <article><span>Recording index</span><strong>{Number(summary.missing_index_rows || 0).toLocaleString()} missing</strong><small>{Number(summary.unindexed_recording_files || 0).toLocaleString()} files need indexing</small></article>
        <article><span>Missing incident media</span><strong>{missingReferences.toLocaleString()}</strong><small>{summary.missing_event_snapshots || 0} incident · {summary.missing_motion_snapshots || 0} motion · {summary.missing_face_snapshots || 0} face images</small></article>
        <article><span>Unlinked media</span><strong>{fullScan ? Number(summary.orphan_media_files || 0).toLocaleString() : "Full scan"}</strong><small>{fullScan ? `${formatBytes(summary.orphan_media_bytes)} reported only; never auto-deleted` : "Not walked during the bounded quick check"}</small></article>
        <article><span>Regenerable cache</span><strong>{fullScan ? formatBytes(summary.regenerable_cache_bytes) : "Full scan"}</strong><small>Playback, event clip, and HLS working files</small></article>
        <article><span>Storage free</span><strong>{formatBytes(summary.storage_free_bytes)}</strong><small>{formatBytes(summary.storage_used_bytes)} used of {formatBytes(summary.storage_total_bytes)}</small></article>
      </div>
      {result.mode === "repair" ? <section className="telemetry-section"><div className="telemetry-section-head"><div><h3>Last repair</h3><p>{repaired.toLocaleString()} records updated; no incidents or media files were deleted.</p></div></div><dl className="telemetry-details maintenance-repair-details"><div><dt>Recording rows removed / added</dt><dd>{repairs.stale_index_rows_removed || 0} / {repairs.recordings_reindexed || 0}</dd></div><div><dt>Recordings validated / fingerprinted</dt><dd>{repairs.recordings_validated || 0} / {repairs.recording_fingerprints_added || 0}</dd></div><div><dt>Incident media links cleared</dt><dd>{repairs.event_media_references_cleared || 0}</dd></div><div><dt>Motion / face links cleared</dt><dd>{repairs.motion_sample_references_cleared || 0} / {repairs.face_media_references_cleared || 0}</dd></div></dl></section> : null}
      {cameraRows.length ? <section className="telemetry-section"><div className="telemetry-section-head"><div><h3>Affected cameras</h3><p>Missing media references grouped by camera.</p></div></div><div className="maintenance-camera-list">{cameraRows.map(([cameraId, counts]) => <div key={cameraId}><strong>{cameraId}</strong><span>{Object.entries(counts).map(([kind, count]) => `${String(kind).replaceAll("_", " ")} ${count}`).join(" · ")}</span></div>)}</div></section> : null}
      {(summary.missing_reference_samples?.length || summary.orphan_media_samples?.length || summary.missing_index_samples?.length || summary.unindexed_samples?.length) ? <details className="maintenance-details"><summary>Technical details and sample paths</summary><div>{summary.missing_reference_samples?.length ? <><h4>Missing media references</h4><pre>{summary.missing_reference_samples.map((item) => `${item.camera_id} · ${item.kind} · ${item.path}`).join("\n")}</pre></> : null}{summary.missing_index_samples?.length ? <><h4>Missing recording files still indexed</h4><pre>{summary.missing_index_samples.join("\n")}</pre></> : null}{summary.unindexed_samples?.length ? <><h4>Recording files not indexed</h4><pre>{summary.unindexed_samples.join("\n")}</pre></> : null}{summary.orphan_media_samples?.length ? <><h4>Unlinked media (report only)</h4><pre>{summary.orphan_media_samples.join("\n")}</pre></> : null}</div></details> : null}
    </div>
  );
}

function CalibrationLab({ cameras, timeZone }) {
  const [runs, setRuns] = useState([]);
  const [changeSets, setChangeSets] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [selectedRecommendations, setSelectedRecommendations] = useState([]);
  const [selectedCameras, setSelectedCameras] = useState(() => cameras.map((camera) => camera.id));
  const [mode, setMode] = useState("standard");
  const [evaluationHours, setEvaluationHours] = useState(24);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const selectedRun = runs.find((run) => run.id === selectedRunId) || runs[0] || null;

  async function loadCalibration() {
    try {
      const [runResponse, changeResponse] = await Promise.all([
        fetch("/api/calibration/runs?limit=20"),
        fetch("/api/calibration/change-sets?limit=50"),
      ]);
      if (!runResponse.ok || !changeResponse.ok) throw new Error("Calibration history could not be loaded");
      const [runPayload, changePayload] = await Promise.all([runResponse.json(), changeResponse.json()]);
      setRuns((current) => (runPayload.runs || []).map((run) => {
        const existing = current.find((item) => item.id === run.id);
        return existing?.result && Object.keys(existing.result).length
          ? { ...run, result: existing.result }
          : run;
      }));
      setChangeSets(changePayload.change_sets || []);
      setSelectedRunId((current) => current || runPayload.runs?.[0]?.id || null);
      setError("");
    } catch (loadError) {
      setError(loadError.message || "Calibration history could not be loaded");
    }
  }

  async function loadCalibrationRun(runId) {
    if (!runId) return;
    const response = await fetch(`/api/calibration/runs/${runId}`);
    if (!response.ok) throw new Error("Calibration run details could not be loaded");
    const run = await response.json();
    setRuns((current) => current.some((item) => item.id === run.id)
      ? current.map((item) => item.id === run.id ? run : item)
      : [run, ...current]);
  }

  useEffect(() => { void loadCalibration(); }, []);
  useEffect(() => {
    if (!selectedRunId) return;
    void loadCalibrationRun(selectedRunId).catch((loadError) => setError(loadError.message));
  }, [selectedRunId]);
  useEffect(() => {
    const activeRun = runs.some((run) => ["queued", "running"].includes(run.status));
    const activeEvaluation = changeSets.some((item) => item.status === "reviewing");
    if (!activeRun && !activeEvaluation) return undefined;
    const timer = window.setInterval(() => {
      void (async () => {
        await loadCalibration();
        if (selectedRunId) await loadCalibrationRun(selectedRunId);
      })().catch((loadError) => setError(loadError.message || "Calibration status could not be refreshed"));
    }, activeRun ? 2000 : 10000);
    return () => window.clearInterval(timer);
  }, [runs, changeSets, selectedRunId]);
  useEffect(() => {
    const ids = new Set(cameras.map((camera) => camera.id));
    setSelectedCameras((current) => current.filter((id) => ids.has(id)));
  }, [cameras]);
  useEffect(() => { setSelectedRecommendations([]); }, [selectedRunId]);

  async function startRun(override = false) {
    if (!selectedCameras.length) return setError("Select at least one camera.");
    if (mode === "deep" && !override && !window.confirm(`Deep analysis can review up to 40 images for each of ${selectedCameras.length} selected cameras and may use substantial AI API capacity. Continue?`)) return;
    setBusy(true); setError("");
    try {
      let response = await fetch("/api/calibration/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_ids: selectedCameras, mode, override_active_evaluation: override }),
      });
      let payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (response.status === 409 && !override && window.confirm(`${payload.detail}\n\nStart a new analysis anyway?`)) {
          response = await fetch("/api/calibration/runs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ camera_ids: selectedCameras, mode, override_active_evaluation: true }),
          });
          payload = await response.json().catch(() => ({}));
        }
      }
      if (!response.ok) {
        throw new Error(typeof payload.detail === "string" ? payload.detail : "Calibration could not start");
      }
      setSelectedRunId(payload.id);
      await loadCalibration();
    } catch (runError) { setError(runError.message || "Calibration could not start"); }
    finally { setBusy(false); }
  }

  async function applySelected() {
    if (!selectedRun || !selectedRecommendations.length) return;
    if (!window.confirm(`Apply ${selectedRecommendations.length} selected calibration change${selectedRecommendations.length === 1 ? "" : "s"}? SurvNG will validate one candidate configuration and reload only affected services.`)) return;
    setBusy(true); setError("");
    try {
      const response = await fetch(`/api/calibration/runs/${selectedRun.id}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recommendation_ids: selectedRecommendations, confirmed: true, configuration_fingerprint: selectedRun.result?.configuration_fingerprint || selectedRun.configuration_fingerprint, evaluation_hours: evaluationHours }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Calibration changes could not be applied");
      setSelectedRecommendations([]);
      await loadCalibration();
    } catch (applyError) { setError(applyError.message || "Calibration changes could not be applied"); }
    finally { setBusy(false); }
  }

  async function rollback(changeSet, { changeIds = [], cameraIds = [] } = {}) {
    const scopeLabel = changeIds.length ? "this setting" : cameraIds.length ? "this camera's settings" : "all settings";
    if (!window.confirm(`Roll back ${scopeLabel} from change set #${changeSet.id}? Newer conflicting values will be preserved.`)) return;
    setBusy(true); setError("");
    try {
      let response = await fetch(`/api/calibration/change-sets/${changeSet.id}/rollback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirmed: true, change_ids: changeIds, camera_ids: cameraIds, force_conflicts: false }) });
      let payload = await response.json().catch(() => ({}));
      if (response.status === 409 && payload.detail?.conflicts?.length && window.confirm(`${payload.detail.message}. Replace the ${payload.detail.conflicts.length} newer conflicting value${payload.detail.conflicts.length === 1 ? "" : "s"} anyway?`)) {
        response = await fetch(`/api/calibration/change-sets/${changeSet.id}/rollback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirmed: true, change_ids: changeIds, camera_ids: cameraIds, force_conflicts: true }) });
        payload = await response.json().catch(() => ({}));
      }
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : payload.detail?.message || "Rollback could not be completed");
      await loadCalibration();
    } catch (rollbackError) { setError(rollbackError.message || "Rollback could not be completed"); }
    finally { setBusy(false); }
  }

  async function evaluate(changeSet) {
    setBusy(true); setError("");
    try {
      const response = await fetch(`/api/calibration/change-sets/${changeSet.id}/evaluate`, { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Evaluation could not start");
      await loadCalibration();
    } catch (evaluationError) { setError(evaluationError.message || "Evaluation could not start"); }
    finally { setBusy(false); }
  }

  const recommendations = selectedRun?.result?.recommendations || [];
  return <>
    <section className="bento-card camera-tree config-tree settings-section-tree calibration-tree">
      <div className="section-head compact"><div><h2>Calibration Lab</h2><p>Analyze, apply, and roll back</p></div></div>
      <div className="calibration-run-controls">
        <label>Analysis depth<select value={mode} onChange={(event) => setMode(event.target.value)}><option value="quick">Quick · 24 hours</option><option value="standard">Standard · 7 days</option><option value="deep">Deep · 30 days</option></select></label>
        <div className="calibration-camera-select"><strong>Cameras</strong><button type="button" onClick={() => setSelectedCameras(selectedCameras.length === cameras.length ? [] : cameras.map((camera) => camera.id))}>{selectedCameras.length === cameras.length ? "Clear all" : "Select all"}</button></div>
        <div className="calibration-camera-list">{cameras.map((camera) => <label key={camera.id}><input type="checkbox" checked={selectedCameras.includes(camera.id)} onChange={(event) => setSelectedCameras((current) => event.target.checked ? [...current, camera.id] : current.filter((id) => id !== camera.id))} />{camera.name || camera.id}</label>)}</div>
        <button className="primary" type="button" onClick={() => void startRun()} disabled={busy || runs.some((run) => ["queued", "running"].includes(run.status))}><Sparkles size={16} />{busy ? "Working…" : "Analyze selected"}</button>
      </div>
      <div className="tree-list calibration-history">{runs.map((run) => <button type="button" className={(selectedRun?.id === run.id) ? "active" : ""} key={run.id} onClick={() => setSelectedRunId(run.id)}><Activity size={16} /><span><strong>Run #{run.id}</strong><small>{run.mode} · {run.status}</small></span></button>)}</div>
    </section>
    <section className="bento-card config-editor settings-panel calibration-panel">
      <div className="section-head"><div><h2>Calibration recommendations</h2><p>AI proposes; you choose and SurvNG validates</p></div><button onClick={() => void loadCalibration()}><RefreshCcw size={16} /> Refresh</button></div>
      {error ? <div className="error-banner">{error}</div> : null}
      {!selectedRun ? <div className="empty-state">Run an analysis to build camera-specific and system-wide recommendations.</div> : ["queued", "running"].includes(selectedRun.status) ? <div className="calibration-progress"><RefreshCcw className="spin" size={20} /><strong>Analyzing cameras</strong><span>{selectedRun.result?.progress?.completed || 0} of {selectedRun.result?.progress?.total || selectedRun.camera_ids?.length || 0} complete</span></div> : selectedRun.status === "failed" ? <div className="error-banner">{selectedRun.error || "Calibration failed"}</div> : <>
        <div className="calibration-summary"><ShieldCheck size={22} /><div><strong>{selectedRun.result?.summary}</strong><span>{recommendations.length} actionable recommendation{recommendations.length === 1 ? "" : "s"} · configuration fingerprint protected</span></div></div>
        {selectedRun.result?.advisories ? <div className="calibration-advisories">{Object.values(selectedRun.result.advisories).map((advisory) => <p key={advisory}>{advisory}</p>)}</div> : null}
        <div className="calibration-recommendations">{recommendations.length ? recommendations.map((item) => <article key={item.id} className={selectedRecommendations.includes(item.id) ? "selected" : ""}>
          <label className="calibration-recommendation-select"><input type="checkbox" checked={selectedRecommendations.includes(item.id)} onChange={(event) => setSelectedRecommendations((current) => event.target.checked ? [...current, item.id] : current.filter((id) => id !== item.id))} /><span>{item.scope === "global" ? "Global" : cameras.find((camera) => camera.id === item.camera_id)?.name || item.camera_id}</span><em>{item.subsystem}</em></label>
          <header><strong>{String(item.setting || "").replaceAll(".", " · ").replaceAll("_", " ")}</strong><code>{JSON.stringify(item.current_effective ?? item.current)} → {JSON.stringify(item.proposed)}</code></header>
          <p>{item.expected_benefit}</p><small><b>Potential downside:</b> {item.downside}</small>
          <footer><span>{item.evidence_strength} evidence · {item.support_count || 0} samples</span><span>{item.compute_impact}</span></footer>
          {item.effective_preview?.length > 1 ? <details className="calibration-effective-preview"><summary>Effective values for {item.effective_preview.length} cameras</summary>{item.effective_preview.map((camera) => <div key={camera.camera_id}><span>{cameras.find((item) => item.id === camera.camera_id)?.name || camera.camera_id}{camera.inherits ? " · inherits" : " · override"}</span><code>{JSON.stringify(camera.current)} → {JSON.stringify(camera.proposed)}</code></div>)}</details> : null}
          {item.evidence?.length ? <div className="calibration-evidence">{item.evidence.map((evidence, index) => evidence.image_url ? <a href={evidence.image_url} target="_blank" rel="noreferrer" key={`${evidence.id}-${index}`}>View evidence {index + 1}</a> : null)}</div> : null}
        </article>) : <div className="empty-state">The evidence did not support a bounded configuration change. Camera and stream-health findings remain available below.</div>}</div>
        {recommendations.length ? <div className="calibration-apply-bar"><label>Evaluate after<select value={evaluationHours} onChange={(event) => setEvaluationHours(Number(event.target.value))}><option value={24}>24 hours</option><option value={72}>3 days</option><option value={168}>7 days</option></select></label><button className="primary" onClick={() => void applySelected()} disabled={busy || !selectedRecommendations.length}><Check size={16} />Apply selected ({selectedRecommendations.length})</button></div> : null}
        <details className="calibration-camera-findings"><summary>Camera findings ({selectedRun.result?.camera_summaries?.length || 0})</summary>{selectedRun.result?.camera_summaries?.map((camera) => <article key={camera.camera_id}><strong>{camera.camera_name}</strong><span>{camera.summary}</span><small>{camera.analyzed} reviewed · {camera.failed} failed</small></article>)}</details>
      </>}
      <div className="calibration-change-history"><h3>Applied change sets</h3>{changeSets.length ? changeSets.map((item) => { const rolledBack = new Set(item.rolled_back_change_ids || []); const remaining = (item.changes || []).filter((change) => !rolledBack.has(change.id)); return <article key={item.id}><div><strong>#{item.id} · {item.action}</strong><span>{formatDateTime(item.created_at, timeZone)} · {String(item.status).replaceAll("_", " ")}</span></div><small>{item.changes?.length || 0} setting changes{rolledBack.size ? ` · ${rolledBack.size} rolled back` : ""}{item.evaluation?.summary ? ` · ${item.evaluation.summary}` : ""}</small><div>{item.action === "apply" && item.status === "collecting" && item.seconds_until_ready <= 0 ? <button onClick={() => void evaluate(item)} disabled={busy}><Activity size={15} />Evaluate</button> : null}{item.action === "apply" && remaining.length ? <button onClick={() => void rollback(item)} disabled={busy}><Undo2 size={15} />Rollback remaining</button> : null}</div>{item.action === "apply" && item.changes?.length ? <details className="calibration-change-details"><summary>Selective rollback</summary>{Object.entries(item.changes.reduce((groups, change) => { const key = change.camera_id || "Global"; return { ...groups, [key]: [...(groups[key] || []), change] }; }, {})).map(([cameraId, changes]) => { const remainingGroup = changes.filter((change) => !rolledBack.has(change.id)); return <section key={cameraId}><header><strong>{cameraId === "Global" ? "Global settings" : cameras.find((camera) => camera.id === cameraId)?.name || cameraId}</strong><button onClick={() => void rollback(item, { changeIds: remainingGroup.map((change) => change.id) })} disabled={busy || !remainingGroup.length}><Undo2 size={14} />Rollback group</button></header>{changes.map((change) => <div key={change.id}><span>{String(change.setting).replaceAll(".", " · ").replaceAll("_", " ")}</span><code>{JSON.stringify(change.before)} → {JSON.stringify(change.after)}</code><button onClick={() => void rollback(item, { changeIds: [change.id] })} disabled={busy || rolledBack.has(change.id)}>{rolledBack.has(change.id) ? "Rolled back" : "Rollback"}</button></div>)}</section>; })}</details> : null}</article>; }) : <div className="empty-state">No calibration changes have been applied.</div>}</div>
    </section>
  </>;
}

function ConfigPage({ timeZone, setTimeZone, theme, setTheme, onAssistantContextChange }) {
  const [config, setConfig] = useState(null);
  const [runtimeStatus, setRuntimeStatus] = useState([]);
  const [accelerator, setAccelerator] = useState(null);
  const [detectorModels, setDetectorModels] = useState([]);
  const [recordingCache, setRecordingCache] = useState(null);
  const [retentionStatus, setRetentionStatus] = useState(null);
  const [retentionError, setRetentionError] = useState("");
  const [mqttStatus, setMqttStatus] = useState(null);
  const [detectorStatus, setDetectorStatus] = useState(null);
  const [motionCatalog, setMotionCatalog] = useState(null);
  const [settingsTab, setSettingsTab] = useStoredState("survng.configTab", "general");
  const [generalSection, setGeneralSection] = useStoredState("survng.generalSection.v1", "general");
  const [cameraSection, setCameraSection] = useStoredState("survng.cameraSection.v1", "settings");
  const [selectedId, setSelectedId] = useState("");
  const [saveNotice, setSaveNotice] = useState(null);
  const [configLoadError, setConfigLoadError] = useState("");
  const [generalSaving, setGeneralSaving] = useState(false);
  const [zonesSaving, setZonesSaving] = useState(false);
  const [cameraSaving, setCameraSaving] = useState(false);
  const [cameraOrderEditing, setCameraOrderEditing] = useState(false);
  const [cameraOrderSaving, setCameraOrderSaving] = useState(false);
  const [dragConfigCameraId, setDragConfigCameraId] = useState("");
  const [dragConfigCameraTarget, setDragConfigCameraTarget] = useState("");
  const [dragConfigCameraAfter, setDragConfigCameraAfter] = useState(false);
  const cameraOrderOriginalRef = useRef([]);
  const [probe, setProbe] = useState(null);
  const [logLines, setLogLines] = useState([]);
  const [logFilter, setLogFilter] = useStoredState("survng.logFilter.v1", "");
  const [logLevel, setLogLevel] = useStoredState("survng.logLevel.v1", "INFO");
  const [logOrder, setLogOrder] = useStoredState("survng.logOrder.v1", "newest");
  const [auditItems, setAuditItems] = useState([]);
  const [auditTotal, setAuditTotal] = useState(0);
  const [auditPage, setAuditPage] = useState(0);
  const [auditCamera, setAuditCamera] = useStoredState("survng.motionAuditCamera.v1", "");
  const [auditCategory, setAuditCategory] = useStoredState("survng.motionAuditCategory.v1", "all");
  const [auditOutcome, setAuditOutcome] = useStoredState("survng.motionAuditOutcome.v1", "all");
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditError, setAuditError] = useState("");
  const [selectedAuditId, setSelectedAuditId] = useState(null);
  const [linkedAudit, setLinkedAudit] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [telemetryLoading, setTelemetryLoading] = useState(false);
  const [telemetryError, setTelemetryError] = useState("");
  const [telemetrySection, setTelemetrySection] = useStoredState("survng.telemetrySection.v1", "overview");
  const [telemetryCamera, setTelemetryCamera] = useStoredState("survng.telemetryCamera.v1", "");
  const [diagnosticScope, setDiagnosticScope] = useState("system");
  const [diagnosticDuration, setDiagnosticDuration] = useState("3600");
  const [maintenance, setMaintenance] = useState(null);
  const [maintenanceError, setMaintenanceError] = useState("");
  const configLoadSequence = useRef(0);
  const auditPageSize = 24;

  useEffect(() => {
    onAssistantContextChange?.({
      page: "config",
      camera_id: settingsTab === "cameras" ? selectedId : "",
      filters: { section: settingsTab, general_section: generalSection, camera_section: cameraSection },
    });
  }, [cameraSection, generalSection, onAssistantContextChange, selectedId, settingsTab]);

  async function load() {
    const sequence = ++configLoadSequence.current;
    setConfigLoadError("");
    try {
      const response = await fetch("/api/config");
      if (!response.ok) throw new Error(`Configuration failed to load (${response.status})`);
      const nextConfig = await response.json();
      if (sequence !== configLoadSequence.current) return false;
      setConfig(nextConfig);
      setSelectedId((current) => nextConfig.cameras?.some((camera) => camera.id === current) ? current : nextConfig.cameras?.[0]?.id || "");

      // These values enrich individual cards but are not required to render
      // editable configuration. Load them independently so a slow storage or
      // hardware status probe cannot strand the entire Admin page.
      const optionalPayload = async (path, timeoutMs = 5000) => {
        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
        try {
          const optionalResponse = await fetch(path, { signal: controller.signal });
          if (!optionalResponse.ok) return null;
          return await optionalResponse.json();
        } catch {
          return null;
        } finally {
          window.clearTimeout(timeout);
        }
      };
      void Promise.all([
        optionalPayload("/api/cameras"),
        optionalPayload("/api/accelerator"),
        optionalPayload("/api/detector/models"),
        optionalPayload("/api/recordings/cache/status"),
        optionalPayload("/api/system/status"),
        optionalPayload("/api/motion/pipeline/catalog"),
        optionalPayload("/api/retention/status"),
      ]).then(([status, acceleratorPayload, models, cache, system, catalog, retention]) => {
        if (sequence !== configLoadSequence.current) return;
        if (Array.isArray(status)) setRuntimeStatus(status);
        if (acceleratorPayload) setAccelerator(acceleratorPayload);
        if (models) setDetectorModels(models.models || []);
        if (cache) setRecordingCache(cache);
        if (system) {
          setMqttStatus(system.mqtt || null);
          setDetectorStatus(system.detector || null);
        }
        if (catalog) setMotionCatalog(catalog);
        if (retention) setRetentionStatus(retention);
      });
      return true;
    } catch (error) {
      if (sequence === configLoadSequence.current) setConfigLoadError(error.message || "Configuration failed to load");
      return false;
    }
  }

  useEffect(() => {
    void load();
    return () => { configLoadSequence.current += 1; };
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("section") !== "audit") return;
    setSettingsTab("audit");
    const auditId = Number(params.get("audit_id"));
    if (!Number.isInteger(auditId) || auditId <= 0) return;
    let active = true;
    fetch(`/api/motion-audit/${auditId}`)
      .then((response) => {
        if (!response.ok) throw new Error(`Motion audit failed to load (${response.status})`);
        return response.json();
      })
      .then((item) => {
        if (!active) return;
        setLinkedAudit(item);
        setSelectedAuditId(item.id);
      })
      .catch((error) => {
        if (active) setAuditError(error.message || "Unable to open the selected motion audit.");
      });
    return () => { active = false; };
  }, [setSettingsTab]);

  async function loadRetention() {
    try {
      const response = await fetch("/api/retention/status");
      if (!response.ok) throw new Error(`Retention status failed (${response.status})`);
      setRetentionStatus(await response.json());
      setRetentionError("");
    } catch (error) {
      setRetentionError(error.message || "Unable to load retention status.");
    }
  }

  async function runRetention(apply = false) {
    if (apply && !window.confirm("Apply the current retention plan? This deletes eligible continuous recordings and incident snapshots older than their configured retention. Pinned face references, databases, motion-audit evidence, and the newest five minutes of recordings remain protected.")) return;
    setRetentionError("");
    try {
      const response = await fetch("/api/retention/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apply }),
      });
      if (!response.ok) throw new Error(`Retention run failed (${response.status})`);
      setRetentionStatus(await response.json());
    } catch (error) {
      setRetentionError(error.message || "Unable to start retention.");
    }
  }

  useEffect(() => {
    if (settingsTab !== "general" || generalSection !== "storage") return undefined;
    void loadRetention();
    const timer = window.setInterval(() => void loadRetention(), 5000);
    return () => window.clearInterval(timer);
  }, [settingsTab, generalSection]);

  useEffect(() => {
    if (settingsTab !== "cameras") return undefined;
    let active = true;
    async function refreshCameraRuntime() {
      try {
        const response = await fetch("/api/cameras");
        if (active && response.ok) setRuntimeStatus(await response.json());
      } catch {
        // Keep the last known runtime state and retry on the next interval.
      }
    }
    refreshCameraRuntime();
    const timer = window.setInterval(refreshCameraRuntime, 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [settingsTab]);


  async function loadLogs() {
    try {
      const params = new URLSearchParams({ limit: "500", level: logLevel, q: logFilter });
      const response = await fetch(`/api/logs?${params.toString()}`);
      if (response.ok) {
        const payload = await response.json();
        setLogLines(payload.lines || []);
      }
    } catch {
      // Preserve the current log view; polling retries automatically.
    }
  }

  useEffect(() => {
    if (settingsTab !== "logs") return undefined;
    loadLogs();
    const timer = window.setInterval(loadLogs, 2000);
    return () => window.clearInterval(timer);
  }, [settingsTab, logLevel, logFilter]);

  async function loadMotionAudit(page = auditPage) {
    setAuditLoading(true);
    setAuditError("");
    try {
      const params = new URLSearchParams({
        limit: String(auditPageSize),
        offset: String(page * auditPageSize),
        outcome: auditOutcome,
        category: auditCategory,
      });
      if (auditCamera) params.set("camera_id", auditCamera);
      const response = await fetch(`/api/motion-audit?${params.toString()}`);
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      setAuditItems(payload.items || []);
      setAuditTotal(Number(payload.total) || 0);
    } catch (error) {
      setAuditError(error.message || "Unable to load motion audit entries.");
    } finally {
      setAuditLoading(false);
    }
  }

  useEffect(() => {
    if (settingsTab !== "audit") return undefined;
    loadMotionAudit(auditPage);
    if (auditPage !== 0) return undefined;
    const timer = window.setInterval(() => loadMotionAudit(0), 10000);
    return () => window.clearInterval(timer);
  }, [settingsTab, auditPage, auditCamera, auditCategory, auditOutcome]);

  async function loadTelemetry() {
    setTelemetryLoading(true);
    setTelemetryError("");
    try {
      const params = new URLSearchParams({ hours: "24" });
      const cameraId = telemetryCamera || config?.cameras?.[0]?.id || "";
      if (telemetrySection === "cameras" && cameraId) params.set("camera_id", cameraId);
      const response = await fetch(`/api/telemetry?${params.toString()}`);
      if (!response.ok) throw new Error(`Telemetry failed to load (${response.status})`);
      setTelemetry(await response.json());
    } catch (error) {
      setTelemetryError(error.message || "Unable to load telemetry.");
    } finally {
      setTelemetryLoading(false);
    }
  }

  async function startTelemetryDiagnostics() {
    const cameraId = telemetryCamera || config?.cameras?.[0]?.id || "";
    const response = await fetch("/api/telemetry/diagnostics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scope: diagnosticScope,
        camera_id: diagnosticScope === "camera" ? cameraId : "",
        duration_seconds: Number(diagnosticDuration),
      }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      setTelemetryError(payload.detail || `Unable to start diagnostics (${response.status})`);
      return;
    }
    await loadTelemetry();
  }

  async function stopTelemetryDiagnostics(sessionId) {
    const response = await fetch(`/api/telemetry/diagnostics/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    if (!response.ok) {
      setTelemetryError(`Unable to stop diagnostics (${response.status})`);
      return;
    }
    await loadTelemetry();
  }

  useEffect(() => {
    if (settingsTab !== "telemetry") return undefined;
    void loadTelemetry();
    const timer = window.setInterval(() => void loadTelemetry(), 10000);
    return () => window.clearInterval(timer);
  }, [settingsTab, telemetryCamera, telemetrySection]);

  async function loadMaintenance() {
    try {
      const response = await fetch("/api/maintenance/storage");
      if (!response.ok) throw new Error(`Maintenance status failed to load (${response.status})`);
      setMaintenance(await response.json());
      setMaintenanceError("");
    } catch (error) {
      setMaintenanceError(error.message || "Unable to load maintenance status.");
    }
  }

  async function startMaintenance(apply = false, full = false) {
    if (full && !apply && !window.confirm("A full scan walks the entire NFS media library and may take a long time. You can cancel it at any point. Continue?")) return;
    if (apply && !window.confirm(`Repair the ${full ? "full" : "recent"} database findings now? Incident history and media files will not be deleted.`)) return;
    setMaintenanceError("");
    try {
      const response = await fetch("/api/maintenance/storage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apply, full }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Maintenance could not start (${response.status})`);
      }
      setMaintenance(await response.json());
    } catch (error) {
      setMaintenanceError(error.message || "Unable to start storage maintenance.");
    }
  }

  async function cancelMaintenance() {
    try {
      const response = await fetch("/api/maintenance/storage", { method: "DELETE" });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Maintenance could not be cancelled (${response.status})`);
      }
      setMaintenance(await response.json());
    } catch (error) {
      setMaintenanceError(error.message || "Unable to cancel storage maintenance.");
    }
  }

  useEffect(() => {
    if (settingsTab !== "maintenance") return undefined;
    void loadMaintenance();
    const timer = window.setInterval(() => void loadMaintenance(), ["running", "cancelling"].includes(maintenance?.status) ? 1000 : 5000);
    return () => window.clearInterval(timer);
  }, [settingsTab, maintenance?.status]);

  const cameras = config?.cameras || [];
  const selectedTelemetryCamera = cameras.some((camera) => camera.id === telemetryCamera)
    ? telemetryCamera
    : cameras[0]?.id || "";
  const selectedCamera = cameras.find((camera) => camera.id === selectedId) || cameras[0] || null;
  const selectedRuntimeStatus = runtimeStatus.find((camera) => camera.id === selectedCamera?.id);
  const selectedAudit = auditItems.find((item) => item.id === selectedAuditId)
    || (linkedAudit?.id === selectedAuditId ? linkedAudit : null);
  const selectedAuditItems = selectedAudit && !auditItems.some((item) => item.id === selectedAudit.id)
    ? [selectedAudit, ...auditItems]
    : auditItems;
  const activeDetectorPath = config?.detector?.model_path || config?.detector?.model_xml || "";
  const activeDetectorModel = detectorModels.find((model) => model.path === activeDetectorPath);
  const zoneClassOptions = activeDetectorModel?.classes?.length
    ? activeDetectorModel.classes
    : config?.detector?.labels || [];



  if (!config) {
    return <main className="bento-grid config-grid"><section className="bento-card config-editor"><div className="empty-state">{configLoadError || "Loading config..."}{configLoadError ? <button type="button" onClick={() => void load()}><RefreshCcw size={15} /> Retry</button> : null}</div></section></main>;
  }

  function updateConfig(path, value) {
    setSaveNotice(null);
    setConfig((current) => {
      const next = structuredClone(current);
      let target = next;
      for (let index = 0; index < path.length - 1; index += 1) target = target[path[index]];
      target[path[path.length - 1]] = value;
      return next;
    });
  }

  function updateCamera(cameraId, path, value) {
    setSaveNotice(null);
    setConfig((current) => {
      const next = structuredClone(current);
      const camera = next.cameras.find((item) => item.id === cameraId);
      let target = camera;
      for (let index = 0; index < path.length - 1; index += 1) target = target[path[index]];
      target[path[path.length - 1]] = value;
      return next;
    });
  }



  function addCamera(seed = {}) {
    const camera = cameraWithDerivedConnection(defaultCamera(cameras, seed));
    setConfig((current) => ({ ...current, cameras: [...(current.cameras || []), camera] }));
    setSelectedId(camera.id);
    setProbe(null);
  }

  function cloneCamera(camera) {
    addCamera(camera);
  }

  function removeCamera(cameraId) {
    const nextCameras = cameras.filter((camera) => camera.id !== cameraId);
    setConfig((current) => ({ ...current, cameras: nextCameras }));
    setSelectedId(nextCameras[0]?.id || "");
    setProbe(null);
  }

  function startCameraOrderEdit() {
    cameraOrderOriginalRef.current = cameras.map((camera) => camera.id);
    setCameraOrderEditing(true);
    setSaveNotice(null);
  }

  function cancelCameraOrderEdit() {
    const originalOrder = cameraOrderOriginalRef.current;
    setConfig((current) => {
      const cameraById = new Map((current.cameras || []).map((camera) => [camera.id, camera]));
      const ordered = originalOrder.map((cameraId) => cameraById.get(cameraId)).filter(Boolean);
      const seen = new Set(ordered.map((camera) => camera.id));
      return { ...current, cameras: [...ordered, ...(current.cameras || []).filter((camera) => !seen.has(camera.id))] };
    });
    setCameraOrderEditing(false);
    setDragConfigCameraId("");
    setDragConfigCameraTarget("");
  }

  function moveConfigCamera(sourceId, targetId, after) {
    if (!sourceId || !targetId || sourceId === targetId) return;
    setConfig((current) => {
      const nextCameras = [...(current.cameras || [])];
      const sourceIndex = nextCameras.findIndex((camera) => camera.id === sourceId);
      if (sourceIndex < 0) return current;
      const [source] = nextCameras.splice(sourceIndex, 1);
      const targetIndex = nextCameras.findIndex((camera) => camera.id === targetId);
      if (targetIndex < 0) return current;
      nextCameras.splice(targetIndex + (after ? 1 : 0), 0, source);
      return { ...current, cameras: nextCameras };
    });
  }

  async function saveCameraOrder() {
    if (cameraOrderSaving) return;
    setCameraOrderSaving(true);
    setSaveNotice({ state: "saving", text: "Saving default camera order..." });
    try {
      const response = await fetch("/api/config/cameras/order", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cameras.map((camera) => camera.id)),
      });
      if (!response.ok) throw new Error(await response.text());
      cameraOrderOriginalRef.current = cameras.map((camera) => camera.id);
      setCameraOrderEditing(false);
      setSaveNotice({ state: "saved", text: "Default live-view order saved." });
    } catch (error) {
      setSaveNotice({ state: "error", text: error.message || "Unable to save camera order." });
    } finally {
      setCameraOrderSaving(false);
      setDragConfigCameraId("");
      setDragConfigCameraTarget("");
    }
  }

  async function save() {
    if (generalSaving) return;
    const ids = new Set();
    const configToSave = {
      ...config,
      cameras: camerasWithGeneratedIds(config.cameras || []),
    };
    for (const camera of configToSave.cameras || []) {
      if (ids.has(camera.id)) {
        setSaveNotice({ state: "error", text: `Duplicate camera ID "${camera.id}". Fix duplicates before saving.` });
        return;
      }
      ids.add(camera.id);
    }
    const mediaStorageError = mediaStorageConfigurationError(configToSave.media_storage);
    if (mediaStorageError) {
      setSaveNotice({ state: "error", text: mediaStorageError });
      return;
    }
    setGeneralSaving(true);
    setSaveNotice({ state: "saving", text: "Saving settings..." });
    try {
      const response = await fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(configToSave),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const detail = typeof payload.detail === "string"
          ? payload.detail
          : `Configuration could not be saved (${response.status}).`;
        throw new Error(detail);
      }
      const payload = await response.json();
      const reloaded = await load();
      setSaveNotice(reloaded
        ? {
          state: "saved",
          text: payload.camera_workers_restarted
            ? payload.camera_ids_restarted?.length === 1
              ? `Saved. ${payload.camera_ids_restarted[0]} motion runtime reloaded; other cameras kept running.`
              : `Saved. ${payload.camera_ids_restarted?.length || "Affected"} camera motion runtimes reloaded.`
            : payload.subsystems_restarted?.includes("recorders") && payload.subsystems_restarted?.includes("mqtt")
              ? "Saved. Recorders restarted and MQTT reconnected; cameras kept running."
              : payload.subsystems_restarted?.includes("recorders")
                ? "Saved. Recorder processes restarted; cameras kept running."
                : payload.subsystems_restarted?.includes("mqtt")
                  ? "Saved. MQTT reconnected; cameras kept running."
                  : payload.subsystems_restarted?.some((name) => (
                    name === "tracking_sessions" || name.endsWith("_inference")
                  ))
                    ? "Saved. Detection services refreshed; camera streams kept running."
              : "Saved without interrupting cameras.",
        }
        : { state: "error", text: "Saved, but the refreshed configuration could not be loaded. Retry this page." });
    } catch (error) {
      setSaveNotice({ state: "error", text: error.message || "Unable to save general settings." });
    } finally {
      setGeneralSaving(false);
    }
  }

  async function saveZones(camera) {
    if (!camera || zonesSaving) return;
    setZonesSaving(true);
    setSaveNotice({ state: "saving", text: "Saving zones..." });
    try {
      const response = await fetch(`/api/config/cameras/${encodeURIComponent(camera.id)}/zones`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(camera.zones || []),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      updateCamera(camera.id, ["zones"], payload.zones || []);
      setSaveNotice({ state: "saved", text: "Zones saved without restarting cameras." });
      const statusResponse = await fetch("/api/cameras");
      if (statusResponse.ok) setRuntimeStatus(await statusResponse.json());
    } catch (error) {
      setSaveNotice({ state: "error", text: error.message || "Unable to save zones." });
    } finally {
      setZonesSaving(false);
    }
  }

  async function saveCamera(camera) {
    if (!camera || cameraSaving) return;
    setCameraSaving(true);
    setSaveNotice({ state: "saving", text: "Saving camera settings..." });
    try {
      const response = await fetch(`/api/config/cameras/${encodeURIComponent(camera.id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(camera),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      const savedCamera = payload.camera;
      setConfig((current) => ({
        ...current,
        cameras: current.cameras.map((item) => item.id === camera.id
          ? { ...savedCamera, zones: item.zones || [] }
          : item),
      }));
      setSelectedId(savedCamera.id);
      setSaveNotice({
        state: "saved",
        text: payload.camera_workers_restarted
          ? `${savedCamera.name || savedCamera.id} motion runtime reloaded; other cameras kept running.`
          : "Camera settings saved without interrupting camera streams.",
      });
      const statusResponse = await fetch("/api/cameras");
      if (statusResponse.ok) setRuntimeStatus(await statusResponse.json());
    } catch (error) {
      setSaveNotice({ state: "error", text: error.message || "Unable to save camera settings." });
    } finally {
      setCameraSaving(false);
    }
  }

  async function deleteCamera(camera) {
    if (!camera || cameraSaving || !window.confirm(`Remove ${camera.name || camera.id}?`)) return;
    const isPersisted = runtimeStatus.some((item) => item.id === camera.id);
    if (!isPersisted) {
      removeCamera(camera.id);
      setSaveNotice({ state: "saved", text: "Unsaved camera removed." });
      return;
    }
    setCameraSaving(true);
    setSaveNotice({ state: "saving", text: "Removing camera..." });
    try {
      const response = await fetch(`/api/config/cameras/${encodeURIComponent(camera.id)}`, { method: "DELETE" });
      if (!response.ok) throw new Error(await response.text());
      removeCamera(camera.id);
      setSaveNotice({ state: "saved", text: "Camera removed. Workers reloaded." });
      const statusResponse = await fetch("/api/cameras");
      if (statusResponse.ok) setRuntimeStatus(await statusResponse.json());
    } catch (error) {
      setSaveNotice({ state: "error", text: error.message || "Unable to remove camera." });
    } finally {
      setCameraSaving(false);
    }
  }

  async function probeCamera(camera) {
    setProbe({ loading: true });
    setCameraSection("info");
    setCameraSection("info");
    const probeCameraConfig = cameraWithDerivedConnection(camera);
    if (probeCameraConfig !== camera) {
      setConfig((current) => ({
        ...current,
        cameras: (current.cameras || []).map((item) => item.id === camera.id ? probeCameraConfig : item),
      }));
    }
    const host = probeCameraConfig.onvif?.host || probeCameraConfig.baichuan?.host || "";
    const username = probeCameraConfig.onvif?.username || probeCameraConfig.baichuan?.username || "";
    const password = probeCameraConfig.onvif?.password || probeCameraConfig.baichuan?.password || "";
    try {
      const response = await fetch("/api/config/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          camera_id: probeCameraConfig.id,
          host,
          username,
          password,
          onvif_port: probeCameraConfig.onvif?.port || 8000,
          baichuan_port: probeCameraConfig.baichuan?.port || 9000,
        }),
      });
      if (!response.ok) throw new Error(`Camera probe failed (${response.status})`);
      const result = await response.json();
      setProbe(result);
      if (result.onvif?.reachable) updateCamera(camera.id, ["onvif", "enabled"], true);
      if (result.baichuan?.reachable) {
        updateCamera(camera.id, ["baichuan", "enabled"], true);
        updateCamera(camera.id, ["video_backend"], "baichuan_native");
      }
    } catch (error) {
      setProbe({ loading: false, error: error.message || "Camera probe failed" });
    }
  }

  return (
    <main className="bento-grid config-grid settings-grid">
      <div className="settings-tabs">
        <div className="settings-tab-list" role="tablist" aria-label="Config sections">
          <button className={settingsTab === "general" ? "active" : ""} onClick={() => setSettingsTab("general")} role="tab" aria-selected={settingsTab === "general"}><Cog size={16} /> General</button>
          <button className={settingsTab === "cameras" ? "active" : ""} onClick={() => setSettingsTab("cameras")} role="tab" aria-selected={settingsTab === "cameras"}><Camera size={16} /> Camera Settings</button>
          <button className={settingsTab === "audit" ? "active" : ""} onClick={() => setSettingsTab("audit")} role="tab" aria-selected={settingsTab === "audit"}><Activity size={16} /> Motion Audit</button>
          <button className={settingsTab === "calibration" ? "active" : ""} onClick={() => setSettingsTab("calibration")} role="tab" aria-selected={settingsTab === "calibration"}><Sparkles size={16} /> Calibration Lab</button>
          <button className={settingsTab === "telemetry" ? "active" : ""} onClick={() => setSettingsTab("telemetry")} role="tab" aria-selected={settingsTab === "telemetry"}><Gauge size={16} /> Telemetry</button>
          <button className={settingsTab === "maintenance" ? "active" : ""} onClick={() => setSettingsTab("maintenance")} role="tab" aria-selected={settingsTab === "maintenance"}><Wrench size={16} /> Maintenance</button>
          <button className={settingsTab === "logs" ? "active" : ""} onClick={() => setSettingsTab("logs")} role="tab" aria-selected={settingsTab === "logs"}><ListTree size={16} /> Logs</button>
        </div>
        {saveNotice ? <span className={`camera-save-indicator settings-header-status ${saveNotice.state}`} role="status">{saveNotice.state === "saving" ? <RefreshCcw size={14} /> : saveNotice.state === "error" ? <CircleAlert size={14} /> : <CircleDot size={14} />}{saveNotice.text}</span> : null}
      </div>

      {settingsTab === "general" ? (
        <>
        <section className="bento-card camera-tree config-tree settings-section-tree">
          <div className="section-head compact"><div><h2>General</h2><p>Configuration sections</p></div></div>
          <div className="tree-list">
            <button type="button" className={generalSection === "general" ? "active" : ""} onClick={() => setGeneralSection("general")}><Cog size={16} /><span>General</span></button>
            <button type="button" className={generalSection === "storage" ? "active" : ""} onClick={() => setGeneralSection("storage")}><HardDrive size={16} /><span>Storage</span></button>
            <button type="button" className={generalSection === "mqtt" ? "active" : ""} onClick={() => setGeneralSection("mqtt")}><Radio size={16} /><span>API</span></button>
            <button type="button" className={generalSection === "detection" ? "active" : ""} onClick={() => setGeneralSection("detection")}><Cpu size={16} /><span>Object Detection</span></button>
            <button type="button" className={generalSection === "motion-review" ? "active" : ""} onClick={() => setGeneralSection("motion-review")}><Sparkles size={16} /><span>Camera Intelligence</span></button>
          </div>
        </section>
        <section className="bento-card config-editor settings-panel">
          <div className="section-head">
            <div><h2>General</h2><p>Application preferences and detector settings</p></div>
            <div className="camera-command-area">
              <button className="primary camera-save-button" onClick={save} disabled={generalSaving}>
                {generalSaving ? <RefreshCcw className="spin" size={16} /> : <Save size={16} />}
                {generalSaving ? "Saving..." : "Save"}
              </button>
            </div>
          </div>
          <GeneralSettings
            config={config}
            updateConfig={updateConfig}
            timeZone={timeZone}
            setTimeZone={setTimeZone}
            theme={theme}
            setTheme={setTheme}
            accelerator={accelerator}
            detectorModels={detectorModels}
            recordingCache={recordingCache}
            retentionStatus={retentionStatus}
            retentionError={retentionError}
            runRetention={runRetention}
            mqttStatus={mqttStatus}
            detectorStatus={detectorStatus}
            motionCatalog={motionCatalog}
            section={generalSection}
          />
        </section>
        </>
      ) : settingsTab === "audit" ? (
        <>
        <section className="bento-card camera-tree config-tree settings-section-tree motion-audit-filters">
          <div className="section-head compact"><div><h2>Motion Audit</h2><p>{auditTotal.toLocaleString()} matching decisions</p></div></div>
          <div className="motion-audit-filter-fields">
            <label>Camera<select value={auditCamera} onChange={(event) => { setAuditCamera(event.target.value); setAuditPage(0); }}>
              <option value="">All cameras</option>
              {cameras.map((camera) => <option value={camera.id} key={camera.id}>{camera.name || camera.id}</option>)}
            </select></label>
            <label>Category<select value={auditCategory} onChange={(event) => { setAuditCategory(event.target.value); setAuditPage(0); }}>
              <option value="all">All categories</option>
              <option value="visual_backup">Visual backup</option>
              <option value="active_followup">Active-event follow-up</option>
              <option value="qualification">Filtered motion</option>
            </select></label>
          </div>
          <div className="tree-list">
            {[
              ["all", "All outcomes"],
              ["object", "Object found"],
              ["clear", "No object"],
              ["not_run", "Detection skipped"],
            ].map(([value, label]) => (
              <button type="button" className={auditOutcome === value ? "active" : ""} key={value} onClick={() => { setAuditOutcome(value); setAuditPage(0); }}><Activity size={16} /><span>{label}</span></button>
            ))}
          </div>
        </section>
        <section className="bento-card config-editor settings-panel motion-audit-panel">
          <div className="section-head">
            <div><h2>{auditCategory === "visual_backup" ? "Visual Backup" : auditCategory === "active_followup" ? "Active-Event Follow-Up" : auditCategory === "qualification" ? "Filtered Motion" : "Motion Decisions"}</h2><p>Qualifier decisions, backup triggers, and detector outcomes</p></div>
            <button onClick={() => loadMotionAudit(auditPage)} disabled={auditLoading}><RefreshCcw className={auditLoading ? "spin" : ""} size={16} /> Refresh</button>
          </div>
          <MotionAuditViewer
            items={auditItems}
            total={auditTotal}
            page={auditPage}
            pageSize={auditPageSize}
            setPage={setAuditPage}
            loading={auditLoading}
            error={auditError}
            timeZone={timeZone}
            onOpen={(item) => setSelectedAuditId(item.id)}
          />
        </section>
        </>
      ) : settingsTab === "calibration" ? (
        <CalibrationLab cameras={cameras} timeZone={timeZone} />
      ) : settingsTab === "telemetry" ? (
        <>
        <section className="bento-card camera-tree config-tree settings-section-tree telemetry-camera-filter">
          <div className="section-head compact"><div><h2>Cameras</h2><p>Select for camera statistics</p></div></div>
          <div className="tree-list">
            {cameras.map((camera) => <button type="button" className={telemetrySection === "cameras" && selectedTelemetryCamera === camera.id ? "active" : ""} key={camera.id} onClick={() => { setTelemetryCamera(camera.id); setTelemetrySection("cameras"); }}><Camera size={16} /><span>{camera.name || camera.id}</span></button>)}
          </div>
        </section>
        <section className="bento-card config-editor settings-panel telemetry-panel">
          <div className="section-head telemetry-panel-head">
            <div><h2>Telemetry</h2><p>System, detection, event, and camera health</p></div>
            {telemetrySection === "overview" ? <TelemetryContinuity data={telemetry} /> : null}
            <button onClick={() => void loadTelemetry()} disabled={telemetryLoading}><RefreshCcw className={telemetryLoading ? "spin" : ""} size={16} /> Refresh</button>
          </div>
          {telemetryError ? <div className="error-banner telemetry-error">{telemetryError}</div> : null}
          <div className="camera-section-tabs telemetry-section-tabs" role="tablist" aria-label="Telemetry sections">
            <button type="button" className={telemetrySection === "overview" ? "active" : ""} onClick={() => setTelemetrySection("overview")} role="tab" aria-selected={telemetrySection === "overview"}><Gauge size={15} />Overview</button>
            <button type="button" className={telemetrySection === "cameras" ? "active" : ""} onClick={() => setTelemetrySection("cameras")} role="tab" aria-selected={telemetrySection === "cameras"}><Camera size={15} />Per-camera</button>
            <button type="button" className={telemetrySection === "diagnostics" ? "active" : ""} onClick={() => setTelemetrySection("diagnostics")} role="tab" aria-selected={telemetrySection === "diagnostics"}><Wrench size={15} />Diagnostics</button>
          </div>
          {telemetrySection === "diagnostics" ? <div className="telemetry-diagnostics">
            <section className="telemetry-section">
              <div className="telemetry-section-head"><div><h3>Temporary diagnostics</h3><p>Capture detailed troubleshooting data for a limited time. Sessions stop automatically and never include images, video, or credentials.</p></div></div>
              <div className="telemetry-diagnostic-controls">
                <label><span>Scope</span><select value={diagnosticScope} onChange={(event) => setDiagnosticScope(event.target.value)}><option value="system">Entire system</option><option value="detector">Object detector</option><option value="storage">Storage</option><option value="camera">One camera</option></select></label>
                {diagnosticScope === "camera" ? <label><span>Camera</span><select value={selectedTelemetryCamera} onChange={(event) => setTelemetryCamera(event.target.value)}>{cameras.map((camera) => <option value={camera.id} key={camera.id}>{camera.name || camera.id}</option>)}</select></label> : null}
                <label><span>Duration</span><select value={diagnosticDuration} onChange={(event) => setDiagnosticDuration(event.target.value)}><option value="900">15 minutes</option><option value="3600">1 hour</option><option value="21600">6 hours</option><option value="86400">24 hours</option></select></label>
                <button type="button" className="primary" onClick={() => void startTelemetryDiagnostics()} disabled={diagnosticScope === "camera" && !selectedTelemetryCamera}>Start diagnostics</button>
              </div>
              {(telemetry?.diagnostics?.active || []).length ? <div className="telemetry-diagnostic-list">{telemetry.diagnostics.active.map((session) => {
                const camera = cameras.find((item) => item.id === session.camera_id);
                return <article className="telemetry-diagnostic-card" key={session.id}><div><strong>{session.scope === "camera" ? camera?.name || session.camera_id : String(session.scope).replaceAll("_", " ")} diagnostics</strong><span>Active until {formatDateTime(session.expires_at, timeZone)}</span></div><div className="button-row"><a className="button" href={`/api/telemetry/diagnostics/${encodeURIComponent(session.id)}`} download={`survng-diagnostics-${session.id}.json`}><Download size={14} />Download</a><button type="button" onClick={() => void stopTelemetryDiagnostics(session.id)}>Stop</button></div></article>;
              })}</div> : <p className="telemetry-diagnostic-empty">No diagnostic capture is active.</p>}
            </section>
            {(telemetry?.diagnostics?.recent || []).some((session) => session.stopped_at || new Date(session.expires_at).getTime() <= Date.now()) ? <section className="telemetry-section"><details className="telemetry-technical"><summary>Recent diagnostic reports</summary><div className="telemetry-diagnostic-list">{telemetry.diagnostics.recent.filter((session) => session.stopped_at || new Date(session.expires_at).getTime() <= Date.now()).map((session) => <article className="telemetry-diagnostic-card" key={session.id}><div><strong>{session.scope === "camera" ? cameras.find((camera) => camera.id === session.camera_id)?.name || session.camera_id : String(session.scope).replaceAll("_", " ")} diagnostics</strong><span>{formatDateTime(session.started_at, timeZone)}</span></div><a className="button" href={`/api/telemetry/diagnostics/${encodeURIComponent(session.id)}`} download={`survng-diagnostics-${session.id}.json`}><Download size={14} />Download</a></article>)}</div></details></section> : null}
            {(telemetry?.operational_events || []).length ? <section className="telemetry-section"><details className="telemetry-technical"><summary>Recent health events</summary><div className="telemetry-health-event-list">{telemetry.operational_events.slice(0, 10).map((event) => <div key={event.id}><span>{event.summary}{Number(event.count || 1) > 1 ? ` · ${event.count} occurrences` : ""}</span><time>{formatDateTime(event.occurred_at, timeZone)}</time></div>)}</div></details></section> : null}
          </div> : <TelemetryViewer data={telemetry} cameraId={telemetrySection === "cameras" ? selectedTelemetryCamera : ""} timeZone={timeZone} />}
        </section>
        </>
      ) : settingsTab === "maintenance" ? (
        <>
        <section className="bento-card camera-tree config-tree settings-section-tree">
          <div className="section-head compact"><div><h2>Maintenance</h2><p>Safe system tools</p></div></div>
          <div className="tree-list">
            <button type="button" className="active"><HardDrive size={16} /><span>Storage Reconciliation</span></button>
          </div>
          <div className="maintenance-help"><strong>What it does</strong><p>Quick Check is bounded to recent media and the newest index rows, so it will not saturate network storage.</p><p>Full Scan checks the entire library, reports progress, and can be cancelled. Repair Database also checks a small bounded batch of older recording metadata. Repairs never delete media or incident history.</p></div>
        </section>
        <section className="bento-card config-editor settings-panel maintenance-panel">
          <div className="section-head">
            <div><h2>Storage Reconciliation</h2><p>Find missing references, stale recording rows, and unlinked media</p></div>
            <div className="camera-command-area maintenance-actions">
              {["running", "cancelling"].includes(maintenance?.status) ? <button onClick={() => void cancelMaintenance()} disabled={maintenance?.status === "cancelling"}><X size={16} /> {maintenance?.status === "cancelling" ? "Cancelling" : "Cancel"}</button> : <><button onClick={() => void startMaintenance(false, false)}><RefreshCcw size={16} /> Quick Check</button><button onClick={() => void startMaintenance(false, true)}><Search size={16} /> Full Scan</button><button className="primary" onClick={() => void startMaintenance(true, maintenance?.result?.full === true)}><Wrench size={16} /> Repair Database</button></>}
            </div>
          </div>
          {maintenanceError ? <div className="error-banner">{maintenanceError}</div> : null}
          <MaintenanceViewer state={maintenance} />
        </section>
        </>
      ) : settingsTab === "logs" ? (
        <>
        <section className="bento-card camera-tree config-tree settings-section-tree">
          <div className="section-head compact"><div><h2>Logs</h2><p>Minimum severity</p></div></div>
          <div className="tree-list">
            {[["DEBUG", "Debug+"], ["INFO", "Info+"], ["WARNING", "Warning+"], ["ERROR", "Error+"], ["CRITICAL", "Critical"]].map(([value, label]) => (
              <button type="button" className={logLevel === value ? "active" : ""} key={value} onClick={() => setLogLevel(value)}><ListTree size={16} /><span>{label}</span></button>
            ))}
          </div>
        </section>
        <section className="bento-card config-editor settings-panel log-panel">
          <div className="section-head">
            <div><h2>Logs</h2><p>Live application log stream</p></div>
            <button onClick={loadLogs}><RefreshCcw size={16} /> Refresh</button>
          </div>
          <LogViewer
            lines={logLines}
            filter={logFilter}
            setFilter={setLogFilter}
            order={logOrder}
            setOrder={setLogOrder}
            timeZone={timeZone}
          />
        </section>
        </>
      ) : (
        <>
      <section className="bento-card camera-tree config-tree">
        <div className="section-head compact">
          <div><h2>Cameras</h2><p>Add, clone, or select</p></div>
          <div className="camera-tree-actions">
            {cameraOrderEditing ? (
              <>
                <button type="button" onClick={cancelCameraOrderEdit} disabled={cameraOrderSaving}>Cancel</button>
                <button type="button" className="primary" onClick={saveCameraOrder} disabled={cameraOrderSaving}><Save size={14} /> {cameraOrderSaving ? "Saving" : "Save"}</button>
              </>
            ) : (
              <>
                <button type="button" onClick={startCameraOrderEdit}><GripVertical size={14} /> Edit Order</button>
                <button type="button" onClick={() => addCamera()}><Plus size={14} /> Add</button>
              </>
            )}
          </div>
        </div>
        <div className="tree-list">
          {cameras.map((camera) => (
            <button
              type="button"
              key={camera.id}
              draggable={cameraOrderEditing}
              className={`${camera.id === selectedCamera?.id ? "active" : ""} ${cameraOrderEditing ? "ordering" : ""} ${dragConfigCameraTarget === camera.id ? (dragConfigCameraAfter ? "drag-after" : "drag-before") : ""}`}
              onClick={() => { if (!cameraOrderEditing) { setSelectedId(camera.id); setProbe(null); } }}
              onDragStart={(event) => {
                if (!cameraOrderEditing) return;
                setDragConfigCameraId(camera.id);
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData("text/plain", camera.id);
              }}
              onDragOver={(event) => {
                if (!dragConfigCameraId || dragConfigCameraId === camera.id) return;
                event.preventDefault();
                const bounds = event.currentTarget.getBoundingClientRect();
                setDragConfigCameraTarget(camera.id);
                setDragConfigCameraAfter(event.clientY > bounds.top + bounds.height / 2);
              }}
              onDrop={(event) => {
                event.preventDefault();
                const sourceId = event.dataTransfer.getData("text/plain") || dragConfigCameraId;
                const bounds = event.currentTarget.getBoundingClientRect();
                moveConfigCamera(sourceId, camera.id, event.clientY > bounds.top + bounds.height / 2);
                setDragConfigCameraId("");
                setDragConfigCameraTarget("");
              }}
              onDragEnd={() => {
                setDragConfigCameraId("");
                setDragConfigCameraTarget("");
              }}
            >
              {cameraOrderEditing ? <GripVertical size={16} /> : <Camera size={16} />}
              <span>{camera.name || camera.id}</span>
            </button>
          ))}
        </div>
      </section>
      <section className="bento-card config-editor">
        <div className="section-head">
          <div><h2>{selectedCamera ? selectedCamera.name : "Camera Config"}</h2><p>Changes save to config.json; only structural motion or camera changes interrupt the affected camera</p></div>
          {selectedCamera ? (
            <div className="camera-command-area">
              <div className="camera-command-bar">
                <button type="button" onClick={() => cloneCamera(selectedCamera)} disabled={cameraSaving}><Copy size={16} /> Clone</button>
                <button type="button" onClick={() => probeCamera(selectedCamera)} disabled={cameraSaving}><Radar size={16} /> Auto-detect</button>
                <button type="button" className="danger" onClick={() => deleteCamera(selectedCamera)} disabled={cameraSaving}><Trash2 size={16} /> Remove</button>
                <button type="button" className="primary camera-save-button" onClick={() => saveCamera(selectedCamera)} disabled={cameraSaving}>
                  {cameraSaving ? <RefreshCcw className="spin" size={16} /> : <Save size={16} />}
                  {cameraSaving ? "Saving..." : "Save Camera"}
                </button>
              </div>
            </div>
          ) : null}
        </div>

        <div className="config-form">
          {selectedCamera ? (
            <>
              <div className="camera-section-tabs" role="tablist" aria-label={`${selectedCamera.name} settings sections`}>
                <button type="button" className={cameraSection === "settings" ? "active" : ""} onClick={() => setCameraSection("settings")} role="tab" aria-selected={cameraSection === "settings"}><Cog size={15} />Settings</button>
                <button type="button" className={cameraSection === "motion" ? "active" : ""} onClick={() => setCameraSection("motion")} role="tab" aria-selected={cameraSection === "motion"}><Activity size={15} />Motion/Object</button>
                <button type="button" className={cameraSection === "zones" ? "active" : ""} onClick={() => setCameraSection("zones")} role="tab" aria-selected={cameraSection === "zones"}><Crop size={15} />Zones</button>
                <button type="button" className={cameraSection === "info" ? "active" : ""} onClick={() => setCameraSection("info")} role="tab" aria-selected={cameraSection === "info"}><Gauge size={15} />Info</button>
              </div>

              {cameraSection === "settings" ? <>
              <div className="field-row camera-identity-fields">
                <label>Name<input value={selectedCamera.name} onChange={(event) => updateCamera(selectedCamera.id, ["name"], event.target.value)} /></label>
              </div>
              </> : null}

              {cameraSection === "motion" ? <div className="field-row camera-object-policy-fields">
                <label>Incident eligibility<select value={selectedCamera.require_incident_zone == null ? "" : String(selectedCamera.require_incident_zone)} onChange={(event) => updateCamera(selectedCamera.id, ["require_incident_zone"], event.target.value === "" ? null : event.target.value === "true")}>
                  <option value="">Use global ({(config.detector?.require_incident_zone ?? true) ? "Zones" : "Zones + Full Frame"})</option>
                  <option value="true">Zones</option>
                  <option value="false">Zones + Full Frame</option>
                </select><small>Ignore zones always suppress their matching object classes.</small></label>
                <label>Repeated scene context<select value={selectedCamera.object_activity_attribution || "inherit"} onChange={(event) => updateCamera(selectedCamera.id, ["object_activity_attribution"], event.target.value)}>
                  <option value="inherit">Use global ({config.detector?.object_activity_attribution === "shadow" ? "Observe" : config.detector?.object_activity_attribution === "off" ? "Off" : "Prevent labels"})</option>
                  <option value="enforce">Prevent false incident labels</option>
                  <option value="shadow">Observe only</option>
                  <option value="off">Off</option>
                </select><small>Controls whether stable objects repeatedly seen in one location can remain evidence without labeling the incident.</small></label>
              </div> : null}

              {cameraSection === "info" ? <div className="field-row camera-info-fields">
                <label>Generated Camera ID<input value={slugify(selectedCamera.name || selectedCamera.id || "camera")} readOnly /></label>
                <label>Detected Backend<input value={inferredBackendLabel(selectedCamera)} readOnly /></label>
              </div> : null}

              {cameraSection === "settings" ? <>
              <div className="field-row stream-field-row">
                <div className="stream-field">
                  <div className="stream-field-head">
                    <label htmlFor={`main-stream-${selectedCamera.id}`}>Main Stream URL</label>
                    <label className="stream-record-toggle"><input type="checkbox" checked={selectedCamera.record} onChange={(event) => updateCamera(selectedCamera.id, ["record"], event.target.checked)} /> Record</label>
                  </div>
                  <input id={`main-stream-${selectedCamera.id}`} value={selectedCamera.stream_url || ""} onChange={(event) => updateCamera(selectedCamera.id, ["stream_url"], event.target.value)} />
                </div>
                <div className="stream-field">
                  <div className="stream-field-head">
                    <label htmlFor={`sub-stream-${selectedCamera.id}`}>Live/Sub Stream URL</label>
                    <label className="stream-record-toggle"><input type="checkbox" checked={selectedCamera.record_sub || false} onChange={(event) => updateCamera(selectedCamera.id, ["record_sub"], event.target.checked)} /> Record</label>
                  </div>
                  <input id={`sub-stream-${selectedCamera.id}`} value={selectedCamera.live_stream_url || ""} onChange={(event) => updateCamera(selectedCamera.id, ["live_stream_url"], event.target.value)} />
                </div>
              </div>
              <details className="camera-retention-details">
                <summary>Camera recording retention</summary>
                <div className="field-row">
                  <label>Main stream history<input type="number" min="1" max="3650" step="1" placeholder={`Global: ${config.retention?.main_days ?? 7} days`} value={selectedCamera.retention?.main_days ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["retention", "main_days"], event.target.value === "" ? null : Number(event.target.value))} /><small>Leave blank to inherit the global policy.</small></label>
                  <label>Substream history<input type="number" min="1" max="3650" step="1" placeholder={`Global: ${config.retention?.live_days ?? 21} days`} value={selectedCamera.retention?.live_days ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["retention", "live_days"], event.target.value === "" ? null : Number(event.target.value))} /><small>Leave blank to inherit the global policy.</small></label>
                </div>
              </details>
              </> : null}

              <div className="config-panels">
                {cameraSection === "settings" ? <div className="sub-panel">
                  <h3>ONVIF</h3>
                  <label className="check-field"><input type="checkbox" checked={selectedCamera.onvif?.enabled || false} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "enabled"], event.target.checked)} /> Enabled</label>
                  <div className="onvif-field-grid">
                    <label>Host<input value={selectedCamera.onvif?.host || ""} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "host"], event.target.value)} /></label>
                    <label>Port<input type="number" value={selectedCamera.onvif?.port || 8000} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "port"], Number(event.target.value))} /></label>
                    <label>Username<input value={selectedCamera.onvif?.username || ""} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "username"], event.target.value)} /></label>
                    <label>Password<input type="password" value={secretInputValue(selectedCamera.onvif?.password)} placeholder={secretInputHint(selectedCamera.onvif?.password)} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "password"], event.target.value)} /></label>
                  </div>
                </div> : null}
                {cameraSection === "motion" ? <div className="sub-panel">
                  <h3>Motion Triggers &amp; Filtering</h3>
                  <MotionDecisionEditor
                    cameraName={selectedCamera.name}
                    fusion={selectedCamera.motion_qualification?.pipeline?.fusion}
                    mode={selectedCamera.motion_qualification?.mode || "inherit"}
                    globalMode={config.motion_qualification?.mode || "camera_rescue"}
                    inherited={selectedCamera.motion_qualification?.pipeline?.fusion == null}
                    inheritedFusion={config.motion_qualification?.pipeline?.fusion}
                    onModeChange={(mode) => updateCamera(selectedCamera.id, ["motion_qualification", "mode"], mode)}
                    onSetInherited={(shouldInherit) => {
                      const pipeline = { ...(selectedCamera.motion_qualification?.pipeline || {}) };
                      pipeline.fusion = shouldInherit
                        ? null
                        : buildMotionDecisionFusion(
                          readMotionDecisionFusion(config.motion_qualification?.pipeline?.fusion).settings,
                        );
                      updateCamera(selectedCamera.id, ["motion_qualification", "pipeline"], pipeline);
                    }}
                    onChange={(fusion) => updateCamera(
                      selectedCamera.id,
                      ["motion_qualification", "pipeline"],
                      { ...(selectedCamera.motion_qualification?.pipeline || {}), fusion },
                    )}
                    onRestoreDefaults={() => updateCamera(
                      selectedCamera.id,
                      ["motion_qualification"],
                      defaultCameraMotionQualification(),
                    )}
                    configurationInherited={cameraMotionQualificationInherited(selectedCamera.motion_qualification)}
                  />
                  <MotionAnalysisPresetEditor
                    qualification={selectedCamera.motion_qualification?.pipeline?.qualification}
                    inherited={selectedCamera.motion_qualification?.pipeline?.qualification == null}
                    catalog={motionCatalog}
                    onSetInherited={() => updateCamera(
                      selectedCamera.id,
                      ["motion_qualification", "pipeline"],
                      { ...(selectedCamera.motion_qualification?.pipeline || {}), qualification: null },
                    )}
                    onChange={(qualification) => updateCamera(
                      selectedCamera.id,
                      ["motion_qualification", "pipeline"],
                      { ...(selectedCamera.motion_qualification?.pipeline || {}), qualification },
                    )}
                  />
                  <details className="motion-tuning-details">
                    <summary>Advanced camera tuning</summary>
                    <div className="motion-camera-tuning">
                  <label>Sensitivity<select value={selectedCamera.motion_qualification?.sensitivity || "inherit"} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "sensitivity"], event.target.value)}>
                    <option value="inherit">Use global setting</option>
                    <option value="high">High</option>
                    <option value="balanced">Balanced</option>
                    <option value="low">Low</option>
                  </select></label>
                  <label>Stationary object policy<select value={selectedCamera.motion_qualification?.stationary_object_tolerance || "inherit"} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "stationary_object_tolerance"], event.target.value)}>
                    <option value="inherit">Use global setting</option>
                    <option value="low">Light</option>
                    <option value="balanced">Standard</option>
                    <option value="high">Strong</option>
                  </select><small>Controls how aggressively EMA rejects confined outline shimmer and reflections before object detection. Strong may ignore unusually slow or distant movement.</small></label>
                  <label>Light and shadow filtering<select value={selectedCamera.motion_qualification?.illumination_filter_enabled == null ? "" : String(selectedCamera.motion_qualification.illumination_filter_enabled)} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "illumination_filter_enabled"], event.target.value === "" ? null : event.target.value === "true")}><option value="">Use global setting</option><option value="true">Enabled</option><option value="false">Disabled</option></select><small>Ignores clear moving illumination while uncertain motion continues to object detection.</small></label>
                  <label>Analysis size<select value={selectedCamera.motion_qualification?.frame_width ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "frame_width"], event.target.value ? Number(event.target.value) : null)}>
                    <option value="">Use global setting</option>
                    <option value="320">320 px</option>
                    <option value="480">480 px</option>
                    <option value="640">640 px</option>
                    <option value="720">720 px</option>
                    <option value="800">800 px</option>
                  </select></label>
                  <label>Visual confidence<input type="number" min="0" max="1" step="0.01" placeholder={`Global: ${config.motion_qualification?.visual_backup_min_score ?? 0.7}`} value={selectedCamera.motion_qualification?.visual_backup_min_score ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "visual_backup_min_score"], event.target.value === "" ? null : Number(event.target.value))} /><small>Leave blank to inherit. Higher values require stronger visual motion before camera-notification rescue runs detection.</small></label>
                  <label>Strong samples<input type="number" min="2" max="10" step="1" placeholder={`Global: ${config.motion_qualification?.visual_backup_min_consecutive ?? 3}`} value={selectedCamera.motion_qualification?.visual_backup_min_consecutive ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "visual_backup_min_consecutive"], event.target.value === "" ? null : Number(event.target.value))} /><small>Consecutive qualifying samples required before rescue.</small></label>
                  <label>Visual grace<input type="number" min="0" max="5" step="0.1" placeholder={`Global: ${config.motion_qualification?.visual_backup_grace_seconds ?? 1.5}s`} value={selectedCamera.motion_qualification?.visual_backup_grace_seconds ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "visual_backup_grace_seconds"], event.target.value === "" ? null : Number(event.target.value))} /><small>How long strong motion must persist. Leave blank to inherit.</small></label>
                  <label>Rescue cooldown<input type="number" min="5" max="300" step="5" placeholder={`Global: ${config.motion_qualification?.visual_backup_cooldown_seconds ?? 20}s`} value={selectedCamera.motion_qualification?.visual_backup_cooldown_seconds ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "visual_backup_cooldown_seconds"], event.target.value === "" ? null : Number(event.target.value))} /><small>Minimum seconds between visual rescue attempts.</small></label>
                  <label>Rescues per 5 minutes<input type="number" min="1" max="30" step="1" placeholder={`Global: ${config.motion_qualification?.visual_backup_max_triggers_5m ?? 3}`} value={selectedCamera.motion_qualification?.visual_backup_max_triggers_5m ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "visual_backup_max_triggers_5m"], event.target.value === "" ? null : Number(event.target.value))} /><small>Per-camera ceiling for visual rescue detection attempts.</small></label>
                  <label>Borderline Rescue<select value={selectedCamera.motion_qualification?.borderline_rescue_enabled == null ? "" : String(selectedCamera.motion_qualification.borderline_rescue_enabled)} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "borderline_rescue_enabled"], event.target.value === "" ? null : event.target.value === "true")}>
                    <option value="">Use global setting</option>
                    <option value="true">Enabled</option>
                    <option value="false">Disabled</option>
                  </select></label>
                  <label>Rescue Margin<input type="number" min="0" max="0.1" step="0.005" placeholder="Global" value={selectedCamera.motion_qualification?.borderline_margin ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "borderline_margin"], event.target.value === "" ? null : Number(event.target.value))} /></label>
                  <label>Double-check filtered motion<select value={selectedCamera.motion_qualification?.suppression_verification_rate == null ? "" : String(selectedCamera.motion_qualification.suppression_verification_rate)} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "suppression_verification_rate"], event.target.value === "" ? null : Number(event.target.value))}><option value="">Use global setting</option><option value="0">Off</option><option value="0.01">About 1 in 100</option><option value="0.05">About 1 in 20</option><option value="0.1">About 1 in 10</option></select><small>Runs object detection on a small sample that visual motion would filter. A configured object safely restores the incident.</small></label>
                    </div>
                  </details>
                </div> : null}
              </div>

              {cameraSection === "zones" ? <ZoneEditor
                camera={selectedCamera}
                classOptions={zoneClassOptions}
                onChange={(zones) => updateCamera(selectedCamera.id, ["zones"], zones)}
                onSave={() => saveZones(selectedCamera)}
                saving={zonesSaving}
              /> : null}

              {cameraSection === "info" ? <>
                <RuntimeStatus status={selectedRuntimeStatus} timeZone={timeZone} motionCatalog={motionCatalog} />
                <MotionDebugViewer cameraId={selectedCamera.id} timeZone={timeZone} />
                {probe ? <ProbeResult probe={probe} /> : null}
              </> : null}
            </>
          ) : (
            <div className="empty-state">Add a camera to begin.</div>
          )}
        </div>
      </section>
        </>
      )}
      {selectedAudit ? (
        <MotionAuditOverlay
          item={selectedAudit}
          items={selectedAuditItems}
          timeZone={timeZone}
          onClose={() => setSelectedAuditId(null)}
          onSelect={(item) => setSelectedAuditId(item.id)}
        />
      ) : null}
    </main>
  );
}

function ZoneEditor({ camera, classOptions = [], onChange, onSave, saving = false }) {
  const zones = camera.zones || [];
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [dragPoint, setDragPoint] = useState(null);
  const [pointAddHistory, setPointAddHistory] = useState({});
  const [snapshotSize, setSnapshotSize] = useState(null);
  const [canvasSize, setCanvasSize] = useState(null);
  const canvasRef = useRef(null);
  const snapshotUrl = useMemo(() => appUrl(`/api/cameras/${camera.id}/zone-snapshot.jpg?source=live&t=${Date.now()}`), [camera.id]);
  const selectedZone = zones[selectedIndex] || null;

  useEffect(() => {
    setSelectedIndex(0);
    setDragPoint(null);
    setPointAddHistory({});
    setSnapshotSize(null);
  }, [camera.id]);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const updateSize = () => {
      setCanvasSize((current) => {
        const next = {
          width: Math.max(0, canvas.clientWidth),
          height: Math.max(0, canvas.clientHeight),
        };
        return current && Math.abs(current.width - next.width) < 0.5 && Math.abs(current.height - next.height) < 0.5
          ? current
          : next;
      });
    };
    updateSize();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", updateSize);
      return () => window.removeEventListener("resize", updateSize);
    }
    const observer = new ResizeObserver(updateSize);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [camera.id]);

  const mediaSize = useMemo(() => {
    if (!snapshotSize?.width || !snapshotSize?.height || !canvasSize?.width || !canvasSize?.height) return null;
    const scale = Math.min(canvasSize.width / snapshotSize.width, canvasSize.height / snapshotSize.height);
    return {
      width: Math.max(1, Math.floor(snapshotSize.width * scale)),
      height: Math.max(1, Math.floor(snapshotSize.height * scale)),
    };
  }, [canvasSize, snapshotSize]);

  useEffect(() => {
    if (!dragPoint) return undefined;
    const preventSelection = (event) => event.preventDefault();
    document.documentElement.classList.add("zone-vertex-dragging");
    document.addEventListener("selectstart", preventSelection);
    return () => {
      document.documentElement.classList.remove("zone-vertex-dragging");
      document.removeEventListener("selectstart", preventSelection);
      window.getSelection()?.removeAllRanges();
    };
  }, [dragPoint]);

  function replaceZone(index, patch) {
    onChange(zones.map((zone, zoneIndex) => zoneIndex === index ? { ...zone, ...patch } : zone));
  }

  function addZone() {
    const next = [...zones, {
      name: `Zone ${zones.length + 1}`,
      color: ["#22c55e", "#38bdf8", "#f59e0b", "#e879f9"][zones.length % 4],
      enabled: true,
      points: [],
      object_classes: [],
      confidence_threshold: null,
      behavior: "incident",
      exclude_from_ema: false,
      trigger: "bottom_center",
    }];
    onChange(next);
    setSelectedIndex(next.length - 1);
  }

  function removeZone(index) {
    onChange(zones.filter((_, zoneIndex) => zoneIndex !== index));
    setPointAddHistory((current) => Object.fromEntries(
      Object.entries(current).flatMap(([zoneIndex, history]) => {
        const numericIndex = Number(zoneIndex);
        if (numericIndex === index) return [];
        return [[numericIndex > index ? numericIndex - 1 : numericIndex, history]];
      }),
    ));
    setSelectedIndex((current) => Math.max(0, Math.min(current, zones.length - 2)));
  }

  function undoPoint() {
    const history = pointAddHistory[selectedIndex] || [];
    const insertionIndex = history.at(-1);
    const points = selectedZone?.points || [];
    if (!selectedZone || insertionIndex == null || insertionIndex >= points.length) return;
    replaceZone(selectedIndex, {
      points: points.filter((_, pointIndex) => pointIndex !== insertionIndex),
    });
    setPointAddHistory((current) => {
      const remaining = (current[selectedIndex] || [])
        .slice(0, -1)
        .map((pointIndex) => pointIndex > insertionIndex ? pointIndex - 1 : pointIndex);
      return { ...current, [selectedIndex]: remaining };
    });
  }

  function pointerPosition(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
    };
  }

  function addPoint(event) {
    if (!selectedZone || event.target !== event.currentTarget) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const point = pointerPosition(event);
    const inserted = insertZonePointWithIndex(
      selectedZone.points,
      point,
      { x: rect.width, y: rect.height },
    );
    setPointAddHistory((current) => {
      const shifted = (current[selectedIndex] || [])
        .map((pointIndex) => pointIndex >= inserted.insertionIndex ? pointIndex + 1 : pointIndex);
      return { ...current, [selectedIndex]: [...shifted, inserted.insertionIndex] };
    });
    replaceZone(selectedIndex, {
      points: inserted.points,
    });
  }

  function movePoint(event) {
    if (!dragPoint || dragPoint.zoneIndex !== selectedIndex || !selectedZone) return;
    const points = [...(selectedZone.points || [])];
    points[dragPoint.pointIndex] = pointerPosition(event);
    replaceZone(selectedIndex, { points });
  }

  return (
    <div className="sub-panel zone-settings">
      <div className="zone-settings-head">
        <div><h3>Detection Zones</h3><p>Objects match using the bottom-center of their detection box.</p></div>
        <div className="zone-settings-actions">
          <button type="button" onClick={undoPoint} disabled={!(pointAddHistory[selectedIndex]?.length)} title="Remove the last point added"><Undo2 size={15} /> Undo Point</button>
          <button type="button" onClick={addZone}><Plus size={15} /> Add Zone</button>
          <button type="button" className="primary" onClick={onSave} disabled={saving}><Save size={15} /> {saving ? "Saving..." : "Save Zones"}</button>
        </div>
      </div>
      <div className="zone-editor-layout">
        <div className="zone-canvas" ref={canvasRef}>
          <div className="zone-canvas-media" style={mediaSize || undefined}>
            <img
              src={snapshotUrl}
              alt={`${camera.name} zone editor`}
              onLoad={(event) => setSnapshotSize({
                width: event.currentTarget.naturalWidth,
                height: event.currentTarget.naturalHeight,
              })}
            />
            <svg
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              onPointerDown={addPoint}
              onPointerMove={movePoint}
              onPointerUp={(event) => {
                movePoint(event);
                setDragPoint(null);
              }}
              onPointerCancel={() => setDragPoint(null)}
              onDragStart={(event) => event.preventDefault()}
              aria-label="Zone polygon editor"
            >
              {zones.map((zone, zoneIndex) => {
                const points = (zone.points || []).map((point) => `${point.x * 100},${point.y * 100}`).join(" ");
                return (
                  <g key={`${zone.name}-${zoneIndex}`} opacity={zone.enabled === false ? 0.35 : 1}>
                    {zone.points?.length >= 3 ? <polygon points={points} fill={`${zone.color || "#22c55e"}33`} stroke={zone.color || "#22c55e"} strokeWidth="0.55" vectorEffect="non-scaling-stroke" pointerEvents="none" /> : null}
                    {zone.points?.length === 2 ? <polyline points={points} fill="none" stroke={zone.color || "#22c55e"} strokeWidth="0.55" vectorEffect="non-scaling-stroke" pointerEvents="none" /> : null}
                    {zoneIndex === selectedIndex ? (zone.points || []).map((point, pointIndex) => (
                      <circle
                        key={pointIndex}
                        cx={point.x * 100}
                        cy={point.y * 100}
                        r="0.85"
                        fill="#fff"
                        stroke={zone.color || "#22c55e"}
                        strokeWidth="0.35"
                        vectorEffect="non-scaling-stroke"
                        onPointerDown={(event) => {
                          event.stopPropagation();
                          event.currentTarget.setPointerCapture(event.pointerId);
                          setDragPoint({ zoneIndex, pointIndex });
                        }}
                      />
                    )) : null}
                  </g>
                );
              })}
            </svg>
          </div>
          {!selectedZone ? <div className="zone-canvas-empty">Add a zone to begin</div> : selectedZone.points?.length < 3 ? <div className="zone-canvas-hint">Click at least three points</div> : null}
        </div>
        <aside className="zone-sidebar">
          <div className="zone-sidebar-panel">
            <h4>Zones</h4>
            <div className="zone-list">
              {zones.map((zone, index) => (
                <button type="button" key={`${zone.name}-${index}`} className={index === selectedIndex ? "active" : ""} onClick={() => setSelectedIndex(index)}>
                  <span className="zone-swatch" style={{ background: zone.color || "#22c55e" }} />
                  <span>{zone.name || `Zone ${index + 1}`}</span>
                  <small>{zone.behavior === "none" ? "no object effect" : zone.behavior}{zone.exclude_from_ema ? " · EMA excluded" : ""}</small>
                </button>
              ))}
              {!zones.length ? <div className="empty-state compact">No zones configured.</div> : null}
            </div>
          </div>
          {selectedZone ? (
            <div className="zone-sidebar-panel zone-config-panel">
              <h4>Zone settings</h4>
              <div className="zone-fields">
                <label className="zone-field-name">Name<input value={selectedZone.name || ""} onChange={(event) => replaceZone(selectedIndex, { name: event.target.value })} /></label>
                <label className="zone-field-color">Color<input className="zone-color-input" type="color" value={selectedZone.color || "#22c55e"} onChange={(event) => replaceZone(selectedIndex, { color: event.target.value })} /></label>
                <label className="zone-field-behavior">Behavior<select value={selectedZone.behavior || "incident"} onChange={(event) => replaceZone(selectedIndex, { behavior: event.target.value })}><option value="incident">Incident</option><option value="ignore">Ignore</option><option value="none">No object effect</option></select></label>
                <div className="zone-class-field">
                  <span>Object Classes</span>
                  <details className={`zone-class-dropdown${selectedZone.behavior === "none" ? " disabled" : ""}`}>
                    <summary>{selectedZone.behavior === "none" ? "Not used" : selectedZone.object_classes?.length ? selectedZone.object_classes.join(", ") : "All classes"}</summary>
                    <div className="zone-class-menu">
                      <label>
                        <input type="checkbox" checked={!selectedZone.object_classes?.length} onChange={() => replaceZone(selectedIndex, { object_classes: [] })} />
                        All classes
                      </label>
                      {classOptions.map((className) => {
                        const selectedClasses = selectedZone.object_classes || [];
                        const checked = selectedClasses.includes(className);
                        return (
                          <label key={className}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => replaceZone(selectedIndex, {
                                object_classes: checked
                                  ? selectedClasses.filter((item) => item !== className)
                                  : [...selectedClasses, className],
                              })}
                            />
                            {className}
                          </label>
                        );
                      })}
                      {!classOptions.length ? <small>No model classes reported</small> : null}
                    </div>
                  </details>
                </div>
                <label className="zone-field-confidence">Confidence<input type="number" min="0.01" max="0.99" step="0.01" placeholder={selectedZone.behavior === "none" ? "N/A" : "Global"} disabled={selectedZone.behavior === "none"} value={selectedZone.confidence_threshold ?? ""} onChange={(event) => replaceZone(selectedIndex, { confidence_threshold: event.target.value === "" ? null : Number(event.target.value) })} /></label>
                <div className="zone-toggle-stack">
                  <label title="Motion inside this zone will not validate or trigger EMA activity. Object incident rules remain unchanged."><input type="checkbox" checked={selectedZone.exclude_from_ema === true} onChange={(event) => replaceZone(selectedIndex, { exclude_from_ema: event.target.checked })} /> Exclude from EMA</label>
                  <label><input type="checkbox" checked={selectedZone.enabled !== false} onChange={(event) => replaceZone(selectedIndex, { enabled: event.target.checked })} /> Enabled</label>
                </div>
                <button type="button" className="danger zone-remove-button" onClick={() => removeZone(selectedIndex)}><Trash2 size={15} /> Remove Zone</button>
              </div>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  );
}

function LogViewer({ lines, filter, setFilter, order, setOrder, timeZone }) {
  const displayedLines = order === "oldest" ? lines : [...lines].reverse();
  return (
    <div className="log-viewer">
      <div className="log-toolbar">
        <div className="log-filter-control">
          <label htmlFor="log-filter-input">Filter</label>
          <div className="log-filter-row">
            <input id="log-filter-input" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="logger, text, error..." />
            <button
              type="button"
              className="log-order-button"
              onClick={() => setOrder(order === "newest" ? "oldest" : "newest")}
              title={order === "newest" ? "Show oldest messages first" : "Show newest messages first"}
              aria-label={order === "newest" ? "Newest messages first; switch to oldest first" : "Oldest messages first; switch to newest first"}
            >
              <ArrowUpDown size={15} /> {order === "newest" ? "Newest first" : "Oldest first"}
            </button>
          </div>
        </div>
      </div>
      <div className="log-lines" role="log" aria-live="polite">
        {displayedLines.length ? displayedLines.map((line, index) => (
          <div className={`log-line ${String(line.level || "").toLowerCase()}`} key={`${line.time}-${index}`}>
            <time>{formatTimeOnly(line.time, timeZone)}</time>
            <span>{line.level}</span>
            <strong>{line.logger}</strong>
            <code>{line.message}</code>
          </div>
        )) : <div className="empty-state">No log lines match the current filters.</div>}
      </div>
    </div>
  );
}

function motionAuditOutcome(item) {
  const visualBackup = item.category === "visual_backup";
  const activeFollowup = item.category === "active_followup";
  if (item.reason === "illumination_change") return { label: "Filtered · light or shadow change", className: "not-run" };
  if (item.features?.illumination_verification_probe) return { label: item.object_detected ? "Light filter check · object rescued" : "Light filter check · no object", className: item.object_detected ? "object" : "clear" };
  if (item.interpretation?.category === "visual_backup_scene_learning") return { label: "Visual backup · scene learning", className: "not-run" };
  if (item.interpretation?.category === "visual_backup_below_threshold") return { label: "Credible EMA motion · below backup threshold", className: "not-run" };
  if (item.interpretation?.category === "object_not_motion_correlated") return { label: "Motion confirmed · detected object outside motion area", className: "clear" };
  if (item.interpretation?.category === "duplicate_active_event") return { label: "Duplicate · event active", className: "not-run" };
  if (item.interpretation?.category === "duplicate_event_cooldown") return { label: "Duplicate · cooldown", className: "not-run" };
  if (item.interpretation?.category === "filtered_before_object_detection") return { label: "Filtered before detection", className: "not-run" };
  if (item.object_detected === true) return { label: visualBackup ? "Visual backup · object found" : activeFollowup ? "Active follow-up · object found" : "Object found", className: "object" };
  if (item.object_detected === false) return { label: visualBackup ? "Visual backup · no object" : activeFollowup ? "Active follow-up · no object" : "No object", className: "clear" };
  return { label: visualBackup ? "Visual backup · incomplete" : activeFollowup ? "Active follow-up · incomplete" : "Not run", className: "not-run" };
}

function MotionAuditAnnotatedImage({ item, alt, loading, onImageSize, interactive = false }) {
  const frameRef = useRef(null);
  const [frameSize, setFrameSize] = useState(null);
  const [imageSize, setImageSize] = useState(null);
  const [zoom, setZoom] = useState({ scale: 1, x: 0, y: 0 });
  const zoomRef = useRef(zoom);
  const pointersRef = useRef(new Map());
  const gestureRef = useRef(null);
  const regions = motionAuditRegions(item.features);
  const renderedImage = useMemo(() => {
    if (!frameSize?.width || !frameSize?.height || !imageSize?.width || !imageSize?.height) return null;
    const scale = Math.min(frameSize.width / imageSize.width, frameSize.height / imageSize.height);
    const width = imageSize.width * scale;
    const height = imageSize.height * scale;
    return { left: (frameSize.width - width) / 2, top: (frameSize.height - height) / 2, width, height };
  }, [frameSize, imageSize]);

  useEffect(() => {
    const reset = { scale: 1, x: 0, y: 0 };
    zoomRef.current = reset;
    setZoom(reset);
    setImageSize(null);
    pointersRef.current.clear();
    gestureRef.current = null;
  }, [item.id]);

  useLayoutEffect(() => {
    const frame = frameRef.current;
    if (!frame) return undefined;
    function updateFrameSize() {
      const rect = frame.getBoundingClientRect();
      if (rect.width && rect.height) setFrameSize({ width: rect.width, height: rect.height });
    }
    updateFrameSize();
    const observer = new ResizeObserver(updateFrameSize);
    observer.observe(frame);
    return () => observer.disconnect();
  }, []);

  function imageLoaded(event) {
    const size = { width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight };
    setImageSize(size);
    onImageSize?.(size);
  }

  function updateZoom(next) {
    const scale = Math.max(1, Math.min(8, Number(next.scale) || 1));
    const xLimit = renderedImage ? renderedImage.width * (scale - 1) / 2 : 0;
    const yLimit = renderedImage ? renderedImage.height * (scale - 1) / 2 : 0;
    const value = {
      scale,
      x: scale === 1 ? 0 : Math.max(-xLimit, Math.min(xLimit, Number(next.x) || 0)),
      y: scale === 1 ? 0 : Math.max(-yLimit, Math.min(yLimit, Number(next.y) || 0)),
    };
    zoomRef.current = value;
    setZoom(value);
  }

  function zoomAt(clientX, clientY, scale) {
    if (!interactive || !renderedImage || !frameRef.current) return;
    const current = zoomRef.current;
    const rect = frameRef.current.getBoundingClientRect();
    const nextScale = Math.max(1, Math.min(8, scale));
    const localX = clientX - rect.left - frameSize.width / 2;
    const localY = clientY - rect.top - frameSize.height / 2;
    const ratio = nextScale / current.scale;
    updateZoom({
      scale: nextScale,
      x: localX - (localX - current.x) * ratio,
      y: localY - (localY - current.y) * ratio,
    });
  }

  function onWheel(event) {
    if (!interactive) return;
    event.preventDefault();
    zoomAt(event.clientX, event.clientY, zoomRef.current.scale * Math.exp(-event.deltaY * 0.0017));
  }

  function onPointerDown(event) {
    if (!interactive) return;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const pointers = [...pointersRef.current.values()];
    if (pointers.length === 2) {
      gestureRef.current = {
        mode: "pinch",
        distance: Math.hypot(pointers[1].x - pointers[0].x, pointers[1].y - pointers[0].y),
        scale: zoomRef.current.scale,
        centerX: (pointers[0].x + pointers[1].x) / 2,
        centerY: (pointers[0].y + pointers[1].y) / 2,
      };
    } else {
      gestureRef.current = { mode: "pan", pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX: zoomRef.current.x, panY: zoomRef.current.y };
    }
  }

  function onPointerMove(event) {
    if (!interactive || !pointersRef.current.has(event.pointerId)) return;
    pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const pointers = [...pointersRef.current.values()];
    const gesture = gestureRef.current;
    if (pointers.length === 2) {
      const distance = Math.hypot(pointers[1].x - pointers[0].x, pointers[1].y - pointers[0].y);
      const centerX = (pointers[0].x + pointers[1].x) / 2;
      const centerY = (pointers[0].y + pointers[1].y) / 2;
      if (!gesture || gesture.mode !== "pinch") {
        gestureRef.current = { mode: "pinch", distance, scale: zoomRef.current.scale, centerX, centerY };
        return;
      }
      zoomAt(centerX, centerY, gesture.scale * distance / Math.max(1, gesture.distance));
      return;
    }
    if (gesture?.mode === "pan" && gesture.pointerId === event.pointerId && zoomRef.current.scale > 1) {
      updateZoom({ scale: zoomRef.current.scale, x: gesture.panX + event.clientX - gesture.x, y: gesture.panY + event.clientY - gesture.y });
    }
  }

  function onPointerEnd(event) {
    pointersRef.current.delete(event.pointerId);
    const remaining = [...pointersRef.current.entries()];
    if (remaining.length === 1) {
      const [pointerId, point] = remaining[0];
      gestureRef.current = { mode: "pan", pointerId, x: point.x, y: point.y, panX: zoomRef.current.x, panY: zoomRef.current.y };
    } else if (!remaining.length) {
      gestureRef.current = null;
    }
  }

  const canvasStyle = renderedImage ? {
    left: `${renderedImage.left}px`,
    top: `${renderedImage.top}px`,
    width: `${renderedImage.width}px`,
    height: `${renderedImage.height}px`,
    transform: `translate3d(${zoom.x}px, ${zoom.y}px, 0) scale(${zoom.scale})`,
  } : undefined;

  return (
    <div
      className={`motion-audit-annotated-image ${interactive ? "interactive" : ""} ${zoom.scale > 1 ? "zoomed" : ""}`}
      ref={frameRef}
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerEnd}
      onPointerCancel={onPointerEnd}
      onDoubleClick={(event) => interactive && (zoom.scale > 1 ? updateZoom({ scale: 1, x: 0, y: 0 }) : zoomAt(event.clientX, event.clientY, 2))}
    >
      <div className="motion-audit-image-canvas" style={canvasStyle}>
        <img src={appUrl(`/api/motion-audit/${item.id}/snapshot.jpg`)} alt={alt} loading={loading} onLoad={imageLoaded} draggable="false" />
        {renderedImage && regions.length ? <div className="motion-audit-region-layer" aria-hidden="true">
          {regions.map(([x1, y1, x2, y2], index) => <span
            className="motion-audit-region"
            key={`${x1}-${y1}-${x2}-${y2}-${index}`}
            style={{ left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` }}
          >{index === regions.length - 1 ? <strong>motion</strong> : null}</span>)}
        </div> : null}
      </div>
      {interactive && zoom.scale > 1 ? <button type="button" className="motion-audit-zoom-reset" onClick={(event) => { event.stopPropagation(); updateZoom({ scale: 1, x: 0, y: 0 }); }}>Reset {zoom.scale.toFixed(1)}×</button> : null}
    </div>
  );
}

function MotionAuditViewer({ items, total, page, pageSize, setPage, loading, error, timeZone, onOpen }) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  return (
    <div className="motion-audit-viewer">
      {error ? <div className="save-status motion-audit-error">{error}</div> : null}
      <div className="motion-audit-grid">
        {items.map((item) => {
          const outcome = motionAuditOutcome(item);
          const features = Object.entries(item.features || {}).filter(([name, value]) => (
            typeof value === "number"
            && Number.isFinite(value)
          ));
          return (
            <article className="motion-audit-card" key={item.id}>
              <button type="button" className="motion-audit-media" onClick={() => onOpen(item)} aria-label={`Open ${item.camera_id} motion audit image`}>
                {item.has_snapshot
                  ? <MotionAuditAnnotatedImage item={item} alt={`${item.camera_id} motion decision`} loading="lazy" />
                  : <div className="empty-thumb"><Camera size={28} /><span>Audit image unavailable</span></div>}
                <span className={`motion-audit-outcome ${outcome.className}`}>{outcome.label}</span>
              </button>
              <div className="motion-audit-body">
                <div className="motion-audit-title"><strong>{item.camera_id}</strong><time>{formatDateTime(item.created_at, timeZone)}</time></div>
                <div className="motion-audit-decision">
                  <span>{String(item.reason || "rejected").replaceAll("_", " ")}</span>
                  <strong>{Number(item.score || 0).toFixed(3)} / {Number(item.threshold || 0).toFixed(3)}</strong>
                </div>
                <div className="motion-audit-meter" aria-label={`Score ${item.score}, threshold ${item.threshold}`}>
                  <i style={{ width: `${Math.max(0, Math.min(100, Number(item.score || 0) * 100))}%` }} />
                  <b style={{ left: `${Math.max(0, Math.min(100, Number(item.threshold || 0) * 100))}%` }} />
                </div>
                <div className="motion-audit-features">
                  {features.map(([name, value]) => <span key={name}>{name.replaceAll("_", " ")} <strong>{Number(value).toFixed(2)}</strong></span>)}
                </div>
                <div className="motion-audit-meta"><span>{item.mode} / {item.sensitivity}</span><span>{item.trigger_count} trigger{item.trigger_count === 1 ? "" : "s"}</span></div>
              </div>
            </article>
          );
        })}
        {!items.length && !loading ? <div className="empty-state">No motion decisions match these filters.</div> : null}
      </div>
      <div className="motion-audit-pagination">
        <button type="button" aria-label="Previous audit page" onClick={() => setPage(Math.max(0, page - 1))} disabled={page <= 0 || loading}><ChevronLeft size={16} /></button>
        <span>{total ? `${page * pageSize + 1}-${Math.min(total, (page + 1) * pageSize)} of ${total}` : "0 entries"}</span>
        <button type="button" aria-label="Next audit page" onClick={() => setPage(Math.min(pageCount - 1, page + 1))} disabled={page >= pageCount - 1 || loading}><ChevronRight size={16} /></button>
      </div>
    </div>
  );
}

const motionAiSettingLabels = {
  analysis_preset: "Motion analysis method",
  stationary_object_tolerance: "Stationary object policy",
};

function formatMotionAiValue(setting, value) {
  if (setting === "analysis_preset") return value === "adaptive" ? "Enhanced Motion Analysis (EMA)" : String(value);
  if (setting === "stationary_object_tolerance") return ({ low: "Light", balanced: "Standard", high: "Strong", inherit: "Use global setting" })[value] || String(value);
  return String(value);
}

function MotionAuditPipeline({ telemetry }) {
  const graphs = telemetry?.graphs && typeof telemetry.graphs === "object" ? telemetry.graphs : null;
  if (!graphs) return null;
  const graphLabels = { qualification: "Frame analysis", observation: "Supporting sources", fusion: "Final decision" };
  return (
    <details className="motion-audit-pipeline">
      <summary>Processing used for this decision</summary>
      <div>
        {Object.entries(graphs).map(([name, graph]) => {
          const configuration = Array.isArray(graph?.configuration) ? graph.configuration : [];
          const timings = graph?.invocation_timings && typeof graph.invocation_timings === "object" ? Object.values(graph.invocation_timings) : [];
          const duration = timings.reduce((total, timing) => total + Number(timing?.duration_ms || 0), 0);
          return (
            <section key={name}>
              <span>{graphLabels[name] || name}</span>
              <strong>{configuration.length} step{configuration.length === 1 ? "" : "s"}{timings.length ? ` · ${duration.toFixed(1)} ms` : " · continuous"}</strong>
              <small>{telemetry.origins?.[name] || "default"} configuration · {configuration.map((stage) => stage.implementation).join(" → ") || "No stage details"}</small>
            </section>
          );
        })}
      </div>
    </details>
  );
}

function MotionAuditOverlay({ item, items, timeZone, onClose, onSelect }) {
  const outcome = motionAuditOutcome(item);
  const currentIndex = items.findIndex((candidate) => candidate.id === item.id);
  const [aiAdvice, setAiAdvice] = useState(null);
  const [aiError, setAiError] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [aiApplying, setAiApplying] = useState(false);
  const pipelineTelemetry = item.features?.pipeline_telemetry;

  useEffect(() => {
    setAiAdvice(null);
    setAiError("");
    setAiLoading(false);
    setAiApplying(false);
  }, [item.id]);

  async function analyzeWithAi() {
    if (aiLoading || !item.has_snapshot) return;
    setAiLoading(true);
    setAiError("");
    try {
      const response = await fetch(`/api/motion-audit/${item.id}/ai-analyze`, { method: "POST" });
      if (!response.ok) throw new Error(await response.text());
      setAiAdvice(await response.json());
    } catch (error) {
      setAiError(error.message || "Unable to analyze this audit image.");
    } finally {
      setAiLoading(false);
    }
  }

  async function applyAiChanges() {
    const changes = aiAdvice?.advice?.changes || [];
    if (!changes.length || aiApplying) return;
    if (!window.confirm(`Apply ${changes.length} AI-recommended motion setting${changes.length === 1 ? "" : "s"}? Camera workers will restart.`)) return;
    setAiApplying(true);
    setAiError("");
    try {
      const response = await fetch(`/api/motion-audit/${item.id}/ai-apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          changes,
          confirmed: true,
          configuration_fingerprint: aiAdvice?.configuration_fingerprint || "",
          recommendation_proof: aiAdvice?.recommendation_proof || "",
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      const result = await response.json();
      setAiAdvice((current) => ({ ...current, applied: result.applied || [] }));
    } catch (error) {
      setAiError(error.message || "Unable to apply AI recommendations.");
    } finally {
      setAiApplying(false);
    }
  }

  function move(direction) {
    if (currentIndex < 0 || items.length < 2) return;
    onSelect(items[(currentIndex + direction + items.length) % items.length]);
  }

  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "Escape") onClose();
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        move(-1);
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        move(1);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [currentIndex, item.id, items, onClose, onSelect]);

  return (
    <div className="motion-audit-overlay" role="dialog" aria-modal="true" aria-label="Motion audit image">
      <button className="live-overlay-backdrop" type="button" onClick={onClose} aria-label="Close motion audit image" />
      <section className="motion-audit-overlay-panel">
        <header className="motion-audit-overlay-head">
          <div><h2>{item.camera_id}</h2><time>{formatDateTime(item.created_at, timeZone)}</time></div>
          <div className="overlay-actions">
            <button type="button" className="icon-only" onClick={() => move(-1)} disabled={items.length < 2} aria-label="Previous audit image"><ChevronLeft size={19} /></button>
            <span>{currentIndex + 1} / {items.length}</span>
            <button type="button" className="icon-only" onClick={() => move(1)} disabled={items.length < 2} aria-label="Next audit image"><ChevronRight size={19} /></button>
            <button type="button" className="icon-only" onClick={onClose} aria-label="Close motion audit image"><X size={19} /></button>
          </div>
        </header>
        <div className="motion-audit-overlay-content">
          <div className="motion-audit-overlay-media">
            {item.has_snapshot
              ? <MotionAuditAnnotatedImage item={item} alt={`${item.camera_id} rejected motion`} interactive />
              : <div className="empty-thumb"><Camera size={42} /><span>Audit image unavailable</span></div>}
          </div>
          <aside className="motion-audit-overlay-details">
            <span className={`motion-audit-outcome ${outcome.className}`}>{outcome.label}</span>
            <div className="motion-audit-overlay-score"><span>{String(item.reason || "rejected").replaceAll("_", " ")}</span><strong>{Number(item.score || 0).toFixed(3)} / {Number(item.threshold || 0).toFixed(3)}</strong></div>
            {item.interpretation?.explanation ? <div className="motion-analysis-warning">{item.interpretation.explanation}</div> : null}
            <div className="motion-audit-meter"><i style={{ width: `${Math.max(0, Math.min(100, Number(item.score || 0) * 100))}%` }} /><b style={{ left: `${Math.max(0, Math.min(100, Number(item.threshold || 0) * 100))}%` }} /></div>
            <dl>
              {Object.entries(item.features || {}).filter(([, value]) => typeof value === "number").map(([name, value]) => <div key={name}><dt>{name.replaceAll("_", " ")}</dt><dd>{Number(value).toFixed(3)}</dd></div>)}
              <div><dt>Mode</dt><dd>{item.mode}</dd></div>
              <div><dt>Sensitivity</dt><dd>{item.sensitivity}</dd></div>
              <div><dt>Triggers</dt><dd>{item.trigger_count}</dd></div>
            </dl>
            <MotionAuditPipeline telemetry={pipelineTelemetry} />
            <div className="motion-audit-ai">
              <div className="motion-audit-ai-head">
                <strong><Sparkles size={15} /> AI Advisor</strong>
                <button type="button" onClick={analyzeWithAi} disabled={aiLoading || aiApplying || !item.has_snapshot} title={item.has_snapshot ? "Analyze this motion decision audit image" : "AI analysis requires a saved audit image"}><Sparkles size={15} /> {aiLoading ? "Analyzing..." : "Analyze"}</button>
              </div>
              {!item.has_snapshot ? <span className="motion-audit-ai-none">AI analysis requires an audit image. This older audit was not sampled or has passed the retention limit.</span> : null}
              {aiError ? <div className="motion-audit-ai-error">{aiError}</div> : null}
              {aiAdvice?.advice ? (
                <div className="motion-audit-ai-result">
                  {aiAdvice.motion_paradigm ? <small>Analyzed as {String(aiAdvice.motion_paradigm.paradigm || "motion decision").replaceAll("_", " ")} · {String(aiAdvice.motion_paradigm.automatic_trigger?.source || "configured trigger").replaceAll("_", " ")}</small> : null}
                  <div className="motion-audit-ai-verdict"><span>{aiAdvice.advice.verdict.replaceAll("_", " ")}</span><strong>{Math.round(Number(aiAdvice.advice.confidence || 0) * 100)}%</strong></div>
                  <p>{aiAdvice.advice.summary}</p>
                  {aiAdvice.advice.visible_subjects?.length ? <div className="motion-audit-ai-subjects">{aiAdvice.advice.visible_subjects.map((subject) => <span key={subject}>{subject}</span>)}</div> : null}
                  {aiAdvice.advice.explanation?.length ? <ul>{aiAdvice.advice.explanation.map((line) => <li key={line}>{line}</li>)}</ul> : null}
                  {aiAdvice.advice.changes?.length ? (
                    <>
                      <div className="motion-audit-ai-changes">
                        {aiAdvice.advice.changes.map((change, index) => <div key={`${change.scope}-${change.setting}-${index}`}><strong>{change.scope} · {motionAiSettingLabels[change.setting] || change.setting.replaceAll("_", " ")}</strong><code>{formatMotionAiValue(change.setting, change.value)}</code><small>{change.reason}</small></div>)}
                      </div>
                      <button type="button" className="primary" onClick={applyAiChanges} disabled={aiApplying || Boolean(aiAdvice.applied)}><Save size={15} /> {aiAdvice.applied ? "Applied" : aiApplying ? "Applying..." : "Apply Recommendations"}</button>
                    </>
                  ) : <span className="motion-audit-ai-none">No setting changes recommended.</span>}
                </div>
              ) : null}
            </div>
          </aside>
        </div>
      </section>
    </div>
  );
}

function ProbeResult({ probe }) {
  if (probe.loading) return <div className="probe-result">Probing camera capabilities...</div>;
  if (probe.error) return <div className="probe-result"><strong>Auto-detection failed</strong><span>{probe.error}</span></div>;
  return (
    <div className="probe-result">
      <strong>Auto-detection</strong>
      <span>Reolink Baichuan: {probe.baichuan?.reachable ? `reachable on ${probe.baichuan.port}` : "not reachable"}</span>
      <span>ONVIF: {probe.onvif?.reachable ? `reachable on ${probe.onvif.port}` : "not reachable"}</span>
      {probe.onvif?.capabilities ? <span>Capabilities: {Object.entries(probe.onvif.capabilities).filter(([, value]) => value).map(([key]) => key).join(", ") || "none reported"}</span> : null}
      {probe.onvif?.error ? <span>{probe.onvif.error}</span> : null}
    </div>
  );
}

function MotionAiReviewPanel({ cameras, advisorEnabled }) {
  const [cameraId, setCameraId] = useState(cameras[0]?.id || "");
  const [hours, setHours] = useState(24);
  const [imageLimit, setImageLimit] = useState(12);
  const [evaluationHours, setEvaluationHours] = useState(24);
  const [review, setReview] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    if (!cameraId && cameras[0]?.id) setCameraId(cameras[0].id);
    if (cameraId && !cameras.some((camera) => camera.id === cameraId)) {
      setCameraId(cameras[0]?.id || "");
    }
  }, [cameraId, cameras]);

  async function loadReview(selectedCameraId, quiet = false) {
    if (!selectedCameraId) return;
    if (!quiet) setLoading(true);
    try {
      const response = await fetch(`/api/motion-ai-reviews/latest?camera_id=${encodeURIComponent(selectedCameraId)}`);
      if (!response.ok) throw new Error(await response.text());
      setReview(await response.json());
      setError("");
    } catch (loadError) {
      setError(loadError.message || "Unable to load the latest camera review.");
    } finally {
      if (!quiet) setLoading(false);
    }
  }

  async function loadEvaluation(selectedCameraId, quiet = false) {
    if (!selectedCameraId) return;
    try {
      const response = await fetch(`/api/camera-intelligence/evaluations/latest?camera_id=${encodeURIComponent(selectedCameraId)}`);
      if (!response.ok) throw new Error(await response.text());
      setEvaluation(await response.json());
    } catch (loadError) {
      if (!quiet) setError(loadError.message || "Unable to load the latest effectiveness check.");
    }
  }

  useEffect(() => {
    setReview(null);
    setEvaluation(null);
    setError("");
    setNotice("");
    void loadReview(cameraId);
    void loadEvaluation(cameraId);
  }, [cameraId]);

  useEffect(() => {
    if (!cameraId || evaluation?.status !== "reviewing") return undefined;
    const timer = window.setInterval(() => void loadEvaluation(cameraId, true), 2000);
    return () => window.clearInterval(timer);
  }, [cameraId, evaluation?.status]);

  useEffect(() => {
    if (!cameraId || !["queued", "running"].includes(review?.status)) return undefined;
    const timer = window.setInterval(() => void loadReview(cameraId, true), 2000);
    return () => window.clearInterval(timer);
  }, [cameraId, review?.status]);

  async function startReview() {
    if (!cameraId || loading) return;
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const response = await fetch("/api/motion-ai-reviews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_id: cameraId, hours, record_limit: 100, image_limit: imageLimit }),
      });
      if (!response.ok) throw new Error(await response.text());
      setReview(await response.json());
    } catch (startError) {
      setError(startError.message || "Unable to start the camera review.");
    } finally {
      setLoading(false);
    }
  }

  async function applyRecommendations() {
    const recommendations = report.recommendations || [];
    if (!review?.id || !recommendations.length || applying) return;
    const cameraName = selectedCamera?.name || cameraId;
    if (!window.confirm(`Apply ${recommendations.length} reviewed setting change${recommendations.length === 1 ? "" : "s"} to ${cameraName}? SurvNG will validate the changes and reload only the affected camera services.`)) return;
    setApplying(true);
    setError("");
    setNotice("");
    try {
      const response = await fetch(`/api/motion-ai-reviews/${review.id}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confirmed: true,
          configuration_fingerprint: report.configuration_fingerprint,
          evaluation_hours: evaluationHours,
          changes: recommendations.map((recommendation) => ({
            scope: "camera",
            setting: recommendation.setting,
            value: recommendation.proposed ?? recommendation.value,
            reason: recommendation.reasons?.[0] || recommendation.reason || "Repeated review evidence supports this change.",
          })),
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      const applied = await response.json();
      setEvaluation(applied.effectiveness_evaluation || null);
      setReview((current) => current ? ({ ...current, result: { ...(current.result || {}), can_apply: false } }) : current);
      setNotice("Recommended camera settings were applied successfully.");
    } catch (applyError) {
      setError(applyError.message || "Unable to apply the reviewed settings.");
    } finally {
      setApplying(false);
    }
  }

  async function runFollowup() {
    if (!evaluation?.id || evaluation.status !== "ready" || loading) return;
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const response = await fetch(`/api/camera-intelligence/evaluations/${evaluation.id}/follow-up`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_limit: imageLimit }),
      });
      if (!response.ok) throw new Error(await response.text());
      setEvaluation(await response.json());
    } catch (followupError) {
      setError(followupError.message || "Unable to start the effectiveness check.");
    } finally {
      setLoading(false);
    }
  }

  const running = ["queued", "running"].includes(review?.status);
  const report = review?.result || {};
  const completedWork = Number(review?.analyzed || 0) + Number(review?.failed || 0);
  const selectedCamera = cameras.find((camera) => camera.id === cameraId);
  const isCameraIntelligence = report.review_type === "camera_intelligence";
  const verdictLabels = {
    consistent: "Looks correct",
    likely_miss: "Likely missed subject",
    likely_false_alarm: "Likely nuisance alert",
    likely_misclassification: "Likely wrong label",
    uncertain: "Uncertain",
  };
  const categoryLabels = {
    possible_miss: "Possible miss",
    visual_backup: "Visual backup",
    motion_filtered: "Filtered motion",
    motion_only_incident: "Motion-only incident",
    recognized_incident: "Recognized incident",
    other: "Other",
  };

  return (
    <div className="sub-panel motion-ai-review-panel">
      <h3>Camera Intelligence</h3>
      <p className="settings-help">Review how one camera has performed across recent incidents and motion decisions. SurvNG deliberately samples successes, possible misses, visual rescues, and filtered motion, then recommends a change only when multiple images support it. Nothing is applied automatically.</p>
      <div className="field-row motion-ai-review-controls">
        <label>Camera<select value={cameraId} onChange={(event) => setCameraId(event.target.value)} disabled={running}>
          {cameras.map((camera) => <option key={camera.id} value={camera.id}>{camera.name || camera.id}</option>)}
        </select></label>
        <label>Review period<select value={hours} onChange={(event) => setHours(Number(event.target.value))} disabled={running}>
          <option value={24}>Last 24 hours</option>
          <option value={72}>Last 3 days</option>
          <option value={168}>Last 7 days</option>
        </select></label>
        <label>Images to inspect<select value={imageLimit} onChange={(event) => setImageLimit(Number(event.target.value))} disabled={running}>
          <option value={8}>8 · Lower cost</option>
          <option value={12}>12 · Balanced</option>
          <option value={16}>16 · More evidence</option>
          <option value={24}>24 · Most thorough</option>
        </select></label>
        <button type="button" className="primary" onClick={startReview} disabled={!cameraId || !advisorEnabled || running || loading}>
          {running ? <RefreshCcw className="spin" size={16} /> : <Sparkles size={16} />}
          {running ? "Reviewing..." : "Review camera"}
        </button>
        <button type="button" onClick={() => { void loadReview(cameraId); void loadEvaluation(cameraId, true); }} disabled={!cameraId || loading}><RefreshCcw className={loading ? "spin" : ""} size={16} /> Refresh</button>
      </div>
      {!advisorEnabled ? <div className="save-status motion-audit-error">Enable and save AI analysis under Object Detection before running a review.</div> : null}
      <div className="probe-result">
        <strong>What this uses</strong>
        <span>Up to 100 recent records for {selectedCamera?.name || cameraId || "the selected camera"}, balanced across different outcomes instead of simply choosing the newest images.</span>
        <span>At most {imageLimit} images are sent for analysis. Each image is one provider request; missing or expired images are skipped.</span>
      </div>
      {error ? <div className="save-status motion-audit-error">{error}</div> : null}
      {notice ? <div className="save-status">{notice}</div> : null}
      {review?.status && review.status !== "never" ? (
        <section className="motion-ai-review-report">
          <header>
            <div><strong>{selectedCamera?.name || review.camera_id}</strong><span>Review #{review.id} · {String(review.status).replaceAll("_", " ")}</span></div>
            <time>{review.updated_at ? formatDateTime(review.updated_at) : ""}</time>
          </header>
          {running ? (
            <div className="motion-ai-review-progress">
              <div><i style={{ width: `${Math.min(100, Number(review.images_available || 0) ? completedWork / Number(review.images_available) * 100 : 0)}%` }} /></div>
              <span>{review.analyzed || 0} reviewed · {review.failed || 0} unavailable · {review.images_available || 0} selected images from {review.audits_considered || 0} recent records</span>
            </div>
          ) : null}
          {review.error ? <div className="motion-runtime-warning">{review.error}</div> : null}
          {review.status === "completed" ? (
            <>
              <p>{report.summary}</p>
              {!isCameraIntelligence && report.review_context?.motion_paradigm ? (
                <div className="probe-result">
                  <strong>Configuration analyzed</strong>
                  <span>{report.review_context.motion_paradigm.paradigm === "camera_triggered" ? "ONVIF-triggered" : report.review_context.motion_paradigm.paradigm === "camera_triggered_with_visual_backup" ? "ONVIF + EMA backup" : report.review_context.motion_paradigm.paradigm === "visual_triggered" ? "EMA-triggered" : "Legacy trigger mode"} · {report.review_context.effective_settings?.incident_eligibility_policy === "zones_only" ? "Zones only" : "Zones + Full Frame"}</span>
                </div>
              ) : null}
              <div className="motion-ai-review-stats">
                <span><strong>{report.verdict_counts?.likely_miss ?? report.verdict_counts?.real_motion ?? 0}</strong>{isCameraIntelligence ? " likely missed" : " likely real motion"}</span>
                <span><strong>{report.verdict_counts?.likely_false_alarm ?? report.verdict_counts?.noise ?? 0}</strong> likely nuisance</span>
                {isCameraIntelligence ? <span><strong>{report.verdict_counts?.consistent || 0}</strong> looks correct</span> : null}
                <span><strong>{report.verdict_counts?.uncertain || 0}</strong> uncertain</span>
              </div>
              {isCameraIntelligence && report.samples?.length ? (
                <>
                  <h4>Images reviewed</h4>
                  <div className="camera-intelligence-samples">
                    {report.samples.map((sample) => (
                      <article key={`${sample.kind}-${sample.record_id}`}>
                        {sample.image_url?.startsWith("/api/") ? <img src={sample.image_url} alt={`${selectedCamera?.name || cameraId} review sample`} loading="lazy" /> : <div className="camera-intelligence-image-missing">Image unavailable</div>}
                        <div><strong>{verdictLabels[sample.verdict] || String(sample.verdict || "Uncertain").replaceAll("_", " ")}</strong><span>{categoryLabels[sample.category] || String(sample.category || "Other").replaceAll("_", " ")}</span></div>
                        <p>{sample.summary}</p>
                      </article>
                    ))}
                  </div>
                </>
              ) : null}
              <h4>Suggested camera changes</h4>
              {report.recommendations?.length ? (
                <div className="motion-ai-review-recommendations">
                  {report.recommendations.map((recommendation) => (
                    <article key={`${recommendation.setting}-${JSON.stringify(recommendation.value)}`}>
                      <div><strong>{motionAiSettingLabels[recommendation.setting] || recommendation.setting.replaceAll("_", " ")}</strong><code>{(recommendation.current ?? recommendation.current_value) == null ? "Current unavailable" : formatMotionAiValue(recommendation.setting, recommendation.current ?? recommendation.current_value)} → {formatMotionAiValue(recommendation.setting, recommendation.proposed ?? recommendation.value)}</code></div>
                      <span>Supported by {recommendation.support_count} analyzed image{recommendation.support_count === 1 ? "" : "s"} · {Math.round(Number(recommendation.average_confidence || 0) * 100)}% average confidence</span>
                      <p>{recommendation.reasons?.[0]}</p>
                      {recommendation.evidence_audit_ids?.length ? <small>Evidence: audit {recommendation.evidence_audit_ids.join(", ")}</small> : null}
                    </article>
                  ))}
                </div>
              ) : <span>No setting change was recommended consistently enough across the analyzed images.</span>}
              {isCameraIntelligence && report.recommendations?.length && !notice ? <div className="camera-intelligence-apply-row"><label>Check results after<select value={evaluationHours} onChange={(event) => setEvaluationHours(Number(event.target.value))}><option value={24}>24 hours</option><option value={168}>7 days</option></select></label><button type="button" className="primary camera-intelligence-apply" onClick={applyRecommendations} disabled={!report.can_apply || applying}>{applying ? <RefreshCcw className="spin" size={16} /> : <Check size={16} />}{report.can_apply ? (applying ? "Applying..." : "Review and apply suggestions") : "Applying AI suggestions is disabled"}</button></div> : null}
            </>
          ) : null}
        </section>
      ) : <div className="probe-result"><strong>No review yet</strong><span>Choose a camera and run its first manual review.</span></div>}
      {evaluation?.status && evaluation.status !== "never" ? (
        <section className={`camera-intelligence-effectiveness ${evaluation.comparison?.outcome || evaluation.status}`}>
          <header><div><strong>Did the change help?</strong><span>Applied {formatDateTime(evaluation.applied_at)}</span></div><span>{String(evaluation.status).replaceAll("_", " ")}</span></header>
          {evaluation.applied_changes?.length ? <div className="camera-intelligence-applied">{evaluation.applied_changes.map((change) => <span key={change.setting}>{motionAiSettingLabels[change.setting] || change.setting.replaceAll("_", " ")}: {formatMotionAiValue(change.setting, change.current)} → {formatMotionAiValue(change.setting, change.proposed)}</span>)}</div> : null}
          {evaluation.status === "collecting" ? <p>SurvNG is gathering normal camera activity. The follow-up becomes available {evaluation.ready_at ? formatDateTime(evaluation.ready_at) : "after the selected period"}.</p> : null}
          {evaluation.status === "ready" ? <><p>Enough time has passed to compare a new balanced image sample against the review made before the change.</p><button type="button" className="primary" onClick={runFollowup} disabled={loading}><Sparkles size={16} />Run follow-up review</button></> : null}
          {evaluation.status === "reviewing" ? <p><RefreshCcw className="spin" size={16} /> Reviewing post-change camera activity…</p> : null}
          {evaluation.status === "completed" && evaluation.comparison ? <><p className="camera-intelligence-outcome">{evaluation.comparison.summary}</p><div className="camera-intelligence-comparison">{evaluation.comparison.metrics?.map((metric) => <article key={metric.key}><span>{metric.label}</span><strong>{Math.round(Number(metric.before_rate || 0) * 100)}% → {Math.round(Number(metric.after_rate || 0) * 100)}%</strong><small>{Number(metric.change_points || 0) > 0 ? "+" : ""}{metric.change_points} percentage points</small></article>)}</div><small>{evaluation.comparison.caution}</small></> : null}
          {evaluation.error ? <div className="motion-runtime-warning">{evaluation.error}</div> : null}
        </section>
      ) : null}
    </div>
  );
}

function RetentionSummary({ status }) {
  const plan = status.plan || {};
  const storage = plan.storage || {};
  const indexed = plan.indexed || {};
  const reclaim = plan.reclaim || {};
  const lastRun = status.last_run || null;
  const cameraStorageRows = plan.per_camera_storage || [];
  const snapshots = plan.snapshots || {};
  const headroom = indexed.days_to_minimum_free;
  return (
    <div className="retention-summary">
      <div className={`retention-alert ${storage.emergency ? "critical" : Number(reclaim.planned_bytes || 0) > 0 ? "warning" : "healthy"}`}>
        {storage.emergency || Number(reclaim.planned_bytes || 0) > 0 ? <CircleAlert size={20} /> : <CircleDot size={20} />}
        <div>
          <strong>{storage.emergency ? "Storage is critically low" : Number(reclaim.planned_bytes || 0) > 0 ? `${formatBytes(reclaim.planned_bytes)} eligible for cleanup` : "Storage is within the configured policy"}</strong>
          <span>{Number(storage.free_percent || 0).toFixed(1)}% free · {formatBytes(storage.free_bytes)} available{headroom == null ? "" : ` · approximately ${headroom} days to the cleanup threshold`}</span>
        </div>
      </div>
      <div className="retention-metrics">
        <article><span>Continuous recordings</span><strong>{formatBytes(indexed.bytes)}</strong><small>{Number(indexed.file_count || 0).toLocaleString()} indexed segments</small></article>
        <article><span>Incident snapshots</span><strong>{formatBytes(snapshots.bytes)}</strong><small>{Number(snapshots.file_count || 0).toLocaleString()} indexed images</small></article>
        <article><span>Current growth</span><strong>{formatBytes(indexed.bytes_per_day)}/day</strong><small>Estimated from indexed history</small></article>
        <article><span>Age-expired</span><strong>{formatBytes(reclaim.expired_bytes)}</strong><small>{Number(reclaim.expired_files || 0).toLocaleString()} segments</small></article>
        <article><span>Capacity pressure</span><strong>{formatBytes(Math.max(Number(reclaim.quota_bytes || 0), Number(reclaim.free_space_bytes || 0)))}</strong><small>{(reclaim.reasons || []).length ? reclaim.reasons.join(" + ").replaceAll("_", " ") : "None"}</small></article>
      </div>
      {lastRun ? <div className="retention-last-run"><strong>Last cleanup</strong><span>{Number(lastRun.deleted_files || 0).toLocaleString()} files · {formatBytes(lastRun.deleted_bytes)} reclaimed{lastRun.failed_files ? ` · ${lastRun.failed_files} failed` : ""}</span></div> : null}
      {cameraStorageRows.length ? <details className="retention-camera-details"><summary>Per-camera storage</summary><div className="retention-camera-table retention-camera-storage-table">
        <div className="heading"><span>Camera</span><span>Used-Recordings</span><span>Used-Snapshots</span><span>Recording files</span><span>Snapshot files</span></div>
        {cameraStorageRows.map((row) => <div key={row.camera_id}><strong>{row.camera_id}</strong><span>{formatBytes(row.recording_bytes)}</span><span>{formatBytes(row.snapshot_bytes)}</span><span>{Number(row.recording_files || 0).toLocaleString()}</span><span>{Number(row.snapshot_files || 0).toLocaleString()}</span></div>)}
      </div></details> : null}
    </div>
  );
}

function GeneralSettings({ config, updateConfig, timeZone, setTimeZone, theme, setTheme, accelerator, detectorModels, recordingCache, retentionStatus, retentionError, runRetention, mqttStatus, detectorStatus, motionCatalog, section }) {
  const [liveOrderReset, setLiveOrderReset] = useState(false);
  const [serverRestart, setServerRestart] = useState({ state: "idle", text: "" });
  const [apiTokenDraft, setApiTokenDraft] = useState({ id: "", name: "", scopes: ["read"] });
  const [apiTokenSecret, setApiTokenSecret] = useState("");
  const [apiTokenBusy, setApiTokenBusy] = useState(false);
  const [apiTokenError, setApiTokenError] = useState("");
  const activeModelPath = config.detector?.model_path || config.detector?.model_xml || "";
  const validEvaluationModels = detectorModels.filter((model) => model.valid).sort((left, right) => String(left.path).localeCompare(String(right.path)));
  const defaultBaselinePath = validEvaluationModels.filter((model) => String(model.path) < activeModelPath).at(-1)?.path
    || validEvaluationModels.filter((model) => model.path !== activeModelPath).at(-1)?.path
    || "";
  const [modelEvaluationDraft, setModelEvaluationDraft] = useState({ baseline_path: "", candidate_path: "", sample_count: 200, confidence: 0.25 });
  const [modelEvaluation, setModelEvaluation] = useState({ status: "idle" });
  const [modelEvaluationError, setModelEvaluationError] = useState("");
  const [modelEvaluationPreview, setModelEvaluationPreview] = useState(null);
  const mediaLocations = config.media_storage?.locations || [];
  const reidStatus = detectorStatus?.reid || null;
  const cameraTransitionRoutes = config.detector?.tracking?.camera_transition_routes || [];
  const routeCameras = config.cameras || [];
  const updateCameraRoute = (index, field, value) => updateConfig(
    ["detector", "tracking", "camera_transition_routes"],
    cameraTransitionRoutes.map((route, routeIndex) => routeIndex === index ? { ...route, [field]: value } : route),
  );
  const addCameraRoute = () => {
    if (routeCameras.length < 2) return;
    const existing = new Set(cameraTransitionRoutes.map((route) => `${route.from_camera}->${route.to_camera}`));
    let pair = null;
    for (const from of routeCameras) {
      for (const to of routeCameras) {
        if (from.id !== to.id && !existing.has(`${from.id}->${to.id}`)) {
          pair = [from.id, to.id];
          break;
        }
      }
      if (pair) break;
    }
    if (!pair) return;
    updateConfig(["detector", "tracking", "camera_transition_routes"], [
      ...cameraTransitionRoutes,
      { from_camera: pair[0], to_camera: pair[1], min_seconds: 0, max_seconds: 30, bidirectional: false, enabled: true, name: "" },
    ]);
  };
  const openvinoDevices = accelerator?.openvino_devices || [];
  const hasOpenvinoGpu = openvinoDevices.includes("GPU");
  const detectorBackend = config.detector?.backend || "openvino";
  const coremlLabel = accelerator?.is_macos
    ? accelerator?.coreml_available
      ? "Core ML available"
      : "Core ML not installed"
    : "Core ML is macOS only";
  const gpuLabel = accelerator?.is_apple_silicon
    ? "Mac GPU detected, OpenVINO GPU not available on Apple GPU"
    : accelerator?.has_nvidia
      ? "NVIDIA GPU detected"
      : hasOpenvinoGpu
        ? "OpenVINO GPU device available"
        : "No OpenVINO GPU device reported";
  const deviceOptions = [
    ["CPU", "CPU"],
    ["GPU", hasOpenvinoGpu ? "GPU" : "GPU (if OpenVINO plugin is available)"],
    ["AUTO", "AUTO"],
  ];
  const ffmpegAcceleration = accelerator?.ffmpeg_hardware_acceleration || {};
  const vaapi = ffmpegAcceleration.vaapi || {};
  const qsv = ffmpegAcceleration.qsv || {};
  const vaapiLabel = vaapi.available
    ? `VAAPI available (${(vaapi.encoders || []).join(", ") || "encoders detected"})`
    : vaapi.listed
      ? "VAAPI listed by FFmpeg but runtime init failed"
      : "VAAPI not available to FFmpeg";
  const qsvLabel = qsv.available
    ? `Intel QSV available (${(qsv.encoders || []).join(", ") || "encoders detected"})`
    : qsv.listed
      ? "Intel QSV listed by FFmpeg but runtime init failed"
      : "Intel QSV not available to FFmpeg";
  const activeModel = detectorModels.find((model) => model.path === activeModelPath);
  const eventClassConfirmations = config.detector?.event_class_confirmation_frames || {};
  const eventClassConfidences = config.detector?.event_class_confidence_thresholds || {};
  const eventConfirmationClasses = [...new Set([
    ...(activeModel?.classes || []),
    ...Object.keys(eventClassConfirmations),
    ...Object.keys(eventClassConfidences),
  ].map((label) => String(label).trim().toLowerCase()).filter(Boolean))]
    .sort((left, right) => left.localeCompare(right));
  const trackingExcludedLabels = config.detector?.tracking?.excluded_labels || ["face"];
  const trackingClassOptions = [...new Set([
    ...(activeModel?.classes || []),
    ...trackingExcludedLabels,
  ].map((label) => String(label).trim().toLowerCase()).filter(Boolean))]
    .sort((left, right) => left.localeCompare(right));

  function selectOpenvinoModel(path) {
    updateConfig(["detector", "model_path"], path);
    updateConfig(["detector", "model_xml"], "");
    if (path.endsWith(".xml")) updateConfig(["detector", "labels_path"], "");
  }

  function setEventClassConfirmation(label, value) {
    const next = { ...eventClassConfirmations };
    if (value === "") delete next[label];
    else next[label] = Number(value);
    updateConfig(["detector", "event_class_confirmation_frames"], next);
  }

  function setEventClassConfidence(label, value) {
    const next = { ...eventClassConfidences };
    if (value === "") delete next[label];
    else next[label] = Number(value);
    updateConfig(["detector", "event_class_confidence_thresholds"], next);
  }

  function resetLiveCameraOrder() {
    localStorage.removeItem("survng.liveCameraOrder.v1");
    setLiveOrderReset(true);
  }

  function updateMediaLocation(index, field, value) {
    updateConfig(["media_storage", "locations"], mediaLocations.map((item, itemIndex) => (
      itemIndex === index ? { ...item, [field]: value } : item
    )));
  }

  function toggleMediaRole(index, role, enabled) {
    const current = mediaLocations[index]?.roles || [];
    const roles = enabled ? [...new Set([...current, role])] : current.filter((item) => item !== role);
    if (!roles.length) return;
    updateMediaLocation(index, "roles", roles);
  }

  function addMediaLocation() {
    const index = mediaLocations.length + 1;
    updateConfig(["media_storage", "locations"], [...mediaLocations, {
      id: `media-${index}`,
      name: `Media ${index}`,
      path: "",
      enabled: true,
      roles: ["recordings", "snapshots", "motion_audits", "clips", "exports"],
      reserve_percent: 15,
      priority: 100,
      require_mount: true,
    }]);
  }

  useEffect(() => {
    setModelEvaluationDraft((current) => ({
      ...current,
      candidate_path: current.candidate_path || activeModelPath,
      baseline_path: current.baseline_path || defaultBaselinePath,
    }));
  }, [activeModelPath, defaultBaselinePath]);

  useEffect(() => {
    if (section !== "detection") return undefined;
    let cancelled = false;
    let timer;
    const load = async () => {
      try {
        const response = await fetch("/api/detector/model-evaluation", { cache: "no-store" });
        if (!response.ok) return;
        const payload = await response.json();
        if (cancelled) return;
        setModelEvaluation(payload);
        if (["queued", "running", "cancelling"].includes(payload.status)) timer = window.setTimeout(load, 1_000);
      } catch {
        // An evaluation result is optional configuration telemetry.
      }
    };
    void load();
    return () => { cancelled = true; if (timer) window.clearTimeout(timer); };
  }, [section, modelEvaluation.status]);

  async function startModelEvaluation() {
    setModelEvaluationError("");
    try {
      const response = await fetch("/api/detector/model-evaluation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(modelEvaluationDraft),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Unable to start model evaluation.");
      setModelEvaluation(payload);
    } catch (error) {
      setModelEvaluationError(error.message || "Unable to start model evaluation.");
    }
  }

  async function cancelModelEvaluation() {
    const response = await fetch("/api/detector/model-evaluation", { method: "DELETE" });
    const payload = await response.json().catch(() => ({}));
    if (response.ok) setModelEvaluation(payload);
    else setModelEvaluationError(payload.detail || "Unable to cancel model evaluation.");
  }

  async function restartServer() {
    if (serverRestart.state === "requesting" || serverRestart.state === "waiting") return;
    if (!window.confirm("Restart SurvNG now? Live view, recording playback, and detection will be briefly unavailable.")) return;
    setServerRestart({ state: "requesting", text: "Requesting restart..." });
    try {
      const response = await fetch("/api/system/restart", { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Unable to restart SurvNG.");
      const previousInstance = String(payload.instance_id || "");
      setServerRestart({ state: "waiting", text: "Restarting SurvNG..." });
      const deadline = Date.now() + 90_000;
      while (Date.now() < deadline) {
        await new Promise((resolve) => window.setTimeout(resolve, 1_500));
        try {
          const statusResponse = await fetch(`/api/system/status?restart_check=${Date.now()}`, { cache: "no-store" });
          if (!statusResponse.ok) continue;
          const status = await statusResponse.json();
          if (previousInstance && String(status.instance_id || "") === previousInstance) continue;
          window.location.reload();
          return;
        } catch {
          // The expected unavailable window while the service restarts.
        }
      }
      setServerRestart({ state: "error", text: "Restart is taking longer than expected. Refresh this page shortly." });
    } catch (error) {
      setServerRestart({ state: "error", text: error.message || "Unable to restart SurvNG." });
    }
  }

  function toggleApiTokenScope(scope) {
    setApiTokenDraft((current) => ({
      ...current,
      scopes: current.scopes.includes(scope)
        ? current.scopes.filter((item) => item !== scope)
        : [...current.scopes, scope],
    }));
  }

  async function createApiToken() {
    if (apiTokenBusy || !apiTokenDraft.id.trim() || !apiTokenDraft.name.trim() || !apiTokenDraft.scopes.length) return;
    setApiTokenBusy(true);
    setApiTokenError("");
    setApiTokenSecret("");
    try {
      const response = await fetch("/api/config/api-tokens", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: apiTokenDraft.id.trim(),
          name: apiTokenDraft.name.trim(),
          scopes: apiTokenDraft.scopes,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not create API token");
      updateConfig(["api_auth", "tokens"], [
        ...(config.api_auth?.tokens || []),
        { ...payload.credential, token_hash: "__SURVNG_SECRET_SET__" },
      ]);
      setApiTokenSecret(payload.token || "");
      setApiTokenDraft({ id: "", name: "", scopes: ["read"] });
    } catch (error) {
      setApiTokenError(error.message || "Could not create API token");
    } finally {
      setApiTokenBusy(false);
    }
  }

  async function deleteApiToken(tokenId) {
    if (apiTokenBusy || !window.confirm(`Delete API token “${tokenId}”? Clients using it will stop working immediately.`)) return;
    setApiTokenBusy(true);
    setApiTokenError("");
    try {
      const response = await fetch(`/api/config/api-tokens/${encodeURIComponent(tokenId)}`, { method: "DELETE" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not delete API token");
      updateConfig(["api_auth", "tokens"], (config.api_auth?.tokens || []).filter((token) => token.id !== tokenId));
      if (!payload.enabled) updateConfig(["api_auth", "enabled"], false);
      setApiTokenSecret("");
    } catch (error) {
      setApiTokenError(error.message || "Could not delete API token");
    } finally {
      setApiTokenBusy(false);
    }
  }

  return (
    <div className="general-settings-content config-form">
        {section === "general" ? (
        <div className="sub-panel">
          <h3>General</h3>
          <label>Timezone<select value={timeZone} onChange={(event) => setTimeZone(event.target.value)}>
            {US_TIME_ZONES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          <label>Theme<select value={theme} onChange={(event) => setTheme(event.target.value)}>
            {THEMES.map((value) => <option key={value} value={value}>{THEME_META[value].label}</option>)}
          </select></label>
          <label>Web Base Path<input value={config.base_path ?? "/survng"} onChange={(event) => updateConfig(["base_path"], event.target.value)} placeholder="/survng" /></label>
          <div className="preference-action general-server-actions">
            <strong>Live Camera Order</strong>
            <div className="preference-action-buttons">
              <button type="button" onClick={resetLiveCameraOrder}><RotateCcw size={15} /> Reset Order</button>
              <button type="button" className="danger" onClick={restartServer} disabled={["requesting", "waiting"].includes(serverRestart.state)}>
                {serverRestart.state === "requesting" || serverRestart.state === "waiting" ? <RefreshCcw className="spin" size={15} /> : <Power size={15} />}
                Restart Server
              </button>
            </div>
          </div>
          {liveOrderReset ? <span className="preference-status"><CircleDot size={13} /> Reset for this browser</span> : null}
          {serverRestart.text ? <span className={`preference-status ${serverRestart.state === "error" ? "error" : ""}`} role="status">{serverRestart.text}</span> : null}
        </div>
        ) : null}

        {section === "storage" ? (
        <div className="sub-panel">
          <h3>Storage</h3>
          <div className="admin-field-grid">
            <label>Storage Directory<input value={config.storage_dir || ""} onChange={(event) => updateConfig(["storage_dir"], event.target.value)} /></label>
            <label>Metadata Database Directory<input value={config.database_dir || ""} onChange={(event) => updateConfig(["database_dir"], event.target.value)} placeholder="Defaults to storage directory" /></label>
            <label>Recording Index Directory<input value={config.recording_index_dir || ""} onChange={(event) => updateConfig(["recording_index_dir"], event.target.value)} placeholder="Defaults to storage directory" /></label>
          </div>
          <section className="media-storage-settings">
            <div className="retention-heading">
              <div><h4>Media locations</h4><p>Spread recordings and related media across independently managed filesystems.</p></div>
              <button type="button" onClick={addMediaLocation}><Plus size={15} /> Add location</button>
            </div>
            <div className="admin-field-grid">
              <label>Placement<select value={config.media_storage?.placement || "balanced"} onChange={(event) => updateConfig(["media_storage", "placement"], event.target.value)}><option value="balanced">Balanced free space</option><option value="priority">Location priority</option></select></label>
            </div>
            {!mediaLocations.length ? <div className="probe-result ok"><strong>Single media location</strong><span>All media continues to use Storage Directory. Add locations only when the additional filesystems are mounted and writable.</span></div> : null}
            <div className="media-location-list">
              {mediaLocations.map((location, index) => {
                const candidateStatus = retentionStatus?.plan?.storage?.locations?.find((item) => item.id === location.id);
                const normalizePath = (value) => String(value || "").replace(/\/+$/, "");
                const status = candidateStatus && normalizePath(candidateStatus.path) === normalizePath(location.path)
                  ? candidateStatus
                  : null;
                return <article className="media-location-card" key={index}>
                  <header><strong>{location.name || location.id || `Location ${index + 1}`}</strong><span className={`retention-state ${status?.state === "online" ? "running" : status?.state || "idle"}`}>{status?.state || "save to inspect"}</span><button type="button" className="danger compact" aria-label={`Remove ${location.name || location.id}`} onClick={() => updateConfig(["media_storage", "locations"], mediaLocations.filter((_item, itemIndex) => itemIndex !== index))}><Trash2 size={14} /></button></header>
                  <div className="admin-field-grid">
                    <label>ID<input value={location.id || ""} onChange={(event) => updateMediaLocation(index, "id", event.target.value)} /></label>
                    <label>Name<input value={location.name || ""} onChange={(event) => updateMediaLocation(index, "name", event.target.value)} /></label>
                    <label className="wide-field">Filesystem path<input value={location.path || ""} onChange={(event) => updateMediaLocation(index, "path", event.target.value)} placeholder="/mnt/survng-media-2" /></label>
                    <label>Reserve free space<input type="number" min="0" max="95" step="1" value={location.reserve_percent ?? 15} onChange={(event) => updateMediaLocation(index, "reserve_percent", Number(event.target.value))} /></label>
                    <label>Priority<input type="number" min="1" max="1000" step="1" value={location.priority ?? 100} onChange={(event) => updateMediaLocation(index, "priority", Number(event.target.value))} /></label>
                  </div>
                  <div className="media-location-roles">{MEDIA_STORAGE_ROLES.map(([role, label]) => <label key={role}><input type="checkbox" checked={(location.roles || []).includes(role)} onChange={(event) => toggleMediaRole(index, role, event.target.checked)} />{label}</label>)}</div>
                  <div className="media-location-flags"><label><input type="checkbox" checked={location.enabled ?? true} onChange={(event) => updateMediaLocation(index, "enabled", event.target.checked)} />Accept new media</label><label><input type="checkbox" checked={location.require_mount ?? false} onChange={(event) => updateMediaLocation(index, "require_mount", event.target.checked)} />Require a real mount</label></div>
                  {status ? <small>{formatBytes(status.free_bytes)} free of {formatBytes(status.total_bytes)} · {Number(status.free_percent || 0).toFixed(1)}% free</small> : null}
                </article>;
              })}
            </div>
            <p className="retention-protection"><ShieldCheck size={15} /> Databases, indexes, models, and playback cache remain on local SurvNG storage. A required mount is never replaced by its empty mountpoint.</p>
          </section>
          <div className="prewarm-setting">
            <h4>Evidence image storage</h4>
            <div className="field-row">
              <label>Format<select value={config.image_storage?.format || "webp"} onChange={(event) => updateConfig(["image_storage", "format"], event.target.value)}><option value="webp">WebP (recommended)</option><option value="jpeg">JPEG</option></select></label>
              <label>Quality<input type="number" min="1" max="100" step="1" value={config.image_storage?.quality ?? 95} onChange={(event) => updateConfig(["image_storage", "quality"], Number(event.target.value))} /></label>
            </div>
            <p>Controls newly saved incident and motion-audit images. Higher quality preserves more forensic detail but uses more space. Existing images are left unchanged, and live snapshots remain JPEG for compatibility.</p>
          </div>
          <div className="admin-field-grid">
            <label>FFmpeg Path<input value={config.ffmpeg_path || ""} onChange={(event) => updateConfig(["ffmpeg_path"], event.target.value)} /></label>
            <label>Hardware Acceleration<select value={config.hardware_acceleration || "auto"} onChange={(event) => updateConfig(["hardware_acceleration"], event.target.value)}>
              <option value="auto">Auto (VAAPI preferred)</option>
              <option value="vaapi">VAAPI</option>
              <option value="qsv">Intel QSV</option>
              <option value="off">Off</option>
            </select></label>
            <label>Recording Segment Seconds<input type="number" min="2" max="300" step="1" value={config.recording_segment_seconds ?? 10} onChange={(event) => updateConfig(["recording_segment_seconds"], Number(event.target.value))} /></label>
            <label>Event Clip Before<input type="number" min="0" max="30" step="1" value={config.event_clip_before_seconds ?? 5} onChange={(event) => updateConfig(["event_clip_before_seconds"], Number(event.target.value))} /></label>
            <label>Event Clip After<input type="number" min="0" max="30" step="1" value={config.event_clip_after_seconds ?? 5} onChange={(event) => updateConfig(["event_clip_after_seconds"], Number(event.target.value))} /></label>
            <label>Playback Cache GB<input type="number" min="0.5" max="100" step="0.5" value={config.recording_cache_max_gb ?? 5} onChange={(event) => updateConfig(["recording_cache_max_gb"], Number(event.target.value))} /></label>
            <label>Playback Cache Days<input type="number" min="1" max="90" step="1" value={config.recording_cache_max_days ?? 7} onChange={(event) => updateConfig(["recording_cache_max_days"], Number(event.target.value))} /></label>
          </div>
          <div className="prewarm-setting">
            <label className="check-field"><input type="checkbox" checked={config.recording_cache_prewarm ?? true} onChange={(event) => updateConfig(["recording_cache_prewarm"], event.target.checked)} /> Prewarm finalized recordings</label>
            <p>Prepares each completed recording in the background so it opens faster on iPhone and in browsers. It trades additional remux work and playback-cache space for a shorter initial loading delay.</p>
          </div>
          {recordingCache ? <div className="probe-result"><strong>Playback Cache</strong><span>{formatBytes(recordingCache.bytes)} used across {recordingCache.entries} fragments</span><span>{formatBytes(recordingCache.max_bytes)} limit, {recordingCache.max_days} day maximum age</span><span>{recordingCache.metrics?.playback_hits || 0} hits / {recordingCache.metrics?.playback_misses || 0} misses, {recordingCache.metrics?.playback_avg_remux_ms || 0} ms average remux</span></div> : null}
          <div className="retention-settings">
            <div className="retention-heading">
              <div><h4>Media retention</h4><p>Daily recording and incident-snapshot planning with lightweight cleanup checks every 15 minutes.</p></div>
              <span className={`retention-state ${retentionStatus?.state || "starting"}`}>
                {String(retentionStatus?.state || "calculating").replaceAll("_", " ")}
                {retentionStatus?.progress && Number(retentionStatus.progress.initial_bytes || 0) > 0
                  ? ` · ${Number(retentionStatus.progress.percent || 0).toFixed(1)}% · ${retentionStatus.progress.eta_seconds == null ? "calculating" : `~${formatCompactDuration(retentionStatus.progress.eta_seconds)} left`}`
                  : ""}
              </span>
            </div>
            <div className="retention-fields">
              <label className="check-field"><input type="checkbox" checked={config.retention?.enabled ?? true} onChange={(event) => updateConfig(["retention", "enabled"], event.target.checked)} /> Monitor storage retention</label>
              <label className="check-field"><input type="checkbox" checked={config.retention?.automatic_cleanup ?? false} onChange={(event) => updateConfig(["retention", "automatic_cleanup"], event.target.checked)} /> Automatically remove expired recordings and snapshots</label>
              <label>SurvNG storage limit<input type="number" min="0.1" max="1000" step="0.5" value={config.retention?.storage_limit_tb ?? 13} onChange={(event) => updateConfig(["retention", "storage_limit_tb"], Number(event.target.value))} /><small>TiB allocated to indexed continuous recordings.</small></label>
              <label>Main stream history<input type="number" min="1" max="3650" step="1" value={config.retention?.main_days ?? 7} onChange={(event) => updateConfig(["retention", "main_days"], Number(event.target.value))} /><small>Days of high-resolution continuous video.</small></label>
              <label>Substream history<input type="number" min="1" max="3650" step="1" value={config.retention?.live_days ?? 21} onChange={(event) => updateConfig(["retention", "live_days"], Number(event.target.value))} /><small>Days of lower-bandwidth continuous video.</small></label>
              <label>Incident snapshot history<input type="number" min="1" max="3650" step="1" value={config.retention?.snapshot_days ?? 1095} onChange={(event) => updateConfig(["retention", "snapshot_days"], Number(event.target.value))} /><small>Days to keep clean incident evidence images. Default: 1,095 days.</small></label>
              <label>Start cleanup below<input type="number" min="1" max="95" step="1" value={config.retention?.minimum_free_percent ?? 15} onChange={(event) => updateConfig(["retention", "minimum_free_percent"], Number(event.target.value))} /><small>Percent free on the entire storage filesystem.</small></label>
              <label>Clean back to<input type="number" min="2" max="99" step="1" value={config.retention?.target_free_percent ?? 20} onChange={(event) => updateConfig(["retention", "target_free_percent"], Number(event.target.value))} /><small>Higher than the start threshold to prevent repeated cycling.</small></label>
              <label>Emergency threshold<input type="number" min="0.5" max="50" step="0.5" value={config.retention?.emergency_free_percent ?? 5} onChange={(event) => updateConfig(["retention", "emergency_free_percent"], Number(event.target.value))} /><small>Raises a critical storage state.</small></label>
            </div>
            {retentionStatus?.plan ? <RetentionSummary status={retentionStatus} /> : <div className="probe-result"><strong>Calculating retention projection</strong><span>The first index-only plan normally appears within a few seconds.</span></div>}
            {retentionError ? <div className="error-banner">{retentionError}</div> : null}
            <div className="retention-actions">
              <button type="button" onClick={() => runRetention(false)}><RefreshCcw size={15} /> Recalculate</button>
              <button type="button" className="danger" onClick={() => runRetention(true)} disabled={["queued", "planning", "cleaning", "waiting"].includes(retentionStatus?.state)}><Trash2 size={15} /> Clean Up Now</button>
            </div>
            <p className="retention-protection"><ShieldCheck size={15} /> Incident snapshots expire only after their configured history; face-reference images, incident clips, metadata databases, and the newest five minutes of recording remain protected.</p>
          </div>
        </div>
        ) : null}

        {section === "mqtt" ? (
        <div className="sub-panel">
          <h3>API</h3>
          <section className="api-access-settings">
            <div className="detection-settings-subhead">
              <div><strong>API access tokens</strong><small>Long-lived credentials for Home Assistant and other integrations. Secrets are displayed only once and are never stored in readable form.</small></div>
              <span className={`retention-state ${config.api_auth?.enabled ? "running" : "idle"}`}>{config.api_auth?.enabled ? "Enforced" : "Not enforced"}</span>
            </div>
            <div className="api-token-list">
              {(config.api_auth?.tokens || []).map((token) => (
                <article key={token.id}>
                  <div><strong>{token.name}</strong><code>{token.id}</code><small>{(token.scopes || []).join(" · ")}</small></div>
                  <button type="button" className="danger" onClick={() => deleteApiToken(token.id)} disabled={apiTokenBusy}><Trash2 size={14} /> Delete</button>
                </article>
              ))}
              {!(config.api_auth?.tokens || []).length ? <p className="settings-help">No integration tokens configured.</p> : null}
            </div>
            <div className="api-token-create">
              <label>Token ID<input value={apiTokenDraft.id} onChange={(event) => setApiTokenDraft((current) => ({ ...current, id: event.target.value }))} placeholder="home-assistant" /></label>
              <label>Name<input value={apiTokenDraft.name} onChange={(event) => setApiTokenDraft((current) => ({ ...current, name: event.target.value }))} placeholder="Home Assistant" /></label>
              <div className="api-token-scopes" role="group" aria-label="API token scopes">
                {[["read", "Read"], ["camera:control", "Camera control"], ["admin", "Admin"]].map(([value, label]) => <label className="check-field" key={value}><input type="checkbox" checked={apiTokenDraft.scopes.includes(value)} onChange={() => toggleApiTokenScope(value)} /> {label}</label>)}
              </div>
              <button type="button" className="primary" onClick={createApiToken} disabled={apiTokenBusy || !apiTokenDraft.id.trim() || !apiTokenDraft.name.trim() || !apiTokenDraft.scopes.length}>{apiTokenBusy ? <RefreshCcw className="spin" size={15} /> : <Plus size={15} />} Create token</button>
            </div>
            {apiTokenSecret ? <div className="api-token-secret" role="status"><strong>Copy this token now</strong><code>{apiTokenSecret}</code><button type="button" onClick={() => navigator.clipboard?.writeText(apiTokenSecret)}><Copy size={14} /> Copy</button><small>It cannot be displayed again after you leave this page.</small></div> : null}
            {apiTokenError ? <div className="error-banner">{apiTokenError}</div> : null}
          </section>
          <section className="api-access-settings mqtt-access-settings">
            <div className="detection-settings-subhead">
              <div><strong>MQTT</strong><small>Broker connection, Home Assistant discovery, incident publishing, and server telemetry.</small></div>
              <span className={`retention-state ${mqttStatus?.connected ? "running" : "idle"}`}>{mqttStatus?.connected ? "Connected" : config.mqtt?.enabled ? "Disconnected" : "Disabled"}</span>
            </div>
            <div className="admin-field-grid">
              <label className="check-field"><input type="checkbox" checked={config.mqtt?.enabled || false} onChange={(event) => updateConfig(["mqtt", "enabled"], event.target.checked)} /> Enabled</label>
              <label>Broker Host<input value={config.mqtt?.host || ""} onChange={(event) => updateConfig(["mqtt", "host"], event.target.value)} placeholder="mqtt.local" /></label>
              <label>Port<input type="number" min="1" max="65535" value={config.mqtt?.port || 1883} onChange={(event) => updateConfig(["mqtt", "port"], Number(event.target.value))} /></label>
              <label>Username<input value={config.mqtt?.username || ""} onChange={(event) => updateConfig(["mqtt", "username"], event.target.value)} /></label>
              <label>Password<input type="password" value={secretInputValue(config.mqtt?.password)} placeholder={secretInputHint(config.mqtt?.password)} onChange={(event) => updateConfig(["mqtt", "password"], event.target.value)} /></label>
              <label>Client ID<input value={config.mqtt?.client_id || "survng"} onChange={(event) => updateConfig(["mqtt", "client_id"], event.target.value)} /></label>
              <label>Topic Prefix<input value={config.mqtt?.topic_prefix || "survng"} onChange={(event) => updateConfig(["mqtt", "topic_prefix"], event.target.value)} /></label>
              <label className="check-field"><input type="checkbox" checked={config.mqtt?.incident_events_enabled ?? true} onChange={(event) => updateConfig(["mqtt", "incident_events_enabled"], event.target.checked)} /> Publish incident events</label>
              <label className="check-field"><input type="checkbox" checked={config.mqtt?.discovery_enabled ?? true} onChange={(event) => updateConfig(["mqtt", "discovery_enabled"], event.target.checked)} /> Home Assistant Discovery</label>
              <label>Discovery Prefix<input value={config.mqtt?.discovery_prefix || "homeassistant"} onChange={(event) => updateConfig(["mqtt", "discovery_prefix"], event.target.value)} disabled={config.mqtt?.discovery_enabled === false} /></label>
              <label className="check-field"><input type="checkbox" checked={config.mqtt?.server_status_enabled ?? true} onChange={(event) => updateConfig(["mqtt", "server_status_enabled"], event.target.checked)} /> Publish SurvNG server status</label>
              <label>Server Device Name<input value={config.mqtt?.server_name || "SurvNG Server"} onChange={(event) => updateConfig(["mqtt", "server_name"], event.target.value)} disabled={config.mqtt?.server_status_enabled === false} /></label>
              <label>Server Metrics Interval<input type="number" min="10" max="3600" step="5" value={config.mqtt?.server_metrics_interval_seconds ?? 30} onChange={(event) => updateConfig(["mqtt", "server_metrics_interval_seconds"], Number(event.target.value))} disabled={config.mqtt?.server_status_enabled === false} /><small>Seconds between retained system, camera, detector, and storage updates.</small></label>
              <label>QoS<select value={config.mqtt?.qos ?? 0} onChange={(event) => updateConfig(["mqtt", "qos"], Number(event.target.value))}><option value={0}>0</option><option value={1}>1</option><option value={2}>2</option></select></label>
              <label className="check-field"><input type="checkbox" checked={config.mqtt?.tls || false} onChange={(event) => updateConfig(["mqtt", "tls"], event.target.checked)} /> TLS</label>
            </div>
            {mqttStatus ? <div className={`probe-result ${mqttStatus.connected ? "ok" : ""}`}><strong>Connection details</strong><span>{mqttStatus.host || "No broker"}:{mqttStatus.port || 1883}</span><span>{mqttStatus.messages_published || 0} published · {mqttStatus.publish_failures || 0} publish failures</span><span>Commands: {mqttStatus.command_subscriptions_active ? "ready" : mqttStatus.connected ? "not subscribed" : "offline"} · {mqttStatus.commands_received || 0} accepted · {mqttStatus.commands_rejected || 0} rejected · {mqttStatus.command_errors || 0} failed · {mqttStatus.command_queue_depth || 0} queued</span>{mqttStatus.server_status_enabled ? <span>Server: {mqttStatus.server_lifecycle || "starting"} · {mqttStatus.server_state?.health || "pending"} · {mqttStatus.server_state?.activity || "idle"} · every {mqttStatus.server_metrics_interval_seconds || 30}s · {mqttStatus.server_state_topic}</span> : null}{mqttStatus.incident_events_enabled ? <span>Incidents: {mqttStatus.incident_topic} ({mqttStatus.pending_incidents || 0} pending)</span> : null}{mqttStatus.server_status_error ? <span>Server metrics: {mqttStatus.server_status_error}</span> : null}{mqttStatus.last_error ? <span>{mqttStatus.last_error}</span> : null}</div> : null}
          </section>
        </div>
        ) : null}

      {section === "detection" ? (
      <div className="detection-settings">
        <section className="detection-settings-card primary">
          <header className="detection-settings-card-head">
            <div className="detection-settings-card-icon"><ScanFace size={18} /></div>
            <div><h3>Detection</h3><p>Choose the model, accelerator, and rules that turn motion into object incidents.</p></div>
            <label className="compact-toggle"><input type="checkbox" checked={config.detector?.enabled || false} onChange={(event) => updateConfig(["detector", "enabled"], event.target.checked)} /><span>Detector enabled</span></label>
          </header>
          <div className="detection-field-grid">
          <label>Backend<select value={detectorBackend} onChange={(event) => updateConfig(["detector", "backend"], event.target.value)}>
            <option value="openvino">OpenVINO / ONNX</option>
            <option value="coreml">Core ML (Mac)</option>
          </select></label>
          <label>OpenVINO Device<select value={config.detector?.device || "CPU"} onChange={(event) => updateConfig(["detector", "device"], event.target.value)}>
            {deviceOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          <label>Parallel detectors<select value={String(config.detector?.object_worker_count ?? 1)} onChange={(event) => updateConfig(["detector", "object_worker_count"], Number(event.target.value))} disabled={detectorBackend !== "openvino"}>
            <option value="1">1 detector</option>
            <option value="2">2 detectors</option>
            <option value="3">3 detectors</option>
            <option value="4">4 detectors</option>
          </select><small>Independent OpenVINO workers can process simultaneous camera events. More workers use more accelerator and memory capacity.</small></label>
          <label>Incident confidence<input type="number" min="0.01" max="0.99" step="0.01" value={config.detector?.confidence_threshold ?? 0.45} onChange={(event) => updateConfig(["detector", "confidence_threshold"], Number(event.target.value))} /><small>A single detection must meet this confidence. Repeated candidates can still qualify through confirmation.</small></label>
          <label>Candidate confidence<input type="number" min="0.01" max="0.95" step="0.01" value={config.detector?.event_candidate_confidence_threshold ?? 0.25} onChange={(event) => updateConfig(["detector", "event_candidate_confidence_threshold"], Number(event.target.value))} /><small>Retains weaker detections only as temporal evidence; they require at least three consistent frames.</small></label>
          <label>Object confirmation<select value={String(config.detector?.event_confirmation_frames ?? 2)} onChange={(event) => updateConfig(["detector", "event_confirmation_frames"], Number(event.target.value))}><option value="1">Immediate (1 frame)</option><option value="2">Confirmed (2 frames)</option><option value="3">Strong (3 frames)</option><option value="4">Very strict (4 frames)</option><option value="5">Maximum (5 frames)</option></select><small>Requires the same label in this many of five event-time frames. Confirmed is recommended and suppresses one-frame false identifications.</small></label>
          <label>Incident eligibility<select value={String(config.detector?.require_incident_zone ?? true)} onChange={(event) => updateConfig(["detector", "require_incident_zone"], event.target.value === "true")}>
            <option value="true">Zones</option>
            <option value="false">Zones + Full Frame</option>
          </select><small>Default for cameras using the global rule.</small></label>
          <label className="wide-field">Model<select value={detectorModels.some((model) => model.path === activeModelPath) ? activeModelPath : ""} onChange={(event) => selectOpenvinoModel(event.target.value)}>
            <option value="">Custom path</option>
            {detectorModels.map((model) => {
              const directory = String(model.path || "").split("/").slice(0, -1).pop();
              return <option key={model.path} value={model.path} disabled={!model.valid}>{directory ? `${directory} / ` : ""}{model.name} ({model.task || "detect"}, {model.valid ? "ready" : "incomplete"})</option>;
            })}
          </select></label>
          </div>
          <details className="detection-compact-details">
            <summary>Model paths and startup options</summary>
            <div className="detection-field-grid">
              <label className="wide-field">OpenVINO / ONNX path<input value={activeModelPath} onChange={(event) => selectOpenvinoModel(event.target.value)} placeholder="openvino_model/best.xml or best.onnx" /></label>
              <label>Labels path<input value={config.detector?.labels_path || ""} onChange={(event) => updateConfig(["detector", "labels_path"], event.target.value)} placeholder="Automatic from metadata" /></label>
              <label>Compiled model cache<input value={config.detector?.cache_dir || ".cache/openvino"} onChange={(event) => updateConfig(["detector", "cache_dir"], event.target.value)} disabled={config.detector?.cache_enabled === false} /></label>
              <label className="compact-toggle"><input type="checkbox" checked={config.detector?.cache_enabled ?? true} onChange={(event) => updateConfig(["detector", "cache_enabled"], event.target.checked)} /><span>Cache compiled model</span></label>
              <label className="compact-toggle"><input type="checkbox" checked={config.detector?.warmup_enabled ?? true} onChange={(event) => updateConfig(["detector", "warmup_enabled"], event.target.checked)} /><span>Warm up at startup</span></label>
            </div>
          </details>
          <details className="detection-compact-details">
            <summary>Per-object confirmation and confidence</summary>
            <p className="settings-help">Tune how often and how confidently each object must be recognized. Higher confirmation reduces one-frame mistakes; higher confidence rejects weaker matches. Leaving either setting on global uses the values above.</p>
            {eventConfirmationClasses.length ? <div className="per-object-detection-grid">
              {eventConfirmationClasses.map((label) => <div className="per-object-detection-row" key={label}>
                <strong>{label.replaceAll("_", " ")}</strong>
                <label>Confirmation<select value={eventClassConfirmations[label] == null ? "" : String(eventClassConfirmations[label])} onChange={(event) => setEventClassConfirmation(label, event.target.value)}><option value="">Global ({config.detector?.event_confirmation_frames ?? 2} frames)</option><option value="1">1 frame</option><option value="2">2 frames</option><option value="3">3 frames</option><option value="4">4 frames</option><option value="5">5 frames</option></select></label>
                <label>Confidence<input type="number" min="0.01" max="0.99" step="0.01" placeholder={`Global (${config.detector?.confidence_threshold ?? 0.45})`} value={eventClassConfidences[label] == null ? "" : String(eventClassConfidences[label])} onChange={(event) => setEventClassConfidence(label, event.target.value)} /></label>
              </div>)}
            </div> : <span className="settings-help">Select a model with class metadata to configure per-object overrides.</span>}
          </details>
        </section>

        <section className="detection-settings-card wide-card model-evaluation-card">
          <header className="detection-settings-card-head">
            <div className="detection-settings-card-icon"><Gauge size={18} /></div>
            <div><h3>Model Evaluation</h3><p>Compare two models on the same recent clean incident images without changing production detection.</p></div>
          </header>
          <div className="detection-field-grid">
            <label>Baseline model<select value={modelEvaluationDraft.baseline_path} onChange={(event) => setModelEvaluationDraft((current) => ({ ...current, baseline_path: event.target.value }))}>
              <option value="">Select model</option>
              {detectorModels.filter((model) => model.valid).map((model) => <option key={model.path} value={model.path}>{String(model.path).split("/").slice(-2, -1)[0] || model.name}</option>)}
            </select></label>
            <label>Candidate model<select value={modelEvaluationDraft.candidate_path} onChange={(event) => setModelEvaluationDraft((current) => ({ ...current, candidate_path: event.target.value }))}>
              <option value="">Select model</option>
              {detectorModels.filter((model) => model.valid).map((model) => <option key={model.path} value={model.path}>{String(model.path).split("/").slice(-2, -1)[0] || model.name}{model.path === activeModelPath ? " (active)" : ""}</option>)}
            </select></label>
            <label>Recent images<input type="number" min="10" max="500" step="10" value={modelEvaluationDraft.sample_count} onChange={(event) => setModelEvaluationDraft((current) => ({ ...current, sample_count: Number(event.target.value) }))} /><small>Round-robin sampled across cameras; 200 is a useful first pass.</small></label>
            <label>Candidate threshold<input type="number" min="0.01" max="0.99" step="0.01" value={modelEvaluationDraft.confidence} onChange={(event) => setModelEvaluationDraft((current) => ({ ...current, confidence: Number(event.target.value) }))} /><small>Use the same low evidence threshold for both models.</small></label>
          </div>
          <div className="model-evaluation-actions">
            <button type="button" className="primary" onClick={startModelEvaluation} disabled={!modelEvaluationDraft.baseline_path || !modelEvaluationDraft.candidate_path || ["queued", "running", "cancelling"].includes(modelEvaluation.status)}><Activity size={15} />Run comparison</button>
            {["queued", "running", "cancelling"].includes(modelEvaluation.status) ? <button type="button" onClick={cancelModelEvaluation} disabled={modelEvaluation.status === "cancelling"}><X size={15} />Cancel</button> : null}
            <span className={`model-evaluation-state ${modelEvaluation.status}`}>{String(modelEvaluation.status || "idle").replaceAll("_", " ")}{modelEvaluation.progress?.total ? ` · ${modelEvaluation.progress.completed}/${modelEvaluation.progress.total}` : ""}</span>
          </div>
          <p className="settings-help">Runs sequentially at low priority from the user’s perspective, but shares the configured accelerator with production detection. Start with 200 images and run during a quiet period.</p>
          {modelEvaluationError || modelEvaluation.error ? <div className="error-banner">{modelEvaluationError || modelEvaluation.error}</div> : null}
          {modelEvaluation.result ? <div className="model-evaluation-results">
            <div className="model-evaluation-summary">
              <span><strong>{modelEvaluation.result.sample_count}</strong> images</span>
              <span><strong>{modelEvaluation.result.camera_count}</strong> cameras</span>
              <span><strong>{modelEvaluation.result.source_counts?.incident || 0}</strong> incidents</span>
              <span><strong>{modelEvaluation.result.source_counts?.motion_audit || 0}</strong> negatives</span>
              <span><strong>{modelEvaluation.result.disagreement_frames}</strong> disagreements</span>
              <span><strong>{modelEvaluation.result.candidate.average_ms} ms</strong> candidate average</span>
              <span><strong>{modelEvaluation.result.candidate.p95_ms} ms</strong> candidate p95</span>
            </div>
            <div className="model-evaluation-models">
              {[['Baseline', modelEvaluation.result.baseline], ['Candidate', modelEvaluation.result.candidate]].map(([label, result]) => <div key={label}><strong>{label}</strong><span>{String(result.path).split("/").slice(-2, -1)[0]}</span><span>{result.frames_with_objects} frames with objects</span><span>{result.average_ms} ms average · {result.p95_ms} ms p95</span><span>{Object.entries(result.label_counts || {}).map(([name, count]) => `${name} ${count}`).join(" · ") || "No detections"}</span></div>)}
            </div>
            <p className="settings-help">Stored-evidence recall is diagnostic, not verified accuracy: baseline {modelEvaluation.result.stored_evidence_recall?.baseline ?? "—"}, candidate {modelEvaluation.result.stored_evidence_recall?.candidate ?? "—"}. Review disagreements before promoting a model.</p>
            {modelEvaluation.result.disagreements?.length ? <div className="model-evaluation-disagreements">
              {modelEvaluation.result.disagreements.map((item) => <article key={`${item.source_kind}-${item.source_id}`}>
                <button type="button" className="model-evaluation-image-button" onClick={() => setModelEvaluationPreview(item)} aria-label={`Enlarge ${item.camera_id} comparison image`}><img src={appUrl(item.image_url)} alt="" loading="lazy" /></button>
                <span><strong>{item.camera_id}</strong><small>{item.source_kind === "motion_audit" ? "Motion audit negative" : "Incident"} · {item.created_at}</small><small>Old: {item.baseline_labels.join(", ") || "none"}</small><small>New: {item.candidate_labels.join(", ") || "none"}</small><a href={appUrl(item.source_kind === "motion_audit" ? `/config?section=audit&audit_id=${item.source_id}` : `/incidents?event_ids=${item.event_id}`)}>Open {item.source_kind === "motion_audit" ? "audit" : "incident"}</a></span>
              </article>)}
            </div> : <div className="probe-result ok"><strong>No label disagreements</strong><span>Both models returned the same label sets on this corpus.</span></div>}
          </div> : null}
          {modelEvaluationPreview ? <div className="model-evaluation-preview" role="dialog" aria-modal="true" aria-label="Model comparison image">
            <button type="button" className="live-overlay-backdrop" onClick={() => setModelEvaluationPreview(null)} aria-label="Close comparison image" />
            <section>
              <header><div><strong>{modelEvaluationPreview.camera_id}</strong><small>{modelEvaluationPreview.created_at}</small></div><button type="button" className="icon-only" onClick={() => setModelEvaluationPreview(null)} aria-label="Close comparison image"><X size={19} /></button></header>
              <img src={appUrl(modelEvaluationPreview.image_url)} alt={`${modelEvaluationPreview.camera_id} model comparison source`} />
              <footer><span><strong>Old</strong> {modelEvaluationPreview.baseline_labels.join(", ") || "none"}</span><span><strong>New</strong> {modelEvaluationPreview.candidate_labels.join(", ") || "none"}</span><a className="primary" href={appUrl(modelEvaluationPreview.source_kind === "motion_audit" ? `/config?section=audit&audit_id=${modelEvaluationPreview.source_id}` : `/incidents?event_ids=${modelEvaluationPreview.event_id}`)}>Open {modelEvaluationPreview.source_kind === "motion_audit" ? "audit" : "incident"}</a></footer>
            </section>
          </div> : null}
        </section>

        <section className="detection-settings-card detection-feature-card wide-card">
          <header className="detection-settings-card-head">
            <div className="detection-settings-card-icon"><Activity size={18} /></div>
            <div><h3>Stationary objects &amp; scene context</h3><p>Separate visual-motion filtering from object-level incident attribution.</p></div>
          </header>
          <div className="detection-field-grid">
            <label>Stationary object policy<select value={config.motion_qualification?.stationary_object_tolerance || "balanced"} onChange={(event) => updateConfig(["motion_qualification", "stationary_object_tolerance"], event.target.value)}><option value="low">Light</option><option value="balanced">Standard</option><option value="high">Strong</option></select><small>Coordinates EMA background learning, stationary-motion scoring, and parked-object scene memory. Strong may ignore unusually slow or distant travel.</small></label>
            <label>Repeated scene context<select value={config.detector?.object_activity_attribution || "enforce"} onChange={(event) => updateConfig(["detector", "object_activity_attribution"], event.target.value)}>
              <option value="enforce">Prevent false incident labels</option>
              <option value="shadow">Observe without changing incidents</option>
              <option value="off">Off</option>
            </select><small>Runs after object detection. Repeated stable objects remain stored as evidence without being treated as the cause; moving or uncertain objects remain eligible.</small></label>
            <div className="detection-settings-subhead"><strong>Fixed areas remain explicit</strong><small>Object Ignore zones suppress only their matching classes. “Exclude from EMA” independently removes all visual motion in that polygon.</small></div>
          </div>
        </section>

        <section className="detection-settings-card">
          <header className="detection-settings-card-head">
            <div className="detection-settings-card-icon"><Activity size={18} /></div>
            <div><h3>Continuous tracking</h3><p>Keep one numbered identity while an object moves through an active incident.</p></div>
          </header>
          <div className="detection-field-grid">
          <label>Tracking detail<select value={String(config.detector?.tracking?.sample_fps ?? 2)} onChange={(event) => updateConfig(["detector", "tracking", "sample_fps"], Number(event.target.value))}><option value="1">Lower CPU (1 frame/sec)</option><option value="2">Balanced (2 frames/sec)</option><option value="3">Smoother (3 frames/sec)</option><option value="5">Maximum detail (5 frames/sec)</option></select><small>OpenVINO runs once for every analyzed tracking frame.</small></label>
          <div className="zone-class-field tracking-class-field">
            <span>Do not track</span>
            <details className="zone-class-dropdown">
              <summary>{trackingExcludedLabels.length ? trackingExcludedLabels.join(", ") : "Track all classes"}</summary>
              <div className="zone-class-menu">
                <label><input type="checkbox" checked={!trackingExcludedLabels.length} onChange={() => updateConfig(["detector", "tracking", "excluded_labels"], [])} /> Track all classes</label>
                {trackingClassOptions.map((label) => {
                  const checked = trackingExcludedLabels.includes(label);
                  return <label key={label}><input type="checkbox" checked={checked} onChange={() => updateConfig(["detector", "tracking", "excluded_labels"], checked ? trackingExcludedLabels.filter((item) => item !== label) : [...trackingExcludedLabels, label])} /> {label}</label>;
                })}
              </div>
            </details>
            <small>Select classes to exclude. Face detection and recognition continue normally; excluded classes simply do not receive track IDs.</small>
          </div>
          <label>Maximum duration<input type="number" min="3" max="120" step="1" value={config.detector?.tracking?.max_session_seconds ?? 15} onChange={(event) => updateConfig(["detector", "tracking", "max_session_seconds"], Number(event.target.value))} /><small>Seconds after initial detection.</small></label>
          <label>Lost-object grace<input type="number" min="0.5" max="15" step="0.5" value={config.detector?.tracking?.lost_timeout_seconds ?? 3} onChange={(event) => updateConfig(["detector", "tracking", "lost_timeout_seconds"], Number(event.target.value))} /><small>Seconds to retain an obstructed object.</small></label>
          <label>Baseline camera limit<input type="number" min="1" max="16" step="1" value={config.detector?.tracking?.max_active_cameras ?? 2} onChange={(event) => updateConfig(["detector", "tracking", "max_active_cameras"], Number(event.target.value))} /><small>Normal simultaneous tracking sessions.</small></label>
          <label className="compact-toggle"><input type="checkbox" checked={config.detector?.tracking?.adaptive_burst_enabled ?? true} onChange={(event) => updateConfig(["detector", "tracking", "adaptive_burst_enabled"], event.target.checked)} /><span>Allow an extra tracker when healthy</span><small>Temporarily uses the burst limit only while inference has no backlog and system memory is healthy.</small></label>
          <label>Burst camera limit<input type="number" min={config.detector?.tracking?.max_active_cameras ?? 2} max="16" step="1" value={config.detector?.tracking?.burst_max_active_cameras ?? 3} onChange={(event) => updateConfig(["detector", "tracking", "burst_max_active_cameras"], Number(event.target.value))} /><small>Maximum only during a healthy short burst.</small></label>
          <label>Wait for tracking capacity<input type="number" min="0" max="30" step="0.5" value={config.detector?.tracking?.capacity_wait_seconds ?? 5} onChange={(event) => updateConfig(["detector", "tracking", "capacity_wait_seconds"], Number(event.target.value))} /><small>Wait briefly for a busy tracking slot, then recover the gap from recordings. Zero skips immediately.</small></label>
          </div>
        <details className="detection-compact-details">
          <summary>Association tuning</summary>
          <div className="detection-field-grid advanced-tracking-grid">
            <label className="compact-toggle"><input type="checkbox" checked={config.detector?.tracking?.enabled ?? true} onChange={(event) => updateConfig(["detector", "tracking", "enabled"], event.target.checked)} /><span>Enable core tracking</span><small>Troubleshooting escape hatch. Leave enabled unless this camera or hardware cannot sustain tracking.</small></label>
            <div className="detection-settings-subhead"><strong>SurvNG Hybrid tracking</strong><small>Production tracking uses SurvNG’s timestamp-aware geometry and selective appearance recovery. FastTrack is available only through the incident Compare tool.</small></div>
            <label>Confirm after detections<input type="number" min="1" max="10" step="1" value={config.detector?.tracking?.min_confirmations ?? 2} onChange={(event) => updateConfig(["detector", "tracking", "min_confirmations"], Number(event.target.value))} /><small>New objects found during an active session need this many matching observations. Incident-starting objects have already passed the event-frame confirmation above.</small></label>
            <label>Tracking confidence floor<input type="number" min="0.01" max="0.95" step="0.01" value={config.detector?.tracking?.low_confidence_threshold ?? 0.25} onChange={(event) => updateConfig(["detector", "tracking", "low_confidence_threshold"], Number(event.target.value))} /><small>Allows an existing track to survive weaker detections without creating a new incident object.</small></label>
            <label>Box match overlap<input type="number" min="0.05" max="0.9" step="0.05" value={config.detector?.tracking?.match_iou_threshold ?? 0.2} onChange={(event) => updateConfig(["detector", "tracking", "match_iou_threshold"], Number(event.target.value))} /><small>How much predicted and detected boxes must overlap to retain an ID.</small></label>
            <label>Movement match distance<input type="number" min="0.1" max="2" step="0.05" value={config.detector?.tracking?.match_center_distance_ratio ?? 0.65} onChange={(event) => updateConfig(["detector", "tracking", "match_center_distance_ratio"], Number(event.target.value))} /><small>Reconnects nearby boxes when overlap changes because someone moves quickly or approaches the camera.</small></label>
            <label>Maximum tracks per incident<input type="number" min="1" max="1000" step="10" value={config.detector?.tracking?.max_tracks_per_session ?? 100} onChange={(event) => updateConfig(["detector", "tracking", "max_tracks_per_session"], Number(event.target.value))} /><small>Safety limit for unusually noisy detector output.</small></label>
          </div>
        </details>
        <details className="detection-compact-details">
          <summary>Appearance matching (ReID)</summary>
          <div className="detection-field-grid advanced-tracking-grid">
            <div className="detection-settings-subhead"><strong>Person appearance matching</strong><small>Reconnect a person after geometry briefly loses them.</small></div>
            <label className="compact-toggle"><input type="checkbox" checked={config.detector?.tracking?.reid_enabled ?? false} onChange={(event) => updateConfig(["detector", "tracking", "reid_enabled"], event.target.checked)} /><span>Person ReID enabled</span></label>
            <label>Person ReID model<input value={config.detector?.tracking?.reid_model_path ?? ""} onChange={(event) => updateConfig(["detector", "tracking", "reid_model_path"], event.target.value)} placeholder="person-reidentification-retail-0286.xml" /><small>OpenVINO whole-person embedding model. Intel's 0286 model is the recommended accuracy-focused option; face-recognition models are not compatible.</small></label>
            <label>ReID device<input value={config.detector?.tracking?.reid_device ?? "AUTO"} onChange={(event) => updateConfig(["detector", "tracking", "reid_device"], event.target.value)} /><small>Runs in a separate isolated inference worker.</small></label>
            <label>Appearance similarity<input type="number" min="0" max="1" step="0.01" value={config.detector?.tracking?.reid_match_threshold ?? 0.7} onChange={(event) => updateConfig(["detector", "tracking", "reid_match_threshold"], Number(event.target.value))} /><small>0.70 is the conservative default. Higher values reduce accidental joins but make lost identities harder to recover.</small></label>
            <label>Remember lost appearance<input type="number" min="1" max="300" step="1" value={config.detector?.tracking?.reid_max_age_seconds ?? 30} onChange={(event) => updateConfig(["detector", "tracking", "reid_max_age_seconds"], Number(event.target.value))} /><small>Seconds a lost person can recover the same track ID.</small></label>
            <div className="detection-settings-subhead"><strong>Vehicle appearance matching</strong><small>Use vehicle appearance to recover car, truck, bus, and motorcycle identities.</small></div>
            <label className="compact-toggle"><input type="checkbox" checked={config.detector?.tracking?.vehicle_reid_enabled ?? false} onChange={(event) => updateConfig(["detector", "tracking", "vehicle_reid_enabled"], event.target.checked)} /><span>Vehicle ReID enabled</span></label>
            <label>Vehicle ReID model<input value={config.detector?.tracking?.vehicle_reid_model_path ?? ""} onChange={(event) => updateConfig(["detector", "tracking", "vehicle_reid_model_path"], event.target.value)} placeholder="vehicle-reid-0001.xml" /><small>OpenVINO whole-vehicle embedding model. This is separate from the person model.</small></label>
            <label>Vehicle labels<input value={(config.detector?.tracking?.vehicle_reid_labels || ["car", "truck", "bus", "motorcycle"]).join(", ")} onChange={(event) => updateConfig(["detector", "tracking", "vehicle_reid_labels"], event.target.value.split(",").map((value) => value.trim().toLowerCase()).filter(Boolean))} /><small>Comma-separated detector labels that use vehicle appearance matching.</small></label>
            <label>Vehicle ReID device<input value={config.detector?.tracking?.vehicle_reid_device ?? "AUTO"} onChange={(event) => updateConfig(["detector", "tracking", "vehicle_reid_device"], event.target.value)} /><small>Shares the isolated appearance worker but uses its own OpenVINO model.</small></label>
            <label>Vehicle appearance similarity<input type="number" min="0" max="1" step="0.01" value={config.detector?.tracking?.vehicle_reid_match_threshold ?? 0.8} onChange={(event) => updateConfig(["detector", "tracking", "vehicle_reid_match_threshold"], Number(event.target.value))} /><small>Higher values reduce accidental merging of similar-looking vehicles.</small></label>
            <label>Maximum appearance checks<input type="number" min="1" max="64" step="1" value={config.detector?.tracking?.reid_max_embeddings_per_frame ?? 8} onChange={(event) => updateConfig(["detector", "tracking", "reid_max_embeddings_per_frame"], Number(event.target.value))} /><small>Bounds combined person and vehicle ReID work in a crowded frame.</small></label>
            <label>Refresh appearance every<input type="number" min="1" max="120" step="1" value={config.detector?.tracking?.reid_refresh_interval_frames ?? 8} onChange={(event) => updateConfig(["detector", "tracking", "reid_refresh_interval_frames"], Number(event.target.value))} /><small>Matched samples between appearance refreshes. Geometry handles the intervening frames; lower values use more GPU.</small></label>
            <div className="detection-settings-subhead"><strong>Missed-session recovery</strong><small>Recover durable appearance evidence from the saved incident image after full tracking finishes or is skipped.</small></div>
            <label className="compact-toggle"><input type="checkbox" checked={config.detector?.tracking?.deferred_reid_enabled ?? true} onChange={(event) => updateConfig(["detector", "tracking", "deferred_reid_enabled"], event.target.checked)} /><span>Recover missed appearance evidence</span></label>
            <label>Recovery delay<input type="number" min="0" max="300" step="1" value={config.detector?.tracking?.deferred_reid_delay_seconds ?? 20} onChange={(event) => updateConfig(["detector", "tracking", "deferred_reid_delay_seconds"], Number(event.target.value))} /><small>Waits for stronger multi-frame tracking evidence before using a single saved snapshot.</small></label>
            <label>Nearby-camera window<input type="number" min="1" max="300" step="1" value={config.detector?.tracking?.related_sequence_window_seconds ?? 30} onChange={(event) => updateConfig(["detector", "tracking", "related_sequence_window_seconds"], Number(event.target.value))} /><small>Seconds on either side used to show clearly labeled sequence candidates. Time alone never claims identity.</small></label>
            <div className="detection-settings-subhead camera-route-heading"><div><strong>Expected camera routes</strong><small>Describe physically plausible camera-to-camera movement. Direction follows event time; routes strengthen ordering but never establish identity by themselves.</small></div><button type="button" onClick={addCameraRoute} disabled={routeCameras.length < 2}>Add route</button></div>
            <div className="camera-route-list">
              {cameraTransitionRoutes.length ? cameraTransitionRoutes.map((route, index) => <div className="camera-route-row" key={`${route.from_camera}-${route.to_camera}-${index}`}>
                <label>From<select value={route.from_camera} onChange={(event) => updateCameraRoute(index, "from_camera", event.target.value)}>{routeCameras.map((camera) => <option value={camera.id} key={camera.id}>{camera.name || camera.id}</option>)}</select></label>
                <span className="camera-route-arrow">→</span>
                <label>To<select value={route.to_camera} onChange={(event) => updateCameraRoute(index, "to_camera", event.target.value)}>{routeCameras.map((camera) => <option value={camera.id} key={camera.id}>{camera.name || camera.id}</option>)}</select></label>
                <label>Earliest<input type="number" min="0" max="299" step="1" value={route.min_seconds ?? 0} onChange={(event) => updateCameraRoute(index, "min_seconds", Number(event.target.value))} /><small>seconds</small></label>
                <label>Latest<input type="number" min="1" max="300" step="1" value={route.max_seconds ?? 30} onChange={(event) => updateCameraRoute(index, "max_seconds", Number(event.target.value))} /><small>seconds</small></label>
                <label className="compact-toggle"><input type="checkbox" checked={route.bidirectional ?? false} onChange={(event) => updateCameraRoute(index, "bidirectional", event.target.checked)} /><span>Both directions</span></label>
                <label className="compact-toggle"><input type="checkbox" checked={route.enabled ?? true} onChange={(event) => updateCameraRoute(index, "enabled", event.target.checked)} /><span>Enabled</span></label>
                <button type="button" className="danger" onClick={() => updateConfig(["detector", "tracking", "camera_transition_routes"], cameraTransitionRoutes.filter((_item, routeIndex) => routeIndex !== index))}>Remove</button>
              </div>) : <p className="settings-help">No expected routes yet. Nearby incidents still appear as general sequence candidates.</p>}
            </div>
          </div>
          {config.detector?.tracking?.reid_enabled ? (
            reidStatus?.enabled ? (
              <div className={`probe-result ${(reidStatus.person?.ready ?? reidStatus.ready) ? "ok" : "bad"}`}>
                <strong>{(reidStatus.person?.ready ?? reidStatus.ready) ? "Person appearance matching is ready" : "Person appearance matching is unavailable"}</strong>
                <span>{(reidStatus.person?.ready ?? reidStatus.ready) ? `${reidStatus.person?.device || reidStatus.device || "AUTO"} · ${reidStatus.person?.embedding_size || reidStatus.embedding_size || 0}-value appearance signature` : reidStatus.person?.error || reidStatus.error || "The isolated ReID worker did not start."}</span>
                {(reidStatus.person?.ready ?? reidStatus.ready) && (reidStatus.person?.model_load_ms ?? reidStatus.model_load_ms) != null ? <span>Model loaded in {Math.round(reidStatus.person?.model_load_ms ?? reidStatus.model_load_ms)} ms</span> : null}
              </div>
            ) : <div className="probe-result"><strong>Person appearance matching is not active yet</strong><span>Save the configuration and restart SurvNG to start its isolated model worker.</span></div>
          ) : null}
          {config.detector?.tracking?.vehicle_reid_enabled ? (
            reidStatus?.enabled ? (
              <div className={`probe-result ${reidStatus.vehicle?.ready ? "ok" : "bad"}`}>
                <strong>{reidStatus.vehicle?.ready ? "Vehicle appearance matching is ready" : "Vehicle appearance matching is unavailable"}</strong>
                <span>{reidStatus.vehicle?.ready ? `${reidStatus.vehicle.device || "AUTO"} · ${reidStatus.vehicle.embedding_size || 0}-value vehicle signature · ${(reidStatus.vehicle.labels || []).join(", ")}` : reidStatus.vehicle?.error || "The vehicle ReID model did not start."}</span>
                {reidStatus.vehicle?.ready && reidStatus.vehicle.model_load_ms != null ? <span>Model loaded in {Math.round(reidStatus.vehicle.model_load_ms)} ms</span> : null}
              </div>
            ) : <div className="probe-result"><strong>Vehicle appearance matching is not active yet</strong><span>Save the configuration and restart SurvNG to start the model.</span></div>
          ) : null}
        </details>
        </section>

        <details className="detection-settings-card detection-feature-card">
          <summary><span className="detection-settings-card-icon"><Search size={18} /></span><span><strong>Smart Search</strong><small>Find indexed incidents by describing visible details in plain language.</small></span></summary>
          <div className="detection-feature-body detection-field-grid">
            <label className="compact-toggle"><input type="checkbox" checked={config.semantic_search?.enabled ?? false} onChange={(event) => updateConfig(["semantic_search", "enabled"], event.target.checked)} /><span>Smart Search enabled</span></label>
            <label>Model package<input value={config.semantic_search?.model_dir ?? ""} onChange={(event) => updateConfig(["semantic_search", "model_dir"], event.target.value)} placeholder="/path/to/SurvNG/models/mobileclip2-b-openvino-fp16" /><small>Use the host path for systemd or the mounted container path for Docker. The package contains semantic_model.json, both encoders, and tokenizer assets.</small></label>
            <label>Inference device<input value={config.semantic_search?.device ?? "GPU"} onChange={(event) => updateConfig(["semantic_search", "device"], event.target.value)} /><small>GPU is recommended on Intel systems. This does not share the object detector queue.</small></label>
            <label>Historical batch size<input type="number" min="1" max="250" step="1" value={config.semantic_search?.backfill_batch_size ?? 25} onChange={(event) => updateConfig(["semantic_search", "backfill_batch_size"], Number(event.target.value))} /><small>How many older incidents are scheduled at a time. Existing indexed generations are skipped.</small></label>
            <label>Historical pacing<input type="number" min="0.01" max="5" step="0.05" value={config.semantic_search?.backfill_pause_seconds ?? 0.25} onChange={(event) => updateConfig(["semantic_search", "backfill_pause_seconds"], Number(event.target.value))} /><small>Pause between older incidents so object detection and new Smart Search evidence retain priority.</small></label>
            <label className="compact-toggle"><input type="checkbox" checked={config.semantic_search?.index_full_frame ?? true} onChange={(event) => updateConfig(["semantic_search", "index_full_frame"], event.target.checked)} /><span>Index whole incident image</span></label>
            <label className="compact-toggle"><input type="checkbox" checked={config.semantic_search?.index_object_crops ?? true} onChange={(event) => updateConfig(["semantic_search", "index_object_crops"], event.target.checked)} /><span>Index detected object crops</span></label>
            <label>Object crops per incident<input type="number" min="1" max="100" step="1" value={config.semantic_search?.max_object_crops_per_event ?? 24} onChange={(event) => updateConfig(["semantic_search", "max_object_crops_per_event"], Number(event.target.value))} /><small>Caps crop inference and memory for unusually busy incidents; highest-confidence detections are indexed first.</small></label>
          </div>
        </details>

        <details className="detection-settings-card detection-feature-card wide-card">
          <summary><span className="detection-settings-card-icon"><Gauge size={18} /></span><span><strong>Motion validation</strong><small>How camera and visual motion decide when object detection runs.</small></span></summary>
          <div className="detection-feature-body">
        <MotionAnalysisPresetEditor
          qualification={config.motion_qualification?.pipeline?.qualification || []}
          catalog={motionCatalog}
          onChange={(qualification) => updateConfig(
            ["motion_qualification", "pipeline"],
            { ...(config.motion_qualification?.pipeline || {}), qualification },
          )}
        />
        <details className="motion-tuning-details">
          <summary>Advanced motion tuning</summary>
          <div className="field-row">
          <label>Sensitivity<select value={config.motion_qualification?.sensitivity || "balanced"} onChange={(event) => updateConfig(["motion_qualification", "sensitivity"], event.target.value)}><option value="high">High</option><option value="balanced">Balanced</option><option value="low">Low</option></select></label>
          <label>Light and shadow filtering<select value={String(config.motion_qualification?.illumination_filter_enabled ?? false)} onChange={(event) => updateConfig(["motion_qualification", "illumination_filter_enabled"], event.target.value === "true")}><option value="false">Disabled</option><option value="true">Enabled</option></select><small>Ignores clear moving illumination while uncertain motion continues to object detection. Disabled still records evidence for evaluation.</small></label>
          <label>Analysis size<select value={config.motion_qualification?.frame_width ?? 320} onChange={(event) => updateConfig(["motion_qualification", "frame_width"], Number(event.target.value))}><option value="320">320 px</option><option value="480">480 px</option><option value="640">640 px</option><option value="720">720 px</option><option value="800">800 px</option></select><small>Maximum image edge used by EMA; portrait cameras no longer expand beyond this size.</small></label>
          <label>Sample FPS<input type="number" min="2" max="10" step="1" value={config.motion_qualification?.sample_fps ?? 5} onChange={(event) => updateConfig(["motion_qualification", "sample_fps"], Number(event.target.value))} /></label>
          <label>ONVIF background upkeep<select value={String(config.motion_qualification?.camera_mode_background_fps ?? 2)} onChange={(event) => updateConfig(["motion_qualification", "camera_mode_background_fps"], Number(event.target.value))}><option value="1">Low CPU (1 frame/sec)</option><option value="2">Balanced (2 frames/sec)</option><option value="3">Faster adaptation (3 frames/sec)</option><option value="5">Maximum adaptation (5 frames/sec)</option></select><small>When camera alerts trigger motion, SurvNG maintains the visual background at this lower rate. Trigger validation still analyzes the full buffered window.</small></label>
          {config.motion_qualification?.mode === "camera_rescue" ? <>
            <div className="detection-settings-subhead"><strong>Visual backup safeguards</strong><small>These conservative limits control when SurvNG may compensate for a missing camera notice.</small></div>
            <label>Scene learning time<input type="number" min="0" max="120" step="1" value={config.motion_qualification?.visual_backup_warmup_seconds ?? 10} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_warmup_seconds"], Number(event.target.value))} /><small>After this unchanged startup period, EMA also waits for a quiet scene baseline. Camera alerts continue normally throughout.</small></label>
            <label>Wait for camera notice<input type="number" min="0" max="5" step="0.25" value={config.motion_qualification?.visual_backup_grace_seconds ?? 1.5} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_grace_seconds"], Number(event.target.value))} /><small>Seconds strong visual motion must persist while SurvNG waits for ONVIF.</small></label>
            <label>Minimum visual confidence<input type="number" min="0" max="1" step="0.01" value={config.motion_qualification?.visual_backup_min_score ?? 0.7} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_min_score"], Number(event.target.value))} /><small>Absolute adaptive score required before visual backup is considered.</small></label>
            <label>Confidence above normal<input type="number" min="0" max="0.5" step="0.01" value={config.motion_qualification?.visual_backup_score_margin ?? 0.15} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_score_margin"], Number(event.target.value))} /><small>Additional margin above the camera&apos;s adaptive threshold.</small></label>
            <label>Consecutive strong samples<input type="number" min="2" max="10" step="1" value={config.motion_qualification?.visual_backup_min_consecutive ?? 3} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_min_consecutive"], Number(event.target.value))} /><small>Prevents a single noisy frame from invoking object detection.</small></label>
            <label>Backup cooldown<input type="number" min="5" max="300" step="5" value={config.motion_qualification?.visual_backup_cooldown_seconds ?? 20} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_cooldown_seconds"], Number(event.target.value))} /><small>Minimum seconds between visual backup attempts and after a camera notice.</small></label>
            <label>Maximum backups per 5 minutes<input type="number" min="1" max="30" step="1" value={config.motion_qualification?.visual_backup_max_triggers_5m ?? 3} onChange={(event) => updateConfig(["motion_qualification", "visual_backup_max_triggers_5m"], Number(event.target.value))} /><small>Hard per-camera safety limit for object-detector work.</small></label>
          </> : null}
          <label>Window Seconds<input type="number" min="0.8" max="4" step="0.1" value={config.motion_qualification?.window_seconds ?? 1.6} onChange={(event) => updateConfig(["motion_qualification", "window_seconds"], Number(event.target.value))} /></label>
          <label>Post-trigger Seconds<input type="number" min="0.5" max="6" step="0.1" value={config.motion_qualification?.post_trigger_seconds ?? 2.5} onChange={(event) => updateConfig(["motion_qualification", "post_trigger_seconds"], Number(event.target.value))} /></label>
          <label>Burst Quiet Seconds<input type="number" min="0.1" max="2" step="0.1" value={config.motion_qualification?.burst_quiet_seconds ?? 0.5} onChange={(event) => updateConfig(["motion_qualification", "burst_quiet_seconds"], Number(event.target.value))} /></label>
          <label>Save rejected motion images<select value={String(config.motion_qualification?.rejected_sample_rate ?? 1)} onChange={(event) => updateConfig(["motion_qualification", "rejected_sample_rate"], Number(event.target.value))}><option value="1">Every rejection (Recommended)</option><option value="0.5">About half</option><option value="0.1">About 1 in 10</option><option value="0.05">About 1 in 20</option><option value="0">Never</option></select><small>Used by Motion Audit and the AI Advisor. SurvNG keeps the latest 100 per camera.</small></label>
          <label>Double-check filtered motion<select value={String(config.motion_qualification?.suppression_verification_rate ?? 0.05)} onChange={(event) => updateConfig(["motion_qualification", "suppression_verification_rate"], Number(event.target.value))}><option value="0">Off</option><option value="0.01">About 1 in 100</option><option value="0.05">About 1 in 20 (Recommended)</option><option value="0.1">About 1 in 10</option></select><small>Runs object detection on a sample of visual rejections. If a configured object is found, SurvNG restores the incident; otherwise only Motion Audit records the check.</small></label>
          <label className="check-field"><input type="checkbox" checked={config.motion_qualification?.borderline_rescue_enabled ?? true} onChange={(event) => updateConfig(["motion_qualification", "borderline_rescue_enabled"], event.target.checked)} /> Borderline object rescue</label>
          <label>Rescue Margin<input type="number" min="0" max="0.1" step="0.005" value={config.motion_qualification?.borderline_margin ?? 0.03} onChange={(event) => updateConfig(["motion_qualification", "borderline_margin"], Number(event.target.value))} /></label>
          </div>
        </details>
        <MotionDecisionEditor
          fusion={config.motion_qualification?.pipeline?.fusion}
          mode={config.motion_qualification?.mode || "camera_rescue"}
          onModeChange={(mode) => updateConfig(["motion_qualification", "mode"], mode)}
          onChange={(fusion) => updateConfig(
            ["motion_qualification", "pipeline"],
            { ...(config.motion_qualification?.pipeline || {}), fusion },
          )}
        />
          </div>
        </details>

        <details className="detection-settings-card detection-feature-card">
          <summary><span className="detection-settings-card-icon"><Sparkles size={18} /></span><span><strong>AI analysis &amp; assistant</strong><small>One provider and API key, with your existing analysis model plus an optional deep-reasoning model.</small></span></summary>
          <div className="detection-feature-body detection-field-grid">
          <label className="compact-toggle"><input type="checkbox" checked={config.audit_ai?.enabled ?? false} onChange={(event) => updateConfig(["audit_ai", "enabled"], event.target.checked)} /><span>AI features enabled</span></label>
          <label className="compact-toggle"><input type="checkbox" checked={config.audit_ai?.assistant_enabled ?? true} onChange={(event) => updateConfig(["audit_ai", "assistant_enabled"], event.target.checked)} disabled={!config.audit_ai?.enabled} /><span>SurvNG Assistant enabled</span></label>
          <label>Provider<select value={config.audit_ai?.provider || "openai"} onChange={(event) => updateConfig(["audit_ai", "provider"], event.target.value)}>
            <option value="openai">OpenAI</option>
            <option value="gemini">Google Gemini</option>
            <option value="openai_compatible">OpenAI compatible</option>
          </select></label>
          <label>Everyday AI model<input value={config.audit_ai?.model || ""} onChange={(event) => updateConfig(["audit_ai", "model"], event.target.value)} placeholder={config.audit_ai?.provider === "gemini" ? "gemini-2.5-flash" : "gpt-4.1-mini"} /><small>Used for Motion Audit reviews, finding incidents, status questions, and straightforward answers.</small></label>
          <label>Detailed analysis model<input value={config.audit_ai?.assistant_reasoning_model || ""} onChange={(event) => updateConfig(["audit_ai", "assistant_reasoning_model"], event.target.value)} placeholder="Leave blank to use the everyday model" /><small>Optional second model for visual incident reviews, difficult diagnoses, comparisons, and tuning advice.</small></label>
          <label>API Key<input type="password" value={secretInputValue(config.audit_ai?.api_key)} placeholder={secretInputHint(config.audit_ai?.api_key)} onChange={(event) => updateConfig(["audit_ai", "api_key"], event.target.value)} autoComplete="new-password" /></label>
          <label>Base URL<input value={config.audit_ai?.base_url || ""} onChange={(event) => updateConfig(["audit_ai", "base_url"], event.target.value)} placeholder={config.audit_ai?.provider === "gemini" ? "https://generativelanguage.googleapis.com/v1beta" : config.audit_ai?.provider === "openai_compatible" ? "http://localhost:11434/v1" : "https://api.openai.com/v1"} /></label>
          <label>Timeout Seconds<input type="number" min="5" max="120" step="1" value={config.audit_ai?.timeout_seconds ?? 45} onChange={(event) => updateConfig(["audit_ai", "timeout_seconds"], Number(event.target.value))} /></label>
          <label className="compact-toggle"><input type="checkbox" checked={config.audit_ai?.allow_apply_recommendations ?? false} onChange={(event) => updateConfig(["audit_ai", "allow_apply_recommendations"], event.target.checked)} /><span>Allow confirmed changes</span></label>
          </div>
        </details>

        <details className="detection-settings-card detection-feature-card">
          <summary><span className="detection-settings-card-icon"><ScanFace size={18} /></span><span><strong>Face recognition</strong><small>Identify detected faces using a separate embedding model.</small></span></summary>
          <div className="detection-feature-body detection-field-grid">
          <label className="compact-toggle"><input type="checkbox" checked={config.detector?.face_recognition_enabled ?? false} onChange={(event) => updateConfig(["detector", "face_recognition_enabled"], event.target.checked)} /><span>Recognition enabled</span></label>
          <label>Embedding Model<input value={config.detector?.face_embedding_model_path || ""} onChange={(event) => updateConfig(["detector", "face_embedding_model_path"], event.target.value)} placeholder="face_model/model.xml" /></label>
          <label>Landmark Model<input value={config.detector?.face_landmark_model_path || ""} onChange={(event) => updateConfig(["detector", "face_landmark_model_path"], event.target.value)} placeholder="face_model/landmarks.xml" /></label>
          <label>Face Detector Model<input value={config.detector?.face_detection_model_path || ""} onChange={(event) => updateConfig(["detector", "face_detection_model_path"], event.target.value)} placeholder="face_detector/model.xml" /></label>
          <label>Recognition Device<select value={config.detector?.face_recognition_device || "AUTO"} onChange={(event) => updateConfig(["detector", "face_recognition_device"], event.target.value)}>
            {deviceOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          <label>Face Detection Confidence<input type="number" min="0.01" max="0.99" step="0.01" value={config.detector?.face_detection_threshold ?? 0.6} onChange={(event) => updateConfig(["detector", "face_detection_threshold"], Number(event.target.value))} /></label>
          <label>Suggestion Threshold<input type="number" min="0" max="1" step="0.01" value={config.detector?.face_match_threshold ?? 0.4} onChange={(event) => updateConfig(["detector", "face_match_threshold"], Number(event.target.value))} /></label>
          <label>Minimum Face Size<input type="number" min="16" max="1024" step="8" value={config.detector?.face_min_size ?? 48} onChange={(event) => updateConfig(["detector", "face_min_size"], Number(event.target.value))} /></label>
          <label>References Per Person<input type="number" min="1" max="200" step="1" value={config.detector?.face_max_references ?? 20} onChange={(event) => updateConfig(["detector", "face_max_references"], Number(event.target.value))} /><small>SurvNG chooses the clearest, most varied confirmed faces; pinned references are always retained.</small></label>
          <label>Saved face limit<input type="number" min="100" max="100000" step="100" value={config.detector?.face_max_observations ?? 1000} onChange={(event) => updateConfig(["detector", "face_max_observations"], Number(event.target.value))} /><small>Oldest observations are removed first.</small></label>
          <label className="compact-toggle"><input type="checkbox" checked={config.detector?.face_auto_identify_enabled ?? false} onChange={(event) => updateConfig(["detector", "face_auto_identify_enabled"], event.target.checked)} /><span>Automatically identify very strong matches</span></label>
          <label>Automatic Match Threshold<input type="number" min="0" max="1" step="0.01" value={config.detector?.face_auto_identify_threshold ?? 0.55} onChange={(event) => updateConfig(["detector", "face_auto_identify_threshold"], Number(event.target.value))} /></label>
          <label>Minimum Lead Over Next Person<input type="number" min="0" max="1" step="0.01" value={config.detector?.face_auto_identify_margin ?? 0.12} onChange={(event) => updateConfig(["detector", "face_auto_identify_margin"], Number(event.target.value))} /></label>
          </div>
        </details>

        <details className="detection-settings-card detection-feature-card diagnostics-card">
          <summary><span className="detection-settings-card-icon"><Cpu size={18} /></span><span><strong>Model &amp; accelerator diagnostics</strong><small>Loaded model metadata and available processing hardware.</small></span></summary>
          <div className="detection-feature-body diagnostics-grid">
        {activeModel ? (
          <div className={`probe-result ${activeModel.valid ? "ok" : "bad"}`}>
            <strong>{activeModel.valid ? "OpenVINO IR ready" : "OpenVINO IR incomplete"}</strong>
            <span>XML: {activeModel.path}</span>
            <span>Weights: {activeModel.bin_present ? activeModel.bin_path : "matching .bin file not found"}</span>
            <span>Input: {activeModel.input_shape.join(" x ") || "unknown"}</span>
            <span>Output: {activeModel.output_shapes.map((shape) => shape.join(" x ")).join(", ") || "unknown"}</span>
            <span>Task: {activeModel.task || "detect"}</span>
            <span>Classes: {activeModel.classes.join(", ") || "none found"}</span>
            {activeModel.error ? <span>{activeModel.error}</span> : null}
          </div>
        ) : null}
        {detectorBackend === "coreml" ? (
          <div className="detection-field-grid">
            <label>Core ML Model Path<input value={config.detector?.coreml_model_path || ""} onChange={(event) => updateConfig(["detector", "coreml_model_path"], event.target.value)} placeholder="model.mlpackage or model.mlmodel" /></label>
          </div>
        ) : null}
        <div className="probe-result">
          <strong>Accelerator</strong>
          <span>System: {accelerator ? `${accelerator.system} ${accelerator.machine}` : "checking..."}</span>
          <span>Detector recommendation: {accelerator?.recommended_detector_backend === "coreml" ? "Core ML" : "OpenVINO / ONNX"}</span>
          <span>{coremlLabel}</span>
          <span>OpenVINO devices: {openvinoDevices.length ? openvinoDevices.join(", ") : "none reported"}</span>
          <span>{gpuLabel}</span>
          <span>FFmpeg acceleration: {ffmpegAcceleration.configured || config.hardware_acceleration || "auto"}</span>
          <span>FFmpeg: {ffmpegAcceleration.ffmpeg_path || accelerator?.ffmpeg_path || config.ffmpeg_path || "ffmpeg"}</span>
          <span>FFprobe: {ffmpegAcceleration.ffprobe_path || accelerator?.ffprobe_path || "ffprobe"}</span>
          <span>FFplay: {ffmpegAcceleration.ffplay_path || accelerator?.ffplay_path || "ffplay"}</span>
          <span>{vaapiLabel}</span>
          {vaapi.render_devices?.length ? <span>VAAPI render devices: {vaapi.render_devices.join(", ")}</span> : null}
          {vaapi.filters?.length ? <span>VAAPI filters: {vaapi.filters.join(", ")}</span> : null}
          {vaapi.runtime_error ? <span>VAAPI runtime: {vaapi.runtime_error}</span> : null}
          <span>{qsvLabel}</span>
          {qsv.render_devices?.length ? <span>QSV render devices: {qsv.render_devices.join(", ")}</span> : null}
          {qsv.decoders?.length ? <span>QSV decoders: {qsv.decoders.join(", ")}</span> : null}
          {qsv.runtime_error ? <span>QSV runtime: {qsv.runtime_error}</span> : null}
          {accelerator?.recommended_openvino_device ? <span>Recommended OpenVINO device: {accelerator.recommended_openvino_device}</span> : null}
          {accelerator?.coreml_error ? <span>{accelerator.coreml_error}</span> : null}
          {accelerator?.openvino_error ? <span>{accelerator.openvino_error}</span> : null}
        </div>
          </div>
        </details>
      </div>
      ) : null}

      {section === "motion-review" ? (
        <MotionAiReviewPanel
          cameras={config.cameras || []}
          advisorEnabled={config.audit_ai?.enabled ?? false}
        />
      ) : null}
    </div>
  );
}

function MotionPipelineRuntimeCard({ label, pipeline, origin, motionCatalog }) {
  if (!pipeline) return null;
  const metrics = Object.values(pipeline.stages || {});
  const calls = Math.max(0, ...metrics.map((item) => Number(item.calls) || 0));
  const failures = metrics.reduce((total, item) => total + (Number(item.failures) || 0), 0);
  const averageMs = metrics.reduce((total, item) => total + (Number(item.average_ms) || 0), 0);
  const lastMs = metrics.reduce((total, item) => total + (Number(item.last_ms) || 0), 0);
  const health = failures ? "attention" : calls ? "healthy" : "ready";
  const parallelGroups = (pipeline.execution_groups || []).filter((group) => group.mode === "parallel");
  const parallelStages = parallelGroups.reduce((total, group) => total + (group.stages?.length || 0), 0);
  const stageNames = new Map(
    (motionCatalog?.stages || []).map((stage) => [stage.implementation, stage.name]),
  );
  const originLabel = origin === "camera" ? "Camera override" : origin === "global" ? "Global setting" : "Built-in default";
  return (
    <div className={`motion-pipeline-runtime-card ${health}`}>
      <div className="motion-pipeline-runtime-head">
        <strong>{label}</strong>
        <span>{health === "attention" ? "Needs attention" : health === "healthy" ? "Healthy" : "Ready"}</span>
      </div>
      <small>{originLabel} · {pipeline.configuration?.length || 0} steps · {calls.toLocaleString()} cycles{parallelStages ? ` · ${parallelStages} parallel branches` : ""}</small>
      <div className="motion-pipeline-timing">
        <span>Last <strong>{lastMs.toFixed(2)} ms</strong></span>
        <span>Average <strong>{averageMs.toFixed(2)} ms</strong></span>
        <span>Failures <strong>{failures}</strong></span>
      </div>
      <details>
        <summary>Processing steps</summary>
        <div className="motion-pipeline-stage-list">
          {(pipeline.configuration || []).map((stage) => {
            const stageMetrics = pipeline.stages?.[stage.stage_id] || {};
            return (
              <div key={stage.stage_id}>
                <span><strong>{stageNames.get(stage.implementation) || stage.implementation}</strong><small>{stage.stage_id}</small></span>
                <span>{Number(stageMetrics.average_ms || 0).toFixed(2)} ms avg{stageMetrics.failures ? ` · ${stageMetrics.failures} failed` : ""}</span>
              </div>
            );
          })}
        </div>
      </details>
    </div>
  );
}

function MotionDebugViewer({ cameraId, timeZone }) {
  const [status, setStatus] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState("overlay");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const ownedRef = useRef(false);
  const debugRequestSequence = useRef(0);

  async function loadStatus(renew = false) {
    const sequence = ++debugRequestSequence.current;
    const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/motion-debug`, renew ? {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: true }),
    } : undefined);
    if (!response.ok) throw new Error("Could not load motion diagnostics");
    const payload = await response.json();
    if (sequence !== debugRequestSequence.current) return null;
    setStatus(payload);
    const layers = payload.snapshot?.layers || [];
    if (layers.length && !layers.some((layer) => layer.id === selectedLayer)) {
      setSelectedLayer(layers[0].id);
    }
    return payload;
  }

  useEffect(() => {
    let active = true;
    setStatus(null);
    setError("");
    fetch(`/api/cameras/${encodeURIComponent(cameraId)}/motion-debug`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Could not load motion diagnostics")))
      .then((payload) => { if (active) setStatus(payload); })
      .catch((loadError) => { if (active) setError(loadError.message); });
    return () => {
      active = false;
      debugRequestSequence.current += 1;
      if (ownedRef.current) {
        fetch(`/api/cameras/${encodeURIComponent(cameraId)}/motion-debug`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: false }),
          keepalive: true,
        }).catch(() => {});
        ownedRef.current = false;
      }
    };
  }, [cameraId]);

  useEffect(() => {
    if (!status?.enabled) return undefined;
    let inFlight = false;
    let active = true;
    const timer = window.setInterval(() => {
      if (inFlight) return;
      inFlight = true;
      loadStatus(ownedRef.current && Number(status.expires_in_seconds || 0) < 70)
        .catch((loadError) => { if (active) setError(loadError.message); })
        .finally(() => { inFlight = false; });
    }, 2000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [cameraId, status?.enabled, status?.expires_in_seconds]);

  async function setEnabled(enabled) {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/motion-debug`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      });
      if (!response.ok) throw new Error("Could not update motion diagnostics");
      ownedRef.current = enabled;
      setStatus(await response.json());
    } catch (updateError) {
      setError(updateError.message);
    } finally {
      setBusy(false);
    }
  }

  const snapshot = status?.snapshot;
  const layers = snapshot?.layers || [];
  const imageUrl = snapshot && selectedLayer
    ? appUrl(`/api/cameras/${encodeURIComponent(cameraId)}/motion-debug/${encodeURIComponent(selectedLayer)}.jpg?t=${snapshot.captured_at}`)
    : "";
  return (
    <div className="sub-panel motion-debug-viewer">
      <div className="motion-debug-heading">
        <div>
          <h3>Motion Diagnostics</h3>
          <span>See what each processing step sees. Runs only for this camera and expires automatically.</span>
        </div>
        <button type="button" className={status?.enabled ? "danger" : ""} disabled={busy} onClick={() => setEnabled(!status?.enabled)}>
          {busy ? <RefreshCcw className="spin" size={15} /> : <Activity size={15} />}
          {status?.enabled ? "Stop Diagnostics" : "Start Diagnostics"}
        </button>
      </div>
      {error ? <div className="motion-analysis-warning">{error}</div> : null}
      {status?.enabled && !snapshot ? <div className="motion-debug-waiting"><RefreshCcw className="spin" size={16} /> Collecting the first diagnostic frame...</div> : null}
      {snapshot ? (
        <div className="motion-debug-content">
          <div className="motion-debug-image-panel">
            <label>Diagnostic view<select value={selectedLayer} onChange={(event) => setSelectedLayer(event.target.value)}>
              {layers.map((layer) => <option key={layer.id} value={layer.id}>{layer.label}</option>)}
            </select></label>
            {imageUrl ? <img src={imageUrl} alt={layers.find((layer) => layer.id === selectedLayer)?.label || "Motion diagnostic"} /> : null}
          </div>
          <div className="motion-debug-details">
            <strong>{snapshot.accepted ? "Motion accepted" : "Motion not accepted"}</strong>
            <span>{Math.round(Number(snapshot.score || 0) * 100)}% score · {Math.round(Number(snapshot.threshold || 0) * 100)}% needed</span>
            <span>{snapshot.frame_count || 0} frames · {snapshot.blob_count || 0} regions · {snapshot.track_points || 0} tracked points</span>
            <span>{snapshot.reason || "No reason reported"}</span>
            <span>{formatDateTime(new Date(Number(snapshot.captured_at) * 1000).toISOString(), timeZone)}</span>
            <details>
              <summary>Stage timing</summary>
              {Object.entries(snapshot.timings || {}).map(([stage, milliseconds]) => <span key={stage}>{stage}: {Number(milliseconds).toFixed(2)} ms</span>)}
            </details>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function MotionEffectiveness({ cameraId, mode }) {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const response = await fetch("/api/motion-effectiveness?days=7");
        if (!response.ok) throw new Error("Effectiveness history unavailable");
        const payload = await response.json();
        if (active) {
          setSummary(payload?.by_camera?.[cameraId]?.[mode] || null);
          setError("");
        }
      } catch (loadError) {
        if (active) setError(loadError.message || "Effectiveness history unavailable");
      }
    };
    void load();
    const timer = window.setInterval(load, 60000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [cameraId, mode]);

  if (error) return <span className="motion-runtime-warning">{error}</span>;
  if (!summary) return <span>No durable motion decisions for this mode in the last 7 days.</span>;
  return (
    <div className="motion-effectiveness-summary">
      <strong>Last 7 days in this mode</strong>
      <span>{summary.allowed_events || 0} allowed · {summary.visual_filtered || 0} visually filtered · {summary.state_deduplicated || 0} merged with ongoing activity</span>
      <span>{summary.object_events || 0} allowed events found a configured object · {summary.no_object_events || 0} found none</span>
      <span>{Math.round(Number(summary.visual_rejection_rate || 0) * 100)}% visually filtered · {Math.round(Number(summary.object_yield_rate || 0) * 100)}% object yield · {summary.borderline_rescued || 0} borderline rescues</span>
      <span>{summary.suppression_verification_checks || 0} filtered events double-checked · {summary.suppression_verification_rescues || 0} restored after finding an object</span>
      {summary.visual_backup_attempts ? <span>{summary.visual_backup_attempts} visual backup attempts · {summary.visual_backup_objects || 0} found an object · {summary.visual_backup_no_object || 0} found none{summary.visual_backup_incomplete ? ` · ${summary.visual_backup_incomplete} incomplete` : ""}</span> : null}
      {summary.unreviewed_visual_filters ? <span className="motion-runtime-warning">{summary.unreviewed_visual_filters} visual filters were not independently checked by object detection.</span> : null}
    </div>
  );
}

function RuntimeStatus({ status, timeZone, motionCatalog }) {
  if (!status) {
    return <div className="probe-result"><strong>Runtime</strong><span>Save this camera to start workers.</span></div>;
  }
  const motionMode = status.motion_qualification?.mode;
  const cameraAlertsOnly = !["adaptive", "enforce"].includes(motionMode);
  const visualBackupEnabled = motionMode === "camera_rescue";
  const missingMotionNotices = cameraAlertsOnly
    && status.onvif_enabled
    && Number(status.onvif_motion_events_received || 0) === 0;
  const missingCameraTrigger = cameraAlertsOnly && !status.onvif_enabled;
  return (
    <div className="probe-result runtime-result">
      <strong>Runtime</strong>
      <span>Stream worker: {status.running ? "running" : "not running"}</span>
      <span>Recording: {status.recording ? "running" : "stopped"}</span>
      <span>ONVIF: {status.onvif_enabled ? (status.onvif_connected ? "connected" : `not connected${status.onvif_last_error ? `: ${status.onvif_last_error}` : ""}`) : "disabled"}</span>
      {status.onvif_last_event_at ? <span>Last ONVIF notification (any type): {formatDateTime(status.onvif_last_event_at, timeZone)}</span> : null}
      {status.onvif_enabled ? <span>{status.onvif_notifications_received || 0} notifications · {status.onvif_motion_events_received || 0} active motion · {status.onvif_inactive_motion_events || 0} inactive motion · {status.onvif_renewals || 0} subscription renewals</span> : null}
      {status.motion_qualification ? (
        <div className="motion-runtime-status">
          <div className="motion-runtime-summary">
            <strong>Motion processing</strong>
            <span>{motionModeInfo(status.motion_qualification.mode).status} · {status.motion_qualification.sensitivity} sensitivity · {status.motion_qualification.frame_width || 320}px</span>
            <span>{status.motion_qualification.passed || 0} accepted · {status.motion_qualification.audit_rejected || 0} legacy preview rejects · {status.motion_qualification.suppressed || 0} filtered</span>
            <span>{status.motion_qualification.continuous_frames || 0} visual frames analyzed · {status.motion_qualification.continuous_candidates || 0} accepted analysis frames · {status.motion_qualification.triggers || 0} triggers delivered · {status.motion_qualification.analysis_frames_dropped || 0} stale requests replaced</span>
            <span>Capture-to-analysis p95 {formatMilliseconds(status.motion_qualification.analysis_runtime?.capture_to_analysis_p95_ms)} · preprocessing p99 {formatMilliseconds(status.motion_qualification.analysis_runtime?.preprocess_p99_ms)} · {formatBytes(status.motion_qualification.analysis_runtime?.copy_bytes || 0)} copied for motion analysis</span>
            <span>Light and shadow filtering {status.motion_qualification.illumination_filter_enabled ? "enabled" : "measuring only"} · {status.motion_qualification.illumination_evaluations || 0} evaluated · {status.motion_qualification.illumination_candidates || 0} likely illumination changes · {status.motion_qualification.illumination_filtered || 0} filtered</span>
            <span>{status.motion_qualification.validation_failures || 0} validator errors · {status.motion_qualification.validation_fail_opens || 0} allowed through safely</span>
            <span>{status.motion_qualification.active_followup_triggers || 0} active-event follow-ups · {status.motion_qualification.active_followup_objects || 0} found an object · {status.motion_qualification.active_followup_no_object || 0} found none · {status.motion_qualification.active_followup_episode_limited || 0} held by the episode limit</span>
            {missingCameraTrigger ? <span className="motion-runtime-warning">{visualBackupEnabled ? "ONVIF is disabled, so the conservative visual backup is the only automatic trigger. Restore ONVIF for primary coverage." : "ONVIF is disabled. Camera-triggered mode has no automatic trigger source; only manual tests can run object detection."}</span> : null}
            {missingMotionNotices ? <span className="motion-runtime-warning">{visualBackupEnabled ? "No recognized ONVIF motion notices since this worker started. Strong persistent visual motion can still invoke the backup detector path." : "No recognized ONVIF motion notices since this worker started. In this mode, visual analysis alone cannot create an incident."}</span> : null}
            {visualBackupEnabled ? <>
              <span>{status.motion_qualification.visual_backup?.scene_ready ? "EMA background ready" : "EMA learning scene"} · {status.motion_qualification.visual_backup_triggers || 0} visual backups · {status.motion_qualification.visual_backup_onvif_matches || 0} strong candidates matched to camera notices · {status.motion_qualification.visual_backup_rate_limited || 0} limited</span>
              <span>{status.motion_qualification.visual_backup_not_ready || 0} strong candidates held during scene learning · {status.motion_qualification.visual_backup_uncorrelated_objects || 0} detected objects outside motion areas rejected</span>
            </> : null}
            <MotionEffectiveness cameraId={status.id} mode={status.motion_qualification.mode} />
          </div>
          <div className="motion-pipeline-runtime-grid">
            <MotionPipelineRuntimeCard label="Motion analysis" pipeline={status.motion_qualification.pipeline} origin={status.motion_qualification.pipeline_origins?.qualification} motionCatalog={motionCatalog} />
            <MotionPipelineRuntimeCard label="Extra sources" pipeline={status.motion_qualification.observation_pipeline} origin={status.motion_qualification.pipeline_origins?.observation} motionCatalog={motionCatalog} />
            <MotionPipelineRuntimeCard label="Decision" pipeline={status.motion_qualification.fusion_pipeline} origin={status.motion_qualification.pipeline_origins?.fusion} motionCatalog={motionCatalog} />
          </div>
          <div className="motion-evidence-runtime">
            {Object.entries(status.motion_qualification.evidence_sources || {}).map(([source, evidence]) => (
              <span key={source} className={evidence.enabled ? "enabled" : "disabled"}>
                <strong>{source === "onvif" ? "Camera signal" : source}</strong>
                {evidence.enabled ? `${evidence.sample_count || 0} samples${evidence.last?.score != null ? ` · ${Math.round(Number(evidence.last.score) * 100)}% last confidence` : ""}` : "Disabled"}
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function FaceReviewDialog({ observation, people, timeZone, onClose, onUpdated }) {
  const [newName, setNewName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setNewName("");
    setError("");
  }, [observation.id]);

  useEffect(() => {
    function closeOnEscape(event) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  async function assignPerson(nextPersonId) {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`/api/faces/observations/${observation.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ person_id: nextPersonId ? Number(nextPersonId) : null }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Could not update this face");
      }
      await onUpdated?.("", { advance: true, observationId: observation.id });
    } catch (requestError) {
      setError(requestError.message || "Could not update this face");
    } finally {
      setBusy(false);
    }
  }

  async function createPerson() {
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/faces/people", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, observation_id: observation.id }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Could not create this person");
      }
      await onUpdated?.(`${name} enrolled`, { advance: true, observationId: observation.id });
    } catch (requestError) {
      setError(requestError.message || "Could not create this person");
    } finally {
      setBusy(false);
    }
  }

  async function updateReference(pinned) {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`/api/faces/observations/${observation.id}/reference`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pinned }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Could not update this reference");
      }
      await onUpdated?.(pinned ? "Reference pinned" : "Reference unpinned", { advance: false });
    } catch (requestError) {
      setError(requestError.message || "Could not update this reference");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="face-review-dialog" role="dialog" aria-modal="true" aria-label="Review face">
        <button type="button" className="overlay-close" onClick={onClose} aria-label="Close"><X size={22} /></button>
        <img src={appUrl(`/api/faces/observations/${observation.id}/crop.jpg?padding=0.45`)} alt="Selected face" />
        <div className="face-review-form">
          <div><strong>{observation.person_name || "Unknown face"}</strong><span>{observation.camera_id} · {formatDateTime(observation.observed_at, timeZone)}</span></div>
          <div className="face-match-summary">
            <span>Face quality <strong>{observation.quality_score != null ? `${Math.round(Number(observation.quality_score) * 100)}%` : "Not scored"}</strong></span>
            {Number(observation.consensus?.candidate_count || 0) > 1 ? <span>Selected from <strong>{observation.consensus.candidate_count} incident frames</strong>{Number(observation.consensus?.agreement_count || 0) > 1 ? ` · ${observation.consensus.agreement_count} agreed on identity` : ""}</span> : null}
            {observation.match_details?.reference_ids?.length ? <span>Match supported by <strong>{observation.match_details.reference_ids.length} strongest references</strong></span> : null}
            {observation.match_details?.margin != null ? <span>Lead over next person <strong>{Math.round(Number(observation.match_details.margin) * 100)} points</strong></span> : null}
          </div>
          {observation.candidate_person_id ? <div className="face-enroll-row"><button type="button" disabled={busy} onClick={() => assignPerson(observation.candidate_person_id)}><ScanFace size={16} /> Confirm {observation.candidate_person_name} ({Math.round(Number(observation.candidate_confidence || 0) * 100)}%)</button><button type="button" className="subtle" disabled={busy} onClick={() => assignPerson(null)}><X size={16} /> Reject</button></div> : null}
          {observation.auto_identified && observation.person_id ? <div className="face-enroll-row"><button type="button" disabled={busy} onClick={() => assignPerson(observation.person_id)}><ShieldCheck size={16} /> Confirm automatic match</button></div> : null}
          <label>Assign to person<select value={observation.person_id || ""} disabled={busy} onChange={(event) => assignPerson(event.target.value)}><option value="">Unknown</option>{people.map((person) => <option value={person.id} key={person.id}>{person.name}</option>)}</select></label>
          {observation.person_id && observation.review_status === "confirmed" ? <button type="button" className="subtle" disabled={busy} onClick={() => updateReference(!observation.reference_pinned)}><ShieldCheck size={16} /> {observation.reference_pinned ? "Unpin reference" : "Pin as reference"}</button> : null}
          <div className="face-enroll-row"><input value={newName} disabled={busy} onChange={(event) => setNewName(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") createPerson(); }} placeholder="New person name" /><button type="button" onClick={createPerson} disabled={busy || !newName.trim()}><UserPlus size={16} /> Enroll</button></div>
          {error ? <span className="save-status error">{error}</span> : null}
        </div>
      </section>
    </div>
  );
}

function FacesPage({ timeZone, onAssistantContextChange }) {
  const [people, setPeople] = useState([]);
  const [observations, setObservations] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [status, setStatus] = useState(null);
  const [calibration, setCalibration] = useState(null);
  const [calibrating, setCalibrating] = useState(false);
  const [filter, setFilter] = useState("unknown");
  const [cameraId, setCameraId] = useState("");
  const [personId, setPersonId] = useState("");
  const [selected, setSelected] = useState(null);
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [totalObservations, setTotalObservations] = useState(0);
  const faceLoadSequence = useRef(0);
  const pageSize = isMobileViewport() ? 24 : 48;
  const pageCount = Math.max(1, Math.ceil(totalObservations / pageSize));

  useEffect(() => {
    onAssistantContextChange?.({
      page: "faces",
      camera_id: selected?.camera_id || cameraId,
      incident_event_id: Number(selected?.event_id) || null,
      filters: { status: filter, person_id: personId },
    });
  }, [cameraId, filter, onAssistantContextChange, personId, selected?.camera_id, selected?.event_id]);

  async function load() {
    const sequence = ++faceLoadSequence.current;
    setLoading(true);
    try {
      const query = new URLSearchParams({ status: personId ? "all" : filter, limit: String(pageSize), offset: String(page * pageSize) });
      if (cameraId) query.set("camera_id", cameraId);
      if (personId) query.set("person_id", personId);
      const countQuery = new URLSearchParams(query);
      countQuery.delete("limit");
      countQuery.delete("offset");
      const [peopleResponse, observationResponse, countResponse] = await Promise.all([
        fetch("/api/faces/people"),
        fetch(`/api/faces/observations?${query}`),
        fetch(`/api/faces/observations/count?${countQuery}`),
      ]);
      if (!peopleResponse.ok || !observationResponse.ok) throw new Error("Unable to load the face database");
      const [peoplePayload, observationPayload, countPayload] = await Promise.all([
        peopleResponse.json(),
        observationResponse.json(),
        countResponse.ok ? countResponse.json() : null,
      ]);
      if (sequence !== faceLoadSequence.current) return;
      setPeople(peoplePayload);
      setObservations(observationPayload);
      if (countPayload) setTotalObservations(Number(countPayload.total || 0));
      setNotice("");
      void Promise.all([fetch("/api/cameras"), fetch("/api/faces/status")])
        .then(async ([cameraResponse, statusResponse]) => {
          const [cameraPayload, statusPayload] = await Promise.all([
            cameraResponse.ok ? cameraResponse.json() : null,
            statusResponse.ok ? statusResponse.json() : null,
          ]);
          if (sequence !== faceLoadSequence.current) return;
          if (cameraPayload) setCameras(cameraPayload);
          if (statusPayload) setStatus(statusPayload);
        })
        .catch(() => {});
      return observationPayload;
    } catch (error) {
      if (sequence === faceLoadSequence.current) setNotice(error.message || "Unable to load faces");
      return null;
    } finally {
      if (sequence === faceLoadSequence.current) setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    return () => { faceLoadSequence.current += 1; };
  }, [filter, cameraId, personId, page]);
  useEffect(() => { setPage(0); }, [filter, cameraId, personId]);
  useEffect(() => { if (page >= pageCount) setPage(Math.max(0, pageCount - 1)); }, [page, pageCount]);

  async function deletePerson(person) {
    if (!window.confirm(`Delete ${person.name}? Their observations will return to Unknown.`)) return;
    try {
      const response = await fetch(`/api/faces/people/${person.id}`, { method: "DELETE" });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        return setNotice(payload.detail || "Could not delete this person");
      }
      setPersonId("");
      await load();
    } catch (error) {
      setNotice(error.message || "Could not delete this person");
    }
  }

  async function analyzeCalibration() {
    setCalibrating(true);
    try {
      const response = await fetch("/api/faces/calibration");
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not analyze face matching");
      setCalibration(payload);
    } catch (error) {
      setNotice(error.message || "Could not analyze face matching");
    } finally {
      setCalibrating(false);
    }
  }

  return (
    <main className="faces-page">
      <aside className="faces-people-panel">
        <div className="faces-panel-heading">
          <div><h2>People</h2><span>{people.length} enrolled</span></div>
          <Users size={20} />
        </div>
        <button type="button" className={`face-person-row ${personId === "" ? "active" : ""}`} onClick={() => { setPersonId(""); setPage(0); }}>
          <span className="face-avatar unknown"><ScanFace size={22} /></span>
          <span><strong>All faces</strong><small>{status?.observations || 0} observations</small></span>
        </button>
        <div className="face-person-list">
          {people.map((person) => (
            <div className={`face-person-row ${String(person.id) === personId ? "active" : ""}`} key={person.id}>
              <button type="button" className="face-person-select" onClick={() => { setPersonId(String(person.id)); setPage(0); }}>
                {person.preview_observation_id
                  ? <img src={appUrl(`/api/faces/observations/${person.preview_observation_id}/crop.jpg`)} alt="" />
                  : <span className="face-avatar unknown"><ScanFace size={20} /></span>}
                <span><strong>{person.name}</strong><small>{person.usable_reference_count || 0}/{person.reference_count || 0} usable references · {person.observation_count} total{person.pinned_reference_count ? ` · ${person.pinned_reference_count} pinned` : ""}</small></span>
              </button>
              <button type="button" className="icon-button subtle" onClick={() => deletePerson(person)} title="Delete person" aria-label={`Delete ${person.name}`}><Trash2 size={15} /></button>
            </div>
          ))}
        </div>
      </aside>

      <section className="faces-review-panel">
        <div className="faces-toolbar">
          <div className="faces-filter-group" role="group" aria-label="Face status">
            {["unknown", "suggested", "known", "pending", "unusable", "all"].map((value) => (
              <button type="button" className={filter === value && !personId ? "active" : ""} key={value} onClick={() => { setPersonId(""); setFilter(value); setPage(0); }}>{value}</button>
            ))}
          </div>
          <select value={cameraId} onChange={(event) => { setCameraId(event.target.value); setPage(0); }} aria-label="Filter by camera">
            <option value="">All cameras</option>
            {cameras.map((camera) => <option value={camera.id} key={camera.id}>{camera.name || camera.id}</option>)}
          </select>
          <span className="shown-bubble">{totalObservations} faces</span>
        </div>
        <div className="face-messages">
          {!status?.recognition_ready || status?.recognition?.pending > 0 || status?.recognition?.failed > 0 ? <div className="face-readiness"><Activity size={16} /><span>{status?.recognition_message || "Automatic recognition is not configured."}</span></div> : null}
          <div className="face-readiness face-calibration">
            <Gauge size={16} />
            <span>{calibration?.message || "Measure your confirmed and rejected faces before changing match thresholds."}</span>
            {calibration?.ready ? <strong>{Math.round(Number(calibration.rank_one_accuracy || 0) * 100)}% identity accuracy · suggest at {Math.round(Number(calibration.recommended?.suggestion_threshold || 0) * 100)}% · automatic at {Math.round(Number(calibration.recommended?.automatic_threshold || 0) * 100)}% with a {Math.round(Number(calibration.recommended?.automatic_margin || 0) * 100)}-point lead</strong> : null}
            <button type="button" className="subtle" disabled={calibrating} onClick={analyzeCalibration}>{calibrating ? "Analyzing..." : "Analyze matching"}</button>
          </div>
          {notice ? <div className="save-status">{notice}</div> : null}
        </div>
        <div className="face-observation-grid">
          {loading ? <div className="empty-state">Loading face observations...</div> : null}
          {!loading && !observations.length ? <div className="empty-state">No faces match these filters.</div> : null}
          {observations.map((observation) => (
            <button type="button" className="face-observation-card" key={observation.id} onClick={() => setSelected(observation)}>
              <img src={appUrl(`/api/faces/observations/${observation.id}/crop.jpg`)} alt={observation.person_name || "Unknown face"} loading="lazy" />
              <span className="face-card-hud">
                <strong>{observation.person_name || (observation.candidate_person_name ? `Suggested: ${observation.candidate_person_name}` : "Unknown")}</strong>
                <small>{observation.camera_id} · {formatDateTime(observation.observed_at, timeZone)}</small>
              </span>
              <span className="face-confidence">{observation.candidate_confidence != null ? `${Math.round(Number(observation.candidate_confidence) * 100)}% match` : `${Math.round(Number(observation.confidence || 0) * 100)}%`}</span>
              {Number(observation.consensus?.candidate_count || 0) > 1 ? <span className="face-frame-count">Best of {observation.consensus.candidate_count} frames</span> : null}
            </button>
          ))}
        </div>
        <div className="faces-pagination">
          <button type="button" onClick={() => setPage((current) => Math.max(0, current - 1))} disabled={page === 0 || loading}><ChevronLeft size={16} /> Previous</button>
          <span>Page {Math.min(page + 1, pageCount)} of {pageCount}</span>
          <button type="button" onClick={() => setPage((current) => Math.min(pageCount - 1, current + 1))} disabled={page >= pageCount - 1 || loading}>Next <ChevronRight size={16} /></button>
        </div>
      </section>

      {selected ? <FaceReviewDialog observation={selected} people={people} timeZone={timeZone} onClose={() => setSelected(null)} onUpdated={async (message, action = {}) => {
        const currentObservations = observations;
        if (message) setNotice(message);
        const refreshed = await load();
        setSelected(action.advance && refreshed
          ? nextFaceReviewObservation(action.observationId || selected.id, currentObservations, refreshed)
          : null);
      }} /> : null}
    </main>
  );
}

function App() {
  const [timeZone, setTimeZone] = useStoredState("survng.timeZone", DEFAULT_TIME_ZONE);
  const [theme, setTheme] = useStoredState("survng.theme", "auto");
  const [recordingContext, setRecordingContext] = useState(null);
  const pathname = appPathname();
  const isExportCenter = pathname.startsWith("/recordings/exports");
  const isSemanticSearch = pathname.startsWith("/recordings/search");
  const page = pathname.startsWith("/config")
    ? "config"
    : pathname.startsWith("/recordings")
      ? "recordings"
      : pathname.startsWith("/incidents")
        ? "incidents"
        : pathname.startsWith("/faces")
          ? "faces"
        : "live";
  const [assistantContext, setAssistantContext] = useState({ page });
  useEffect(() => {
    setAssistantContext({ page });
  }, [page]);
  useEffect(() => {
    document.documentElement.dataset.theme = THEMES.includes(theme) ? theme : "auto";
  }, [theme]);
  return (
    <Shell page={page} theme={theme} recordingContext={recordingContext}>
      {page === "config"
        ? <ConfigPage timeZone={timeZone} setTimeZone={setTimeZone} theme={theme} setTheme={setTheme} onAssistantContextChange={setAssistantContext} />
        : page === "recordings"
          ? isExportCenter
            ? <ExportCenterPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
            : isSemanticSearch
              ? <SemanticSearchPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
            : <RecordingsPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
          : page === "incidents"
            ? <IncidentsPage timeZone={timeZone} onRecordingContextChange={setRecordingContext} onAssistantContextChange={setAssistantContext} />
            : page === "faces"
              ? <FacesPage timeZone={timeZone} onAssistantContextChange={setAssistantContext} />
            : <LivePage timeZone={timeZone} onRecordingContextChange={setRecordingContext} onAssistantContextChange={setAssistantContext} />}
      <AssistantPanel pageContext={{ page, ...assistantContext }} timeZone={timeZone} />
    </Shell>
  );
}

createRoot(document.getElementById("root")).render(<App />);

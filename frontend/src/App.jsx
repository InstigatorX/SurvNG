import React, { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowLeft,
  Camera,
  ChevronLeft,
  ChevronRight,
  Copy,
  CircleDot,
  Clock3,
  Crop,
  Cog,
  Download,
  Cpu,
  Film,
  Gauge,
  Grid2X2,
  GripVertical,
  HardDrive,
  Search,
  ListTree,
  Monitor,
  Moon,
  Play,
  Plus,
  Power,
  Radar,
  Radio,
  RefreshCcw,
  RotateCcw,
  Save,
  ScanFace,
  Sparkles,
  Siren,
  SkipBack,
  SkipForward,
  Sun,
  Trash2,
  Undo2,
  UserPlus,
  Users,
  Rows3,
  Video,
  X,
} from "lucide-react";
import "./styles.css";

const DEFAULT_TIME_ZONE = "America/New_York";
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
  onReady,
  onError,
  ...videoProps
}, forwardedRef) {
  const videoRef = useRef(null);
  const [runtime, setRuntime] = useState(null);
  const callbacksRef = useRef({ onReady, onError });
  useImperativeHandle(forwardedRef, () => videoRef.current);

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

  return <video ref={videoRef} muted={muted} {...videoProps} />;
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
      onReady={(_player, video) => onReady?.(video, "recording")}
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
  onReady,
}, forwardedRef) {
  const videoRef = useRef(null);
  const [stage, setStage] = useState("webrtc");
  const [compatibility, setCompatibility] = useState("detect");
  const [snapshotToken, setSnapshotToken] = useState(() => Date.now());
  useImperativeHandle(forwardedRef, () => videoRef.current);

  useEffect(() => {
    setStage("webrtc");
    setCompatibility("detect");
  }, [cameraId, source]);

  useEffect(() => {
    if (stage !== "webrtc" || compatibility !== "detect") return undefined;
    const controller = new AbortController();
    fetch(`/api/cameras/${encodeURIComponent(cameraId)}/live-info?source=${encodeURIComponent(source)}`, {
      signal: controller.signal,
    }).then((response) => response.ok ? response.json() : null).then((info) => {
      if (!controller.signal.aborted) setCompatibility(info?.compatibility === "h264" ? "h264" : "native");
    }).catch(() => {
      if (!controller.signal.aborted) setCompatibility("native");
    });
    return () => controller.abort();
  }, [cameraId, source, stage, compatibility]);

  useEffect(() => {
    if (stage !== "webrtc" || compatibility === "detect") return undefined;
    let disposed = false;
    let connected = false;
    let socket = null;
    let peer = null;
    let disconnectTimer = null;
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const fallback = () => {
      if (!disposed) setStage("mse");
    };
    const failTimer = window.setTimeout(() => !connected && fallback(), 3000);

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
        if (socket?.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: "webrtc/candidate", value: event.candidate?.candidate || "" }));
        }
      });
      peer.addEventListener("connectionstatechange", () => {
        if (peer.connectionState === "connected") {
          connected = true;
          window.clearTimeout(failTimer);
          window.clearTimeout(disconnectTimer);
          disconnectTimer = null;
        } else if (peer.connectionState === "failed" && !disposed) {
          fallback();
        } else if (peer.connectionState === "disconnected" && !disposed && !disconnectTimer) {
          disconnectTimer = window.setTimeout(fallback, 1000);
        }
      });
      socket = new WebSocket(`${protocol}//${location.host}${appUrl(`/api/cameras/${encodeURIComponent(cameraId)}/webrtc?source=${encodeURIComponent(source)}&compat=${compatibility}`)}`);
      socket.addEventListener("open", async () => {
        const offer = await peer.createOffer();
        await peer.setLocalDescription(offer);
        socket.send(JSON.stringify({ type: "webrtc/offer", value: offer.sdp }));
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
          peer.setRemoteDescription({ type: "answer", sdp: message.value }).catch(fallback);
        } else if (message.type === "webrtc/candidate" && message.value) {
          peer.addIceCandidate({ candidate: message.value, sdpMid: "0" }).catch(() => {});
        } else if (message.type === "error" && String(message.value).includes("webrtc")) {
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
      if (videoRef.current) videoRef.current.srcObject = null;
    };
  }, [cameraId, source, stage, compatibility]);

  useEffect(() => {
    if (stage !== "mse") return undefined;
    const MediaSourceApi = window.ManagedMediaSource || window.MediaSource;
    if (!MediaSourceApi) {
      setStage("mjpeg");
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
      if (!disposed) setStage("mjpeg");
    };
    const failTimer = window.setTimeout(() => !ready && fallback(), 12_000);
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
      try {
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
      video.addEventListener("loadeddata", () => {
        ready = true;
        window.clearTimeout(failTimer);
      }, { once: true });

      socket = new WebSocket(`${protocol}//${location.host}${appUrl(`/api/cameras/${encodeURIComponent(cameraId)}/mse?source=${encodeURIComponent(source)}&compat=${compatibility}`)}`);
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
      socket?.close();
      queue.length = 0;
      if (video) {
        video.pause();
        video.removeAttribute("src");
        video.srcObject = null;
        video.load();
      }
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [cameraId, source, stage, compatibility]);

  useEffect(() => {
    if (stage === "webrtc") return undefined;
    const timer = window.setTimeout(() => {
      setCompatibility("detect");
      setStage("webrtc");
      setSnapshotToken(Date.now());
    }, 60_000);
    return () => window.clearTimeout(timer);
  }, [stage]);

  useEffect(() => {
    if (stage !== "snapshot") return undefined;
    const timer = window.setInterval(() => setSnapshotToken(Date.now()), 2000);
    return () => window.clearInterval(timer);
  }, [stage]);

  return (
    <div className="live-stack" data-stage={stage}>
      <img
        className="live-poster"
        src={appUrl(stage === "mjpeg"
          ? `/api/cameras/${cameraId}/stream.mjpg?source=${source}&fps=1&t=${snapshotToken}`
          : `/api/cameras/${cameraId}/snapshot.jpg?source=${source === "main" ? "live" : source}&t=${snapshotToken}`)}
        alt=""
        onLoad={(event) => ["mjpeg", "snapshot"].includes(stage) && onReady?.(event.currentTarget, "snapshot")}
      />
      {["webrtc", "mse"].includes(stage) ? (
        <video
          ref={videoRef}
          className="live-video"
          muted={muted}
          controls={controls}
          autoPlay
          playsInline
          disableRemotePlayback
          onLoadedData={(event) => {
            event.currentTarget.play().catch(() => {});
            onReady?.(event.currentTarget, stage);
          }}
        />
      ) : null}
      {stage === "recording" ? (
        <RecordingFallback
          cameraId={cameraId}
          source={source}
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
  const eventId = Number(event?.representative_event_id || event?.id);
  return Number.isFinite(eventId) ? appUrl(`/api/events/${eventId}/snapshot.jpg`) : "";
}

function useStoredState(key, initialValue) {
  const [value, setValue] = useState(() => localStorage.getItem(key) || initialValue);
  useEffect(() => {
    localStorage.setItem(key, value);
  }, [key, value]);
  return [value, setValue];
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

function formatClock(seconds) {
  if (!Number.isFinite(seconds)) return "00:00";
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function formatDayClock(seconds) {
  const whole = Math.max(0, Math.min(24 * 60 * 60, Math.floor(Number(seconds) || 0)));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
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
  return {
    ...camera,
    video_backend: isReolink ? "baichuan_native" : isRtsp ? "url" : camera.video_backend,
    onvif: {
      ...camera.onvif,
      host: camera.onvif?.host || defaults.host,
      username: camera.onvif?.username || defaults.username,
      password: camera.onvif?.password || defaults.password,
    },
    baichuan: {
      ...camera.baichuan,
      enabled: isReolink,
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
    stream_url: seed.stream_url || "",
    live_stream_url: seed.live_stream_url || "",
    record: seed.record ?? true,
    record_sub: seed.record_sub ?? false,
    motion_qualification: {
      mode: seed.motion_qualification?.mode || "inherit",
      sensitivity: seed.motion_qualification?.sensitivity || "inherit",
      frame_width: seed.motion_qualification?.frame_width ?? null,
      borderline_rescue_enabled: seed.motion_qualification?.borderline_rescue_enabled ?? null,
      borderline_margin: seed.motion_qualification?.borderline_margin ?? null,
      mog2_audit_enabled: seed.motion_qualification?.mog2_audit_enabled ?? null,
    },
    zones: structuredClone(seed.zones || []),
    onvif: {
      enabled: seed.onvif?.enabled || false,
      host,
      port: seed.onvif?.port || 8000,
      username: seed.onvif?.username || "",
      password: seed.onvif?.password || "",
    },
    baichuan: {
      enabled: seed.baichuan?.enabled || false,
      host,
      port: seed.baichuan?.port || 9000,
      username: seed.baichuan?.username || "",
      password: seed.baichuan?.password || "",
      channel: seed.baichuan?.channel || 0,
    },
  };
}

function Shell({ page, theme, recordingContext, children }) {
  const isRecordings = page === "recordings";
  const isConfig = page === "config";
  const isIncidents = page === "incidents";
  const isFaces = page === "faces";
  return (
    <div className={`app-shell page-${page}`}>
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark">
            <img src={appUrl("/static/favicon.svg")} alt="" aria-hidden="true" />
          </div>
          <div>
            <h1>{isConfig ? "Config" : isRecordings ? "Recordings" : isIncidents ? "Incidents" : isFaces ? "Faces" : "SurvNG"}</h1>
            <p>{isConfig ? "Camera inventory, cloning, and capability detection" : isRecordings ? "Continuous review of saved camera history" : isIncidents ? "Motion and object incident review" : isFaces ? "Face enrollment and observation review" : "Streams, events, recordings, and object detections"}</p>
          </div>
          {!isConfig && !isRecordings && !isIncidents && !isFaces ? <LiveHeaderStats /> : null}
        </div>
        <div className="top-actions">
          <nav className="topnav" aria-label="Primary">
            <a className="nav-button" href={appUrl("/")}><Video size={16} /> Live</a>
            <a className="nav-button incidents-nav" href={appUrl("/incidents")}><Siren size={16} /> Incidents</a>
            <a className="nav-button" href={appUrl("/faces")}><ScanFace size={16} /> Faces</a>
            <a className="nav-button" href={recordingsHref(recordingContext)}><Film size={16} /> Recordings</a>
            <a className="nav-button" href={appUrl("/config")}><Cog size={16} /> Config</a>
          </nav>
        </div>
      </header>
      {children}
    </div>
  );
}

const APP_EVENT_TYPES = ["camera_state", "cameras_state", "motion", "object", "incident", "system_state"];
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
    motion: "--",
    objects: "--",
    storage: null,
    detector: null,
    cameras: null,
  });
  const eventRefreshTimer = useRef(null);

  async function loadRecentEvents() {
    const eventResponse = await fetch("/api/events?limit=50");
    const events = await eventResponse.json();
    setStats((current) => ({
      ...current,
      motion: events.filter((event) => event.kind === "motion").length,
      objects: events.filter(hasDetectedObjects).length,
    }));
  }

  async function loadSystem() {
    const systemResponse = await fetch("/api/system/status");
    const system = systemResponse.ok ? await systemResponse.json() : {};
    setStats((current) => ({
      ...current,
      storage: system.storage || null,
      detector: system.detector || null,
      cameras: system.cameras || null,
    }));
  }

  useAppEvents(({ type, data }) => {
    if (type === "system_state") {
      setStats((current) => ({
        ...current,
        storage: data.storage || null,
        detector: data.detector || null,
        cameras: data.cameras || null,
      }));
    } else if (type === "incident") {
      window.clearTimeout(eventRefreshTimer.current);
      eventRefreshTimer.current = window.setTimeout(loadRecentEvents, 250);
    }
  });

  useEffect(() => {
    Promise.all([loadRecentEvents(), loadSystem()]);
    const timer = window.setInterval(() => Promise.all([loadRecentEvents(), loadSystem()]), 60_000);
    return () => {
      window.clearInterval(timer);
      window.clearTimeout(eventRefreshTimer.current);
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
  const detectorLoaded = detector.coreml_loaded || detector.openvino_loaded || detector.opencv_loaded;
  const detectorLabel = detector.enabled
    ? `${detector.loaded_backend || detector.configured_backend || "det"}${detector.loaded_device ? ` ${detector.loaded_device}` : ""}`
    : "off";
  const storageLabel = stats.storage ? `${formatBytes(stats.storage.free_bytes)} free` : "--";
  const cameraLabel = stats.cameras ? `${stats.cameras.recording}/${stats.cameras.total} rec` : "--";

  return (
    <div className="header-stats" aria-label="System and recent event summary">
      <span className="header-stat warn"><Siren size={15} /><small>Motion</small><strong>{stats.motion}</strong></span>
      <span className="header-stat hot"><Activity size={15} /><small>Objects</small><strong>{stats.objects}</strong></span>
      <span className="header-stat"><HardDrive size={15} /><small>Storage</small><strong>{storageLabel}</strong></span>
      <span className={`header-stat ${detectorLoaded ? "ok" : "warn"}`}><Gauge size={15} /><small>Detect</small><strong>{detectorLabel}</strong></span>
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
          <span className="infer-tooltip-foot">1 stream · mmap {detector.mmap_enabled ? "on" : "off"} · cache {detector.cache_enabled ? "on" : "off"} · warm-up {formatMilliseconds(detector.warmup_ms)}</span>
          <span className="infer-tooltip-foot">object {objectWorker.worker_alive ? `#${objectWorker.worker_pid}` : "offline"} · {objectWorker.configured_device || detector.configured_device || "device"} · gen {objectWorker.generation ?? "--"} · restarts {objectWorker.restart_count ?? 0}{objectWorker.fallback_active ? " · CPU fallback" : ""}</span>
          <span className="infer-tooltip-foot">face {faceWorker.enabled ? (faceWorker.worker_alive ? `#${faceWorker.worker_pid}` : "offline") : "disabled"} · {faceWorker.configured_device || "AUTO"} · gen {faceWorker.generation ?? "--"} · restarts {faceWorker.restart_count ?? 0}{faceWorker.fallback_active ? " · CPU fallback" : ""}</span>
        </span>
      </span>
      <span className={runtime.queue_depth > 0 ? "header-stat warn" : "header-stat"}><ListTree size={15} /><small>Queue</small><strong>{Number.isFinite(runtime.queue_depth) ? runtime.queue_depth : "--"}</strong></span>
      <span className="header-stat hot"><Clock3 size={15} /><small>Last Hit</small><strong>{formatAge(runtime.last_detection_age_seconds)}</strong></span>
      <span className="header-stat"><Camera size={15} /><small>Cameras</small><strong>{cameraLabel}</strong></span>
    </div>
  );
}

function StatCard({ icon, label, value, tone = "default" }) {
  return (
    <section className={`bento-card stat-card ${tone}`}>
      <div className="icon-badge">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </section>
  );
}

function usePollingData(includeIncidents = true) {
  const [cameras, setCameras] = useState([]);
  const [incidents, setIncidents] = useState([]);
  const [appConfig, setAppConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const incidentRefreshTimer = useRef(null);

  async function loadIncidents() {
    if (!includeIncidents) return;
    const response = await fetch("/api/incidents?limit=120&gap_seconds=45");
    if (response.ok) setIncidents(await response.json());
  }

  async function load() {
    const [cameraResponse, incidentResponse, configResponse] = await Promise.all([
      fetch("/api/cameras"),
      includeIncidents ? fetch("/api/incidents?limit=120&gap_seconds=45") : Promise.resolve(null),
      fetch("/api/config"),
    ]);
    setCameras(await cameraResponse.json());
    if (incidentResponse?.ok) setIncidents(await incidentResponse.json());
    if (configResponse.ok) setAppConfig(await configResponse.json());
    setLoading(false);
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
    } else if (type === "incident" && includeIncidents) {
      window.clearTimeout(incidentRefreshTimer.current);
      incidentRefreshTimer.current = window.setTimeout(loadIncidents, 250);
    }
  });

  useEffect(() => {
    load();
    const timer = window.setInterval(load, 60_000);
    return () => {
      window.clearInterval(timer);
      window.clearTimeout(incidentRefreshTimer.current);
    };
  }, [includeIncidents]);

  return { cameras, incidents, appConfig, loading, refresh: load };
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
  if (!width || !height) return "16 / 9";
  return `${width} / ${height}`;
}

function CameraTile({ camera, timeZone, refresh, onOpen, startDelayMs = 0, dropProps = {}, dragHandleProps = {}, dragging = false, dragOver = false }) {
  const [streamMode, setStreamMode] = useStoredState(`survng.streamMode.v3.${camera.id}`, "motion");
  const [sourceMode, setSourceMode] = useStoredState(`survng.sourceMode.${camera.id}`, "live");
  const normalizedStreamMode = STREAM_MODES.includes(streamMode) ? streamMode : "motion";
  const lastMotionMs = new Date(camera.last_motion_at || 0).getTime();
  const motionActive = Number.isFinite(lastMotionMs) && Date.now() - lastMotionMs <= MOTION_WEBRTC_HOLD_MS;
  const activeTransport = normalizedStreamMode === "motion" ? (motionActive ? "webrtc" : "snapshot") : normalizedStreamMode;
  const [aspect, setAspect] = useState("16 / 9");
  const [mjpegToken, setMjpegToken] = useState(() => String(Date.now()));
  const [snapshotToken, setSnapshotToken] = useState(() => String(Date.now()));
  const [streamReady, setStreamReady] = useState(false);
  const [recordingBusy, setRecordingBusy] = useState(false);
  const [recordingError, setRecordingError] = useState("");
  const [detectionBusy, setDetectionBusy] = useState(false);
  const [detectionError, setDetectionError] = useState("");
  const shouldUseWebRtc = camera.running && streamReady && activeTransport === "webrtc";
  const shouldUseMjpegStream = camera.running && streamReady && activeTransport === "mjpeg";
  const cameraConnected = camera.connected ?? camera.running;

  useEffect(() => {
    if (!STREAM_MODES.includes(streamMode)) setStreamMode("motion");
  }, [streamMode, setStreamMode]);

  useEffect(() => {
    setMjpegToken(String(Date.now()));
    setSnapshotToken(String(Date.now()));
    setStreamReady(false);
  }, [camera.id, sourceMode, activeTransport]);

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
    await fetch(`/api/cameras/${camera.id}/${action}`, { method: "POST" });
    refresh();
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
    setSourceMode(sourceMode === "main" ? "live" : "main");
  }

  const posterSource = activeTransport === "webrtc" && sourceMode === "main" ? "live" : sourceMode;
  const imageUrl = appUrl(shouldUseMjpegStream
    ? `/api/cameras/${camera.id}/stream.mjpg?source=${sourceMode}&t=${mjpegToken}`
    : `/api/cameras/${camera.id}/snapshot.jpg?source=${posterSource}&t=${snapshotToken}`);

  return (
    <article className={`bento-card camera-tile ${dragging ? "dragging" : ""} ${dragOver ? "drag-over" : ""}`} {...dropProps}>
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
        ) : shouldUseWebRtc ? (
          <WebRtcLive
            cameraId={camera.id}
            source={sourceMode}
            timeZone={timeZone}
            muted
            onReady={(media) => setAspect(mediaAspect(media))}
          />
        ) : (
          <img
            src={imageUrl}
            alt={`${camera.name} ${sourceMode === "main" ? "main" : "sub"} live stream`}
            onLoad={(event) => setAspect(mediaAspect(event.currentTarget))}
          />
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
            <button
              type="button"
              className="tile-control-button icon-only camera-drag-handle"
              title="Drag to reorder camera"
              aria-label={`Reorder ${camera.name}`}
              {...dragHandleProps}
            >
              <GripVertical size={16} />
            </button>
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
              className={`tile-control-button icon-only ${camera.running ? "danger" : ""}`}
              onClick={() => post(camera.running ? "camera/stop" : "camera/start")}
              title={camera.running ? "Stop camera" : "Start camera"}
              aria-label={`${camera.running ? "Stop" : "Start"} ${camera.name}`}
            >
              <Power size={16} />
            </button>
          </div>
        </div>
      </div>
    </article>
  );
}

function LiveCameraOverlay({ camera, timeZone, onClose }) {
  const mediaRef = useRef(null);
  const [aspect, setAspect] = useState("16 / 9");
  const [source, setSource] = useStoredState(`survng.liveOverlaySource.${camera.id}`, preferredStreamSource());
  const [showControls, setShowControls] = useState(false);
  const [mediaReady, setMediaReady] = useState(false);
  const activeSource = source === "main" ? "main" : "live";

  useEffect(() => {
    setShowControls(false);
    setMediaReady(false);
  }, [camera.id, activeSource]);

  useEffect(() => {
    function onKey(event) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  function updateMediaAspect() {
    setAspect(mediaAspect(mediaRef.current));
  }

  return (
    <div className="live-overlay" role="dialog" aria-modal="true" aria-label={`${camera.name} full live view`}>
      <button className="live-overlay-backdrop" type="button" onClick={onClose} aria-label="Close live view" />
      <section className="live-overlay-panel" style={{ "--media-aspect": aspect }}>
        <div className="live-overlay-head">
          <div>
            <h2>{camera.name}</h2>
            <span>{sourceLabel(activeSource)} stream</span>
          </div>
          <button type="button" className="tile-control-button" onClick={() => setSource(activeSource === "main" ? "live" : "main")} aria-label="Switch live stream">
            <Radio size={15} /> {sourceLabel(activeSource)}
          </button>
          <button type="button" className="tile-control-button icon-only" onClick={onClose} aria-label="Close live view">
            <X size={18} />
          </button>
        </div>
        <div
          className="live-overlay-media"
          onClick={() => {
            if (!showControls) setShowControls(true);
          }}
        >
          {!mediaReady ? (
            <div className="live-media-status" role="status" aria-live="polite">
              <RefreshCcw className="spin" size={20} />
              <span>Connecting live stream...</span>
            </div>
          ) : null}
          <WebRtcLive
            ref={mediaRef}
            cameraId={camera.id}
            source={activeSource}
            timeZone={timeZone}
            muted
            controls={showControls}
            onReady={(media) => {
              setAspect(mediaAspect(media));
              setMediaReady(true);
            }}
          />
        </div>
      </section>
    </div>
  );
}

function Segmented({ value, onChange, options }) {
  return (
    <div className="segmented">
      {options.map(([id, label]) => (
        <button key={id} className={value === id ? "active" : ""} onClick={() => onChange(id)}>{label}</button>
      ))}
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
    .filter(({ object, box }) => object?.label && (!incidentEligibleOnly || object.incident_eligible !== false) && box && [box.x1, box.y1, box.x2, box.y2].every((value) => Number.isFinite(Number(value))))
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

function SnapshotImage({ event, alt, iconSize = 24, className = "", layerStyle = null, allowObjectFocus = true, showAnnotations = true, incidentEligibleOnly = false, onImageSize, children }) {
  const boxes = objectBoxes(event, incidentEligibleOnly);
  const frameRef = useRef(null);
  const [imageSize, setImageSize] = useState(null);
  const [frameSize, setFrameSize] = useState(null);
  const [objectFocused, setObjectFocused] = useState(false);
  const renderedImage = useMemo(() => {
    if (!imageSize?.width || !imageSize?.height || !frameSize?.width || !frameSize?.height) return null;
    const scale = Math.min(frameSize.width / imageSize.width, frameSize.height / imageSize.height);
    const width = imageSize.width * scale;
    const height = imageSize.height * scale;
    return {
      x: (frameSize.width - width) / 2,
      y: (frameSize.height - height) / 2,
      width,
      height,
      scale,
    };
  }, [frameSize, imageSize]);
  const canFocus = allowObjectFocus && showAnnotations && boxes.length > 0 && renderedImage;

  useEffect(() => {
    setObjectFocused(false);
    setImageSize(null);
  }, [event?.id, event?.snapshot_path]);

  useEffect(() => {
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

  function onImageLoad(loadEvent) {
    const image = loadEvent.currentTarget;
    if (image.naturalWidth && image.naturalHeight) {
      const size = { width: image.naturalWidth, height: image.naturalHeight };
      setImageSize(size);
      onImageSize?.(size);
    }
  }

  const renderedBoxes = useMemo(() => {
    if (!renderedImage || !frameSize) return [];
    return boxes.map((box) => ({
      ...box,
      left: renderedImage.x + box.x1 * renderedImage.scale,
      top: renderedImage.y + box.y1 * renderedImage.scale,
      width: (box.x2 - box.x1) * renderedImage.scale,
      height: (box.y2 - box.y1) * renderedImage.scale,
      maskPoints: box.maskPolygon.map(([x, y]) => `${renderedImage.x + x * renderedImage.scale},${renderedImage.y + y * renderedImage.scale}`).join(" "),
    })).filter((box) => box.width > 0 && box.height > 0);
  }, [boxes, frameSize, renderedImage]);

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

  const activeLayerStyle = objectFocused && focusStyle ? focusStyle : layerStyle;
  const aspect = imageSize ? `${imageSize.width} / ${imageSize.height}` : undefined;

  return (
    <div ref={frameRef} className={`snapshot-frame ${objectFocused ? "object-focused" : ""} ${className}`} style={aspect ? { "--snapshot-aspect": aspect } : undefined}>
      <div className="snapshot-layer" style={activeLayerStyle || undefined}>
        {event?.snapshot_path && eventSnapshotUrl(event) ? <img src={eventSnapshotUrl(event)} alt={alt} onLoad={onImageLoad} /> : <div className="empty-thumb"><Camera size={iconSize} /></div>}
        {showAnnotations && renderedBoxes.length ? (
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
      </div>
      {canFocus ? (
        <button
          type="button"
          className={`snapshot-focus-button ${objectFocused ? "active" : ""}`}
          onClick={(event) => {
            event.stopPropagation();
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


function IncidentClipLayer({ event, active, onEnded }) {
  const videoRef = useRef(null);
  const [clipInfo, setClipInfo] = useState(null);
  const [clipLoading, setClipLoading] = useState(false);
  const [clipError, setClipError] = useState("");
  const [playback, setPlayback] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function loadClipSettings() {
      const eventId = Number(event?.representative_event_id || event?.id);
      if (!active || !Number.isFinite(eventId)) {
        setClipInfo(null);
        setPlayback(null);
        setClipLoading(false);
        setClipError(active ? "No event video available" : "");
        return;
      }
      setClipInfo(null);
      setPlayback(null);
      setClipLoading(true);
      setClipError("");
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
        // Defaults keep inline playback useful if settings are temporarily unavailable.
      }
      if (cancelled) return;
      const safeBefore = Number.isFinite(before) ? before : 5;
      const safeAfter = Number.isFinite(after) ? after : 5;
      const window = incidentClipWindow(event, safeBefore, safeAfter);
      const info = {
        streamUrl: eventStreamUrl(eventId, window.before, window.after),
        downloadUrl: eventClipUrl(eventId, window.before, window.after),
      };
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
            onReady={() => { setClipLoading(false); setClipError(""); }}
            onError={() => {
              if (playback.url !== clipInfo.downloadUrl) {
                setClipLoading(true);
                setPlayback({ url: clipInfo.downloadUrl, mimeType: "video/mp4" });
              } else {
                setClipLoading(false);
                setClipError("No recording window found");
              }
            }}
            onEnded={onEnded}
          />
          {clipLoading ? <div className="incident-video-status preparing">Preparing incident video...</div> : null}
        </>
      ) : (
        <div className="incident-video-status">{clipLoading ? "Preparing video..." : clipError || "No event video available"}</div>
      )}
    </div>
  );
}

function IncidentCard({ incident, timeZone, expanded, selected = false, thumbnailAnnotations = true, onToggle, onSelect, onPreviewChange }) {
  const rawEvents = incident.events || [];
  const showSubEvents = rawEvents.length > 1;
  const [selectedPreview, setSelectedPreview] = useState(null);
  const [subEventsOpen, setSubEventsOpen] = useState(false);
  const [inlineVideoActive, setInlineVideoActive] = useState(false);
  const preview = selectedPreview || incident;
  const labels = incidentLabels(incident);
  const eventCount = incident.event_count || rawEvents.length || 1;
  const countText = `${eventCount} ${eventCount === 1 ? "event" : "events"}`;
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
    setSubEventsOpen(false);
    setInlineVideoActive(false);
  }, [incident.id]);

  useEffect(() => {
    setInlineVideoActive(false);
  }, [preview.id, preview.created_at]);

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
    if (expanded) setInlineVideoActive(true);
    else toggle();
  }

  function openOverlay(pointerEvent) {
    pointerEvent.stopPropagation();
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
      <div className="incident-preview" onClick={openPreview} aria-label={expanded ? "Play selected event video" : "Expand incident"}>
        <SnapshotImage
          event={preview}
          alt="incident snapshot"
          showAnnotations={expanded || thumbnailAnnotations}
          incidentEligibleOnly={!expanded}
        >
          <div className="incident-snapshot-hud">
            <div className="incident-snapshot-main">
              <strong>{incident.camera_id}</strong>
              <time>{expanded ? previewTimeText : timeText}</time>
            </div>
            <div className="pill-row compact incident-labels">
              {labels.length ? labels.slice(0, 3).map((item) => <span className="pill" key={item}>{item}</span>) : <span className="pill quiet">motion</span>}
            </div>
          </div>
          <IncidentClipLayer event={incident} active={expanded && inlineVideoActive} onEnded={() => setInlineVideoActive(false)} />
          <button type="button" className="event-count" onClick={openOverlay} onKeyDown={(event) => event.stopPropagation()} aria-label="Open event overlay" title="Open event overlay">{countText}</button>
        </SnapshotImage>
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
                {rawEvents.map((event, index) => {
                  const eventLabels = incidentLabels(event);
                  const eventLabelText = eventLabels.length ? eventLabels.join(", ") : "motion";
                  const isActive = (preview.id || incident.id) === event.id && (preview.created_at || incident.created_at) === event.created_at;
                  return (
                    <button type="button" key={`${event.id || "event"}-${index}`} className={isActive ? "active" : ""} onClick={() => { setSelectedPreview(event); setInlineVideoActive(false); }}>
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

function IncidentInspector({ incident, faceEvent, appConfig, timeZone, onOpen, onFaceOpen }) {
  if (!incident) return <aside className="incident-inspector"><div className="empty-state">Select an incident.</div></aside>;
  const objects = eventObjects(incident).filter((object) => object.label && object.incident_eligible !== false);
  const faces = faceEvent?.faces || [];
  const zones = incidentZones(incident);
  const eventId = Number(incident.representative_event_id || incident.id);
  const before = Number(appConfig?.event_clip_before_seconds ?? 5);
  const after = Number(appConfig?.event_clip_after_seconds ?? 5);
  const window = incidentClipWindow(incident, before, after);
  const clipUrl = Number.isFinite(eventId) ? eventClipUrl(eventId, window.before, window.after) : "";

  return (
    <aside className="incident-inspector">
      <div className="incident-inspector-head">
        <div><strong>{incident.camera_id}</strong><time>{formatDateTime(incident.created_at, timeZone)}</time></div>
        <button type="button" onClick={() => onOpen(incident)}>Open viewer</button>
      </div>
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
            <span>{Math.round(Number(face.confidence || 0) * 100)}%</span>
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
          <div><dt>Duration</dt><dd>{formatDuration(incident.duration_seconds || 0)}</dd></div>
          <div><dt>Start</dt><dd>{formatTimeOnly(incident.start_at || incident.created_at, timeZone)}</dd></div>
          <div><dt>End</dt><dd>{formatTimeOnly(incident.end_at || incident.created_at, timeZone)}</dd></div>
        </dl>
      </section>
      <div className="incident-inspector-actions">
        {clipUrl ? <a href={clipUrl} download={`survng-${incident.camera_id}-${eventId}.mp4`}><Download size={15} /> Video</a> : null}
        {incident.snapshot_path && eventSnapshotUrl(incident) ? <a href={eventSnapshotUrl(incident)} download><Download size={15} /> Snapshot</a> : null}
        <button type="button" onClick={() => onOpen(incident)}><Cpu size={15} /> Manual detect</button>
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
        const response = await fetch(`/api/detector/frame?confidence=${Number(confidence).toFixed(2)}`, {
          method: "POST",
          headers: { "Content-Type": "image/jpeg" },
          body: blob,
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        const tracks = updateTracks(payload.objects || []);
        draw(tracks, payload.width || width, payload.height || height);
        onStats?.({ inferenceMs: payload.elapsed_ms, objects: tracks.length, tracks: tracks.map((track) => track.id) });
      } catch (error) {
        onStats?.({ error: error.message || "Detection failed" });
      }
      if (!disposed) timer = window.setTimeout(sample, 500);
    }

    sample();
    return () => {
      disposed = true;
      window.clearTimeout(timer);
    };
  }, [active, confidence, videoRef, onStats]);

  return <canvas ref={canvasRef} className="event-detection-canvas" aria-hidden="true" />;
}

function EventOverlay({ event, events, timeZone, onClose, onSelect, onRefresh }) {
  const clipVideoRef = useRef(null);
  const mediaRef = useRef(null);
  const gestureRef = useRef({ mode: null, pointerId: null, startX: 0, startY: 0, panX: 0, panY: 0, moved: false, pinchDistance: 0, scale: 1 });
  const [clipInfo, setClipInfo] = useState(null);
  const [clipLoading, setClipLoading] = useState(false);
  const [clipError, setClipError] = useState("");
  const [playback, setPlayback] = useState(null);
  const [videoActive, setVideoActive] = useState(false);
  const [detectionDebug, setDetectionDebug] = useState(false);
  const [detectionDebugStats, setDetectionDebugStats] = useState(null);
  const [zoom, setZoom] = useState({ scale: 1, x: 0, y: 0 });
  const [mediaSize, setMediaSize] = useState(null);
  const zoomRef = useRef(zoom);
  const [manualConfidence, setManualConfidence] = useStoredState("survng.manualDetectionConfidence.v1", "0.35");
  const [manualDetection, setManualDetection] = useState(null);
  const [manualLoading, setManualLoading] = useState(false);
  const [manualError, setManualError] = useState("");
  const displayedEvent = manualDetection ? { ...event, objects: manualDetection.objects || [] } : event;
  const objects = eventObjects(displayedEvent);
  const detectedObjects = objects.filter((object) => object.label);
  const manualConfidenceNumber = Number(manualConfidence);
  const safeManualConfidence = Number.isFinite(manualConfidenceNumber) ? Math.max(0.01, Math.min(0.99, manualConfidenceNumber)) : 0.35;
  const manualEventId = Number(event.representative_event_id || event.id);
  const downloadName = `survng-${String(event.camera_id || "camera")}-${String(event.created_at || event.id || "event").replace(/[^0-9A-Za-z_-]+/g, "-")}.mp4`;

  useEffect(() => {
    let cancelled = false;
    async function loadClipSettings() {
      const eventId = Number(event.representative_event_id || event.id);
      if (!Number.isFinite(eventId)) {
        setClipInfo(null);
        setClipLoading(false);
        setClipError("No event video available");
        return;
      }
      setClipInfo(null);
      setPlayback(null);
      setClipLoading(true);
      setClipError("");
      setVideoActive(false);
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
        // Keep the default review window if settings are temporarily unavailable.
      }
      if (cancelled) return;
      const safeBefore = Number.isFinite(before) ? before : 5;
      const safeAfter = Number.isFinite(after) ? after : 5;
      const window = incidentClipWindow(event, safeBefore, safeAfter);
      const info = {
        streamUrl: eventStreamUrl(eventId, window.before, window.after),
        downloadUrl: eventClipUrl(eventId, window.before, window.after),
        before: window.before,
        after: window.after,
        duration: window.before + window.after,
      };
      setClipInfo(info);
      setPlayback({ url: info.streamUrl, mimeType: "application/vnd.apple.mpegurl" });
    }
    loadClipSettings();
    return () => { cancelled = true; };
  }, [event.id, event.representative_event_id, event.start_epoch, event.last_epoch]);

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

  useEffect(() => {
    resetZoom();
    setMediaSize(null);
    setDetectionDebug(false);
    setDetectionDebugStats(null);
    setManualDetection(null);
    setManualError("");
    setManualLoading(false);
  }, [event.id]);

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

  useEffect(() => {
    function onKey(keyEvent) {
      if (keyEvent.key === "Escape") {
        onClose();
        return;
      }
      if (keyEvent.key !== "ArrowLeft" && keyEvent.key !== "ArrowRight") return;
      const navItems = events.flatMap((candidate) => [candidate, ...(candidate.events || [])]);
      if (!navItems.length) return;
      const currentIndex = navItems.findIndex((candidate) => candidate.id === event.id);
      if (currentIndex < 0) return;
      keyEvent.preventDefault();
      const direction = keyEvent.key === "ArrowRight" ? 1 : -1;
      const nextIndex = (currentIndex + direction + navItems.length) % navItems.length;
      onSelect(navItems[nextIndex]);
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
            <h2>{event.camera_id}</h2>
            <time>{formatDateTime(event.created_at, timeZone)}</time>
          </div>
          <div className="overlay-actions">
            <button
              type="button"
              className={`tile-control-button debug-detection-toggle ${detectionDebug ? "active" : ""}`}
              onClick={() => {
                if (!videoActive) playEventClip();
                setDetectionDebug((enabled) => !enabled);
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
            event={displayedEvent}
            alt="selected event snapshot"
            iconSize={42}
            className="event-snapshot-frame"
            layerStyle={{ transform: `translate3d(${zoom.x}px, ${zoom.y}px, 0) scale(${zoom.scale})` }}
            allowObjectFocus={zoom.scale === 1 && !videoActive}
            showAnnotations
            onImageSize={setMediaSize}
          />
          {videoActive && clipInfo && playback && !clipError ? (
            <>
              <ShakaVideo
                className="event-video-layer"
                ref={clipVideoRef}
                src={playback.url}
                mimeType={playback.mimeType}
                autoPlay
                controls
                playsInline
                preload="metadata"
                onReady={() => { setClipLoading(false); setClipError(""); }}
                onError={() => {
                  if (playback.url !== clipInfo.downloadUrl) {
                    setClipLoading(true);
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
          <div>
            <span className="muted">Detection</span>
            <div className="pill-row">
              {detectedObjects.length ? detectedObjects.map((object, index) => (
                <span className="pill" key={`${object.label}-${index}`}>
                  {object.label}
                  {object.confidence ? ` ${(object.confidence * 100).toFixed(1)}%` : ""}
                </span>
              )) : <span className="pill quiet">motion only</span>}
            </div>
          </div>
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
      </section>
    </div>
  );
}

function IncidentsPage({ timeZone, onRecordingContextChange }) {
  const { cameras, appConfig, refresh: refreshBase } = usePollingData(false);
  const thumbnailAnnotations = appConfig?.incident_thumbnail_annotations ?? true;
  const [eventFilter, setEventFilter] = useStoredState("survng.liveEventFilter.v2", "object");
  const [incidentCameraFilter, setIncidentCameraFilter] = useStoredState("survng.incidentCameraFilter.v1", "all");
  const [incidentObjectFilter, setIncidentObjectFilter] = useStoredState("survng.incidentObjectFilter.v1", "all");
  const [incidentZoneFilter, setIncidentZoneFilter] = useStoredState("survng.incidentZoneFilter.v1", "all");
  const [incidentDensity, setIncidentDensity] = useStoredState("survng.incidentDensity.v1", "compact");
  const today = dateKeyForTimeZone(Date.now(), timeZone);
  const [incidentDay, setIncidentDay] = useStoredState("survng.incidentDay.v1", today);
  const [incidents, setIncidents] = useState([]);
  const [incidentTotal, setIncidentTotal] = useState(0);
  const [incidentFacets, setIncidentFacets] = useState({ camera_ids: [], labels: [], zones: [] });
  const [incidentLoading, setIncidentLoading] = useState(true);
  const [incidentLoadError, setIncidentLoadError] = useState("");
  const [incidentRefreshToken, setIncidentRefreshToken] = useState(0);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [focusedFaceEventId, setFocusedFaceEventId] = useState(null);
  const [selectedFace, setSelectedFace] = useState(null);
  const [facePeople, setFacePeople] = useState([]);
  const [expandedIncidentId, setExpandedIncidentId] = useState(null);
  const [incidentPage, setIncidentPage] = useState(0);
  const mobileView = isMobileViewport();
  const incidentsPerPage = mobileView ? 12 : incidentDensity === "comfortable" ? 10 : 16;
  const cameraNameById = useMemo(() => new Map(cameras.map((camera) => [camera.id, camera.name || camera.id])), [cameras]);
  const incidentCameraOptions = incidentFacets.camera_ids || [];
  const incidentObjectOptions = incidentFacets.labels || [];
  const incidentZoneOptions = incidentFacets.zones || [];
  const visibleIncidents = incidents;
  const explicitlyFocusedIncident = visibleIncidents.find((incident) => incident.id === expandedIncidentId) || null;
  const focusedIncident = mobileView ? explicitlyFocusedIncident : explicitlyFocusedIncident || visibleIncidents[0] || null;
  const focusedEvent = (focusedIncident?.events || []).find((event) => Number(event.id) === Number(focusedFaceEventId))
    || (focusedIncident?.events || []).find((event) => Number(event.id) === Number(focusedIncident.representative_event_id))
    || focusedIncident;
  const galleryIncidents = visibleIncidents;
  const incidentPageCount = Math.max(1, Math.ceil(incidentTotal / incidentsPerPage));
  const clampedIncidentPage = Math.min(incidentPage, incidentPageCount - 1);
  const pagedIncidents = galleryIncidents;

  function refresh() {
    refreshBase();
    setIncidentRefreshToken((value) => value + 1);
  }

  useAppEvents(({ type }) => {
    if (type === "incident") setIncidentRefreshToken((value) => value + 1);
  });

  async function openFaceReview(face) {
    const observationId = Number(face?.observation_id);
    if (!Number.isFinite(observationId)) return;
    const [observationResponse, peopleResponse] = await Promise.all([
      fetch(`/api/faces/observations/${observationId}`),
      fetch("/api/faces/people"),
    ]);
    if (!observationResponse.ok || !peopleResponse.ok) return;
    setFacePeople(await peopleResponse.json());
    setSelectedFace(await observationResponse.json());
  }

  useEffect(() => {
    if (incidentDay > today) setIncidentDay(today);
  }, [incidentDay, setIncidentDay, today]);

  useEffect(() => {
    let cancelled = false;
    async function loadIncidentPage() {
      setIncidentLoading(true);
      setIncidentLoadError("");
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
      try {
        const response = await fetch(`/api/incidents/search?${query}`);
        if (!response.ok) throw new Error("Unable to load incidents");
        const payload = await response.json();
        if (cancelled) return;
        setIncidents(payload.items || []);
        setIncidentTotal(Number(payload.total || 0));
        setIncidentFacets(payload.facets || { camera_ids: [], labels: [], zones: [] });
      } catch (error) {
        if (!cancelled) {
          setIncidents([]);
          setIncidentTotal(0);
          setIncidentLoadError(error.message || "Unable to load incidents");
        }
      } finally {
        if (!cancelled) setIncidentLoading(false);
      }
    }
    loadIncidentPage();
    return () => {
      cancelled = true;
    };
  }, [incidentDay, today, timeZone, eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, incidentPage, incidentsPerPage, incidentRefreshToken]);

  useEffect(() => {
    setIncidentPage(0);
  }, [eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter, incidentDay, incidentDensity]);

  useEffect(() => {
    if (incidentPage >= incidentPageCount) setIncidentPage(Math.max(0, incidentPageCount - 1));
  }, [incidentPage, incidentPageCount]);

  useEffect(() => {
    setFocusedFaceEventId(null);
  }, [focusedIncident?.id]);

  useEffect(() => {
    const context = incidentRecordingContext(selectedEvent || focusedEvent);
    if (context) onRecordingContextChange(context);
  }, [selectedEvent?.id, focusedEvent?.id, focusedEvent?.created_at, focusedEvent?.camera_id, onRecordingContextChange]);

  useEffect(() => {
    if (selectedEvent && !visibleIncidents.some((incident) => {
      if (incident.id === selectedEvent.id) return true;
      return (incident.events || []).some((event) => event.id === selectedEvent.id);
    })) {
      setSelectedEvent(null);
    }
    if (expandedIncidentId && !visibleIncidents.some((incident) => incident.id === expandedIncidentId)) {
      setExpandedIncidentId(null);
    }
  }, [selectedEvent, expandedIncidentId, visibleIncidents]);

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
      if (incident) setSelectedEvent(incident);
      return;
    }
    setExpandedIncidentId(incidentId);
  }

  const focusedIndex = focusedIncident ? visibleIncidents.findIndex((incident) => incident.id === focusedIncident.id) : -1;

  function moveFocus(direction) {
    if (!visibleIncidents.length) return;
    const nextIndex = Math.max(0, Math.min(visibleIncidents.length - 1, focusedIndex + direction));
    setExpandedIncidentId(visibleIncidents[nextIndex].id);
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

  if (!mobileView) {
    return (
      <main className="incidents-desktop-page with-inspector">
        <section className="bento-card incidents-desktop-shell">
          <div className="incidents-desktop-toolbar">
            <div className="incident-filter-toggle compact" aria-label="Incident type filter">
              <button className={eventFilter === "object" ? "active" : ""} onClick={() => setEventFilter("object")}>Object</button>
              <button className={eventFilter === "motion" ? "active" : ""} onClick={() => setEventFilter("motion")}>Motion</button>
            </div>
            <div className="incident-filter-selects desktop">
              <label className="incident-day-field"><span>Day</span><input type="date" value={incidentDay} max={today} onChange={(event) => setIncidentDay(event.target.value || today)} aria-label="Incident day" /></label>
              <label><span>Camera</span><select value={incidentCameraFilter} onChange={(event) => setIncidentCameraFilter(event.target.value)}><option value="all">All cameras</option>{incidentCameraOptions.map((id) => <option value={id} key={id}>{cameraNameById.get(id) || id}</option>)}</select></label>
              <label><span>Object</span><select value={incidentObjectFilter} onChange={(event) => setIncidentObjectFilter(event.target.value)}><option value="all">All objects</option>{incidentObjectOptions.map((label) => <option value={label} key={label}>{label}</option>)}</select></label>
              <label><span>Zone</span><select value={incidentZoneFilter} onChange={(event) => setIncidentZoneFilter(event.target.value)}><option value="all">All zones</option>{incidentZoneOptions.map((zone) => <option value={zone} key={zone}>{zone}</option>)}</select></label>
            </div>
            <span className="shown-bubble">{incidentTotal} shown</span>
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
              <div className="incident-rail-list">
                {incidentLoading ? <div className="empty-state">Loading incidents...</div> : null}
                {!incidentLoading && incidentLoadError ? <div className="empty-state">{incidentLoadError}</div> : null}
                {!incidentLoading && !incidentLoadError && galleryIncidents.length ? pagedIncidents.map((incident) => (
                  <IncidentCard key={incident.id} incident={incident} timeZone={timeZone} expanded={false} selected={incident.id === focusedIncident?.id} thumbnailAnnotations={thumbnailAnnotations} onToggle={toggleIncident} onSelect={setSelectedEvent} />
                )) : null}
                {!incidentLoading && !incidentLoadError && !galleryIncidents.length ? <div className="empty-state">No other incidents.</div> : null}
              </div>
              {incidentTotal > incidentsPerPage ? (
                <div className="incident-pager" aria-label="Incident pages">
                  <button type="button" onClick={() => setIncidentPage((page) => Math.max(0, page - 1))} disabled={clampedIncidentPage === 0}>Prev</button>
                  <span>{clampedIncidentPage + 1} / {incidentPageCount}</span>
                  <button type="button" onClick={() => setIncidentPage((page) => Math.min(incidentPageCount - 1, page + 1))} disabled={clampedIncidentPage >= incidentPageCount - 1}>Next</button>
                </div>
              ) : null}
            </aside>

            <section className="incident-investigation">
              <div className="incident-focus-nav">
                <span>{focusedIndex >= 0 ? `${incidentPage * incidentsPerPage + focusedIndex + 1} of ${incidentTotal}` : "No incident selected"}</span>
              </div>
              <div className="incident-desktop-focus">
                {focusedIncident ? (
                  <>
                    <button type="button" className="incident-focus-arrow previous" onClick={() => moveFocus(-1)} disabled={focusedIndex <= 0} title="Previous incident" aria-label="Previous incident"><ChevronLeft size={26} /></button>
                    <button type="button" className="incident-focus-arrow next" onClick={() => moveFocus(1)} disabled={focusedIndex < 0 || focusedIndex >= visibleIncidents.length - 1} title="Next incident" aria-label="Next incident"><ChevronRight size={26} /></button>
                  </>
                ) : null}
                {focusedIncident ? <IncidentCard incident={focusedIncident} timeZone={timeZone} expanded thumbnailAnnotations={thumbnailAnnotations} onToggle={toggleIncident} onSelect={setSelectedEvent} onPreviewChange={setFocusedFaceEventId} /> : <div className="empty-state">No incidents match the current filters.</div>}
              </div>
            </section>

            <IncidentInspector incident={focusedIncident} faceEvent={focusedEvent} appConfig={appConfig} timeZone={timeZone} onOpen={setSelectedEvent} onFaceOpen={openFaceReview} />
          </div>
        </section>
        {selectedEvent ? <EventOverlay event={selectedEvent} events={visibleIncidents} timeZone={timeZone} onClose={() => setSelectedEvent(null)} onSelect={setSelectedEvent} onRefresh={refresh} /> : null}
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
              <button className={eventFilter === "motion" ? "active" : ""} onClick={() => setEventFilter("motion")}>Motion</button>
            </div>
            <span className="shown-bubble">{incidentTotal} shown</span>
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
        </div>
        <div className="incident-gallery">
          {incidentLoading ? <div className="empty-state">Loading incidents...</div> : null}
          {!incidentLoading && incidentLoadError ? <div className="empty-state">{incidentLoadError}</div> : null}
          {!incidentLoading && !incidentLoadError && visibleIncidents.length
            ? pagedIncidents.map((incident) => (
              <IncidentCard
                key={incident.id}
                incident={incident}
                timeZone={timeZone}
                expanded={false}
                thumbnailAnnotations={thumbnailAnnotations}
                onToggle={toggleIncident}
                onSelect={setSelectedEvent}
              />
            ))
            : null}
          {!incidentLoading && !incidentLoadError && !visibleIncidents.length ? <div className="empty-state">No incidents match the current filters.</div> : null}
        </div>
        {incidentTotal > incidentsPerPage ? (
          <div className="incident-pager" aria-label="Incident pages">
            <button type="button" onClick={() => setIncidentPage((page) => Math.max(0, page - 1))} disabled={clampedIncidentPage === 0}>Prev</button>
            <span>{clampedIncidentPage + 1} / {incidentPageCount}</span>
            <button type="button" onClick={() => setIncidentPage((page) => Math.min(incidentPageCount - 1, page + 1))} disabled={clampedIncidentPage >= incidentPageCount - 1}>Next</button>
          </div>
        ) : null}
      </section>
      {selectedEvent ? <EventOverlay event={selectedEvent} events={visibleIncidents} timeZone={timeZone} onClose={() => setSelectedEvent(null)} onSelect={setSelectedEvent} onRefresh={refresh} /> : null}
    </main>
  );
}

function LivePage({ timeZone, onRecordingContextChange }) {
  const { cameras, incidents, appConfig, refresh } = usePollingData();
  const thumbnailAnnotations = appConfig?.incident_thumbnail_annotations ?? true;
  const [eventFilter, setEventFilter] = useStoredState("survng.liveEventFilter.v2", "object");
  const [incidentCameraFilter, setIncidentCameraFilter] = useStoredState("survng.incidentCameraFilter.v1", "all");
  const [incidentObjectFilter, setIncidentObjectFilter] = useStoredState("survng.incidentObjectFilter.v1", "all");
  const [incidentZoneFilter, setIncidentZoneFilter] = useStoredState("survng.incidentZoneFilter.v1", "all");
  const [cameraOrder, setCameraOrder] = useStoredState("survng.liveCameraOrder.v1", "[]");
  const [dragCameraId, setDragCameraId] = useState("");
  const [dragOverCameraId, setDragOverCameraId] = useState("");
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [expandedIncidentId, setExpandedIncidentId] = useState(null);
  const [expandedCamera, setExpandedCamera] = useState(null);
  const [incidentPage, setIncidentPage] = useState(0);
  const incidentsPerPage = 12;
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
  const cameraNameById = useMemo(() => new Map(cameras.map((camera) => [camera.id, camera.name || camera.id])), [cameras]);
  const incidentCameraOptions = useMemo(() => {
    const ids = Array.from(new Set(incidents.map((incident) => incident.camera_id).filter(Boolean)));
    return ids.sort((left, right) => (cameraNameById.get(left) || left).localeCompare(cameraNameById.get(right) || right));
  }, [incidents, cameraNameById]);
  const incidentObjectOptions = useMemo(() => {
    const labels = new Set();
    incidents.forEach((incident) => incidentLabels(incident).forEach((label) => labels.add(label)));
    return Array.from(labels).sort((left, right) => left.localeCompare(right));
  }, [incidents]);
  const incidentZoneOptions = useMemo(() => {
    const zones = new Set();
    incidents.forEach((incident) => incidentZones(incident).forEach((zone) => zones.add(zone)));
    return Array.from(zones).sort((left, right) => left.localeCompare(right));
  }, [incidents]);
  const visibleIncidents = useMemo(() => incidents.filter((incident) => {
    if (eventFilter === "object" && !hasDetectedObjects(incident)) return false;
    if (incidentCameraFilter !== "all" && incident.camera_id !== incidentCameraFilter) return false;
    if (incidentObjectFilter !== "all" && !incidentLabels(incident).includes(incidentObjectFilter)) return false;
    if (incidentZoneFilter !== "all" && !incidentZones(incident).includes(incidentZoneFilter)) return false;
    return true;
  }), [incidents, eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter]);
  const focusedIncident = visibleIncidents.find((incident) => incident.id === expandedIncidentId) || null;
  const galleryIncidents = visibleIncidents;
  const incidentPageCount = Math.max(1, Math.ceil(galleryIncidents.length / incidentsPerPage));
  const clampedIncidentPage = Math.min(incidentPage, incidentPageCount - 1);
  const pagedIncidents = galleryIncidents.slice(clampedIncidentPage * incidentsPerPage, (clampedIncidentPage + 1) * incidentsPerPage);

  function moveCameraBefore(sourceId, targetId) {
    if (!sourceId || !targetId || sourceId === targetId) return;
    const ids = orderedCameras.map((camera) => camera.id);
    const withoutSource = ids.filter((id) => id !== sourceId);
    const targetIndex = withoutSource.indexOf(targetId);
    if (targetIndex < 0) return;
    withoutSource.splice(targetIndex, 0, sourceId);
    setCameraOrder(JSON.stringify(withoutSource));
  }

  useEffect(() => {
    setIncidentPage(0);
  }, [eventFilter, incidentCameraFilter, incidentObjectFilter, incidentZoneFilter]);

  useEffect(() => {
    if (incidentPage >= incidentPageCount) setIncidentPage(Math.max(0, incidentPageCount - 1));
  }, [incidentPage, incidentPageCount]);

  useEffect(() => {
    const context = incidentRecordingContext(selectedEvent || focusedIncident);
    if (context) onRecordingContextChange(context);
  }, [selectedEvent?.id, focusedIncident?.id, focusedIncident?.created_at, focusedIncident?.camera_id, onRecordingContextChange]);

  useEffect(() => {
    if (selectedEvent && !visibleIncidents.some((incident) => {
      if (incident.id === selectedEvent.id) return true;
      return (incident.events || []).some((event) => event.id === selectedEvent.id);
    })) {
      setSelectedEvent(null);
    }
    if (expandedIncidentId && !visibleIncidents.some((incident) => incident.id === expandedIncidentId)) {
      setExpandedIncidentId(null);
    }
  }, [selectedEvent, expandedIncidentId, visibleIncidents]);

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
    setExpandedIncidentId((current) => current === incidentId ? null : incidentId);
  }

  return (
    <main className="bento-grid live-grid">
      <section className="bento-card camera-zone">
        <div className="camera-grid">
          {orderedCameras.map((camera, index) => (
            <CameraTile
              key={camera.id}
              camera={camera}
              timeZone={timeZone}
              refresh={refresh}
              onOpen={setExpandedCamera}
              startDelayMs={index * 450}
              dragging={dragCameraId === camera.id}
              dragOver={dragOverCameraId === camera.id && dragCameraId !== camera.id}
              dragHandleProps={{
                draggable: true,
                onDragStart: (event) => {
                  setDragCameraId(camera.id);
                  event.dataTransfer.effectAllowed = "move";
                  event.dataTransfer.setData("text/plain", camera.id);
                },
                onDragEnd: () => {
                  setDragCameraId("");
                  setDragOverCameraId("");
                },
              }}
              dropProps={{
                onDragOver: (event) => {
                  if (!dragCameraId || dragCameraId === camera.id) return;
                  event.preventDefault();
                  event.dataTransfer.dropEffect = "move";
                  setDragOverCameraId(camera.id);
                },
                onDragLeave: (event) => {
                  if (!event.currentTarget.contains(event.relatedTarget)) setDragOverCameraId("");
                },
                onDrop: (event) => {
                  event.preventDefault();
                  const sourceId = event.dataTransfer.getData("text/plain") || dragCameraId;
                  moveCameraBefore(sourceId, camera.id);
                  setDragCameraId("");
                  setDragOverCameraId("");
                },
              }}
            />
          ))}
        </div>
      </section>
      <section className="bento-card events-zone">
        <div className="section-head compact incident-head">
          <div><h2>Incidents</h2></div>
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
              onSelect={setSelectedEvent}
            />
          </div>
        ) : null}
        <div className="incident-gallery">
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
                onSelect={setSelectedEvent}
              />
            ))
            : <div className="empty-state">No incidents match the current filters.</div>}
        </div>
        {galleryIncidents.length > incidentsPerPage ? (
          <div className="incident-pager" aria-label="Incident pages">
            <button type="button" onClick={() => setIncidentPage((page) => Math.max(0, page - 1))} disabled={clampedIncidentPage === 0}>Prev</button>
            <span>{clampedIncidentPage + 1} / {incidentPageCount}</span>
            <button type="button" onClick={() => setIncidentPage((page) => Math.min(incidentPageCount - 1, page + 1))} disabled={clampedIncidentPage >= incidentPageCount - 1}>Next</button>
          </div>
        ) : null}
      </section>
      {selectedEvent ? <EventOverlay event={selectedEvent} events={visibleIncidents} timeZone={timeZone} onClose={() => setSelectedEvent(null)} onSelect={setSelectedEvent} onRefresh={refresh} /> : null}
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

function mergeRecordingAvailability(current, updates) {
  const ranges = [...current, ...updates]
    .map((item) => ({
      ...item,
      start_epoch: Number(item.start_epoch),
      end_epoch: Number(item.end_epoch),
    }))
    .filter((item) => Number.isFinite(item.start_epoch) && Number.isFinite(item.end_epoch))
    .sort((a, b) => a.start_epoch - b.start_epoch);
  const merged = [];
  ranges.forEach((item) => {
    const previous = merged[merged.length - 1];
    if (previous && item.start_epoch <= previous.end_epoch + 5) {
      previous.end_epoch = Math.max(previous.end_epoch, item.end_epoch);
      previous.duration_seconds = previous.end_epoch - previous.start_epoch;
      previous.segment_count = Math.max(
        Number(previous.segment_count) || 0,
        Number(item.segment_count) || 0,
      );
      return;
    }
    merged.push({ ...item });
  });
  return merged;
}

function mergeRecordingEvents(current, updates) {
  const byId = new Map(current.map((event) => [event.id, event]));
  updates.forEach((event) => byId.set(event.id, event));
  return [...byId.values()]
    .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    .slice(-5000);
}

function RecordingsPage({ timeZone }) {
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
  const today = dateKeyForTimeZone(Date.now(), timeZone);
  const queryDate = initialQuery.get("date") || (initialEpoch ? dateKeyForTimeZone(initialEpoch * 1000, timeZone) : today);
  const querySource = initialQuery.get("source");
  const [cameras, setCameras] = useState([]);
  const [cameraId, setCameraId] = useState(initialQuery.get("camera") || "");
  const [source, setSource] = useState(querySource === "live" || querySource === "main" ? querySource : preferredStreamSource());
  const [date, setDate] = useState(/^\d{4}-\d{2}-\d{2}$/.test(queryDate) && queryDate <= today ? queryDate : today);
  const [recordings, setRecordings] = useState([]);
  const [playbackDetail, setPlaybackDetail] = useState(null);
  const [events, setEvents] = useState([]);
  const [availableSources, setAvailableSources] = useState([]);
  const [playhead, setPlayhead] = useState(null);
  const [loading, setLoading] = useState(true);
  const [playbackError, setPlaybackError] = useState("");
  const [playbackNotice, setPlaybackNotice] = useState("");
  const [playbackBlocked, setPlaybackBlocked] = useState(false);
  const [playbackWindow, setPlaybackWindow] = useState(null);
  const [followTarget, setFollowTarget] = useState(null);

  const activeCameraId = cameras.some((camera) => camera.id === cameraId) ? cameraId : cameras[0]?.id || "";
  const dayStart = useMemo(() => zonedDateSecondToEpoch(date, 0, timeZone), [date, timeZone]);
  const nextDate = addDaysToDateKey(date, 1);
  const dayEnd = useMemo(() => zonedDateSecondToEpoch(nextDate, 0, timeZone), [nextDate, timeZone]);
  const daySeconds = Math.max(1, dayEnd - dayStart);

  const timeline = useMemo(() => {
    let mediaOffset = 0;
    return recordings
      .map((item) => ({ ...item, start_epoch: Number(item.start_epoch), end_epoch: Number(item.end_epoch) }))
      .filter((item) => Number.isFinite(item.start_epoch) && Number.isFinite(item.end_epoch))
      .sort((a, b) => a.start_epoch - b.start_epoch)
      .map((item) => {
        const duration = Math.max(0.01, Number(item.duration_seconds) || item.end_epoch - item.start_epoch);
        const mapped = { ...item, media_start: mediaOffset, media_end: mediaOffset + duration };
        mediaOffset += duration;
        return mapped;
      });
  }, [recordings]);
  latestAvailabilityRef.current = timeline[timeline.length - 1]?.end_epoch ?? null;
  const playbackTimeline = useMemo(() => {
    if (!playbackDetail) return [];
    let mediaOffset = 0;
    return playbackDetail.rows
      .map((item) => ({ ...item, start_epoch: Number(item.start_epoch), end_epoch: Number(item.end_epoch) }))
      .filter((item) => Number.isFinite(item.start_epoch) && Number.isFinite(item.end_epoch))
      .sort((a, b) => a.start_epoch - b.start_epoch)
      .map((item) => {
        const duration = Math.max(0.01, Number(item.duration_seconds) || item.end_epoch - item.start_epoch);
        const mapped = { ...item, media_start: mediaOffset, media_end: mediaOffset + duration };
        mediaOffset += duration;
        return mapped;
      });
  }, [playbackDetail]);
  const loadedPlaybackWindow = playbackDetail;
  const manifestUrl = activeCameraId && playbackDetail && playbackTimeline.length
    ? recordingDayHlsUrl(activeCameraId, playbackDetail.start, playbackDetail.end, source)
    : "";
  const manifestStartTime = useMemo(() => {
    if (!playbackTimeline.length) return null;
    const retainedEpoch = desiredEpochRef.current;
    const initialEpoch = Number.isFinite(retainedEpoch) && retainedEpoch >= dayStart && retainedEpoch < dayEnd
      ? retainedEpoch
      : date === today ? Date.now() / 1000 : timeline[0].start_epoch;
    return epochToPlaybackMediaTime(initialEpoch);
  }, [manifestUrl, playbackTimeline]);

  const nearbyEvents = useMemo(() => {
    if (!Number.isFinite(playhead)) return events.slice(0, 20);
    return events
      .map((event) => ({ ...event, distance: Math.abs(new Date(event.created_at).getTime() / 1000 - playhead) }))
      .filter((event) => event.distance <= 15 * 60)
      .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
      .slice(0, 24);
  }, [events, playhead]);

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
    autoplayRef.current = autoplay;
    setFollowTarget(null);
    setPlaybackError("");
    setPlayhead(target);
    desiredEpochRef.current = target;
    const nextWindow = windowAround(target);
    const inCurrentWindow = loadedPlaybackWindow
      && target >= loadedPlaybackWindow.start
      && target < loadedPlaybackWindow.end;
    const video = videoRef.current;
    if (autoplay && video) requestRecordingPlay(video, false);
    const mediaTime = epochToPlaybackMediaTime(target);
    if (inCurrentWindow && video && Number.isFinite(mediaTime)) {
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
      setPlaybackWindow(nextWindow);
    }
  }

  useEffect(() => {
    fetch("/api/cameras")
      .then((response) => response.json())
      .then(setCameras)
      .catch(() => setPlaybackError("Unable to load cameras"));
  }, []);

  useEffect(() => {
    if (!activeCameraId) return undefined;
    const controller = new AbortController();
    setLoading(true);
    setPlaybackError("");
    setPlaybackBlocked(false);
    setRecordings([]);
    playbackRequestRef.current += 1;
    setPlaybackDetail(null);
    setEvents([]);
    setAvailableSources([]);
    setPlaybackWindow(null);
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
    fetch(recordingDayUrl(activeCameraId, dayStart, dayEnd, source), { signal: controller.signal })
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
          setPlaybackNotice("No Main recording exists for this day; using Sub.");
          setSource("live");
          return;
        }
        setRecordings(nextAvailability);
        setEvents(payload.events || []);
      })
      .catch((error) => {
        if (error.name !== "AbortError") setPlaybackError(error.message || "Unable to load recordings");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [activeCameraId, source, dayStart, dayEnd]);

  useEffect(() => {
    if (!activeCameraId || date !== today) return undefined;
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
        const eventUpdates = payload.events || [];
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
  }, [activeCameraId, source, date, today, dayStart, dayEnd]);

  useEffect(() => {
    if (!activeCameraId || !playbackWindow) return undefined;
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
        });
      })
      .catch((error) => {
        if (error.name !== "AbortError" && requestId === playbackRequestRef.current) {
          pendingSeekEpochRef.current = null;
          pendingSeekModeRef.current = null;
          setPlaybackNotice("");
          setPlaybackError(error.message || "Unable to load recording window");
        }
      });
    return () => controller.abort();
  }, [activeCameraId, source, playbackWindow?.start, playbackWindow?.end]);

  useEffect(() => {
    if (!timeline.length || Number.isFinite(playhead)) return;
    const retainedEpoch = desiredEpochRef.current;
    const initialEpoch = Number.isFinite(retainedEpoch) && retainedEpoch >= dayStart && retainedEpoch < dayEnd
      ? retainedEpoch
      : date === today ? Date.now() / 1000 : timeline[0].start_epoch;
    playAt(initialEpoch, false);
  }, [timeline, playhead]);

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

  function handleRecordingError() {
    if (source === "main" && availableSources.includes("live") && !codecFallbackRef.current) {
      codecFallbackRef.current = true;
      setPlaybackNotice("Main stream is not supported by this browser; using Sub.");
      setSource("live");
      return;
    }
    setPlaybackError("This browser could not play the selected recording stream");
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

  return (
    <main className="recordings-v2-page">
      <aside className="recordings-v2-cameras" aria-label="Cameras">
        {cameras.map((camera) => (
          <button key={camera.id} type="button" className={camera.id === activeCameraId ? "active" : ""} onClick={() => setCameraId(camera.id)}>
            <Camera size={16} />
            <span>{camera.name}</span>
            <i className={(source === "main" ? camera.recording : camera.sub_recording) ? "online" : ""} />
          </button>
        ))}
      </aside>

      <section className="recordings-v2-workspace">
        <div className="recordings-v2-player">
          {manifestUrl ? (
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
          {playbackError ? <div className="recordings-v2-error">{playbackError}</div> : null}
          {playbackNotice && !playbackError ? <div className="recordings-v2-notice">{playbackNotice}</div> : null}
          {playbackBlocked && !playbackError ? (
            <button type="button" className="recordings-v2-play" onClick={() => requestRecordingPlay(videoRef.current)}>
              <Play size={22} fill="currentColor" />
              Play recording
            </button>
          ) : null}
        </div>

        <div className="recordings-v2-controls">
          <div className="recordings-v2-toolbar">
            <div className="recordings-v2-source" aria-label="Recording stream">
              <button type="button" className={source === "main" ? "active" : ""} onClick={() => setSource("main")} disabled={availableSources.length > 0 && !availableSources.includes("main")}>Main</button>
              <button type="button" className={source === "live" ? "active" : ""} onClick={() => setSource("live")} disabled={availableSources.length > 0 && !availableSources.includes("live")}>Sub</button>
            </div>
            <div className="recordings-v2-date">
              <button type="button" onClick={() => changeDate(addDaysToDateKey(date, -1))} aria-label="Previous day"><SkipBack size={16} /></button>
              <input type="date" value={date} max={today} onChange={(event) => changeDate(event.target.value || today)} aria-label="Recording day" />
              <button type="button" onClick={() => changeDate(addDaysToDateKey(date, 1))} disabled={date >= today} aria-label="Next day"><SkipForward size={16} /></button>
              <button type="button" onClick={() => changeDate(today)} disabled={date === today}>Today</button>
            </div>
          </div>

          <RecordingTimeline
            startEpoch={dayStart}
            endEpoch={dayEnd}
            recordings={timeline}
            playhead={playhead ?? dayStart}
            onSeek={(epoch) => playAt(epoch, true)}
          />
        </div>

        <div className="recordings-v2-events">
          {nearbyEvents.length ? nearbyEvents.map((event) => (
            <button key={event.id} type="button" onClick={() => playAt(new Date(event.created_at).getTime() / 1000, true)}>
              <time>{formatTimeOnly(event.created_at, timeZone)}</time>
              <span>{event.labels?.length ? event.labels.join(", ") : "motion"}</span>
            </button>
          )) : <div className="recordings-v2-no-events"><Radar size={17} />No events near this time</div>}
        </div>
      </section>
    </main>
  );
}

function RecordingTimeline({ startEpoch, endEpoch, recordings, playhead, onSeek }) {
  const duration = Math.max(1, endEpoch - startEpoch);
  const offset = Math.max(0, Math.min(duration, playhead - startEpoch));
  const [draft, setDraft] = useState(offset);
  const draftRef = useRef(offset);
  const dragRef = useRef(null);
  useEffect(() => {
    if (dragRef.current) return;
    draftRef.current = offset;
    setDraft(offset);
  }, [offset]);
  const percent = (draft / duration) * 100;

  function updateDraft(value) {
    const next = Math.max(0, Math.min(duration, Number(value) || 0));
    draftRef.current = next;
    setDraft(next);
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
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    if (!drag.precise) updateDraft(pointerValue(event, drag));
  }

  function moveDrag(event) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    updateDraft(pointerValue(event, drag));
  }

  function finishDrag(event, cancelled = false) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
    if (cancelled) updateDraft(offset);
    else commit(pointerValue(event, drag));
  }

  return (
    <div className="recordings-v2-timeline">
      <div className="recordings-v2-track">
        {recordings.map((item) => (
          <span
            key={item.path || `${item.start_epoch}:${item.end_epoch}`}
            style={{
              left: `${((Math.max(startEpoch, item.start_epoch) - startEpoch) / duration) * 100}%`,
              width: `${((Math.min(endEpoch, item.end_epoch) - Math.max(startEpoch, item.start_epoch)) / duration) * 100}%`,
            }}
          />
        ))}
        <i style={{ left: `${percent}%` }} />
        <output style={{ left: `${Math.max(4, Math.min(96, percent))}%` }}>{formatDayClock(draft)}</output>
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

function ConfigPage({ timeZone, setTimeZone, theme, setTheme }) {
  const [config, setConfig] = useState(null);
  const [runtimeStatus, setRuntimeStatus] = useState([]);
  const [accelerator, setAccelerator] = useState(null);
  const [detectorModels, setDetectorModels] = useState([]);
  const [recordingCache, setRecordingCache] = useState(null);
  const [mqttStatus, setMqttStatus] = useState(null);
  const [settingsTab, setSettingsTab] = useStoredState("survng.configTab", "general");
  const [generalSection, setGeneralSection] = useStoredState("survng.generalSection.v1", "general");
  const [selectedId, setSelectedId] = useState("");
  const [saveNotice, setSaveNotice] = useState(null);
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
  const [auditItems, setAuditItems] = useState([]);
  const [auditTotal, setAuditTotal] = useState(0);
  const [auditPage, setAuditPage] = useState(0);
  const [auditCamera, setAuditCamera] = useStoredState("survng.motionAuditCamera.v1", "");
  const [auditOutcome, setAuditOutcome] = useStoredState("survng.motionAuditOutcome.v1", "all");
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditError, setAuditError] = useState("");
  const [selectedAuditId, setSelectedAuditId] = useState(null);
  const auditPageSize = 24;

  async function load() {
    const [response, statusResponse, acceleratorResponse, modelsResponse, cacheResponse, systemResponse] = await Promise.all([
      fetch("/api/config"),
      fetch("/api/cameras"),
      fetch("/api/accelerator"),
      fetch("/api/detector/models"),
      fetch("/api/recordings/cache/status"),
      fetch("/api/system/status"),
    ]);
    const nextConfig = await response.json();
    setConfig(nextConfig);
    if (statusResponse.ok) setRuntimeStatus(await statusResponse.json());
    if (acceleratorResponse.ok) setAccelerator(await acceleratorResponse.json());
    if (modelsResponse.ok) setDetectorModels((await modelsResponse.json()).models || []);
    if (cacheResponse.ok) setRecordingCache(await cacheResponse.json());
    if (systemResponse.ok) setMqttStatus((await systemResponse.json()).mqtt || null);
    setSelectedId((current) => nextConfig.cameras?.some((camera) => camera.id === current) ? current : nextConfig.cameras?.[0]?.id || "");
  }

  useEffect(() => {
    load();
  }, []);


  async function loadLogs() {
    const params = new URLSearchParams({ limit: "500", level: logLevel, q: logFilter });
    const response = await fetch(`/api/logs?${params.toString()}`);
    if (response.ok) {
      const payload = await response.json();
      setLogLines(payload.lines || []);
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
  }, [settingsTab, auditPage, auditCamera, auditOutcome]);

  const cameras = config?.cameras || [];
  const selectedCamera = cameras.find((camera) => camera.id === selectedId) || cameras[0] || null;
  const selectedRuntimeStatus = runtimeStatus.find((camera) => camera.id === selectedCamera?.id);
  const selectedAudit = auditItems.find((item) => item.id === selectedAuditId) || null;
  const activeDetectorPath = config?.detector?.model_path || config?.detector?.model_xml || "";
  const activeDetectorModel = detectorModels.find((model) => model.path === activeDetectorPath);
  const zoneClassOptions = activeDetectorModel?.classes?.length
    ? activeDetectorModel.classes
    : config?.detector?.labels || [];



  if (!config) {
    return <main className="bento-grid config-grid"><section className="bento-card config-editor"><div className="empty-state">Loading config...</div></section></main>;
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
    setGeneralSaving(true);
    setSaveNotice({ state: "saving", text: "Saving and reloading cameras..." });
    try {
      const response = await fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(configToSave),
      });
      if (!response.ok) throw new Error(await response.text());
      setSaveNotice({ state: "saved", text: "Saved. Camera workers reloaded." });
      await load();
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
      setSaveNotice({ state: "saved", text: "Camera settings saved. Workers reloaded." });
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
    const response = await fetch("/api/config/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        host,
        username,
        password,
        onvif_port: probeCameraConfig.onvif?.port || 8000,
        baichuan_port: probeCameraConfig.baichuan?.port || 9000,
      }),
    });
    const result = await response.json();
    setProbe(result);
    if (result.onvif?.reachable) updateCamera(camera.id, ["onvif", "enabled"], true);
    if (result.baichuan?.reachable) {
      updateCamera(camera.id, ["baichuan", "enabled"], true);
      updateCamera(camera.id, ["video_backend"], "baichuan_native");
    }
  }

  return (
    <main className="bento-grid config-grid settings-grid">
      <div className="settings-tabs">
        <div className="settings-tab-list" role="tablist" aria-label="Config sections">
          <button className={settingsTab === "general" ? "active" : ""} onClick={() => setSettingsTab("general")} role="tab" aria-selected={settingsTab === "general"}><Cog size={16} /> General</button>
          <button className={settingsTab === "cameras" ? "active" : ""} onClick={() => setSettingsTab("cameras")} role="tab" aria-selected={settingsTab === "cameras"}><Camera size={16} /> Camera Settings</button>
          <button className={settingsTab === "audit" ? "active" : ""} onClick={() => setSettingsTab("audit")} role="tab" aria-selected={settingsTab === "audit"}><Activity size={16} /> Motion Audit</button>
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
            <button type="button" className={generalSection === "mqtt" ? "active" : ""} onClick={() => setGeneralSection("mqtt")}><Radio size={16} /><span>MQTT</span></button>
            <button type="button" className={generalSection === "detection" ? "active" : ""} onClick={() => setGeneralSection("detection")}><Cpu size={16} /><span>Object Detection</span></button>
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
            mqttStatus={mqttStatus}
            section={generalSection}
          />
        </section>
        </>
      ) : settingsTab === "audit" ? (
        <>
        <section className="bento-card camera-tree config-tree settings-section-tree motion-audit-filters">
          <div className="section-head compact"><div><h2>Motion Audit</h2><p>{auditTotal.toLocaleString()} rejected bursts</p></div></div>
          <div className="motion-audit-filter-fields">
            <label>Camera<select value={auditCamera} onChange={(event) => { setAuditCamera(event.target.value); setAuditPage(0); }}>
              <option value="">All cameras</option>
              {cameras.map((camera) => <option value={camera.id} key={camera.id}>{camera.name || camera.id}</option>)}
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
            <div><h2>Rejected Motion</h2><p>Qualifier decisions and detector outcomes</p></div>
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
            level={logLevel}
            setLevel={setLogLevel}
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
          <div><h2>{selectedCamera ? selectedCamera.name : "Camera Config"}</h2><p>Changes save to config.json and reload camera workers</p></div>
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
              <div className="field-row">
                <label>Name<input value={selectedCamera.name} onChange={(event) => updateCamera(selectedCamera.id, ["name"], event.target.value)} /></label>
                <label>Generated Camera ID<input value={slugify(selectedCamera.name || selectedCamera.id || "camera")} readOnly /></label>
                <label>Detected Backend<input value={inferredBackendLabel(selectedCamera)} readOnly /></label>
              </div>
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

              <div className="config-panels">
                <div className="sub-panel">
                  <h3>ONVIF</h3>
                  <label className="check-field"><input type="checkbox" checked={selectedCamera.onvif?.enabled || false} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "enabled"], event.target.checked)} /> Enabled</label>
                  <label>Host<input value={selectedCamera.onvif?.host || ""} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "host"], event.target.value)} /></label>
                  <label>Port<input type="number" value={selectedCamera.onvif?.port || 8000} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "port"], Number(event.target.value))} /></label>
                  <label>Username<input value={selectedCamera.onvif?.username || ""} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "username"], event.target.value)} /></label>
                  <label>Password<input type="password" value={selectedCamera.onvif?.password || ""} onChange={(event) => updateCamera(selectedCamera.id, ["onvif", "password"], event.target.value)} /></label>
                </div>
                <div className="sub-panel">
                  <h3>Motion Qualification</h3>
                  <label>Mode<select value={selectedCamera.motion_qualification?.mode || "inherit"} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "mode"], event.target.value)}>
                    <option value="inherit">Use global setting</option>
                    <option value="off">Off</option>
                    <option value="audit">Audit</option>
                    <option value="enforce">Enforce</option>
                  </select></label>
                  <label>Sensitivity<select value={selectedCamera.motion_qualification?.sensitivity || "inherit"} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "sensitivity"], event.target.value)}>
                    <option value="inherit">Use global setting</option>
                    <option value="high">High</option>
                    <option value="balanced">Balanced</option>
                    <option value="low">Low</option>
                  </select></label>
                  <label>Analysis Width<select value={selectedCamera.motion_qualification?.frame_width ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "frame_width"], event.target.value ? Number(event.target.value) : null)}>
                    <option value="">Use global setting</option>
                    <option value="320">320 px</option>
                    <option value="480">480 px</option>
                    <option value="640">640 px</option>
                    <option value="720">720 px</option>
                    <option value="800">800 px</option>
                  </select></label>
                  <label>Borderline Rescue<select value={selectedCamera.motion_qualification?.borderline_rescue_enabled == null ? "" : String(selectedCamera.motion_qualification.borderline_rescue_enabled)} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "borderline_rescue_enabled"], event.target.value === "" ? null : event.target.value === "true")}>
                    <option value="">Use global setting</option>
                    <option value="true">Enabled</option>
                    <option value="false">Disabled</option>
                  </select></label>
                  <label>Rescue Margin<input type="number" min="0" max="0.1" step="0.005" placeholder="Global" value={selectedCamera.motion_qualification?.borderline_margin ?? ""} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "borderline_margin"], event.target.value === "" ? null : Number(event.target.value))} /></label>
                  <label>MOG2 Audit<select value={selectedCamera.motion_qualification?.mog2_audit_enabled == null ? "" : String(selectedCamera.motion_qualification.mog2_audit_enabled)} onChange={(event) => updateCamera(selectedCamera.id, ["motion_qualification", "mog2_audit_enabled"], event.target.value === "" ? null : event.target.value === "true")}>
                    <option value="">Use global setting</option>
                    <option value="true">Enabled</option>
                    <option value="false">Disabled</option>
                  </select></label>
                </div>
              </div>

              <ZoneEditor
                camera={selectedCamera}
                classOptions={zoneClassOptions}
                onChange={(zones) => updateCamera(selectedCamera.id, ["zones"], zones)}
                onSave={() => saveZones(selectedCamera)}
                saving={zonesSaving}
              />

              <RuntimeStatus status={selectedRuntimeStatus} timeZone={timeZone} />
              {probe ? <ProbeResult probe={probe} /> : null}
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
          items={auditItems}
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
  const snapshotUrl = useMemo(() => appUrl(`/api/cameras/${camera.id}/zone-snapshot.jpg?source=live&t=${Date.now()}`), [camera.id]);
  const selectedZone = zones[selectedIndex] || null;

  useEffect(() => {
    setSelectedIndex(0);
    setDragPoint(null);
  }, [camera.id]);

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
      trigger: "bottom_center",
    }];
    onChange(next);
    setSelectedIndex(next.length - 1);
  }

  function removeZone(index) {
    onChange(zones.filter((_, zoneIndex) => zoneIndex !== index));
    setSelectedIndex((current) => Math.max(0, Math.min(current, zones.length - 2)));
  }

  function undoPoint() {
    if (!selectedZone?.points?.length) return;
    replaceZone(selectedIndex, { points: selectedZone.points.slice(0, -1) });
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
    replaceZone(selectedIndex, { points: [...(selectedZone.points || []), pointerPosition(event)] });
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
          <button type="button" onClick={undoPoint} disabled={!selectedZone?.points?.length} title="Remove last point"><Undo2 size={15} /> Undo Point</button>
          <button type="button" onClick={addZone}><Plus size={15} /> Add Zone</button>
          <button type="button" className="primary" onClick={onSave} disabled={saving}><Save size={15} /> {saving ? "Saving..." : "Save Zones"}</button>
        </div>
      </div>
      <div className="zone-editor-layout">
        <div className="zone-canvas">
          <img src={snapshotUrl} alt={`${camera.name} zone editor`} />
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            onPointerDown={addPoint}
            onPointerMove={movePoint}
            onPointerUp={() => setDragPoint(null)}
            onPointerCancel={() => setDragPoint(null)}
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
          {!selectedZone ? <div className="zone-canvas-empty">Add a zone to begin</div> : selectedZone.points?.length < 3 ? <div className="zone-canvas-hint">Click at least three points</div> : null}
        </div>
        <div className="zone-list">
          {zones.map((zone, index) => (
            <button type="button" key={`${zone.name}-${index}`} className={index === selectedIndex ? "active" : ""} onClick={() => setSelectedIndex(index)}>
              <span className="zone-swatch" style={{ background: zone.color || "#22c55e" }} />
              <span>{zone.name || `Zone ${index + 1}`}</span>
              <small>{zone.behavior}</small>
            </button>
          ))}
          {!zones.length ? <div className="empty-state compact">No zones configured.</div> : null}
        </div>
      </div>
      {selectedZone ? (
        <div className="zone-fields">
          <label>Name<input value={selectedZone.name || ""} onChange={(event) => replaceZone(selectedIndex, { name: event.target.value })} /></label>
          <label>Color<input className="zone-color-input" type="color" value={selectedZone.color || "#22c55e"} onChange={(event) => replaceZone(selectedIndex, { color: event.target.value })} /></label>
          <label>Behavior<select value={selectedZone.behavior || "incident"} onChange={(event) => replaceZone(selectedIndex, { behavior: event.target.value })}><option value="incident">Incident</option><option value="ignore">Ignore</option></select></label>
          <div className="zone-class-field">
            <span>Object Classes</span>
            <details className="zone-class-dropdown">
              <summary>{selectedZone.object_classes?.length ? selectedZone.object_classes.join(", ") : "All classes"}</summary>
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
          <label>Confidence<input type="number" min="0.01" max="0.99" step="0.01" placeholder="Global" value={selectedZone.confidence_threshold ?? ""} onChange={(event) => replaceZone(selectedIndex, { confidence_threshold: event.target.value === "" ? null : Number(event.target.value) })} /></label>
          <label className="check-field"><input type="checkbox" checked={selectedZone.enabled !== false} onChange={(event) => replaceZone(selectedIndex, { enabled: event.target.checked })} /> Enabled</label>
          <button type="button" className="danger" onClick={() => removeZone(selectedIndex)}><Trash2 size={15} /> Remove Zone</button>
        </div>
      ) : null}
    </div>
  );
}

function LogViewer({ lines, filter, setFilter, level, setLevel, timeZone }) {
  return (
    <div className="log-viewer">
      <div className="log-toolbar">
        <label>Filter<input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="logger, text, error..." /></label>
      </div>
      <div className="log-lines" role="log" aria-live="polite">
        {lines.length ? lines.map((line, index) => (
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
  if (item.object_detected === true) return { label: "Object found", className: "object" };
  if (item.object_detected === false) return { label: "No object", className: "clear" };
  return { label: "Not run", className: "not-run" };
}

function MotionAuditViewer({ items, total, page, pageSize, setPage, loading, error, timeZone, onOpen }) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  return (
    <div className="motion-audit-viewer">
      {error ? <div className="save-status motion-audit-error">{error}</div> : null}
      <div className="motion-audit-grid">
        {items.map((item) => {
          const outcome = motionAuditOutcome(item);
          const features = Object.entries(item.features || {}).filter(([name]) => !name.startsWith("mog2_") || ["mog2_score", "mog2_track_persistence"].includes(name));
          return (
            <article className="motion-audit-card" key={item.id}>
              <button type="button" className="motion-audit-media" onClick={() => onOpen(item)} aria-label={`Open ${item.camera_id} motion audit image`}>
                {item.has_snapshot
                  ? <img src={appUrl(`/api/motion-audit/${item.id}/snapshot.jpg`)} alt={`${item.camera_id} rejected motion`} loading="lazy" />
                  : <div className="empty-thumb"><Camera size={28} /><span>No sampled frame</span></div>}
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
        {!items.length && !loading ? <div className="empty-state">No rejected motion matches these filters.</div> : null}
      </div>
      <div className="motion-audit-pagination">
        <button type="button" aria-label="Previous audit page" onClick={() => setPage(Math.max(0, page - 1))} disabled={page <= 0 || loading}><ChevronLeft size={16} /></button>
        <span>{total ? `${page * pageSize + 1}-${Math.min(total, (page + 1) * pageSize)} of ${total}` : "0 entries"}</span>
        <button type="button" aria-label="Next audit page" onClick={() => setPage(Math.min(pageCount - 1, page + 1))} disabled={page >= pageCount - 1 || loading}><ChevronRight size={16} /></button>
      </div>
    </div>
  );
}

function Mog2TrackOverlay({ tracks, bounds }) {
  if (!bounds || !Array.isArray(tracks) || !tracks.length) return null;
  return (
    <div className="mog2-track-overlay" style={bounds} aria-hidden="true">
      <svg viewBox="0 0 1000 1000" preserveAspectRatio="none">
        {tracks.map((track, index) => {
          const box = Array.isArray(track.box) ? track.box : [];
          const path = Array.isArray(track.path) ? track.path : [];
          const points = path.map(([x, y]) => `${Number(x) * 1000},${Number(y) * 1000}`).join(" ");
          if (box.length !== 4) return null;
          return (
            <g className={`mog2-track-color-${index % 6}`} key={track.id ?? index}>
              {points ? <polyline className="mog2-track-trail" points={points} vectorEffect="non-scaling-stroke" /> : null}
              <rect className="mog2-track-box" x={Number(box[0]) * 1000} y={Number(box[1]) * 1000} width={(Number(box[2]) - Number(box[0])) * 1000} height={(Number(box[3]) - Number(box[1])) * 1000} vectorEffect="non-scaling-stroke" />
            </g>
          );
        })}
      </svg>
      {tracks.map((track, index) => {
        const box = Array.isArray(track.box) ? track.box : [];
        if (box.length !== 4) return null;
        return <span className={`mog2-track-label mog2-track-color-${index % 6}`} style={{ left: `${Number(box[0]) * 100}%`, top: `${Number(box[1]) * 100}%` }} key={track.id ?? index}>M{track.id}</span>;
      })}
    </div>
  );
}

function MotionAuditOverlay({ item, items, timeZone, onClose, onSelect }) {
  const outcome = motionAuditOutcome(item);
  const currentIndex = items.findIndex((candidate) => candidate.id === item.id);
  const [aiAdvice, setAiAdvice] = useState(null);
  const [aiError, setAiError] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [aiApplying, setAiApplying] = useState(false);
  const [imageSize, setImageSize] = useState(null);
  const [showMog2Tracks, setShowMog2Tracks] = useState(true);
  const [trackBounds, setTrackBounds] = useState(null);
  const mediaRef = useRef(null);
  const imageRef = useRef(null);
  const mog2Tracks = Array.isArray(item.features?.mog2_tracks) ? item.features.mog2_tracks : [];

  useEffect(() => {
    setAiAdvice(null);
    setAiError("");
    setAiLoading(false);
    setAiApplying(false);
    setImageSize(null);
    setShowMog2Tracks(true);
    setTrackBounds(null);
  }, [item.id]);

  useEffect(() => {
    const media = mediaRef.current;
    const image = imageRef.current;
    if (!media || !image) return undefined;
    function updateBounds() {
      const mediaRect = media.getBoundingClientRect();
      const imageRect = image.getBoundingClientRect();
      setTrackBounds({
        left: `${imageRect.left - mediaRect.left}px`,
        top: `${imageRect.top - mediaRect.top}px`,
        width: `${imageRect.width}px`,
        height: `${imageRect.height}px`,
      });
    }
    const observer = new ResizeObserver(updateBounds);
    observer.observe(media);
    observer.observe(image);
    const frame = window.requestAnimationFrame(updateBounds);
    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [item.id, imageSize]);

  const overlayStyle = useMemo(() => {
    if (!imageSize?.width || !imageSize?.height) return undefined;
    const ratio = imageSize.width / imageSize.height;
    return { "--motion-audit-panel-fit-width": `calc(${ratio * 88}dvh + 316px)` };
  }, [imageSize]);

  async function analyzeWithAi() {
    if (aiLoading) return;
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
        body: JSON.stringify({ changes }),
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
      <section className="motion-audit-overlay-panel" style={overlayStyle}>
        <header className="motion-audit-overlay-head">
          <div><h2>{item.camera_id}</h2><time>{formatDateTime(item.created_at, timeZone)}</time></div>
          <div className="overlay-actions">
            {mog2Tracks.length ? <button type="button" className={`icon-only mog2-track-toggle ${showMog2Tracks ? "active" : ""}`} onClick={() => setShowMog2Tracks((visible) => !visible)} aria-label={`${showMog2Tracks ? "Hide" : "Show"} MOG2 tracks`} title={`${showMog2Tracks ? "Hide" : "Show"} ${mog2Tracks.length} MOG2 track${mog2Tracks.length === 1 ? "" : "s"}`}><Radar size={19} /></button> : null}
            <button type="button" className="icon-only" onClick={() => move(-1)} disabled={items.length < 2} aria-label="Previous audit image"><ChevronLeft size={19} /></button>
            <span>{currentIndex + 1} / {items.length}</span>
            <button type="button" className="icon-only" onClick={() => move(1)} disabled={items.length < 2} aria-label="Next audit image"><ChevronRight size={19} /></button>
            <button type="button" className="icon-only" onClick={onClose} aria-label="Close motion audit image"><X size={19} /></button>
          </div>
        </header>
        <div className="motion-audit-overlay-content">
          <div className="motion-audit-overlay-media" ref={mediaRef}>
            {item.has_snapshot
              ? <><img ref={imageRef} src={appUrl(`/api/motion-audit/${item.id}/snapshot.jpg`)} alt={`${item.camera_id} rejected motion`} onLoad={(event) => setImageSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />{showMog2Tracks ? <Mog2TrackOverlay tracks={mog2Tracks} bounds={trackBounds} /> : null}</>
              : <div className="empty-thumb"><Camera size={42} /><span>No sampled frame</span></div>}
          </div>
          <aside className="motion-audit-overlay-details">
            <span className={`motion-audit-outcome ${outcome.className}`}>{outcome.label}</span>
            <div className="motion-audit-overlay-score"><span>{String(item.reason || "rejected").replaceAll("_", " ")}</span><strong>{Number(item.score || 0).toFixed(3)} / {Number(item.threshold || 0).toFixed(3)}</strong></div>
            <div className="motion-audit-meter"><i style={{ width: `${Math.max(0, Math.min(100, Number(item.score || 0) * 100))}%` }} /><b style={{ left: `${Math.max(0, Math.min(100, Number(item.threshold || 0) * 100))}%` }} /></div>
            <dl>
              {Object.entries(item.features || {}).filter(([, value]) => typeof value === "number").map(([name, value]) => <div key={name}><dt>{name.replaceAll("_", " ")}</dt><dd>{Number(value).toFixed(3)}</dd></div>)}
              {mog2Tracks.length ? <div><dt>MOG2 tracks</dt><dd>{mog2Tracks.length}</dd></div> : null}
              <div><dt>Mode</dt><dd>{item.mode}</dd></div>
              <div><dt>Sensitivity</dt><dd>{item.sensitivity}</dd></div>
              <div><dt>Triggers</dt><dd>{item.trigger_count}</dd></div>
            </dl>
            <div className="motion-audit-ai">
              <div className="motion-audit-ai-head">
                <strong><Sparkles size={15} /> AI Advisor</strong>
                <button type="button" onClick={analyzeWithAi} disabled={aiLoading || aiApplying}><Sparkles size={15} /> {aiLoading ? "Analyzing..." : "Analyze"}</button>
              </div>
              {aiError ? <div className="motion-audit-ai-error">{aiError}</div> : null}
              {aiAdvice?.advice ? (
                <div className="motion-audit-ai-result">
                  <div className="motion-audit-ai-verdict"><span>{aiAdvice.advice.verdict.replaceAll("_", " ")}</span><strong>{Math.round(Number(aiAdvice.advice.confidence || 0) * 100)}%</strong></div>
                  <p>{aiAdvice.advice.summary}</p>
                  {aiAdvice.advice.visible_subjects?.length ? <div className="motion-audit-ai-subjects">{aiAdvice.advice.visible_subjects.map((subject) => <span key={subject}>{subject}</span>)}</div> : null}
                  {aiAdvice.advice.explanation?.length ? <ul>{aiAdvice.advice.explanation.map((line) => <li key={line}>{line}</li>)}</ul> : null}
                  {aiAdvice.advice.changes?.length ? (
                    <>
                      <div className="motion-audit-ai-changes">
                        {aiAdvice.advice.changes.map((change, index) => <div key={`${change.scope}-${change.setting}-${index}`}><strong>{change.scope} · {change.setting.replaceAll("_", " ")}</strong><code>{String(change.value)}</code><small>{change.reason}</small></div>)}
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

function GeneralSettings({ config, updateConfig, timeZone, setTimeZone, theme, setTheme, accelerator, detectorModels, recordingCache, mqttStatus, section }) {
  const [liveOrderReset, setLiveOrderReset] = useState(false);
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
  const activeModelPath = config.detector?.model_path || config.detector?.model_xml || "";
  const activeModel = detectorModels.find((model) => model.path === activeModelPath);

  function selectOpenvinoModel(path) {
    updateConfig(["detector", "model_path"], path);
    updateConfig(["detector", "model_xml"], "");
    if (path.endsWith(".xml")) updateConfig(["detector", "labels_path"], "");
  }

  function resetLiveCameraOrder() {
    localStorage.removeItem("survng.liveCameraOrder.v1");
    setLiveOrderReset(true);
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
          <label className="check-field"><input type="checkbox" checked={config.incident_thumbnail_annotations ?? true} onChange={(event) => updateConfig(["incident_thumbnail_annotations"], event.target.checked)} /> Show boxes on incident thumbnails</label>
          <div className="preference-action">
            <strong>Live Camera Order</strong>
            <button type="button" onClick={resetLiveCameraOrder}><RotateCcw size={15} /> Reset Order</button>
          </div>
          {liveOrderReset ? <span className="preference-status"><CircleDot size={13} /> Reset for this browser</span> : null}
        </div>
        ) : null}

        {section === "storage" ? (
        <div className="sub-panel">
          <h3>Storage</h3>
          <label>Storage Directory<input value={config.storage_dir || ""} onChange={(event) => updateConfig(["storage_dir"], event.target.value)} /></label>
          <label>FFmpeg Path<input value={config.ffmpeg_path || ""} onChange={(event) => updateConfig(["ffmpeg_path"], event.target.value)} /></label>
          <label>Hardware Acceleration<select value={config.hardware_acceleration || "auto"} onChange={(event) => updateConfig(["hardware_acceleration"], event.target.value)}>
            <option value="auto">Auto (VAAPI preferred)</option>
            <option value="vaapi">VAAPI</option>
            <option value="qsv">Intel QSV</option>
            <option value="off">Off</option>
          </select></label>
          <label>Event Clip Before<input type="number" min="0" max="30" step="1" value={config.event_clip_before_seconds ?? 5} onChange={(event) => updateConfig(["event_clip_before_seconds"], Number(event.target.value))} /></label>
          <label>Event Clip After<input type="number" min="0" max="30" step="1" value={config.event_clip_after_seconds ?? 5} onChange={(event) => updateConfig(["event_clip_after_seconds"], Number(event.target.value))} /></label>
          <label>Recording Segment Seconds<input type="number" min="2" max="300" step="1" value={config.recording_segment_seconds ?? 10} onChange={(event) => updateConfig(["recording_segment_seconds"], Number(event.target.value))} /></label>
          <label>Playback Cache GB<input type="number" min="0.5" max="100" step="0.5" value={config.recording_cache_max_gb ?? 5} onChange={(event) => updateConfig(["recording_cache_max_gb"], Number(event.target.value))} /></label>
          <label>Playback Cache Days<input type="number" min="1" max="90" step="1" value={config.recording_cache_max_days ?? 7} onChange={(event) => updateConfig(["recording_cache_max_days"], Number(event.target.value))} /></label>
          <label className="check-field"><input type="checkbox" checked={config.recording_cache_prewarm ?? true} onChange={(event) => updateConfig(["recording_cache_prewarm"], event.target.checked)} /> Prewarm finalized recordings</label>
          {recordingCache ? <div className="probe-result"><strong>Playback Cache</strong><span>{formatBytes(recordingCache.bytes)} used across {recordingCache.entries} fragments</span><span>{formatBytes(recordingCache.max_bytes)} limit, {recordingCache.max_days} day maximum age</span><span>{recordingCache.metrics?.playback_hits || 0} hits / {recordingCache.metrics?.playback_misses || 0} misses, {recordingCache.metrics?.playback_avg_remux_ms || 0} ms average remux</span></div> : null}
        </div>
        ) : null}

        {section === "mqtt" ? (
        <div className="sub-panel">
          <h3>MQTT</h3>
          <label className="check-field"><input type="checkbox" checked={config.mqtt?.enabled || false} onChange={(event) => updateConfig(["mqtt", "enabled"], event.target.checked)} /> Enabled</label>
          <label>Broker Host<input value={config.mqtt?.host || ""} onChange={(event) => updateConfig(["mqtt", "host"], event.target.value)} placeholder="mqtt.local" /></label>
          <label>Port<input type="number" min="1" max="65535" value={config.mqtt?.port || 1883} onChange={(event) => updateConfig(["mqtt", "port"], Number(event.target.value))} /></label>
          <label>Username<input value={config.mqtt?.username || ""} onChange={(event) => updateConfig(["mqtt", "username"], event.target.value)} /></label>
          <label>Password<input type="password" value={config.mqtt?.password || ""} onChange={(event) => updateConfig(["mqtt", "password"], event.target.value)} /></label>
          <label>Client ID<input value={config.mqtt?.client_id || "survng"} onChange={(event) => updateConfig(["mqtt", "client_id"], event.target.value)} /></label>
          <label>Topic Prefix<input value={config.mqtt?.topic_prefix || "survng"} onChange={(event) => updateConfig(["mqtt", "topic_prefix"], event.target.value)} /></label>
          <label className="check-field"><input type="checkbox" checked={config.mqtt?.incident_events_enabled ?? true} onChange={(event) => updateConfig(["mqtt", "incident_events_enabled"], event.target.checked)} /> Publish incident events</label>
          <label className="check-field"><input type="checkbox" checked={config.mqtt?.discovery_enabled ?? true} onChange={(event) => updateConfig(["mqtt", "discovery_enabled"], event.target.checked)} /> Home Assistant Discovery</label>
          <label>Discovery Prefix<input value={config.mqtt?.discovery_prefix || "homeassistant"} onChange={(event) => updateConfig(["mqtt", "discovery_prefix"], event.target.value)} disabled={config.mqtt?.discovery_enabled === false} /></label>
          <label>QoS<select value={config.mqtt?.qos ?? 0} onChange={(event) => updateConfig(["mqtt", "qos"], Number(event.target.value))}><option value={0}>0</option><option value={1}>1</option><option value={2}>2</option></select></label>
          <label className="check-field"><input type="checkbox" checked={config.mqtt?.tls || false} onChange={(event) => updateConfig(["mqtt", "tls"], event.target.checked)} /> TLS</label>
          {mqttStatus ? <div className={`probe-result ${mqttStatus.connected ? "ok" : ""}`}><strong>{mqttStatus.connected ? "Connected" : mqttStatus.enabled ? "Disconnected" : "Disabled"}</strong><span>{mqttStatus.host || "No broker"}:{mqttStatus.port || 1883}</span><span>{mqttStatus.messages_published || 0} published, {mqttStatus.commands_received || 0} commands</span>{mqttStatus.incident_events_enabled ? <span>Incidents: {mqttStatus.incident_topic} ({mqttStatus.pending_incidents || 0} pending)</span> : null}{mqttStatus.last_error ? <span>{mqttStatus.last_error}</span> : null}</div> : null}
        </div>
        ) : null}

      {section === "detection" ? (
      <div className="sub-panel">
        <h3>Object Detection</h3>
        <div className="field-row">
          <label className="check-field"><input type="checkbox" checked={config.detector?.enabled || false} onChange={(event) => updateConfig(["detector", "enabled"], event.target.checked)} /> Enable detector</label>
          <label>Backend<select value={detectorBackend} onChange={(event) => updateConfig(["detector", "backend"], event.target.value)}>
            <option value="openvino">OpenVINO / ONNX</option>
            <option value="coreml">Core ML (Mac)</option>
          </select></label>
          <label>OpenVINO Device<select value={config.detector?.device || "CPU"} onChange={(event) => updateConfig(["detector", "device"], event.target.value)}>
            {deviceOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          <label>Confidence<input type="number" min="0" max="1" step="0.01" value={config.detector?.confidence_threshold ?? 0.45} onChange={(event) => updateConfig(["detector", "confidence_threshold"], Number(event.target.value))} /></label>
        </div>
        <div className="field-row">
          <label>Detected Model<select value={detectorModels.some((model) => model.path === activeModelPath) ? activeModelPath : ""} onChange={(event) => selectOpenvinoModel(event.target.value)}>
            <option value="">Custom path</option>
            {detectorModels.map((model) => <option key={model.path} value={model.path} disabled={!model.valid}>{model.name} ({model.task || "detect"}, {model.valid ? "ready" : "incomplete"})</option>)}
          </select></label>
          <label>OpenVINO / ONNX Model<input value={activeModelPath} onChange={(event) => selectOpenvinoModel(event.target.value)} placeholder="openvino_model/best.xml or best.onnx" /></label>
          <label>Labels Path<input value={config.detector?.labels_path || ""} onChange={(event) => updateConfig(["detector", "labels_path"], event.target.value)} placeholder="Optional; metadata.yaml is automatic" /></label>
          <label>Compiled Model Cache<input value={config.detector?.cache_dir || ".cache/openvino"} onChange={(event) => updateConfig(["detector", "cache_dir"], event.target.value)} disabled={config.detector?.cache_enabled === false} /></label>
          <label className="check-field"><input type="checkbox" checked={config.detector?.cache_enabled ?? true} onChange={(event) => updateConfig(["detector", "cache_enabled"], event.target.checked)} /> Cache compiled model</label>
          <label className="check-field"><input type="checkbox" checked={config.detector?.warmup_enabled ?? true} onChange={(event) => updateConfig(["detector", "warmup_enabled"], event.target.checked)} /> Warm up detector at startup</label>
          <label>Maximum Saved Faces<input type="number" min="100" max="100000" step="100" value={config.detector?.face_max_observations ?? 1000} onChange={(event) => updateConfig(["detector", "face_max_observations"], Number(event.target.value))} /></label>
        </div>
        <h3>Motion Qualification</h3>
        <div className="field-row">
          <label>Mode<select value={config.motion_qualification?.mode || "audit"} onChange={(event) => updateConfig(["motion_qualification", "mode"], event.target.value)}><option value="off">Off</option><option value="audit">Audit</option><option value="enforce">Enforce</option></select></label>
          <label>Sensitivity<select value={config.motion_qualification?.sensitivity || "balanced"} onChange={(event) => updateConfig(["motion_qualification", "sensitivity"], event.target.value)}><option value="high">High</option><option value="balanced">Balanced</option><option value="low">Low</option></select></label>
          <label>Analysis Width<select value={config.motion_qualification?.frame_width ?? 320} onChange={(event) => updateConfig(["motion_qualification", "frame_width"], Number(event.target.value))}><option value="320">320 px</option><option value="480">480 px</option><option value="640">640 px</option><option value="720">720 px</option><option value="800">800 px</option></select></label>
          <label>Sample FPS<input type="number" min="2" max="10" step="1" value={config.motion_qualification?.sample_fps ?? 5} onChange={(event) => updateConfig(["motion_qualification", "sample_fps"], Number(event.target.value))} /></label>
          <label>Window Seconds<input type="number" min="0.8" max="4" step="0.1" value={config.motion_qualification?.window_seconds ?? 1.6} onChange={(event) => updateConfig(["motion_qualification", "window_seconds"], Number(event.target.value))} /></label>
          <label>Post-trigger Seconds<input type="number" min="0.5" max="6" step="0.1" value={config.motion_qualification?.post_trigger_seconds ?? 2.5} onChange={(event) => updateConfig(["motion_qualification", "post_trigger_seconds"], Number(event.target.value))} /></label>
          <label>Burst Quiet Seconds<input type="number" min="0.1" max="2" step="0.1" value={config.motion_qualification?.burst_quiet_seconds ?? 0.5} onChange={(event) => updateConfig(["motion_qualification", "burst_quiet_seconds"], Number(event.target.value))} /></label>
          <label>Rejected Sample Rate<input type="number" min="0" max="1" step="0.01" value={config.motion_qualification?.rejected_sample_rate ?? 0.05} onChange={(event) => updateConfig(["motion_qualification", "rejected_sample_rate"], Number(event.target.value))} /></label>
          <label className="check-field"><input type="checkbox" checked={config.motion_qualification?.borderline_rescue_enabled ?? true} onChange={(event) => updateConfig(["motion_qualification", "borderline_rescue_enabled"], event.target.checked)} /> Borderline object rescue</label>
          <label>Rescue Margin<input type="number" min="0" max="0.1" step="0.005" value={config.motion_qualification?.borderline_margin ?? 0.03} onChange={(event) => updateConfig(["motion_qualification", "borderline_margin"], Number(event.target.value))} /></label>
          <label className="check-field"><input type="checkbox" checked={config.motion_qualification?.mog2_audit_enabled ?? true} onChange={(event) => updateConfig(["motion_qualification", "mog2_audit_enabled"], event.target.checked)} /> MOG2 + blob tracking audit</label>
          <label>MOG2 History Seconds<input type="number" min="5" max="300" step="5" value={config.motion_qualification?.mog2_history_seconds ?? 30} onChange={(event) => updateConfig(["motion_qualification", "mog2_history_seconds"], Number(event.target.value))} /></label>
        </div>
        <h3>AI Audit Advisor</h3>
        <div className="field-row">
          <label className="check-field"><input type="checkbox" checked={config.audit_ai?.enabled ?? false} onChange={(event) => updateConfig(["audit_ai", "enabled"], event.target.checked)} /> Enable AI advisor</label>
          <label>Provider<select value={config.audit_ai?.provider || "openai"} onChange={(event) => updateConfig(["audit_ai", "provider"], event.target.value)}>
            <option value="openai">OpenAI</option>
            <option value="gemini">Google Gemini</option>
            <option value="openai_compatible">OpenAI compatible</option>
          </select></label>
          <label>Model<input value={config.audit_ai?.model || ""} onChange={(event) => updateConfig(["audit_ai", "model"], event.target.value)} placeholder={config.audit_ai?.provider === "gemini" ? "gemini-2.5-flash" : "gpt-4.1-mini"} /></label>
          <label>API Key<input type="password" value={config.audit_ai?.api_key || ""} onChange={(event) => updateConfig(["audit_ai", "api_key"], event.target.value)} autoComplete="new-password" /></label>
          <label>Base URL<input value={config.audit_ai?.base_url || ""} onChange={(event) => updateConfig(["audit_ai", "base_url"], event.target.value)} placeholder={config.audit_ai?.provider === "gemini" ? "https://generativelanguage.googleapis.com/v1beta" : config.audit_ai?.provider === "openai_compatible" ? "http://localhost:11434/v1" : "https://api.openai.com/v1"} /></label>
          <label>Timeout Seconds<input type="number" min="5" max="120" step="1" value={config.audit_ai?.timeout_seconds ?? 45} onChange={(event) => updateConfig(["audit_ai", "timeout_seconds"], Number(event.target.value))} /></label>
          <label className="check-field"><input type="checkbox" checked={config.audit_ai?.allow_apply_recommendations ?? false} onChange={(event) => updateConfig(["audit_ai", "allow_apply_recommendations"], event.target.checked)} /> Allow confirmed recommendation apply</label>
        </div>
        <h3>Face Recognition</h3>
        <div className="field-row">
          <label className="check-field"><input type="checkbox" checked={config.detector?.face_recognition_enabled ?? false} onChange={(event) => updateConfig(["detector", "face_recognition_enabled"], event.target.checked)} /> Enable recognition</label>
          <label>Embedding Model<input value={config.detector?.face_embedding_model_path || ""} onChange={(event) => updateConfig(["detector", "face_embedding_model_path"], event.target.value)} placeholder="face_model/model.xml" /></label>
          <label>Landmark Model<input value={config.detector?.face_landmark_model_path || ""} onChange={(event) => updateConfig(["detector", "face_landmark_model_path"], event.target.value)} placeholder="face_model/landmarks.xml" /></label>
          <label>Recognition Device<select value={config.detector?.face_recognition_device || "AUTO"} onChange={(event) => updateConfig(["detector", "face_recognition_device"], event.target.value)}>
            {deviceOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          <label>Match Threshold<input type="number" min="0" max="1" step="0.01" value={config.detector?.face_match_threshold ?? 0.4} onChange={(event) => updateConfig(["detector", "face_match_threshold"], Number(event.target.value))} /></label>
          <label>Minimum Face Size<input type="number" min="16" max="1024" step="8" value={config.detector?.face_min_size ?? 48} onChange={(event) => updateConfig(["detector", "face_min_size"], Number(event.target.value))} /></label>
          <label>References Per Person<input type="number" min="1" max="200" step="1" value={config.detector?.face_max_references ?? 20} onChange={(event) => updateConfig(["detector", "face_max_references"], Number(event.target.value))} /></label>
        </div>
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
          <div className="field-row">
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
      ) : null}
    </div>
  );
}

function RuntimeStatus({ status, timeZone }) {
  if (!status) {
    return <div className="probe-result"><strong>Runtime</strong><span>Save this camera to start workers.</span></div>;
  }
  return (
    <div className="probe-result runtime-result">
      <strong>Runtime</strong>
      <span>Stream worker: {status.running ? "running" : "not running"}</span>
      <span>Recording: {status.recording ? "running" : "stopped"}</span>
      <span>ONVIF: {status.onvif_enabled ? (status.onvif_connected ? "connected" : `not connected${status.onvif_last_error ? `: ${status.onvif_last_error}` : ""}`) : "disabled"}</span>
      {status.onvif_last_event_at ? <span>Last ONVIF message: {formatDateTime(status.onvif_last_event_at, timeZone)}</span> : null}
      {status.motion_qualification ? <span>Motion qualification: {status.motion_qualification.mode} / {status.motion_qualification.sensitivity} · {status.motion_qualification.frame_width || 320}px · {status.motion_qualification.passed || 0} passed · {status.motion_qualification.audit_rejected || 0} audit rejects · {status.motion_qualification.suppressed || 0} suppressed</span> : null}
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
      if (!response.ok) throw new Error("Could not update this face");
      await onUpdated?.();
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
      if (!response.ok) throw new Error("Could not create this person");
      await onUpdated?.(`${name} enrolled`);
    } catch (requestError) {
      setError(requestError.message || "Could not create this person");
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
          {observation.candidate_person_id ? <div className="face-enroll-row"><button type="button" disabled={busy} onClick={() => assignPerson(observation.candidate_person_id)}><ScanFace size={16} /> Confirm {observation.candidate_person_name} ({Math.round(Number(observation.candidate_confidence || 0) * 100)}%)</button><button type="button" className="subtle" disabled={busy} onClick={() => assignPerson(null)}><X size={16} /> Reject</button></div> : null}
          <label>Assign to person<select value={observation.person_id || ""} disabled={busy} onChange={(event) => assignPerson(event.target.value)}><option value="">Unknown</option>{people.map((person) => <option value={person.id} key={person.id}>{person.name}</option>)}</select></label>
          <div className="face-enroll-row"><input value={newName} disabled={busy} onChange={(event) => setNewName(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") createPerson(); }} placeholder="New person name" /><button type="button" onClick={createPerson} disabled={busy || !newName.trim()}><UserPlus size={16} /> Enroll</button></div>
          {error ? <span className="save-status error">{error}</span> : null}
        </div>
      </section>
    </div>
  );
}

function FacesPage({ timeZone }) {
  const [people, setPeople] = useState([]);
  const [observations, setObservations] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [status, setStatus] = useState(null);
  const [filter, setFilter] = useState("unknown");
  const [cameraId, setCameraId] = useState("");
  const [personId, setPersonId] = useState("");
  const [selected, setSelected] = useState(null);
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [totalObservations, setTotalObservations] = useState(0);
  const pageSize = isMobileViewport() ? 24 : 48;
  const pageCount = Math.max(1, Math.ceil(totalObservations / pageSize));

  async function load() {
    setLoading(true);
    try {
      const query = new URLSearchParams({ status: personId ? "all" : filter, limit: String(pageSize), offset: String(page * pageSize) });
      if (cameraId) query.set("camera_id", cameraId);
      if (personId) query.set("person_id", personId);
      const countQuery = new URLSearchParams(query);
      countQuery.delete("limit");
      countQuery.delete("offset");
      const [peopleResponse, observationResponse, countResponse, cameraResponse, statusResponse] = await Promise.all([
        fetch("/api/faces/people"),
        fetch(`/api/faces/observations?${query}`),
        fetch(`/api/faces/observations/count?${countQuery}`),
        fetch("/api/cameras"),
        fetch("/api/faces/status"),
      ]);
      if (!peopleResponse.ok || !observationResponse.ok) throw new Error("Unable to load the face database");
      setPeople(await peopleResponse.json());
      setObservations(await observationResponse.json());
      if (countResponse.ok) setTotalObservations(Number((await countResponse.json()).total || 0));
      if (cameraResponse.ok) setCameras(await cameraResponse.json());
      if (statusResponse.ok) setStatus(await statusResponse.json());
    } catch (error) {
      setNotice(error.message || "Unable to load faces");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, [filter, cameraId, personId, page]);
  useEffect(() => { setPage(0); }, [filter, cameraId, personId]);
  useEffect(() => { if (page >= pageCount) setPage(Math.max(0, pageCount - 1)); }, [page, pageCount]);

  async function deletePerson(person) {
    if (!window.confirm(`Delete ${person.name}? Their observations will return to Unknown.`)) return;
    const response = await fetch(`/api/faces/people/${person.id}`, { method: "DELETE" });
    if (!response.ok) return setNotice("Could not delete this person");
    setPersonId("");
    await load();
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
                <img src={appUrl(`/api/faces/observations/${person.preview_observation_id}/crop.jpg`)} alt="" />
                <span><strong>{person.name}</strong><small>{person.observation_count} observations</small></span>
              </button>
              <button type="button" className="icon-button subtle" onClick={() => deletePerson(person)} title="Delete person" aria-label={`Delete ${person.name}`}><Trash2 size={15} /></button>
            </div>
          ))}
        </div>
      </aside>

      <section className="faces-review-panel">
        <div className="faces-toolbar">
          <div className="faces-filter-group" role="group" aria-label="Face status">
            {["unknown", "suggested", "known", "all"].map((value) => (
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
          {!status?.recognition_ready ? <div className="face-readiness"><Activity size={16} /><span>{status?.recognition_message || "Automatic recognition is not configured."}</span></div> : null}
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
            </button>
          ))}
        </div>
        <div className="faces-pagination">
          <button type="button" onClick={() => setPage((current) => Math.max(0, current - 1))} disabled={page === 0 || loading}><ChevronLeft size={16} /> Previous</button>
          <span>Page {Math.min(page + 1, pageCount)} of {pageCount}</span>
          <button type="button" onClick={() => setPage((current) => Math.min(pageCount - 1, current + 1))} disabled={page >= pageCount - 1 || loading}>Next <ChevronRight size={16} /></button>
        </div>
      </section>

      {selected ? <FaceReviewDialog observation={selected} people={people} timeZone={timeZone} onClose={() => setSelected(null)} onUpdated={async (message) => { setSelected(null); if (message) setNotice(message); await load(); }} /> : null}
    </main>
  );
}

function App() {
  const [timeZone, setTimeZone] = useStoredState("survng.timeZone", DEFAULT_TIME_ZONE);
  const [theme, setTheme] = useStoredState("survng.theme", "auto");
  const [recordingContext, setRecordingContext] = useState(null);
  const pathname = appPathname();
  const page = pathname.startsWith("/config")
    ? "config"
    : pathname.startsWith("/recordings")
      ? "recordings"
      : pathname.startsWith("/incidents")
        ? "incidents"
        : pathname.startsWith("/faces")
          ? "faces"
        : "live";
  useEffect(() => {
    document.documentElement.dataset.theme = THEMES.includes(theme) ? theme : "auto";
  }, [theme]);
  return (
    <Shell page={page} theme={theme} recordingContext={recordingContext}>
      {page === "config"
        ? <ConfigPage timeZone={timeZone} setTimeZone={setTimeZone} theme={theme} setTheme={setTheme} />
        : page === "recordings"
          ? <RecordingsPage timeZone={timeZone} />
          : page === "incidents"
            ? <IncidentsPage timeZone={timeZone} onRecordingContextChange={setRecordingContext} />
            : page === "faces"
              ? <FacesPage timeZone={timeZone} />
            : <LivePage timeZone={timeZone} onRecordingContextChange={setRecordingContext} />}
    </Shell>
  );
}

createRoot(document.getElementById("root")).render(<App />);
